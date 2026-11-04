"""Practice-only token constraints and stored-precision update envelopes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import time
from types import MappingProxyType

import numpy as np

from ..patches.huggingface import _from_float64, _to_float64
from .content_scoring import score_batch
from .endpoint_update import _tensor_bytes
from .fork_update import (
    AffineUpdate, _batches, _check_fork_identity, _check_memory, _only_train, _region, _resources, _usable_forks,
)
from .generative_fork import fork_statistics, next_token_logits, protected_trajectory, trajectory_diagnostics, trajectory_logits
from .preservation_geometry import DecisionAnchor, feasible_intervals, project_nullspace, solve_inequalities


@dataclass(frozen=True)
class DecisionPool:
    """Parent decisions from the host's complete correct practice trajectories."""

    anchors: tuple[DecisionAnchor, ...]
    trajectories: dict

    @property
    def eligible_anchors(self):
        return self.anchors


@dataclass(frozen=True)
class ForkBasis:
    checkpoint_sha256: str
    tensor_name: str
    columns: tuple[int, ...]
    preferred_left: np.ndarray
    right: np.ndarray
    target_ids: tuple[str, ...]
    target_items: tuple[str, ...]
    target_margins: np.ndarray
    target_gradients: np.ndarray


@dataclass(frozen=True)
class MarginSensitivity:
    anchor: DecisionAnchor
    margin: float
    gradient: np.ndarray
    rival_token_id: int


def _key(anchor):
    return anchor.prompt_id, anchor.position


def _prefix(pool, anchor):
    path = pool.trajectories[anchor.prompt_id]
    return path.prompt_token_ids + path.token_ids[:anchor.position]


def measure_parent_decisions(runner, practice, contract, *, group_for, batch_size, max_memory_bytes):
    """Collect replay-consistent positive-margin decisions for all correct rows."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    rows = [row for row in practice["rows"] if row["correct"]]
    if not 1 <= len(rows) <= 128 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected bounded, uniquely identified correct practice rows")
    trajectories = [protected_trajectory(runner, row, contract) for row in rows]
    anchors, evidence = [], []
    calls = 0
    with torch.inference_mode():
        for batch in _batches(trajectories, batch_size):
            calls += 1
            for path, logits in zip(batch, trajectory_logits(runner, batch), strict=True):
                row = rows[len(evidence)]
                group = group_for(row)
                measured = trajectory_diagnostics(path, logits, runner.tokenizer)
                evidence.append({"item": row["item"], "group": group, "wording": row["wording"], **measured})
                anchors.extend(DecisionAnchor(row["item"], path.prompt_id, group, row["wording"], point["position"],
                                              point["margin"], point["token_id"])
                               for point in measured["decisions"]
                               if point["margin"] > 0 and point["predicted_token_id"] == point["token_id"])
            _check_memory(max_memory_bytes)
    pool = DecisionPool(tuple(anchors), MappingProxyType({path.prompt_id: path for path in trajectories}))
    return pool, {"schema": "weightbridge.practice_decision_pool.v1", "trajectories": evidence,
                  "eligible_decisions": len(anchors), "resources": _resources(wall, cpu, calls, 0)}


def fit_fork_basis(adapter, runner, plan, forks, *, batch_size=8):
    """Choose columns and a fixed right factor from fork CE, then measure each target VJP."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    forks = _usable_forks(forks)
    limits = plan.limits
    _check_fork_identity(forks, plan.checkpoint_sha256)
    if not 1 <= len(forks) <= 32 or len({fork.prompt_id for fork in forks}) != len(forks):
        raise ValueError("Expected bounded, distinct replay-matched target forks")
    batches = tuple(_batches(forks, batch_size))
    reuse_graph = len(batches) == 1
    if (len(batches) * (1 if reuse_graph else 2) > limits.max_forward_calls
            or len(batches) + len(forks) > limits.max_backward_calls):
        raise ValueError("Individual fork gradients exceed the authorized compute budget")
    _check_memory(limits.max_memory_bytes)
    ref, module, raw = _region(adapter, runner, plan, torch)
    width = min(16, ref.shape[1], plan.max_columns, limits.max_update_entries // ref.shape[0])
    if width < 1 or limits.max_rank < 1:
        raise ValueError("Rank-one adaptation requires authorized selected affine columns")
    forwards, backwards = 0, 0

    def forward(batch):
        nonlocal forwards
        forwards += 1
        logits = next_token_logits(runner, [fork.prefix_ids for fork in batch])
        result = fork_statistics(logits, [fork.accepted_token_ids for fork in batch],
                                 actual_token_ids=[fork.actual_token_id for fork in batch])
        if result.predicted_ids.detach().tolist() != [fork.actual_token_id for fork in batch]:
            raise ValueError("Fork gradient replay disagreed with the original greedy token")
        _check_memory(limits.max_memory_bytes)
        return result

    def derivative(value, retain_graph=False):
        nonlocal backwards
        backwards += 1
        gradient, = torch.autograd.grad(value, (module.weight,), retain_graph=retain_graph)
        _check_memory(limits.max_memory_bytes)
        return gradient.detach()

    with _only_train(runner.model, (module.weight,)):
        mean = np.zeros(ref.shape, dtype=np.float64)
        for batch in batches:
            stats = forward(batch)
            mean += derivative(stats.cross_entropy.sum() / len(forks), retain_graph=reuse_graph).float().cpu().numpy()
        column_norms = np.linalg.norm(mean, axis=0)
        columns = tuple(sorted(np.argsort(-column_norms, kind="stable")[:width].tolist()))
        gradient = mean[:, columns]
        if not np.isfinite(gradient).all() or float(np.linalg.norm(gradient)) <= 1e-12:
            return None, {"reason": "no_finite_target_direction", "resources": _resources(wall, cpu, forwards, backwards)}
        left, singular, right = np.linalg.svd(-gradient, full_matrices=False)
        preferred_left, right_factor = left[:, 0], right[0]
        margins, slopes = [], []
        for batch in batches:
            if not reuse_graph:
                stats = forward(batch)
            margins.extend(stats.decision_margins.detach().tolist())
            for index, value in enumerate(stats.decision_margins):
                affine_grad = derivative(value, retain_graph=index + 1 < len(batch))
                slopes.append(affine_grad[:, columns].float().cpu().numpy().astype(np.float64) @ right_factor)
    if _tensor_bytes(module.weight, torch) != raw:
        raise RuntimeError("Target gradient collection changed the live parent")
    basis = ForkBasis(plan.checkpoint_sha256, plan.region.tensor_name, columns, preferred_left, right_factor,
                      tuple(fork.prompt_id for fork in forks), tuple(fork.question_id for fork in forks),
                      np.asarray(margins), np.asarray(slopes))
    return basis, {"schema": "weightbridge.fork_rank_one_basis.v1", "columns": list(columns),
                   "target_prompt_ids": list(basis.target_ids), "target_items": list(basis.target_items),
                   "target_margins": margins, "target_gradient_norm": float(np.linalg.norm(mean)),
                   "top_selected_singular_value": float(singular[0]),
                   "resources": _resources(wall, cpu, forwards, backwards)}


def fit_preservation(adapter, runner, plan, basis, pool, anchors, *, batch_size=8):
    """Measure each selected margin gradient in the fixed-right rank-one coordinates."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    anchors = tuple(anchors)
    limits = plan.limits
    batches = tuple(_batches(anchors, batch_size))
    if (basis.checkpoint_sha256 != plan.checkpoint_sha256 or basis.tensor_name != plan.region.tensor_name
            or len(anchors) > min(64, limits.max_backward_calls) or len(batches) > limits.max_forward_calls):
        raise ValueError("Preservation VJPs exceed the parent, region or compute authorization")
    known = {_key(anchor): anchor for anchor in pool.anchors}
    if len({_key(anchor) for anchor in anchors}) != len(anchors) or any(known.get(_key(anchor)) != anchor for anchor in anchors):
        raise ValueError("Constraints must be distinct replay-validated parent practice decisions")
    if not anchors:
        return {}, {"anchors": [], "rejected": [], "resources": _resources(wall, cpu, 0, 0)}
    _check_memory(limits.max_memory_bytes)
    _, module, raw = _region(adapter, runner, plan, torch)
    values, rejected = {}, []
    backwards = 0
    with _only_train(runner.model, (module.weight,)):
        for batch in batches:
            logits = next_token_logits(runner, [_prefix(pool, anchor) for anchor in batch])
            stats = fork_statistics(logits, [(anchor.token_id,) for anchor in batch])
            margins = stats.decision_margins.detach().tolist()
            winners = stats.predicted_ids.detach().tolist()
            rivals = stats.strongest_rival_ids.detach().tolist()
            eligible = [index for index, anchor in enumerate(batch) if margins[index] > 0 and winners[index] == anchor.token_id]
            for index, anchor in enumerate(batch):
                if index not in eligible:
                    rejected.append({**asdict(anchor), "replay_margin": margins[index], "replay_token_id": winners[index]})
                    continue
                gradient, = torch.autograd.grad(stats.decision_margins[index], (module.weight,),
                                                retain_graph=index != eligible[-1])
                backwards += 1
                projected = gradient.detach()[:, basis.columns].float().cpu().numpy().astype(np.float64) @ basis.right
                if not np.isfinite(projected).all():
                    raise ValueError("Nonfinite preservation sensitivity")
                values[_key(anchor)] = MarginSensitivity(anchor, margins[index], projected, rivals[index])
                _check_memory(limits.max_memory_bytes)
    if _tensor_bytes(module.weight, torch) != raw:
        raise RuntimeError("Preservation gradient collection changed the live parent")
    return values, {"schema": "weightbridge.preservation_vjps.v1",
                    "anchors": [{**asdict(value.anchor), "replay_margin": value.margin, "rival_token_id": value.rival_token_id,
                                 "projected_gradient_norm": float(np.linalg.norm(value.gradient)),
                                 "raw_direction_slope": float(value.gradient @ basis.preferred_left)}
                                for value in values.values()], "rejected": rejected,
                    "resources": _resources(wall, cpu, len(batches), backwards)}


def construct_update(basis, sensitivities, *, radius, floor=0., method="inequality"):
    """Solve in the fixed-right factor space and report per-fork crossing intervals."""
    if not isfinite(radius) or radius <= 0 or not isfinite(floor) or floor < 0:
        raise ValueError("An update needs a positive radius and a nonnegative safety floor")
    values = tuple(sensitivities.values()) if hasattr(sensitivities, "values") else tuple(sensitivities)
    gradients = np.asarray([value.gradient for value in values]).reshape(len(values), len(basis.preferred_left))
    margins = np.asarray([value.margin for value in values])
    if method == "nullspace":
        factor = project_nullspace(basis.preferred_left, gradients, radius=radius)
        solver = {"status": "nullspace_projected" if factor is not None else "empty_nullspace_direction"}
        if factor is not None and np.any(margins + gradients @ factor < floor - 1e-9):
            factor, solver = None, {"status": "preservation_floor_violation"}
    elif method == "inequality":
        solution = solve_inequalities(basis.preferred_left, gradients, margins, radius=radius, floor=floor)
        factor, solver = solution.update, solution.record()
    else:
        raise ValueError("Unknown rank-one preservation solver")
    base_evidence = {"schema": "weightbridge.constrained_fork_update.v1", "method": method, "radius": radius,
                     "floor": floor, "solver": solver, "anchors": [asdict(value.anchor) for value in values]}
    if factor is None:
        return None, {**base_evidence, "reason": "constraint_conflict"}
    norm = float(np.linalg.norm(factor))
    if not isfinite(norm) or norm <= 1e-12:
        return None, {**base_evidence, "reason": "no_nonzero_feasible_direction"}
    direction = factor / norm
    target_slopes = basis.target_gradients @ direction
    preserved_slopes = gradients @ direction
    intervals = feasible_intervals(basis.target_margins, target_slopes, margins, preserved_slopes, radius=radius, floor=floor)
    return AffineUpdate(basis.columns, factor[:, None], basis.right[None, :], method), {
        **base_evidence, "reason": "local_candidate", "unrounded_norm": norm, "local_feasibility": intervals,
        "target_predictions": [{"prompt_id": prompt, "item": item, "parent_margin": float(margin),
                                 "slope": float(slope), "predicted_margin": float(margin + slope * norm)}
                                for prompt, item, margin, slope in zip(basis.target_ids, basis.target_items,
                                    basis.target_margins, target_slopes, strict=True)],
        "preservation_predictions": [{"prompt_id": value.anchor.prompt_id, "position": value.anchor.position,
                                       "parent_margin": value.margin, "slope": float(slope),
                                       "predicted_margin": value.margin + float(slope) * norm}
                                      for value, slope in zip(values, preserved_slopes, strict=True)],
    }


def observe_counterexamples(runner, pool, endpoint_result, *, limit, batch_size, max_memory_bytes, exclude=()):
    """Validate divergent decisions only on endpoint-confirmed regressed practice rows."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    if type(limit) is not int or not 0 <= limit <= 16:
        raise ValueError("Counterexample count exceeds the bounded policy")
    changed = {row["id"]: row for row in endpoint_result["measurement"]["rows"]}
    known = {_key(anchor): anchor for anchor in pool.anchors}
    excluded = set(exclude)
    pending, records = [], []
    for prompt in endpoint_result["guard"]["prohibited_regressions"]:
        path = pool.trajectories[prompt]
        row = changed[prompt]
        if row["correct"]:
            raise ValueError("A correct response cannot supply a preservation counterexample")
        candidate = tuple(row["generated_token_ids"])
        position = next((index for index, pair in enumerate(zip(path.token_ids, candidate)) if pair[0] != pair[1]), None)
        key = prompt, position
        if position is None or key not in known or key in excluded:
            records.append({"prompt_id": prompt, "position": position, "eligible": False,
                            "reason": "no_new_positive_margin_on_path_divergence"})
            continue
        pending.append((known[key], candidate[position], row["status"]))
    # Recheck all eligible divergences before limiting the number inserted. This makes the
    # selection rule independent of whether earlier rows agree with uncached replay.
    accepted, calls = [], 0
    with torch.inference_mode():
        for batch in _batches(pending, batch_size):
            calls += 1
            logits = next_token_logits(runner, [_prefix(pool, anchor) for anchor, _, _ in batch])
            stats = fork_statistics(logits, [(anchor.token_id,) for anchor, _, _ in batch])
            for index, (anchor, candidate_token, status) in enumerate(batch):
                margin = float(stats.decision_margins[index])
                winner = int(stats.predicted_ids[index])
                agrees = winner == candidate_token and margin <= 0
                eligible = agrees and len(accepted) < limit
                if eligible:
                    accepted.append(anchor)
                records.append({**asdict(anchor), "eligible": eligible,
                                "reason": "inserted" if eligible else "refinement_limit" if agrees else "uncached_replay_disagreed",
                                "candidate_status": status, "candidate_token_id": candidate_token,
                                "candidate_replay_token_id": winner, "parent_margin": anchor.margin, "candidate_margin": margin})
            _check_memory(max_memory_bytes)
    return tuple(accepted), {"schema": "weightbridge.practice_counterexamples.v1", "counterexamples": records,
                            "resources": _resources(wall, cpu, calls, 0)}


def _rounded_merge_norm(source, dtype, left, right):
    delta = left.detach().cpu().numpy().astype(np.float64) @ right.detach().cpu().numpy().astype(np.float64)
    rounded = _to_float64(_from_float64(source + delta, dtype), dtype).reshape(source.shape) - source
    return float(np.linalg.norm(delta)), float(np.linalg.norm(rounded)), int(np.count_nonzero(rounded))


def fit_projected_lora(adapter, runner, plan, training_paths, *, radius, steps=4, lr=.005, batch_size=8, rank=4, seed=0):
    """Train native-width Adam LoRA with a rounded merge check after each optimizer step."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    paths = tuple(training_paths)
    limits = plan.limits
    ref, module, raw = _region(adapter, runner, plan, torch)
    out_width, in_width = ref.shape
    if (not 2 <= len(paths) <= 64 or type(steps) is not int or not 1 <= steps <= 32
            or type(rank) is not int or not 1 <= rank <= min(limits.max_rank, *ref.shape)
            or not isfinite(radius) or not 0 < radius <= limits.max_update_norm or not isfinite(lr) or not 0 < lr <= 1
            or plan.max_columns < in_width or out_width * in_width > limits.max_update_entries):
        raise ValueError("Projected LoRA exceeds its bounded affine authorization")
    batches = tuple(_batches(paths, batch_size))
    if len(batches) * steps > min(limits.max_forward_calls, limits.max_backward_calls):
        raise ValueError("Projected LoRA exceeds its compute authorization")
    token_count = sum(len(row.scored_ids) for row in paths)
    if not token_count or any(not row.scored_ids for row in paths):
        raise ValueError("Training requires nonempty validated completion paths")
    _check_memory(limits.max_memory_bytes)
    source = _to_float64(raw, ref.dtype).reshape(ref.shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    right = torch.nn.Parameter(((torch.rand((rank, in_width), generator=generator) * 2 - 1)
                                / np.sqrt(in_width)).to(module.weight.device))
    left = torch.nn.Parameter(torch.zeros((out_width, rank), device=module.weight.device, dtype=torch.float32))
    optimizer = torch.optim.AdamW((left, right), lr=lr, weight_decay=0.)
    history, calls = [], 0

    def hook(_, args, output):
        return output + ((args[0].float() @ right.T) @ left.T).to(output.dtype)

    with _only_train(runner.model, ()):
        handle = module.register_forward_hook(hook)
        try:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                loss = 0.
                for batch in batches:
                    nll = -torch.cat(score_batch(runner, batch)).sum() / token_count
                    if not torch.isfinite(nll).item():
                        raise ValueError("Nonfinite supervised LoRA loss")
                    nll.backward()
                    calls += 1
                    loss += float(nll.detach())
                    _check_memory(limits.max_memory_bytes)
                optimizer.step()
                proposed, rounded, count = _rounded_merge_norm(source, ref.dtype, left, right)
                if not isfinite(proposed) or not isfinite(rounded):
                    raise ValueError("Nonfinite proposed LoRA merge")
                unrounded_before, rounded_before = proposed, rounded
                projections = 0
                with torch.no_grad():
                    # The frozen float radius supplies headroom. Check the separate stored cap directly.
                    while (proposed > radius or rounded > limits.max_update_norm) and projections < 8:
                        float_scale = radius / proposed if proposed > radius else 1.
                        stored_scale = limits.max_update_norm / rounded if rounded > limits.max_update_norm else 1.
                        left.mul_(.999 * min(float_scale, stored_scale))
                        proposed, rounded, count = _rounded_merge_norm(source, ref.dtype, left, right)
                        projections += 1
                    if proposed > radius or rounded > limits.max_update_norm:
                        left.zero_()
                        proposed, rounded, count = 0., 0., 0
                history.append({"mean_token_nll": loss, "proposed_unrounded_norm": unrounded_before,
                                "proposed_stored_norm": rounded_before, "unrounded_norm": proposed,
                                "realized_update_norm": rounded, "changed_entries": count,
                                "projection_steps": projections})
                _check_memory(limits.max_memory_bytes)
        finally:
            handle.remove()
            if _tensor_bytes(module.weight, torch) != raw:
                raise RuntimeError("Projected LoRA mutated the parent affine tensor")
    update = AffineUpdate(tuple(range(in_width)), left.detach().cpu().numpy().astype(np.float64),
                          right.detach().cpu().numpy().astype(np.float64), "lora_adam")
    return update, {"schema": "weightbridge.projected_lora.v1", "reason": "trained", "rank": rank,
                    "radius": radius, "stored_norm_cap": limits.max_update_norm,
                    "steps": steps, "seed": seed, "learning_rate": lr,
                    "trainable_parameters": rank * sum(ref.shape), "training_rows": len(paths),
                    "training_tokens_per_step": token_count, "step_history": history,
                    "parent_restored_exactly": True, "resources": _resources(wall, cpu, calls, calls)}
