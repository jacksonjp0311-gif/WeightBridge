"""Bounded affine updates from observed greedy forks and parent trajectories."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from math import isfinite
import time

import numpy as np

from ..patches.huggingface import _from_float64, _to_float64
from ..provenance.resources import current_rss_bytes, peak_rss_bytes
from .content_scoring import score_batch
from .endpoint_update import _first_divergences, _live_probe, _tensor_bytes
from .universal_update import _patch


@dataclass(frozen=True)
class AffineUpdate:
    columns: tuple[int, ...]
    left: np.ndarray
    right: np.ndarray
    method: str

    @property
    def rank(self):
        return self.left.shape[1]

    def delta(self):
        return self.left @ self.right


def _resources(wall, cpu, forwards, backwards, **extra):
    return {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
            "peak_rss_bytes": peak_rss_bytes(), "forward_calls": forwards, "backward_calls": backwards, **extra}


def _check_memory(limit):
    rss = current_rss_bytes()
    if rss is not None and rss > limit:
        raise ValueError("Fork update exceeded the host memory budget")


def _region(adapter, runner, plan, torch, checkpoint_sha256=None):
    region = plan.region
    checkpoint = adapter.checkpoint
    observed = checkpoint.checkpoint_sha256() if checkpoint_sha256 is None else checkpoint_sha256
    if region is None or observed != plan.checkpoint_sha256 or runner.model.training:
        raise ValueError("Fork update requires a checkpoint-bound affine plan")
    ref = checkpoint.tensors[region.tensor_name]
    module = runner.model.get_submodule(region.module_name)
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
    if (not isinstance(module, torch.nn.Linear) or tuple(module.weight.shape) != ref.shape
            or region.shape != ref.shape or module.weight.dtype != dtypes.get(ref.dtype)
            or ref.dtype != region.dtype or ref.nbytes > plan.limits.max_memory_bytes // 8):
        raise ValueError("Live affine region does not match the storage and resource plan")
    raw = checkpoint.read_raw(region.tensor_name, max_bytes=ref.nbytes)
    if _tensor_bytes(module.weight, torch) != raw:
        raise ValueError("Live affine bytes differ from the checkpoint")
    return ref, module, raw


@contextmanager
def _only_train(model, parameters):
    allowed = {id(parameter) for parameter in parameters}
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        for parameter, _ in flags:
            parameter.requires_grad_(id(parameter) in allowed)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def _batches(items, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("Batch size must be between one and sixteen")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _usable_forks(forks):
    return tuple(fork for fork in forks if fork.status in {"semantic", "syntactic"}
                 and fork.replay_status == "matched")


def _check_fork_identity(forks, checkpoint_sha256):
    if any(fork.checkpoint_sha256 != checkpoint_sha256 for fork in forks):
        raise ValueError("Fork evidence belongs to another checkpoint")


def _trajectory_scores(logits, tokens, torch):
    wanted = torch.as_tensor(tokens, device=logits.device).reshape(-1, 1)
    scores = logits.float()
    rivals = scores.scatter(1, wanted, -torch.inf).max(dim=1).values
    return scores.gather(1, wanted).squeeze(1) - rivals


def _trajectory_report(trajectories, values, torch):
    return [{"token_ids": list(row.token_ids), "margins": _trajectory_scores(logits, row.token_ids, torch).tolist(),
             "replay_argmax_ids": logits.argmax(dim=1).tolist()}
            for row, logits in zip(trajectories, values, strict=True)]


def _rank_one(target, margin_gradient, guards, initial_margins, radius, floor, width):
    """Preserve rank one by projecting the output factor with the input factor fixed."""
    column_norms = np.linalg.norm(target, axis=0)
    columns = tuple(sorted(np.argsort(-column_norms, kind="stable")[:width].tolist()))
    target_slice = target[:, columns]
    gradient_norm = float(np.linalg.norm(target_slice))
    if not isfinite(gradient_norm) or gradient_norm <= 1e-12:
        return {}, {"reason": "no_target_gradient_in_authorized_columns", "selectivity_score": None}
    left, _, right = np.linalg.svd(-target_slice, full_matrices=False)
    u, v = left[:, 0], right[0]
    # Normalize each preservation gradient so its size cannot determine numerical rank.
    guard_rows = np.asarray([gradient[:, columns] @ v for gradient in guards], dtype=np.float64)
    active = guard_rows[np.linalg.norm(guard_rows, axis=1) > 1e-12] if len(guard_rows) else guard_rows
    subspace = np.empty((0, target_slice.shape[0]))
    if len(active):
        normalized = active / np.linalg.norm(active, axis=1, keepdims=True)
        _, singular_values, right_vectors = np.linalg.svd(normalized, full_matrices=False)
        subspace = right_vectors[singular_values > singular_values[0] * 1e-8]
    projected = u - subspace.T @ (subspace @ u)
    retained = float(np.linalg.norm(projected))
    target_slope = margin_gradient[:, columns] @ v

    def witness(direction):
        guard_slopes = guard_rows @ direction if len(guards) else np.asarray([])
        margins = np.asarray(initial_margins)
        headroom = margins - floor
        radius_bound = radius
        if np.any(headroom < 0):
            radius_bound = 0.0
        elif np.any(guard_slopes < -1e-12):
            negative = guard_slopes < -1e-12
            radius_bound = max(0.0, min(radius, float(np.min(headroom[negative] / -guard_slopes[negative]))))
        return {"target_margin_gain_at_radius": float(target_slope @ direction) * radius,
                "target_margin_gain_at_feasible_radius": float(target_slope @ direction) * radius_bound,
                "predicted_protected_margins": (margins + radius * guard_slopes).tolist(),
                "feasible_radius": radius_bound,
                "sampled_constraints_satisfied_at_radius": bool(np.all(margins + radius * guard_slopes >= floor - 1e-9))}

    direct = witness(u)
    directions = {"fork": AffineUpdate(columns, u[:, None], v[None, :], "first_fork")}
    details = {"reason": "direction_available", "columns": list(columns),
               "selected_column_target_gradient_norm": gradient_norm,
               "full_region_target_gradient_norm": float(np.linalg.norm(target)),
               "rank_one_output_factor_reachability": retained,
               "sampled_constraint_rank": int(len(subspace)), "unconstrained": direct,
               "selectivity_score": direct["target_margin_gain_at_feasible_radius"] if len(guards) else None,
               "protected": None}
    if len(guards) and retained > 1e-9:
        protected_u = projected / retained
        protected = witness(protected_u)
        if protected["target_margin_gain_at_feasible_radius"] > 0 and protected["sampled_constraints_satisfied_at_radius"]:
            directions["protected"] = AffineUpdate(columns, protected_u[:, None], v[None, :], "trajectory_projected")
        details["protected"] = protected
        if "protected" not in directions:
            details["protected_failure"] = "no_target_improvement_in_preservation_compatible_direction"
    elif len(guards):
        details["protected_failure"] = "sampled_preservation_conflict"
    return directions, details


def scan_fork_regions(adapter, runner, plans, forks, trajectories, *, radius, margin_floor=0., batch_size=8,
                      max_constraints=4, objective="cross_entropy"):
    """Share bounded VJPs across authorized regions; return measured and predicted quantities separately."""
    import torch
    from .generative_fork import fork_losses, fork_statistics, next_token_logits, trajectory_logits

    wall, cpu = time.perf_counter(), time.process_time()
    plans, forks, trajectories = tuple(plans), _usable_forks(forks), tuple(trajectories)
    if (not 1 <= len(plans) <= 4 or len({plan.region.layer_index for plan in plans}) != len(plans)
            or not 0 < radius <= min(plan.limits.max_update_norm for plan in plans)
            or not isfinite(margin_floor) or margin_floor < 0 or type(max_constraints) is not int
            or not 1 <= max_constraints <= 4 or not 1 <= len(forks) <= 32 or not 1 <= len(trajectories) <= 16):
        raise ValueError("Fork region scan exceeds the bounded research design")
    target_batches = tuple(_batches(forks, batch_size))
    trajectory_batches = tuple(_batches(trajectories, batch_size))
    if (len(target_batches) + len(trajectory_batches) + 1 > min(p.limits.max_forward_calls for p in plans)
            or 2 + max_constraints > min(p.limits.max_backward_calls for p in plans)):
        raise ValueError("Fork scan exceeds the AWL forward/backward allowance")
    memory = min(p.limits.max_memory_bytes for p in plans)
    _check_memory(memory)
    source_sha = adapter.checkpoint.checkpoint_sha256()
    _check_fork_identity(forks, source_sha)
    resolved = [_region(adapter, runner, plan, torch, source_sha) for plan in plans]
    weights = tuple(module.weight for _, module, _ in resolved)
    forwards, backwards = 0, 0
    with torch.inference_mode():
        trajectory_values = []
        for batch in trajectory_batches:
            trajectory_values.extend(trajectory_logits(runner, batch))
            forwards += 1
            _check_memory(memory)
        parent_paths = _trajectory_report(trajectories, trajectory_values, torch)
    del trajectory_values
    # One weakest replay-consistent position per trajectory, ordered by exact parent margin.
    weakest = []
    for index, path in enumerate(parent_paths):
        matched = [(margin, position) for position, margin in enumerate(path["margins"])
                   if margin >= margin_floor and path["replay_argmax_ids"][position] == path["token_ids"][position]]
        if matched:
            margin, position = min(matched)
            weakest.append((margin, index, position))
    anchors = sorted(weakest)[:max_constraints]

    def derivative(loss, retain_graph=False):
        nonlocal backwards
        result = torch.autograd.grad(loss, weights, retain_graph=retain_graph, allow_unused=True)
        backwards += 1
        _check_memory(memory)
        return [np.asarray((g if g is not None else torch.zeros_like(w)).detach().float().cpu(), dtype=np.float64)
                for w, g in zip(weights, result, strict=True)]

    with _only_train(runner.model, weights):
        losses, target_margins = [], []
        for batch in target_batches:
            logits = next_token_logits(runner, [row.prefix_ids for row in batch])
            forwards += 1
            accepted = [row.accepted_token_ids for row in batch]
            actual = [row.actual_token_id for row in batch]
            stats = fork_statistics(logits, accepted, actual_token_ids=actual)
            losses.append(fork_losses(logits, accepted, objective=objective, actual_token_ids=actual))
            target_margins.append(stats.decision_margins)
            _check_memory(memory)
        mean_margin = torch.cat(target_margins).mean()
        parent_target_margin = float(mean_margin.detach())
        target_grad = derivative(torch.cat(losses).mean(), retain_graph=True)
        margin_grad = derivative(mean_margin)
        del logits, stats, losses, target_margins, mean_margin
        guards = []
        if anchors:
            sequences = [trajectories[index].prompt_token_ids + trajectories[index].token_ids[:position]
                         for _, index, position in anchors]
            decisions = next_token_logits(runner, sequences)
            forwards += 1
            wanted = [trajectories[index].token_ids[position] for _, index, position in anchors]
            guard_margins = _trajectory_scores(decisions, wanted, torch)
            measured_anchors = []
            for number, (_, index, position) in enumerate(anchors):
                measured = float(guard_margins[number].detach())
                if measured >= margin_floor and int(decisions[number].argmax().detach()) == wanted[number]:
                    measured_anchors.append((measured, index, position))
                    guards.append(derivative(guard_margins[number], retain_graph=number < len(anchors) - 1))
            anchors = measured_anchors

    directions, regions = {}, []
    for index, plan in enumerate(plans):
        ref, module, raw = resolved[index]
        if _tensor_bytes(module.weight, torch) != raw:
            raise RuntimeError("Sensitivity scan mutated an authorized affine tensor")
        width = min(16, ref.shape[1], plan.max_columns, plan.limits.max_update_entries // ref.shape[0])
        if width < 1 or plan.limits.max_rank < 1:
            raise ValueError("The plan cannot hold a rank-one fork update")
        layer = plan.region.layer_index
        layer_directions, details = _rank_one(target_grad[index], margin_grad[index], [g[index] for g in guards],
                                              [margin for margin, _, _ in anchors], radius, margin_floor, width)
        directions[layer] = layer_directions
        for kind in ("unconstrained", "protected"):
            if details.get(kind) is not None:
                gain = details[kind]["target_margin_gain_at_feasible_radius"]
                details[kind]["predicted_mean_target_margin"] = parent_target_margin + gain
                details[kind]["aggregate_margin_crossing_predicted"] = bool(parent_target_margin + gain > 0)
        regions.append({"layer_index": layer, "tensor_name": plan.region.tensor_name, **details})
    eligible = [row for row in regions if row["selectivity_score"] is not None and row["selectivity_score"] > 0]
    selected = max(eligible, key=lambda row: (row["selectivity_score"], -row["layer_index"])) if eligible else None
    selected_index = selected["layer_index"] if selected else None
    return selected_index, directions, {"schema": "weightbridge.fork_region_scan.v1", "selected_layer": selected_index,
        "selection_rule": "Maximum predicted mean target margin gain along the raw rank-one direction within the radius and sampled preservation floor",
        "limitation": "Local aggregate directional witness; failure to cross does not prove infeasibility outside the tested direction, and crossing does not establish per-item or nonlinear correction",
        "radius": radius, "parent_mean_target_margin": parent_target_margin,
        "reason": "direction_selected" if selected else "no_selective_first_order_witness",
        "regions": regions, "parent_trajectories": parent_paths,
        "protected_anchors": [{"trajectory_index": i, "position": p, "margin": m} for m, i, p in anchors],
        "resources": _resources(wall, cpu, forwards, backwards, shared_region_count=len(plans))}


def _measure_logits(runner, forks, trajectories, batch_size, max_memory_bytes, torch):
    from .generative_fork import next_token_logits, trajectory_logits

    target_values, trajectory_values = [], []
    forwards = 0
    for batch in _batches(forks, batch_size):
        target_values.append(next_token_logits(runner, [row.prefix_ids for row in batch]).float().cpu())
        forwards += 1
        _check_memory(max_memory_bytes)
    for batch in _batches(trajectories, batch_size):
        trajectory_values.extend(value.float().cpu() for value in trajectory_logits(runner, batch))
        forwards += 1
        _check_memory(max_memory_bytes)
    return (torch.cat(target_values) if target_values else None, tuple(trajectory_values)), forwards


@dataclass(frozen=True)
class ProbeReference:
    checkpoint_sha256: str
    model: object
    forks: tuple
    trajectories: tuple
    values: tuple
    resources: dict


def prepare_probe_reference(adapter, runner, forks, trajectories, *, batch_size=8, max_memory_bytes):
    """Cache read-only parent logits for candidate comparisons on the same runner."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    forks, trajectories = _usable_forks(forks), tuple(trajectories)
    if len(forks) > 32 or len(trajectories) > 16:
        raise ValueError("Parent logit cache exceeds the bounded research design")
    source_sha = adapter.checkpoint.checkpoint_sha256()
    _check_fork_identity(forks, source_sha)
    with torch.inference_mode():
        values, forwards = _measure_logits(runner, forks, trajectories, batch_size, max_memory_bytes, torch)
    return ProbeReference(source_sha, runner.model, forks, trajectories, values,
                          _resources(wall, cpu, forwards, 0))


def _fork_comparison(forks, parent, candidate):
    from .generative_fork import fork_statistics

    if not forks:
        return None
    accepted = [fork.accepted_token_ids for fork in forks]
    actual = [fork.actual_token_id for fork in forks]
    a = fork_statistics(parent, accepted, actual_token_ids=actual)
    b = fork_statistics(candidate, accepted, actual_token_ids=actual)
    return {"parent_cross_entropy": float(a.cross_entropy.mean()), "candidate_cross_entropy": float(b.cross_entropy.mean()),
            "cross_entropy_gain": float((a.cross_entropy - b.cross_entropy).mean()),
            "parent_margin_loss": float(a.margin_loss.mean()), "candidate_margin_loss": float(b.margin_loss.mean()),
            "parent_actual_margin_loss": float(a.actual_margin_loss.mean()),
            "candidate_actual_margin_loss": float(b.actual_margin_loss.mean()),
            "parent_decision_margins": a.decision_margins.tolist(), "candidate_decision_margins": b.decision_margins.tolist(),
            "candidate_predicted_ids": b.predicted_ids.tolist(), "candidate_strongest_rival_ids": b.strongest_rival_ids.tolist(),
            "accepted_next_token_count": sum(int(token) in choices for token, choices in zip(b.predicted_ids, accepted, strict=True)),
            "probe_count": len(forks)}


def _trajectory_comparison(trajectories, parent, candidate, torch):
    rows = []
    for row, before, after in zip(trajectories, parent, candidate, strict=True):
        parent_margin = _trajectory_scores(before, row.token_ids, torch)
        margin = _trajectory_scores(after, row.token_ids, torch)
        parent_lp = before.double().log_softmax(dim=1)
        after_lp = after.double().log_softmax(dim=1)
        kl = (parent_lp.exp() * (parent_lp - after_lp)).sum(dim=1).clamp_min(0.)
        rows.append({"token_ids": list(row.token_ids), "parent_margins": parent_margin.tolist(),
                     "candidate_margins": margin.tolist(), "candidate_argmax_ids": after.argmax(dim=1).tolist(),
                     "margin_shrink_penalty": (parent_margin - margin).clamp_min(0.).tolist(), "kl": kl.tolist()})
    count = sum(len(row["token_ids"]) for row in rows)
    if not count:
        return None
    return {"positions": count, "trajectories": rows,
            "candidate_positive_margin_positions": sum(sum(m > 0 for m in row["candidate_margins"]) for row in rows),
            "new_nonpositive_margin_positions": sum(sum(a > 0 and b <= 0 for a, b in
                zip(row["parent_margins"], row["candidate_margins"], strict=True)) for row in rows),
            "mean_kl": sum(sum(row["kl"]) for row in rows) / count,
            "mean_margin_shrink_penalty": sum(sum(row["margin_shrink_penalty"]) for row in rows) / count}


def measure_probe_reference(runner, reference, *, batch_size=8, max_memory_bytes):
    """Measure an active probe against the cached parent, including legacy probes."""
    import torch

    if reference.model is not runner.model:
        raise ValueError("Parent logit cache belongs to another runtime")
    wall, cpu = time.perf_counter(), time.process_time()
    with torch.inference_mode():
        after, forwards = _measure_logits(runner, reference.forks, reference.trajectories, batch_size,
                                          max_memory_bytes, torch)
        fork_result = _fork_comparison(reference.forks, reference.values[0], after[0])
        paths = _trajectory_comparison(reference.trajectories, reference.values[1], after[1], torch)
    return {"fork": fork_result, "trajectory": paths, "resources": _resources(wall, cpu, forwards, 0)}


def probe_update(adapter, runner, plan, update, *, radius, seen, evaluate_candidate, forks=(), trajectories=(),
                 baseline_endpoint=None, batch_size=8, reference=None):
    """Evaluate a unique stored-precision update and emit a patch only after the host endpoint passes."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    limits = plan.limits
    _check_memory(limits.max_memory_bytes)
    ref, module, raw = _region(adapter, runner, plan, torch)
    columns = update.columns
    if (not columns or tuple(sorted(set(columns))) != columns or columns[0] < 0 or columns[-1] >= ref.shape[1]
            or len(columns) > plan.max_columns or ref.shape[0] * len(columns) > limits.max_update_entries
            or update.left.ndim != 2 or update.right.ndim != 2 or update.left.shape[0] != ref.shape[0]
            or update.right.shape != (update.rank, len(columns)) or not 1 <= update.rank <= limits.max_rank
            or not np.isfinite(update.left).all() or not np.isfinite(update.right).all()):
        raise ValueError("Fork update violates the affine plan")
    if radius is not None and (not isfinite(radius) or not 0 < radius <= limits.max_update_norm):
        raise ValueError("Requested probe radius exceeds the affine plan")
    scale = 1.
    delta = update.delta()
    natural_norm = float(np.linalg.norm(delta))
    if radius is not None and natural_norm:
        scale = radius / natural_norm
    source = _to_float64(raw, ref.dtype).reshape(ref.shape)
    base_columns = source[:, columns]
    rounded = _to_float64(_from_float64(base_columns + scale * delta, ref.dtype), ref.dtype).reshape(base_columns.shape)
    if update.method != "lora_adam":
        rounded[rounded == base_columns] = base_columns[rounded == base_columns]
    realized = rounded - base_columns
    norm = float(np.linalg.norm(realized))
    entries = int(np.count_nonzero(realized))
    evidence = {"schema": "weightbridge.fork_probe.v1", "method": update.method, "rank": update.rank,
                "tensor_name": plan.region.tensor_name, "layer_index": plan.region.layer_index,
                "dose": radius, "proposed_norm": natural_norm * scale, "realized_update_norm": norm,
                "changed_entries": entries, "touched_entries": realized.size,
                "max_abs_delta": float(np.abs(realized).max()), "restored_exactly": True}

    def result(reason, patch=None, forwards=0, **extra):
        return patch, {**evidence, **extra, "reason": reason,
                       "resources": _resources(wall, cpu, forwards, 0)}

    if not entries:
        return result("no_stored_change")
    if not isfinite(norm) or norm > limits.max_update_norm:
        return result("realized_norm_exceeds_budget")
    expected = source.copy()
    expected[:, columns] = rounded
    expected_raw = _from_float64(expected, ref.dtype)
    digest = hashlib.sha256(expected_raw).hexdigest()
    evidence["candidate_tensor_sha256"] = digest
    key = (plan.region.tensor_name, digest)
    if key in seen:
        return result("duplicate_stored_candidate")
    forks, trajectories = _usable_forks(forks), tuple(trajectories)
    if len(forks) > 32 or len(trajectories) > 16:
        raise ValueError("Fork probe evidence exceeds the bounded research design")
    _check_fork_identity(forks, plan.checkpoint_sha256)
    if reference is not None and (reference.checkpoint_sha256 != plan.checkpoint_sha256
            or reference.model is not runner.model or reference.forks != forks or reference.trajectories != trajectories):
        raise ValueError("Parent logit cache does not match this checkpoint and probe")
    batches = len(tuple(_batches(forks, batch_size))) + len(tuple(_batches(trajectories, batch_size)))
    if batches * (1 if reference is not None else 2) > limits.max_forward_calls:
        raise ValueError("Fork probe exceeds the AWL forward allowance")
    with torch.inference_mode():
        if reference is None:
            before, forwards = _measure_logits(runner, forks, trajectories, batch_size, limits.max_memory_bytes, torch)
        else:
            before, forwards = reference.values, 0
        with _live_probe(module.weight, columns, rounded, expected_raw, raw, torch):
            after, calls = _measure_logits(runner, forks, trajectories, batch_size, limits.max_memory_bytes, torch)
            forwards += calls
            fork_result = _fork_comparison(forks, before[0], after[0])
            paths = _trajectory_comparison(trajectories, before[1], after[1], torch)
            endpoint = evaluate_candidate()
            guard = endpoint.get("guard", {})
            if type(guard.get("passed")) is not bool:
                raise ValueError("Endpoint evaluator must return a boolean guard decision")
            if guard["passed"] and (type(guard.get("target_item_gain")) is not int or guard["target_item_gain"] < 1
                                    or guard.get("prohibited_regressions") or guard.get("lost_valid_presentations")):
                raise ValueError("Endpoint guard passed without positive gain and zero regressions")
            _check_memory(limits.max_memory_bytes)
    seen.add(key)
    patch = None
    if guard["passed"]:
        base_tensor_sha = hashlib.sha256(raw).hexdigest()
        if update.method == "lora_adam" and len(columns) == ref.shape[1]:
            patch = {"schema": "weightbridge.patch.v1", "base_sha256": plan.checkpoint_sha256,
                     "reason": "AWL merged Adam-trained affine LoRA", "tensor_sha256": {plan.region.tensor_name: base_tensor_sha},
                     "low_rank": [{"tensor": plan.region.tensor_name, "u": update.left.tolist(),
                                   "v": update.right.tolist(), "scale": scale}]}
        else:
            patch = _patch(plan, base_tensor_sha, list(columns), rounded)
    divergences = (_first_divergences(baseline_endpoint, endpoint["measurement"], runner.tokenizer)
                   if baseline_endpoint is not None else [])
    return result("endpoint_pass" if patch else "endpoint_rejected", patch, forwards,
                  fork=fork_result, trajectory=paths, endpoint=endpoint, first_divergences=divergences)


def fit_lora(adapter, runner, plan, training_paths, *, steps=4, lr=.005, batch_size=8, rank=1, seed=0):
    """Conventional supervised continuation tuning with additive LoRA and an unmodified parent."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    training_paths = tuple(training_paths)
    ref, module, raw = _region(adapter, runner, plan, torch)
    width = ref.shape[1]
    limits = plan.limits
    if (not 2 <= len(training_paths) <= 64 or type(steps) is not int or not 1 <= steps <= 32
            or type(rank) is not int or not 1 <= rank <= min(limits.max_rank, *ref.shape)
            or not isfinite(lr) or not 0 < lr <= 1 or plan.max_columns < width
            or ref.shape[0] * width > limits.max_update_entries):
        raise ValueError("LoRA configuration exceeds the bounded full affine baseline plan")
    batches = tuple(_batches(training_paths, batch_size))
    if len(batches) * steps > min(limits.max_forward_calls, limits.max_backward_calls):
        raise ValueError("LoRA training exceeds the AWL forward/backward allowance")
    token_count = sum(len(row.scored_ids) for row in training_paths)
    if token_count < 1 or any(not row.scored_ids for row in training_paths):
        raise ValueError("LoRA needs nonempty validated continuation paths")
    _check_memory(limits.max_memory_bytes)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # LoRA uses randomized down-projection and zero up-projection, alpha/rank=1.
    right = (torch.rand((rank, width), generator=generator) * 2 - 1) / np.sqrt(width)
    right = torch.nn.Parameter(right.to(module.weight.device))
    left = torch.nn.Parameter(torch.zeros((ref.shape[0], rank), device=module.weight.device, dtype=torch.float32))
    optimizer = torch.optim.AdamW((left, right), lr=lr, weight_decay=0.)
    losses, calls = [], 0

    def additive_adapter(_, args, output):
        return output + ((args[0].float() @ right.T) @ left.T).to(output.dtype)

    with _only_train(runner.model, ()):
        handle = module.register_forward_hook(additive_adapter)
        try:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                for batch in batches:
                    scores = score_batch(runner, batch)
                    nll = -torch.cat(scores).sum() / token_count
                    if not torch.isfinite(nll).item():
                        raise ValueError("Nonfinite supervised LoRA loss")
                    nll.backward()
                    calls += 1
                    total += float(nll.detach())
                    _check_memory(limits.max_memory_bytes)
                optimizer.step()
                losses.append(total)
        finally:
            handle.remove()
            if _tensor_bytes(module.weight, torch) != raw:
                raise RuntimeError("LoRA training mutated the parent affine tensor")
    update = AffineUpdate(tuple(range(width)), left.detach().cpu().numpy().astype(np.float64),
                          right.detach().cpu().numpy().astype(np.float64), "lora_adam")
    return update, {"schema": "weightbridge.fork_lora_training.v1", "method": "additive LoRA with AdamW",
        "layer_index": plan.region.layer_index, "rank": rank, "trainable_parameters": rank * sum(ref.shape),
        "input_columns": width, "steps": steps, "learning_rate": lr, "alpha_over_rank": 1., "seed": seed,
        "training_rows": len(training_paths), "training_tokens_per_step": token_count, "mean_token_nll_by_step": losses,
        "unrounded_merged_norm": float(np.linalg.norm(update.delta())),
        "parent_restored_exactly": True, "resources": _resources(wall, cpu, calls, calls)}
