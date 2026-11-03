"""Bounded affine updates for the host's architecture-specific AWL plan."""
from __future__ import annotations

import hashlib
from math import ceil
import time

import numpy as np

from ..patches.huggingface import _from_float64, _to_float64
from ..provenance.resources import current_rss_bytes, peak_rss_bytes
from .content_scoring import prepare_continuations, score_batch


def _rounded(matrix, update, dtype):
    intended = matrix + update
    return _to_float64(_from_float64(intended, dtype), dtype).reshape(matrix.shape)


def _patch(plan, tensor_hash, columns, values):
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": plan.checkpoint_sha256,
        "reason": f"AWL v3 {plan.method} in a validated affine output",
        "tensor_sha256": {plan.region.tensor_name: tensor_hash},
        "slices": [{"tensor": plan.region.tensor_name, "axis": "columns", "indices": columns,
                    "op": "set", "values": values.T.tolist()}],
    }


def fit_candidate(adapter, runner, plan, rows, *, max_input_tokens=96, batch_size=8):
    """Construct one patch using authorized practice rows and a compiled plan.

    The live module is restored after every stored-precision probe. The host
    owns candidate application and all independent evaluation.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    limits, region = plan.limits, plan.region
    checkpoint = adapter.checkpoint
    if (region is None or checkpoint.checkpoint_sha256() != plan.checkpoint_sha256
            or plan.method not in {"first_order", "constrained", "gradient", "direct_low_rank"}):
        raise ValueError("AWL update requires a bound affine execution plan")
    ref = checkpoint.tensors[region.tensor_name]
    module = runner.model.get_submodule(region.module_name)
    if not isinstance(module, torch.nn.Linear) or tuple(module.weight.shape) != ref.shape:
        raise ValueError("Live affine module differs from the checkpoint plan")
    output_width, input_width = ref.shape
    width = min(16, input_width, limits.max_update_entries // output_width)
    if width < 1 or limits.max_candidates < 1 or limits.max_rank < 1 or limits.max_update_norm <= 0:
        raise ValueError("Host budget cannot support a rank-one column update")
    if ref.nbytes > limits.max_memory_bytes // 8:
        raise ValueError("Source affine tensor exceeds the bounded workspace")

    def check_memory():
        rss = current_rss_bytes()
        if rss is not None and rss > limits.max_memory_bytes:
            raise ValueError("Update process exceeds the host memory budget")

    check_memory()
    raw = checkpoint.read_raw(region.tensor_name, max_bytes=ref.nbytes)
    source = _to_float64(raw, ref.dtype).reshape(ref.shape)
    tensor_hash = hashlib.sha256(raw).hexdigest()
    if not np.array_equal(module.weight.detach().cpu().double().numpy(), source):
        raise ValueError("Live affine values differ from the bound checkpoint precision")
    calls = {"forward_calls": 0, "backward_calls": 0, "candidate_probes": 0}

    def receipt(reason, **extra):
        return {"schema": "weightbridge.universal_update.v1", "method": plan.method, "reason": reason,
                "tensor_name": region.tensor_name, "source_dtype": ref.dtype,
                **calls, **extra, "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}

    def realized(columns, delta):
        base_values = source[:, columns]
        after = _rounded(base_values, delta, ref.dtype)
        change = after - base_values
        norm = float(np.linalg.norm(change))
        changed = int(np.count_nonzero(change))
        if not changed or not np.isfinite(norm) or norm > limits.max_update_norm:
            return None
        return after, {"planned_rank": 1, "changed_entries": changed, "touched_entries": change.size,
                       "realized_update_norm": norm, "max_abs_delta": float(np.max(np.abs(change)))}

    if plan.method == "direct_low_rank":
        columns = list(range(min(width, 2)))
        left = np.zeros(output_width)
        left[0] = min(.03125, limits.max_update_norm / 4)
        right = np.linspace(1., .5, len(columns))
        stored = realized(columns, np.outer(left, right))
        if stored is None:
            return None, receipt("no_stored_change")
        after, evidence = stored
        return _patch(plan, tensor_hash, columns, after), receipt("engineering_probe", **evidence)

    if (type(batch_size) is not int or not 2 <= batch_size <= 16 or batch_size % 2
            or not 1 <= len(rows) <= 64
            or any(type(row.get("preserve")) is not bool for row in rows)):
        raise ValueError("Expected at most 64 practice rows and a bounded even batch size")
    preservation = torch.tensor([row["preserve"] for row in rows], device=runner.device, dtype=torch.bool)
    if not preservation.any().item() or preservation.all().item():
        raise ValueError("Numerical adaptation requires target and preservation practice")
    prepared = prepare_continuations(runner.tokenizer, [row["prompt"] for row in rows],
                                     [row["answers"] for row in rows], max_input_tokens=max_input_tokens)
    batches = ceil(len(prepared) / batch_size)
    probe_scales = (1., .5)[:limits.max_candidates]
    passes = 4 if plan.method == "gradient" else 1
    backward = passes + int(plan.method == "constrained")
    if (batches * (passes + 1 + len(probe_scales)) > limits.max_forward_calls
            or backward > limits.max_backward_calls):
        return None, receipt("insufficient_solver_budget")

    def margins():
        groups = []
        for start in range(0, len(prepared), batch_size):
            check_memory()
            token_scores = score_batch(runner, prepared[start:start + batch_size])
            calls["forward_calls"] += 1
            totals = torch.stack([scores.sum() for scores in token_scores])
            groups.append(totals[::2] - totals[1::2])
        check_memory()
        return torch.cat(groups)

    def derivative(loss, *, retain_graph=False):
        gradient, = torch.autograd.grad(loss, module.weight, retain_graph=retain_graph)
        calls["backward_calls"] += 1
        check_memory()
        return gradient.detach().float()

    def score_stored(columns, after):
        saved = module.weight[:, columns].detach().clone()
        with torch.no_grad():
            module.weight[:, columns] = torch.as_tensor(after, device=runner.device, dtype=module.weight.dtype)
            try:
                return margins()
            finally:
                module.weight[:, columns] = saved

    parameters = list(runner.model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(parameter is module.weight)
    try:
        with torch.no_grad():
            base_margins = margins()
        observed = margins()
        target_loss = torch.nn.functional.softplus(-observed[~preservation]).mean()
        target_gradient = derivative(target_loss, retain_graph=plan.method == "constrained")
        if not torch.isfinite(target_gradient).all().item():
            return None, receipt("nonfinite_sensitivity")
        importance = torch.linalg.vector_norm(target_gradient, dim=0).cpu().numpy()
        columns = sorted(np.argsort(-importance, kind="stable")[:width].tolist())
        direction = -target_gradient[:, columns].cpu().numpy().astype(float)
        if plan.method == "constrained":
            preservation_gradient = derivative(-observed[preservation].mean())[:, columns].cpu().numpy()
            if not np.isfinite(preservation_gradient).all():
                return None, receipt("nonfinite_preservation_sensitivity")
            adverse = float(np.sum(direction * preservation_gradient))
            norm_squared = float(np.sum(preservation_gradient**2))
            if adverse > 0 and norm_squared > 0:
                direction -= adverse / norm_squared * preservation_gradient
        if np.linalg.norm(direction) <= 1e-12:
            return None, receipt("no_local_direction")
        left, _, right = np.linalg.svd(direction, full_matrices=False)
        radius = .95 * limits.max_update_norm
        delta = radius * np.outer(left[:, 0], right[0])
        if plan.method == "gradient":
            # Rank-one Adam tuning on the same columns, initialized from the
            # leading practice direction. Parent probabilities anchor KL.
            # Update factors in float32 and measure probes at stored precision.
            u = torch.nn.Parameter(torch.zeros(output_width, device=runner.device))
            v = torch.nn.Parameter(torch.as_tensor(right[0], dtype=torch.float32, device=runner.device))
            optimizer = torch.optim.Adam((u, v), lr=radius / np.sqrt(output_width))
            parent = base_margins[preservation].sigmoid()
            saved = module.weight[:, columns].detach().clone()
            try:
                for _ in range(passes - 1):
                    observed = margins()
                    target_loss = torch.nn.functional.softplus(-observed[~preservation]).mean()
                    preserve_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        observed[preservation], parent)
                    gradient = derivative(target_loss + preserve_loss)[:, columns]
                    if not torch.isfinite(gradient).all().item():
                        return None, receipt("nonfinite_gradient")
                    u.grad, v.grad = gradient @ v.detach(), gradient.T @ u.detach()
                    optimizer.step()
                    with torch.no_grad():
                        length = u.norm() * v.norm()
                        if length.item() > radius:
                            u.mul_(radius / length)
                        module.weight[:, columns] = saved + torch.outer(u, v).to(saved.dtype)
                delta = torch.outer(u, v).detach().cpu().numpy().astype(float)
            finally:
                with torch.no_grad():
                    module.weight[:, columns] = saved

        before = base_margins.detach().cpu().numpy()
        preserve_rows = preservation.cpu().numpy()
        baseline_loss = float(np.logaddexp(0, -before[~preserve_rows]).mean())
        trials, best = [], None
        for scale in probe_scales:
            calls["candidate_probes"] += 1
            stored = realized(columns, delta * scale)
            if stored is None:
                trials.append({"scale": scale, "reason": "no_bounded_stored_change"})
                continue
            after, evidence = stored
            measured = score_stored(columns, after).cpu().numpy()
            stable = bool(np.all((before[preserve_rows] > 0) == (measured[preserve_rows] > 0)))
            target_regressions = int(np.count_nonzero((before > 0) & (measured <= 0) & ~preserve_rows))
            corrections = int(np.count_nonzero((before <= 0) & (measured > 0) & ~preserve_rows))
            loss = float(np.logaddexp(0, -measured[~preserve_rows]).mean())
            passed = stable and target_regressions == 0 and corrections > 0 and loss < baseline_loss
            trial = {"scale": scale, "corrections": corrections, "target_regressions": target_regressions,
                     "preservation_stable": stable, "loss": loss, "passed": passed, **evidence}
            trials.append(trial)
            if passed and (best is None or (corrections, -loss) > (best[0]["corrections"], -best[0]["loss"])):
                best = trial, after
        if best is None:
            return None, receipt("practice_guard", baseline_loss=baseline_loss, trials=trials)
        return _patch(plan, tensor_hash, columns, best[1]), receipt("practice_pass", selected=best[0], trials=trials)
    finally:
        for parameter, required in zip(parameters, requires_grad, strict=True):
            parameter.requires_grad_(required)
