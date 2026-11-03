"""Stored-precision affine dose probes guarded by actual endpoint generation."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
import hashlib
import inspect
from math import ceil, isfinite
import time

import numpy as np

from ..patches.huggingface import _from_float64, _to_float64
from ..provenance.resources import current_rss_bytes, peak_rss_bytes
from .content_scoring import _left_pad, _pad_id, prepare_continuations, score_batch
from .universal_update import _patch


def _tensor_bytes(tensor, torch) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


@contextmanager
def _live_probe(weight, columns, after, expected, parent, torch):
    """Check the entire affine payload and restore it on every exit path."""
    saved = weight.detach().clone()
    if _tensor_bytes(saved, torch) != parent:
        raise ValueError("Live affine tensor is not the exact parent before the probe")
    with torch.no_grad():
        try:
            weight[:, columns] = torch.as_tensor(after, device=weight.device, dtype=weight.dtype)
            if _tensor_bytes(weight, torch) != expected:
                raise ValueError("Live affine payload differs from the intended stored bytes")
            yield
            if _tensor_bytes(weight, torch) != expected:
                raise ValueError("Probe evaluator mutated the live affine tensor")
        finally:
            weight.copy_(saved)
            if _tensor_bytes(weight, torch) != parent:
                raise RuntimeError("Failed to restore exact live parent affine bytes")


def _first_divergences(parent: dict, candidate: dict, tokenizer) -> list[dict]:
    by_id = {row["id"]: row for row in parent["rows"]}
    if (len(by_id) != len(parent["rows"]) or len(by_id) != len(candidate["rows"])
            or by_id.keys() != {row["id"] for row in candidate["rows"]}):
        raise ValueError("Candidate endpoint IDs must match the parent roster")
    changes = []
    for row in candidate["rows"]:
        parent_row = by_id[row["id"]]
        before = parent_row["generated_token_ids"]
        after = row["generated_token_ids"]
        if before == after:
            continue
        index = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), min(len(before), len(after)))
        change = {"id": row["id"], "position": index,
                  "parent_token_id": before[index] if index < len(before) else None,
                  "candidate_token_id": after[index] if index < len(after) else None}
        if parent_row.get("completed") is True:
            boundary = _completion_position(tokenizer, before)
            if boundary is None:
                raise ValueError("Completed parent response lacks a token at the completion boundary")
            if index > boundary:
                change["parent_phase"] = "after_completion"
            elif index == boundary:
                change["parent_phase"] = "completion_boundary"
            else:
                change["parent_phase"] = "first_token" if index == 0 else "inside_answer"
        changes.append(change)
    return changes


def _completion_position(tokenizer, ids: Sequence[int]) -> int | None:
    for position, token in enumerate(ids):
        if token == tokenizer.eos_token_id or "\n" in tokenizer.decode(ids[:position + 1], skip_special_tokens=False):
            return position
    return None


def _completion_ids(tokenizer, ids: Sequence[int]) -> tuple[int, ...]:
    """Retain the accepted first line and the actual token ending that line."""
    if not 1 <= len(ids) <= 16 or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Distribution diagnostics need 1 to 16 observed generation tokens")
    boundary = _completion_position(tokenizer, ids)
    if boundary is None:
        raise ValueError("Preservation distribution prefix has no observed completion boundary")
    return tuple(ids[:boundary + 1])


def _distribution_batch(runner, prefixes, torch):
    """Full-vocabulary log probabilities along completed parent responses."""
    sequences = [prefix + answer for _, prefix, answer in prefixes]
    input_ids, mask = _left_pad(torch, sequences, _pad_id(runner.tokenizer), runner.device)
    keep = max(len(answer) for _, _, answer in prefixes) + 1
    forward = inspect.signature(runner.model.forward).parameters
    kwargs = {"use_cache": False}
    if "position_ids" in forward:
        kwargs["position_ids"] = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    if "logits_to_keep" in forward:
        kwargs["logits_to_keep"] = keep
    elif "num_logits_to_keep" in forward:
        kwargs["num_logits_to_keep"] = keep
    logits = runner.model(input_ids=input_ids, attention_mask=mask, **kwargs).logits
    if logits.ndim != 3 or logits.shape[0] != len(prefixes) or logits.shape[1] < keep:
        raise ValueError("Model returned insufficient logits for preservation distributions")
    distributions = tuple(logits[i, -len(answer) - 1:-1].double().log_softmax(-1).cpu()
                          for i, (_, _, answer) in enumerate(prefixes))
    if not all(torch.isfinite(distribution).all().item() for distribution in distributions):
        raise ValueError("Preservation distribution has nonfinite log probabilities")
    return distributions


def _top_tokens(log_probs, tokenizer):
    values, tokens = log_probs.topk(min(5, log_probs.shape[-1]))
    return [{"token_id": token, "text": tokenizer.decode([token], skip_special_tokens=False),
             "log_probability": value} for value, token in zip(values.tolist(), tokens.tolist(), strict=True)]


def _distribution_report(prefixes, parent, candidate, tokenizer):
    reports, all_positions = [], []
    for (row_id, _, answer), p, q in zip(prefixes, parent, candidate, strict=True):
        # Clamp tiny negative floating-point roundoff in an otherwise nonnegative KL.
        kl = (p.exp() * (p - q)).sum(-1).clamp_min(0).tolist()
        all_positions.extend(kl)
        reports.append({"id": row_id, "positions": [
            {"position": position, "parent_response_token_id": token, "kl": value,
             "parent_top": _top_tokens(p[position], tokenizer),
             "candidate_top": _top_tokens(q[position], tokenizer)}
            for position, (token, value) in enumerate(zip(answer, kl, strict=True))]})
    return {"mean_kl": float(np.mean(all_positions)), "positions": len(all_positions), "prefixes": reports}


def _margin_report(before, after, preserve):
    target = ~preserve
    baseline_loss = float(np.logaddexp(0, -before[target]).mean())
    loss = float(np.logaddexp(0, -after[target]).mean())
    stable = bool(np.array_equal(before[preserve] > 0, after[preserve] > 0))
    corrections = int(np.count_nonzero(target & (before <= 0) & (after > 0)))
    regressions = int(np.count_nonzero(target & (before > 0) & (after <= 0)))
    return {"passed": stable and corrections > 0 and regressions == 0 and loss < baseline_loss,
            "before": before.tolist(), "after": after.tolist(), "baseline_loss": baseline_loss, "loss": loss,
            "surrogate_gain": baseline_loss - loss, "corrections": corrections,
            "target_regressions": regressions, "preservation_stable": stable}


def fit_endpoint_probes(adapter, runner, plan, rows, *, evaluate_candidate: Callable[[], dict],
                        baseline_endpoint: dict, doses: Sequence[float], max_input_tokens: int = 96,
                        batch_size: int = 8, preservation_prefix_rows: Sequence[int] = (),
                        kl_weight: float = .2, norm_weight: float = .1):
    """Fit one rank-one direction and measure doses without opening evaluation data.

    The host supplies validated practice contrasts and a read-only callback that
    performs actual decoding. Only endpoint-passing probes yield patch documents.
    Full-vocabulary KL regularizes measured-point selection, not the fit gradient.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    limits, region = plan.limits, plan.region
    checkpoint = adapter.checkpoint
    if (region is None or plan.method not in {"first_order", "constrained"}
            or checkpoint.checkpoint_sha256() != plan.checkpoint_sha256):
        raise ValueError("Endpoint fitting requires a bound first-order affine execution plan")
    if (not 1 <= len(doses) <= limits.max_candidates
            or any(type(dose) not in (float, int) or not isfinite(dose) or not 0 < dose <= limits.max_update_norm
                   for dose in doses)
            or tuple(sorted(set(doses))) != tuple(doses)):
        raise ValueError("Endpoint probes need increasing positive doses within the authorized budget")
    if any(type(value) not in (float, int) or not isfinite(value) or value < 0 for value in (kl_weight, norm_weight)):
        raise ValueError("Distribution and norm penalty weights must be finite and nonnegative")
    if (not 1 <= len(rows) <= 64 or type(batch_size) is not int or not 2 <= batch_size <= 16
            or batch_size % 2 or any(type(row.get("preserve")) is not bool for row in rows)
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("Expected uniquely identified practice rows and a bounded even batch size")
    preserve = np.asarray([row["preserve"] for row in rows], dtype=bool)
    if not preserve.any() or preserve.all():
        raise ValueError("Endpoint fitting needs target and preservation practice contrasts")
    if (len(preservation_prefix_rows) > 4 or len(set(preservation_prefix_rows)) != len(preservation_prefix_rows)
            or any(type(i) is not int or not 0 <= i < len(rows) or not preserve[i] for i in preservation_prefix_rows)):
        raise ValueError("Distribution diagnostics need up to four distinct preservation row indices")

    ref = checkpoint.tensors[region.tensor_name]
    module = runner.model.get_submodule(region.module_name)
    dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}.get(ref.dtype)
    if not isinstance(module, torch.nn.Linear) or tuple(module.weight.shape) != ref.shape or module.weight.dtype != dtype:
        raise ValueError("Live affine shape or dtype differs from the source checkpoint")
    if ref.nbytes > limits.max_memory_bytes // 8:
        raise ValueError("Source affine tensor exceeds the bounded workspace")
    width = min(16, ref.shape[1], plan.max_columns, limits.max_update_entries // ref.shape[0])
    if width < 1 or limits.max_rank < 1:
        raise ValueError("Host budget cannot support a rank-one column update")

    def check_memory():
        rss = current_rss_bytes()
        if rss is not None and rss > limits.max_memory_bytes:
            raise ValueError("Endpoint probe process exceeds the host memory budget")

    check_memory()
    raw = checkpoint.read_raw(region.tensor_name, max_bytes=ref.nbytes)
    if _tensor_bytes(module.weight, torch) != raw:
        raise ValueError("Live affine values differ from the bound checkpoint bytes")
    source = _to_float64(raw, ref.dtype).reshape(ref.shape)
    tensor_hash = hashlib.sha256(raw).hexdigest()
    prepared = prepare_continuations(runner.tokenizer, [row["prompt"] for row in rows],
                                     [row["answers"] for row in rows], max_input_tokens=max_input_tokens)
    parent_rows = {row["id"]: row for row in baseline_endpoint["rows"]}
    if len(parent_rows) != len(baseline_endpoint["rows"]):
        raise ValueError("Baseline endpoint rows need unique IDs")
    if any(row["id"] not in parent_rows or parent_rows[row["id"]].get("prompt", row["prompt"]) != row["prompt"]
           for row in rows):
        raise ValueError("Fitting rows must use the same prompts as the baseline endpoint")
    prefixes = []
    for index in preservation_prefix_rows:
        row_id = rows[index]["id"]
        measured = parent_rows.get(row_id, {})
        if measured.get("correct") is not True:
            raise ValueError("Distribution diagnostics require accepted parent preservation responses")
        answer = _completion_ids(runner.tokenizer, measured["generated_token_ids"])
        prefixes.append((row_id, prepared[2 * index].prefix_ids, answer))

    probes = []
    chosen = {"margin": None, "endpoint": None, "distribution": None}
    counts = {"forward_calls": 0, "distribution_forward_calls": 0,
              "backward_calls": 0, "candidate_probes": 0, "endpoint_evaluations": 0}

    def receipt(reason, **extra):
        return {"schema": "weightbridge.endpoint_update.v1", "reason": reason, "method": plan.method,
                "tensor_name": region.tensor_name, "tensor_sha256": tensor_hash, "source_dtype": ref.dtype,
                "fit_row_ids": [row["id"] for row in rows], "probes": probes, "selected": chosen,
                "distribution_penalties": {"kl_weight": kl_weight, "norm_weight": norm_weight},
                "resources": {**counts, "total_numerical_forward_calls": counts["forward_calls"] +
                              counts["distribution_forward_calls"], "wall_seconds": time.perf_counter() - wall,
                              "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}, **extra}

    batches = ceil(len(prepared) / batch_size)
    constrained = plan.method == "constrained"
    if (batches * (1 + len(doses)) + (1 + len(doses)) * bool(prefixes) > limits.max_forward_calls
            or 1 + int(constrained) > limits.max_backward_calls):
        return {}, receipt("insufficient_solver_budget")

    def margins():
        groups = []
        for start in range(0, len(prepared), batch_size):
            check_memory()
            counts["forward_calls"] += 1
            scores = score_batch(runner, prepared[start:start + batch_size])
            groups.append(torch.stack([item.sum() for item in scores]).reshape(-1, 2))
        check_memory()
        return torch.cat(groups)

    def distributions():
        check_memory()
        counts["distribution_forward_calls"] += 1
        values = _distribution_batch(runner, prefixes, torch)
        check_memory()
        return values

    def derivative(loss, *, retain_graph=False):
        counts["backward_calls"] += 1
        gradient, = torch.autograd.grad(loss, module.weight, retain_graph=retain_graph)
        check_memory()
        return gradient.detach().float()

    parameters = list(runner.model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(parameter is module.weight)
        base_scores = margins()
        base_margins = base_scores[:, 0] - base_scores[:, 1]
        preservation = torch.as_tensor(preserve, device=base_scores.device)
        target_gradient = derivative(torch.nn.functional.softplus(-base_margins[~preservation]).mean(),
                                     retain_graph=constrained)
        if not torch.isfinite(target_gradient).all().item():
            return {}, receipt("nonfinite_sensitivity")
        importance = torch.linalg.vector_norm(target_gradient, dim=0).cpu().numpy()
        columns = sorted(np.argsort(-importance, kind="stable")[:width].tolist())
        direction = -target_gradient[:, columns].cpu().numpy().astype(np.float64)
        if constrained:
            protect = derivative(-base_margins[preservation].mean())[:, columns].cpu().numpy().astype(np.float64)
            if not np.isfinite(protect).all():
                return {}, receipt("nonfinite_preservation_sensitivity")
            adverse, squared = float(np.sum(direction * protect)), float(np.sum(protect**2))
            if adverse > 0 and squared > 0:
                direction -= adverse / squared * protect
        before = base_margins.detach().cpu().numpy()
        before_scores = base_scores.detach().cpu().numpy().tolist()
    finally:
        for parameter, required in zip(parameters, original_requires_grad, strict=True):
            parameter.requires_grad_(required)
    del base_margins, base_scores, target_gradient
    if np.linalg.norm(direction) <= 1e-12:
        return {}, receipt("no_local_direction")

    left, _, right = np.linalg.svd(direction, full_matrices=False)
    unit = np.outer(left[:, 0], right[0])
    with torch.inference_mode():
        parent_distributions = distributions() if prefixes else ()
    base_columns = source[:, columns]
    patches, seen = {}, set()
    for index, dose in enumerate(doses):
        counts["candidate_probes"] += 1
        probe_id = f"dose-{index + 1}"
        after = _to_float64(_from_float64(base_columns + dose * unit, ref.dtype), ref.dtype).reshape(base_columns.shape)
        # The patch engine counts numeric changes. Preserve the source sign of zero on numeric no-ops.
        after[after == base_columns] = base_columns[after == base_columns]
        change = after - base_columns
        norm = float(np.linalg.norm(change))
        trial = {"id": probe_id, "dose": dose, "changed_entries": int(np.count_nonzero(change)),
                 "realized_update_norm": norm, "touched_entries": change.size,
                 "max_abs_delta": float(np.max(np.abs(change))), "planned_rank": 1}
        probes.append(trial)
        if trial["changed_entries"] == 0 or not isfinite(norm) or norm > limits.max_update_norm:
            trial["reason"] = "no_stored_change" if trial["changed_entries"] == 0 else "realized_norm_exceeds_budget"
            continue
        expected = source.copy()
        expected[:, columns] = after
        expected_raw = _from_float64(expected, ref.dtype)
        del expected
        digest = hashlib.sha256(expected_raw).hexdigest()
        trial["candidate_tensor_sha256"] = digest
        if digest in seen:
            trial["reason"] = "duplicate_stored_candidate"
            continue
        seen.add(digest)
        with _live_probe(module.weight, columns, after, expected_raw, raw, torch), torch.inference_mode():
            scores = margins().cpu().numpy()
            trial["margin"] = _margin_report(before, scores[:, 0] - scores[:, 1], preserve)
            trial["margin"]["answer_log_probabilities"] = scores.tolist()
            counts["endpoint_evaluations"] += 1
            endpoint = evaluate_candidate()
            guard = endpoint.get("guard", {})
            if type(guard.get("passed")) is not bool:
                raise ValueError("Endpoint evaluator must return a boolean guard decision")
            if guard["passed"] and (type(guard.get("target_item_gain")) is not int or guard["target_item_gain"] < 1
                                    or guard.get("prohibited_regressions") or guard.get("lost_valid_presentations")):
                raise ValueError("Endpoint guard passed without a positive gain and zero regressions")
            trial["endpoint"] = endpoint
            divergences = _first_divergences(baseline_endpoint, endpoint["measurement"], runner.tokenizer)
            trial["first_divergences"] = divergences
            if prefixes:
                report = _distribution_report(prefixes, parent_distributions, distributions(), runner.tokenizer)
                report["objective"] = trial["margin"]["loss"] + kl_weight * report["mean_kl"] + norm_weight * norm**2
                trial["distribution"] = report
                by_position = {(row["id"], point["position"]): point for row in report["prefixes"]
                               for point in row["positions"]}
                for divergence in divergences:
                    point = by_position.get((divergence["id"], divergence["position"]))
                    if point is not None:
                        divergence["distribution_at_shared_prefix"] = point
            check_memory()
        trial["restored_exactly"] = True
        trial["reason"] = "endpoint_pass" if endpoint["guard"]["passed"] else "endpoint_rejected"
        if endpoint["guard"]["passed"]:
            patches[probe_id] = _patch(plan, tensor_hash, columns, after)

    margin = [probe for probe in probes if probe.get("margin", {}).get("passed")]
    endpoint = [probe for probe in probes if probe["id"] in patches]
    if margin:
        chosen["margin"] = max(margin, key=lambda p: (p["margin"]["corrections"], -p["margin"]["loss"]))["id"]
    if endpoint:
        chosen["endpoint"] = max(endpoint, key=lambda p: (p["endpoint"]["guard"]["target_item_gain"],
                                                         -p["realized_update_norm"]))["id"]
        if prefixes:
            chosen["distribution"] = min(endpoint, key=lambda p: p["distribution"]["objective"])["id"]
    return patches, receipt("practice_pass" if endpoint else "no_endpoint_candidate", columns=columns,
                            baseline_margin=before.tolist(), baseline_answer_log_probabilities=before_scores,
                            parent_restored_exactly=_tensor_bytes(module.weight, torch) == raw)
