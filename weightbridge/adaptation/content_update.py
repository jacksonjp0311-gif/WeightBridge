"""Fit small rank-one edits from measured answer scores on authorized practice."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import time

import numpy as np

from ..agents.hf_direction import _DOWN_PROJECTION
from ..core import EditRejected
from ..inference import InferenceError
from ..inspection.safetensors import load_checkpoint
from ..patches.huggingface import _finite_number, _policy, _to_float64
from ..provenance import peak_rss_bytes
from .bic import _objective, _rounded, _runner, _temporary_weights
from .bic_guard import MeasuredProbe, select_measured_probe
from .content_scoring import prepare_continuations, score_batch


def _practice_rows(targets, preservation, *, use_prior=False):
    if (not isinstance(targets, (list, tuple)) or not isinstance(preservation, (list, tuple))
            or not targets or not preservation or len(targets) + len(preservation) > 64):
        raise ValueError("Content updates require target and preservation practice, at most 64 rows")
    rows = tuple(targets) + tuple(preservation)
    fields = {"id", "prompt", "answers", "correct"}
    if use_prior:
        fields.add("neutral_prompt")
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != fields
                or not isinstance(row["id"], str) or not row["id"].strip()
                or not isinstance(row["prompt"], str) or not row["prompt"].strip()
                or not isinstance(row["answers"], (list, tuple)) or len(row["answers"]) != 2
                or not all(isinstance(answer, str) and answer.strip() for answer in row["answers"])
                or row["answers"][0] == row["answers"][1]
                or type(row["correct"]) is not int or row["correct"] not in (0, 1)):
            raise ValueError("Each practice row requires id, prompt, two answers and correct index")
        if use_prior and (not isinstance(row["neutral_prompt"], str) or not row["neutral_prompt"].strip()
                          or row["neutral_prompt"].strip() == row["prompt"].strip()):
            raise ValueError("Calibrated practice requires a distinct neutral prompt for each row")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Practice row IDs must be unique")
    return rows


def _probe_points(dimensions, radius):
    axes = list(np.eye(dimensions))
    if dimensions == 2:
        axes.append(np.ones(dimensions) / math.sqrt(dimensions))
    for scale in (radius, radius / 2):
        for axis in axes:
            point = scale * axis
            yield tuple(float(value) for value in point) + (0.0,) * (2 - dimensions)


def _prepare_batches(tokenizer, rows, scoring, max_input_tokens, batch_size, use_prior):
    answers = [row["answers"] for row in rows]
    task = prepare_continuations(tokenizer, [row["prompt"] for row in rows], answers,
                                 max_input_tokens=max_input_tokens, include_eos=scoring == "terminated_sum")
    if use_prior:
        neutral = prepare_continuations(tokenizer, [row["neutral_prompt"] for row in rows], answers,
                                        max_input_tokens=max_input_tokens)
        if any(left.answer_ids != right.answer_ids or left.prefix_ids == right.prefix_ids
               for left, right in zip(task, neutral, strict=True)):
            raise InferenceError("Calibrated continuations require matching answer token IDs and distinct prefixes")
        # Each batch keeps both options under both contexts together.
        task = tuple(continuation for index in range(0, len(task), 2)
                     for continuation in (*task[index:index + 2], *neutral[index:index + 2]))
    return [task[start:start + batch_size] for start in range(0, len(task), batch_size)]


def _measure(runner, torch, batches, scoring, resources, signs=(), modules=(), *, prior_alpha=None):
    """Return first-minus-second scores and exact target-loss weight derivatives."""
    gradients = [np.zeros(tuple(module.weight.shape), dtype=np.float64) for module in modules]
    activations, outputs, margins, row_offset = {}, {}, [], 0
    raw_task_margins, raw_neutral_margins = [], []
    continuations_per_row = 2 if prior_alpha is None else 4
    handles = []

    def capture_input(_module, inputs, index):
        activations[index] = inputs[0].detach()

    def capture_output(_module, _inputs, output, index):
        if index in outputs:
            raise InferenceError("A selected projection was called more than once")
        output = output if output.requires_grad else output.detach().requires_grad_(True)
        outputs[index] = output
        return output

    try:
        for batch in batches:
            batch_rows = len(batch) // continuations_per_row
            target_signs = signs[row_offset:row_offset + batch_rows]
            row_offset += batch_rows
            capture = bool(modules) and len(target_signs) > 0
            if capture and not handles:
                for index, module in enumerate(modules):
                    handles.append(module.register_forward_pre_hook(
                        lambda layer, args, index=index: capture_input(layer, args, index)
                    ))
                    handles.append(module.register_forward_hook(
                        lambda layer, args, value, index=index: capture_output(layer, args, value, index)
                    ))
            elif not capture and handles:
                for handle in handles:
                    handle.remove()
                handles.clear()
            activations.clear()
            outputs.clear()
            with torch.set_grad_enabled(capture):
                started = time.perf_counter()
                token_scores = score_batch(runner, batch)
                totals = torch.stack([value.mean() if scoring == "mean" else value.sum() for value in token_scores])
                if prior_alpha is None:
                    difference = totals[::2] - totals[1::2]
                else:
                    # Match the standalone calibrated evaluator's double precision arithmetic.
                    task_first, task_second, neutral_first, neutral_second = totals.double().reshape(-1, 4).T
                    first = task_first - prior_alpha * neutral_first
                    second = task_second - prior_alpha * neutral_second
                    difference = first - second
                    raw_task_margins.extend((task_first - task_second).detach().cpu().tolist())
                    raw_neutral_margins.extend((neutral_first - neutral_second).detach().cpu().tolist())
                resources["forward_wall_seconds"] += time.perf_counter() - started
                resources["forward_batches"] += 1
                resources["forward_sequences"] += len(batch)
                resources["forward_input_tokens"] += sum(len(continuation.input_ids) for continuation in batch)
                resources["padded_input_token_positions"] += len(batch) * max(len(row.input_ids) for row in batch)
                if not torch.isfinite(difference).all().item():
                    raise InferenceError("Nonfinite continuation score margin")
                margins.extend(difference.detach().double().cpu().tolist())
                if capture:
                    if len(activations) != len(modules) or len(outputs) != len(modules):
                        raise InferenceError("Failed to observe every selected projection")
                    started = time.perf_counter()
                    sign = difference.new_tensor(target_signs)
                    loss = torch.nn.functional.softplus(-sign * difference[:len(target_signs)]).sum() / len(signs)
                    tangents = torch.autograd.grad(loss, [outputs[index] for index in range(len(modules))])
                    for index, tangent in enumerate(tangents):
                        output_gradient = tangent.detach().flatten(0, 1).float()
                        projection_input = activations[index].flatten(0, 1).float()
                        gradients[index] += (output_gradient.T @ projection_input).double().cpu().numpy()
                    resources["backward_wall_seconds"] += time.perf_counter() - started
                    resources["backward_batches"] += 1
                    resources["backward_target_rows"] += len(target_signs)
    finally:
        for handle in handles:
            handle.remove()
    if any(not np.isfinite(gradient).all() for gradient in gradients):
        raise InferenceError("Nonfinite continuation weight sensitivity")
    raw_margins = {} if prior_alpha is None else {"raw_task_margins": raw_task_margins,
                                                  "raw_neutral_margins": raw_neutral_margins}
    return np.asarray(margins, dtype=np.float64), gradients, raw_margins


def _basis(gradient, width):
    scores = np.linalg.norm(gradient, axis=0)
    if not np.any(scores):
        return None
    columns = np.argsort(-scores, kind="stable")[:width]
    left, _, right = np.linalg.svd(gradient[:, columns], full_matrices=False)
    return columns, -np.outer(left[:, 0], right[0])


def _measured(point, objective, margins, labels, target_count):
    first_choice = margins >= 0
    correct = first_choice[:target_count] == (labels[:target_count] == 0)
    return MeasuredProbe(point, objective, tuple(correct.tolist()), tuple(first_choice[target_count:].tolist()))


def propose_content_update(
    base_path,
    targets: Sequence[Mapping],
    preservation: Sequence[Mapping],
    tensor_names,
    inference,
    *,
    policy,
    scoring: str,
    prior_alpha: float | None = None,
    guard: bool = True,
    radius: float = 0.25,
    retention_weight: float = 1.0,
    update_penalty: float = 0.01,
    max_preservation_margin_drop: float = 0.25,
    max_forward_batches: int = 96,
    max_backward_batches: int = 16,
):
    """Return a host-policy patch from measured rounded probes, or abstain.

    ``targets`` and ``preservation`` are authorized practice rows. The answer
    continuations must match the chosen prompt and behavioral representation.
    ``prior_subtracted_sum`` additionally requires a matched ``neutral_prompt``
    per row, a frozen alpha, and the measured guard. Both contexts are measured
    on every candidate with the same model weights.
    Letter and whole-answer arms share the same rank, probe and budget rules.
    Applying and verifying the persisted patch remain separate host actions.
    """
    started, cpu_started = time.perf_counter(), time.process_time()
    use_prior = scoring == "prior_subtracted_sum"
    rows = _practice_rows(targets, preservation, use_prior=use_prior)
    if (scoring not in ("sum", "mean", "terminated_sum", "prior_subtracted_sum") or type(guard) is not bool
            or not _finite_number(radius) or not 1e-6 <= radius <= 8
            or not _finite_number(retention_weight) or not 0 <= retention_weight <= 1000
            or not _finite_number(update_penalty) or not 0 <= update_penalty <= 1000
            or not _finite_number(max_preservation_margin_drop) or not 0 <= max_preservation_margin_drop <= 1000):
        raise ValueError("Invalid content scoring, guard, radius or objective penalty")
    if use_prior:
        if not guard or not _finite_number(prior_alpha) or prior_alpha not in (0.5, 1.0, 1.5):
            raise ValueError("Prior subtraction requires the measured guard and alpha in {0.5, 1.0, 1.5}")
        prior_alpha = float(prior_alpha)
    elif prior_alpha is not None:
        raise ValueError("prior_alpha is only allowed for prior_subtracted_sum scoring")
    if (not isinstance(tensor_names, (list, tuple)) or not 1 <= len(tensor_names) <= 2
            or not all(isinstance(name, str) and _DOWN_PROJECTION.fullmatch(name) for name in tensor_names)
            or len(set(tensor_names)) != len(tensor_names)):
        raise EditRejected("Content updates require one or two distinct Qwen MLP down projections")
    allowed, max_entries, max_fraction, max_delta, max_bytes, total_bytes, _, _ = _policy(policy)
    if not set(tensor_names) <= allowed:
        raise EditRejected("Selected content update tensor is outside host policy")
    batch_size = inference["batch_size"]
    if type(batch_size) is not int or batch_size not in (2, 4, 6, 8):
        raise ValueError("Content update batch size must hold 1 to 4 complete continuation pairs")
    if use_prior and batch_size not in (4, 8):
        raise ValueError("Calibrated batch size must hold complete four-continuation rows")
    rows_per_batch = batch_size // (4 if use_prior else 2)
    probes = tuple(_probe_points(len(tensor_names), radius))
    forward_limit = math.ceil(len(rows) / rows_per_batch) * (1 + len(probes))
    backward_limit = math.ceil(len(targets) / rows_per_batch)
    if (type(max_forward_batches) is not int or not forward_limit <= max_forward_batches <= 512
            or type(max_backward_batches) is not int or not backward_limit <= max_backward_batches <= 64):
        raise ValueError("Content update probe schedule exceeds the batch budget")
    if Path(base_path).is_symlink():
        raise EditRejected("Base checkpoint symlinks are not accepted")
    checkpoint = load_checkpoint(base_path)
    refs = [checkpoint.tensors.get(name) for name in tensor_names]
    if (not checkpoint.root.is_dir() or checkpoint.config.get("model_type") != "qwen3_5"
            or any(ref is None or ref.dtype != "BF16" or len(ref.shape) != 2 or not all(ref.shape) for ref in refs)):
        raise EditRejected("Content updates require existing BF16 Qwen3.5 projection matrices")
    if any(ref.nbytes > max_bytes for ref in refs) or sum(ref.nbytes for ref in refs) > total_bytes:
        raise EditRejected("Content update tensor reads exceed host policy")
    total_parameters = sum(ref.elements for ref in checkpoint.tensors.values())
    entry_limit = min(max_entries, 65536, math.floor(max_fraction * total_parameters))
    widths = [min(64 // len(refs), ref.shape[1], entry_limit // (len(refs) * ref.shape[0])) for ref in refs]
    if not all(widths):
        raise EditRejected("Content update budget cannot touch one column per projection")

    base_hash = checkpoint.checkpoint_sha256()
    resources = {"forward_batches": 0, "backward_batches": 0, "forward_sequences": 0,
                 "forward_input_tokens": 0, "padded_input_token_positions": 0, "backward_target_rows": 0,
                 "forward_wall_seconds": 0.0, "backward_wall_seconds": 0.0}
    labels = np.asarray([row["correct"] for row in rows])
    signs = np.where(labels[:len(targets)] == 0, 1.0, -1.0)
    diagnostics = {"schema": "weightbridge.content_update_report.v1", "base_sha256": base_hash,
                   "scoring": scoring, "guard": guard, "radius": radius, "probe_schedule": probes,
                   "forward_budget": max_forward_batches, "backward_budget": max_backward_batches,
                   "preservation_margin_drop_limit": max_preservation_margin_drop if guard else None,
                   "rows": [row["id"] for row in rows], "probes": [], "basis": []}
    if use_prior:
        diagnostics.update(prior_alpha=prior_alpha, continuations_per_row=4)
    patch = None
    with _runner(checkpoint.root, inference) as (runner, torch):
        tokenize_started = time.perf_counter()
        batches = _prepare_batches(runner.tokenizer, rows, scoring, inference["max_input_tokens"],
                                   batch_size, use_prior)
        resources["tokenization_wall_seconds"] = time.perf_counter() - tokenize_started
        modules = [runner.model.get_submodule(name.removesuffix(".weight")) for name in tensor_names]
        if any(module.weight.dtype != torch.bfloat16 or tuple(module.weight.shape) != ref.shape
               for module, ref in zip(modules, refs, strict=True)):
            raise EditRejected("Loaded projection dtype or shape differs from the BF16 checkpoint")
        baseline, gradients, baseline_raw = _measure(runner, torch, batches, scoring, resources, signs, modules,
                                                      prior_alpha=prior_alpha)
        if use_prior:
            diagnostics["baseline_raw_margins"] = baseline_raw
        origin, _, _ = _objective(baseline, baseline, signs, retention_weight)
        diagnostics.update(origin_objective=origin, baseline_margins=baseline.tolist(), disposition="NO_SENSITIVITY")
        layers = []
        for axis, (name, ref, width, module, gradient) in enumerate(zip(
                tensor_names, refs, widths, modules, gradients, strict=True)):
            basis = _basis(gradient, width)
            if basis is None:
                continue
            columns, direction = basis
            indices = torch.as_tensor(columns, dtype=torch.long, device=module.weight.device)
            stored = _to_float64(checkpoint.read_raw(name, max_bytes=max_bytes), ref.dtype).reshape(ref.shape)[:, columns]
            original = module.weight.detach().index_select(1, indices)
            if not np.isfinite(stored).all() or not np.array_equal(original.double().cpu().numpy(), stored):
                raise EditRejected("Loaded projection values differ from the BF16 checkpoint")
            layers.append({"module": module, "index": indices, "original": original, "base": stored,
                           "dtype": ref.dtype, "direction": direction, "gradient": gradient[:, columns],
                           "axis": axis, "name": name, "columns": columns.tolist()})
            diagnostics["basis"].append({"tensor": name, "columns": columns.tolist(), "rank": 1,
                                         "target_loss_gradient_norm": float(np.linalg.norm(gradient))})
        if layers:
            parent = _measured((0.0, 0.0), origin, baseline, labels, len(targets))
            preserved_margins = baseline[len(targets):]
            preservation_signs = np.where(preserved_margins >= 0, 1.0, -1.0)
            measured_probes, details = [], {}
            seen = set()
            for point in probes:
                selected = [layer for layer in layers if point[layer["axis"]] != 0.0]
                deltas = [point[layer["axis"]] * layer["direction"] for layer in selected]
                rounded = [_rounded(layer["base"], delta, layer["dtype"])
                           for layer, delta in zip(selected, deltas, strict=True)]
                realized = [new - layer["base"] for new, layer in zip(rounded, selected, strict=True)]
                changed = sum(int(np.count_nonzero(delta)) for delta in realized)
                norm = math.sqrt(sum(float(np.sum(delta**2)) for delta in realized))
                requested_norm = math.sqrt(sum(float(np.sum(delta**2)) for delta in deltas))
                probe = {"coefficients": point, "changed_entries": changed, "realized_frobenius_norm": norm,
                         "requested_frobenius_norm": requested_norm}
                diagnostics["probes"].append(probe)
                if not changed:
                    probe["disposition"] = "PRECISION_LOSS"
                    continue
                if norm > radius + 1e-12 or any(np.max(np.abs(delta)) > max_delta for delta in realized):
                    probe["disposition"] = "POLICY_REFUSAL"
                    continue
                rounded_key = tuple((layer["axis"], new.tobytes()) for layer, new, delta in
                                    zip(selected, rounded, realized, strict=True) if np.any(delta))
                if rounded_key in seen:
                    probe["disposition"] = "DUPLICATE_ROUNDED_PROBE"
                    continue
                seen.add(rounded_key)
                with _temporary_weights(torch, selected, rounded):
                    observed, _, observed_raw = _measure(runner, torch, batches, scoring, resources,
                                                          prior_alpha=prior_alpha)
                probe.update(observed_raw)
                loss, target_loss, preservation_kl = _objective(observed, baseline, signs, retention_weight)
                penalty = update_penalty * norm**2
                actual = loss + penalty
                predicted = origin + penalty + sum(float(np.sum(layer["gradient"] * delta))
                                                    for layer, delta in zip(selected, realized, strict=True))
                error = actual - predicted
                reliable = predicted >= 0.0 and abs(error) <= max(0.02, abs(origin - predicted))
                probe.update(predicted_objective=predicted, observed_objective=actual, surrogate_error=error,
                             target_loss=target_loss, preservation_kl=preservation_kl, observed_margins=observed.tolist(),
                             disposition="MEASURED" if reliable else "UNRELIABLE_SURROGATE")
                measurement = _measured(point, actual, observed, labels, len(targets))
                margin_drop = float(np.max(preservation_signs * (preserved_margins - observed[len(targets):])))
                probe.update(target_gain=sum(measurement.target_correct) - sum(parent.target_correct),
                             target_regressions=sum(old and not new for old, new in
                                                    zip(parent.target_correct, measurement.target_correct, strict=True)),
                             preservation_flips=sum(old != new for old, new in
                                                    zip(parent.preservation_first_choice,
                                                        measurement.preservation_first_choice, strict=True)),
                             max_preservation_margin_drop=max(0.0, margin_drop))
                if reliable and guard and margin_drop > max_preservation_margin_drop:
                    probe["disposition"] = "PRESERVATION_MARGIN_DROP"
                elif reliable:
                    measured_probes.append(measurement)
                    details[point] = (probe, selected, deltas)

            if guard:
                winner = select_measured_probe(parent, measured_probes, require_target_gain=True)
            else:
                winner = min((probe for probe in measured_probes if probe.objective < origin),
                             key=lambda probe: probe.objective, default=None)
            diagnostics["disposition"] = "NO_ELIGIBLE_MEASURED_PROBE"
            if winner is not None:
                probe, selected, deltas = details[winner.coefficients]
                diagnostics.update(disposition="PROPOSED", selected_probe=probe,
                                   requested_frobenius_norm=probe["requested_frobenius_norm"],
                                   realized_frobenius_norm=probe["realized_frobenius_norm"],
                                   changed_entries=probe["changed_entries"])
                patch = {"schema": "weightbridge.patch.v1", "base_sha256": base_hash,
                         "tensor_sha256": {layer["name"]: checkpoint.tensor_sha256(layer["name"]) for layer in selected},
                         "reason": "Rank-one content-sensitive edit selected from rounded practice probes",
                         "evidence_ids": ["practice-content-update"],
                         "slices": [{"tensor": layer["name"], "axis": "columns", "indices": layer["columns"],
                                     "op": "add", "values": delta.T.tolist()} for layer, delta in
                                    zip(selected, deltas, strict=True)]}

    if load_checkpoint(checkpoint.root).checkpoint_sha256() != base_hash:
        raise InferenceError("Content update source checkpoint changed during synthesis")
    resources.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started,
                     peak_rss_bytes=peak_rss_bytes())
    diagnostics["resources"] = resources
    return patch, diagnostics
