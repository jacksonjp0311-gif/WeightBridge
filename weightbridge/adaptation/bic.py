"""Compile coordinated Qwen MLP edits from authorized binary practice prompts."""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

from ..agents.hf_direction import _DOWN_PROJECTION
from ..evaluation.hf_edit import _load_suite
from ..inference import HFTextRunner, InferenceConfig, InferenceError
from ..inspection.safetensors import load_checkpoint
from ..patches.huggingface import _finite_number, _from_float64, _to_float64
from ..provenance import peak_rss_bytes
from .bic_math import fit_local_quadratic, select_coefficients


@contextmanager
def _runner(path, inference):
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("Install WeightBridge inference extras to use BIC") from exc
    InferenceConfig(cpu_threads=inference["cpu_threads"], max_input_tokens=inference["max_input_tokens"]).validate(
        max_input_budget=512
    )
    if type(inference["batch_size"]) is not int or not 1 <= inference["batch_size"] <= 8:
        raise ValueError("BIC batch size must be in [1, 8]")
    previous_threads = torch.get_num_threads()
    if inference["cpu_threads"]:
        torch.set_num_threads(inference["cpu_threads"])
    try:
        runner = HFTextRunner.from_pretrained(str(path), device=inference["device"], dtype=inference["dtype"])
        runner.model.eval().requires_grad_(False)
        yield runner, torch
    finally:
        if inference["cpu_threads"]:
            torch.set_num_threads(previous_threads)


class _BinaryProbe:
    def __init__(self, runner, torch, rows, choices, inference):
        if not 1 <= len(rows) <= 128 or len(choices) != 2 or len(set(choices)) != 2:
            raise ValueError("BIC requires 1 to 128 rows and two distinct choices")
        if any(row["label"] not in choices for row in rows):
            raise ValueError("Every BIC label must be an authorized choice")
        self.runner, self.torch, self.choices = runner, torch, choices
        tokenizer = runner.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise InferenceError("Tokenizer needs a padding or end-of-sequence token")
            tokenizer.pad_token = tokenizer.eos_token
        ids = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
        if any(len(item) != 1 for item in ids) or ids[0] == ids[1]:
            raise InferenceError("Each choice must encode to a distinct single token")
        self.ids = [item[0] for item in ids]
        self.batches = []
        batch_size, token_limit = inference["batch_size"], inference["max_input_tokens"]
        for start in range(0, len(rows), batch_size):
            prompts = [row["text"] for row in rows[start:start + batch_size]]
            template = bool(getattr(tokenizer, "chat_template", None))
            if template:
                prompts = tokenizer.apply_chat_template(
                    [[{"role": "user", "content": prompt}] for prompt in prompts],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            tokens = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=not template,
                               truncation=True, max_length=token_limit + 1)
            if not 1 <= tokens["input_ids"].shape[1] <= token_limit:
                raise InferenceError("BIC prompt exceeds max_input_tokens or encodes to no tokens")
            self.batches.append({key: value.to(runner.device) for key, value in tokens.items()})
        forward = inspect.signature(runner.model.forward).parameters
        self.forward_kwargs = {"use_cache": False}
        if "logits_to_keep" in forward:
            self.forward_kwargs["logits_to_keep"] = 1
        elif "num_logits_to_keep" in forward:
            self.forward_kwargs["num_logits_to_keep"] = 1
        self.forward_batches = self.backward_batches = self.forward_examples = self.backward_examples = 0
        self.forward_token_positions = 0
        self.forward_seconds = self.backward_seconds = 0.0

    def embedding_direction(self):
        weights = self.runner.model.get_output_embeddings().weight[self.ids]
        first, second = weights.detach().float().cpu().numpy().astype(np.float64)
        direction = first - second
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or not length:
            raise InferenceError("Choice embeddings do not define an output direction")
        return direction / length

    def measure(self, modules=()):
        """Return margins and optional last-token activation and margin gradients."""
        torch, model = self.torch, self.runner.model
        inputs_seen, outputs_seen = {}, {}
        activations = [[] for _ in modules]
        gradients = [[] for _ in modules]
        margins = []
        handles = []

        def capture_input(_module, args, index):
            inputs_seen[index] = args[0][:, -1].detach().float().cpu().numpy().copy()

        def capture_output(_module, _args, value, index):
            if not value.requires_grad:
                value = value.detach().requires_grad_(True)
            outputs_seen[index] = value
            return value

        try:
            for index, module in enumerate(modules):
                handles.append(module.register_forward_pre_hook(
                    lambda layer, args, index=index: capture_input(layer, args, index)
                ))
                handles.append(module.register_forward_hook(
                    lambda layer, args, value, index=index: capture_output(layer, args, value, index)
                ))
            with torch.set_grad_enabled(bool(modules)):
                for tokens in self.batches:
                    inputs_seen.clear()
                    outputs_seen.clear()
                    started = time.perf_counter()
                    logits = model(**tokens, **self.forward_kwargs).logits[:, -1, self.ids].float()
                    margin = logits[:, 0] - logits[:, 1]
                    self.forward_seconds += time.perf_counter() - started
                    self.forward_batches += 1
                    self.forward_examples += len(margin)
                    self.forward_token_positions += tokens["input_ids"].numel()
                    if not torch.isfinite(margin).all().item():
                        raise InferenceError("Nonfinite BIC choice margin")
                    margins.append(margin.detach().cpu().numpy().astype(np.float64))
                    if modules:
                        if len(inputs_seen) != len(modules) or len(outputs_seen) != len(modules):
                            raise InferenceError("BIC failed to observe a requested projection")
                        started = time.perf_counter()
                        tangents = torch.autograd.grad(margin.sum(), [outputs_seen[i] for i in range(len(modules))])
                        self.backward_seconds += time.perf_counter() - started
                        self.backward_batches += 1
                        self.backward_examples += len(margin)
                        for index, tangent in enumerate(tangents):
                            activations[index].append(inputs_seen[index])
                            gradients[index].append(tangent[:, -1].detach().float().cpu().numpy().copy())
        finally:
            for handle in handles:
                handle.remove()
        return np.concatenate(margins), [np.concatenate(a) for a in activations], [np.concatenate(g) for g in gradients]

    def resources(self, started, cpu_started):
        return {
            "forward_batches": self.forward_batches, "backward_batches": self.backward_batches,
            "forward_examples": self.forward_examples, "backward_examples": self.backward_examples,
            "forward_token_positions": self.forward_token_positions,
            "forward_wall_seconds": self.forward_seconds, "backward_wall_seconds": self.backward_seconds,
            "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
            "peak_rss_bytes": peak_rss_bytes(),
        }


def _cosines(gradients, direction):
    lengths = np.linalg.norm(gradients, axis=1)
    products = gradients @ direction
    return [float(np.clip(dot / length, -1, 1)) if length else None for dot, length in zip(products, lengths)]


def _margin_rows(rows, choices, margins):
    return [{"id": row["id"], "label": row["label"], "margin": float(margin),
             "signed_margin": float(margin if row["label"] == choices[0] else -margin),
             "prediction": choices[int(margin < 0)]} for row, margin in zip(rows, margins, strict=True)]


def score_binary_margins(model_path, rows, inference, *, choices=("A", "B"), jacobian_tensor=None):
    """Measure raw choice margins and optionally the final-token functional Jacobian."""
    started, cpu_started = time.perf_counter(), time.process_time()
    if jacobian_tensor is not None and (not isinstance(jacobian_tensor, str)
                                       or not _DOWN_PROJECTION.fullmatch(jacobian_tensor)):
        raise ValueError("The Jacobian diagnostic requires a Qwen MLP down projection")
    with _runner(model_path, inference) as (runner, torch):
        probe = _BinaryProbe(runner, torch, rows, choices, inference)
        modules = () if jacobian_tensor is None else (runner.model.get_submodule(jacobian_tensor.removesuffix(".weight")),)
        margins, _, gradients = probe.measure(modules)
        scored = _margin_rows(rows, choices, margins)
        if gradients:
            direction = probe.embedding_direction()
            for row, cosine, norm in zip(scored, _cosines(gradients[0], direction),
                                         np.linalg.norm(gradients[0], axis=1), strict=True):
                row["embedding_jacobian_cosine"] = cosine
                row["jacobian_norm"] = float(norm)
        return {"choices": list(choices), "rows": scored, "resources": probe.resources(started, cpu_started)}


def _basis(activations, gradients, signed_margin, signs, width, embedding_direction):
    """Build a unit-Frobenius rank-one direction from last-token sensitivities."""
    activations = activations.astype(np.float64)
    gradients = gradients.astype(np.float64)
    if not np.isfinite(activations).all() or not np.isfinite(gradients).all():
        raise InferenceError("Nonfinite BIC activations or functional gradients")
    count = len(signs)
    scale = np.maximum(np.sqrt(np.mean(activations**2, axis=0)), 1e-6)
    errors = signs * np.exp(-np.logaddexp(0, signed_margin))
    target = activations[:count] / scale
    tangent = (gradients[:count] * errors[:, None]).T @ target
    scores = np.linalg.norm(tangent, axis=0)
    if np.any(scores):
        columns = np.argsort(-scores, kind="stable")[:width]
        u, _, vh = np.linalg.svd(tangent[:, columns] * scale[columns], full_matrices=False)
        return columns, np.outer(u[:, 0], vh[0]), "functional_jacobian"

    # A zero first-order signal cannot rule out a coordinated effect. Use a
    # deterministic activation direction as the measured two-dimensional basis.
    contrast = errors @ target
    columns = np.argsort(-np.abs(contrast), kind="stable")[:width]
    right = contrast[columns] / scale[columns]
    norm = np.linalg.norm(right)
    if not norm:
        raise InferenceError("No BIC practice activation direction was identified")
    return columns, np.outer(embedding_direction, right / norm), "embedding_fallback"


def _objective(margins, baseline, signs, retention_weight):
    count = len(signs)
    target_loss = float(np.mean(np.logaddexp(0, -signs * margins[:count])))
    old, new = baseline[count:], margins[count:]
    parent_probability = np.exp(-np.logaddexp(0, -old))
    kl = np.logaddexp(0, new) - np.logaddexp(0, old) - parent_probability * (new - old)
    preservation_kl = float(np.mean(np.maximum(kl, 0)))
    return target_loss + retention_weight * preservation_kl, target_loss, preservation_kl


def _rounded(base, delta, dtype):
    return _to_float64(_from_float64(base + delta, dtype), dtype).reshape(base.shape)


@contextmanager
def _temporary_weights(torch, layers, rounded):
    try:
        with torch.no_grad():
            for layer, values in zip(layers, rounded, strict=True):
                weight = layer["module"].weight
                weight.index_copy_(1, layer["index"], torch.as_tensor(values, device=weight.device, dtype=weight.dtype))
        yield
    finally:
        with torch.no_grad():
            for layer in layers:
                layer["module"].weight.index_copy_(1, layer["index"], layer["original"])


def _margin_prediction(baseline, samples, local, coefficients, interaction):
    x_plus, x_minus, y_plus, y_minus, *corners = samples
    step = local.probe_step
    x, y = coefficients
    predicted = baseline + (x * (x_plus - x_minus) + y * (y_plus - y_minus)) / (2 * step)
    if interaction:
        pp, pm, mp, mm = corners
        predicted += x * y * ((pp - pm) - (mp - mm)) / (4 * step**2)
    return predicted


def propose_bic(
    base_path,
    practice_suite,
    tensor_names,
    inference,
    *,
    interaction: bool,
    radius: float,
    retention_weight: float = 1.0,
    update_penalty: float = 0.01,
    max_entries: int = 65536,
):
    """Return a host-policy candidate patch and practice diagnostics.

    The same two-layer basis and requested norm cap apply to both modes.
    Interaction mode makes four additional mixed probes. Every weight probe
    uses the checkpoint storage rounding and is reverted in memory.
    """
    if type(interaction) is not bool:
        raise ValueError("BIC interaction mode must be an explicit boolean")
    if (not _finite_number(radius) or not 1e-6 <= radius <= 8
            or not _finite_number(retention_weight) or not 0 <= retention_weight <= 1000
            or not _finite_number(update_penalty) or not 0 <= update_penalty <= 1000):
        raise ValueError("BIC radius must be in [1e-6, 8] and penalties in [0, 1000]")
    if type(max_entries) is not int or not 2 <= max_entries <= 65536:
        raise ValueError("BIC supports at most 65,536 jointly touched coordinates")
    if (not isinstance(tensor_names, (tuple, list)) or len(tensor_names) != 2
            or not all(isinstance(name, str) and _DOWN_PROJECTION.fullmatch(name) for name in tensor_names)
            or tensor_names[0] == tensor_names[1]):
        raise ValueError("BIC requires two distinct Qwen MLP down projections")
    if not isinstance(practice_suite, dict) or set(practice_suite) != {"schema", "choices", "target", "sentinel"}:
        raise ValueError("BIC requires a practice-only suite with target and preservation rows")
    suite = _load_suite(json.dumps(practice_suite).encode())
    choices = suite["choices"]
    rows = suite["target"] + suite["sentinel"]
    if len(choices) != 2 or len(rows) > 64:
        raise ValueError("BIC synthesis requires two choices and at most 64 practice rows")
    signs = np.array([1.0 if row["label"] == choices[0] else -1.0 for row in suite["target"]])
    started, cpu_started = time.perf_counter(), time.process_time()
    checkpoint = load_checkpoint(base_path)
    refs = [checkpoint.tensors.get(name) for name in tensor_names]
    if (not checkpoint.root.is_dir() or checkpoint.config.get("model_type") != "qwen3_5"
            or any(ref is None or len(ref.shape) != 2 or not all(ref.shape) for ref in refs)):
        raise ValueError("The authorized projections must exist in a local Qwen3.5 checkpoint")
    widths = [min(32, ref.shape[1], max_entries // (2 * ref.shape[0])) for ref in refs]
    if not all(widths):
        raise ValueError("BIC entry budget cannot cover one column per layer")
    base_hash = checkpoint.checkpoint_sha256()
    with _runner(checkpoint.root, inference) as (runner, torch):
        probe = _BinaryProbe(runner, torch, rows, choices, inference)
        modules = [runner.model.get_submodule(name.removesuffix(".weight")) for name in tensor_names]
        baseline, activations, gradients = probe.measure(modules)
        embedding = probe.embedding_direction()
        layers, basis_summary = [], []
        for name, ref, width, module, activation, gradient in zip(
                tensor_names, refs, widths, modules, activations, gradients, strict=True):
            columns, direction, basis_kind = _basis(activation, gradient, signs * baseline[:len(signs)],
                                                    signs, width, embedding)
            original = _to_float64(checkpoint.read_raw(name), ref.dtype).reshape(ref.shape)[:, columns]
            index = torch.as_tensor(columns, dtype=torch.long, device=runner.device)
            layers.append({"module": module, "index": index, "base": original, "direction": direction,
                           "dtype": ref.dtype, "original": module.weight.detach().index_select(1, index)})
            basis_summary.append({"tensor": name, "columns": columns.tolist(), "rank": 1, "kind": basis_kind,
                                  "embedding_jacobian_cosines": _cosines(gradient, embedding)})

        origin, _, _ = _objective(baseline, baseline, signs, retention_weight)
        probe_margins = []

        def round_values(coefficients):
            deltas = [value * layer["direction"] for value, layer in zip(coefficients, layers, strict=True)]
            rounded = [_rounded(layer["base"], delta, layer["dtype"])
                       for layer, delta in zip(layers, deltas, strict=True)]
            return deltas, rounded

        def evaluate(coefficients):
            _, rounded = round_values(coefficients)
            with _temporary_weights(torch, layers, rounded):
                margins, _, _ = probe.measure()
            probe_margins.append(margins)
            return _objective(margins, baseline, signs, retention_weight)[0]

        local = fit_local_quadratic(evaluate, origin, radius=radius, interaction=interaction)
        coefficients = select_coefficients(local, radius=radius, update_penalty=update_penalty)
        requested_norm = float(np.linalg.norm(coefficients))
        prediction = local.predict(coefficients) + update_penalty * requested_norm**2
        deltas, rounded = round_values(coefficients)
        realized = [values - layer["base"] for values, layer in zip(rounded, layers, strict=True)]
        realized_norm = math.sqrt(sum(float(np.sum(delta**2)) for delta in realized))
        rounding_error = math.sqrt(sum(float(np.sum((a - b)**2)) for a, b in zip(realized, deltas, strict=True)))
        changed_entries = sum(int(np.count_nonzero(delta)) for delta in realized)
        selected = [index for index, value in enumerate(coefficients) if value != 0]
        diagnostics = {
            "mode": "mixed_curvature" if interaction else "first_order",
            "base_sha256": base_hash, "basis": basis_summary, "coefficients": coefficients.tolist(),
            "requested_frobenius_norm": requested_norm, "realized_frobenius_norm": realized_norm,
            "rounding_error_frobenius_norm": rounding_error, "changed_entries": changed_entries,
            "touched_entries": sum(deltas[index].size for index in selected),
            "origin_objective": origin, "predicted_objective": prediction,
            "local_model": {"probe_step": local.probe_step, "gradient": local.gradient.tolist(),
                            "axial_curvature": local.axial_curvature.tolist(), "hessian": local.hessian.tolist(),
                            "probes": [{"coefficients": list(point), "objective": loss}
                                       for point, loss in local.probes]},
            "rows": _margin_rows(rows, choices, baseline),
        }
        patch = None
        if not selected:
            diagnostics["disposition"] = "NO_PREDICTED_GAIN"
        elif not changed_entries:
            diagnostics["disposition"] = "PRECISION_LOSS"
        else:
            with _temporary_weights(torch, layers, rounded):
                measured, _, _ = probe.measure()
            loss, target_loss, preservation_kl = _objective(measured, baseline, signs, retention_weight)
            observed = loss + update_penalty * realized_norm**2
            predicted_margins = _margin_prediction(baseline, probe_margins, local, coefficients, interaction)
            for row, predicted, actual in zip(diagnostics["rows"], predicted_margins, measured, strict=True):
                row["predicted_margin"] = float(predicted)
                row["observed_margin"] = float(actual)
                row["observed_prediction"] = choices[int(actual < 0)]
            diagnostics.update({"observed_objective": observed, "observed_target_loss": target_loss,
                                "observed_preservation_kl": preservation_kl,
                                "objective_prediction_error": observed - prediction})
            if observed >= origin:
                diagnostics["disposition"] = "NO_OBSERVED_GAIN"
            else:
                diagnostics["disposition"] = "PROPOSED"
                patch = {
                    "schema": "weightbridge.patch.v1", "base_sha256": base_hash,
                    "tensor_sha256": {tensor_names[i]: checkpoint.tensor_sha256(tensor_names[i]) for i in selected},
                    "reason": "Coordinated rank-one Qwen edit verified on rounded practice probes",
                    "evidence_ids": ["practice-bic-" + diagnostics["mode"]],
                    "slices": [{"tensor": tensor_names[i], "axis": "columns", "op": "add",
                                "indices": basis_summary[i]["columns"], "values": deltas[i].T.tolist()}
                               for i in selected],
                }
        diagnostics["resources"] = probe.resources(started, cpu_started)
    if load_checkpoint(checkpoint.root).checkpoint_sha256() != base_hash:
        raise InferenceError("BIC source checkpoint changed during synthesis")
    return patch, diagnostics
