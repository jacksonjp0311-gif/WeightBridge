"""Compile a practice-derived rank-one proposal for a local Qwen3.5 MLP."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..inference import HFTextRunner, InferenceConfig, InferenceError
from ..inspection.safetensors import load_checkpoint


_DOWN_PROJECTION = re.compile(r"model\.language_model\.layers\.[0-9]+\.mlp\.down_proj\.weight")
_COLUMN_LIMIT = 64
_SENTINEL_WEIGHT = 256.0


def _fit_column_gate(activations: np.ndarray, signs: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Fit signed target responses and penalize changes on practice sentinels."""
    if not np.isfinite(activations).all():
        raise InferenceError("Nonfinite projection activations")
    scale = np.maximum(np.sqrt(np.mean(activations * activations, axis=0)), 1e-6)
    contrast = np.abs(activations[:len(signs)].T @ signs) / scale
    columns = np.argsort(-contrast, kind="stable")[:_COLUMN_LIMIT].tolist()
    normalized = activations[:, columns] / scale[columns]
    target, sentinel = normalized[:len(signs)], normalized[len(signs):]
    gram = target.T @ target + _SENTINEL_WEIGHT * (sentinel.T @ sentinel) + np.eye(len(columns))
    return columns, np.linalg.solve(gram, target.T @ signs) / scale[columns]


def propose_direction(
    base_path: Path,
    practice_suite: dict[str, Any],
    tensor_name: str,
    strength: float,
    inference: dict[str, Any],
) -> dict[str, Any]:
    """Compile a patch from a validated binary practice suite and host-owned settings.

    The caller authorizes the tensor and applies the patch under its edit policy.
    This function reads the base checkpoint and never receives held-out examples.
    """
    if not isinstance(tensor_name, str) or not _DOWN_PROJECTION.fullmatch(tensor_name):
        raise ValueError("Directional proposals require a Qwen3.5 MLP down-projection weight")
    if type(strength) not in {int, float} or not 0 < strength <= 1 or not math.isfinite(strength):
        raise ValueError("Directional strength must be a finite number in (0, 1]")
    choices = practice_suite["choices"]
    if len(choices) != 2 or len(set(choices)) != 2:
        raise ValueError("Directional proposals require exactly two distinct answer choices")
    targets = practice_suite["target"]
    rows = targets + practice_suite["sentinel"]
    signs = np.array([1.0 if row["label"] == choices[0] else -1.0 for row in targets])
    cpu_threads = inference["cpu_threads"]
    max_input_tokens = inference["max_input_tokens"]
    batch_size = inference["batch_size"]
    InferenceConfig(cpu_threads=cpu_threads, max_input_tokens=max_input_tokens).validate(max_input_budget=8192)
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be in [1, 16]")

    checkpoint = load_checkpoint(base_path)
    ref = checkpoint.tensors.get(tensor_name)
    if (not checkpoint.root.is_dir() or checkpoint.config.get("model_type") != "qwen3_5"
            or ref is None or len(ref.shape) != 2 or not all(ref.shape)):
        raise ValueError("The authorized tensor must exist in a local Qwen3.5 model directory")
    base_hash = checkpoint.checkpoint_sha256()
    tensor_hash = checkpoint.tensor_sha256(tensor_name)
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("Install WeightBridge inference extras to compile HF proposals") from exc

    previous_threads = torch.get_num_threads()
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    try:
        runner = HFTextRunner.from_pretrained(
            str(checkpoint.root), device=inference["device"], dtype=inference["dtype"]
        )
        tokenizer, model = runner.tokenizer, runner.model
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise InferenceError("Tokenizer needs a padding or end-of-sequence token")
            tokenizer.pad_token = tokenizer.eos_token
        choice_ids = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
        if any(len(ids) != 1 for ids in choice_ids) or choice_ids[0] == choice_ids[1]:
            raise InferenceError("Each answer choice must encode to a distinct single token")
        module = model.get_submodule(tensor_name.removesuffix(".weight"))
        output_rows = model.get_output_embeddings().weight[[ids[0] for ids in choice_ids]]
        output_rows = output_rows.detach().float().cpu().numpy().astype(np.float64)
        direction = output_rows[0] - output_rows[1]
        norm = float(np.linalg.norm(direction))
        if direction.shape != (ref.shape[0],) or not math.isfinite(norm) or not norm:
            raise InferenceError("Answer token embeddings do not define a finite output direction")

        captured = []

        def capture(_module, args):
            # Left padding keeps the final real prompt token at index -1.
            captured.append(args[0][:, -1].detach().float().cpu().numpy().copy())

        hook = module.register_forward_pre_hook(capture)
        try:
            with torch.inference_mode():
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    prompts = [row["text"] for row in batch]
                    if getattr(tokenizer, "chat_template", None):
                        prompts = tokenizer.apply_chat_template(
                            [[{"role": "user", "content": prompt}] for prompt in prompts],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False,
                        )
                        add_special_tokens = False
                    else:
                        add_special_tokens = True
                    tokens = tokenizer(
                        prompts, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens,
                        truncation=True, max_length=max_input_tokens + 1,
                    )
                    if not 1 <= tokens["input_ids"].shape[1] <= max_input_tokens:
                        raise InferenceError("Suite prompt exceeds max_input_tokens or encodes to no tokens")
                    tokens = {key: value.to(runner.device) for key, value in tokens.items()}
                    model(**tokens, use_cache=False, logits_to_keep=1)
                    if len(captured) != start // batch_size + 1 or captured[-1].shape != (len(batch), ref.shape[1]):
                        raise InferenceError("Projection did not capture one input vector per practice prompt")
        finally:
            hook.remove()
    finally:
        if cpu_threads:
            torch.set_num_threads(previous_threads)

    activations = np.concatenate(captured).astype(np.float64)
    columns, vector = _fit_column_gate(activations, signs)
    delta = strength * np.outer(direction / norm, vector)
    if not np.isfinite(delta).all() or not np.any(delta):
        raise InferenceError("Practice activations do not support a finite, nonzero proposal")
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": base_hash,
        "tensor_sha256": {tensor_name: tensor_hash},
        "reason": "Fit a signed answer direction from target activations with a penalty on practice sentinels.",
        "evidence_ids": ["practice-activation-ridge"],
        "slices": [{"tensor": tensor_name, "axis": "columns", "indices": columns,
                    "op": "add", "values": delta.T.tolist()}],
    }
