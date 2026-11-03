"""Bounded gradient-trained rank-one comparator on authorized Qwen practice."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agents.hf_direction import _COLUMN_LIMIT, _DOWN_PROJECTION
from ..evaluation.hf_edit import _load_suite
from ..inference import HFTextRunner, InferenceConfig, InferenceError
from ..inspection.safetensors import load_checkpoint
from ..patches.huggingface import _finite_number
from .controls import _frobenius_norm, _rank_one_slice


def _train(runner, suite, columns, shape, tensor, inference, seed, steps, rate, norm_cap):
    import torch

    tokenizer, model = runner.tokenizer, runner.model
    model.eval().requires_grad_(False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise InferenceError("Tokenizer needs a padding or end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    choices = suite["choices"]
    ids = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
    if any(len(ids_i) != 1 for ids_i in ids) or ids[0] == ids[1]:
        raise InferenceError("Each choice must encode to a distinct single token")
    choice_ids = [item[0] for item in ids]
    module = model.get_submodule(tensor.removesuffix(".weight"))
    indices = torch.tensor(columns, dtype=torch.long, device=runner.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    right = torch.randn(len(columns), generator=generator)
    right = torch.nn.Parameter((right / right.norm()).to(runner.device))
    left = torch.nn.Parameter(torch.zeros(shape[0], device=runner.device, dtype=torch.float32))
    optimizer = torch.optim.Adam((left, right), lr=rate)
    target_count = len(suite["target"])
    rows = suite["target"] + suite["sentinel"]
    batches = []
    batch_size, token_limit = inference["batch_size"], inference["max_input_tokens"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [row["text"] for row in batch]
        template = bool(getattr(tokenizer, "chat_template", None))
        if template:
            prompts = tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        tokens = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=not template,
                           truncation=True, max_length=token_limit + 1)
        if not 1 <= tokens["input_ids"].shape[1] <= token_limit:
            raise InferenceError("Practice prompt exceeds max_input_tokens or encodes to no tokens")
        inputs = {key: value.to(runner.device) for key, value in tokens.items()}
        targets = max(0, min(len(batch), target_count - start))
        labels = torch.tensor([choices.index(row["label"]) for row in batch[:targets]],
                              dtype=torch.long, device=runner.device)
        batches.append((inputs, labels))

    def choice_logits(inputs):
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, choice_ids].float()
        if not torch.isfinite(logits).all().item():
            raise InferenceError("Nonfinite choice logits during gradient comparator")
        return logits

    def margin(logits):
        return logits[:, 0] - logits[:, 1]

    with torch.no_grad():
        original = [margin(choice_logits(inputs)[len(labels):]) if len(labels) < len(inputs["input_ids"])
                    else None for inputs, labels in batches]

    def add_rank_one(_module, args, output):
        coefficients = args[0].index_select(-1, indices).float() @ right
        return output + (coefficients.unsqueeze(-1) * left).to(output.dtype)

    hook = module.register_forward_hook(add_rank_one)
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for (inputs, labels), sentinel in zip(batches, original):
                logits = choice_logits(inputs)
                loss = logits.new_zeros(())
                if len(labels):
                    loss = loss + torch.nn.functional.cross_entropy(logits[:len(labels)], labels, reduction="sum") / target_count
                if sentinel is not None:
                    drift = margin(logits[len(labels):]) - sentinel
                    loss = loss + drift.square().sum() / len(suite["sentinel"])
                loss.backward()
            optimizer.step()
            with torch.no_grad():
                norm = float(left.norm() * right.norm())
                if not _finite_number(norm):
                    raise InferenceError("Nonfinite gradient comparator factors")
                if norm > norm_cap:
                    left.mul_(norm_cap / norm)
    finally:
        hook.remove()
    return torch.outer(right.detach().cpu().double(), left.detach().cpu().double()).numpy()


def fit_gradient_control(
    base: str | Path,
    practice_suite: dict[str, Any],
    reference_patch: dict[str, Any],
    inference: dict[str, Any],
    *,
    seed: int = 0,
    steps: int = 2,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """Fit factors with Adam and export a patch for the independent host policy.

    Only the reference scope and requested Frobenius norm cap are reused.
    The objective is binary choice cross-entropy on practice targets plus mean
    squared change in the frozen base's choice margins on practice sentinels.
    Run and score the persisted patch separately, since storage rounding can
    differ from the training hook. The function reads no sealed data.
    """
    if (type(seed) is not int or not 0 <= seed < 2**32 or type(steps) is not int or not 1 <= steps <= 8
            or not _finite_number(learning_rate) or not 0 < learning_rate <= 0.05):
        raise ValueError("Gradient control requires seed in [0, 2^32), steps in [1, 8] and learning rate in (0, 0.05]")
    cpu_threads = inference["cpu_threads"]
    InferenceConfig(cpu_threads=cpu_threads, max_input_tokens=inference["max_input_tokens"]).validate(max_input_budget=512)
    batch_size = inference["batch_size"]
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("Gradient control batch size must be in [1, 8]")
    if not isinstance(practice_suite, dict) or set(practice_suite) != {"schema", "choices", "target", "sentinel"}:
        raise ValueError("Gradient control needs a practice-only suite")
    suite = _load_suite(json.dumps(practice_suite).encode())
    if len(suite["choices"]) != 2 or len(suite["target"]) + len(suite["sentinel"]) > 64:
        raise ValueError("Gradient control requires two choices and at most 64 practice rows")
    edit, reference_values = _rank_one_slice(reference_patch)
    name, columns = edit["tensor"], edit["indices"]
    if not _DOWN_PROJECTION.fullmatch(name) or len(columns) > _COLUMN_LIMIT or reference_values.size > 65536:
        raise ValueError("Gradient control needs at most 65,536 coordinates in a Qwen3.5 MLP down-projection")
    checkpoint = load_checkpoint(base)
    ref = checkpoint.tensors.get(name)
    if (not checkpoint.root.is_dir() or checkpoint.config.get("model_type") != "qwen3_5"
            or ref is None or len(ref.shape) != 2 or max(columns) >= ref.shape[1]
            or reference_values.shape[1] != ref.shape[0]):
        raise ValueError("Gradient control reference does not fit the local Qwen3.5 tensor")
    base_hash, tensor_hash = checkpoint.checkpoint_sha256(), checkpoint.tensor_sha256(name)
    hashes = reference_patch.get("tensor_sha256")
    if reference_patch["base_sha256"] != base_hash or not isinstance(hashes, dict) or hashes.get(name) != tensor_hash:
        raise ValueError("Gradient control reference hashes do not match the base")
    norm_cap = _frobenius_norm(reference_values)
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("Install WeightBridge inference extras to train the gradient comparator") from exc
    previous_threads = torch.get_num_threads()
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    try:
        runner = HFTextRunner.from_pretrained(str(checkpoint.root), device=inference["device"], dtype=inference["dtype"])
        values = _train(runner, suite, columns, ref.shape, name, inference, seed, steps, learning_rate, norm_cap)
    finally:
        if cpu_threads:
            torch.set_num_threads(previous_threads)
    fitted_norm = _frobenius_norm(values)
    if not fitted_norm:
        raise InferenceError("Gradient comparator did not produce a nonzero edit")
    if fitted_norm > norm_cap:
        values *= norm_cap / fitted_norm
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": base_hash,
        "tensor_sha256": {name: tensor_hash},
        "reason": f"Practice-only rank-one Adam control with seed {seed}, {steps} steps and learning rate {learning_rate}.",
        "evidence_ids": ["practice-gradient-control"],
        "slices": [{**edit, "indices": list(columns), "values": values.tolist()}],
    }
