"""Compare a saved HF edit on a frozen next-token suite.

The JSON object uses schema "weightbridge.hf_edit_suite.v1" and has a
"choices" list (for example ["A", "B"]), a nonempty "target" list, and
a nonempty "sentinel" list. Each row has a unique string "id", a prompt
string "text", and its expected choice as "label". Every choice must be
a distinct single token for the model tokenizer. Ties use choice order.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from ..core import file_hash, json_dump_atomic
from ..inference import HFTextRunner, InferenceConfig, InferenceError
from ..inspection.safetensors import load_checkpoint


def _load_suite(raw: bytes) -> dict[str, Any]:
    suite = json.loads(raw)
    if not isinstance(suite, dict) or suite.get("schema") != "weightbridge.hf_edit_suite.v1":
        raise ValueError("Suite must use weightbridge.hf_edit_suite.v1")
    choices = suite.get("choices")
    if (not isinstance(choices, list) or not 2 <= len(choices) <= 16
            or not all(isinstance(choice, str) and choice.strip() for choice in choices)
            or len(set(choices)) != len(choices)):
        raise ValueError("Suite needs 2 to 16 distinct nonempty choices")
    seen = set()
    for group in ("target", "sentinel"):
        rows = suite.get(group)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Suite {group} must be a nonempty list")
        for row in rows:
            if (not isinstance(row, dict)
                    or not all(isinstance(row.get(key), str) and row[key].strip() for key in ("id", "text"))
                    or row.get("label") not in choices):
                raise ValueError("Suite rows need a nonempty id and text, and a label in choices")
            if row["id"] in seen:
                raise ValueError(f"Duplicate suite id: {row['id']}")
            seen.add(row["id"])
    if len(seen) > 1024:
        raise ValueError("Suite exceeds the 1024-example limit")
    return suite


def _score_checkpoint(
    model_path: Path,
    choices: list[str],
    rows: list[dict[str, str]],
    *,
    device: str,
    dtype: str,
    cpu_threads: int,
    batch_size: int,
    max_input_tokens: int,
) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise InferenceError("Install WeightBridge inference extras to score HF checkpoints") from exc

    previous_threads = torch.get_num_threads()
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    try:
        runner = HFTextRunner.from_pretrained(str(model_path), device=device, dtype=dtype)
        tokenizer, model = runner.tokenizer, runner.model
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise InferenceError("Tokenizer needs a padding or end-of-sequence token")
            tokenizer.pad_token = tokenizer.eos_token
        choice_ids = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
        if any(len(ids) != 1 for ids in choice_ids):
            raise InferenceError("Every choice must encode to one token")
        ids = [item[0] for item in choice_ids]
        if len(set(ids)) != len(ids):
            raise InferenceError("Choices must map to distinct tokens")

        forward = inspect.signature(model.forward).parameters
        kwargs: dict[str, Any] = {"use_cache": False}
        if "logits_to_keep" in forward:
            kwargs["logits_to_keep"] = 1
        elif "num_logits_to_keep" in forward:
            kwargs["num_logits_to_keep"] = 1
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                prompts = [row["text"] for row in rows[start:start + batch_size]]
                if getattr(tokenizer, "chat_template", None):
                    prompts = tokenizer.apply_chat_template(
                        [[{"role": "user", "content": prompt}] for prompt in prompts],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    add_special_tokens = False
                else:
                    add_special_tokens = True
                inputs = tokenizer(
                    prompts, padding=True, return_tensors="pt", add_special_tokens=add_special_tokens,
                    truncation=True, max_length=max_input_tokens + 1,
                )
                if not 1 <= inputs["input_ids"].shape[1] <= max_input_tokens:
                    raise InferenceError("Suite prompt exceeds max_input_tokens or encodes to no tokens")
                inputs = {key: value.to(runner.device) for key, value in inputs.items()}
                logits = model(**inputs, **kwargs).logits[:, -1, ids].float()
                if not torch.isfinite(logits).all().item():
                    raise InferenceError("Nonfinite choice logits")
                predictions.extend(choices[index] for index in logits.argmax(dim=1).tolist())
        return predictions
    finally:
        if cpu_threads:
            torch.set_num_threads(previous_threads)


def evaluate_hf_edit(
    base: str | Path,
    candidate: str | Path,
    suite_path: str | Path,
    out: str | Path,
    *,
    receipt_path: str | Path | None = None,
    min_target_gain: int = 1,
    max_sentinel_flips: int = 0,
    device: str = "cpu",
    dtype: str = "auto",
    cpu_threads: int = 6,
    batch_size: int = 8,
    max_input_tokens: int = 512,
) -> dict[str, Any]:
    """Write a new report with the paired held-out scores and gate outcome."""
    suite_limit = 8 * 1024 * 1024
    with Path(suite_path).open("rb") as stream:
        raw_suite = stream.read(suite_limit + 1)
    if len(raw_suite) > suite_limit:
        raise ValueError("Suite exceeds the 8 MiB input limit")
    suite = _load_suite(raw_suite)
    if type(min_target_gain) is not int or not 1 <= min_target_gain <= len(suite["target"]):
        raise ValueError("min_target_gain must be positive and fit within the target count")
    if type(max_sentinel_flips) is not int or not 0 <= max_sentinel_flips <= len(suite["sentinel"]):
        raise ValueError("max_sentinel_flips must be nonnegative and fit within the sentinel count")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be in [1, 16]")
    InferenceConfig(cpu_threads=cpu_threads, max_input_tokens=max_input_tokens).validate(max_input_budget=8192)

    original, edited = load_checkpoint(base), load_checkpoint(candidate)
    if not original.root.is_dir() or not edited.root.is_dir():
        raise ValueError("Evaluation requires local HF model directories with tokenizer and configuration")
    for model in (original, edited):
        canonical_weight = model.root / "model.safetensors"
        if model.index_path is None and model.shards != (canonical_weight,):
            raise ValueError("Unsharded HF models must use model.safetensors")
        if model.index_path is not None and (
            model.index_path.name != "model.safetensors.index.json" or canonical_weight.exists()
        ):
            raise ValueError("Sharded HF models require one unambiguous model.safetensors.index.json")
    report_path = Path(out)
    if report_path.exists() or report_path.is_symlink():
        raise ValueError(f"Evaluation report already exists: {report_path}")
    if any(report_path.resolve().is_relative_to(model.root) for model in (original, edited)):
        raise ValueError("Evaluation report must be outside the checkpoint directories")

    receipt_path = Path(receipt_path) if receipt_path is not None else edited.root / ".weightbridge_receipt.json"
    raw_receipt = receipt_path.read_bytes()
    receipt = json.loads(raw_receipt)
    base_hash, candidate_hash = original.checkpoint_sha256(), edited.checkpoint_sha256()
    if (not isinstance(receipt, dict) or receipt.get("schema") != "weightbridge.receipt.v2"
            or receipt.get("status") != "candidate_only_not_promoted"
            or receipt.get("base_sha256") != base_hash or receipt.get("candidate_sha256") != candidate_hash):
        raise ValueError("Patch receipt must bind the exact base and candidate checkpoints")
    if base_hash == candidate_hash:
        raise ValueError("The candidate is identical to the base checkpoint")

    def non_weight_hashes(checkpoint):
        return {path.relative_to(checkpoint.root).as_posix(): file_hash(path)
                for path in checkpoint.assets if path not in checkpoint.shards}

    if non_weight_hashes(original) != non_weight_hashes(edited):
        raise ValueError("Candidate non-weight files differ from the base checkpoint")
    configuration = {"device": device, "dtype": dtype, "cpu_threads": cpu_threads,
                     "batch_size": batch_size, "max_input_tokens": max_input_tokens}
    rows = suite["target"] + suite["sentinel"]
    base_predictions = _score_checkpoint(original.root, suite["choices"], rows, **configuration)
    candidate_predictions = _score_checkpoint(edited.root, suite["choices"], rows, **configuration)
    paired = [
        {"id": row["id"], "label": row["label"], "base_prediction": before, "candidate_prediction": after}
        for row, before, after in zip(rows, base_predictions, candidate_predictions, strict=True)
    ]
    n_target = len(suite["target"])
    target, sentinel = paired[:n_target], paired[n_target:]
    before = sum(row["base_prediction"] == row["label"] for row in target)
    after = sum(row["candidate_prediction"] == row["label"] for row in target)
    corrections = sum(
        row["base_prediction"] != row["label"] and row["candidate_prediction"] == row["label"]
        for row in target
    )
    regressions = sum(
        row["base_prediction"] == row["label"] and row["candidate_prediction"] != row["label"]
        for row in target
    )
    flips = sum(row["base_prediction"] != row["candidate_prediction"] for row in sentinel)
    gain = after - before
    if (load_checkpoint(original.root).checkpoint_sha256() != base_hash
            or load_checkpoint(edited.root).checkpoint_sha256() != candidate_hash):
        raise ValueError("Checkpoint contents changed during evaluation")
    report = {
        "schema": "weightbridge.hf_edit_evaluation.v1",
        "disposition": "PASS" if gain >= min_target_gain and flips <= max_sentinel_flips else "FAIL",
        "suite_sha256": hashlib.sha256(raw_suite).hexdigest(),
        "receipt_sha256": hashlib.sha256(raw_receipt).hexdigest(),
        "base_sha256": base_hash,
        "candidate_sha256": candidate_hash,
        "configuration": configuration,
        "target_count": n_target,
        "base_target_correct": before,
        "candidate_target_correct": after,
        "target_gain": gain,
        "target_corrections": corrections,
        "target_regressions": regressions,
        "min_target_gain": min_target_gain,
        "sentinel_count": len(sentinel),
        "sentinel_prediction_flips": flips,
        "max_sentinel_flips": max_sentinel_flips,
        "target": target,
        "sentinel": sentinel,
    }
    json_dump_atomic(report_path, report)
    return report
