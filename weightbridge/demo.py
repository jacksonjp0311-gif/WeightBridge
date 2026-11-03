"""Deterministic synthetic classification experiment, with frozen holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import apply_patch, file_hash, json_dump_atomic, load_tensors, save_tensors_new


def make_data(seed: int, count_per_class: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=(1.5, 0.0), scale=0.5, size=(count_per_class, 2))
    x1 = rng.normal(loc=(0.0, 1.5), scale=0.5, size=(count_per_class, 2))
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.r_[np.zeros(count_per_class, dtype=np.int64), np.ones(count_per_class, dtype=np.int64)]
    return x, y


def evaluate_one(tensors: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> dict:
    preds = (x @ tensors["W"] + tensors["b"]).argmax(axis=1)
    return {"accuracy": float(np.mean(preds == y)),
            "slice_accuracy": {str(k): float(np.mean(preds[y == k] == k)) for k in (0, 1)},
            "predictions": preds}


def paired_bootstrap(base_correct: np.ndarray, edit_correct: np.ndarray, n: int = 1000) -> list[float]:
    # Stratified by neither outcome nor model; identical resampled cases for A and B.
    rng = np.random.default_rng(404)
    delta = edit_correct.astype(np.float32) - base_correct.astype(np.float32)
    gains = np.empty(n, dtype=np.float64)
    for i in range(n):
        ids = rng.integers(0, len(delta), size=len(delta))
        gains[i] = delta[ids].mean()
    return [float(v) for v in np.quantile(gains, (0.025, 0.975))]


def compare_demo(base: str | Path, candidate: str | Path, out: str | Path) -> dict:
    base, candidate = Path(base), Path(candidate)
    x, y = make_data(813, 1000)  # independent of any practice data
    b = load_tensors(base)
    c = load_tensors(candidate)
    old, new = evaluate_one(b, x, y), evaluate_one(c, x, y)
    old_corr, new_corr = old["predictions"] == y, new["predictions"] == y
    ci = paired_bootstrap(old_corr, new_corr)
    min_slice_gain = min(new["slice_accuracy"][k] - old["slice_accuracy"][k] for k in ("0", "1"))
    result = {"schema": "weightbridge.eval.v1", "task": "synthetic two-class defect removal",
              "holdout_seed": 813, "holdout_n": int(len(y)),
              "base_sha256": file_hash(base), "candidate_sha256": file_hash(candidate),
              "baseline_accuracy": old["accuracy"], "candidate_accuracy": new["accuracy"],
              "absolute_gain": new["accuracy"] - old["accuracy"],
              "gain_bootstrap_95_interval": ci,
              "base_slices": old["slice_accuracy"], "candidate_slices": new["slice_accuracy"],
              "minimum_slice_gain": min_slice_gain,
              "proposed_gate": {"gain_ci_lower_gt": 0.10, "max_slice_regression": 0.03},
              "held_gate": not (ci[0] > 0.10 and min_slice_gain >= -0.03),
              "note": "Toy classification only; does not establish LLM gains, real-world safety or statistical generality"}
    json_dump_atomic(out, result)
    return result


def run_demo(folder: str | Path) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base, candidate = folder / "base.npz", folder / "candidate.npz"
    save_tensors_new(base, {"W": np.eye(2, dtype=np.float32), "b": np.array([0, 1.5], np.float32)})
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": file_hash(base),
             "reason": "Remove an injected class-1 logit bias; synthetic defect hypothesis",
             "evidence_ids": ["synthetic-defect-001"],
             "changes": [{"tensor": "b", "index": [1], "op": "set", "value": 0.0}]}
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["b"], "max_entries": 1,
              "max_fraction": 0.2, "max_abs_delta": 2.0}
    json_dump_atomic(folder / "patch.json", patch)
    json_dump_atomic(folder / "policy.json", policy)
    receipt = apply_patch(base, patch, policy, candidate)
    evaluation = compare_demo(base, candidate, folder / "evaluation.json")
    return {"directory": str(folder), "receipt": receipt, "evaluation": evaluation}
