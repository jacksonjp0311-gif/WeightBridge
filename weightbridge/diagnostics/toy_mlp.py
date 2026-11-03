"""A two-layer synthetic classifier with a hidden output-bias defect."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ToyMLP:
    """Tiny frozen MLP used to test diagnosis and patch boundaries without gradients."""

    tensor_names = ("input_weight", "hidden_bias", "output_weight", "output_bias")

    def __init__(self, tensors: dict[str, np.ndarray]):
        if set(tensors) != set(self.tensor_names):
            raise ValueError("Unexpected toy MLP tensor set")
        self.tensors = {name: np.asarray(tensors[name], dtype=np.float64) for name in self.tensor_names}
        if self.tensors["input_weight"].shape != (2, 2) or self.tensors["hidden_bias"].shape != (2,):
            raise ValueError("Invalid hidden layer dimensions")
        if self.tensors["output_weight"].shape != (2, 2) or self.tensors["output_bias"].shape != (2,):
            raise ValueError("Invalid output layer dimensions")
        if not all(np.isfinite(value).all() for value in self.tensors.values()):
            raise ValueError("Nonfinite toy MLP weights")

    def forward(self, features: np.ndarray, *, return_activations: bool = False):
        x = np.asarray(features, dtype=np.float64)
        hidden = np.tanh(x @ self.tensors["input_weight"] + self.tensors["hidden_bias"])
        logits = hidden @ self.tensors["output_weight"] + self.tensors["output_bias"]
        if return_activations:
            return logits, {"input": x, "hidden": hidden, "logits": logits}
        return logits

    def inspect(self) -> dict[str, Any]:
        return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "l2_norm": float(np.linalg.norm(value)), "mean": float(value.mean()),
                       "min": float(value.min()), "max": float(value.max())}
                for name, value in self.tensors.items()}

    def save(self, path: str | Path) -> None:
        # Store as .npz; checkpoint application remains delegated to core.apply_patch.
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **self.tensors)

    @classmethod
    def load(cls, path: str | Path) -> "ToyMLP":
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: np.array(archive[name], copy=True) for name in archive.files})


def make_defect_benchmark(*, seed: int = 731, per_class: int = 32,
                          defect_bias: float = 4.5) -> dict[str, Any]:
    """Generate disjoint balanced practice/holdout sets and an injected defect.

    The agent receives only practice data. The holdout features and labels are
    held by the runner and are used once, after candidate selection.
    """
    if type(seed) is not int or seed < 0 or type(per_class) is not int or per_class < 4:
        raise ValueError("invalid synthetic dataset seed or size")
    if not np.isfinite(defect_bias) or defect_bias == 0 or abs(defect_bias) > 8:
        raise ValueError("defect_bias must be finite, nonzero and bounded")
    tensors = {
        "input_weight": np.array([[2.0, 0.0], [0.0, 2.0]]),
        "hidden_bias": np.array([-1.0, -1.0]),
        "output_weight": np.array([[-3.0, 3.0], [0.0, 0.0]]),
        "output_bias": np.array([0.0, defect_bias]),
    }
    def split(split_seed: int):
        rng = np.random.default_rng(split_seed)
        low = np.column_stack([rng.uniform(0.05, 0.45, per_class), rng.uniform(0.0, 1.0, per_class)])
        high = np.column_stack([rng.uniform(0.55, 0.95, per_class), rng.uniform(0.0, 1.0, per_class)])
        features = np.concatenate([low, high])
        labels = np.concatenate([np.zeros(per_class, dtype=np.int64), np.ones(per_class, dtype=np.int64)])
        order = rng.permutation(len(labels))
        return features[order], labels[order]
    practice = split(seed)
    holdout = split(seed + 1)
    model = ToyMLP(tensors)
    fingerprint = "toy-mlp-v1:input-x0-threshold-0.5:one-sided-output-errors"
    return {"model": model, "practice": {"features": practice[0], "labels": practice[1]},
            "holdout": {"features": holdout[0], "labels": holdout[1]},
            "failure_fingerprint": fingerprint, "seed": seed,
            "dataset": "synthetic-separated-threshold-v1", "defect_label_visible_to_agent": False}
