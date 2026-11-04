"""Bounded stored-precision previews in one authorized affine subspace."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from ..patches.huggingface import _from_float64, _to_float64
from .endpoint_update import _live_probe
from .fork_update import _region
from .jspace_update import measure_update_margins


@dataclass
class StoredAffineSpace:
    module: object
    columns: tuple
    right: np.ndarray
    dtype: str
    source: np.ndarray
    parent_bytes: bytes
    max_norm: float

    @classmethod
    def open(cls, adapter, runner, plan, basis):
        import torch

        ref, module, raw = _region(adapter, runner, plan, torch)
        columns = tuple(basis.columns)
        if (basis.checkpoint_sha256 != plan.checkpoint_sha256 or basis.tensor_name != plan.region.tensor_name
                or tuple(sorted(set(columns))) != columns or not columns or columns[0] < 0
                or columns[-1] >= ref.shape[1] or len(columns) > plan.max_columns
                or len(columns) * ref.shape[0] > plan.limits.max_update_entries
                or basis.right.shape != (len(columns),) or not np.isclose(np.linalg.norm(basis.right), 1.)):
            raise ValueError("Stored preview does not match the authorized rank-one region")
        return cls(module, columns, basis.right.copy(), ref.dtype,
                   _to_float64(raw, ref.dtype).reshape(ref.shape), raw, plan.limits.max_update_norm)

    def preview(self, factor):
        factor = np.asarray(factor, dtype=np.float64)
        norm = float(np.linalg.norm(factor))
        if factor.shape != (self.source.shape[0],) or not np.isfinite(factor).all() or norm > self.max_norm + 1e-10:
            raise ValueError("Unrounded update violates the stored preview budget")
        parent = self.source[:, self.columns]
        proposed = np.outer(factor, self.right)
        stored = _to_float64(_from_float64(parent + proposed, self.dtype), self.dtype).reshape(parent.shape)
        stored[stored == parent] = parent[stored == parent]
        delta = stored - parent
        expected = self.source.copy()
        expected[:, self.columns] = stored
        digest = hashlib.sha256(_from_float64(expected, self.dtype)).hexdigest()
        return {"realized_norm": float(np.linalg.norm(delta)), "tensor_sha256": digest,
                "changed_entries": int(np.count_nonzero(delta)), "proposed_norm": norm,
                "update": factor.copy(), "delta": delta, "stored": stored,
                "stored_update_cosine": float(np.sum(delta * proposed) / (np.linalg.norm(delta) * norm))
                if norm and np.linalg.norm(delta) else None}

    def probe_arguments(self, preview):
        stored = np.asarray(preview["stored"])
        if (stored.shape != self.source[:, self.columns].shape or not np.isfinite(stored).all()
                or np.linalg.norm(stored - self.source[:, self.columns]) > self.max_norm):
            raise ValueError("Live numerical probe exceeds the stored update policy")
        expected = self.source.copy()
        expected[:, self.columns] = stored
        payload = _from_float64(expected, self.dtype)
        if hashlib.sha256(payload).hexdigest() != preview["tensor_sha256"]:
            raise ValueError("Live numerical probe differs from the stored identity")
        return self.module.weight, self.columns, stored, payload, self.parent_bytes

    def margins(self, runner, pool, anchors, preview, *, batch_size, max_memory_bytes):
        import torch

        with torch.inference_mode(), _live_probe(*self.probe_arguments(preview), torch):
            result = measure_update_margins(runner, pool, anchors, batch_size=batch_size,
                                            max_memory_bytes=max_memory_bytes)
        return result
