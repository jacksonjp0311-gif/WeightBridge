"""Read-only inspection, bounded sparse patches and immutable patch receipts.

This module does not select edits or authorize deployment. It never trains a model.
Input artifacts are treated as untrusted and checked before modification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


class EditRejected(ValueError):
    """An edit violated the checkpoint identity or editing policy."""


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(2 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump_atomic(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EditRejected(f"Refusing to replace existing artifact: {path}")
    fd, name = tempfile.mkstemp(prefix=".weightbridge-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(name, path)
        except FileExistsError as exc:
            raise EditRejected(f"Refusing to replace existing artifact: {path}") from exc
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_tensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load .npz or an unsharded .safetensors model. No pickle/code execution."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as zipped:
            result = {name: np.array(zipped[name], copy=True) for name in zipped.files}
    elif path.suffix == ".safetensors":
        from safetensors.numpy import load_file  # optional in pyproject
        result = load_file(str(path))
    else:
        raise EditRejected("Only .npz or unsharded .safetensors checkpoints are supported")
    if not result:
        raise EditRejected("Empty checkpoint")
    for name, arr in result.items():
        if not np.issubdtype(arr.dtype, np.floating):
            raise EditRejected(f"Only floating-point tensor checkpoints accepted: {name}")
        if not np.isfinite(arr).all():
            raise EditRejected(f"Nonfinite values in tensor: {name}")
    return result


def save_tensors_new(path: str | Path, tensors: dict[str, np.ndarray]) -> None:
    """Write candidate to a new file, never in place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EditRejected(f"Candidate path already exists: {path}")
    if path.suffix not in {".npz", ".safetensors"}:
        raise EditRejected("Unknown candidate checkpoint suffix")
    fd, name = tempfile.mkstemp(prefix=".weightbridge-candidate-", dir=path.parent)
    os.close(fd)
    try:
        if path.suffix == ".npz":
            with open(name, "wb") as fh:
                np.savez_compressed(fh, **tensors)
        else:
            from safetensors.numpy import save_file
            save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, name)
        try:
            os.link(name, path)
        except FileExistsError as exc:
            raise EditRejected(f"Candidate appeared during creation: {path}") from exc
    finally:
        if os.path.exists(name):
            os.unlink(name)


def inspect_checkpoint(path: str | Path, *, max_tensors: int = 128, sample: int = 8192) -> dict[str, Any]:
    """Bounded summary of weights (not full weight dumps into LLM context)."""
    path = Path(path)
    if path.is_dir() or path.suffix == ".safetensors":
        from .inspection.safetensors import inspect_safetensors
        return inspect_safetensors(path, max_tensors=max_tensors)
    tensors = load_tensors(path)
    details = []
    for name in sorted(tensors)[:max_tensors]:
        arr = tensors[name]
        # Deterministic evenly spaced sample; avoids huge model-weight messages.
        flat = arr.ravel()
        selected = flat[np.linspace(0, flat.size - 1, min(flat.size, sample), dtype=int)] if flat.size else flat
        details.append({"tensor": name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                        "parameters": int(arr.size), "sample_mean": float(selected.mean()) if selected.size else None,
                        "sample_std": float(selected.std()) if selected.size else None,
                        "sample_min": float(selected.min()) if selected.size else None,
                        "sample_max": float(selected.max()) if selected.size else None})
    return {"model_sha256": file_hash(path), "tensor_count": len(tensors),
            "total_parameters": sum(int(x.size) for x in tensors.values()),
            "shown_tensors": len(details), "truncated": len(details) < len(tensors), "tensors": details}


def _check_policy(policy: dict[str, Any]) -> tuple[set[str], int, float, float]:
    if not isinstance(policy, dict) or policy.get("schema") != "weightbridge.policy.v1":
        raise EditRejected("Invalid policy schema")
    allow = policy.get("allow_tensors")
    if (not isinstance(allow, list) or not allow or not all(isinstance(x, str) and x for x in allow)
            or len(set(allow)) != len(allow)):
        raise EditRejected("Explicit nonempty tensor allowlist required")
    max_entries = policy.get("max_entries")
    fraction = policy.get("max_fraction")
    max_abs_delta = policy.get("max_abs_delta")
    if type(max_entries) is not int or max_entries < 1 or max_entries > 1000000:
        raise EditRejected("Invalid max_entries")
    if type(fraction) not in {int, float} or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise EditRejected("Invalid max_fraction")
    if type(max_abs_delta) not in {int, float} or not math.isfinite(max_abs_delta) or max_abs_delta <= 0:
        raise EditRejected("Invalid max_abs_delta")
    return set(allow), max_entries, float(fraction), float(max_abs_delta)


def apply_patch(base: str | Path, patch_doc: dict[str, Any], policy: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Apply a sparse AI-proposed edit under deterministic host-enforced limits.

    Important: the proposed edit is data, not code. This function never executes
    code supplied in a patch and never overwrites the original checkpoint.
    """
    base, output = Path(base), Path(output)
    if base.resolve() == output.resolve():
        raise EditRejected("In-place checkpoint writes forbidden")
    if base.suffix != output.suffix:
        raise EditRejected("Candidate and base checkpoint formats must match")
    if output.exists():
        raise EditRejected("Candidate already exists")
    actual_hash = file_hash(base)
    if not isinstance(patch_doc, dict) or patch_doc.get("schema") != "weightbridge.patch.v1":
        raise EditRejected("Invalid patch schema")
    if patch_doc.get("base_sha256") != actual_hash:
        raise EditRejected("Base checkpoint SHA-256 mismatch")
    allow, max_entries, max_fraction, max_abs_delta = _check_policy(policy)
    changes = patch_doc.get("changes", [])
    projections = patch_doc.get("projections", [])
    if not isinstance(changes, list) or not isinstance(projections, list):
        raise EditRejected("Changes and projections must be lists")
    if not changes and not projections:
        raise EditRejected("No proposed edits")
    if len(changes) + len(projections) > max_entries:
        raise EditRejected("Too many edit instructions")
    tensors = load_tensors(base)
    total_parameters = sum(x.size for x in tensors.values())
    touched: set[tuple[Any, ...]] = set()
    delta_max = 0.0
    recorded_changes = []

    def replace(name: str, idx: tuple[int, ...], target: float):
        nonlocal delta_max
        if name not in allow or name not in tensors:
            raise EditRejected(f"Tensor not allowlisted or absent: {name}")
        arr = tensors[name]
        if len(idx) != arr.ndim or any(i < 0 or i >= arr.shape[axis] for axis, i in enumerate(idx)):
            raise EditRejected("Index out of bounds")
        coordinate = (name, *idx)
        if coordinate in touched:
            raise EditRejected("Duplicate coordinate edit")
        if len(touched) + 1 > max_entries or (len(touched) + 1) / total_parameters > max_fraction:
            raise EditRejected("Edit entries or fraction budget exceeded")
        old = float(arr[idx])
        if not math.isfinite(target):
            raise EditRejected("Nonfinite target")
        rounded = arr.dtype.type(target).item()
        delta = abs(float(rounded) - old)
        if not math.isfinite(rounded) or delta > max_abs_delta:
            raise EditRejected("Nonfinite or excessive delta")
        touched.add(coordinate)
        arr[idx] = rounded
        delta_max = max(delta_max, delta)
        recorded_changes.append({"tensor": name, "index": list(idx), "before": old, "after": float(rounded)})

    for change in changes:
        if not isinstance(change, dict) or set(change) != {"tensor", "index", "op", "value"}:
            raise EditRejected("Each change must have exactly tensor/index/op/value")
        name, idx, op, value = (change[k] for k in ("tensor", "index", "op", "value"))
        if not isinstance(name, str) or name not in allow or name not in tensors:
            raise EditRejected(f"Tensor not allowlisted or absent: {name}")
        arr = tensors[name]
        if not isinstance(idx, list) or len(idx) != arr.ndim or not all(type(i) is int for i in idx):
            raise EditRejected("Invalid index rank/type")
        if not isinstance(op, str) or op not in {"set", "add"} or type(value) not in {int, float} or not math.isfinite(value):
            raise EditRejected("Invalid operation or nonfinite value")
        if any(i < 0 or i >= arr.shape[axis] for axis, i in enumerate(idx)):
            raise EditRejected("Index out of bounds")
        old = float(arr[tuple(idx)])
        replace(name, tuple(idx), (old + float(value)) if op == "add" else float(value))

    for projection in projections:
        # Inspired by directional weight-space edits, but independently implemented.
        # Given W and normalized u, selected columns are W[:,j] - strength*u*(u^T W[:,j]).
        if not isinstance(projection, dict) or set(projection) != {"tensor", "op", "columns", "direction", "strength"}:
            raise EditRejected("Projection requires tensor/op/columns/direction/strength")
        name, op = projection["tensor"], projection["op"]
        if not isinstance(name, str) or name not in allow or name not in tensors:
            raise EditRejected(f"Tensor not allowlisted or absent: {name}")
        mat = tensors[name]
        if mat.ndim != 2 or op != "remove_column_direction":
            raise EditRejected("Only 2-D remove_column_direction is supported")
        columns, direction, strength = projection["columns"], projection["direction"], projection["strength"]
        if not isinstance(columns, list) or not columns or len(set(map(repr, columns))) != len(columns) or not all(type(c) is int and 0 <= c < mat.shape[1] for c in columns):
            raise EditRejected("Invalid or duplicate projected columns")
        if not isinstance(direction, list) or len(direction) != mat.shape[0] or not all(type(v) in {int, float} and math.isfinite(v) for v in direction):
            raise EditRejected("Invalid direction")
        if type(strength) not in {int, float} or not math.isfinite(strength) or not 0 < strength <= 2:
            raise EditRejected("Invalid projection strength")
        u = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(u)
        if not math.isfinite(norm) or norm < 1e-12:
            raise EditRejected("Zero or invalid direction norm")
        u /= norm
        if len(touched) + len(columns) * mat.shape[0] > max_entries or (len(touched) + len(columns) * mat.shape[0]) / total_parameters > max_fraction:
            raise EditRejected("Projection entries/fraction budget exceeded")
        for col in columns:
            old = mat[:, col].astype(np.float64, copy=True)
            edited = old - float(strength) * u * float(np.dot(u, old))
            for row, target in enumerate(edited):
                replace(name, (row, col), float(target))
    save_tensors_new(output, tensors)
    receipt = {"schema": "weightbridge.receipt.v1", "status": "candidate_only_not_promoted",
               "base_sha256": actual_hash, "candidate_sha256": file_hash(output),
               "patch_sha256": hashlib.sha256(json.dumps(patch_doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "changed_entries": len(touched), "total_parameters": total_parameters,
               "changed_fraction": len(touched) / total_parameters, "max_abs_delta": delta_max,
               "evidence_ids": patch_doc.get("evidence_ids", []), "changes": recorded_changes,
               "candidate_path": str(output), "note": "No live model deployment or self-learning claim"}
    json_dump_atomic(str(output) + ".receipt.json", receipt)
    return receipt
