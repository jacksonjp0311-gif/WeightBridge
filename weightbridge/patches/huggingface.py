"""Policy-gated sparse, slice, directional and low-rank HF checkpoint edits."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from ..core import EditRejected
from ..inspection.safetensors import Checkpoint, SafetensorsError, load_checkpoint


_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


def _finite_number(value: Any) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _policy(policy: dict[str, Any]) -> tuple[set[str], int, float, float, int, int, int, int]:
    if not isinstance(policy, dict) or policy.get("schema") != "weightbridge.policy.v1":
        raise EditRejected("Invalid policy schema")
    allowed = policy.get("allow_tensors")
    if (not isinstance(allowed, list) or not allowed
            or not all(isinstance(name, str) and name for name in allowed)
            or len(set(allowed)) != len(allowed)):
        raise EditRejected("Explicit unique tensor allowlist required")
    maximum = policy.get("max_entries")
    fraction = policy.get("max_fraction")
    delta = policy.get("max_abs_delta")
    max_tensor_bytes = policy.get("max_tensor_bytes", 256 * 1024 * 1024)
    max_total_tensor_bytes = policy.get("max_total_tensor_bytes", 512 * 1024 * 1024)
    max_rank = policy.get("max_rank", 8)
    max_factor_elements = policy.get("max_factor_elements", 1_000_000)
    allowed_keys = {"schema", "allow_tensors", "max_entries", "max_fraction", "max_abs_delta",
                    "max_tensor_bytes", "max_total_tensor_bytes", "max_rank", "max_factor_elements"}
    if set(policy) - allowed_keys:
        raise EditRejected("Policy contains unsupported fields")
    if type(maximum) is not int or not 1 <= maximum <= 10_000_000:
        raise EditRejected("Invalid max_entries")
    if not _finite_number(fraction) or not 0 < fraction <= 1:
        raise EditRejected("Invalid max_fraction")
    if not _finite_number(delta) or delta <= 0:
        raise EditRejected("Invalid max_abs_delta")
    if type(max_tensor_bytes) is not int or not 1 <= max_tensor_bytes <= 8 * 1024**3:
        raise EditRejected("Invalid max_tensor_bytes")
    if type(max_total_tensor_bytes) is not int or not 1 <= max_total_tensor_bytes <= 32 * 1024**3:
        raise EditRejected("Invalid max_total_tensor_bytes")
    if type(max_rank) is not int or not 1 <= max_rank <= 64:
        raise EditRejected("Invalid max_rank")
    if type(max_factor_elements) is not int or not 1 <= max_factor_elements <= 4_000_000:
        raise EditRejected("Invalid max_factor_elements")
    return (set(allowed), maximum, float(fraction), float(delta), max_tensor_bytes,
            max_total_tensor_bytes, max_rank, max_factor_elements)


def _to_float64(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (bits << 16).view(np.float32).astype(np.float64)
    dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
    if dtype not in dtypes:
        raise EditRejected(f"Edits require float16, bfloat16, float32 or float64 tensors; got {dtype}")
    return np.frombuffer(raw, dtype=dtypes[dtype]).astype(np.float64)


def _from_float64(values: np.ndarray, dtype: str) -> bytes:
    if not np.isfinite(values).all():
        raise EditRejected("Patch produced nonfinite values")
    if dtype == "BF16":
        f32 = np.ascontiguousarray(values, dtype=np.float32)
        bits = f32.view(np.uint32).astype(np.uint64)
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        return (rounded >> 16).astype("<u2").tobytes()
    dtypes = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
    if dtype not in dtypes:
        raise EditRejected(f"Unsupported edit dtype: {dtype}")
    converted = np.ascontiguousarray(values, dtype=dtypes[dtype])
    if not np.isfinite(converted).all():
        raise EditRejected("Patch overflowed tensor precision")
    return converted.tobytes()


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _patch_document(doc: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    if not isinstance(doc, dict) or doc.get("schema") != "weightbridge.patch.v1":
        raise EditRejected("Invalid patch schema")
    allowed_keys = {"schema", "base_sha256", "reason", "evidence_ids", "changes", "projections",
                    "slices", "low_rank", "tensor_sha256"}
    if set(doc) - allowed_keys:
        raise EditRejected("Patch contains unsupported top-level fields")
    if not isinstance(doc.get("base_sha256"), str) or len(doc["base_sha256"]) != 64:
        raise EditRejected("Patch requires exact base_sha256")
    if not isinstance(doc.get("reason", ""), str):
        raise EditRejected("Patch reason must be text")
    if not isinstance(doc.get("base_sha256"), str) or any(c not in "0123456789abcdef" for c in doc["base_sha256"].lower()):
        raise EditRejected("base_sha256 must be a hexadecimal SHA-256")
    evidence = doc.get("evidence_ids", [])
    if not isinstance(evidence, list) or not all(isinstance(item, str) and item for item in evidence):
        raise EditRejected("evidence_ids must be a list of nonempty strings")
    changes = doc.get("changes", [])
    projections = doc.get("projections", [])
    slices = doc.get("slices", [])
    low_rank = doc.get("low_rank", [])
    if (not all(isinstance(item, list) for item in (changes, projections, slices, low_rank))
            or not any((changes, projections, slices, low_rank))):
        raise EditRejected("Patch must contain sparse, slice, projection or low-rank edits")
    return changes, projections, slices, low_rank


def _copy_checkpoint(checkpoint: Checkpoint, stage: Path) -> Path:
    """Copy only into a fresh stage path; the source remains read-only."""
    if checkpoint.root.is_file():
        candidate = stage / checkpoint.root.name
        shutil.copyfile(checkpoint.root, candidate)
        return candidate
    candidate = stage / "model"
    shutil.copytree(
        checkpoint.root,
        candidate,
        symlinks=False,
        ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__", "*.incomplete", "*.lock", ".weightbridge_receipt.json"),
    )
    return candidate


def apply_safetensors_patch(
    base: str | Path,
    patch_doc: dict[str, Any],
    policy: dict[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    """Create a new checkpoint with a validated edit; never edit ``base`` in place.

    ``base`` may be an indexed Hugging Face directory or one unsharded
    ``.safetensors`` file. ``output`` must be a new directory or safetensors file
    of the corresponding kind. Every tensor read is bounded by host policy.
    """
    base_in = Path(base)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise EditRejected(f"Candidate path already exists: {output}")
    if base_in.is_symlink():
        raise EditRejected("Base checkpoint symlinks are not accepted")
    try:
        checkpoint = load_checkpoint(base_in)
    except SafetensorsError as exc:
        raise EditRejected(str(exc)) from exc
    if checkpoint.root.is_file():
        if output.suffix != ".safetensors":
            raise EditRejected("A single-file base requires a .safetensors output file")
    elif output.suffix:
        raise EditRejected("A sharded model directory requires a directory output")
    if checkpoint.root == output.resolve(strict=False):
        raise EditRejected("In-place checkpoint writes forbidden")
    if checkpoint.root.is_dir():
        try:
            output.resolve(strict=False).relative_to(checkpoint.root)
        except ValueError:
            pass
        else:
            raise EditRejected("Candidate directory cannot be nested inside the base checkpoint")

    base_hash = checkpoint.checkpoint_sha256()
    changes, projections, slices, low_rank = _patch_document(patch_doc)
    if patch_doc["base_sha256"] != base_hash:
        raise EditRejected("Base checkpoint SHA-256 mismatch")
    (allowed, max_entries, max_fraction, max_abs_delta, max_tensor_bytes,
     max_total_tensor_bytes, max_rank, max_factor_elements) = _policy(policy)
    if sum(map(len, (changes, projections, slices, low_rank))) > max_entries:
        raise EditRejected("Patch instruction count exceeds host entry budget")
    tensor_hashes = patch_doc.get("tensor_sha256", {})
    if not isinstance(tensor_hashes, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in tensor_hashes.items()):
        raise EditRejected("tensor_sha256 must map tensor names to hashes")
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()) for value in tensor_hashes.values()):
        raise EditRejected("tensor_sha256 values must be hexadecimal SHA-256 digests")

    total_parameters = sum(ref.elements for ref in checkpoint.tensors.values())
    touched: set[tuple[str, ...]] = set()
    states: dict[str, tuple[bytearray, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    delta_max = 0.0
    loaded_tensor_bytes = 0

    def state_for(name: Any) -> tuple[bytearray, np.ndarray]:
        nonlocal loaded_tensor_bytes
        if not isinstance(name, str) or name not in allowed or name not in checkpoint.tensors:
            raise EditRejected(f"Tensor not allowlisted or absent: {name}")
        if name not in states:
            ref = checkpoint.tensors[name]
            if ref.dtype not in _FLOAT_DTYPES:
                raise EditRejected(f"Tensor is not a supported floating dtype: {name} ({ref.dtype})")
            if name not in tensor_hashes:
                raise EditRejected(f"Patch is missing a base tensor hash: {name}")
            if loaded_tensor_bytes + ref.nbytes > max_total_tensor_bytes:
                raise EditRejected("Patch exceeds total tensor memory budget")
            raw = checkpoint.read_raw(name, max_bytes=max_tensor_bytes)
            expected_hash = tensor_hashes.get(name)
            if expected_hash is not None and hashlib.sha256(raw).hexdigest() != expected_hash:
                raise EditRejected(f"Base tensor hash mismatch: {name}")
            loaded_tensor_bytes += ref.nbytes
            states[name] = (bytearray(raw), _to_float64(raw, ref.dtype).reshape(ref.shape).copy())
        return states[name]

    def replace(name: Any, index: tuple[int, ...], target: float, op: str, *, record: bool = True) -> None:
        nonlocal delta_max
        ref = checkpoint.tensors.get(name) if isinstance(name, str) else None
        if ref is None or len(index) != len(ref.shape) or any(i < 0 or i >= ref.shape[axis] for axis, i in enumerate(index)):
            raise EditRejected("Invalid tensor coordinate")
        coordinate = (name, *(str(i) for i in index))
        if coordinate in touched:
            raise EditRejected("Duplicate coordinate edit")
        if len(touched) + 1 > max_entries or (len(touched) + 1) / total_parameters > max_fraction:
            raise EditRejected("Edit entries or parameter fraction budget exceeded")
        _, values = state_for(name)
        before = float(values[index])
        if not _finite_number(target):
            raise EditRejected("Nonfinite patch value")
        rounded_raw = _from_float64(np.asarray([target], dtype=np.float64), ref.dtype)
        after = float(_to_float64(rounded_raw, ref.dtype)[0])
        delta = abs(after - before)
        if not math.isfinite(after) or delta > max_abs_delta:
            raise EditRejected("Nonfinite or excessive rounded delta")
        touched.add(coordinate)
        values[index] = after
        delta_max = max(delta_max, delta)
        if record:
            records.append({"tensor": name, "index": list(index), "op": op, "before": before, "after": after})

    for change in changes:
        if not isinstance(change, dict) or set(change) != {"tensor", "index", "op", "value"}:
            raise EditRejected("Each sparse change must have tensor/index/op/value")
        name, index, op, value = (change[key] for key in ("tensor", "index", "op", "value"))
        if not isinstance(index, list) or not all(type(v) is int for v in index):
            raise EditRejected("Invalid sparse index")
        if not isinstance(op, str) or op not in {"set", "add"} or not _finite_number(value):
            raise EditRejected("Invalid sparse operation or value")
        ref = checkpoint.tensors.get(name) if isinstance(name, str) else None
        if ref is None or len(index) != len(ref.shape) or any(i < 0 or i >= ref.shape[axis] for axis, i in enumerate(index)):
            raise EditRejected("Invalid tensor coordinate")
        _, values = state_for(name)
        target = float(value) if op == "set" else float(values[tuple(index)]) + float(value)
        replace(name, tuple(index), target, op)

    for slice_edit in slices:
        required = {"tensor", "axis", "indices", "op", "values"}
        if not isinstance(slice_edit, dict) or set(slice_edit) != required:
            raise EditRejected("Slice edit must contain tensor/axis/indices/op/values")
        name, axis, indices, op, vectors = (slice_edit[key] for key in ("tensor", "axis", "indices", "op", "values"))
        ref = checkpoint.tensors.get(name) if isinstance(name, str) else None
        if ref is None or len(ref.shape) != 2 or ref.dtype not in _FLOAT_DTYPES:
            raise EditRejected("Slice target must be an allowlisted floating matrix")
        if axis not in {"rows", "columns"} or op not in {"set", "add"}:
            raise EditRejected("Slice axis or operation is invalid")
        if (not isinstance(indices, list) or not indices or not all(type(i) is int for i in indices)
                or len(set(indices)) != len(indices)):
            raise EditRejected("Slice indices must be nonempty, unique integers")
        axis_size, vector_size = (ref.shape[0], ref.shape[1]) if axis == "rows" else (ref.shape[1], ref.shape[0])
        if any(i < 0 or i >= axis_size for i in indices):
            raise EditRejected("Slice index is outside the tensor dimensions")
        if len(indices) * vector_size + len(touched) > max_entries:
            raise EditRejected("Slice edit exceeds host entry budget")
        if (not isinstance(vectors, list) or len(vectors) != len(indices)
                or any(not isinstance(row, list) or len(row) != vector_size for row in vectors)
                or any(not _finite_number(value)
                       for row in vectors for value in row)):
            raise EditRejected("Slice values must provide one finite vector per selected row or column")
        _, matrix = state_for(name)
        for index, vector in zip(indices, vectors):
            for offset, value in enumerate(vector):
                coordinate = (index, offset) if axis == "rows" else (offset, index)
                target = float(value) if op == "set" else float(matrix[coordinate]) + float(value)
                replace(name, coordinate, target, f"slice_{op}", record=False)
        records.append({"tensor": name, "op": f"slice_{op}", "axis": axis, "indices": indices,
                        "parameter_updates": len(indices) * vector_size,
                        "values_sha256": _canonical_sha256(vectors)})

    for projection in projections:
        if not isinstance(projection, dict) or set(projection) != {"tensor", "op", "columns", "direction", "strength"}:
            raise EditRejected("Projection must contain tensor/op/columns/direction/strength")
        name = projection["tensor"]
        ref = checkpoint.tensors.get(name) if isinstance(name, str) else None
        if ref is None or len(ref.shape) != 2 or ref.dtype not in _FLOAT_DTYPES:
            raise EditRejected("Projection target must be an allowlisted floating matrix")
        if projection["op"] != "remove_column_direction":
            raise EditRejected("Only remove_column_direction is supported")
        columns = projection["columns"]
        direction = projection["direction"]
        strength = projection["strength"]
        if (not isinstance(columns, list) or not columns or not all(type(col) is int for col in columns)
                or len(set(columns)) != len(columns) or any(col < 0 or col >= ref.shape[1] for col in columns)):
            raise EditRejected("Invalid or duplicate projection columns")
        if (not isinstance(direction, list) or len(direction) != ref.shape[0]
                or not all(_finite_number(v) for v in direction)):
            raise EditRejected("Projection direction length or values are invalid")
        if not _finite_number(strength) or not 0 < strength <= 2:
            raise EditRejected("Projection strength must be in (0, 2]")
        _, matrix = state_for(name)
        u = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(u))
        if not math.isfinite(norm) or norm < 1e-12:
            raise EditRejected("Projection direction has zero or invalid norm")
        u /= norm
        for col in columns:
            old = matrix[:, col].copy()
            new = old - float(strength) * u * float(np.dot(u, old))
            for row, target in enumerate(new):
                replace(name, (row, col), float(target), "projection")

    for update in low_rank:
        if not isinstance(update, dict) or set(update) != {"tensor", "u", "v", "scale"}:
            raise EditRejected("Low-rank edit must contain tensor/u/v/scale")
        name = update["tensor"]
        ref = checkpoint.tensors.get(name) if isinstance(name, str) else None
        if ref is None or len(ref.shape) != 2 or ref.dtype not in _FLOAT_DTYPES:
            raise EditRejected("Low-rank target must be an allowlisted floating matrix")
        rows, columns = ref.shape
        factor_u, factor_v = update["u"], update["v"]
        if not isinstance(factor_u, list) or not factor_u or any(not isinstance(row, list) for row in factor_u):
            raise EditRejected("Low-rank U must be a nonempty nested list")
        rank = len(factor_u[0])
        if (not 1 <= rank <= max_rank or len(factor_u) != rows
                or any(len(row) != rank for row in factor_u)
                or not isinstance(factor_v, list) or len(factor_v) != rank
                or any(not isinstance(row, list) or len(row) != columns for row in factor_v)):
            raise EditRejected("Low-rank factor shapes or rank exceed host limits")
        factor_elements = rows * rank + rank * columns
        if factor_elements > max_factor_elements:
            raise EditRejected("Low-rank factors exceed host factor-element budget")
        factors = [value for matrix in (factor_u, factor_v) for row in matrix for value in row]
        if any(not _finite_number(value) for value in factors):
            raise EditRejected("Low-rank factors must contain finite numbers")
        scale = update["scale"]
        if not _finite_number(scale):
            raise EditRejected("Low-rank scale must be a finite number")
        if rows * columns + len(touched) > max_entries:
            raise EditRejected("Low-rank edit exceeds host entry budget")
        _, matrix = state_for(name)
        dense_delta = (np.asarray(factor_u, dtype=np.float64) @ np.asarray(factor_v, dtype=np.float64)) * float(scale)
        if dense_delta.shape != (rows, columns) or not np.isfinite(dense_delta).all():
            raise EditRejected("Low-rank product is malformed or nonfinite")
        for row in range(rows):
            for column in range(columns):
                coordinate = (row, column)
                replace(name, coordinate, float(matrix[coordinate] + dense_delta[coordinate]),
                        "low_rank_add", record=False)
        records.append({"tensor": name, "op": "low_rank_add", "rank": rank,
                        "scale": float(scale), "parameter_updates": rows * columns,
                        "factor_sha256": _canonical_sha256({"u": factor_u, "v": factor_v, "scale": scale})})

    if not states:
        raise EditRejected("Patch did not affect any tensor")
    changed_entries = 0
    output_raw: dict[str, bytes] = {}
    for name, (raw_before, values) in states.items():
        ref = checkpoint.tensors[name]
        raw_after = _from_float64(values.ravel(), ref.dtype)
        original = checkpoint.read_raw(name, max_bytes=max_tensor_bytes)
        a = np.frombuffer(original, dtype=np.uint8)
        b = np.frombuffer(raw_after, dtype=np.uint8)
        item_bytes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}[ref.dtype]
        if len(a) != len(b) or len(a) % item_bytes:
            raise EditRejected("Internal tensor byte length mismatch")
        before_values = _to_float64(original, ref.dtype)
        after_values = _to_float64(raw_after, ref.dtype)
        changed_entries += int(np.count_nonzero(before_values != after_values))
        output_raw[name] = raw_after

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise EditRejected(f"Candidate path already exists: {output}")
    stage_parent = Path(tempfile.mkdtemp(prefix=".weightbridge-stage-", dir=output.parent))
    try:
        staged = _copy_checkpoint(checkpoint, stage_parent)
        staged_checkpoint = load_checkpoint(staged)
        for name, raw in output_raw.items():
            ref = staged_checkpoint.tensors[name]
            target_path = ref.path
            with target_path.open("r+b") as stream:
                stream.seek(ref.data_start + ref.start)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        verified = load_checkpoint(staged)
        tensor_receipts = {}
        for name, expected_raw in output_raw.items():
            observed = verified.read_raw(name, max_bytes=max_tensor_bytes)
            if observed != expected_raw:
                raise EditRejected(f"Candidate tensor verification failed: {name}")
            tensor_receipts[name] = {
                "base_sha256": checkpoint.tensor_sha256(name),
                "candidate_sha256": hashlib.sha256(observed).hexdigest(),
                "shape": list(checkpoint.tensors[name].shape),
                "dtype": checkpoint.tensors[name].dtype,
                "bytes_changed": int(np.count_nonzero(
                    np.frombuffer(checkpoint.read_raw(name, max_bytes=max_tensor_bytes), dtype=np.uint8)
                    != np.frombuffer(observed, dtype=np.uint8)
                )),
            }
        candidate_hash = verified.checkpoint_sha256()
        receipt = {
            "schema": "weightbridge.receipt.v2",
            "status": "candidate_only_not_promoted",
            "base_sha256": base_hash,
            "candidate_sha256": candidate_hash,
            "patch_sha256": _canonical_sha256(patch_doc),
            "policy_sha256": _canonical_sha256(policy),
            "changed_entries": changed_entries,
            "touched_entries": len(touched),
            "total_parameters": total_parameters,
            "parameter_fraction_changed": changed_entries / total_parameters,
            "max_abs_delta": delta_max,
            "evidence_ids": patch_doc.get("evidence_ids", []),
            "tensor_receipts": tensor_receipts,
            "changes": records,
            "source_shards": [p.name for p in checkpoint.shards],
            "note": "Candidate is isolated and has not been promoted or deployed",
        }
        receipt_path = staged / ".weightbridge_receipt.json" if staged.is_dir() else stage_parent / f"{staged.name}.receipt.json"
        _write_new_json(receipt_path, receipt)
        if output.exists():
            raise EditRejected(f"Candidate path appeared during creation: {output}")
        if staged.is_dir():
            os.rename(staged, output)
        else:
            os.link(staged, output)
            sidecar = Path(str(output) + ".receipt.json")
            try:
                _write_new_json(sidecar, receipt)
            except Exception:
                output.unlink(missing_ok=True)
                raise
        return receipt
    except OSError as exc:
        if isinstance(exc, FileExistsError):
            raise EditRejected(f"Candidate path already exists: {output}") from exc
        raise
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)
