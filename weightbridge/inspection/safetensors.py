"""Bounded, read-only inspection for sharded Hugging Face safetensors models.

The index and safetensors headers are read eagerly; tensor payloads are read only
when requested. Hashing streams the file so model size does not imply RAM use.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np


class SafetensorsError(ValueError):
    """Invalid or unsupported safetensors checkpoint metadata."""


_MAX_HEADER_BYTES = 100 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1, "F8_E5M2": 1, "F8_E5M2FNUZ": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
_NUMPY_DTYPES = {
    "BOOL": np.dtype("?"), "I8": np.dtype("i1"), "U8": np.dtype("u1"),
    "I16": np.dtype("<i2"), "U16": np.dtype("<u2"),
    "I32": np.dtype("<i4"), "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"), "U64": np.dtype("<u8"),
    "F16": np.dtype("<f2"), "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
}
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_IGNORED_DIRS = {".git", ".cache", "__pycache__"}
_RECEIPT_NAME = ".weightbridge_receipt.json"


@dataclass(frozen=True)
class TensorRef:
    """Location and validated metadata for one tensor payload."""

    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    data_start: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Checkpoint:
    """Validated file/header map for an unsharded or indexed safetensors model."""

    root: Path
    tensors: dict[str, TensorRef]
    shards: tuple[Path, ...]
    assets: tuple[Path, ...]
    index_path: Path | None
    config: dict[str, Any]
    tokenizer_config: dict[str, Any]

    def read_raw(self, name: str, *, max_bytes: int | None = 64 * 1024 * 1024) -> bytes:
        ref = self.tensors[name]
        if max_bytes is not None and ref.nbytes > max_bytes:
            raise SafetensorsError(
                f"Tensor {name} is {ref.nbytes} bytes; requested read limit is {max_bytes}"
            )
        with ref.path.open("rb") as stream:
            stream.seek(ref.data_start + ref.start)
            payload = stream.read(ref.nbytes)
        if len(payload) != ref.nbytes:
            raise SafetensorsError(f"Truncated tensor payload: {name}")
        return payload

    def tensor_sha256(self, name: str) -> str:
        ref = self.tensors[name]
        digest = hashlib.sha256()
        with ref.path.open("rb") as stream:
            stream.seek(ref.data_start + ref.start)
            remaining = ref.nbytes
            while remaining:
                chunk = stream.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise SafetensorsError(f"Truncated tensor payload: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()

    def checkpoint_sha256(self) -> str:
        """Hash file checkpoints directly; hash directory assets by relative path."""
        if len(self.shards) == 1 and self.root.is_file() and self.root == self.shards[0]:
            return _file_sha256(self.root)
        digest = hashlib.sha256()
        for path in sorted(self.assets, key=lambda p: p.relative_to(self.root).as_posix()):
            rel = path.relative_to(self.root).as_posix().encode("utf-8")
            digest.update(rel + b"\0" + bytes.fromhex(_file_sha256(path)))
        return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafetensorsError(f"Missing safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > _MAX_HEADER_BYTES or 8 + header_size > size:
            raise SafetensorsError(f"Invalid safetensors header length: {path}")
        raw = stream.read(header_size)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"Invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"Safetensors header must be a mapping: {path}")
    return header, 8 + header_size


def _parse_shard(path: Path) -> dict[str, TensorRef]:
    header, data_start = _read_header(path)
    file_payload_size = path.stat().st_size - data_start
    refs: dict[str, TensorRef] = {}
    intervals: list[tuple[int, int, str]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(metadata, dict):
            raise SafetensorsError(f"Invalid tensor entry in {path}")
        dtype = metadata.get("dtype")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise SafetensorsError(f"Invalid tensor metadata for {name}")
        if not all(type(d) is int and d >= 0 for d in shape):
            raise SafetensorsError(f"Invalid tensor shape for {name}")
        if not all(type(v) is int and v >= 0 for v in offsets):
            raise SafetensorsError(f"Invalid data offsets for {name}")
        start, end = offsets
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end < start or end - start != expected or end > file_payload_size:
            raise SafetensorsError(f"Tensor payload bounds do not match metadata for {name}")
        refs[name] = TensorRef(name, path, dtype, tuple(shape), start, end, data_start)
        if end > start:
            intervals.append((start, end, name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] > current[0]:
            raise SafetensorsError(f"Overlapping tensor payloads in {path}: {previous[2]}, {current[2]}")
    return refs


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"Invalid JSON asset: {path.name}") from exc
    return value if isinstance(value, dict) else {}


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read headers only and validate the tensor-to-shard map."""
    path = Path(path)
    if path.is_symlink():
        raise SafetensorsError("Checkpoint symlinks are not accepted")
    path = path.resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise SafetensorsError("Checkpoint file must end in .safetensors")
        root = path
        shard_paths = (path,)
        index_path = None
        assets = (path,)
        config = {}
        tokenizer_config = {}
    elif path.is_dir():
        root = path
        for child in root.rglob("*"):
            if child.is_symlink():
                raise SafetensorsError(f"Symlinks are not accepted in model directories: {child}")
        indexes = list(root.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            raise SafetensorsError("Multiple safetensors index files found")
        index_path = indexes[0] if indexes else None
        if index_path:
            index = _load_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise SafetensorsError("Safetensors index requires a nonempty weight_map")
            shard_names = set()
            normalized_map: dict[str, str] = {}
            for tensor_name, shard_name in weight_map.items():
                if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                    raise SafetensorsError("Invalid safetensors weight_map entry")
                shard = (root / shard_name).resolve()
                try:
                    shard.relative_to(root)
                except ValueError as exc:
                    raise SafetensorsError("Shard path escapes model directory") from exc
                if not shard.is_file() or shard.suffix != ".safetensors":
                    raise SafetensorsError(f"Missing safetensors shard: {shard_name}")
                shard_names.add(shard)
                normalized_map[tensor_name] = shard.relative_to(root).as_posix()
            discovered_shards = {p.resolve() for p in root.rglob("*.safetensors")}
            if discovered_shards != shard_names:
                raise SafetensorsError("Indexed shard set does not match checkpoint safetensors files")
            shard_paths = tuple(sorted(shard_names))
        else:
            shard_paths = tuple(sorted(root.glob("*.safetensors")))
            if len(shard_paths) != 1:
                raise SafetensorsError("Multiple shards require a safetensors index file")
            if not shard_paths:
                raise SafetensorsError("No safetensors checkpoint found")
        config = _load_json(root / "config.json")
        tokenizer_config = _load_json(root / "tokenizer_config.json")
        assets = tuple(
            sorted(
                (p for p in root.rglob("*")
                 if p.is_file() and p.name != _RECEIPT_NAME
                 and not any(part in _IGNORED_DIRS for part in p.relative_to(root).parts)),
                key=lambda p: p.relative_to(root).as_posix(),
            )
        )
    else:
        raise SafetensorsError(f"Checkpoint path does not exist: {path}")

    tensors: dict[str, TensorRef] = {}
    for shard in shard_paths:
        shard_map = _parse_shard(shard)
        overlap = tensors.keys() & shard_map.keys()
        if overlap:
            raise SafetensorsError(f"Duplicate tensor names across shards: {sorted(overlap)[:3]}")
        tensors.update(shard_map)
    if not tensors:
        raise SafetensorsError("No tensors found")

    if path.is_dir() and index_path:
        expected = _load_json(index_path)["weight_map"]
        if set(expected) != set(tensors):
            raise SafetensorsError("Index tensor names do not exactly match checkpoint tensors")
        for name, shard_name in expected.items():
            ref = tensors.get(name)
            if ref is None or ref.path.relative_to(root).as_posix() != normalized_map[name]:
                raise SafetensorsError(f"Index does not match shard tensor: {name}")
    return Checkpoint(root, tensors, shard_paths, assets, index_path, config, tokenizer_config)


def _decode_float(raw: bytes, dtype: str) -> np.ndarray | None:
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (bits << 16).view(np.float32)
    numpy_dtype = _NUMPY_DTYPES.get(dtype)
    if numpy_dtype is None or not dtype.startswith("F"):
        return None
    return np.frombuffer(raw, dtype=numpy_dtype).astype(np.float64, copy=False)


def _layer_fields(name: str) -> tuple[int | None, str]:
    match = _LAYER.search(name)
    layer = int(match.group(1)) if match else None
    if ".mlp." in name:
        component = "mlp"
    elif ".self_attn." in name or ".linear_attn." in name or ".attention." in name:
        component = "attention"
    elif "embed" in name:
        component = "embedding"
    elif "norm" in name:
        component = "normalization"
    elif "lm_head" in name:
        component = "output"
    else:
        component = "other"
    return layer, component


def summarize_tensor(
    checkpoint: Checkpoint,
    name: str,
    *,
    max_stats_bytes: int = 64 * 1024 * 1024,
    compute_svd: bool = False,
    max_svd_elements: int = 1_000_000,
) -> dict[str, Any]:
    ref = checkpoint.tensors[name]
    layer, component = _layer_fields(name)
    result: dict[str, Any] = {
        "name": name,
        "shape": list(ref.shape),
        "dtype": ref.dtype,
        "parameters": ref.elements,
        "bytes": ref.nbytes,
        "layer": layer,
        "component": component,
        "shard": ref.path.name,
        "sha256": checkpoint.tensor_sha256(name),
    }
    if ref.nbytes > max_stats_bytes:
        result["statistics"] = None
        result["statistics_omitted_reason"] = f"tensor exceeds {max_stats_bytes} byte statistics budget"
        return result
    raw = checkpoint.read_raw(name, max_bytes=max_stats_bytes)
    values = _decode_float(raw, ref.dtype)
    if values is None:
        result["statistics"] = None
        result["statistics_omitted_reason"] = "numeric summary is not implemented for this dtype"
        return result
    if not np.isfinite(values).all():
        raise SafetensorsError(f"Nonfinite values found in tensor: {name}")
    result["statistics"] = {
        "mean": float(values.mean()) if values.size else None,
        "std": float(values.std()) if values.size else None,
        "min": float(values.min()) if values.size else None,
        "max": float(values.max()) if values.size else None,
        "l2_norm": float(np.linalg.norm(values.astype(np.float64, copy=False))) if values.size else 0.0,
        "zero_fraction": float(np.count_nonzero(values == 0) / values.size) if values.size else 0.0,
    }
    if compute_svd and len(ref.shape) == 2:
        if ref.elements > max_svd_elements:
            result["svd"] = {"computed": False, "reason": "matrix exceeds SVD element budget"}
        else:
            matrix = values.reshape(ref.shape).astype(np.float64, copy=False)
            singular = np.linalg.svd(matrix, compute_uv=False)
            tolerance = singular.max(initial=0.0) * max(matrix.shape) * np.finfo(np.float64).eps
            result["svd"] = {
                "computed": True,
                "rank_estimate": int(np.count_nonzero(singular > tolerance)),
                "largest_singular_value": float(singular[0]) if singular.size else 0.0,
                "smallest_singular_value": float(singular[-1]) if singular.size else 0.0,
            }
    return result


def inspect_safetensors(
    path: str | Path,
    *,
    max_tensors: int = 128,
    max_stats_bytes: int = 64 * 1024 * 1024,
    compute_svd: bool = False,
    max_svd_elements: int = 1_000_000,
) -> dict[str, Any]:
    for label, value in (("max_tensors", max_tensors), ("max_stats_bytes", max_stats_bytes),
                         ("max_svd_elements", max_svd_elements)):
        if type(value) is not int or value < 0:
            raise SafetensorsError(f"{label} must be a nonnegative integer")
    if max_stats_bytes > 256 * 1024 * 1024 or max_svd_elements > 1_000_000:
        raise SafetensorsError("Requested inspection budget exceeds the hard host limit")
    if max_tensors == 0:
        raise SafetensorsError("max_tensors must be at least one")
    checkpoint = load_checkpoint(path)
    names = sorted(checkpoint.tensors)
    model_config = checkpoint.config
    text_config = model_config.get("text_config") if isinstance(model_config.get("text_config"), dict) else model_config
    tokenizer = checkpoint.tokenizer_config
    file_hashes = [
        {"path": p.relative_to(checkpoint.root).as_posix() if checkpoint.root.is_dir() else p.name,
         "sha256": _file_sha256(p), "bytes": p.stat().st_size}
        for p in checkpoint.assets
    ]
    model_sha256 = checkpoint.checkpoint_sha256()
    return {
        "schema": "weightbridge.inspect.v2",
        "source": str(checkpoint.root),
        "checkpoint_sha256": model_sha256,
        "model_sha256": model_sha256,
        "file_hashes": file_hashes,
        "model": {
            "model_type": model_config.get("model_type"),
            "architectures": model_config.get("architectures"),
            "source_id": model_config.get("_name_or_path"),
            "hidden_size": text_config.get("hidden_size"),
            "intermediate_size": text_config.get("intermediate_size"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "layer_types": text_config.get("layer_types"),
        },
        "tokenizer": {
            "tokenizer_class": tokenizer.get("tokenizer_class"),
            "vocab_size": tokenizer.get("vocab_size"),
            "bos_token": tokenizer.get("bos_token"),
            "eos_token": tokenizer.get("eos_token"),
            "chat_template_sha256": _file_sha256(checkpoint.root / "chat_template.jinja")
            if checkpoint.root.is_dir() and (checkpoint.root / "chat_template.jinja").is_file() else None,
        },
        "shards": [p.name for p in checkpoint.shards],
        "tensor_count": len(names),
        "total_parameters": sum(checkpoint.tensors[name].elements for name in names),
        "shown_tensors": min(len(names), max_tensors),
        "truncated": len(names) > max_tensors,
        "tensors": [
            summarize_tensor(checkpoint, name, max_stats_bytes=max_stats_bytes,
                             compute_svd=compute_svd, max_svd_elements=max_svd_elements)
            for name in names[:max_tensors]
        ],
    }
