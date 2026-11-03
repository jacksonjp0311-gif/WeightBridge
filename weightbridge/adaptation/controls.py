"""Host-only matched control for one activation-ridge column slice."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..core import EditRejected
from ..inspection.safetensors import load_checkpoint
from ..patches.huggingface import _finite_number, _patch_document, _policy, _to_float64, apply_safetensors_patch


def _rank_one_slice(patch: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    changes, projections, slices, low_rank = _patch_document(patch)
    if changes or projections or low_rank or len(slices) != 1:
        raise EditRejected("Matched control requires exactly one additive column slice")
    edit = slices[0]
    if (not isinstance(edit, dict) or set(edit) != {"tensor", "axis", "indices", "op", "values"}
            or edit["axis"] != "columns" or edit["op"] != "add"
            or not isinstance(edit["tensor"], str) or not edit["tensor"]):
        raise EditRejected("Matched control requires exactly one additive column slice")
    indices = edit["indices"]
    vectors = edit["values"]
    if (not isinstance(indices, list) or not indices
            or not all(type(index) is int and index >= 0 for index in indices)
            or len(set(indices)) != len(indices)
            or not isinstance(vectors, list) or len(vectors) != len(indices)
            or not all(isinstance(vector, list) and vector for vector in vectors)
            or any(len(vector) != len(vectors[0]) or not all(map(_finite_number, vector)) for vector in vectors)):
        raise EditRejected("Matched control requires distinct columns and equally sized finite coefficient vectors")
    values = np.asarray(vectors, dtype=np.float64)
    row, column = np.unravel_index(np.abs(values).argmax(), values.shape)
    pivot = values[row, column]
    if pivot == 0 or not np.allclose(values / pivot, np.outer(values[:, column] / pivot, values[row] / pivot),
                                    rtol=1e-12, atol=1e-14):
        raise EditRejected("Matched control requires a nonzero rank-one proposal")
    return edit, values


def _permuted_patch(patch, edit, values, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Matched control seed must be a nonnegative integer")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(values.shape[1])
    control = deepcopy({key: value for key, value in patch.items() if key != "slices"})
    control["slices"] = [{**edit, "indices": list(edit["indices"]), "values": values[:, permutation].tolist()}]
    control["reason"] = "Matched control with a single seeded permutation of the proposal's output rows."
    return control, permutation.tolist()


def matched_random_patch(patch: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Permute output rows jointly across selected columns with a frozen seed.

    The permutation preserves the requested norm, rank, and number of
    coordinates. The realized rounded changes depend on the base weights.
    """
    edit, values = _rank_one_slice(patch)
    return _permuted_patch(patch, edit, values, seed)[0]


def _frobenius_norm(values: np.ndarray) -> float:
    maximum = float(np.abs(values).max())
    norm = float(np.linalg.norm(values / maximum)) * maximum if maximum else 0.0
    if not math.isfinite(norm):
        raise EditRejected("Matched control norm exceeds report precision")
    return norm


def build_matched_control(
    base: str | Path,
    patch_path: str | Path,
    policy: dict[str, Any],
    output: str | Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Replay the proposal, apply its matched control, and report rounded deltas.

    Both edits pass the regular host patcher. The proposal replay uses temporary
    storage, and the control is written to the caller's new checkpoint path.
    The report measures weight changes only. It contains no evaluation scores.
    """
    with Path(patch_path).open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Proposal exceeds the 8 MiB input limit")
    patch = json.loads(raw)
    edit, values = _rank_one_slice(patch)
    control, permutation = _permuted_patch(patch, edit, values, seed)
    requested_norm = _frobenius_norm(values)
    max_tensor_bytes = _policy(policy)[4]
    checkpoint = load_checkpoint(base)
    name, columns = edit["tensor"], edit["indices"]
    ref = checkpoint.tensors.get(name)
    if (ref is None or len(ref.shape) != 2 or any(i >= ref.shape[1] for i in columns)
            or ref.shape[0] != values.shape[1]):
        raise EditRejected("Matched proposal slice does not fit the base tensor")

    def selected_columns(model):
        data = model.read_raw(name, max_bytes=max_tensor_bytes)
        return _to_float64(data, ref.dtype).reshape(ref.shape)[:, columns]

    def rounded(path, receipt, original):
        delta = selected_columns(load_checkpoint(path)) - original
        changed = int(np.count_nonzero(delta))
        if changed != receipt["changed_entries"]:
            raise EditRejected("Rounded slice changes differ from the patch receipt")
        return {"patch_sha256": receipt["patch_sha256"], "candidate_sha256": receipt["candidate_sha256"],
                "changed_entries": changed, "frobenius_norm": _frobenius_norm(delta)}

    with tempfile.TemporaryDirectory(prefix="weightbridge-control-") as work:
        replay = Path(work) / ("proposal.safetensors" if checkpoint.root.is_file() else "proposal")
        proposal_receipt = apply_safetensors_patch(base, patch, policy, replay)
        original = selected_columns(checkpoint)
        proposal_result = rounded(replay, proposal_receipt, original)
    control_receipt = apply_safetensors_patch(base, control, policy, output)
    control_result = rounded(output, control_receipt, original)
    return {
        "schema": "weightbridge.adaptation_control.v1",
        "method": "row_permutation",
        "seed": seed,
        "row_permutation": permutation,
        "base_sha256": proposal_receipt["base_sha256"],
        "tensor": name,
        "tensor_sha256": proposal_receipt["tensor_receipts"][name]["base_sha256"],
        "requested": {"columns": columns, "rank": 1, "frobenius_norm": requested_norm,
                      "parameter_count": values.size, "nonzero_coefficients": int(np.count_nonzero(values))},
        "proposal": proposal_result,
        "control": control_result,
        "same_candidate_as_proposal": control_result["candidate_sha256"] == proposal_result["candidate_sha256"],
        "control_minus_proposal": {key: control_result[key] - proposal_result[key]
                                   for key in ("changed_entries", "frobenius_norm")},
    }
