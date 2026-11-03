import json
import math
import struct

import numpy as np
import pytest

from weightbridge.adaptation.controls import build_matched_control, matched_random_patch
from weightbridge.core import EditRejected
from weightbridge.inspection.safetensors import load_checkpoint


TENSOR = "model.layers.0.mlp.down_proj.weight"


def _proposal(values, *, base_hash="a" * 64, tensor_hash="b" * 64):
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": base_hash,
        "reason": "activation-guided repair",
        "evidence_ids": ["practice-only"],
        "tensor_sha256": {TENSOR: tensor_hash},
        "slices": [{"tensor": TENSOR, "axis": "columns", "indices": [0, 1],
                    "op": "add", "values": values}],
    }


def _bf16_file(path):
    weights = np.array([[1.0, 1.0], [64.0, 1.0]], dtype=np.float32)
    payload = (weights.view(np.uint32) >> 16).astype("<u2").tobytes()
    header = json.dumps({TENSOR: {
        "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)],
    }}).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_matched_patch_freezes_joint_permutation_and_bindings():
    patch = _proposal([[0.125, 0.5], [-0.25, -1.0]])
    control = matched_random_patch(patch, seed=3)
    assert control == matched_random_patch(patch, seed=3)
    assert control["base_sha256"] == patch["base_sha256"]
    assert control["tensor_sha256"] == patch["tensor_sha256"]
    assert control["evidence_ids"] == patch["evidence_ids"]
    assert control["slices"] == [{**patch["slices"][0], "values": [[0.5, 0.125], [-1.0, -0.25]]}]
    assert patch["slices"][0]["values"] == [[0.125, 0.5], [-0.25, -1.0]]


def test_matched_patch_rejects_unmatched_edits():
    with pytest.raises(EditRejected, match="rank-one"):
        matched_random_patch(_proposal([[1.0, 0.0], [0.0, 1.0]]), seed=3)
    patch = _proposal([[0.125, 0.5], [-0.25, -1.0]])
    patch["changes"] = [{"tensor": TENSOR, "index": [0, 0], "op": "set", "value": 1.0}]
    with pytest.raises(EditRejected, match="exactly one"):
        matched_random_patch(patch, seed=3)


def test_build_control_validates_policy_and_measures_bf16_rounding(tmp_path):
    base = tmp_path / "base.safetensors"
    _bf16_file(base)
    checkpoint = load_checkpoint(base)
    base_hash = checkpoint.checkpoint_sha256()
    tensor_hash = checkpoint.tensor_sha256(TENSOR)
    patch = _proposal([[0.001, 0.1], [0.001, 0.1]], base_hash=base_hash, tensor_hash=tensor_hash)
    patch_path = tmp_path / "proposal.json"
    patch_path.write_text(json.dumps(patch), encoding="utf-8")
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": [TENSOR],
              "max_entries": 4, "max_fraction": 1.0, "max_abs_delta": 1.0, "max_tensor_bytes": 1024}
    output = tmp_path / "control.safetensors"
    with pytest.raises(EditRejected, match="not allowlisted"):
        build_matched_control(base, patch_path, {**policy, "allow_tensors": ["other"]}, output, seed=3)
    assert not output.exists()

    report = build_matched_control(base, patch_path, policy, output, seed=3)
    assert report["base_sha256"] == load_checkpoint(base).checkpoint_sha256() == base_hash
    assert report["tensor_sha256"] == tensor_hash
    assert report["row_permutation"] == [1, 0]
    assert report["requested"] == {"columns": [0, 1], "rank": 1, "parameter_count": 4,
                                   "nonzero_coefficients": 4, "frobenius_norm": pytest.approx(math.sqrt(0.020002))}
    assert report["proposal"]["changed_entries"] == 1
    assert report["proposal"]["frobenius_norm"] == 0.1015625
    assert report["control"]["changed_entries"] == 2
    assert report["control"]["frobenius_norm"] == pytest.approx(math.sqrt(2) * 0.1015625)
    assert report["control_minus_proposal"]["changed_entries"] == 1
    assert report["same_candidate_as_proposal"] is False
    assert report["control"]["candidate_sha256"] == load_checkpoint(output).checkpoint_sha256()
    receipt = json.loads(output.with_suffix(".safetensors.receipt.json").read_text(encoding="utf-8"))
    assert report["control"]["patch_sha256"] == receipt["patch_sha256"]

    identity = build_matched_control(base, patch_path, policy, tmp_path / "identity.safetensors", seed=0)
    assert identity["row_permutation"] == [0, 1]
    assert identity["same_candidate_as_proposal"] is True
    assert identity["control_minus_proposal"] == {"changed_entries": 0, "frobenius_norm": 0.0}
