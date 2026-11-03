import copy
import json
from pathlib import Path

import numpy as np
import pytest

from weightbridge.core import EditRejected, apply_patch, file_hash, inspect_checkpoint, load_tensors, save_tensors_new
from weightbridge.demo import compare_demo, run_demo


def fixture_inputs(tmp_path):
    base = tmp_path / "base.npz"
    save_tensors_new(base, {"W": np.eye(2, dtype=np.float32), "b": np.array([0, 1.5], np.float32)})
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": file_hash(base),
             "evidence_ids": ["test-event"],
             "changes": [{"tensor": "b", "index": [1], "op": "set", "value": 0.0}]}
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["b"],
              "max_entries": 1, "max_fraction": 0.2, "max_abs_delta": 2.0}
    return base, patch, policy


def test_demo_improves_synthetic_holdout(tmp_path):
    run = run_demo(tmp_path / "case")
    assert run["evaluation"]["candidate_accuracy"] > run["evaluation"]["baseline_accuracy"] + 0.10
    assert run["evaluation"]["held_gate"] is False
    assert run["receipt"]["status"] == "candidate_only_not_promoted"


def test_original_is_immutable_and_receipted(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    original_hash = file_hash(base)
    out = tmp_path / "candidate.npz"
    r = apply_patch(base, patch, policy, out)
    assert file_hash(base) == original_hash
    assert file_hash(out) == r["candidate_sha256"]
    assert (tmp_path / "candidate.npz.receipt.json").exists()
    assert float(load_tensors(base)["b"][1]) == 1.5
    assert float(load_tensors(out)["b"][1]) == 0.0


def test_base_hash_mismatch_rejected(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    patch["base_sha256"] = "a" * 64
    with pytest.raises(EditRejected, match="mismatch"):
        apply_patch(base, patch, policy, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_disallowed_tensor_rejected(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    patch["changes"][0]["tensor"] = "W"
    patch["changes"][0]["index"] = [1, 1]
    with pytest.raises(EditRejected, match="allowlisted"):
        apply_patch(base, patch, policy, tmp_path / "bad.npz")


def test_edit_budget_and_delta_rejected(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    policy["max_fraction"] = 0.1
    with pytest.raises(EditRejected, match="fraction"):
        apply_patch(base, patch, policy, tmp_path / "bad.npz")
    policy["max_fraction"] = 0.2
    patch["changes"][0]["value"] = -100
    with pytest.raises(EditRejected, match="delta"):
        apply_patch(base, patch, policy, tmp_path / "bad2.npz")


def test_duplicate_and_nonfinite_rejected(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    policy["max_entries"] = 2
    policy["max_fraction"] = 0.4
    patch["changes"].append(copy.deepcopy(patch["changes"][0]))
    with pytest.raises(EditRejected, match="Duplicate"):
        apply_patch(base, patch, policy, tmp_path / "bad.npz")
    patch["changes"] = patch["changes"][:1]
    patch["changes"][0]["value"] = float("nan")
    with pytest.raises(EditRejected, match="nonfinite"):
        apply_patch(base, patch, policy, tmp_path / "bad2.npz")


def test_cannot_overwrite_base_or_candidate(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    with pytest.raises(EditRejected, match="In-place"):
        apply_patch(base, patch, policy, base)
    out = tmp_path / "candidate.npz"
    apply_patch(base, patch, policy, out)
    with pytest.raises(EditRejected, match="exists"):
        apply_patch(base, patch, policy, out)


def test_tensor_inspection(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    info = inspect_checkpoint(base)
    assert info["tensor_count"] == 2
    assert info["total_parameters"] == 6
    assert len(info["tensors"]) == 2


def test_safetensors_small_checkpoint(tmp_path):
    pytest.importorskip("safetensors")
    base, patch, policy = fixture_inputs(tmp_path)
    st = tmp_path / "base.safetensors"
    save_tensors_new(st, load_tensors(base))
    patch["base_sha256"] = file_hash(st)
    out = tmp_path / "candidate.safetensors"
    apply_patch(st, patch, policy, out)
    assert float(load_tensors(out)["b"][1]) == 0.0


def test_directional_projection_removes_targeted_component(tmp_path):
    from weightbridge.core import apply_patch
    base = tmp_path / "matrix.npz"
    mat = np.zeros((10, 10), dtype=np.float32)
    mat[0, 3] = 2.0
    mat[1, 3] = 1.0
    mat[0, 4] = 7.0
    save_tensors_new(base, {"M": mat})
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["M"],
              "max_entries": 10, "max_fraction": 0.1, "max_abs_delta": 2.1}
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": file_hash(base),
             "projections": [{"tensor": "M", "op": "remove_column_direction", "columns": [3],
                              "direction": [1.0] + [0.0] * 9, "strength": 1.0}]}
    out = tmp_path / "projected.npz"
    receipt = apply_patch(base, patch, policy, out)
    observed = load_tensors(out)["M"]
    assert observed[0, 3] == 0.0
    assert observed[1, 3] == 1.0
    assert observed[0, 4] == 7.0
    assert receipt["changed_entries"] == 10  # patch budget counts all touched entries


def test_projection_budget_rejected(tmp_path):
    base = tmp_path / "matrix.npz"
    mat = np.ones((10, 10), dtype=np.float32)
    save_tensors_new(base, {"M": mat})
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["M"],
              "max_entries": 5, "max_fraction": 0.1, "max_abs_delta": 2.1}
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": file_hash(base),
             "projections": [{"tensor": "M", "op": "remove_column_direction", "columns": [3],
                              "direction": [1.0] + [0.0] * 9, "strength": 1.0}]}
    with pytest.raises(EditRejected, match="budget"):
        apply_patch(base, patch, policy, tmp_path / "reject.npz")


@pytest.mark.parametrize("key,value", [("max_entries", True), ("max_fraction", True),
                                       ("max_abs_delta", True)])
def test_policy_rejects_boolean_numeric_limits(tmp_path, key, value):
    base, patch, policy = fixture_inputs(tmp_path)
    policy[key] = value
    with pytest.raises(EditRejected, match="Invalid"):
        apply_patch(base, patch, policy, tmp_path / "bad.npz")


def test_patch_rejects_unhashable_operation_and_duplicate_allowlist(tmp_path):
    base, patch, policy = fixture_inputs(tmp_path)
    patch["changes"][0]["op"] = []
    with pytest.raises(EditRejected, match="operation"):
        apply_patch(base, patch, policy, tmp_path / "bad-op.npz")
    patch["changes"][0]["op"] = "set"
    policy["allow_tensors"] = ["b", "b"]
    with pytest.raises(EditRejected, match="allowlist"):
        apply_patch(base, patch, policy, tmp_path / "bad-policy.npz")
