import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiis_tool_transfer import merge
from weightbridge.core import file_hash
from weightbridge.inspection.safetensors import load_checkpoint


def test_merge_authorization_binds_actual_sequential_parent(tmp_path):
    protocol = {"checkpoint_sha256": "b"*64, "task_manifest_sha256": "c"*64, "source_sha256": {}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    targets = [SimpleNamespace(tensor_name="model.layers.0.mlp.down_proj.weight", shape=(4, 4))]
    policy = merge.merge_authorization(path, "a"*64, targets)
    assert policy["base_checkpoint_sha256"] == "b"*64
    merge.validate_merge_authorization(path, policy, "a"*64, "b"*64)
    with pytest.raises(ValueError, match="actual parent"):
        merge.validate_merge_authorization(path, policy, "a"*64, "d"*64)
    with pytest.raises(ValueError, match="bound"):
        merge.validate_merge_authorization(path, policy, "e"*64, "b"*64)


def test_cumulative_scan_uses_stored_delta_and_exact_hashes(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    original, candidate = tmp_path / "original.safetensors", tmp_path / "candidate.safetensors"
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.bfloat16)
    y = torch.tensor([1., 2.03125, 3., 4.0625], dtype=torch.bfloat16)
    st.save_file({"weight": x, "frozen": torch.tensor([1.], dtype=torch.float32)}, str(original))
    st.save_file({"weight": y, "frozen": torch.tensor([1.], dtype=torch.float32)}, str(candidate))
    a, b = load_checkpoint(original).checkpoint_sha256(), load_checkpoint(candidate).checkpoint_sha256()
    result = merge.checkpoint_difference(original, candidate, expected_original=a, expected_candidate=b,
                                         chunk_elements=1)
    assert result["changed_entries"] == 2
    assert result["stored_update_norm"] == pytest.approx(float(torch.linalg.vector_norm(y.float()-x.float())))
    assert result["original_parameter_norm"] == pytest.approx(np.sqrt(31))
    assert result["changed_tensors"][0]["tensor"] == "weight"
    assert file_hash(original) == a
    with pytest.raises(ValueError, match="identity"):
        merge.checkpoint_difference(original, candidate, expected_original="0"*64, expected_candidate=b)


def test_two_stage_provenance_rejects_single_round_delta_sum():
    first = {"base_sha256": "a"*64, "candidate_sha256": "b"*64, "base_unchanged": True,
             "lineage_accepted": False, "production_deployment": False}
    second = {**first, "base_sha256": "b"*64, "candidate_sha256": "c"*64}
    assert merge.verify_two_stage_provenance("a"*64, "b"*64, "c"*64, first, second)["passed"]
    collapsed = {**second, "base_sha256": "a"*64}
    assert not merge.verify_two_stage_provenance("a"*64, "b"*64, "c"*64, first, collapsed)["passed"]
