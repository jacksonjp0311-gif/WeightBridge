import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

pytest.importorskip("safetensors")

from weightbridge.core import EditRejected, inspect_checkpoint
from weightbridge.inspection.safetensors import SafetensorsError, load_checkpoint
from weightbridge.patches import apply_safetensors_patch


def make_sharded_model(folder: Path) -> Path:
    from safetensors.numpy import save_file

    folder.mkdir()
    first = folder / "model-00001-of-00002.safetensors"
    second = folder / "model-00002-of-00002.safetensors"
    save_file({"model.layers.0.mlp.down_proj.weight": np.array([[1, 2], [3, 4]], dtype=np.float32)}, str(first))
    save_file({"model.layers.1.self_attn.q_proj.weight": np.eye(2, dtype=np.float16)}, str(second))
    (folder / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 24},
        "weight_map": {
            "model.layers.0.mlp.down_proj.weight": first.name,
            "model.layers.1.self_attn.q_proj.weight": second.name,
        },
    }), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps({
        "model_type": "toy_transformer", "architectures": ["ToyForCausalLM"],
        "hidden_size": 2, "num_hidden_layers": 2,
    }), encoding="utf-8")
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "ToyTokenizer", "eos_token": "<eos>", "vocab_size": 4,
    }), encoding="utf-8")
    (folder / "tokenizer.json").write_text('{"model":{"vocab":{}}}', encoding="utf-8")
    (folder / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    return folder


def make_bf16_file(path: Path) -> Path:
    values = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
    bits = (values.view(np.uint32) >> 16).astype("<u2")
    header = {
        "model.layers.15.mlp.down_proj.weight": {
            "dtype": "BF16", "shape": [2, 2], "data_offsets": [0, int(bits.nbytes)],
        },
        "__metadata__": {"format": "pt"},
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        stream.write(bits.tobytes())
    return path


def policy(name: str, **overrides):
    result = {
        "schema": "weightbridge.policy.v1", "allow_tensors": [name],
        "max_entries": 8, "max_fraction": 1.0, "max_abs_delta": 10.0,
        "max_tensor_bytes": 1024 * 1024,
    }
    result.update(overrides)
    return result


def patch_for(base: Path, name: str, changes=None, projections=None, slices=None, low_rank=None):
    checkpoint = load_checkpoint(base)
    return {
        "schema": "weightbridge.patch.v1",
        "base_sha256": checkpoint.checkpoint_sha256(),
        "reason": "test proposal",
        "evidence_ids": ["test-001"],
        "tensor_sha256": {name: checkpoint.tensor_sha256(name)},
        "changes": changes or [],
        "projections": projections or [],
        "slices": slices or [],
        "low_rank": low_rank or [],
    }


def test_sharded_inspection_is_lazy_and_identifies_model_and_tokenizer(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    checkpoint = load_checkpoint(base)
    info = inspect_checkpoint(base, max_tensors=10)
    assert info["schema"] == "weightbridge.inspect.v2"
    assert info["model"]["model_type"] == "toy_transformer"
    assert info["model"]["num_hidden_layers"] == 2
    assert info["tokenizer"]["tokenizer_class"] == "ToyTokenizer"
    assert len(info["tokenizer"]["chat_template_sha256"]) == 64
    assert info["tensor_count"] == 2
    assert info["total_parameters"] == 8
    assert info["tensors"][0]["layer"] == 0
    assert info["tensors"][0]["component"] == "mlp"
    assert checkpoint.shards == (base / "model-00001-of-00002.safetensors", base / "model-00002-of-00002.safetensors")
    small_budget = inspect_checkpoint(base, max_tensors=1)
    assert small_budget["shown_tensors"] == 1
    assert small_budget["truncated"] is True
    with pytest.raises(SafetensorsError, match="read limit"):
        checkpoint.read_raw("model.layers.0.mlp.down_proj.weight", max_bytes=1)
    bounded = __import__("weightbridge.inspection.safetensors", fromlist=["summarize_tensor"]).summarize_tensor(
        checkpoint, "model.layers.0.mlp.down_proj.weight", max_stats_bytes=1
    )
    assert bounded["statistics"] is None
    assert "budget" in bounded["statistics_omitted_reason"]


def test_small_matrix_svd_summary_is_bounded(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    checkpoint = load_checkpoint(base)
    from weightbridge.inspection.safetensors import summarize_tensor
    result = summarize_tensor(checkpoint, "model.layers.0.mlp.down_proj.weight", compute_svd=True)
    assert result["svd"]["computed"] is True
    assert result["svd"]["rank_estimate"] == 2
    skipped = summarize_tensor(checkpoint, "model.layers.0.mlp.down_proj.weight", compute_svd=True, max_svd_elements=2)
    assert skipped["svd"]["computed"] is False


def test_sparse_patch_sharded_checkpoint_creates_immutable_candidate(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    before_hash = load_checkpoint(base).checkpoint_sha256()
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 1], "op": "set", "value": 9.5}])
    out = tmp_path / "candidate"
    receipt = apply_safetensors_patch(base, patch, policy(name, max_entries=1), out)
    assert receipt["status"] == "candidate_only_not_promoted"
    assert receipt["touched_entries"] == 1
    assert receipt["changed_entries"] == 1
    assert receipt["parameter_fraction_changed"] == pytest.approx(1 / 8)
    assert load_checkpoint(base).checkpoint_sha256() == before_hash
    candidate = load_checkpoint(out)
    assert float(np.frombuffer(candidate.read_raw(name), dtype="<f4")[1]) == 9.5
    assert candidate.tensor_sha256("model.layers.1.self_attn.q_proj.weight") == load_checkpoint(base).tensor_sha256(
        "model.layers.1.self_attn.q_proj.weight"
    )
    assert (out / ".weightbridge_receipt.json").is_file()
    with pytest.raises(EditRejected, match="already exists"):
        apply_safetensors_patch(base, patch, policy(name, max_entries=1), out)


def test_bfloat16_sparse_patch_rounds_to_storage_precision(tmp_path):
    base = make_bf16_file(tmp_path / "base.safetensors")
    name = "model.layers.15.mlp.down_proj.weight"
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 1.5}])
    out = tmp_path / "candidate.safetensors"
    receipt = apply_safetensors_patch(base, patch, policy(name, max_entries=1), out)
    edited = load_checkpoint(out).read_raw(name)
    assert np.frombuffer(edited, dtype="<u2")[0] == np.uint16(0x3FC0)
    assert receipt["tensor_receipts"][name]["dtype"] == "BF16"
    assert hashlib.sha256(edited).hexdigest() == receipt["tensor_receipts"][name]["candidate_sha256"]
    assert load_checkpoint(base).read_raw(name) != edited


def test_projection_changes_only_selected_columns_and_counts_touched_entries(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, projections=[{
        "tensor": name, "op": "remove_column_direction", "columns": [0],
        "direction": [1.0, 0.0], "strength": 1.0,
    }])
    receipt = apply_safetensors_patch(base, patch, policy(name, max_entries=2), tmp_path / "candidate")
    assert receipt["touched_entries"] == 2
    assert receipt["changed_entries"] == 1
    matrix = np.frombuffer(load_checkpoint(tmp_path / "candidate").read_raw(name), dtype="<f4").reshape(2, 2)
    assert np.array_equal(matrix[:, 0], [0.0, 3.0])
    assert np.array_equal(matrix[:, 1], [2.0, 4.0])


def test_selected_row_slice_modification_is_bounded_and_receipted(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, slices=[{
        "tensor": name, "axis": "rows", "indices": [1], "op": "add", "values": [[2.0, -1.0]],
    }])
    receipt = apply_safetensors_patch(base, patch, policy(name, max_entries=2), tmp_path / "candidate")
    matrix = np.frombuffer(load_checkpoint(tmp_path / "candidate").read_raw(name), dtype="<f4").reshape(2, 2)
    assert np.array_equal(matrix, [[1.0, 2.0], [5.0, 3.0]])
    assert receipt["touched_entries"] == 2
    assert receipt["changed_entries"] == 2
    assert receipt["changes"][0]["op"] == "slice_add"


def test_rank_one_patch_is_applied_with_dtype_and_parameter_budgets(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, low_rank=[{
        "tensor": name, "u": [[1.0], [0.0]], "v": [[1.0, 2.0]], "scale": 1.0,
    }])
    receipt = apply_safetensors_patch(base, patch, policy(name, max_entries=4, max_rank=1), tmp_path / "candidate")
    matrix = np.frombuffer(load_checkpoint(tmp_path / "candidate").read_raw(name), dtype="<f4").reshape(2, 2)
    assert np.array_equal(matrix, [[2.0, 4.0], [3.0, 4.0]])
    assert receipt["touched_entries"] == 4
    assert receipt["changed_entries"] == 2
    assert receipt["changes"][0]["op"] == "low_rank_add"
    assert receipt["changes"][0]["rank"] == 1


def test_low_rank_patch_rejects_rank_shape_and_parameter_budget_violations(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    wrong_shape = patch_for(base, name, low_rank=[{
        "tensor": name, "u": [[1.0], [0.0]], "v": [[1.0]], "scale": 1.0,
    }])
    with pytest.raises(EditRejected, match="factor shapes"):
        apply_safetensors_patch(base, wrong_shape, policy(name), tmp_path / "bad-shape")
    too_large = patch_for(base, name, low_rank=[{
        "tensor": name, "u": [[1.0], [0.0]], "v": [[1.0, 2.0]], "scale": 1.0,
    }])
    with pytest.raises(EditRejected, match="entry budget"):
        apply_safetensors_patch(base, too_large, policy(name, max_entries=3), tmp_path / "over-budget")
    rank_limited = patch_for(base, name, low_rank=[{
        "tensor": name, "u": [[1.0, 1.0], [0.0, 0.0]], "v": [[1.0, 2.0], [1.0, 1.0]], "scale": 1.0,
    }])
    with pytest.raises(EditRejected, match="rank exceed"):
        apply_safetensors_patch(base, rank_limited, policy(name, max_entries=4, max_rank=1), tmp_path / "over-rank")


def test_unrepresentably_large_python_integers_are_rejected_as_patch_data(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, changes=[{
        "tensor": name, "index": [0, 0], "op": "set", "value": 10 ** 1000,
    }])
    with pytest.raises(EditRejected, match="Invalid sparse"):
        apply_safetensors_patch(base, patch, policy(name), tmp_path / "too-large")
    small_patch = patch_for(base, name, changes=[{
        "tensor": name, "index": [0, 0], "op": "set", "value": 1.0,
    }])
    with pytest.raises(EditRejected, match="max_abs_delta"):
        apply_safetensors_patch(base, small_patch, policy(name, max_abs_delta=10 ** 1000), tmp_path / "bad-policy")


@pytest.mark.parametrize("bad_policy", [
    {"max_entries": True}, {"max_fraction": True}, {"max_abs_delta": True},
])
def test_boolean_policy_numbers_are_rejected(tmp_path, bad_policy):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 1.0}])
    with pytest.raises(EditRejected, match="Invalid"):
        apply_safetensors_patch(base, patch, policy(name, **bad_policy), tmp_path / "candidate")


def test_rejects_wrong_base_tensor_hash_disallowed_names_and_duplicate_coordinates(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 2.0}])
    patch["tensor_sha256"][name] = "a" * 64
    with pytest.raises(EditRejected, match="tensor hash"):
        apply_safetensors_patch(base, patch, policy(name), tmp_path / "wrong-tensor")
    patch = patch_for(base, name, changes=[
        {"tensor": name, "index": [0, 0], "op": "set", "value": 2.0},
        {"tensor": name, "index": [0, 0], "op": "set", "value": 3.0},
    ])
    with pytest.raises(EditRejected, match="Duplicate"):
        apply_safetensors_patch(base, patch, policy(name), tmp_path / "duplicate")
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 2.0}])
    with pytest.raises(EditRejected, match="allowlisted"):
        apply_safetensors_patch(base, patch, policy("other.tensor"), tmp_path / "disallowed")


def test_rejects_projection_over_budget_and_patch_schema_extras(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, projections=[{
        "tensor": name, "op": "remove_column_direction", "columns": [0, 1],
        "direction": [1.0, 0.0], "strength": 1.0,
    }])
    with pytest.raises(EditRejected, match="budget"):
        apply_safetensors_patch(base, patch, policy(name, max_entries=2), tmp_path / "too-large")
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 2.0}])
    patch["unexpected"] = "ignored?"
    with pytest.raises(EditRejected, match="unsupported top-level"):
        apply_safetensors_patch(base, patch, policy(name), tmp_path / "extra")


def test_rejects_index_path_traversal_and_bad_offsets(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["model.layers.0.mlp.down_proj.weight"] = "../escape.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(SafetensorsError, match="escapes"):
        load_checkpoint(base)


def test_rejects_index_that_omits_or_misassigns_a_tensor(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"].pop("model.layers.1.self_attn.q_proj.weight")
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(SafetensorsError, match="shard set does not match"):
        load_checkpoint(base)


def test_rejects_malformed_sparse_operation_and_missing_tensor_hash(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": [], "value": 1.0}])
    with pytest.raises(EditRejected, match="Invalid sparse"):
        apply_safetensors_patch(base, patch, policy(name), tmp_path / "bad-op")
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 1.0}])
    del patch["tensor_sha256"][name]
    with pytest.raises(EditRejected, match="missing a base tensor hash"):
        apply_safetensors_patch(base, patch, policy(name), tmp_path / "missing-hash")


def test_rejects_candidate_nested_in_base_and_overlarge_inspection_budget(tmp_path):
    base = make_sharded_model(tmp_path / "model")
    name = "model.layers.0.mlp.down_proj.weight"
    patch = patch_for(base, name, changes=[{"tensor": name, "index": [0, 0], "op": "set", "value": 2.0}])
    with pytest.raises(EditRejected, match="nested"):
        apply_safetensors_patch(base, patch, policy(name), base / "candidate")
    from weightbridge.inspection.safetensors import inspect_safetensors
    with pytest.raises(SafetensorsError, match="hard host limit"):
        inspect_safetensors(base, max_stats_bytes=2**30)

