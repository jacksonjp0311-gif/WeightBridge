import json

import numpy as np
import pytest

from weightbridge.evaluation import hf_edit
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch


def test_hf_edit_evaluation_binds_receipt_and_gates_gain_and_locality(tmp_path, monkeypatch):
    save_file = pytest.importorskip("safetensors.numpy").save_file
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    save_file({"weight": np.array([0.0, 0.0], dtype=np.float32)}, str(base / "model.safetensors"))
    candidate = tmp_path / "candidate"
    checkpoint = load_checkpoint(base)
    patch = {
        "schema": "weightbridge.patch.v1",
        "base_sha256": checkpoint.checkpoint_sha256(),
        "tensor_sha256": {"weight": checkpoint.tensor_sha256("weight")},
        "reason": "Test a policy-applied candidate",
        "changes": [{"tensor": "weight", "index": [0], "op": "add", "value": 0.25}],
    }
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["weight"],
              "max_entries": 1, "max_fraction": 0.5, "max_abs_delta": 0.5}
    receipt = apply_safetensors_patch(base, patch, policy, candidate)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps({
        "schema": "weightbridge.hf_edit_suite.v1",
        "choices": ["A", "B"],
        "target": [{"id": "t1", "text": "Target question", "label": "A"}],
        "sentinel": [{"id": "s1", "text": "Unrelated question", "label": "B"}],
    }), encoding="utf-8")

    def fake_score(path, choices, rows, **configuration):
        assert choices == ["A", "B"]
        assert [row["id"] for row in rows] == ["t1", "s1"]
        return predictions[path.name]

    predictions = {"base": ["B", "B"], "candidate": ["A", "B"]}
    monkeypatch.setattr(hf_edit, "_score_checkpoint", fake_score)
    out = tmp_path / "report.json"
    passed = hf_edit.evaluate_hf_edit(base, candidate, suite_path, out)
    assert passed["disposition"] == "PASS"
    assert passed["target_gain"] == 1
    assert passed["target_corrections"] == 1
    assert passed["target_regressions"] == 0
    assert passed["candidate_sha256"] == receipt["candidate_sha256"]
    assert json.loads(out.read_text(encoding="utf-8")) == passed

    predictions["candidate"] = ["A", "A"]
    failed = hf_edit.evaluate_hf_edit(base, candidate, suite_path, tmp_path / "failed.json")
    assert failed["disposition"] == "FAIL"
    assert failed["target_gain"] == failed["sentinel_prediction_flips"] == 1

    with pytest.raises(ValueError, match="already exists"):
        hf_edit.evaluate_hf_edit(base, candidate, suite_path, out)

    raced = tmp_path / "raced.json"

    def score_with_competing_writer(path, choices, rows, **configuration):
        if path == candidate:
            raced.write_text("competing report", encoding="utf-8")
        return fake_score(path, choices, rows, **configuration)

    monkeypatch.setattr(hf_edit, "_score_checkpoint", score_with_competing_writer)
    with pytest.raises(ValueError, match="Refusing to replace"):
        hf_edit.evaluate_hf_edit(base, candidate, suite_path, raced)
    assert raced.read_text(encoding="utf-8") == "competing report"

    (candidate / "config.json").write_text('{"tampered":true}', encoding="utf-8")
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="Patch receipt must bind"):
        hf_edit.evaluate_hf_edit(base, candidate, suite_path, rejected)
    assert not rejected.exists()
