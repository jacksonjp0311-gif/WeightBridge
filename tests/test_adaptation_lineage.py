import json
from pathlib import Path

import numpy as np
import pytest

from weightbridge.adaptation import lineage
from weightbridge.evaluation import hf_edit
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def research(tmp_path, monkeypatch):
    save_file = pytest.importorskip("safetensors.numpy").save_file
    base = tmp_path / "base"
    base.mkdir()
    _write(base / "config.json", {})
    save_file({"weight": np.zeros(2, dtype=np.float32)}, str(base / "model.safetensors"))
    workspace = tmp_path / "research"
    workspace.mkdir()
    state = lineage.init_lineage(workspace / "lineage", base)
    assert state["head_path"] == str(base)

    predictions = {}

    def score(path, choices, rows, **kwargs):
        return predictions[path]

    monkeypatch.setattr(hf_edit, "_score_checkpoint", score)

    def cycle(cycle_id, parent, *, pass_gate=True, merged=False):
        cycle_dir = workspace / "cycles" / cycle_id
        cycle_dir.mkdir(parents=True)
        candidate = cycle_dir / "model"
        checkpoint = load_checkpoint(parent)
        patch = {
            "schema": "weightbridge.patch.v1", "base_sha256": checkpoint.checkpoint_sha256(),
            "tensor_sha256": {"weight": checkpoint.tensor_sha256("weight")},
            "reason": "Test candidate", "changes": [{"tensor": "weight", "index": [0], "op": "add", "value": 0.25}],
        }
        policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["weight"],
                  "max_entries": 1, "max_fraction": 0.5, "max_abs_delta": 0.5}
        apply_safetensors_patch(parent, patch, policy, candidate)
        source = {
            "schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
            "target": [{"id": f"{cycle_id}-target", "text": "Target", "label": "A"}],
            "sentinel": [{"id": f"{cycle_id}-sentinel", "text": "Sentinel", "label": "B"}],
        }
        source_path = _write(cycle_dir / "holdout.json", source)
        suite_path = source_path
        if merged:
            suite = {**source, "sentinel": [*source["sentinel"], {"id": "old-target", "text": "Old", "label": "A"}]}
            suite_path = _write(cycle_dir / "merged.json", suite)
        predictions[parent] = ["B", "B"] + (["A"] if merged else [])
        predictions[candidate] = ["A", "B" if pass_gate else "A"] + (["A"] if merged else [])
        report = cycle_dir / "report.json"
        hf_edit.evaluate_hf_edit(parent, candidate, suite_path, report)
        return candidate, report, suite_path, source_path

    return workspace / "lineage", base, cycle


def test_accept_reject_and_rollback_preserve_checkpoints(research):
    directory, base, cycle = research
    initial = lineage.lineage_status(directory)
    first, report, suite, source = cycle("first", base, merged=True)
    accepted = lineage.record_lineage_decision(
        directory, first, report, suite=suite, source_holdout=source,
        cycle_id="first", accept=True, reason="Verified target gain",
        expected_parent_sha256=initial["head_sha256"],
    )
    assert accepted["head_path"] == str(first)
    assert accepted["ancestry"][-1]["cycle_id"] == "first"
    assert accepted["ancestry"][-1]["holdout_path"] == str(source)
    assert accepted["events"][0]["kind"] == "ACCEPT"
    assert lineage.audit_lineage(directory) == accepted

    bad, report, suite, _ = cycle("bad", first, pass_gate=False)
    with pytest.raises(ValueError, match="failed verification"):
        lineage.record_lineage_decision(directory, bad, report, suite=suite,
                                        cycle_id="bad", accept=True, reason="Invalid acceptance")
    rejected = lineage.record_lineage_decision(directory, bad, report, suite=suite,
                                               cycle_id="bad", accept=False, reason="Sentinel changed")
    assert rejected["head_sha256"] == accepted["head_sha256"]
    assert rejected["events"][-1]["verification"]["disposition"] == "FAIL"
    with pytest.raises(ValueError, match="parent changed"):
        lineage.record_lineage_rejection(directory, cycle_id="stale", reason="Stale",
                                         expected_parent_sha256=initial["head_sha256"])
    lineage.record_lineage_rejection(directory, cycle_id="stopped", reason="No candidate",
                                     expected_parent_sha256=accepted["head_sha256"])

    rolled_back = lineage.rollback_lineage(directory, initial["head_sha256"], reason="Restore original")
    assert rolled_back["head_path"] == str(base)
    assert rolled_back["events"][-1]["kind"] == "ROLLBACK"
    assert load_checkpoint(base).checkpoint_sha256() == initial["head_sha256"]
    assert load_checkpoint(first).checkpoint_sha256() == accepted["head_sha256"]
    with pytest.raises(ValueError, match="earlier accepted ancestor"):
        lineage.rollback_lineage(directory, accepted["head_sha256"], reason="Detached branch")


def test_lineage_separates_status_from_evidence_audit(research):
    directory, base, cycle = research
    candidate, report, suite, source = cycle("first", base, merged=True)
    original_report = report.read_bytes()
    forged = json.loads(original_report)
    forged["target_gain"] = 2
    _write(report, forged)
    with pytest.raises(ValueError, match="differs from paired prediction"):
        lineage.record_lineage_decision(directory, candidate, report, suite=suite,
                                        cycle_id="first", accept=True, reason="Forged report")
    report.write_bytes(original_report)
    forged_source = json.loads(source.read_bytes())
    forged_source["target"][0]["text"] = "Different source prompt"
    mismatched = _write(source.with_name("different.json"), forged_source)
    with pytest.raises(ValueError, match="preserve the source holdout"):
        lineage.record_lineage_decision(directory, candidate, report, suite=suite, source_holdout=mismatched,
                                        cycle_id="first", accept=True, reason="Mismatched holdout")

    accepted = lineage.record_lineage_decision(directory, candidate, report, suite=suite, source_holdout=source,
                                               cycle_id="first", accept=True, reason="Verified report")
    report.unlink()
    assert lineage.lineage_status(directory) == accepted
    with pytest.raises(OSError):
        lineage.audit_lineage(directory)
    _write(Path(accepted["head_path"]) / "config.json", {"changed": True})
    with pytest.raises(ValueError, match="Active checkpoint changed"):
        lineage.lineage_status(directory)
    restored = lineage.rollback_lineage(directory, accepted["root_sha256"], reason="Recover from checkpoint drift")
    assert restored["head_path"] == str(base)

    final_event = directory / "events" / "00000002.json"
    changed = json.loads(final_event.read_bytes())
    changed["reason"] = "Edited after rollback"
    _write(final_event, changed)
    with pytest.raises(ValueError, match="event sequence or hash chain differs"):
        lineage.lineage_status(directory)
