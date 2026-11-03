import json

import pytest

from weightbridge.adaptation.behavior_stage import compare_behavior_rows, compile_stage_one_candidate, stage_one_admission
from weightbridge.core import file_hash


def test_stage_one_refuses_unqualified_measurement_before_model_access(tmp_path):
    base_hash = "a" * 64
    protocol = tmp_path / "protocol.json"
    config = {"schema": "weightbridge.phase_iiic.measurement_protocol.v1", "model_sha256": base_hash}
    protocol.write_text(json.dumps(config), encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema": "weightbridge.phase_iiic.measurement_report.v1",
                                  "protocol_sha256": file_hash(protocol), "base_sha256": base_hash,
                                  "selected_rule": "mean", "qualification_gate": {"passed": False},
                                  "headroom_gate": {"passed": True, "eligible_families": ["date-order"]}}),
                      encoding="utf-8")
    admission = stage_one_admission(protocol, report)
    assert stage_one_admission(protocol, report) == admission
    assert admission.reasons == ("MEASUREMENT_NOT_QUALIFIED",)
    with pytest.raises(ValueError, match="requires qualified measurement"):
        compile_stage_one_candidate(admission, arm="C", base=tmp_path / "missing", output=tmp_path / "out",
                                    family="date-order", targets=[], preservation={}, tensor_names=[],
                                    inference={}, policy={})
    assert not (tmp_path / "out").exists()
    protocol.write_text(json.dumps({**config, "changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        stage_one_admission(protocol, report)


def _row(row_id, item_id, role, score, generated):
    return {"id": row_id, "item_id": item_id, "role": role, "correct": 0, "scored": score, "generated": generated}


def test_teacher_forced_improvement_requires_completed_generation_and_preservation():
    target = [_row(f"target-{index}", "target", "target", 1, 1) for index in range(4)]
    sentinel = _row("sentinel-0", "sentinel", "preservation", 0, 0)
    before = target + [sentinel]
    scores_only = [{**row, "scored": 0} for row in target] + [sentinel]
    assert compare_behavior_rows(before, scores_only)["joint_item_gain"] == 0
    both = [{**row, "scored": 0, "generated": 0} for row in target]
    assert compare_behavior_rows(before, both + [sentinel])["passed"]
    assert not compare_behavior_rows(before, both + [{**sentinel, "generated": None}])["passed"]
    assert not compare_behavior_rows(before, both[:3] + [scores_only[3], sentinel])["passed"]
    previously_generated = [{**row, "generated": 0} for row in target] + [sentinel]
    assert not compare_behavior_rows(previously_generated, both + [sentinel])["passed"]


def test_stage_runner_records_hold_without_opening_model_or_sealed_material(tmp_path):
    from experiments.phase_iiic.run_stage_one import PROTOCOL, run

    report = tmp_path / "measurement.json"
    config = json.loads(PROTOCOL.read_bytes())
    report.write_text(json.dumps({"schema": "weightbridge.phase_iiic.measurement_report.v1",
                                  "protocol_sha256": file_hash(PROTOCOL), "base_sha256": config["model_sha256"],
                                  "selected_rule": "mean", "qualification_gate": {"passed": True},
                                  "headroom_gate": {"passed": False, "eligible_families": []}}), encoding="utf-8")
    out = tmp_path / "out"
    assert run(tmp_path / "no-model", tmp_path / "no-material", report, out) == "HOLD"
    evidence = json.loads((out / "report.json").read_bytes())
    assert evidence["candidate_attempts"] == 0
    assert evidence["reasons"] == ["NO_AUTHORIZED_HEADROOM"]
