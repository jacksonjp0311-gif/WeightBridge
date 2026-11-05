from copy import deepcopy
from pathlib import Path

import pytest

from weightbridge.core import file_hash, json_dump_atomic
from experiments.phase_iiir_replication import study
from experiments.phase_iiir_replication import execute


def measurement(correct, valid=None):
    valid = set(range(8)) if valid is None else valid
    rows = []
    for counterfactual in (0, 1):
        for order in (0, 1):
            for wording in (0, 1):
                index = counterfactual*4+order*2+wording
                rows.append({"id": str(index), "item": f"{counterfactual}/{order}",
                    "comparison_id": str(counterfactual), "pair_id": "family", "wording": wording,
                    "order": order, "group": "target", "family": "larger-decimal",
                    "prompt": f"question {index}", "answer": "-1", "tier": "negative-close",
                    "correct": index in correct, "valid": index in valid, "completed": True,
                    "status": "correct" if index in correct else "incorrect", "stop_reason": "token_budget",
                    "generated_token_ids": [index], "text": " -1\nextra"})
    return {"endpoint": {}, "rows": rows}


def q_final():
    return {"schema": "weightbridge.supervised_baseline.final_verification.v1", "passed": True,
        "disposition": "VERIFIED_BASELINE_PASS", "parent_sha256": study.PARENT,
        "representation_sha256": "b"*64, "protocol_sha256": None,
        "exact_reconstruction": {"passed": True}, "rollback": {"passed": True},
        "lineage_accepted": False, "production_authority": False,
        "replays": {stage: {"passed": True, "reconstructed": {"exact_native_generation": True},
                            "rollback": {"exact_native_generation": True}} for stage in study.STAGES}}


def protocols(tmp_path):
    common = {"master_protocol_sha256": "a"*64, "global_data_manifest_sha256": "c"*64,
        "checkpoint_sha256": study.PARENT, "training": {"learning_rates": [0.0003]},
        "inference": {"max_new_tokens": 16}, "endpoint": {"parser": "original"},
        "gates": {"hidden": {"min_strict_gain": 2}}, "source_sha256": {}}
    a, b = tmp_path / "r8.json", tmp_path / "r4.json"
    json_dump_atomic(a, {**common, "adapter": {"rank": 8, "alpha": 16}})
    json_dump_atomic(b, {**common, "adapter": {"rank": 4, "alpha": 8}})
    return a, b


def test_paired_retention_rejects_tied_score_with_different_losses():
    before, after = measurement({0, 1, 2, 3}), measurement({4, 5, 6, 7})
    result = study.retention(before, after)
    assert result["strict_comparison_gain"] == 0
    assert result["correct_response_gain"] == 0
    assert not result["passed"]
    assert result["prohibited_regressions"] == ["0", "1", "2", "3"]
    assert len(result["correctness_disagreements"]) == 8


def test_paired_retention_checks_validity_and_controls():
    before = measurement(set(range(8)))
    control = {**before["rows"][0], "id": "control", "group": "preserve", "family": "copy"}
    before["rows"].append(control)
    after = deepcopy(before)
    after["rows"][-1].update(correct=False, valid=False)
    result = study.retention(before, after)
    assert not result["passed"]
    assert result["lost_required_controls"] == ["control"]
    assert result["lost_valid_presentations"] == ["control"]


def test_first_line_pass_does_not_claim_complete_continuation_identity():
    before = measurement(set(range(8)))
    after = deepcopy(before)
    after["rows"][0]["text"] = " -1\ndifferent trailing text"
    result = study.retention(before, after)
    assert result["passed"]
    assert result["complete_continuation_disagreements"] == ["0"]


def test_paired_retention_requires_identical_semantic_cases():
    a, b = measurement(set()), measurement(set())
    b["rows"][0]["prompt"] += " changed"
    with pytest.raises(ValueError, match="different frozen"):
        study.retention(a, b)


@pytest.mark.parametrize("stage", study.STAGES)
def test_verified_replication_requires_every_exact_replay(stage):
    value = q_final()
    value["replays"][stage]["rollback"]["exact_native_generation"] = False
    with pytest.raises(ValueError, match="Every protected stage"):
        study._verified_q(value)


def test_rank4_gate_binds_protocol_rank_artifact_and_master(tmp_path, monkeypatch):
    a, b = protocols(tmp_path)
    final = q_final()
    final["protocol_sha256"] = file_hash(a)
    final_path, gate = tmp_path / "final.json", tmp_path / "replication.json"
    json_dump_atomic(final_path, final)
    study.seal_replication(a, final_path, gate)
    monkeypatch.setattr(execute, "validate_stage_execution", lambda *args: {
        "outcome": {"disposition": "VERIFIED_REPLICATION_PASS"}})
    assert study.validate_replication_gate(study.read(b), gate)["passed"]
    changed = {**study.read(b), "master_protocol_sha256": "d"*64}
    with pytest.raises(ValueError, match="this study"):
        study.validate_replication_gate(changed, gate)
    final_path.write_text("{}")
    with pytest.raises(ValueError, match="identity changed"):
        study.validate_replication_gate(study.read(b), gate)


def test_rank4_gate_requires_completed_outer_verifier(tmp_path, monkeypatch):
    a, b = protocols(tmp_path)
    final = {**q_final(), "protocol_sha256": file_hash(a)}
    final_path, gate = tmp_path / "final.json", tmp_path / "replication.json"
    json_dump_atomic(final_path, final)
    study.seal_replication(a, final_path, gate)
    def failed(*args):
        raise ValueError("Enclosing study stage did not qualify")
    monkeypatch.setattr(execute, "validate_stage_execution", failed)
    with pytest.raises(ValueError, match="Enclosing"):
        study.validate_replication_gate(study.read(b), gate)


def test_replication_disposition_write_once_and_never_from_rank4(tmp_path):
    a, b = protocols(tmp_path)
    final_path, gate = tmp_path / "final.json", tmp_path / "replication.json"
    json_dump_atomic(final_path, {**q_final(), "protocol_sha256": file_hash(b)})
    with pytest.raises(ValueError, match="rank-eight arm"):
        study.seal_replication(b, final_path, gate)
    assert not gate.exists()
    final_path.write_text(__import__("json").dumps({**q_final(), "protocol_sha256": file_hash(a)}))
    study.seal_replication(a, final_path, gate)
    with pytest.raises(FileExistsError):
        study.seal_replication(a, final_path, gate)


def test_bank_b_recipes_cannot_change_after_bank_a(tmp_path):
    a, b = protocols(tmp_path)
    study._matched_arms(a, b)
    changed = study.read(b)
    changed["training"]["learning_rates"] = [0.001]
    b.write_text(__import__("json").dumps(changed))
    with pytest.raises(ValueError, match="matched configurations"):
        study._matched_arms(a, b)


def test_bank_b_unopened_if_development_retention_fails(tmp_path, monkeypatch):
    a, b = protocols(tmp_path)
    monkeypatch.setattr(study, "validate_replication_gate", lambda *args: {"representation_sha256": "b"*64})
    monkeypatch.setattr(study, "_bound_reference", lambda *args: None)
    monkeypatch.setattr(study, "load_checkpoint", lambda path: type("Checkpoint", (), {"checkpoint_sha256": lambda self: "b"*64})())
    monkeypatch.setattr(study.qstudy, "_host_output", lambda *args: None)
    monkeypatch.setattr(study, "development_retention", lambda *args, **kwargs: {"passed": False})
    def forbidden(*args, **kwargs):
        raise AssertionError("A protected bank was opened before its development gate")
    monkeypatch.setattr(study, "read_manifest", forbidden)
    monkeypatch.setattr(study, "load_split", forbidden)
    gate = tmp_path / "replication.json"
    json_dump_atomic(gate, {})
    out = tmp_path / "assessment"
    out.mkdir()
    result = study.qualify_bank_b(tmp_path / "base", tmp_path / "r8", a, tmp_path / "assessment8", gate,
        tmp_path / "r4", tmp_path / "merge.json", tmp_path / "auth.json", b,
        tmp_path / "material", out, tmp_path / "hidden.json")
    assert result["reason"] == "development_retention_failed"
    assert result["stages"] == {}
    assert not (out / "private-independent-rosters").exists()


def test_rank8_reference_failure_blocks_hidden_even_if_rank4_improves(tmp_path, monkeypatch):
    a, b = protocols(tmp_path)
    config = study.read(b)
    config["gates"] = {"preservation-validation": {}, "selection": {"min_strict_gain": 1},
                       "hidden": {"min_strict_gain": 2, "min_family_gain": 1}}
    monkeypatch.setattr(study, "_matched_arms", lambda *args: (config, config))
    monkeypatch.setattr(study, "validate_replication_gate", lambda *args: {"representation_sha256": "b"*64})
    monkeypatch.setattr(study, "_bound_reference", lambda *args: None)
    monkeypatch.setattr(study, "load_checkpoint", lambda path: type("Checkpoint", (), {
        "checkpoint_sha256": lambda self: study.PARENT if Path(path).name == "base" else "b"*64})())
    monkeypatch.setattr(study.qstudy, "_host_output", lambda *args: None)
    monkeypatch.setattr(study, "development_retention", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(study, "_checked_merged", lambda *args: ({}, "d"*64))
    monkeypatch.setattr(study, "_merged_development", lambda *args: {"passed": True})
    manifest = {"splits": {stage: {"sha256": "e"*64} for stage in study.STAGES}}
    monkeypatch.setattr(study, "read_manifest", lambda *args, **kwargs: manifest)
    opened = []
    def load(*args, **kwargs):
        stage = args[1]
        opened.append(stage)
        assert stage != "hidden", "Failed reference must block sealed evaluation"
        return []
    monkeypatch.setattr(study, "load_split", load)
    monkeypatch.setattr(study, "authorize_stage", lambda *args, **kwargs: {"representation_sha256": "d"*64})
    def launch(*args, **kwargs):
        stage, digest = kwargs["stage"], kwargs["checkpoint"]
        correct = set(range(8))
        if stage == "selection" and digest == study.PARENT:
            correct = {0}
        elif stage == "selection" and digest == "b"*64:
            correct = set(range(1, 8))
        return {"measurement": measurement(correct)}
    monkeypatch.setattr(study.qstudy, "_launch", launch)
    gate, receipt = tmp_path / "replication.json", tmp_path / "merge.json"
    json_dump_atomic(gate, {})
    json_dump_atomic(receipt, {})
    material, out = tmp_path / "material", tmp_path / "assessment"
    material.mkdir()
    out.mkdir()
    json_dump_atomic(material / "manifest.json", {})
    result = study.qualify_bank_b(tmp_path / "base", tmp_path / "r8", a, tmp_path / "assessment8", gate,
        tmp_path / "r4", receipt, tmp_path / "auth.json", b, material, out, tmp_path / "hidden.json")
    assert opened == ["preservation-validation", "selection"]
    assert result["reason"] == "bank_b_selection_failed"
    selected = result["stages"]["selection"]
    assert selected["rank4_ordinary"]["passed"]
    assert selected["rank4_retention"]["passed"]
    assert not selected["rank8_reference"]["passed"]
    assert "hidden" not in result["stages"]
    assert study.read(out / "independent.json")["reason"] != study.qverify.PENDING_REASON
