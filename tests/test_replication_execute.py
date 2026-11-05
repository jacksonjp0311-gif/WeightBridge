from pathlib import Path

import pytest

from experiments.phase_iiir_replication import execute
from experiments.phase_iiir_replication import training
from weightbridge.core import file_hash, json_dump_atomic


def fixture(tmp_path, arm="rank8", passed=True):
    local = tmp_path / arm
    protocol = local / "protocol.json"
    json_dump_atomic(protocol, {"source_sha256": {}, "stage_resources": {
        action: {"wall_seconds": 10, "cpu_seconds": 20, "memory_bytes": 100}
        for action in ("assess", "qualify", "verify")}})
    json_dump_atomic(local / "main-execution.json", {"passed": passed, "protocol_sha256": file_hash(protocol)})
    return local


def test_assessment_refuses_incomplete_main_before_generation(tmp_path, monkeypatch):
    fixture(tmp_path, passed=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("An incomplete arm cannot be scored for finalist selection")
    monkeypatch.setattr(execute.qstudy, "assess", forbidden)
    def incomplete(*args, **kwargs):
        raise ValueError("Incomplete recorded training")
    monkeypatch.setattr(training, "validate_main_execution", incomplete, raising=False)
    with pytest.raises(ValueError):
        execute.worker(tmp_path / "base", tmp_path, "rank8", "assess")


def test_rank4_cannot_touch_confirmation_without_verified_replication(tmp_path, monkeypatch):
    fixture(tmp_path, arm="rank4")
    def refused(*args, **kwargs):
        raise ValueError("Replication did not qualify")
    monkeypatch.setattr(execute.study, "validate_replication_gate", refused)
    monkeypatch.setattr(execute.study, "prepare_merge", lambda *args: pytest.fail("merge was entered"))
    monkeypatch.setattr(training, "validate_main_execution", lambda *args: {"passed": True}, raising=False)
    monkeypatch.setattr(execute, "previous_stage", lambda *args: None)
    with pytest.raises(ValueError):
        execute.worker(tmp_path / "base", tmp_path, "rank4", "qualify")


def test_rank4_development_loss_blocks_merge_and_bank_b(tmp_path, monkeypatch):
    fixture(tmp_path, arm="rank4")
    monkeypatch.setattr(execute.study, "validate_replication_gate", lambda *args: {"passed": True})
    monkeypatch.setattr(execute.study, "development_retention", lambda *args: {"passed": False})
    monkeypatch.setattr(execute.study, "prepare_merge", lambda *args: pytest.fail("merge was entered"))
    monkeypatch.setattr(execute.study, "qualify_bank_b", lambda *args: pytest.fail("Bank B was entered"))
    # This fixture focuses on the post-completion path. The main training
    # receipt replayer itself is independently exercised in training tests.
    monkeypatch.setattr(training, "validate_main_execution", lambda *args: {"passed": True}, raising=False)
    monkeypatch.setattr(execute, "previous_stage", lambda *args: None)
    result = execute.worker(tmp_path / "base", tmp_path, "rank4", "qualify")
    assert result["reason"] == "rank4_development_retention_failed"
    assert result["bank_b_opened"] is False
    assert (tmp_path / "rank4/adapter-development-retention.json").is_file()


def test_inner_pass_cannot_bypass_enclosing_budget_failure(tmp_path):
    local = fixture(tmp_path)
    worker_path = tmp_path / "execution/rank8-assess.worker.json"
    json_dump_atomic(worker_path, {"passed": True})
    json_dump_atomic(tmp_path / "execution/rank8-assess.json", {
        "schema": "weightbridge.phase_iiir.stage_execution.v1", "arm": "rank8", "action": "assess",
        "protocol_sha256": file_hash(local / "protocol.json"), "passed": False,
        "within_resource_limits": False, "outcome": {"passed": True}, "outcome_sha256": file_hash(worker_path)})
    with pytest.raises(ValueError):
        execute.previous_stage(tmp_path, "rank8", "qualify", local / "protocol.json")
