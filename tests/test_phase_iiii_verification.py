import json
from pathlib import Path

import pytest

from experiments.phase_iiii import verification
from weightbridge.adaptation.fork_policy import decide_fork_action
from weightbridge.core import file_hash


MATERIAL = Path(__file__).resolve().parents[1] / "results/phase-iii-i/causal-fork-v1/material"


def test_hidden_worker_refuses_an_unlocked_or_changed_seal_before_reading(tmp_path, monkeypatch):
    hidden = tmp_path / "private-roster.json"
    lock = tmp_path / "candidate-lock.json"
    lock.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(verification, "load_split", lambda *_args, **_kw: pytest.fail("Read before lock"))
    args = (tmp_path / "base", "unavailable", MATERIAL, "hidden", tmp_path / "out.json",
            file_hash(MATERIAL / "manifest.json"))
    with pytest.raises(ValueError, match="finalist lock"):
        verification.worker(*args, hidden=hidden)
    with pytest.raises(ValueError, match="hash-bound"):
        verification.worker(*args, hidden=hidden, lock=lock, lock_sha="changed")
    with pytest.raises(ValueError, match="does not authorize"):
        verification.worker(*args, hidden=hidden, lock=lock, lock_sha=file_hash(lock))


def test_verifier_rejects_a_report_with_substituted_roster_rows(tmp_path):
    manifest = verification.read_manifest(MATERIAL)
    config = verification._read(verification.PROTOCOL)
    rows = verification.load_split(MATERIAL, "selection", manifest)
    rows[0] = {**rows[0], "prompt": "substituted"}
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema": verification.WORKER_SCHEMA, "split": "selection",
        "checkpoint_sha256": config["checkpoint_sha256"], "manifest_sha256": file_hash(MATERIAL / "manifest.json"),
        "roster_sha256": manifest["splits"]["selection"]["sha256"],
        "measurement": {"endpoint": verification.endpoint(config).manifest(), "rows": rows}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen checkpoint, endpoint, or roster"):
        verification._validate_report(report, config["checkpoint_sha256"], MATERIAL, manifest, config, "selection")


def test_post_verification_policy_refuses_failed_replay_even_after_hidden_pass():
    verified = {"disposition": "PASS", "hidden_gate_passed": True, "restart_exact": True,
                "rollback_exact": True, "lineage_event_sha256": "bound"}
    parameters = {"measurement_valid": True, "persistent_errors": 2, "practice_pass": True, "selection_pass": True}
    assert decide_fork_action(**parameters, independent=verified).action == "record_verified"
    assert decide_fork_action(**parameters, independent={**verified, "restart_exact": False}).action == "hold"
    assert decide_fork_action(**parameters, independent={**verified, "lineage_event_sha256": None}).action == "hold"


def test_finish_recomputes_lock_gate_before_launching_any_hidden_worker(tmp_path, monkeypatch):
    config = verification._read(verification.PROTOCOL)
    receipt = tmp_path / "checkpoints" / "A" / ".weightbridge_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    workers = tmp_path / "verification"
    workers.mkdir()
    reports = {}
    for name in ("base", "A"):
        path = workers / f"selection-{name}.json"
        path.write_text("{}", encoding="utf-8")
        reports[name] = file_hash(path)
    lock = {"winner": "A", "winner_sha256": "candidate", "base_sha256": config["checkpoint_sha256"],
            "selection": {"A": {"passed": True}}, "reports": reports,
            "candidates": {"A": {"receipt_sha256": file_hash(receipt)}}}
    monkeypatch.setattr(verification, "_check_lock", lambda *args: lock)
    monkeypatch.setattr(verification, "_validate_report", lambda *args: {"measurement": {}})
    monkeypatch.setattr(verification, "_paired", lambda *args: {"passed": False})
    monkeypatch.setattr(verification, "_launch", lambda *_args, **_kwargs: pytest.fail("Opened hidden too early"))
    with pytest.raises(ValueError, match="independently recomputed"):
        verification.finish(tmp_path / "parent", MATERIAL, tmp_path, tmp_path / "private.json", "sha")
    assert not (tmp_path / "independent-outcome.json").exists()
