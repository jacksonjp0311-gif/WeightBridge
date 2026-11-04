from pathlib import Path

import pytest

from experiments.phase_iiij import verification
from weightbridge.adaptation.preservation_policy import PreservationEvidence, preservation_decision
from weightbridge.core import file_hash


MATERIAL = Path(__file__).resolve().parents[1] / "results/phase-iii-j/preservation-v1/material"


def test_hidden_data_requires_private_finalist_lock_before_loading_roster(tmp_path, monkeypatch):
    hidden = tmp_path / "private-roster.json"
    lock = tmp_path / "candidate-lock.json"
    lock.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(verification, "load_split", lambda *_args, **_kwargs: pytest.fail("Roster read before lock"))
    args = (tmp_path / "base", "unavailable", MATERIAL, "hidden", tmp_path / "out.json",
            file_hash(MATERIAL / "manifest.json"))
    with pytest.raises(ValueError, match="finalist lock"):
        verification.worker(*args, hidden=hidden)
    with pytest.raises(ValueError, match="hash-bound"):
        verification.worker(*args, hidden=hidden, lock=lock, lock_sha="changed")


def test_finalist_rank_is_recomputed_before_hidden_measurement(tmp_path, monkeypatch):
    receipt = tmp_path / "checkpoints" / "A" / ".weightbridge_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    workers = tmp_path / "verification"
    workers.mkdir()
    reports = {}
    gates = {"A": {"passed": True, "target_item_gain": 1}, "B": {"passed": True, "target_item_gain": 2}}
    for name in ("base", "A", "B"):
        path = workers / f"selection-{name}.json"
        path.write_text("{}", encoding="utf-8")
        reports[name] = file_hash(path)
    lock = {"winner": "A", "winner_sha256": "A", "base_sha256": "parent", "selection": gates,
            "reports": reports, "candidates": {
                name: {"checkpoint_sha256": name, "realized_update_norm": .05,
                       "receipt_sha256": file_hash(receipt)} for name in ("A", "B")}}
    monkeypatch.setattr(verification, "_check_lock", lambda *args: lock)
    monkeypatch.setattr(verification, "_validate_report", lambda _path, sha, *args: {"measurement": sha})
    monkeypatch.setattr(verification, "_paired", lambda _parent, candidate, *args: gates[candidate])
    monkeypatch.setattr(verification, "_launch", lambda *_args, **_kwargs: pytest.fail("Opened hidden early"))
    with pytest.raises(ValueError, match="finalist differs"):
        verification.finish(tmp_path / "parent", MATERIAL, tmp_path, tmp_path / "private.json", "sha")
    assert not (tmp_path / "independent-outcome.json").exists()


def test_preservation_policy_cannot_authorize_failed_restart():
    decision = preservation_decision(PreservationEvidence(True, 2, missing_groups=("copy",)))
    assert (decision.action, decision.reason) == ("broaden_preservation", "missing_preservation_coverage")
    decision = preservation_decision(PreservationEvidence(True, 2, locally_feasible=False, alternate_layer_available=True))
    assert (decision.action, decision.reason) == ("alternate_layer", "empty_local_interval")
    decision = preservation_decision(PreservationEvidence(True, 2, corrected_with_regressions=True,
                                                            refinement_available=True))
    assert decision.action == "refine_constraints"
    verified = {"disposition": "PASS", "hidden_gate_passed": True, "restart_exact": False,
                "rollback_exact": True, "lineage_event_sha256": "hash"}
    evidence = PreservationEvidence(True, 2, practice_pass=True, selection_pass=True)
    assert preservation_decision(evidence, independent=verified).action == "hold"
    assert preservation_decision(evidence, independent={**verified, "restart_exact": True}).action == "record_verified"
