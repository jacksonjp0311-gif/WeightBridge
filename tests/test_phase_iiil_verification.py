import json
from types import SimpleNamespace

import pytest

from experiments.phase_iiil import semantic_roster as roster, verification as verify
from weightbridge.adaptation.behavior_endpoint import _observation
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def trial(tmp_path):
    material, hidden = tmp_path / "material", tmp_path / "host-hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="test-only-phase-iiil-verifier")
    config = verify._read(roster.PROTOCOL)
    out, base = tmp_path / "trial", tmp_path / "model"
    out.mkdir()
    return SimpleNamespace(material=material, hidden=hidden, manifest=manifest, config=config, out=out,
                           base=base, base_sha=config["checkpoint_sha256"], candidate_sha="b" * 64)


def _fit(t, *, candidate=False):
    artifacts, candidates = {}, {}
    if candidate:
        receipt = t.out / "patches" / "A1-receipt.json"
        receipt.parent.mkdir()
        receipt.write_bytes((json.dumps({"fixture": True}, indent=2, sort_keys=True) + "\n").encode())
        arm = t.out / "arm-A1.json"
        json_dump_atomic(arm, {"probe": {"restored_exactly": True, "candidate_tensor_sha256": "a" * 64,
                                        "endpoint": {"guard": {"passed": True}}}})
        candidates["A1"] = {"checkpoint_sha256": t.candidate_sha, "tensor_sha256": "a" * 64,
                            "receipt_sha256": file_hash(receipt), "realized_update_norm": .01, "changed_entries": 12}
        artifacts = {path.relative_to(t.out).as_posix(): file_hash(path) for path in (arm, receipt)}
    json_dump_atomic(t.out / "fit-report.json", {"schema": "weightbridge.phase_iiil.fit.v1",
        "checkpoint_sha256": t.base_sha, "sources": verify.source_hashes(), "artifacts": artifacts,
        "manifest_sha256": file_hash(t.material / "manifest.json"), "candidates": candidates,
        "optimization_splits": ["adaptation", "preservation-fit", "calibration"], "hidden_opened": False})


def _mock_processes(monkeypatch, t, *, regress_transfer=False):
    calls = []
    monkeypatch.setattr(verify, "load_checkpoint", lambda _: SimpleNamespace(checkpoint_sha256=lambda: t.base_sha))
    monkeypatch.setattr(verify, "_checkpoint", lambda *args: t.out / "checkpoint")

    def launch(path, sha, material, split, out, manifest, config, **kwargs):
        assert split in verify.PUBLIC_SPLITS and not kwargs
        calls.append((split, sha))
        rows = roster.load_split(material, split, manifest)
        affected = rows[0]["comparison_id"]
        contract = verify.endpoint(config)
        measured = []
        for index, row in enumerate(rows):
            wrong = (sha == t.base_sha and split == "selection" and row["comparison_id"] == affected)
            wrong |= bool(regress_transfer and sha == t.candidate_sha and split == verify.TRANSFER and index == 0)
            answer = next(value for value in row["operands"] if value != row["answer"]) if wrong else row["answer"]
            generated = {**row, "text": answer + "\n", "stop_reason": "token_budget"}
            measured.append({**generated, **_observation(generated, contract.question(row), contract)})
        record = {"schema": verify.WORKER_SCHEMA, "split": split, "checkpoint_sha256": sha,
                  "roster_sha256": manifest["splits"][split]["sha256"],
                  "manifest_sha256": file_hash(material / "manifest.json"),
                  "measurement": {"endpoint": contract.manifest(), "rows": measured, "resources": {}}, "resources": {}}
        json_dump_atomic(out, record)
        return record

    monkeypatch.setattr(verify, "_launch", launch)
    return calls


def test_no_practice_candidate_keeps_hidden_sealed_and_measures_both_parents(monkeypatch, trial):
    _fit(trial)
    calls = _mock_processes(monkeypatch, trial)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert calls == [(verify.TRANSFER, trial.base_sha), ("selection", trial.base_sha)]
    assert result["reason"] == "no_practice_candidate" and not result["hidden_opened"]
    with pytest.raises(ValueError, match="does not authorize"):
        verify.finish(trial.base, trial.material, trial.out, trial.hidden, result["candidate_lock_sha256"])


def test_transfer_regression_blocks_candidate_selection(monkeypatch, trial):
    _fit(trial, candidate=True)
    calls = _mock_processes(monkeypatch, trial, regress_transfer=True)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert calls[-1] == (verify.TRANSFER, trial.candidate_sha)
    assert result["reason"] == "transfer_preservation_failure" and not result["selection"]
    assert not result["transfer"]["A1"]["passed"]


def test_distinct_passer_locks_and_report_replay_rejects_changed_score(monkeypatch, trial):
    _fit(trial, candidate=True)
    calls = _mock_processes(monkeypatch, trial)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert calls[-2:] == [(verify.TRANSFER, trial.candidate_sha), ("selection", trial.candidate_sha)]
    assert result["winner"] == "A1" and result["selection"]["A1"]["target_gain"] == 1
    lock = verify._read(trial.out / "candidate-lock.json")
    assert verify.replay_lock(trial.out, trial.material, lock)["winner"] == "A1"
    # Rebinding a modified receipt is insufficient if its parser result disagrees with the raw generation.
    receipt = trial.out / "verification" / "selection-A1.json"
    forged = verify._read(receipt)
    forged["measurement"]["rows"][0]["correct"] = False
    receipt.write_text(json.dumps(forged), encoding="utf-8")
    lock["reports"][receipt.name] = file_hash(receipt)
    with pytest.raises(ValueError, match="parser classification"):
        verify.replay_lock(trial.out, trial.material, lock)
