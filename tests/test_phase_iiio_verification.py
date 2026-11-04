import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from experiments.phase_iiio import semantic_roster as roster, verification as verify
from weightbridge.adaptation.behavior_endpoint import _observation
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def trial(tmp_path, monkeypatch):
    material, hidden = tmp_path / "material", tmp_path / "hidden.json"
    manifest = roster.freeze(material, hidden, hidden_seed="phase-m-verifier-hidden")
    config = verify._read(roster.PROTOCOL)
    out = tmp_path / "trial"
    out.mkdir()
    monkeypatch.setattr(verify, "source_hashes", lambda **kwargs: {"test-only-source": "a" * 64})
    return SimpleNamespace(material=material, hidden=hidden, manifest=manifest, config=config, out=out,
                           base=tmp_path / "model", base_sha=config["checkpoint_sha256"], candidate_sha="b" * 64)


def _fit(t, candidate=True):
    artifacts, candidates = {}, {}
    if candidate:
        receipt = t.out / "patches" / "A1-receipt.json"
        receipt.parent.mkdir()
        receipt.write_bytes((json.dumps({"fixture": True}, indent=2, sort_keys=True) + "\n").encode())
        arm = t.out / "arm-A1.json"
        json_dump_atomic(arm, {"left_factor": [.1], "norm_match": {"status": "MATCHED"}, "probe": {
            "restored_exactly": True, "candidate_tensor_sha256": "c" * 64, "endpoint": {"guard": {"passed": True}},
            "native_measurement": {"passed": True, "restored_exactly": True, "tensor_sha256": "c" * 64,
                "targets": {"rows": [{"prompt_id": "test-fork", "measured_margin": .125,
                    "accepted_token_ids": [1], "greedy_token_id": 1}]}}}})
        basis = t.out / "basis.json"
        json_dump_atomic(basis, {"forks": [{"prompt_id": "test-fork", "accepted_token_ids": [1],
            "parent_margin": -.01, "projected_margin_gradient": [1.]}]})
        artifacts = {path.relative_to(t.out).as_posix(): file_hash(path) for path in (receipt, arm, basis)}
        candidates["A1"] = {"checkpoint_sha256": t.candidate_sha, "tensor_sha256": "c" * 64,
                            "receipt_sha256": file_hash(receipt), "realized_update_norm": .0125, "changed_entries": 10}
    json_dump_atomic(t.out / "fit-report.json", {"schema": "weightbridge.phase_iiio.fit.v1", "sources": verify.source_hashes(),
        "checkpoint_sha256": t.base_sha, "manifest_sha256": file_hash(t.material / "manifest.json"),
        "artifacts": artifacts, "candidates": candidates, "optimization_splits": ["adaptation", "preservation-fit", "calibration"],
        "hidden_opened": False})


def _processes(monkeypatch, t, *, complete_family=True, preservation_regression=False):
    calls = []
    monkeypatch.setattr(verify, "load_checkpoint", lambda _: SimpleNamespace(checkpoint_sha256=lambda: t.base_sha))
    monkeypatch.setattr(verify, "_checkpoint", lambda *args: t.out / "checkpoint")

    def launch(path, sha, material, split, out, manifest, config, *, lock=None, hidden=None):
        assert hidden is None
        assert (lock is None) == (split == verify.DEVELOPMENT)
        if lock:
            earlier = verify._read(lock)
            assert earlier["eligible"] and earlier["winner"] == "A1"
        calls.append((split, sha))
        rows, measured = roster.load_split(material, split, manifest), []
        family, comparison = rows[0]["pair_id"], rows[0]["comparison_id"]
        contract = verify.endpoint(config)
        for i, row in enumerate(rows):
            wrong = sha == t.base_sha and ((split == verify.DEVELOPMENT and row["pair_id"] == family)
                                          or (split == "selection" and row["comparison_id"] == comparison))
            wrong |= sha == t.candidate_sha and split == verify.DEVELOPMENT and not complete_family and row["pair_id"] == family and row["counterfactual"] == 1
            wrong |= sha == t.candidate_sha and split == verify.PRESERVATION and preservation_regression and i == 0
            answer = next(value for value in row["operands"] if value != row["answer"]) if wrong else row["answer"]
            generated = {**row, "text": answer + "\n", "stop_reason": "token_budget"}
            measured.append({**generated, **_observation(generated, contract.question(row), contract)})
        record = {"schema": verify.WORKER_SCHEMA, "split": split, "checkpoint_sha256": sha,
            "roster_sha256": manifest["splits"][split]["sha256"], "manifest_sha256": file_hash(material / "manifest.json"),
            "measurement": {"endpoint": contract.manifest(), "rows": measured, "resources": {}}, "resources": {}}
        json_dump_atomic(out, record)
        return record

    monkeypatch.setattr(verify, "_launch", launch)
    return calls


def test_no_candidate_opens_no_postfit_partition(monkeypatch, trial):
    _fit(trial, candidate=False)
    calls = _processes(monkeypatch, trial)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert calls == [] and result["reason"] == "no_development_transfer_candidate"
    with pytest.raises(FileExistsError):
        verify.assess(trial.base, trial.material, trial.out)
    assert not result["hidden_opened"]


def test_strict_gain_without_complete_truthflip_blocks_later_partitions(monkeypatch, trial):
    _fit(trial)
    calls = _processes(monkeypatch, trial, complete_family=False)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert result["development"]["A1"]["target_gain"] == 1
    assert result["development"]["A1"]["complete_family_gain"] == 0
    assert not result["winner"] and {split for split, _ in calls} == {verify.DEVELOPMENT}


def test_preservation_failure_stops_before_selection(monkeypatch, trial):
    _fit(trial)
    calls = _processes(monkeypatch, trial, preservation_regression=True)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert result["development_winner"] == "A1" and not result["winner"]
    assert result["reason"] == "preservation_validation_failure"
    assert {split for split, _ in calls} == {verify.DEVELOPMENT, verify.PRESERVATION}


def test_locked_stages_replay_and_forged_parser_cannot_authorize_hidden(monkeypatch, trial):
    _fit(trial)
    _processes(monkeypatch, trial)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert result["winner"] == "A1" and result["reason"] == "hidden_verification_required"
    lock = verify._read(trial.out / "candidate-lock.json")
    assert verify.replay_lock(trial.out, trial.material, lock)["eligible"]
    report = trial.out / "verification" / "selection-A1.json"
    forged = verify._read(report)
    forged["measurement"]["rows"][0]["correct"] = False
    report.write_text(json.dumps(forged), encoding="utf-8")
    lock["reports"][report.name] = file_hash(report)
    with pytest.raises(ValueError, match="deterministic answer parsing"):
        verify.replay_lock(trial.out, trial.material, lock)


def test_selection_worker_requires_eligible_preservation_lock(trial):
    with pytest.raises(ValueError, match="earlier private verification lock"):
        verify.worker(trial.base, trial.base_sha, trial.material, "selection", trial.out / "unscored.json",
                      file_hash(trial.material / "manifest.json"))


def test_interrupted_development_worker_records_hold_without_advancing(monkeypatch, trial):
    _fit(trial)
    _processes(monkeypatch, trial)

    def fail(*args, **kwargs):
        raise RuntimeError("deterministic fixture interruption")

    monkeypatch.setattr(verify, "_launch", fail)
    result = verify.assess(trial.base, trial.material, trial.out)
    assert result["reason"] == "incomplete_independent_verifier" and result["disposition"] == "HOLD"
    assert not result["winner"] and not result["selection"] and not result["hidden_opened"]
    lock = verify._read(trial.out / result["active_lock"])
    assert not verify.replay_lock(trial.out, trial.material, lock)["eligible"]


def test_changed_fitting_source_is_rejected_before_evaluation(monkeypatch, trial):
    _fit(trial)
    monkeypatch.setattr(verify, "source_hashes", lambda **kwargs: {"changed-source": "b" * 64})
    with pytest.raises(ValueError, match="bind source"):
        verify.validate_fit_report(trial.out, trial.material)


def test_inherited_verifier_globals_remain_isolated(monkeypatch):
    from experiments.phase_iiin import verification as prior
    before = prior.PROTOCOL
    assert before != verify.PROTOCOL
    monkeypatch.setattr(verify, "PROTOCOL", Path("fixture-protocol.json"))
    assert prior.PROTOCOL == before
    assert verify.assess.__globals__ is vars(verify)
    assert prior.assess.__globals__ is vars(prior)


@pytest.mark.parametrize("kind", ["measured", "predicted", "missing"])
def test_completed_endpoint_cannot_hide_an_uncorrected_individual_fork(trial, kind):
    _fit(trial)
    arm_path = trial.out / "arm-A1.json"
    record = verify._read(arm_path)
    if kind == "measured":
        record["probe"]["native_measurement"]["targets"]["rows"][0]["measured_margin"] = -.01
    elif kind == "predicted":
        record["left_factor"] = [.01]
    else:
        record["probe"]["native_measurement"]["targets"]["rows"] = []
    # Deliberately forge a fixture artifact, bypassing write-once creation.
    arm_path.write_text(json.dumps(record), encoding="utf-8")
    fit_path = trial.out / "fit-report.json"
    fit = verify._read(fit_path)
    fit["artifacts"][arm_path.name] = file_hash(arm_path)
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    with pytest.raises(ValueError, match="every individually corrected"):
        verify.validate_fit_report(trial.out, trial.material)


def test_candidate_cannot_use_an_unbound_basis_file(trial):
    _fit(trial)
    path = trial.out / "fit-report.json"
    report = verify._read(path)
    report["artifacts"].pop("basis.json")
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hash-bound basis"):
        verify.validate_fit_report(trial.out, trial.material)
