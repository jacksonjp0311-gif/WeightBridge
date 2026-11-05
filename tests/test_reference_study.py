"""The external benchmark has its own data, frozen endpoint and no admission power."""
from decimal import Decimal
import json
import pytest
from experiments.phase_iiip import reference_study as study
from experiments.phase_iiip import reference_execute
from weightbridge.core import file_hash


def test_reference_population_is_exact_paired_and_in_disjoint_domain(tmp_path):
    out = tmp_path / "reference.json"
    study.freeze(out)
    frozen = json.loads(out.read_bytes())
    targets = [row for row in frozen["rows"] if row["group"] == "target"]
    assert len(frozen["rows"]) == 72 and len(targets) == 48
    assert len({row["pair_id"] for row in targets}) == 6
    assert all(abs(Decimal(value)) >= 1000 for row in frozen["rows"] for value in row["operands"])
    assert all(row["answer"] == max(row["operands"], key=Decimal) for row in targets)
    for family in {row["pair_id"] for row in targets}:
        rows = [row for row in targets if row["pair_id"] == family]
        assert {(row["counterfactual"], row["order"], row["wording"]) for row in rows} == {
            (cf, order, wording) for cf in (0, 1) for order in (0, 1) for wording in (0, 1)}
        assert {row["truth_precision"] for row in rows} == {"shorter", "longer"}
    with pytest.raises(FileExistsError):
        study.freeze(out)


def test_bad_roster_identity_is_refused_before_loading_any_model(tmp_path, monkeypatch):
    out = tmp_path / "reference.json"
    study.freeze(out)
    monkeypatch.setattr(study, "_open", lambda *a, **k: pytest.fail("model loaded before roster identity refusal"))
    with pytest.raises(ValueError, match="identity changed"):
        study.worker(tmp_path / "model", "a" * 64, out, tmp_path / "worker.json", "b" * 64)


def test_reference_endpoint_remains_greedy_and_never_claims_long_benchmark():
    config = json.loads(study.PROTOCOL.read_bytes())
    assert config["inference"]["max_new_tokens"] == 12
    assert config["budgets"]["models"] == 2
    assert config["budgets"]["paid_compute"] is False
    assert "Not a reproduction" in config["result_scope"]
    assert "Track A" in config["domain"]


def test_interrupted_reference_worker_retains_cost_without_inventing_behavior(tmp_path, monkeypatch):
    roster = tmp_path / "roster.json"
    study.freeze(roster)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic runtime interruption")
    monkeypatch.setattr(study, "_open", fail)
    out = tmp_path / "worker.json"
    with pytest.raises(RuntimeError):
        study.worker(tmp_path / "model", "a" * 64, roster, out, file_hash(roster))
    record = json.loads(out.with_name("worker.failure.json").read_bytes())
    assert record["behavioral_measurement"] is None and not out.exists()
    assert record["resources"]["cpu_seconds"] >= 0


def test_interrupted_reference_attempt_is_write_once(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_execute, "assess", lambda *a: (_ for _ in ()).throw(ValueError("fixture")))
    args = [tmp_path / name for name in ("base", "candidate", "roster", "reconstruction", "out", "execution.json")]
    with pytest.raises(ValueError):
        reference_execute.execute(*args)
    assert json.loads(args[-1].read_bytes())["status"] == "INTERRUPTED"
    with pytest.raises(FileExistsError):
        reference_execute.execute(*args)
