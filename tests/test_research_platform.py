import json

import numpy as np
import pytest

from weightbridge.evaluation.metrics import paired_bootstrap_interval
from weightbridge.inference.huggingface import InferenceConfig, InferenceError
from weightbridge.memory import ExperienceStore, MemoryToWeightsConsolidation
from weightbridge.provenance import append_experiment, load_ledger
from weightbridge.provenance.ledger import LedgerError


def test_inference_configuration_enforces_host_budgets():
    InferenceConfig(max_new_tokens=32768, timeout_seconds=3600).validate()
    with pytest.raises(InferenceError, match="host budget"):
        InferenceConfig(max_new_tokens=32769).validate()
    with pytest.raises(InferenceError, match="nonnegative integer"):
        InferenceConfig(seed=True).validate()
    with pytest.raises(InferenceError, match="presence_penalty"):
        InferenceConfig(presence_penalty=6).validate()
    with pytest.raises(InferenceError, match="cpu_threads"):
        InferenceConfig(cpu_threads=257).validate()
    with pytest.raises(InferenceError, match="max_rss_bytes"):
        InferenceConfig(max_rss_bytes=1).validate()


def test_paired_bootstrap_is_deterministic_and_rejects_unpaired_data():
    before = [0, 1, 0, 1]
    after = [1, 1, 1, 1]
    first = paired_bootstrap_interval(before, after, seed=11, resamples=1000)
    second = paired_bootstrap_interval(before, after, seed=11, resamples=1000)
    assert first == second
    assert first["mean_difference"] == 0.5
    assert first["n"] == 4
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_interval([0], [0, 1])


def test_experiment_ledger_is_append_only_unique_and_correctable(tmp_path):
    path = tmp_path / "experiment_ledger.jsonl"
    original = {"experiment_id": "run-1", "disposition": "HOLD", "observation": "first"}
    append_experiment(path, original)
    with pytest.raises(LedgerError, match="Duplicate"):
        append_experiment(path, original)
    correction = {"experiment_id": "run-1-correction-1", "disposition": "PASS",
                  "supersedes": "run-1", "observation": "versioned correction"}
    append_experiment(path, correction)
    append_experiment(path, {"experiment_id": "run-2", "disposition": "PASS",
                             "measurement_key": "measurement-2"})
    with pytest.raises(LedgerError, match="measurement_key"):
        append_experiment(path, {"experiment_id": "run-2-copy", "disposition": "PASS",
                                 "measurement_key": "measurement-2"})
    records = load_ledger(path)
    assert [record["experiment_id"] for record in records] == ["run-1", "run-1-correction-1", "run-2"]
    assert records[1]["supersedes"] == "run-1"
    assert "recorded_at" in json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_schema_experiment_requires_identity_and_explicit_protocol_fields(tmp_path):
    path = tmp_path / "strict_ledger.jsonl"
    invalid = {"schema": "weightbridge.experiment.v1", "experiment_id": "incomplete", "disposition": "HOLD"}
    with pytest.raises(LedgerError, match="required fields"):
        append_experiment(path, invalid)
    complete = {
        "schema": "weightbridge.experiment.v1", "experiment_id": "complete",
        "measurement_key": "unique-run", "timestamp_utc": "2026-01-01T00:00:00Z",
        "hypothesis": "A checkpoint can be loaded.", "source_model": {"id": "toy", "sha256": "a" * 64},
        "proposed_modification": {"type": "none"}, "affected_parameters": {"count": 0},
        "evaluation": {"type": "checkpoint reload", "not_applicable": "no dataset or sampling"},
        "resources": {"wall_seconds": 0.1}, "artifact_root": "results/run",
        "artifacts": ["receipt.json"], "disposition": "PASS",
    }
    append_experiment(path, complete)
    assert load_ledger(path)[0]["experiment_id"] == "complete"


def test_mwc_store_accepts_synthetic_only_and_deduplicates_runs(tmp_path):
    store = ExperienceStore(tmp_path / "memory")
    workflow = MemoryToWeightsConsolidation(store)
    with pytest.raises(ValueError, match="synthetic"):
        workflow.consolidate(failure_fingerprint="fp", observation_id="private-run",
                             diagnosis="private", patch_sha256="a" * 64,
                             evaluation={}, data_class="private")
    record_id = workflow.consolidate(failure_fingerprint="fp", observation_id="synthetic-run-1",
                                     diagnosis="one-sided output bias", patch_sha256="a" * 64,
                                     evaluation={"accuracy": 1.0})
    assert len(record_id) == 64
    assert len(store.retrieve("fp")) == 1
    with pytest.raises(FileExistsError):
        workflow.consolidate(failure_fingerprint="fp", observation_id="synthetic-run-1",
                             diagnosis="changed duplicate", patch_sha256="b" * 64,
                             evaluation={"accuracy": 0.0})


def test_mwc_experience_records_the_expected_lifecycle(tmp_path):
    store = ExperienceStore(tmp_path / "memory")
    workflow = MemoryToWeightsConsolidation(store)
    workflow.consolidate(failure_fingerprint="fp", observation_id="run-2",
                         diagnosis="one-sided output bias", patch_sha256="b" * 64,
                         evaluation={"accuracy": 1.0})
    record = store.retrieve("fp")[0]
    assert record["workflow"] == ["execution_failure", "immutable_evidence", "pattern_analysis",
                                  "neural_diagnosis", "candidate_edit", "independent_evaluation",
                                  "retention_or_rollback"]


def test_agent_worker_rejects_hidden_holdout_fields(tmp_path):
    from weightbridge.agents.worker import run
    task = {"schema": "weightbridge.agent_task.v1", "base_path": "unused",
            "practice_features": [], "practice_labels": [], "candidate_dir": "unused",
            "result_path": "unused", "iteration_budget": 1, "allowed_tensors": ["output_bias"],
            "holdout_answers": [1, 2, 3]}
    request = tmp_path / "task.json"
    request.write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid agent worker request"):
        run(request)

