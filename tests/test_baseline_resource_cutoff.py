"""Synthetic host receipts for the resource-only continuation boundary."""
import json
import pytest

from experiments.phase_iiiq_baseline import resource_limited_assessment as cutoff, study
from experiments.phase_iiiq_baseline.integrity import PARENT
from test_baseline_stages import processes, trial  # Reuse the frozen-stage fixtures.
from weightbridge.core import file_hash, json_dump_atomic


@pytest.fixture
def continuation(trial):
    root, protocol = trial.root, trial.protocol
    auth, bundle, smoke = (root/name for name in ("authorization.json", "bundle.json", "smoke.json"))
    json_dump_atomic(bundle, {"fixture": "training only"})
    json_dump_atomic(smoke, {"passed": True, "protocol_sha256": file_hash(protocol)})
    json_dump_atomic(auth, {"parent_sha256": PARENT, "max_runs": 2, "training_bundle_sha256": file_hash(bundle),
        "smoke_gate_sha256": file_hash(smoke)})
    identity = cutoff._identity(protocol, auth, bundle, smoke)
    complete_identity = {key: identity[key] for key in ("protocol_sha256", "authorization_sha256", "bundle_sha256", "parent_sha256")}
    manifest_sha = file_hash(trial.material/"manifest.json")
    interrupted, run = root/"run-0-retry1", trial.runs[1]
    json_dump_atomic(interrupted/"started.json", {**complete_identity, "manifest_sha256": manifest_sha, "run_index": 0, "smoke": False})
    step = interrupted/"steps/0001.json"
    json_dump_atomic(step, {"step": 1, "counts": {"optimizer_steps": 1, "forward_calls": 4}})
    failed_process, process = root/"failed-process.json", root/"survived-process.json"

    def process_base(path, index):
        log = root/f"{path.name}.log"
        log.write_text("fixture", encoding="utf-8")
        return {"schema": cutoff.PROCESS_SCHEMA, **identity, "attempt_directory": path.name, "run_index": index,
            "production_authority": False, "lineage_authority": False, "frozen_training_sources_modified": False,
            "log_sha256": file_hash(log)}

    json_dump_atomic(failed_process, {**process_base(interrupted, 0), "passed": False, "status": "HOLD",
        "checks": {"completed_training": False, "frozen_sources_unchanged": True, "whole_tree_wall_within_limit": False},
        "execution": {"budget_failure": "whole_tree_wall_budget", "exit_code": 15,
            "resources": {"wall_seconds": 5401, "cpu_seconds": 11}}, "limits": {"wall_seconds": 5400},
        "training_completed_receipt_sha256": None})
    completed = run/"completed.json"
    completed.write_text(json.dumps({**study.read(completed), **complete_identity}), encoding="utf-8")
    for path in (run/"checkpoints").glob("*/receipt.json"):
        receipt = study.read(path)
        receipt["host_metadata"].update(complete_identity, smoke=False)
        path.write_text(json.dumps(receipt), encoding="utf-8")
    check_names = ("worker_exit_success", "no_tree_budget_refusal", "complete_observed_process_cpu", "completed_training",
        "completed_identity", "worker_cpu_covered", "worker_peak_covered", "frozen_sources_unchanged",
        "whole_tree_wall_within_limit", "whole_tree_cpu_within_limit", "conservative_tree_memory_within_limit")
    json_dump_atomic(process, {**process_base(run, 1), "passed": True, "status": "COMPLETED",
        "checks": dict.fromkeys(check_names, True), "execution": {"exit_code": 0, "budget_failure": None},
        "training_completed_receipt_sha256": file_hash(completed)})
    addendum = root/"continuation.json"
    json_dump_atomic(addendum, {"schema": cutoff.SCHEMA, **complete_identity, "material_manifest_sha256": manifest_sha,
        "active_run_index": 1, "failed_run_index": 0, "failed_attempt_directory": interrupted.name,
        "failed_completed_optimizer_steps": 1, "failed_last_step_sha256": file_hash(step),
        "failed_process_sha256": file_hash(failed_process), "failure": "whole_tree_wall_budget",
        "behavior_gates": trial.config["gates"], "checkpoint_steps": [64, 128], "optimizer_steps": 128,
        "learning_rate": .0003, "data_and_behavior_gates_changed": False, "two_rate_comparison_available": False,
        "independent_populations_opened": False, "new_training_attempt_index0_authorized": False,
        "prospective_no_development_accuracy_seen": True})
    return dict(base=trial.base, protocol=protocol, authorization=auth, bundle=bundle, smoke_gate=smoke,
        material=trial.material, addendum=addendum, failed_run=interrupted, failed_process=failed_process,
        run=run, process=process, out=trial.out)


@pytest.mark.parametrize("complete_family", [True, False])
def test_survivor_keeps_original_gates_and_only_screening_access(continuation, trial, monkeypatch, complete_family):
    calls = processes(monkeypatch, trial, development_family=complete_family)
    checkpoint_provider = study._checkpoints
    result = cutoff.assess(**continuation)
    assert study._checkpoints is checkpoint_provider
    assert set(result["candidates"]) == {"run-1-step-0064", "run-1-step-0128"}
    assert {stage for stage, _, _ in calls} == {"practice", "development"}
    assert result["hidden"] is result["selection"] is result["preservation_validation"] is None
    if complete_family:
        assert result["winner"] == "run-1-step-0064"
        assert study._replay_screening(trial.out, trial.protocol) == result
    else:
        assert result["winner"] is None
    receipt = study.read(trial.out.with_name(f"{trial.out.name}-resource-continuation.json"))
    assert receipt["interrupted_run"]["candidate_outcomes"] is None
    assert receipt["learning_rate_comparison"] is None


@pytest.mark.parametrize("tamper,expected", [("gates", "original gates"), ("cutoff", "terminal wall-budget"),
    ("survivor", "process-tree-qualified"), ("checkpoint", "checkpoint receipt")])
def test_changed_bindings_refuse_before_screening(continuation, monkeypatch, tamper, expected):
    if tamper == "gates":
        path = continuation["addendum"]
        value = study.read(path)
        value["behavior_gates"]["development"]["min_family_gain"] = 0
    elif tamper == "cutoff":
        path = continuation["failed_process"]
        value = study.read(path)
        value["execution"]["budget_failure"] = "whole_tree_cpu_budget"
    elif tamper == "survivor":
        path = continuation["process"]
        value = study.read(path)
        value["checks"]["worker_cpu_covered"] = False
    else:
        path = continuation["run"]/"checkpoints/step-0064/receipt.json"
        value = study.read(path)
        value["host_metadata"]["run_index"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(study, "_launch", lambda *args, **kwargs: pytest.fail("Unbound experiment cannot evaluate"))
    with pytest.raises(ValueError, match=expected):
        cutoff.assess(**continuation)
    assert not continuation["out"].exists()
