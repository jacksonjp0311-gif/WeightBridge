"""Host-stage fixtures; generated responses here are synthetic, not model evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.phase_iiis_tool_transfer import study as s
from experiments.phase_iiis_tool_transfer import verification as v
from experiments.phase_iiiq_baseline import data as qdata, measurement as qmeasure
from weightbridge.adaptation.behavior_endpoint import _observation
from weightbridge.core import file_hash
from test_tool_measurement import rows, measured, SETTINGS, END


DECIMAL_SETTINGS = {**SETTINGS, "max_input_tokens": 128, "max_new_tokens": 16}


def legacy(wrong=()):
    raw = qdata.generate("development", qdata.PopulationSpec(target_families=2, comparison_controls=2,
        copy_controls=1, addition_controls=1, successor_controls=1), "synthetic-host-only", set(), set())
    endpoint = qmeasure.endpoint(DECIMAL_SETTINGS)
    outputs = []
    for i, row in enumerate(raw):
        output = {**row, "text": " " + ("0" if i in wrong else row["answer"]) + "\n",
                  "stop_reason": "token_budget", "generated_token_ids": [8]*16, "prefix_tokens": 3}
        outputs.append({**output, **_observation(output, endpoint.question(row), endpoint)})
    return {"endpoint": endpoint.manifest(), "rows": outputs}


def bundle(tool_wrong=(), decimal_wrong=(), *, families=8):
    return {"tool": measured(rows(families), wrong=tool_wrong), "decimal": legacy(decimal_wrong)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def protocol(tmp_path, arm="original-joint"):
    return write(tmp_path / arm / "protocol.json", {"arm": arm, "checkpoint_sha256": s.PARENTS[arm],
        "source_sha256": {}, "task_manifest_sha256": "c"*64,
        "endpoints": {"tool": SETTINGS, "decimal": DECIMAL_SETTINGS}, "tool_completion": END,
        "authorized_arms": ["sequential", "original-joint"]})


def execution(path, protocol_path, out, *, passed=True):
    path = Path(path)
    log = path.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Synthetic owned execution fixture; no model process executed.")
    return write(path, {"schema": "weightbridge.phase_iiis.execution.v1", "passed": passed,
        "name": path.stem, "log_sha256": file_hash(log),
        "error": None if passed else {"type": "BudgetExceeded"},
        "command": ["python", "-m", "experiments.phase_iiis_tool_transfer.evaluate", "--protocol", str(protocol_path), "--out", str(out)],
        "limits": {"wall_limit": 10, "cpu_limit": 10, "memory_limit": 100},
        "temporary_sleep_request": {"requested": True, "succeeded": True, "released": True},
        "execution": {"exit_code": 0 if passed else 1, "budget_failure": None,
            "processes": [{"cpu_seconds": 1.5, "peak_rss_bytes": 5}],
            "resources": {"wall_seconds": 1, "cpu_seconds": 2, "peak_rss_bytes": 5,
                          "observed_tree_cpu_seconds": 1.5, "host_cpu_seconds": .5,
                          "cpu_accounting_complete_for_observed_processes": True}}})


def worker_specs(tmp_path, protocol_path, role, measurement, *, checkpoint, adapter=None, stage="development"):
    p = s.read(protocol_path)
    result = {}
    for task in s.SUITES:
        dest = tmp_path / role / task
        observed = measurement[task]
        raw = [{k: val for k, val in row.items() if k not in __import__(
            "experiments.phase_iiis_tool_transfer.measurement", fromlist=["_OUTPUT"])._OUTPUT}
            for row in observed["rows"]] if task == "tool" else qdata.generate("development",
                qdata.PopulationSpec(target_families=2, comparison_controls=2, copy_controls=1,
                    addition_controls=1, successor_controls=1), "synthetic-host-only", set(), set())
        roster = write(dest / "rows.json", raw)
        authorization = dest / "authorization.json"
        v.authorize_evaluation(authorization, protocol_path, stage=stage, task=task,
            roster_sha256=file_hash(roster), representation_sha256=checkpoint, adapter_sha256=adapter)
        record = write(dest / "worker.json", {"schema": v.SCHEMA, "stage": stage, "task": task,
            "checkpoint_sha256": checkpoint, "parent_sha256": p["checkpoint_sha256"], "adapter_sha256": adapter,
            "representation": "fp32_adapter_on_bf16_parent" if adapter else "stored_checkpoint",
            "protocol_sha256": file_hash(protocol_path), "roster_sha256": file_hash(roster),
            "task_manifest_sha256": p["task_manifest_sha256"], "source_sha256": {},
            "authorization_sha256": file_hash(authorization), "measurement": observed,
            "summary": s.summarize_tools(observed) if task == "tool" else qmeasure.summarize(observed)})
        process = execution(dest / "process.json", protocol_path, record)
        result[task] = {"path": str(record), "rows": str(roster), "authorization": str(authorization),
            "protocol": str(protocol_path), "stage": stage, "execution": str(process)}
    return result


def test_own_parent_gates_and_original_joint_not_forced_to_keep_r_gains():
    original = bundle(range(64), [0])
    r8 = bundle(range(64))
    candidate = bundle((), [0])
    assert s.behavior_gate("original-joint", original, r8, candidate)["passed"]
    gate = s.behavior_gate("sequential", original, r8, candidate)
    assert not gate["passed"]
    assert gate["disposition"] == "TOOL_TRANSFER_WITH_DECIMAL_REGRESSION"
    assert gate["inherited_decimal"]["response_denominator"] == 1
    assert gate["inherited_decimal"]["responses_retained"] == 0


def test_sequential_general_protects_union_even_when_r8_already_lost_one():
    original = bundle(range(64))
    general_index = next(i for i, r in enumerate(original["decimal"]["rows"]) if r["group"] != "target")
    r8 = bundle(range(64), [general_index])
    candidate = bundle((), [general_index])
    assert not s.behavior_gate("sequential", original, r8, candidate)["general"]["passed"]


def test_recovered_original_answers_are_separate_from_new_improvement():
    original = bundle([0])
    r8 = bundle([0, 1])
    candidate = bundle()
    result = s.recovery_report(original["tool"], r8["tool"], candidate["tool"])
    assert result["original_to_r8_losses"] == [rows()[1]["id"]]
    assert result["recovered_original_successes"] == [rows()[1]["id"]]
    assert result["corrected_beyond_original"] == [rows()[0]["id"]]


def test_formal_headroom_insufficiency_overrides_high_candidate_accuracy():
    original, r8, candidate = bundle(range(256), families=32), bundle([0], families=32), bundle(families=32)
    sequential = s.behavior_gate("sequential", original, r8, candidate, formal=True)
    control = s.behavior_gate("original-joint", original, r8, candidate, formal=True)
    assert not sequential["passed"]
    assert sequential["disposition"] == "INSUFFICIENT_IMPROVEMENT_HEADROOM"
    assert control["passed"]


def test_clustered_uncertainty_uses_family_units_and_is_deterministic():
    before, after = bundle(range(64)), bundle()
    a = s.paired_tool_interval(before["tool"], after["tool"], draws=100)
    assert a == s.paired_tool_interval(before["tool"], after["tool"], draws=100)
    assert a["families"] == 8
    assert a["strict_accuracy_gain_interval"] == [1., 1.]


def test_worker_bound_development_replay_and_step64_is_only_diagnostic(tmp_path):
    p = protocol(tmp_path)
    parent, candidate = bundle(range(64)), bundle()
    workers = {role: worker_specs(tmp_path, p, role, candidate if role == "candidate" else parent,
        checkpoint=s.DECIMAL_PARENT if role == "r8" else s.ORIGINAL,
        adapter="a"*64 if role == "candidate" else None) for role in s.ROLES}
    path = tmp_path / "gate.json"
    gate = s.record_stage(path, p, stage="development", workers=workers, candidate_checkpoint=s.ORIGINAL,
                          adapter_sha256="a"*64, step=64)
    assert gate["passed"] and not gate["finalist_eligible"]
    assert s.replay_stage(path) == gate
    tampered = deepcopy(gate)
    tampered["finalist_eligible"] = True
    write(path, tampered)
    with pytest.raises(ValueError, match="replay"):
        s.replay_stage(path)


def test_failed_enclosing_worker_cannot_be_hidden_by_passing_outcomes(tmp_path):
    p = protocol(tmp_path)
    specs = worker_specs(tmp_path, p, "candidate", bundle(), checkpoint=s.ORIGINAL, adapter="a"*64)
    process = s.read(specs["tool"]["execution"])
    process["execution"]["exit_code"] = 1
    write(specs["tool"]["execution"], process)
    with pytest.raises(ValueError, match="execution failed"):
        s.load_bundle(specs, p, expected_checkpoint=s.ORIGINAL, expected_adapter="a"*64)


def test_protected_stage_refuses_before_loading_any_bank(tmp_path, monkeypatch):
    p = protocol(tmp_path)
    monkeypatch.setattr(s, "load_bundle", lambda *args, **kwargs: pytest.fail("Opened unopened bank"))
    with pytest.raises(ValueError, match="prior stages"):
        s.record_stage(tmp_path / "gate.json", p, stage="qualification", workers={r: {} for r in s.ROLES},
            candidate_checkpoint="e"*64)
    assert not (tmp_path / "gate.json").exists()


def test_operational_arm_failure_locks_without_inventing_training_evidence(tmp_path):
    locks = {}
    for arm in s.PARENTS:
        p = protocol(tmp_path, arm)
        failed = execution(p.parent / "failed.json", p, p.parent / "out.json", passed=False)
        locks[arm] = p.parent / "lock.json"
        value = s.lock_operational_failure(locks[arm], p, failed, disposition="INTERRUPTED_INCOMPLETE")
        assert value["training_validation"] is None
        assert s.validate_arm_lock(locks[arm]) == value
    both = tmp_path / "both-arms.json"
    decision = s.lock_arms(both, locks)
    assert decision["passed"]  # Both terminal decisions are fixed; neither candidate passed.
    assert not decision["matched_training_observed_for_every_arm"]
    assert decision["matched_training"] is None
    assert s.validate_arm_locks(both) == decision


def test_cannot_omit_other_authorized_arm_or_mutate_its_failure(tmp_path):
    p = protocol(tmp_path)
    failed = execution(tmp_path / "failed.json", p, tmp_path / "out.json", passed=False)
    lock = tmp_path / "lock.json"
    s.lock_operational_failure(lock, p, failed, disposition="INTERRUPTED_INCOMPLETE")
    with pytest.raises(ValueError, match="omit"):
        s.lock_arms(tmp_path / "both.json", {"original-joint": lock}, expected_arms=("original-joint",))
    record = s.read(failed)
    record["passed"] = True
    write(failed, record)
    with pytest.raises(ValueError, match="changed"):
        s.validate_arm_lock(lock)


def test_missing_independent_stages_remain_null_not_pass(tmp_path, monkeypatch):
    p = protocol(tmp_path)
    lock = {"status": "FINALIST_LOCKED", "protocol_sha256": file_hash(p), "passed": True,
            "input_sha256": {}, "representation_sha256": "e"*64}
    monkeypatch.setattr(s, "validate_arm_lock", lambda path: lock)
    result = s.finalize(tmp_path / "final.json", p, tmp_path / "lock.json")
    assert not result["passed"]
    assert result["independent_stages"] == {"qualification": None, "confirmation": None}
    assert result["restoration"] is None
    assert result["native_replays"] == {}


def test_main_admission_requires_fixed_complete_exposure_and_128_checkpoint(tmp_path, monkeypatch):
    from experiments.phase_iiis_tool_transfer import initialize
    p = protocol(tmp_path)
    root = p.parent
    policy = write(root / "main-authorization.json", {})
    bundle_path = write(root / "training-bundle.json", {})
    admission = {"passed": True, "protocol_sha256": file_hash(p), "parent_sha256": s.ORIGINAL, "input_sha256": {}}
    monkeypatch.setattr(initialize, "validate_main_admission", lambda path: admission)
    training = root / "training"
    adapter = training / "checkpoints/step-0128/adapter.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"synthetic factor artifact; no training executed")
    identity = {"protocol_sha256": file_hash(p), "parent_sha256": s.ORIGINAL,
                "authorization_sha256": file_hash(policy), "bundle_sha256": file_hash(bundle_path)}
    checkpoint = write(adapter.with_name("receipt.json"), {"sha256": file_hash(adapter), "step": 128,
        "finalist_eligible_step": True, "configuration": {"checkpoint_sha256": s.ORIGINAL,
            "rank": 8, "alpha": 16., "dropout": 0., "seed": 606118},
        "host_metadata": {**identity, "step": 128}})
    completion = write(training / "completed.json", {"schema": "weightbridge.phase_iiis.training_completed.v1",
        **identity, "status": "COMPLETED", "mode": "main", "smoke": False, "run_index": 0,
        "base_unchanged": True, "initial_factor_sha256": "a"*64, "all_factors_changed": True,
        "all_factors_received_nonzero_gradient": True, "checkpoints": [str(adapter.parent)],
        "counts": {"optimizer_steps": 128, "forward_calls": 512, "backward_calls": 512, "examples_seen": 2048},
        "parameter_changes": [{"changed_entries": 1, "nonzero_gradient_steps": 1, "norm": .01}]*372})
    write(training / "inventory.json", {"trainable_parameters": 5411328, "trainable_tensors": [{}]*372,
        "initial_factor_sha256": "a"*64, "data_order_sha256": "b"*64,
        "ordered_row_ids_sha256": "c"*64, "tokenization_sha256": "d"*64})
    process = execution(root / "main-process.json", p, training)
    record = s.read(process)
    record["command"][2] = "experiments.phase_iiis_tool_transfer.train"
    record["command"] += ["--mode", "main"]
    write(process, record)
    result = s.validate_main_checkpoint(p, completion, process, adapter)
    assert result["passed"] and result["main_admission"] == admission
    changed = s.read(completion)
    changed["counts"]["forward_calls"] = 511
    write(completion, changed)
    with pytest.raises(ValueError, match="complete_exposure"):
        s.validate_main_checkpoint(p, completion, process, adapter)
    changed["counts"]["forward_calls"] = 512
    write(completion, changed)
    saved = s.read(checkpoint)
    saved["step"] = 64
    write(checkpoint, saved)
    with pytest.raises(ValueError, match="fixed_final_step"):
        s.validate_main_checkpoint(p, completion, process, adapter)
