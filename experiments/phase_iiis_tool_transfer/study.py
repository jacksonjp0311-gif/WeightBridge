"""Host-only Phase S decisions from complete, hash-validated worker records.

No inference or training is launched here. Fixed step 128 is the only finalist;
all arm decisions are locked before either protected bank may be opened.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from experiments.phase_iiiq_baseline import measurement as legacy
from weightbridge.core import file_hash
from weightbridge.inspection.safetensors import load_checkpoint
from .integrity import ORIGINAL, DECIMAL_PARENT, PARENTS, read, verify_sources, write_once, checked_sha256
from .measurement import _aligned, compare_tools, legacy_retention, summarize_tools
from .verification import validate_worker, exact_native_replay, _requirements

ROLES = ("original", "r8", "candidate")
SUITES = ("tool", "decimal")


def _command_value(command, flag):
    if command.count(flag) != 1 or command.index(flag)+1 >= len(command):
        raise ValueError("Owned worker command lacks one unambiguous bound argument")
    return command[command.index(flag)+1]


def validate_execution(path, *, module, out, protocol_path):
    """Recompute enclosing process success rather than trusting its top-level boolean."""
    record = read(path)
    execution, limits = record.get("execution") or {}, record.get("limits") or {}
    resources = execution.get("resources") or {}
    sleep = record.get("temporary_sleep_request") or {}
    command = record.get("command") or []
    checks = {"kind": record.get("schema") == "weightbridge.phase_iiis.execution.v1",
        "status": record.get("passed") is True and record.get("error") is None,
        "successful_exit": execution.get("exit_code") == 0 and execution.get("budget_failure") is None,
        "cpu_scope": resources.get("cpu_accounting_complete_for_observed_processes") is True,
        "sleep_release": sleep.get("requested") is False or
            sleep.get("requested") is True and sleep.get("succeeded") is True and sleep.get("released") is True,
        "module": _command_value(command, "-m") == module,
        "output_binding": Path(_command_value(command, "--out")).resolve() == Path(out).resolve(),
        "protocol_binding": Path(_command_value(command, "--protocol")).resolve() == Path(protocol_path).resolve()}
    processes = execution.get("processes") or []
    checks["tree_resource_replay"] = bool(processes) and (
        math.isclose(sum(p["cpu_seconds"] for p in processes), resources.get("observed_tree_cpu_seconds", -1), abs_tol=1e-7)
        and math.isclose(resources.get("host_cpu_seconds", -1)+resources.get("observed_tree_cpu_seconds", -1),
                         resources.get("cpu_seconds", -3), abs_tol=1e-7)
        and sum(p["peak_rss_bytes"] for p in processes) == resources.get("peak_rss_bytes"))
    log = Path(path).parent / f"{record.get('name')}.log"
    checks["owned_log_identity"] = log.is_file() and file_hash(log) == record.get("log_sha256")
    for metric, bound in (("wall_seconds", "wall_limit"), ("cpu_seconds", "cpu_limit"), ("peak_rss_bytes", "memory_limit")):
        value, limit = resources.get(metric), limits.get(bound)
        checks[metric] = isinstance(value, (int, float)) and isinstance(limit, (int, float)) and math.isfinite(value) and 0 <= value <= limit
    if not all(checks.values()):
        raise ValueError("Enclosing worker execution failed or differs: " + ", ".join(k for k, ok in checks.items() if not ok))
    return record


def legacy_parts(measured):
    target = {**measured, "rows": [r for r in measured["rows"] if r["group"] == "target"]}
    general = {**measured, "rows": [r for r in measured["rows"] if r["group"] != "target"]}
    if not target["rows"] or not general["rows"]:
        raise ValueError("Combined legacy suite must retain both negative comparisons and general controls")
    legacy.summarize(target)
    legacy.summarize(general)
    return target, general


def paired_tool_interval(parent, candidate, *, seed=606118, draws=2000):
    """Deterministic paired bootstrap over whole families, never eight independent forms."""
    if type(draws) is not int or not 100 <= draws <= 10000 or type(seed) is not int:
        raise ValueError("Uncertainty requires a frozen bounded bootstrap")
    families = defaultdict(list)
    for a, b in _aligned(parent, candidate):
        families[a["family_id"]].append((a["strict_correct"], b["strict_correct"]))
    values = np.array([[sum(b-a for a, b in forms)/len(forms),
                        int(all(b for a, b in forms))-int(all(a for a, b in forms))]
                       for _, forms in sorted(families.items())], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty((draws, 2))
    for i in range(draws):
        means[i] = values[rng.integers(0, len(values), len(values))].mean(axis=0)
    low, high = np.quantile(means, (.025, .975), axis=0)
    return {"method": "paired_family_percentile_bootstrap", "families": len(values), "draws": draws,
        "seed": seed, "confidence": .95, "strict_accuracy_gain": float(values[:, 0].mean()),
        "strict_accuracy_gain_interval": [float(low[0]), float(high[0])],
        "complete_family_rate_gain": float(values[:, 1].mean()),
        "complete_family_rate_gain_interval": [float(low[1]), float(high[1])],
        "limitations": "Conditional on this finite generated family population and one trained seed; not universal competence."}


def recovery_report(original, r8, candidate):
    first, second = _aligned(original, r8), _aligned(r8, candidate)
    recovered = []
    beyond = []
    original_lost = []
    for (a, b), (_, c) in zip(first, second, strict=True):
        if a["strict_correct"] and not b["strict_correct"] and c["strict_correct"]:
            recovered.append(c["id"])
        if not a["strict_correct"] and c["strict_correct"]:
            beyond.append(c["id"])
        if a["strict_correct"] and not c["strict_correct"]:
            original_lost.append(c["id"])
    return {"original_to_r8_gains": [b["id"] for a, b in first if not a["strict_correct"] and b["strict_correct"]],
        "original_to_r8_losses": [b["id"] for a, b in first if a["strict_correct"] and not b["strict_correct"]],
        "recovered_original_successes": recovered, "corrected_beyond_original": beyond,
        "final_losses_relative_original": original_lost,
        "scope": "Recovery of a response lost in R is distinct from correction beyond the original checkpoint."}


def inherited_decimal_report(original, r8, candidate):
    # Alignment and correctness are independently checked by the historical contract.
    legacy.compare(original, r8, min_strict_gain=0, min_family_gain=0)
    legacy.compare(r8, candidate, min_strict_gain=0, min_family_gain=0)
    gains = [c["id"] for a, b, c in zip(original["rows"], r8["rows"], candidate["rows"], strict=True)
             if not a["correct"] and b["correct"]]
    outcomes = {r["id"]: r["correct"] for r in candidate["rows"]}
    before, intermediate, after = map(legacy.summarize, (original, r8, candidate))
    families = sorted(set(intermediate["complete_family_ids"]) - set(before["complete_family_ids"]))
    retained = sorted(set(families) & set(after["complete_family_ids"]))
    return {"original_wrong_r8_correct_responses": gains, "response_denominator": len(gains),
        "responses_retained": sum(outcomes[key] for key in gains),
        "retained_fraction": sum(outcomes[key] for key in gains)/len(gains) if gains else None,
        "new_r8_complete_families": families, "family_denominator": len(families),
        "new_r8_families_retained": retained,
        "scope": "Fixed fresh-bank inherited-gain denominator; an empty or small denominator limits retention evidence."}


def behavior_gate(arm, original, r8, candidate, *, formal=False):
    if arm not in PARENTS or any(set(bundle) != set(SUITES) for bundle in (original, r8, candidate)):
        raise ValueError("Unknown arm or incomplete paired task suite")
    own = r8 if arm == "sequential" else original
    tool = compare_tools(own["tool"], candidate["tool"], formal=formal)
    odec, ogen = legacy_parts(original["decimal"])
    rdec, rgen = legacy_parts(r8["decimal"])
    cdec, cgen = legacy_parts(candidate["decimal"])
    decimal = legacy_retention([rdec if arm == "sequential" else odec], cdec)
    general = legacy_retention([ogen, rgen] if arm == "sequential" else [ogen], cgen)
    passed = all(g["passed"] for g in (tool, decimal, general))
    if tool["disposition"] == "INSUFFICIENT_IMPROVEMENT_HEADROOM":
        disposition = "INSUFFICIENT_IMPROVEMENT_HEADROOM"
    elif tool["passed"] and not decimal["passed"]:
        disposition = "TOOL_TRANSFER_WITH_DECIMAL_REGRESSION"
    elif passed:
        disposition = "BEHAVIOR_QUALIFIED_PENDING_REPLAY" if formal else "DEVELOPMENT_QUALIFIED"
    elif tool["strict_response_gain"] > 0 and not tool["corrected_schema_valid_semantic_errors"]:
        disposition = "FORMAT_IMPROVEMENT_WITHOUT_SEMANTIC_QUALIFICATION"
    else:
        disposition = "NO_QUALIFIED_CANDIDATE"
    return {"passed": passed, "disposition": disposition, "arm": arm, "formal": formal,
        "tool": tool, "decimal": decimal, "general": general,
        "tool_recovery": recovery_report(original["tool"], r8["tool"], candidate["tool"]),
        "inherited_decimal": inherited_decimal_report(odec, rdec, cdec),
        "tool_uncertainty": paired_tool_interval(own["tool"], candidate["tool"]),
        "production_authority": False, "lineage_authority": False}


def _protocol(path):
    protocol = read(path)
    if protocol.get("arm") not in PARENTS or protocol.get("checkpoint_sha256") != PARENTS[protocol["arm"]]:
        raise ValueError("Study arm uses another explicit training parent")
    verify_sources(protocol["source_sha256"])
    return protocol


def load_bundle(specs, protocol_path, *, expected_checkpoint, expected_adapter=None):
    """Worker specs contain receipt/rows/authorization/protocol/stage paths and hashes."""
    protocol = _protocol(protocol_path)
    if set(specs) != set(SUITES):
        raise ValueError("Every model requires both tool and combined legacy worker records")
    measurements, commitments = {}, {}
    for task, spec in specs.items():
        if set(spec) != {"path", "rows", "authorization", "protocol", "stage", "execution"}:
            raise ValueError("Unexpected or incomplete worker specification")
        worker_protocol = read(spec["protocol"])
        if any(worker_protocol.get(key) != protocol.get(key) for key in
               ("task_manifest_sha256", "source_sha256", "endpoints", "tool_completion")):
            raise ValueError("Cached worker used different sources, populations or endpoint semantics")
        result = validate_worker(spec["path"], spec["rows"], spec["protocol"], expected_checkpoint,
            expected_adapter, spec["stage"], task, authorization=spec["authorization"])
        validate_execution(spec["execution"], module="experiments.phase_iiis_tool_transfer.evaluate",
                           out=spec["path"], protocol_path=spec["protocol"])
        measurements[task] = result["measurement"]
        for name in ("path", "rows", "authorization", "protocol", "execution"):
            commitments[f"{task}/{name}"] = {"path": str(Path(spec[name]).resolve()), "sha256": file_hash(spec[name])}
    return measurements, commitments


def _worker_stage_scope(workers, stage):
    expected_candidate = {"development": {"development"}, "merged-development": {"merged-development"},
                          "qualification": {"qualification"}, "confirmation": {"confirmation"}}[stage]
    for role, bundle in workers.items():
        allowed = expected_candidate if role == "candidate" else ({"development"} if stage == "merged-development" else {stage})
        if any(spec["stage"] not in allowed for spec in bundle.values()):
            raise ValueError("A worker from a later or different stage cannot supply this decision")


def _assert_hashes(commitments):
    for item in commitments.values():
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Recorded decision evidence changed")


def validate_prerequisites(protocol_path, stage, lock_path, prerequisite_paths):
    """Run before accessing any protected roster or measurement record."""
    protocol = _protocol(protocol_path)
    required = _requirements(stage)
    if set(prerequisite_paths or {}) != required or lock_path is None:
        raise ValueError("Protected stage lacks locked and completed prior stages")
    lock = validate_arm_lock(lock_path)
    if (lock.get("status") != "FINALIST_LOCKED" or lock.get("passed") is not True
            or lock.get("protocol_sha256") != file_hash(protocol_path)
            or lock.get("task_manifest_sha256") != protocol["task_manifest_sha256"]):
        raise ValueError("Protected stage requires this exact qualified stored finalist")
    _assert_hashes(lock["input_sha256"])
    commitments = {}
    for name, path in prerequisite_paths.items():
        value = read(path)
        if value.get("passed") is not True or value.get("task_manifest_sha256") != protocol["task_manifest_sha256"]:
            raise ValueError(f"Failed or missing prerequisite {name}")
        if name == "both-arms-locked":
            validate_arm_locks(path)
        else:
            if (value.get("protocol_sha256") != file_hash(protocol_path)
                    or value.get("representation_sha256") != lock["representation_sha256"]):
                raise ValueError("Prerequisite belongs to another finalized checkpoint")
            if value.get("schema") == "weightbridge.phase_iiis.behavior_gate.v1":
                replay_stage(path)
            elif "input_sha256" in value:
                _assert_hashes(value["input_sha256"])
        commitments[name] = {"path": str(Path(path).resolve()), "sha256": file_hash(path)}
    return lock, commitments


def record_stage(out, protocol_path, *, stage, workers, candidate_checkpoint, adapter_sha256=None,
                 step=128, lock_path=None, prerequisite_paths=None):
    protocol = _protocol(protocol_path)
    if stage not in ("development", "merged-development", "qualification", "confirmation"):
        raise ValueError("Unknown Phase S behavioral decision stage")
    if set(workers) != set(ROLES) or step not in (64, 128):
        raise ValueError("Fixed two diagnostic checkpoints and three model roles required")
    checked_sha256(candidate_checkpoint)
    prerequisites = None
    if stage in ("qualification", "confirmation"):
        lock, prerequisites = validate_prerequisites(protocol_path, stage, lock_path, prerequisite_paths)
        if adapter_sha256 is not None or candidate_checkpoint != lock["representation_sha256"] or step != 128:
            raise ValueError("Only the locked stored step-128 finalist reaches independent data")
    elif stage == "development" and (candidate_checkpoint != protocol["checkpoint_sha256"] or adapter_sha256 is None):
        raise ValueError("Adapter development must use its exact training parent and factors")
    elif stage == "merged-development" and (adapter_sha256 is not None or step != 128):
        raise ValueError("Only the fixed step-128 stored merge receives merged development qualification")
    _worker_stage_scope(workers, stage)
    workers = {role: {task: {key: str(value) for key, value in spec.items()}
                        for task, spec in bundle.items()} for role, bundle in workers.items()}
    bundles, commitments = {}, {}
    for role in ROLES:
        checkpoint = {"original": ORIGINAL, "r8": DECIMAL_PARENT, "candidate": candidate_checkpoint}[role]
        bundles[role], hashes = load_bundle(workers[role], protocol_path, expected_checkpoint=checkpoint,
            expected_adapter=adapter_sha256 if role == "candidate" else None)
        commitments.update({f"{role}/{key}": value for key, value in hashes.items()})
    gate = behavior_gate(protocol["arm"], bundles["original"], bundles["r8"], bundles["candidate"],
                         formal=stage in ("qualification", "confirmation"))
    result = {"schema": "weightbridge.phase_iiis.behavior_gate.v1", **gate,
        "protocol_sha256": file_hash(protocol_path), "protocol_path": str(Path(protocol_path).resolve()),
        "task_manifest_sha256": protocol["task_manifest_sha256"], "source_sha256": protocol["source_sha256"],
        "stage": stage, "step": step, "finalist_eligible": step == 128 and gate["passed"],
        "representation_sha256": candidate_checkpoint, "adapter_sha256": adapter_sha256,
        "workers": workers, "input_sha256": commitments, "prerequisites": prerequisites,
        "finalist_lock": {"path": str(Path(lock_path).resolve()), "sha256": file_hash(lock_path)} if lock_path else None}
    write_once(out, result)
    return result


def replay_stage(path):
    receipt = read(path)
    if receipt.get("schema") != "weightbridge.phase_iiis.behavior_gate.v1":
        raise ValueError("Expected a replayable Phase S behavior gate")
    protocol = _protocol(receipt["protocol_path"])
    if (file_hash(receipt["protocol_path"]) != receipt["protocol_sha256"]
            or protocol["task_manifest_sha256"] != receipt["task_manifest_sha256"]
            or protocol["source_sha256"] != receipt["source_sha256"]):
        raise ValueError("Frozen decision protocol changed")
    if receipt["stage"] in ("qualification", "confirmation"):
        lock = receipt.get("finalist_lock") or {}
        if not lock or file_hash(lock["path"]) != lock["sha256"]:
            raise ValueError("Independent stage's locked finalist changed")
        _assert_hashes(receipt.get("prerequisites") or {})
        validate_prerequisites(receipt["protocol_path"], receipt["stage"], lock["path"],
                              {key: value["path"] for key, value in receipt["prerequisites"].items()})
    _assert_hashes(receipt["input_sha256"])
    _worker_stage_scope(receipt["workers"], receipt["stage"])
    bundles = {}
    for role in ROLES:
        checkpoint = {"original": ORIGINAL, "r8": DECIMAL_PARENT, "candidate": receipt["representation_sha256"]}[role]
        bundles[role], _ = load_bundle(receipt["workers"][role], receipt["protocol_path"], expected_checkpoint=checkpoint,
            expected_adapter=receipt["adapter_sha256"] if role == "candidate" else None)
    gate = behavior_gate(protocol["arm"], bundles["original"], bundles["r8"], bundles["candidate"],
                         formal=receipt["stage"] in ("qualification", "confirmation"))
    if any(receipt.get(key) != value for key, value in gate.items()) or receipt["finalist_eligible"] != (receipt["step"] == 128 and gate["passed"]):
        raise ValueError("Behavioral admission does not replay from complete outcomes")
    return receipt


def validate_main_checkpoint(protocol_path, completed_path, execution_path, adapter_path):
    protocol = _protocol(protocol_path)
    from .initialize import validate_main_admission
    admission = validate_main_admission(Path(protocol_path).parent)
    completed, process = read(completed_path), read(execution_path)
    adapter_path = Path(adapter_path)
    validate_execution(execution_path, module="experiments.phase_iiis_tool_transfer.train",
                       out=Path(completed_path).parent, protocol_path=protocol_path)
    if _command_value(process["command"], "--mode") != "main":
        raise ValueError("A smoke or timing worker cannot supply the fixed main finalist")
    inventory = read(Path(completed_path).with_name("inventory.json"))
    saved = read(adapter_path.with_name("receipt.json"))
    checks = {"completed": completed.get("schema") == "weightbridge.phase_iiis.training_completed.v1"
        and completed.get("status") == "COMPLETED" and completed.get("mode") == "main"
        and completed.get("smoke") is False and completed.get("run_index") == 0,
        "protocol": completed.get("protocol_sha256") == file_hash(protocol_path),
        "parent": completed.get("parent_sha256") == protocol["checkpoint_sha256"],
        "base_unchanged": completed.get("base_unchanged") is True,
        "full_steps": completed.get("counts", {}).get("optimizer_steps") == 128,
        "complete_exposure": all(completed.get("counts", {}).get(key) == value for key, value in
            {"forward_calls": 512, "backward_calls": 512, "examples_seen": 2048}.items()),
        "inventory": inventory.get("trainable_parameters") == 5411328 and len(inventory.get("trainable_tensors", [])) == 372,
        "initial_factors": inventory.get("initial_factor_sha256") == completed.get("initial_factor_sha256"),
        "all_factor_gradients": completed.get("all_factors_received_nonzero_gradient") is True,
        "all_factor_changes": completed.get("all_factors_changed") is True,
        "enclosing_process": process.get("schema") == "weightbridge.phase_iiis.execution.v1"
            and process.get("passed") is True and process.get("error") is None
            and (process.get("execution") or {}).get("exit_code") == 0
            and (process.get("execution") or {}).get("budget_failure") is None,
        "fixed_final_step": saved.get("step") == 128 and saved.get("finalist_eligible_step") is True,
        "artifact": saved.get("sha256") == file_hash(adapter_path),
        "configuration": all(saved.get("configuration", {}).get(key) == value for key, value in
            {"checkpoint_sha256": protocol["checkpoint_sha256"], "rank": 8, "alpha": 16., "dropout": 0., "seed": 606118}.items()),
        "host_binding": saved.get("host_metadata", {}).get("protocol_sha256") == file_hash(protocol_path)
            and saved.get("host_metadata", {}).get("parent_sha256") == protocol["checkpoint_sha256"]
            and saved.get("host_metadata", {}).get("step") == 128,
        "recorded_checkpoint": str(adapter_path.parent.resolve()) in {str(Path(p).resolve()) for p in completed.get("checkpoints", [])}}
    arm_root = Path(protocol_path).parent
    for key, relative in (("authorization_sha256", "main-authorization.json"), ("bundle_sha256", "training-bundle.json")):
        checks[key] = completed.get(key) == file_hash(arm_root / relative) and saved.get("host_metadata", {}).get(key) == completed.get(key)
    checks["fixed_main_path"] = Path(completed_path).parent.resolve() == (arm_root / "training").resolve()
    checks["admitted_parent"] = admission["protocol_sha256"] == file_hash(protocol_path) and admission["parent_sha256"] == protocol["checkpoint_sha256"]
    changes = completed.get("parameter_changes") or []
    checks["measured_factor_updates"] = len(changes) == 372 and all(r.get("changed_entries", 0) > 0
        and r.get("nonzero_gradient_steps", 0) > 0 and np.isfinite(r.get("norm", float("nan"))) and r["norm"] > 0 for r in changes)
    if not all(checks.values()):
        raise ValueError("Step-128 finalist requires complete exact training and enclosing resource evidence: " +
                         ", ".join(key for key, passed in checks.items() if not passed))
    return {"passed": True, "checks": checks, "adapter_sha256": file_hash(adapter_path),
        "main_admission": admission,
        "matched_training": {key: inventory.get(key) for key in
            ("initial_factor_sha256", "data_order_sha256", "ordered_row_ids_sha256", "tokenization_sha256")},
        "input_sha256": {name: {"path": str(Path(path).resolve()), "sha256": file_hash(path)} for name, path in
            (("completion", completed_path), ("execution", execution_path), ("adapter", adapter_path),
             ("checkpoint_receipt", adapter_path.with_name("receipt.json")),
             ("inventory", Path(completed_path).with_name("inventory.json")))}}


def lock_arm(out, protocol_path, development_path, *, adapter_path, training_validation,
             merged_development_path=None, merge_verification_path=None, reconstruction_path=None):
    protocol = _protocol(protocol_path)
    development = replay_stage(development_path)
    if development["step"] != 128 or development["stage"] != "development":
        raise ValueError("Step 64 is diagnostic and can never become the finalist")
    _assert_hashes(training_validation["input_sha256"])
    if (not training_validation.get("passed") or training_validation["adapter_sha256"] != file_hash(adapter_path)
            or development["adapter_sha256"] != file_hash(adapter_path)):
        raise ValueError("Development candidate differs from the completed main training")
    paths = {"development": development_path, "adapter": adapter_path}
    passed, representation, disposition = False, None, development["disposition"]
    if development["finalist_eligible"]:
        if not all((merged_development_path, merge_verification_path, reconstruction_path)):
            raise ValueError("A development-qualified adapter must finish merged checks or receive an explicit operational failure record")
        merged = replay_stage(merged_development_path)
        merge, rebuilt = read(merge_verification_path), read(reconstruction_path)
        paths.update(merged_development=merged_development_path, merge_verification=merge_verification_path,
                     reconstruction=reconstruction_path)
        representation = merged["representation_sha256"]
        if (merged["stage"] != "merged-development" or merge.get("adapter_sha256") != file_hash(adapter_path)
                or merge.get("candidate_sha256") != representation or rebuilt.get("candidate_sha256") != representation
                or merge.get("parent_checkpoint_sha256") != protocol["checkpoint_sha256"]
                or rebuilt.get("parent_checkpoint_sha256") != protocol["checkpoint_sha256"]):
            raise ValueError("Stored finalist or reconstruction differs from the actual training parent")
        passed = (merged["finalist_eligible"] and merge.get("passed") is True and rebuilt.get("passed") is True
                  and rebuilt.get("exact_reconstruction") is True)
        disposition = merged["disposition"] if not merged["passed"] else "MERGE_OR_RECONSTRUCTION_FAILURE" if not passed else "FINALIST_LOCKED_PENDING_INDEPENDENT"
    result = {"schema": "weightbridge.phase_iiis.arm_lock.v1", "arm": protocol["arm"],
        "protocol_sha256": file_hash(protocol_path), "protocol_path": str(Path(protocol_path).resolve()),
        "task_manifest_sha256": protocol["task_manifest_sha256"], "source_sha256": protocol["source_sha256"],
        "passed": passed, "status": "FINALIST_LOCKED" if passed else "TERMINAL_FAILURE_LOCKED",
        "disposition": "FINALIST_LOCKED_PENDING_INDEPENDENT" if passed else disposition,
        "step": 128, "adapter_sha256": file_hash(adapter_path), "representation_sha256": representation,
        "training_validation": training_validation,
        "input_sha256": {name: {"path": str(Path(path).resolve()), "sha256": file_hash(path)} for name, path in paths.items()},
        "production_authority": False, "lineage_authority": False}
    write_once(out, result)
    return result


def lock_operational_failure(out, protocol_path, execution_path, *, disposition):
    """A failed owned execution can close one arm, but cannot qualify any model."""
    protocol, execution = _protocol(protocol_path), read(execution_path)
    if disposition not in {"INTERRUPTED_INCOMPLETE", "RESOURCE_INFEASIBLE", "TRAINING_PIPELINE_FAILURE"}:
        raise ValueError("Unknown terminal operational disposition")
    if (execution.get("schema") != "weightbridge.phase_iiis.execution.v1" or execution.get("passed") is not False
            or execution.get("error") is None or Path(_command_value(execution.get("command", []), "--protocol")).resolve() != Path(protocol_path).resolve()):
        raise ValueError("Operational terminal record requires this arm's failed owned execution")
    result = {"schema": "weightbridge.phase_iiis.arm_lock.v1", "arm": protocol["arm"],
        "protocol_sha256": file_hash(protocol_path), "protocol_path": str(Path(protocol_path).resolve()),
        "task_manifest_sha256": protocol["task_manifest_sha256"], "source_sha256": protocol["source_sha256"],
        "status": "TERMINAL_FAILURE_LOCKED", "passed": False, "terminal_kind": "operational",
        "disposition": disposition, "step": None, "adapter_sha256": None, "representation_sha256": None,
        "training_validation": None, "input_sha256": {"failed_execution": {
            "path": str(Path(execution_path).resolve()), "sha256": file_hash(execution_path)}},
        "production_authority": False, "lineage_authority": False}
    write_once(out, result)
    return result


def lock_arms(out, locks, *, expected_arms=("sequential", "original-joint")):
    if set(locks) != set(expected_arms) or set(expected_arms) - PARENTS.keys() or not expected_arms:
        raise ValueError("Every prospectively authorized arm must have a finalist or terminal failure")
    entries, manifest, matched_training = {}, None, None
    for arm, path in locks.items():
        value = validate_arm_lock(path)
        if (value.get("schema") != "weightbridge.phase_iiis.arm_lock.v1" or value.get("arm") != arm
                or value.get("status") not in ("FINALIST_LOCKED", "TERMINAL_FAILURE_LOCKED")
                or value.get("passed") != (value["status"] == "FINALIST_LOCKED")):
            raise ValueError("Arm decision has not been completed and locked")
        _assert_hashes(value["input_sha256"])
        if value["training_validation"] is not None:
            _assert_hashes(value["training_validation"]["input_sha256"])
        protocol = _protocol(value["protocol_path"])
        if set(protocol.get("authorized_arms", PARENTS)) != set(expected_arms):
            raise ValueError("All-arm boundary cannot omit a prospectively authorized arm")
        if file_hash(value["protocol_path"]) != value["protocol_sha256"]:
            raise ValueError("Locked arm protocol changed")
        if manifest is not None and manifest != value["task_manifest_sha256"]:
            raise ValueError("Arms do not share one frozen population")
        manifest = value["task_manifest_sha256"]
        configuration = (value["training_validation"] or {}).get("matched_training")
        if value["training_validation"] is not None and (configuration is None or any(v is None for v in configuration.values())):
            raise ValueError("Matched training identities are incomplete")
        if matched_training is not None and configuration is not None and matched_training != configuration:
            raise ValueError("Arms did not share adapter initialization and ordered exposure")
        matched_training = configuration if configuration is not None else matched_training
        entries[arm] = {"path": str(Path(path).resolve()), "sha256": file_hash(path), "status": value["status"],
                        "matched_training": configuration}
    result = {"schema": "weightbridge.phase_iiis.all_arms_locked.v1", "passed": True,
        "task_manifest_sha256": manifest, "arms": entries, "expected_arms": list(expected_arms),
        "matched_training": matched_training,
        "matched_training_observed_for_every_arm": all(v["matched_training"] is not None for v in entries.values()),
        "scope": "All arm decisions are fixed; this does not mean every arm qualified.",
        "production_authority": False, "lineage_authority": False}
    write_once(out, result)
    return result


def validate_arm_locks(path):
    record = read(path)
    if (record.get("schema") != "weightbridge.phase_iiis.all_arms_locked.v1" or record.get("passed") is not True
            or set(record.get("arms", {})) != set(record.get("expected_arms", [])) or not record.get("arms")):
        raise ValueError("All prospectively admitted arms must be locked before protected access")
    for arm, commitment in record["arms"].items():
        if file_hash(commitment["path"]) != commitment["sha256"]:
            raise ValueError("A locked arm changed after the shared evaluation boundary")
        lock = validate_arm_lock(commitment["path"])
        if lock.get("arm") != arm or lock.get("status") != commitment["status"] or lock.get("task_manifest_sha256") != record["task_manifest_sha256"]:
            raise ValueError("Arm lock identity differs")
        _assert_hashes(lock["input_sha256"])
        if lock["training_validation"] is not None:
            _assert_hashes(lock["training_validation"]["input_sha256"])
        protocol = _protocol(lock["protocol_path"])
        if (file_hash(lock["protocol_path"]) != lock["protocol_sha256"]
                or set(protocol.get("authorized_arms", PARENTS)) != set(record["expected_arms"])
                or (lock["training_validation"] or {}).get("matched_training") != commitment["matched_training"]):
            raise ValueError("All-arm protocol or matched exposure commitment changed")
    return record


def validate_arm_lock(path):
    """Recompute lock eligibility; changing a boolean or rewriting its hash is insufficient."""
    value = read(path)
    if value.get("schema") != "weightbridge.phase_iiis.arm_lock.v1":
        raise ValueError("Expected a Phase S arm lock")
    protocol = _protocol(value["protocol_path"])
    if (value.get("protocol_sha256") != file_hash(value["protocol_path"])
            or value.get("task_manifest_sha256") != protocol["task_manifest_sha256"]
            or value.get("arm") != protocol["arm"]):
        raise ValueError("Arm lock changed its fixed model, protocol or finalist step")
    _assert_hashes(value["input_sha256"])
    if value.get("terminal_kind") == "operational":
        execution = read(value["input_sha256"]["failed_execution"]["path"])
        if (value.get("passed") is not False or value.get("status") != "TERMINAL_FAILURE_LOCKED"
                or value.get("representation_sha256") is not None or value.get("adapter_sha256") is not None
                or value.get("training_validation") is not None or value.get("step") is not None
                or execution.get("schema") != "weightbridge.phase_iiis.execution.v1"
                or execution.get("passed") is not False or execution.get("error") is None
                or Path(_command_value(execution.get("command", []), "--protocol")).resolve() != Path(value["protocol_path"]).resolve()):
            raise ValueError("Operational terminal record does not reproduce a failed owned execution")
        return value
    if value.get("step") != 128:
        raise ValueError("Only the fixed step 128 may supply a measured arm decision")
    training = value["training_validation"]
    _assert_hashes(training["input_sha256"])
    required = training["input_sha256"]
    recomputed = validate_main_checkpoint(value["protocol_path"], required["completion"]["path"],
        required["execution"]["path"], required["adapter"]["path"])
    if recomputed != training or value["adapter_sha256"] != recomputed["adapter_sha256"]:
        raise ValueError("Training admission does not replay from the recorded main worker")
    development = replay_stage(value["input_sha256"]["development"]["path"])
    if development["step"] != 128 or development["adapter_sha256"] != value["adapter_sha256"]:
        raise ValueError("Locked development candidate differs from the exact step-128 adapter")
    passed, representation = False, None
    if development["finalist_eligible"]:
        required = value["input_sha256"]
        if not {"merged_development", "merge_verification", "reconstruction"} <= required.keys():
            raise ValueError("Qualified adapter lacks its measured stored-model decision")
        merged = replay_stage(required["merged_development"]["path"])
        merge, rebuilt = read(required["merge_verification"]["path"]), read(required["reconstruction"]["path"])
        representation = merged["representation_sha256"]
        if (merge.get("candidate_sha256") != representation or rebuilt.get("candidate_sha256") != representation
                or merge.get("adapter_sha256") != value["adapter_sha256"]
                or any(r.get("parent_checkpoint_sha256") != protocol["checkpoint_sha256"] for r in (merge, rebuilt))
                or any(r.get("protocol_sha256") != value["protocol_sha256"] for r in (merge, rebuilt))):
            raise ValueError("Merge or reconstruction identity differs from the locked actual parent")
        passed = merged["finalist_eligible"] and merge.get("passed") is True and rebuilt.get("passed") is True and rebuilt.get("exact_reconstruction") is True
    if (value["passed"] != passed or value["status"] != ("FINALIST_LOCKED" if passed else "TERMINAL_FAILURE_LOCKED")
            or value["representation_sha256"] != representation):
        raise ValueError("Finalist lock does not reproduce its actual behavioral and storage gates")
    return value


def finalize(out, protocol_path, lock_path, *, qualification_path=None, confirmation_path=None,
             reconstruction_path=None, replays=None, original=None, actual_parent=None):
    """Exact full replay receipts and current checkpoint bytes remain required for PASS."""
    protocol, lock = _protocol(protocol_path), validate_arm_lock(lock_path)
    if (lock.get("status") != "FINALIST_LOCKED" or lock.get("protocol_sha256") != file_hash(protocol_path)
            or lock.get("passed") is not True):
        raise ValueError("Final verification requires this arm's locked stored finalist")
    _assert_hashes(lock["input_sha256"])
    stages, recorded_stages = {}, {}
    for stage, path in (("qualification", qualification_path), ("confirmation", confirmation_path)):
        if path is None:
            stages[stage] = None
            continue
        value = replay_stage(path)
        if value["stage"] != stage or value["representation_sha256"] != lock["representation_sha256"]:
            raise ValueError("Final verification received another stage or checkpoint")
        stages[stage] = {"passed": value["passed"], "sha256": file_hash(path), "disposition": value["disposition"]}
        recorded_stages[stage] = value
    expected = {f"{stage}/{role}/{task}" for stage in ("qualification", "confirmation")
                for role in ("candidate", "parent") for task in SUITES}
    native = {}
    for key, pair in (replays or {}).items():
        if key not in expected or set(pair) != {"recorded", "replayed"}:
            raise ValueError("Unknown or incomplete native replay identity")
        stage, role, task = key.split("/")
        if stage not in recorded_stages:
            raise ValueError("A replay cannot open a stage that lacks its original completed evaluation")
        reference_role = "candidate" if role == "candidate" else "r8" if protocol["arm"] == "sequential" else "original"
        if pair["recorded"] != recorded_stages[stage]["workers"][reference_role][task]:
            raise ValueError("Fresh replay reference differs from the exact independently scored worker")
        if (file_hash(pair["recorded"]["rows"]) != file_hash(pair["replayed"]["rows"])
                or file_hash(pair["recorded"]["protocol"]) != file_hash(pair["replayed"]["protocol"])):
            raise ValueError("Fresh replay changed its original roster or protocol")
        checkpoint = lock["representation_sha256"] if role == "candidate" else protocol["checkpoint_sha256"]
        values = []
        for spec in (pair["recorded"], pair["replayed"]):
            if spec["stage"] != stage:
                raise ValueError("Fresh replay must reproduce its own previously opened stage")
            values.append(validate_worker(spec["path"], spec["rows"], spec["protocol"], checkpoint, None,
                stage, task, authorization=spec["authorization"])["measurement"])
            validate_execution(spec["execution"], module="experiments.phase_iiis_tool_transfer.evaluate",
                               out=spec["path"], protocol_path=spec["protocol"])
        if (Path(pair["recorded"]["path"]).resolve() == Path(pair["replayed"]["path"]).resolve()
                or Path(pair["recorded"]["execution"]).resolve() == Path(pair["replayed"]["execution"]).resolve()):
            raise ValueError("A fresh replay cannot reuse its original worker or process record")
        native[key] = {"passed": exact_native_replay(*values),
            "recorded_sha256": file_hash(pair["recorded"]["path"]), "replayed_sha256": file_hash(pair["replayed"]["path"])}
    reconstructed = read(reconstruction_path) if reconstruction_path is not None else None
    if reconstruction_path is not None and file_hash(reconstruction_path) != lock["input_sha256"]["reconstruction"]["sha256"]:
        raise ValueError("Final verification reconstruction differs from the locked exact rebuild")
    restoration = None
    if original is not None and actual_parent is not None:
        restoration = {"original_sha256": load_checkpoint(original).checkpoint_sha256(),
                       "actual_parent_sha256": load_checkpoint(actual_parent).checkpoint_sha256()}
        restoration["passed"] = restoration["original_sha256"] == ORIGINAL and restoration["actual_parent_sha256"] == protocol["checkpoint_sha256"]
    checks = {"independent_gates": all((v or {}).get("passed") is True for v in stages.values()),
        "all_native_replays": set(native) == expected and all(r["passed"] for r in native.values()),
        "exact_reconstruction": (reconstructed or {}).get("passed") is True
            and (reconstructed or {}).get("exact_reconstruction") is True
            and (reconstructed or {}).get("candidate_sha256") == lock["representation_sha256"],
        "immutable_parent_restoration": (restoration or {}).get("passed") is True}
    passed = all(checks.values())
    result = {"schema": "weightbridge.phase_iiis.final_verification.v1", "passed": passed, "checks": checks,
        "protocol_sha256": file_hash(protocol_path), "task_manifest_sha256": protocol["task_manifest_sha256"],
        "arm": protocol["arm"], "parent_sha256": protocol["checkpoint_sha256"],
        "representation_sha256": lock["representation_sha256"], "independent_stages": stages,
        "native_replays": native, "restoration": restoration,
        "reconstruction_sha256": file_hash(reconstruction_path) if reconstruction_path else None,
        "disposition": "VERIFIED_TOOL_TRANSFER_AND_RETENTION" if passed else "DEVELOPMENT_GAIN_WITHOUT_SEALED_CONFIRMATION",
        "lineage_accepted": False, "production_authority": False}
    write_once(out, result)
    return result
