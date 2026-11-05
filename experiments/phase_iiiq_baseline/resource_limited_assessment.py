"""Host addendum for a pre-evaluation resource cutoff in the baseline study."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import FunctionType

from weightbridge.core import file_hash, json_dump_atomic
from . import study
from .integrity import PARENT, read, verify_sources


SCHEMA = "weightbridge.supervised_baseline.resource_continuation.v1"
PROCESS_SCHEMA = "weightbridge.supervised_baseline.training_process_tree.v1"


def _identity(protocol, authorization, bundle, smoke_gate):
    return {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization),
        "bundle_sha256": file_hash(bundle), "smoke_gate_sha256": file_hash(smoke_gate), "parent_sha256": PARENT}


def _process(path, root, identity, index, *, legacy_pretraining=False):
    record = read(path)
    expected_attempt = root.name if legacy_pretraining else None
    if (record.get("schema") != PROCESS_SCHEMA or record.get("attempt_directory", expected_attempt) != root.name
            or record.get("run_index") != index or any(record.get(key) != value for key, value in identity.items())
            or record.get("production_authority") is not False or record.get("lineage_authority") is not False
            or record.get("frozen_training_sources_modified") is not False
            or record.get("log_sha256") != file_hash(root.parent/f"{root.name}.log")):
        raise ValueError("Training process receipt is not bound to this frozen attempt")
    return record


def _cutoff(config, failed_run, process_path, identity):
    record = _process(process_path, failed_run, identity, 0)
    execution, checks = record["execution"], record["checks"]
    if (record.get("passed") is not False or record.get("status") != "HOLD"
            or execution.get("budget_failure") != "whole_tree_wall_budget" or not execution.get("exit_code")
            or checks.get("completed_training") is not False or checks.get("frozen_sources_unchanged") is not True
            or checks.get("whole_tree_wall_within_limit") is not False
            or execution["resources"]["wall_seconds"] <= record["limits"]["wall_seconds"]
            or record.get("training_completed_receipt_sha256") is not None
            or (failed_run/"completed.json").exists() or any((failed_run/"checkpoints").glob("*"))):
        raise ValueError("Only a terminal wall-budget cutoff without saved checkpoints can narrow screening")
    started = read(failed_run/"started.json")
    if (started.get("smoke") is not False or started.get("run_index") != 0
            or any(started.get(key) != identity[key] for key in
                ("protocol_sha256", "authorization_sha256", "bundle_sha256", "parent_sha256"))):
        raise ValueError("Interrupted worker start identity differs from its process receipt")
    steps = sorted((failed_run/"steps").glob("*.json"))
    if not 0 < len(steps) < min(config["training"]["checkpoint_steps"]) or [path.name for path in steps] != [
            f"{number:04d}.json" for number in range(1, len(steps)+1)]:
        raise ValueError("Interrupted worker lacks a contiguous pre-checkpoint step history")
    last = read(steps[-1])
    if last.get("step") != len(steps) or last["counts"]["optimizer_steps"] != len(steps):
        raise ValueError("Interrupted optimizer step count differs from its durable receipts")
    return {"run_index": 0, "attempt_directory": failed_run.name, "status": "WALL_BUDGET_HOLD",
        "candidate_outcomes": None, "completed_steps": len(steps), "last_durable_counts": last["counts"],
        "process_receipt_sha256": file_hash(process_path), "log_sha256": record["log_sha256"],
        "started_sha256": file_hash(failed_run/"started.json"),
        "step_sha256": {path.name: file_hash(path) for path in steps}, "resources": execution["resources"]}


def _survivor(run, process_path, identity, config, manifest_sha):
    run = run.resolve(strict=True)
    process = _process(process_path, run, identity, 1)
    complete_path = run/"completed.json"
    complete_sha = file_hash(complete_path)
    checks = {"worker_exit_success", "no_tree_budget_refusal", "complete_observed_process_cpu",
        "completed_training", "completed_identity", "worker_cpu_covered", "worker_peak_covered",
        "frozen_sources_unchanged", "whole_tree_wall_within_limit", "whole_tree_cpu_within_limit",
        "conservative_tree_memory_within_limit"}
    if (process.get("passed") is not True or process.get("status") != "COMPLETED"
            or process["execution"].get("exit_code") != 0 or process["execution"].get("budget_failure") is not None
            or any(process["checks"].get(key) is not True for key in checks)
            or process.get("training_completed_receipt_sha256") != complete_sha):
        raise ValueError("Only a complete process-tree-qualified surviving run supplies candidates")
    complete = read(complete_path)
    binding = {key: identity[key] for key in ("protocol_sha256", "authorization_sha256", "bundle_sha256", "parent_sha256")}
    binding.update(manifest_sha256=manifest_sha, run_index=1, smoke=False)
    if (any(complete.get(key) != value for key, value in binding.items()) or complete.get("status") != "COMPLETED"
            or complete.get("base_unchanged") is not True
            or complete["counts"]["optimizer_steps"] != config["training"]["optimizer_steps"]):
        raise ValueError("Surviving training receipt differs from the frozen parent, run, or schedule")
    expected = [run/"checkpoints"/f"step-{step:04d}" for step in config["training"]["checkpoint_steps"]]
    if ([Path(path).resolve() for path in complete["checkpoints"]] != expected
            or set((run/"checkpoints").iterdir()) != set(expected)):
        raise ValueError("Surviving checkpoint artifacts differ from the frozen schedule")
    candidates = []
    for step, directory in zip(config["training"]["checkpoint_steps"], expected, strict=True):
        adapter, receipt_path = directory/"adapter.safetensors", directory/"receipt.json"
        receipt = read(receipt_path)
        host, norm = receipt["host_metadata"], receipt["effective_update"]["stored_update_norm"]
        if (any(host.get(key) != value for key, value in binding.items()) or host.get("step") != step
                or receipt["step"] != step or file_hash(adapter) != receipt["sha256"]
                or receipt.get("base_unchanged") is not True or receipt["configuration"]["checkpoint_sha256"] != PARENT
                or not math.isfinite(norm) or norm < 0):
            raise ValueError("Surviving checkpoint receipt differs from the original authorization")
        candidates.append({"id": f"run-1-step-{step:04d}", "run_index": 1, "step": step,
            "adapter": adapter, "directory": directory, "adapter_sha256": receipt["sha256"],
            "factor_sha256": receipt["factor_sha256"], "stored_update_norm": norm,
            "training_receipt_sha256": file_hash(receipt_path), "completed_run_sha256": complete_sha})
    return candidates


def assess(base, protocol, authorization, bundle, smoke_gate, material, addendum, failed_run, failed_process,
           run, process, out):
    """Screen both frozen checkpoints of the remaining run under the original behavioral gates."""
    base, protocol, authorization, bundle, smoke_gate, material, addendum, failed_run, failed_process, run, process, out = map(
        Path, (base, protocol, authorization, bundle, smoke_gate, material, addendum, failed_run, failed_process, run, process, out))
    study._host_output(base, out)
    sidecar = out.parent/f"{out.name}-resource-continuation.json"
    study._host_output(base, sidecar)
    if out.exists() or sidecar.exists():
        raise FileExistsError("Resource-limited screening and its binding receipt are write-once")
    if any(out.resolve().is_relative_to(path.resolve()) for path in (failed_run, run)):
        raise ValueError("Assessment output must stay outside training attempts")
    if run.name != "run-1" or run.parent.resolve() != failed_run.parent.resolve():
        raise ValueError("Only the originally authorized run-1 location may supply surviving candidates")
    config, auth, admitted = read(protocol), read(authorization), read(smoke_gate)
    verify_sources(config["source_sha256"])
    identity = _identity(protocol, authorization, bundle, smoke_gate)
    if (config.get("checkpoint_sha256") != PARENT or auth.get("parent_sha256") != PARENT
            or auth.get("training_bundle_sha256") != identity["bundle_sha256"]
            or auth.get("smoke_gate_sha256") != identity["smoke_gate_sha256"]
            or admitted.get("passed") is not True or admitted.get("protocol_sha256") != identity["protocol_sha256"]
            or config["training"]["learning_rates"] != [0.0001, 0.0003]
            or config["training"]["checkpoint_steps"] != [64, 128]
            or config["training"]["optimizer_steps"] != 128 or auth.get("max_runs") != 2):
        raise ValueError("Resource cutoff requires the original two-run authorization and frozen schedule")
    failed = _cutoff(config, failed_run, failed_process, identity)
    manifest_sha = file_hash(material/"manifest.json")
    frozen = read(addendum)
    expected = {"schema": SCHEMA, "protocol_sha256": identity["protocol_sha256"],
        "authorization_sha256": identity["authorization_sha256"], "bundle_sha256": identity["bundle_sha256"],
        "parent_sha256": PARENT, "material_manifest_sha256": manifest_sha, "active_run_index": 1,
        "failed_run_index": 0, "failed_attempt_directory": failed_run.name,
        "failed_completed_optimizer_steps": failed["completed_steps"],
        "failed_last_step_sha256": failed["step_sha256"][f"{failed['completed_steps']:04d}.json"],
        "failed_process_sha256": failed["process_receipt_sha256"], "failure": "whole_tree_wall_budget",
        "behavior_gates": config["gates"], "checkpoint_steps": [64, 128], "optimizer_steps": 128,
        "learning_rate": 0.0003, "data_and_behavior_gates_changed": False, "two_rate_comparison_available": False,
        "independent_populations_opened": False, "new_training_attempt_index0_authorized": False,
        "prospective_no_development_accuracy_seen": True}
    if (any(frozen.get(key) != value for key, value in expected.items())
            or read(failed_run/"started.json").get("manifest_sha256") != manifest_sha):
        raise ValueError("Prospective addendum differs from the interrupted attempt or original gates")
    candidates = _survivor(run, process, identity, config, manifest_sha)
    earlier = failed_run.parent/"run-0-process.json"
    pretraining_failure = None
    if earlier.is_file() and earlier.resolve() != failed_process.resolve():
        previous = _process(earlier, failed_run.parent/"run-0", identity, 0, legacy_pretraining=True)
        if (previous.get("passed") is not False or previous.get("training_completed_receipt_sha256") is not None
                or (failed_run.parent/"run-0").exists()):
            raise ValueError("Earlier invocation has unexpected model training evidence")
        pretraining_failure = {"process_receipt_sha256": file_hash(earlier), "log_sha256": previous["log_sha256"],
            "resources": previous["execution"]["resources"], "status": "PRETRAINING_PROCESS_FAILURE"}
    binding = {"schema": "weightbridge.supervised_baseline.resource_assessment_binding.v1", **identity,
        "addendum_sha256": file_hash(addendum), "host_assessor_sha256": file_hash(__file__),
        "frozen_assessor_sha256": file_hash(study.__file__), "material_manifest_sha256": manifest_sha,
        "interrupted_run": failed, "earlier_pretraining_failure": pretraining_failure,
        "surviving_run_index": 1, "surviving_process_sha256": file_hash(process),
        "eligible_checkpoints": [row["id"] for row in candidates],
        "learning_rate_comparison": None, "additional_training_authority": False,
        "production_authority": False, "lineage_authority": False}
    json_dump_atomic(sidecar, binding)

    def authorized_checkpoints(run_dirs, protocol_path, manifest_hash):
        if (tuple(map(Path, run_dirs)) != (run,) or file_hash(protocol_path) != identity["protocol_sha256"]
                or manifest_hash != manifest_sha):
            raise ValueError("Screening input differs from the frozen resource continuation")
        return candidates

    # Keep the frozen assessor and all of its gates. Bind only the candidate
    # provider in a private namespace to avoid mutating the imported module.
    namespace = {**vars(study), "_checkpoints": authorized_checkpoints}
    frozen_assessor = FunctionType(study.assess.__code__, namespace)
    return frozen_assessor(base, protocol, material, out, [run])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "authorization", "bundle", "smoke-gate", "material", "addendum", "failed-run",
                 "failed-process", "run", "process", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = assess(**vars(args))
    print(result["reason"])
