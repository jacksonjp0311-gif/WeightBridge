"""Prospective rank-specific admission around the unchanged Phase III-Q trainer.

No new optimizer or resume path is introduced. A launch claim is write-once,
including failed/interrupted launches, and excludes replacement training runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from weightbridge.adaptation.training_policy import TrainingAuthorization
from weightbridge.core import file_hash, json_dump_atomic
from experiments.phase_iiiq_baseline.integrity import PARENT, ROOT, read, verify_sources
from experiments.phase_iiiq_baseline.run_training import run as launch_q_training
from experiments.phase_iiiq_baseline.train import data_order


PARAMETERS = {8: 5_411_328, 4: 2_705_664}
MAIN_SETTINGS = {"learning_rates": [.0003], "optimizer_steps": 128, "checkpoint_steps": [64, 128],
    "seed": 606117, "effective_batch_size": 16, "micro_batch_size": 4, "epochs": 1,
    "optimizer": "AdamW", "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": .01,
    "gradient_clip": 1., "warmup_fraction": .05, "logits_mode": "tail",
    "scheduler": "linear warmup then linear decay", "norm_measurement_interval": 8}
SMOKE_SETTINGS = {**MAIN_SETTINGS, "learning_rates": [.001], "optimizer_steps": 32,
    "checkpoint_steps": [32], "seed": 606118, "effective_batch_size": 4, "epochs": 4,
    "norm_measurement_interval": 4}


def validate_recipe(config):
    """Reject an undeclared recipe before any training process starts."""
    adapter = config.get("adapter", {})
    rank = adapter.get("rank")
    if (config.get("schema") != "weightbridge.phase_iiir.protocol.v1"
            or config.get("checkpoint_sha256") != PARENT
            or rank not in PARAMETERS or config.get("replication_arm") != f"rank{rank}"
            or adapter.get("alpha") != rank*2 or adapter.get("dropout") != 0
            or adapter.get("base_dtype") != "bfloat16" or adapter.get("factor_dtype") != "float32"):
        raise ValueError("Phase III-R requires the exact original parent and declared rank configuration")
    for name, required in (("training", MAIN_SETTINGS), ("smoke_training", SMOKE_SETTINGS)):
        if any(config.get(name, {}).get(key) != value for key, value in required.items()):
            raise ValueError(f"Phase III-R {name} differs from the frozen replication recipe")
    limits = config.get("resources", {})
    required_limits = {"max_wall_seconds_per_run": 5400., "max_cpu_seconds_per_run": 30000.,
        "max_memory_bytes": 24*1024**3, "smoke_max_wall_seconds": 900., "smoke_max_cpu_seconds": 4500.,
        "max_answer_tokens_per_run": 40000, "max_input_tokens_per_run": 400000}
    if any(limits.get(key) != value for key, value in required_limits.items()):
        raise ValueError("Phase III-R resource limits differ from the declared budgets")
    inference = config.get("inference", {})
    for key, value in {"cpu_threads": 6, "batch_size": 8, "max_input_tokens": 128, "max_new_tokens": 16}.items():
        if inference.get(key) != value:
            raise ValueError("Phase III-R requires the unchanged complete-generation endpoint")
    return rank


def authorization(config, inventory, bundle, *, smoke=False, smoke_gate=None):
    """Use the existing narrow training policy with one run for this exact rank."""
    rank = validate_recipe(config)
    rows = tuple(inventory)
    if (len(rows) != 186 or len({row["name"] for row in rows}) != 186
            or {row["layer"] for row in rows} != set(range(24))
            or rank*sum(sum(row["weight_shape"]) for row in rows) != PARAMETERS[rank]):
        raise ValueError("Rank-specific authorization requires the measured 186-module architecture")
    settings = config["smoke_training" if smoke else "training"]
    return TrainingAuthorization(schema="weightbridge.supervised_training_authorization.v1", parent_sha256=PARENT,
        allowed_modules=tuple(row["name"] for row in rows), max_trainable_parameters=PARAMETERS[rank],
        rank=rank, alpha=2.*rank, dropout=0., learning_rates=tuple(settings["learning_rates"]), optimizer="AdamW",
        max_runs=1, max_steps_per_run=32 if smoke else 128,
        max_answer_tokens_per_run=10000 if smoke else 40000,
        max_input_tokens_per_run=50000 if smoke else 400000, max_memory_bytes=24*1024**3,
        max_wall_seconds_per_run=900. if smoke else 5400.,
        max_cpu_seconds_per_run=4500. if smoke else 30000.,
        max_development_checkpoints_per_run=1 if smoke else 2, allow_merge=not smoke,
        training_bundle_sha256=file_hash(bundle), smoke_gate_sha256=file_hash(smoke_gate) if smoke_gate else None)


def verify_authorization(config, auth, *, smoke):
    rank = validate_recipe(config)
    expected = {"parent_sha256": PARENT, "rank": rank, "alpha": rank*2., "dropout": 0.,
        "max_trainable_parameters": PARAMETERS[rank], "max_runs": 1, "max_steps_per_run": 32 if smoke else 128,
        "learning_rates": (.001,) if smoke else (.0003,), "max_answer_tokens_per_run": 10000 if smoke else 40000,
        "max_input_tokens_per_run": 50000 if smoke else 400000, "max_memory_bytes": 24*1024**3,
        "max_wall_seconds_per_run": 900. if smoke else 5400.,
        "max_cpu_seconds_per_run": 4500. if smoke else 30000.,
        "max_development_checkpoints_per_run": 1 if smoke else 2, "allow_merge": not smoke}
    if len(auth.allowed_modules) != 186 or any(getattr(auth, key) != value for key, value in expected.items()):
        raise ValueError("Rank authorization differs from the prospectively frozen bounds")
    return rank


def claim_launch(protocol, authorization_path, bundle, out, *, smoke):
    """One attempt per arm and stage; an interruption consumes its launch slot."""
    protocol, authorization_path, bundle, out = map(Path, (protocol, authorization_path, bundle, out))
    config, auth = read(protocol), TrainingAuthorization.from_dict(read(authorization_path))
    rank = verify_authorization(config, auth, smoke=smoke)
    if not config.get("source_sha256"):
        raise ValueError("Phase III-R execution requires a populated prospective source manifest")
    verify_sources(config["source_sha256"])
    material = read(bundle)
    expected_splits = {"smoke"} if smoke else {"train", "replay"}
    if (set(material) != {"schema", "protocol_sha256", "rows", "manifest_sha256", "split_ids"}
            or set(material["split_ids"]) != expected_splits or file_hash(bundle) != auth.training_bundle_sha256
            or material["protocol_sha256"] != file_hash(protocol)):
        raise ValueError("Training can receive only exact rank-bound authorized fitting material")
    if (out.exists() or out.resolve().is_relative_to(ROOT)
            or authorization_path.resolve().parent != protocol.resolve().parent):
        raise ValueError("Launch needs a fresh private output and canonical per-arm authorization")
    path = protocol.parent/("smoke-launch-claim.json" if smoke else "main-launch-claim.json")
    value = {"schema": "weightbridge.phase_iiir.launch_claim.v1", "rank": rank, "smoke": smoke,
        "protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization_path),
        "bundle_sha256": file_hash(bundle), "out": str(out.resolve()), "run_index": 0,
        "retries_authorized": False, "exact_resume_implemented": False,
        "interruption_disposition": "RESOURCE_LIMITED_OR_INCONCLUSIVE"}
    json_dump_atomic(path, value)
    return path, value


def data_exposure_receipt(bundle, config):
    """Batch order is independent of shape-dependent Torch adapter initialization."""
    import hashlib
    import json
    rows = read(bundle)["rows"]
    settings = config["training"]
    order = data_order(len(rows), settings["seed"], settings["epochs"])
    if len(order) != 2048 or len(order) != settings["optimizer_steps"]*settings["effective_batch_size"]:
        raise ValueError("The one-epoch comparison must expose the same 2048 presentations")
    ordered_ids = [rows[index]["id"] for index in order]
    return {"schema": "weightbridge.phase_iiir.data_order.v1", "seed": settings["seed"],
        "order_rng": "independent random.Random(str(seed) + '/epoch-' + str(epoch))",
        "adapter_rng": "independent torch.Generator(cpu), randomized A then zero B, native sorted module order",
        "ordered_row_ids_sha256": hashlib.sha256(json.dumps(ordered_ids, separators=(",", ":")).encode()).hexdigest(),
        "data_order_sha256": hashlib.sha256(bytes(str(order), "ascii")).hexdigest(),
        "examples_seen": len(order), "batches": len(order)//16, "initialization_seed": 606117,
        "cross_rank_initial_factors_identical": False}


def _main_checks(result, completed, inventory, exposure, rank):
    checks = {"process_training_passed": result["passed"]}
    if completed is None or inventory is None:
        checks["complete_training_evidence"] = False
        return checks
    counts = completed["counts"]
    checks.update(actual_parameter_count=inventory["trainable_parameters"] == PARAMETERS[rank],
        actual_module_count=inventory["module_count"] == 186,
        actual_data_order=inventory["data_order_sha256"] == exposure["data_order_sha256"],
        exact_main_exposure=counts["examples_seen"] == 2048 and counts["optimizer_steps"] == 128,
        exact_forward_backward_schedule=counts["forward_calls"] == counts["backward_calls"] == 512,
        all_factors_changed=completed.get("all_factors_changed") is True,
        all_factors_received_gradients=completed.get("all_factors_received_nonzero_gradient") is True,
        frozen_base_unchanged=completed.get("base_unchanged") is True)
    return checks


def validate_main_execution(arm_root):
    """Recompute R main admission from worker artifacts, never a bare pass flag."""
    import math
    from .smoke import validate_gate
    root = Path(arm_root)
    protocol, authorization_path, bundle = root/"protocol.json", root/"main-authorization.json", root/"training-bundle.json"
    config = read(protocol)
    auth = TrainingAuthorization.from_dict(read(authorization_path))
    rank = verify_authorization(config, auth, smoke=False)
    verify_sources(config["source_sha256"])
    receipt, claim = read(root/"main-execution.json"), read(root/"main-launch-claim.json")
    out = Path(claim["out"])
    process_path = out.parent/f"{out.name}-process.json"
    if (not out.resolve().is_relative_to(root.resolve()) or receipt.get("schema") != "weightbridge.phase_iiir.main_execution.v1"
            or receipt.get("passed") is not True or receipt.get("status") != "COMPLETED" or receipt.get("rank") != rank
            or receipt.get("protocol_sha256") != file_hash(protocol)
            or receipt.get("launch_claim_sha256") != file_hash(root/"main-launch-claim.json")
            or receipt.get("process_receipt_sha256") != file_hash(process_path)):
        raise ValueError("Rank execution, launch claim or process identity differs")
    material = read(bundle)
    if (file_hash(bundle) != auth.training_bundle_sha256 or material["protocol_sha256"] != file_hash(protocol)
            or material["split_ids"] != ["train", "replay"]):
        raise ValueError("Completed run used a different fitting bundle")
    expected = {"protocol_sha256": file_hash(protocol), "authorization_sha256": file_hash(authorization_path),
        "bundle_sha256": file_hash(bundle), "parent_sha256": PARENT, "run_index": 0}
    if any(claim.get(key) != value for key, value in expected.items() if key != "parent_sha256"):
        raise ValueError("Main launch claim differs from frozen inputs")
    process, completed, inventory = read(process_path), read(out/"completed.json"), read(out/"inventory.json")
    for record in (process, completed):
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Completed worker or process is bound to another study")
    if (process.get("passed") is not True or not all(process.get("checks", {}).values())
            or process.get("training_completed_receipt_sha256") != file_hash(out/"completed.json")
            or process.get("log_sha256") != file_hash(out.parent/f"{out.name}.log")
            or completed.get("status") != "COMPLETED" or completed.get("smoke") is not False
            or completed.get("manifest_sha256") != material["manifest_sha256"]):
        raise ValueError("Incomplete, altered or failed training cannot supply eligible checkpoints")
    gate_path = Path(receipt["smoke_gate_path"])
    if file_hash(gate_path) != auth.smoke_gate_sha256 or process.get("smoke_gate_sha256") != auth.smoke_gate_sha256:
        raise ValueError("Main training smoke admission changed")
    validate_gate(protocol, root/"smoke-authorization.json", root/"smoke-bundle.json", root/"material/smoke.json", gate_path)
    execution, resources = process["execution"], process["execution"]["resources"]
    limits = {"wall_seconds": 5400., "cpu_seconds": 30000., "memory_bytes": 24*1024**3}
    sleep = process["temporary_sleep_request"]
    if (execution.get("exit_code") != 0 or execution.get("budget_failure") is not None
            or resources.get("cpu_accounting_complete_for_observed_processes") is not True
            or process.get("limits") != limits or receipt["resources"] != resources
            or (sleep["requested"] and (sleep["succeeded"] is not True or sleep["released"] is not True))):
        raise ValueError("Main resource monitoring or temporary sleep release failed")
    if (not math.isclose(sum(row["cpu_seconds"] for row in execution["processes"]), resources["observed_tree_cpu_seconds"], abs_tol=1e-7)
            or not math.isclose(resources["host_cpu_seconds"]+resources["observed_tree_cpu_seconds"], resources["cpu_seconds"], abs_tol=1e-7)
            or resources["peak_rss_bytes"] != sum(row["peak_rss_bytes"] for row in execution["processes"])):
        raise ValueError("Main process-tree aggregation differs from per-process observations")
    for observed in (resources, completed["resources"]):
        auth.check_usage(steps=completed["counts"]["optimizer_steps"], answer_tokens=completed["counts"]["answer_tokens"],
            input_tokens=completed["counts"]["input_tokens"], wall_seconds=observed["wall_seconds"],
            cpu_seconds=observed["cpu_seconds"], rss_bytes=observed["peak_rss_bytes"])
    if resources["observed_tree_cpu_seconds"]+.01 < completed["resources"]["cpu_seconds"] or resources["peak_rss_bytes"] < completed["resources"]["peak_rss_bytes"]:
        raise ValueError("Process-tree accounting omitted measured worker resources")
    exposure = data_exposure_receipt(bundle, config)
    if any(read(root/name) != exposure for name in ("prospective-data-order.json", "main-data-order.json")) or receipt["data_order"] != exposure:
        raise ValueError("Actual and prospective ordered training exposure differ")
    checks = _main_checks(process, completed, inventory, exposure, rank)
    if checks != receipt["checks"] or not all(checks.values()):
        raise ValueError("Exact rank counts, batch order or training completion failed independent replay")
    changes = completed["parameter_changes"]
    if (len(changes) != 372 or len({row["name"] for row in changes}) != 372
            or any(row["changed_entries"] <= 0 or row["nonzero_gradient_steps"] <= 0 for row in changes)):
        raise ValueError("Every declared factor must receive a gradient and change")
    expected_paths = [out/"checkpoints"/f"step-{step:04d}" for step in (64, 128)]
    if [Path(value).resolve() for value in completed["checkpoints"]] != [path.resolve() for path in expected_paths]:
        raise ValueError("Completed run changed its predeclared checkpoint schedule")
    for step, path in zip((64, 128), expected_paths, strict=True):
        saved = read(path/"receipt.json")
        conf, host = saved["configuration"], saved["host_metadata"]
        if (saved["sha256"] != file_hash(path/"adapter.safetensors") or saved["step"] != step
                or saved["trainable_parameters"] != PARAMETERS[rank]
                or conf["rank"] != rank or conf["alpha"] != rank*2 or conf["seed"] != 606117
                or conf["checkpoint_sha256"] != PARENT or conf["factor_dtype"] != "F32" or conf["base_dtype"] != "BF16"
                or tuple(row["module_name"] for row in conf["modules"]) != auth.allowed_modules
                or any(host.get(key) != value for key, value in {**expected, "step": step, "smoke": False}.items())):
            raise ValueError("Saved adapter identity differs from rank, source, fitted data or checkpoint step")
    if rank == 4:
        from .study import validate_replication_gate
        replication_path = Path(receipt["replication_receipt_path"])
        if file_hash(replication_path) != receipt["replication_receipt_sha256"]:
            raise ValueError("Rank-four launch replication gate changed")
        validate_replication_gate(config, replication_path)
    return receipt


def run(base, protocol, authorization_path, bundle, out, *, smoke_gate, replication_receipt=None):
    config = read(protocol)
    rank = validate_recipe(config)
    if rank == 4:
        if replication_receipt is None:
            raise ValueError("Rank-four main training requires an independently verified rank-eight receipt")
        from .study import validate_replication_gate
        validate_replication_gate(config, replication_receipt)
    elif replication_receipt is not None:
        raise ValueError("Rank-eight replication cannot inherit an earlier learned checkpoint")
    from .smoke import validate_gate
    arm_root = Path(protocol).parent
    validate_gate(protocol, arm_root/"smoke-authorization.json", arm_root/"smoke-bundle.json",
                  arm_root/"material/smoke.json", smoke_gate)
    claim, _ = claim_launch(protocol, authorization_path, bundle, out, smoke=False)
    exposure = data_exposure_receipt(bundle, config)
    json_dump_atomic(Path(protocol).parent/"main-data-order.json", exposure)
    # Q's worker remains unchanged: it validates native module identity, frozen
    # base bytes, both factor gradients, exact schedule, token budget and saves.
    result = launch_q_training(base, protocol, authorization_path, bundle, out, run_index=0, smoke_gate=smoke_gate)
    completed_path, inventory_path = Path(out)/"completed.json", Path(out)/"inventory.json"
    completed = read(completed_path) if completed_path.is_file() else None
    inventory = read(inventory_path) if inventory_path.is_file() else None
    checks = _main_checks(result, completed, inventory, exposure, rank)
    passed = all(checks.values())
    json_dump_atomic(Path(protocol).parent/"main-execution.json", {
        "schema": "weightbridge.phase_iiir.main_execution.v1", "rank": rank,
        "protocol_sha256": file_hash(protocol), "launch_claim_sha256": file_hash(claim),
        "data_order": exposure, "process_receipt_sha256": file_hash(Path(out).parent/f"{Path(out).name}-process.json"),
        "passed": passed, "status": "COMPLETED" if passed else "RESOURCE_LIMITED_OR_INCONCLUSIVE", "checks": checks,
        "smoke_gate_path": str(Path(smoke_gate).resolve()),
        "replication_receipt_path": str(Path(replication_receipt).resolve()) if replication_receipt else None,
        "replication_receipt_sha256": file_hash(replication_receipt) if replication_receipt else None,
        "resources": result["execution"]["resources"], "no_replacement_run": True})
    return {**result, "passed": passed, "status": "COMPLETED" if passed else "RESOURCE_LIMITED_OR_INCONCLUSIVE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "authorization", "bundle", "out", "smoke-gate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replication-receipt", type=Path)
    args = parser.parse_args()
    result = run(args.base, args.protocol, args.authorization, args.bundle, args.out,
        smoke_gate=args.smoke_gate, replication_receipt=args.replication_receipt)
    print(result["status"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)
