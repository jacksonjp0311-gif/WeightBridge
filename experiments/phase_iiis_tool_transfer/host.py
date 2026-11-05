"""Bounded host commands for Phase S development and evidence-only decisions.

Only `baselines` launches model workers, using the existing shared resource
ledger. `assess` and versioned requests replay existing evidence. None trains,
opens a bank for fitting, selects step64, or grants production authority.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from weightbridge.core import file_hash
from . import execute, study
from .integrity import ROOT, ORIGINAL, DECIMAL_PARENT, PARENTS, read, verify_sources, write_once
from .verification import authorize_evaluation, validate_worker


def _private(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT.resolve()):
        raise ValueError("Raw host evidence must stay outside the public source checkout")
    return path


def _initialized(root):
    root = _private(root)
    initialized = read(root / "initialized.json")
    manifest_path = root / "manifest.json"
    if (initialized.get("schema") != "weightbridge.phase_iiis.initialization.v1"
            or initialized.get("manifest_sha256") != file_hash(manifest_path)):
        raise ValueError("Host command requires the unchanged prospectively initialized study")
    verify_sources(initialized["source_sha256"])
    for arm, value in initialized["arms"].items():
        protocol = root / arm / "protocol.json"
        if file_hash(protocol) != value["protocol_sha256"]:
            raise ValueError("Frozen arm protocol changed")
    return root, initialized, read(manifest_path)


def baselines(study_root, original, decimal_parent, ledger):
    """Run the same private DEVELOPMENT rows on both immutable parents once."""
    root, initialized, manifest = _initialized(study_root)
    out, ledger = root / "baselines", _private(ledger)
    if out.exists():
        raise FileExistsError("Parent development attempts cannot be repeated or overwritten")
    for base in (original, decimal_parent):
        if out.resolve().is_relative_to(Path(base).resolve()):
            raise ValueError("Development evidence must stay outside immutable model directories")
    if ledger.is_relative_to(out) or ledger.is_relative_to(root / "sequential") or ledger.is_relative_to(root / "original-joint"):
        raise ValueError("Parent work must use the shared cumulative model-work ledger")
    arm = "original-joint" if "original-joint" in initialized["arms"] else "sequential"
    protocol_path = root / arm / "protocol.json"
    protocol = read(protocol_path)
    limits = protocol["resources"]["evaluation_limits"]
    out.mkdir()
    write_once(out / "started.json", {"schema": "weightbridge.phase_iiis.parent_development_started.v1",
        "manifest_sha256": file_hash(root / "manifest.json"), "protocol_sha256": file_hash(protocol_path),
        "retry_authorized": False, "stage": "development", "protected_banks_opened": False})
    specifications, attempts = {}, []
    try:
        for role, base, identity in (("original", original, ORIGINAL), ("r8", decimal_parent, DECIMAL_PARENT)):
            specifications[role] = {}
            for task in study.SUITES:
                declared = manifest["splits"]["development"]["files"][task]
                rows = root / "material/development" / f"{task}.json"
                if Path(declared["path"]).resolve() != rows.resolve() or file_hash(rows) != declared["sha256"]:
                    raise ValueError("Development roster differs from its prospective commitment")
                target = out / role
                target.mkdir(exist_ok=True)
                record, authorization = target / f"{task}.json", target / f"{task}-authorization.json"
                authorize_evaluation(authorization, protocol_path, stage="development", task=task,
                    roster_sha256=file_hash(rows), representation_sha256=identity)
                name = f"development-parent-{role}-{task}"
                args = ["--base", str(Path(base).resolve()), "--protocol", str(protocol_path),
                    "--rows", str(rows), "--out", str(record), "--authorization", str(authorization),
                    "--stage", "development", "--task", task, "--expected-checkpoint", identity,
                    "--expected-rows", file_hash(rows)]
                result = execute.run(ledger, name, "experiments.phase_iiis_tool_transfer.evaluate", args,
                    wall_limit=limits["max_wall_seconds"], cpu_limit=limits["max_cpu_seconds"],
                    memory_limit=min(limits.get("max_memory_bytes", protocol["resources"]["max_memory_bytes"]),
                                     protocol["resources"]["max_memory_bytes"]))
                process = ledger / f"{name}.receipt.json"
                attempts.append({"path": str(process), "sha256": file_hash(process), "passed": result["passed"]})
                if not result["passed"]:
                    raise RuntimeError("Parent development execution failed its resource or process gate")
                validate_worker(record, rows, protocol_path, identity, None, "development", task, authorization=authorization)
                study.validate_execution(process, module="experiments.phase_iiis_tool_transfer.evaluate",
                                         out=record, protocol_path=protocol_path)
                specifications[role][task] = {"path": str(record), "rows": str(rows), "authorization": str(authorization),
                    "protocol": str(protocol_path), "stage": "development", "execution": str(process)}
        value = {"schema": "weightbridge.phase_iiis.parent_development.v1", "passed": True,
            "manifest_sha256": file_hash(root / "manifest.json"), "workers": specifications,
            "execution_receipts": attempts, "protected_banks_opened": False}
        write_once(out / "completed.json", value)
        return value
    except BaseException as error:
        write_once(out / "failed.json", {"schema": "weightbridge.phase_iiis.parent_development_failure.v1",
            "passed": False, "error": {"type": type(error).__name__, "message": str(error)},
            "execution_receipts": attempts, "retry_authorized": False, "protected_banks_opened": False})
        raise


def assess(study_root, arm):
    """Record both declared checkpoints, while permitting only step128 to qualify."""
    root, _, _ = _initialized(study_root)
    if arm not in PARENTS:
        raise ValueError("Unknown study arm")
    arm_root, protocol_path = root / arm, root / arm / "protocol.json"
    protocol = read(protocol_path)
    parents = read(root / "baselines/completed.json")
    executed = read(arm_root / "run-arm.json")
    if (parents.get("passed") is not True or parents.get("manifest_sha256") != protocol["task_manifest_sha256"]
            or set(parents.get("workers", {})) != {"original", "r8"}
            or executed.get("completed") is not True or executed.get("status") != "DEVELOPMENT_MEASURED"
            or executed.get("protocol_sha256") != file_hash(protocol_path)
            or set(executed.get("development", {})) != {"64", "128"}):
        raise ValueError("Assessment needs complete parent and two fixed checkpoint development measurements")
    out = arm_root / "assessment"
    if out.exists():
        raise FileExistsError("An assessment cannot be repeated or replaced")
    out.mkdir()
    gates = {}
    for step in (64, 128):
        declared = executed["development"][str(step)]
        checkpoint = arm_root / "training/checkpoints" / f"step-{step:04d}"
        saved = read(checkpoint / "receipt.json")
        if (saved.get("step") != step or saved.get("sha256") != file_hash(checkpoint / "adapter.safetensors")
                or declared["adapter_sha256"] != saved["sha256"]
                or saved.get("configuration", {}).get("checkpoint_sha256") != PARENTS[arm]):
            raise ValueError("Development specification differs from its exact declared checkpoint")
        workers = {**parents["workers"], "candidate": declared["workers"]}
        gate = study.record_stage(out / f"step-{step:04d}.json", protocol_path, stage="development", workers=workers,
            candidate_checkpoint=PARENTS[arm], adapter_sha256=saved["sha256"], step=step)
        gates[str(step)] = {"path": str(out / f"step-{step:04d}.json"),
            "sha256": file_hash(out / f"step-{step:04d}.json"), "passed": gate["passed"],
            "finalist_eligible": gate["finalist_eligible"], "disposition": gate["disposition"]}
    value = {"schema": "weightbridge.phase_iiis.development_assessment.v1", "arm": arm,
        "protocol_sha256": file_hash(protocol_path), "task_manifest_sha256": protocol["task_manifest_sha256"],
        "fixed_finalist_step": 128, "selected_step": 128 if gates["128"]["finalist_eligible"] else None,
        "passed": gates["128"]["finalist_eligible"], "gates": gates,
        "protected_banks_opened": False, "production_authority": False, "lineage_authority": False}
    write_once(out / "assessment.json", value)
    return value


REQUESTS = {"record-stage": study.record_stage, "lock-arm": study.lock_arm, "lock-arms": study.lock_arms,
            "lock-operational-failure": study.lock_operational_failure, "finalize": study.finalize}


def request(path):
    """Execute one versioned, explicit evidence request; no arbitrary code or bank routing."""
    value = read(path)
    if (set(value) != {"schema", "operation", "arguments"}
            or value["schema"] != "weightbridge.phase_iiis.host_request.v1"
            or value["operation"] not in REQUESTS or not isinstance(value["arguments"], dict)
            or "out" not in value["arguments"]):
        raise ValueError("Unknown or malformed versioned host evidence request")
    _private(value["arguments"]["out"])
    return REQUESTS[value["operation"]](**value["arguments"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baselines")
    for name in ("study", "original", "decimal-parent", "ledger"):
        baseline.add_argument(f"--{name}", type=Path, required=True)
    assessment = sub.add_parser("assess")
    assessment.add_argument("--study", type=Path, required=True)
    assessment.add_argument("--arm", choices=tuple(PARENTS), required=True)
    evidence = sub.add_parser("request")
    evidence.add_argument("--file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baselines":
        result = baselines(args.study, args.original, args.decimal_parent, args.ledger)
    elif args.command == "assess":
        result = assess(args.study, args.arm)
    else:
        result = request(args.file)
    print({key: result.get(key) for key in ("schema", "passed", "disposition", "selected_step")}, flush=True)


if __name__ == "__main__":
    main()
