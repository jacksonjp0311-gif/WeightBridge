"""Host-only frozen-checkpoint screening and staged independent verification.

This module never trains a model, changes a gate, merges an adapter, or promotes
a checkpoint. Workers receive one hash-bound roster, not the private data root.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase_iiil.analyze import _checked_measurement
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from .data import load_split, read_manifest
from .integrity import PARENT, ROOT, authorize_stage, directory_hash, lock_finalist, read, verify_sources
from .measurement import compare, endpoint, summarize


def _host_output(base, out):
    parent, destination = Path(base).resolve(), Path(out).resolve()
    if (destination.is_relative_to(ROOT.resolve()) or destination.is_relative_to(parent)
            or parent.is_relative_to(destination)):
        raise ValueError("Host evidence must stay outside the public checkout and immutable parent checkpoint")


def _gate(parent, candidate, protocol, stage):
    rule = protocol["gates"][stage]
    return compare(parent, candidate, min_strict_gain=rule.get("min_strict_gain", 0),
        min_family_gain=rule.get("min_family_gain", 0),
        min_repeatable=rule.get("min_repeatable_corrections", 0))


def probe_rosters(train, replay, development, protocol):
    """Whole families/tasks only; the replay control probe has eighty rows."""
    family_ids = sorted({row["pair_id"] for row in train if row["group"] == "target"})
    count = protocol["data"]["practice_probe_families"]
    if len(family_ids) < count or count % 2:
        raise ValueError("Practice probe requires the prospectively balanced complete-family count")
    chosen = set()
    for tier in ("negative-close", "negative-far"):
        eligible = sorted({row["pair_id"] for row in train if row["group"] == "target" and row["tier"] == tier})
        if len(eligible) < count//2:
            raise ValueError("Practice probe lacks a balanced close/far family quota")
        chosen.update(eligible[:count//2])
    practice = [row for row in train if row["pair_id"] in chosen]
    control_probe = []
    labels = {"comparison_controls": "larger-decimal", "copy_controls": "copy",
              "addition_controls": "integer-add", "successor_controls": "integer-successor"}
    for kind, family in labels.items():
        tasks = sorted({row["task_id"] for row in replay if row["family"] == family})
        required = protocol["data"]["replay_probe_underlying"][kind]
        if len(tasks) < required:
            raise ValueError("Replay probe lacks the frozen underlying-task quota")
        wanted = set(tasks[:required])
        control_probe.extend(row for row in replay if row["task_id"] in wanted)
    if len(practice) != 8*count:
        raise ValueError("Practice probing cannot truncate a counterfactual family")
    return {"practice": practice+control_probe, "development": list(development)+control_probe}


def _validated_worker(path, rows_path, protocol_path, checkpoint, adapter_sha, stage):
    protocol, result = read(protocol_path), read(path)
    if (result.get("schema") != "weightbridge.supervised_baseline.evaluation.v1"
            or result["checkpoint_sha256"] != checkpoint or result["adapter_sha256"] != adapter_sha
            or result["protocol_sha256"] != file_hash(protocol_path) or result["roster_sha256"] != file_hash(rows_path)
            or result["stage"] != stage or result["source_sha256"] != protocol["source_sha256"]):
        raise ValueError("Independent worker identity differs from the frozen endpoint, source or representation")
    _checked_measurement(result["measurement"], read(rows_path), endpoint(protocol["inference"]))
    if summarize(result["measurement"]) != result["summary"]:
        raise ValueError("Independent worker correctness does not replay")
    return result


def _launch(base, protocol_path, rows_path, out, *, checkpoint=PARENT, adapter=None, adapter_sha=None,
            stage, authorization=None):
    if Path(out).exists():
        raise FileExistsError("Independent workers never overwrite evidence")
    protocol = read(protocol_path)
    verify_sources(protocol["source_sha256"])
    args = [sys.executable, "-m", "experiments.phase_iiiq_baseline.evaluate", "--base", str(base),
        "--protocol", str(protocol_path), "--rows", str(rows_path), "--out", str(out),
        "--expected-checkpoint", checkpoint, "--expected-rows", file_hash(rows_path), "--stage", stage]
    if adapter is not None:
        args += ["--adapter", str(adapter), "--expected-adapter", adapter_sha]
    if authorization is not None:
        args += ["--authorization", str(authorization)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(args, cwd=ROOT, check=True, timeout=protocol["resources"]["evaluation_worker_wall_seconds"])
    verify_sources(protocol["source_sha256"])
    return _validated_worker(out, rows_path, protocol_path, checkpoint, adapter_sha, stage)


def _checkpoints(run_dirs, protocol_path, manifest_sha):
    protocol = read(protocol_path)
    expected_runs = len(protocol["training"]["learning_rates"])
    if len(run_dirs) != expected_runs:
        raise ValueError("All prospectively authorized training runs must be accounted for")
    candidates, seen = [], set()
    for run_dir in run_dirs:
        root = Path(run_dir).resolve(strict=True)
        complete = read(root / "completed.json")
        index = complete["run_index"]
        if (complete.get("status") != "COMPLETED" or complete.get("base_unchanged") is not True
                or complete.get("smoke") is not False or complete["parent_sha256"] != PARENT
                or complete["protocol_sha256"] != file_hash(protocol_path) or complete["manifest_sha256"] != manifest_sha
                or complete["counts"]["optimizer_steps"] != protocol["training"]["optimizer_steps"]
                or index in seen or not 0 <= index < expected_runs):
            raise ValueError("Incomplete, duplicated or incompatible training cannot supply a finalist")
        seen.add(index)
        advertised = {Path(path).resolve() for path in complete["checkpoints"]}
        expected = {root / "checkpoints" / f"step-{step:04d}" for step in protocol["training"]["checkpoint_steps"]}
        if advertised != expected:
            raise ValueError("Saved checkpoints differ from the frozen development schedule")
        for step in protocol["training"]["checkpoint_steps"]:
            directory = root / "checkpoints" / f"step-{step:04d}"
            adapter, receipt_path = directory / "adapter.safetensors", directory / "receipt.json"
            receipt = read(receipt_path)
            host = receipt["host_metadata"]
            norm = receipt["effective_update"]["stored_update_norm"]
            if (file_hash(adapter) != receipt["sha256"] or receipt.get("base_unchanged") is not True
                    or receipt["configuration"]["checkpoint_sha256"] != PARENT or receipt["step"] != step
                    or host["protocol_sha256"] != file_hash(protocol_path) or host["manifest_sha256"] != manifest_sha
                    or host["run_index"] != index or host["step"] != step or not math.isfinite(norm) or norm < 0):
                raise ValueError("Adapter checkpoint receipt differs from its authorized immutable artifact")
            candidates.append({"id": f"run-{index}-step-{step:04d}", "run_index": index, "step": step,
                "adapter": adapter, "directory": directory, "adapter_sha256": receipt["sha256"],
                "factor_sha256": receipt["factor_sha256"],
                "stored_update_norm": norm, "training_receipt_sha256": file_hash(receipt_path),
                "completed_run_sha256": file_hash(root / "completed.json")})
    return sorted(candidates, key=lambda value: (value["run_index"], value["step"]))


def _rank(value):
    gate = value["development_gate"]
    return (-gate["complete_family_gain"], -gate["strict_comparison_gain"], value["step"],
            value["stored_update_norm"], value["run_index"], value["id"])


def assess(base, protocol_path, material, out, run_dirs):
    """Screen only fixed checkpoints; no independent population opens here."""
    base, protocol_path, material, out = map(Path, (base, protocol_path, material, out))
    _host_output(base, out)
    protocol, manifest = read(protocol_path), read_manifest(material, protocol=protocol_path)
    verify_sources(protocol["source_sha256"])
    if out.exists():
        raise FileExistsError("Use a fresh host assessment path")
    out.mkdir(parents=True)
    wall, cpu = time.perf_counter(), time.process_time()
    result = {"schema": "weightbridge.supervised_baseline.assessment.v1", "disposition": "HOLD",
        "reason": "incomplete_behavioral_evaluation", "protocol_sha256": file_hash(protocol_path),
        "manifest_sha256": file_hash(material / "manifest.json"), "candidates": {}, "winner": None,
        "merged": None, "preservation_validation": None, "selection": None, "hidden": None,
        "lineage_accepted": False, "production_authority": False, "errors": {}}
    try:
        candidates = _checkpoints(run_dirs, protocol_path, result["manifest_sha256"])
        inputs = {name: load_split(material, name, manifest, protocol=protocol_path) for name in ("train", "replay", "development")}
        rosters = probe_rosters(inputs["train"], inputs["replay"], inputs["development"], protocol)
        parents = {}
        for stage, rows in rosters.items():
            roster_path = out / "rosters" / f"{stage}.json"
            json_dump_atomic(roster_path, rows)
            parents[stage] = _launch(base, protocol_path, roster_path,
                out / "evaluation" / f"{stage}-parent.json", stage=stage)
        aliases = {}
        for candidate in candidates:
            name, digest = candidate["id"], candidate["adapter_sha256"]
            record = {key: value for key, value in candidate.items() if key not in ("adapter", "directory")}
            if digest in aliases:
                owner = result["candidates"][aliases[digest]]
                record.update(practice_gate=owner["practice_gate"], development_gate=owner["development_gate"],
                              eligible=owner["eligible"], duplicate_of=aliases[digest])
            else:
                for stage in ("practice", "development"):
                    worker = _launch(base, protocol_path, out / "rosters" / f"{stage}.json",
                        out / "evaluation" / f"{stage}-{name}.json", adapter=candidate["adapter"],
                        adapter_sha=digest, stage=stage)
                    record[f"{stage}_gate"] = _gate(parents[stage]["measurement"], worker["measurement"], protocol, stage)
                record["eligible"] = bool(record["practice_gate"]["passed"] and record["development_gate"]["passed"])
                aliases[digest] = name
            result["candidates"][name] = record
        qualified = sorted((record for record in result["candidates"].values() if record["eligible"]), key=_rank)
        if qualified:
            winner = qualified[0]
            selected = next(value for value in candidates if value["id"] == winner["id"])
            lock_finalist(out, adapter_path=selected["directory"], protocol_sha256=result["protocol_sha256"],
                manifest_sha256=result["manifest_sha256"], practice_gate=winner["practice_gate"], development_gate=winner["development_gate"])
            json_dump_atomic(out / "finalist-reference.json", {"candidate_id": selected["id"],
                "adapter_directory": str(selected["directory"]), "adapter_file_sha256": selected["adapter_sha256"],
                "factor_sha256": selected["factor_sha256"]})
            result.update(winner=winner["id"], reason="merged_checkpoint_and_independent_verification_required")
        else:
            result["reason"] = "no_practice_and_development_qualified_adapter"
        verify_sources(protocol["source_sha256"])
    except BaseException as error:
        result["errors"]["assessment"] = {"type": type(error).__name__, "message": str(error)}
        result.update(winner=None, reason="incomplete_behavioral_evaluation")
        if not isinstance(error, Exception):
            json_dump_atomic(out / "assessment.json", result)
            raise
    result["resources"] = {"host_wall_seconds_including_workers": time.perf_counter()-wall,
                           "host_cpu_seconds": time.process_time()-cpu}
    result["artifact_sha256"] = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))}
    json_dump_atomic(out / "assessment.json", result)
    return result


def _replay_screening(out, protocol_path):
    result, protocol = read(out / "assessment.json"), read(protocol_path)
    if result["protocol_sha256"] != file_hash(protocol_path) or result["errors"] or not result["winner"]:
        raise ValueError("Independent stages require a complete locked adapter assessment")
    for name, digest in result["artifact_sha256"].items():
        path = (out / name).resolve(strict=True)
        if not path.is_relative_to(out.resolve()) or file_hash(path) != digest:
            raise ValueError("Adapter screening evidence changed before independent verification")
    parents = {stage: _validated_worker(out / "evaluation" / f"{stage}-parent.json", out / "rosters" / f"{stage}.json",
        protocol_path, PARENT, None, stage) for stage in ("practice", "development")}
    for name, record in result["candidates"].items():
        if record.get("duplicate_of"):
            owner = result["candidates"][record["duplicate_of"]]
            if record["adapter_sha256"] != owner["adapter_sha256"]:
                raise ValueError("Checkpoint alias differs from its original artifact")
            if any(record[key] != owner[key] for key in ("practice_gate", "development_gate", "eligible")):
                raise ValueError("Checkpoint alias changes the measured qualification")
            continue
        for stage in ("practice", "development"):
            worker = _validated_worker(out / "evaluation" / f"{stage}-{name}.json", out / "rosters" / f"{stage}.json",
                protocol_path, PARENT, record["adapter_sha256"], stage)
            if _gate(parents[stage]["measurement"], worker["measurement"], protocol, stage) != record[f"{stage}_gate"]:
                raise ValueError("Adapter gate does not independently replay")
        if record["eligible"] != bool(record["practice_gate"]["passed"] and record["development_gate"]["passed"]):
            raise ValueError("Adapter eligibility differs from the actual practice/development gates")
    qualified = sorted((record for record in result["candidates"].values() if record["eligible"]), key=_rank)
    if not qualified or qualified[0]["id"] != result["winner"]:
        raise ValueError("Locked finalist differs from the frozen ranking")
    return result


def finish(base, merged, merge_receipt, merge_authorization, protocol_path, material, out, *, hidden_path=None):
    """Consume an explicitly authorized merge; never create or promote one."""
    base, merged, merge_receipt, merge_authorization, protocol_path, material, out = map(Path,
        (base, merged, merge_receipt, merge_authorization, protocol_path, material, out))
    _host_output(base, out)
    protocol = read(protocol_path)
    verify_sources(protocol["source_sha256"])
    if ((out / "independent.json").exists() or (out / "merged-development-gate.json").exists()
            or (out / "evaluation/merged-practice.json").exists()):
        raise FileExistsError("Independent attempts are write-once; an interrupted attempt requires a new host record")
    screening = _replay_screening(out, protocol_path)
    if screening["manifest_sha256"] != file_hash(material / "manifest.json"):
        raise ValueError("Independent material differs from the locked screening cohort")
    reference, lock, merged_receipt = read(out / "finalist-reference.json"), read(out / "finalist-lock.json"), read(merge_receipt)
    policy = read(merge_authorization)
    if (policy.get("schema") != "weightbridge.supervised_lora_merge.v1"
            or policy.get("base_checkpoint_sha256") != PARENT or not policy.get("authority")
            or policy.get("stored_dtype") != "BF16"
            or policy.get("merge_math") != "float64_factor_product_then_round_once_into_BF16"
            or policy.get("production_authority", False) is not False or policy.get("lineage_authority", False) is not False):
        raise ValueError("A separate host policy must authorize the isolated conventional merge")
    directory_sha, files = directory_hash(reference["adapter_directory"])
    representation = load_checkpoint(merged).checkpoint_sha256()
    selected = screening["candidates"][screening["winner"]]
    if (reference["candidate_id"] != screening["winner"]
            or lock["practice_gate"] != selected["practice_gate"] or lock["development_gate"] != selected["development_gate"]
            or directory_sha != lock["adapter_sha256"] or files != lock["adapter_files"]
            or file_hash(Path(reference["adapter_directory"]) / "adapter.safetensors") != reference["adapter_file_sha256"]
            or merged_receipt.get("schema") != "weightbridge.supervised_lora_merge_receipt.v1"
            or merged_receipt.get("status") != "ISOLATED_CANDIDATE" or merged_receipt.get("base_sha256") != PARENT
            or merged_receipt.get("candidate_sha256") != representation
            or merged_receipt.get("factor_sha256") != reference["factor_sha256"]
            or merged_receipt.get("base_unchanged") is not True
            or merged_receipt.get("authorization_sha256") != file_hash(merge_authorization)
            or load_checkpoint(base).checkpoint_sha256() != PARENT):
        raise ValueError("Merged model lacks the locked artifact, separate authorization or exact checkpoint identity")
    result = {"schema": "weightbridge.supervised_baseline.independent.v1", "disposition": "HOLD",
        "representation_sha256": representation, "protocol_sha256": file_hash(protocol_path),
        "manifest_sha256": file_hash(material / "manifest.json"), "merge_receipt_sha256": file_hash(merge_receipt),
        "merged": None, "preservation_validation": None, "selection": None, "hidden": None,
        "exact_reconstruction": None, "lineage_accepted": False, "production_authority": False}
    try:
        merged_gates = {}
        for stage in ("practice", "development"):
            rows_path = out / "rosters" / f"{stage}.json"
            candidate = _launch(merged, protocol_path, rows_path, out / "evaluation" / f"merged-{stage}.json",
                                checkpoint=representation, stage=f"merged-{stage}")
            parent = read(out / "evaluation" / f"{stage}-parent.json")
            merged_gates[stage] = _gate(parent["measurement"], candidate["measurement"], protocol, stage)
        gate = {"passed": all(value["passed"] for value in merged_gates.values()), "gates": merged_gates,
                **{key: result[key] for key in ("representation_sha256", "protocol_sha256", "manifest_sha256")}}
        json_dump_atomic(out / "merged-development-gate.json", gate)
        result["merged"] = gate
        if not gate["passed"]:
            result["reason"] = "merged_representation_failed_practice_or_development"
        else:
            manifest = read_manifest(material, protocol=protocol_path)
            for stage in ("preservation-validation", "selection", "hidden"):
                authorization = authorize_stage(out, stage, representation_sha256=representation,
                    roster_sha256=manifest["splits"][stage]["sha256"])
                auth_path = out / "authorizations" / f"{stage}.json"
                json_dump_atomic(auth_path, authorization)
                rows = load_split(material, stage, manifest, protocol=protocol_path, host_authorized=True,
                    hidden_path=hidden_path if stage == "hidden" else None)
                rows_path = out / "private-independent-rosters" / f"{stage}.json"
                json_dump_atomic(rows_path, rows)
                observations = {}
                for name, model, digest in (("parent", base, PARENT), ("merged", merged, representation)):
                    observations[name] = _launch(model, protocol_path, rows_path, out / "evaluation" / f"{stage}-{name}.json",
                        checkpoint=digest, stage=stage, authorization=auth_path)
                measured = _gate(observations["parent"]["measurement"], observations["merged"]["measurement"], protocol, stage)
                receipt = {**measured, **{key: result[key] for key in ("representation_sha256", "protocol_sha256", "manifest_sha256")},
                    "authorization_sha256": file_hash(auth_path)}
                json_dump_atomic(out / f"{stage}-gate.json", receipt)
                result[stage.replace("-", "_")] = receipt
                if not measured["passed"]:
                    result["reason"] = f"independent_{stage}_failure"
                    break
            else:
                result["reason"] = "independent_behavior_qualified_exact_reconstruction_and_rollback_required"
        verify_sources(protocol["source_sha256"])
        if load_checkpoint(base).checkpoint_sha256() != PARENT or load_checkpoint(merged).checkpoint_sha256() != representation:
            raise ValueError("Independent evaluation changed original or candidate checkpoint identity")
    except BaseException as error:
        result["reason"] = "incomplete_independent_verification"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        if not isinstance(error, Exception):
            json_dump_atomic(out / "independent.json", result)
            raise
    json_dump_atomic(out / "independent.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("assess", "finish"))
    for name in ("base", "protocol", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path)
    for name in ("merged", "merge-receipt", "merge-authorization", "hidden"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.command == "assess":
        if not args.runs:
            parser.error("assess requires the fixed completed training runs")
        print(assess(args.base, args.protocol, args.material, args.out, args.runs)["reason"])
    else:
        if not all((args.merged, args.merge_receipt, args.merge_authorization)):
            parser.error("finish requires an explicitly authorized merged artifact and receipt")
        print(finish(args.base, args.merged, args.merge_receipt, args.merge_authorization,
                     args.protocol, args.material, args.out, hidden_path=args.hidden)["reason"])
