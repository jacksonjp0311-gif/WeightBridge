"""Final host verification for an already-qualified conventional checkpoint.

This post-freeze runner reuses the frozen merge and evaluation workers. It
preserves prior receipts and keeps protected generations in private storage.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from .data import read_manifest
from .integrity import PARENT, authorize_stage, directory_hash, read, verify_sources
from .run_training import _temporary_sleep_request, launch_owned_tree
from .study import _gate, _host_output, _replay_screening, _validated_worker

STAGES = ("preservation-validation", "selection", "hidden")
PENDING_REASON = "independent_behavior_qualified_exact_reconstruction_and_rollback_required"


def _qualified_independent(receipt):
    if (receipt.get("schema") != "weightbridge.supervised_baseline.independent.v1"
            or receipt.get("reason") != PENDING_REASON or receipt.get("disposition") != "HOLD"
            or receipt.get("error") is not None or receipt.get("exact_reconstruction") is not None
            or receipt.get("lineage_accepted") is not False or receipt.get("production_authority") is not False
            or any((receipt.get(name) or {}).get("passed") is not True
                   for name in ("merged", "preservation_validation", "selection", "hidden"))):
        raise ValueError("Final replay requires the original fully qualified independent receipt")


def _merge_report(path, identity, adapter_sha):
    verification = read(path.with_suffix(".verification.json"))
    expected = {"schema": "weightbridge.supervised_baseline.merge_verification.v1",
        "merge_receipt_sha256": file_hash(path), "adapter_sha256": adapter_sha,
        "protocol_sha256": identity["protocol_sha256"], "parent_checkpoint_sha256": PARENT,
        "candidate_sha256": identity["representation_sha256"], "lineage_authority": False,
        "production_authority": False}
    if any(verification.get(key) != value for key, value in expected.items()):
        raise ValueError("Merge verification receipt differs from its authorized checkpoint")
    return verification


def _prior_evidence(assessment, material, protocol_path, protocol, independent, identity, artifacts):
    """Replay recorded gates before starting new workers or consuming their rosters."""
    screening = _replay_screening(assessment, protocol_path)
    reference = read(assessment / "finalist-reference.json")
    lock = read(assessment / "finalist-lock.json")
    selected = screening["candidates"][screening["winner"]]
    adapter_dir = Path(reference["adapter_directory"])
    adapter = adapter_dir / "adapter.safetensors"
    adapter_receipt = adapter_dir / "receipt.json"
    saved = read(adapter_receipt)
    directory_sha, files = directory_hash(adapter_dir)
    if (screening["manifest_sha256"] != identity["manifest_sha256"]
            or reference["candidate_id"] != screening["winner"] or directory_sha != lock["adapter_sha256"]
            or files != lock["adapter_files"] or file_hash(adapter_receipt) != selected["training_receipt_sha256"]
            or file_hash(adapter) != reference["adapter_file_sha256"]
            or saved["sha256"] != reference["adapter_file_sha256"]
            or selected["adapter_sha256"] != reference["adapter_file_sha256"]
            or saved["factor_sha256"] != reference["factor_sha256"]
            or any(lock.get(key) != identity[key] for key in ("protocol_sha256", "manifest_sha256"))
            or any(lock[f"{stage}_gate"] != selected[f"{stage}_gate"] for stage in ("practice", "development"))):
        raise ValueError("Locked adapter and training receipt do not reproduce")
    artifacts.update({"assessment": assessment / "assessment.json", "finalist_reference": assessment / "finalist-reference.json",
        "finalist_lock": assessment / "finalist-lock.json", "adapter": adapter, "adapter_training_receipt": adapter_receipt})

    merged_gates = {}
    for stage in ("practice", "development"):
        roster = assessment / "rosters" / f"{stage}.json"
        before = assessment / "evaluation" / f"{stage}-parent.json"
        after = assessment / "evaluation" / f"merged-{stage}.json"
        parent = _validated_worker(before, roster, protocol_path, PARENT, None, stage)
        candidate = _validated_worker(after, roster, protocol_path, identity["representation_sha256"], None, f"merged-{stage}")
        merged_gates[stage] = _gate(parent["measurement"], candidate["measurement"], protocol, stage)
        artifacts.update({f"merged_{stage}_roster": roster, f"merged_{stage}_parent": before, f"merged_{stage}_candidate": after})
    merged_gate = {**identity, "passed": all(gate["passed"] for gate in merged_gates.values()), "gates": merged_gates}
    merged_path = assessment / "merged-development-gate.json"
    if not merged_gate["passed"] or merged_gate != independent["merged"] or merged_gate != read(merged_path):
        raise ValueError("Recorded merged development gate does not replay")
    artifacts["merged_gate"] = merged_path

    manifest = read_manifest(material, protocol=protocol_path)
    stages = {}
    for stage in STAGES:
        auth = assessment / "authorizations" / f"{stage}.json"
        roster = assessment / "private-independent-rosters" / f"{stage}.json"
        expected_roster_sha = manifest["splits"][stage]["sha256"]
        expected = authorize_stage(assessment, stage, representation_sha256=identity["representation_sha256"],
                                   roster_sha256=expected_roster_sha)
        if read(auth) != expected or file_hash(roster) != expected_roster_sha:
            raise ValueError(f"Recorded {stage} authorization or roster hash differs")
        before = assessment / "evaluation" / f"{stage}-parent.json"
        after = assessment / "evaluation" / f"{stage}-merged.json"
        parent = _validated_worker(before, roster, protocol_path, PARENT, None, stage)
        candidate = _validated_worker(after, roster, protocol_path, identity["representation_sha256"], None, stage)
        gate = {**_gate(parent["measurement"], candidate["measurement"], protocol, stage),
                **identity, "authorization_sha256": file_hash(auth)}
        gate_path = assessment / f"{stage}-gate.json"
        if not gate["passed"] or gate != independent[stage.replace("-", "_")] or gate != read(gate_path):
            raise ValueError(f"Recorded {stage} gate does not replay")
        stages[stage] = (roster, auth, parent, candidate)
        artifacts.update({f"{stage}_{label}": path for label, path in
            (("roster", roster), ("authorization", auth), ("parent", before), ("candidate", after), ("gate", gate_path))})
    return adapter, reference, stages


def _generation_equal(reference, replay):
    if reference["endpoint"] != replay["endpoint"] or len(reference["rows"]) != len(replay["rows"]):
        return False
    fields = ("id", "generated_token_ids", "text", "stop_reason")
    return all(tuple(before[key] for key in fields) == tuple(after[key] for key in fields)
               for before, after in zip(reference["rows"], replay["rows"], strict=True))


def _worker(module, args, name, out, limits, attempts):
    record = {"name": name, "module": module, "execution": None}
    attempts.append(record)
    log = out / f"{name}.log"
    try:
        with _temporary_sleep_request() as sleep:
            record["temporary_sleep_request"] = sleep
            if sleep["requested"] and not sleep["succeeded"]:
                raise RuntimeError("Windows refused the bounded worker sleep request")
            record["execution"] = launch_owned_tree(
                [sys.executable, "-m", f"experiments.phase_iiiq_baseline.{module}", *args], log, **limits)
        execution = record["execution"]
        used = execution["resources"]
        if (execution["exit_code"] != 0 or execution["budget_failure"] is not None
                or not used["cpu_accounting_complete_for_observed_processes"]
                or (sleep["requested"] and not sleep["released"])
                or any(not math.isfinite(used[field]) or not 0 <= used[field] <= limits[bound]
                       for field, bound in (("wall_seconds", "wall_limit"), ("cpu_seconds", "cpu_limit"),
                                            ("peak_rss_bytes", "memory_limit")))):
            raise RuntimeError(f"Final verification worker {name} failed its execution or resource gate")
        return record
    except BaseException as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["log_sha256"] = file_hash(log) if log.exists() else None
        receipt = out / f"{name}-process.json"
        json_dump_atomic(receipt, record)
        record["process_receipt_sha256"] = file_hash(receipt)


def _check_worker_body(record, body):
    used = record["execution"]["resources"]
    if (any(not math.isfinite(body[field]) or body[field] < 0 for field in ("wall_seconds", "cpu_seconds", "peak_rss_bytes"))
            or used["observed_tree_cpu_seconds"] + .01 < body["cpu_seconds"]
            or used["peak_rss_bytes"] < body["peak_rss_bytes"] or used["wall_seconds"] < body["wall_seconds"]):
        raise ValueError("Owned worker resource receipt does not cover its measured body")


def _verify_reconstruction(reconstructed, original, replay_receipt, identity):
    digest = load_checkpoint(reconstructed).checkpoint_sha256()
    same_receipt = read(replay_receipt) == original
    return {"passed": digest == identity["representation_sha256"] and same_receipt,
        "checkpoint_sha256": digest, "expected_checkpoint_sha256": identity["representation_sha256"],
        "numerical_merge_receipt_exact": same_receipt, "merge_receipt_sha256": file_hash(replay_receipt),
        "merge_verification_sha256": file_hash(replay_receipt.with_suffix(".verification.json"))}


def run(base, merged, merge_receipt, merge_authorization, protocol_path, material, assessment, out):
    base, merged, merge_receipt, merge_authorization, protocol_path, material, assessment, out = map(Path,
        (base, merged, merge_receipt, merge_authorization, protocol_path, material, assessment, out))
    _host_output(base, out)
    _host_output(merged, out)
    if out.resolve().is_relative_to(material.resolve()):
        raise ValueError("Replay output must stay outside the frozen material directory")
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    attempts, replay = [], dict.fromkeys(STAGES)
    code = {name: file_hash(Path(__file__).with_name(name)) for name in ("verify_final.py", "run_training.py")}
    result = {"schema": "weightbridge.supervised_baseline.final_verification.v1", "disposition": "HOLD",
        "reason": "independent_gate_refused", "passed": False, "exact_reconstruction": None,
        "replays": replay, "lineage_accepted": False, "production_authority": False, "host_source_sha256": code,
        "protected_generations_private": True}
    json_dump_atomic(out / "started.json", {"host_source_sha256": code, "maximum_reconstruction_workers": 1,
        "maximum_evaluation_workers": 2*len(STAGES)})
    interrupted = None
    try:
        original_path = assessment / "independent.json"
        independent = read(original_path)
        _qualified_independent(independent)
        result["reason"] = "prior_evidence_binding_refused"
        protocol = read(protocol_path)
        verify_sources(protocol["source_sha256"])
        identity = {"protocol_sha256": file_hash(protocol_path), "manifest_sha256": file_hash(material / "manifest.json"),
            "representation_sha256": independent["representation_sha256"]}
        result.update(identity, independent_sha256=file_hash(original_path), parent_sha256=PARENT)
        if (protocol["checkpoint_sha256"] != PARENT or any(independent.get(key) != value for key, value in identity.items())
                or independent["merge_receipt_sha256"] != file_hash(merge_receipt)
                or load_checkpoint(base).checkpoint_sha256() != PARENT
                or load_checkpoint(merged).checkpoint_sha256() != identity["representation_sha256"]):
            raise ValueError("Frozen protocol, material, source, or merged checkpoint identity differs")
        original = read(merge_receipt)
        artifacts = {"independent": original_path, "protocol": protocol_path, "material_manifest": material / "manifest.json",
            "original_merge": merge_receipt, "merge_authorization": merge_authorization,
            "original_merge_verification": merge_receipt.with_suffix(".verification.json")}
        adapter, reference, stages = _prior_evidence(assessment, material, protocol_path, protocol, independent, identity, artifacts)
        adapter_sha = reference["adapter_file_sha256"]
        expected_merge = {"schema": "weightbridge.supervised_lora_merge_receipt.v1", "status": "ISOLATED_CANDIDATE",
            "base_sha256": PARENT, "candidate_sha256": identity["representation_sha256"], "base_unchanged": True,
            "authorization_sha256": file_hash(merge_authorization), "factor_sha256": reference["factor_sha256"],
            "lineage_accepted": False, "production_deployment": False}
        if any(original.get(key) != value for key, value in expected_merge.items()):
            raise ValueError("Original merge does not bind the locked adapter, parent and authorization")
        _merge_report(merge_receipt, identity, adapter_sha)
        input_hashes = {name: file_hash(path) for name, path in artifacts.items()}
        result.update(input_sha256=input_hashes, locked_adapter_sha256=adapter_sha)
        resources = protocol["resources"]
        limits = {"wall_limit": min(1200, resources["evaluation_worker_wall_seconds"]),
            "cpu_limit": resources["max_cpu_seconds_per_run"], "memory_limit": min(24*1024**3, resources["max_memory_bytes"])}
        result["worker_limits"] = limits

        result["reason"] = "exact_reconstruction_failed"
        reconstructed, rebuilt_receipt = out / "reconstructed", out / "reconstruction-merge.json"
        process = _worker("merge", ["--base", base, "--protocol", protocol_path, "--adapter", adapter,
            "--expected-adapter", adapter_sha, "--authorization", merge_authorization, "--out", reconstructed,
            "--receipt", rebuilt_receipt], "reconstruct", out, limits, attempts)
        merge_check = _merge_report(rebuilt_receipt, identity, adapter_sha)
        _check_worker_body(process, merge_check["resources"])
        process["work"] = {"forward_calls": 0, "backward_calls": 0, "generated_examples": 0}
        result["exact_reconstruction"] = _verify_reconstruction(reconstructed, original, rebuilt_receipt, identity)
        if not result["exact_reconstruction"]["passed"]:
            raise ValueError("Reconstructed checkpoint bytes or numerical merge receipt differ")

        result["reason"] = "protected_generation_replay_failed"
        for stage, (roster, auth, parent, candidate) in stages.items():
            responses = {}
            for label, model, digest, prior in (("reconstructed", reconstructed, identity["representation_sha256"], candidate),
                                                 ("rollback", base, PARENT, parent)):
                name = f"{stage}-{label}"
                evaluation = out / "private-evaluation" / f"{name}.json"
                process = _worker("evaluate", ["--base", model, "--protocol", protocol_path, "--rows", roster,
                    "--out", evaluation, "--expected-checkpoint", digest, "--expected-rows", input_hashes[f"{stage}_roster"],
                    "--stage", stage, "--authorization", auth], name, out, limits, attempts)
                observed = _validated_worker(evaluation, roster, protocol_path, digest, None, stage)
                _check_worker_body(process, observed["resources"])
                measurement = observed["measurement"]
                process["work"] = {key: measurement["resources"][key] for key in
                    ("generate_batches", "batched_decode_steps", "generated_examples", "padded_prompt_token_positions")}
                process["work"].update(backward_calls=0,
                    generated_token_records=sum(len(row["generated_token_ids"]) for row in measurement["rows"]))
                replay[stage] = replay[stage] or {"passed": False}
                replay[stage][label] = {"exact_native_generation": _generation_equal(prior["measurement"], measurement),
                    "observations": len(measurement["rows"]), "evaluation_sha256": file_hash(evaluation),
                    "checkpoint_sha256": digest}
                if not replay[stage][label]["exact_native_generation"]:
                    raise ValueError(f"Exact {stage} {label} generations did not replay")
                responses[label] = measurement
            gate = _gate(responses["rollback"], responses["reconstructed"], protocol, stage)
            prior_gate = independent[stage.replace("-", "_")]
            if not gate["passed"] or any(prior_gate[key] != value for key, value in gate.items()):
                raise ValueError(f"Reconstructed {stage} gate did not replay")
            replay[stage]["passed"] = True

        result["reason"] = "end_of_run_integrity_failed"
        verify_sources(protocol["source_sha256"])
        if ({name: file_hash(path) for name, path in artifacts.items()} != input_hashes
                or any(file_hash(Path(__file__).with_name(name)) != digest for name, digest in code.items())
                or any(load_checkpoint(path).checkpoint_sha256() != digest for path, digest in
                       ((base, PARENT), (merged, identity["representation_sha256"]), (reconstructed, identity["representation_sha256"])))):
            raise ValueError("Source, authorized evidence, or checkpoint bytes changed during final replay")
        result.update(passed=True, disposition="VERIFIED_BASELINE_PASS", reason="all_frozen_behavior_and_reconstruction_gates_replayed",
                      rollback={"passed": True, "checkpoint_sha256": PARENT, "method": "fresh-process original checkpoint re-selection"})
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        if not isinstance(error, Exception):
            interrupted = error
    observed = [record["execution"]["resources"] for record in attempts if record["execution"] is not None]
    host_cpu = time.process_time()-cpu
    result.update(worker_attempts=attempts, resources={"wall_seconds": time.perf_counter()-wall, "host_cpu_seconds": host_cpu,
        "known_cpu_lower_bound_seconds": host_cpu+sum(row["observed_tree_cpu_seconds"] for row in observed),
        "exact_total_cpu_seconds": None, "observed_conservative_peak_rss_bytes": max((row["peak_rss_bytes"] for row in observed), default=None),
        "cpu_scope": "Host process plus observed child trees once. Polling cannot exclude unobserved short-lived descendants."})
    json_dump_atomic(out / "final-verification.json", result)
    if interrupted is not None:
        raise interrupted
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "merged", "merge-receipt", "merge-authorization", "protocol", "material", "assessment", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.merged, args.merge_receipt, args.merge_authorization,
                 args.protocol, args.material, args.assessment, args.out)
    print(result["disposition"], flush=True)
    raise SystemExit(0 if result["passed"] else 1)
