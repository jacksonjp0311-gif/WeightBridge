"""Run one frozen cross-model adaptation trial with a separate reload verifier."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiif.matrix_data import PROTOCOL, load_split, read_json, render_rows
from experiments.phase_iiif.matrix_scoring import compare, evaluate, qualify
from weightbridge.adaptation.model_adapter import select_hf_adapter
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64, apply_safetensors_patch
from weightbridge.provenance import peak_rss_bytes


def _sources():
    names = (
        "experiments/phase_iiif/run_matrix.py", "experiments/phase_iiif/matrix_data.py",
        "experiments/phase_iiif/matrix_scoring.py", "experiments/phase_iiif/matrix_protocol.json",
        "weightbridge/adaptation/model_adapter.py", "weightbridge/adaptation/universal_awl.py",
        "weightbridge/adaptation/universal_update.py", "weightbridge/adaptation/content_scoring.py",
        "weightbridge/inference/huggingface.py", "weightbridge/patches/huggingface.py",
    )
    return {name: file_hash(ROOT / name) for name in names}


@contextmanager
def _open(checkpoint, config, expected_sha, *, gradient_probe=False):
    import torch

    adapter = select_hf_adapter(checkpoint)
    if adapter.checkpoint.checkpoint_sha256() != expected_sha:
        raise ValueError("Checkpoint differs from frozen expected SHA-256")
    settings = config["inference"]
    threads = torch.get_num_threads()
    torch.set_num_threads(settings["cpu_threads"])
    runner = None
    try:
        runner = adapter.load_runner(
            device=settings["device"], dtype=settings["dtype"],
            max_rss_bytes=config["intent"]["limits"]["max_memory_bytes"],
        )
        contract = adapter.discover_model(runner, probe_gradients=gradient_probe)
        if contract.checkpoint_sha256 != expected_sha:
            raise ValueError("Runtime discovery changed the expected checkpoint identity")
        yield adapter, runner, contract
    finally:
        runner = None
        gc.collect()
        torch.set_num_threads(threads)


def _read_material(material, config):
    manifest = read_json(material / "manifest.json")
    if (manifest.get("schema") != "weightbridge.phase_iiif.roster.v1"
            or manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["generator_sha256"] != file_hash(Path(__file__).with_name("matrix_data.py"))):
        raise ValueError("Roster must bind the original frozen protocol and generator")
    return manifest


def _fit(base, adapter, runner, contract, method, intent, host, practice, config, out):
    from weightbridge.adaptation.universal_awl import compile_awl_v3
    from weightbridge.adaptation.universal_update import fit_candidate

    plan = compile_awl_v3(intent, contract, host, method=method)
    settings = config["inference"]
    wall, cpu = time.perf_counter(), time.process_time()
    patch, diagnostic = fit_candidate(
        adapter, runner, plan, practice, max_input_tokens=settings["max_input_tokens"],
        batch_size=settings["batch_size"],
    )
    record = {"plan": plan.record(), "fit": diagnostic,
              "resources": {"fit_wall_seconds": time.perf_counter() - wall,
                            "fit_cpu_seconds": time.process_time() - cpu,
                            "peak_rss_bytes": peak_rss_bytes()},
              "candidate": None}
    if patch is None:
        record["disposition"] = "HOLD"
        return record
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": [plan.region.tensor_name],
              "max_entries": plan.limits.max_update_entries, "max_fraction": .001,
              "max_abs_delta": plan.limits.max_update_norm, "max_rank": plan.limits.max_rank}
    candidate = out / "checkpoints" / method
    receipt = apply_safetensors_patch(base, patch, policy, candidate)
    after = load_checkpoint(candidate)
    name = plan.region.tensor_name
    before_values = _to_float64(adapter.checkpoint.read_raw(name), plan.region.dtype)
    after_values = _to_float64(after.read_raw(name), plan.region.dtype)
    delta = after_values - before_values
    count, norm = int(np.count_nonzero(delta)), float(np.linalg.norm(delta))
    if (receipt["base_sha256"] != contract.checkpoint_sha256
            or receipt["candidate_sha256"] != after.checkpoint_sha256()
            or set(receipt["tensor_receipts"]) != {name} or not 0 < count <= plan.limits.max_update_entries
            or count != receipt["changed_entries"] or norm > plan.limits.max_update_norm + 1e-8
            or not set(after.tensors) == set(adapter.checkpoint.tensors)
            or receipt["tensor_receipts"][name]["candidate_sha256"] != after.tensor_sha256(name)):
        raise ValueError("Stored-precision candidate violates plan or receipt")
    record.update(disposition="CANDIDATE", candidate={
        "checkpoint_sha256": receipt["candidate_sha256"], "receipt": receipt,
        "verified_changed_entries": count, "verified_update_norm": norm,
        "max_abs_stored_delta": float(np.max(np.abs(delta))),
    })
    json_dump_atomic(out / "patches" / f"{method}.json", patch)
    json_dump_atomic(out / "patches" / f"{method}-policy.json", policy)
    json_dump_atomic(out / "patches" / f"{method}-receipt.json", receipt)
    return record


def _worker(checkpoint, expected, material, manifest_sha, out):
    start, cpu = time.perf_counter(), time.process_time()
    config = read_json(PROTOCOL)
    manifest = _read_material(material, config)
    if file_hash(material / "manifest.json") != manifest_sha:
        raise ValueError("Verifier received different roster commitments")
    evaluation = render_rows(load_split(material, "evaluation", manifest, config["roster_counts_per_split"]))
    with _open(checkpoint, config, expected) as (_, runner, contract):
        measured = evaluate(runner, evaluation, config["inference"])
        json_dump_atomic(out, {
            "schema": "weightbridge.phase_iiif.independent_evaluation.v1",
            "checkpoint_sha256": expected,
            "evaluation_sha256": manifest["artifacts"]["evaluation"]["sha256"],
            "contract": contract.to_dict(), "measurement": measured,
            "worker_resources": {"wall_seconds": time.perf_counter() - start,
                                 "cpu_seconds": time.process_time() - cpu,
                                 "peak_rss_bytes": peak_rss_bytes()},
        })


def _independent(path, expected, material, manifest_sha, out, timeout):
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "worker", "--base", str(path),
         "--checkpoint-sha256", expected, "--material", str(material),
         "--manifest-sha256", manifest_sha, "--out", str(out)],
        check=True, timeout=timeout,
    )
    result = read_json(out)
    if result["checkpoint_sha256"] != expected:
        raise ValueError("Independent evaluator reported another checkpoint")
    return result


def run(base, base_sha, material, out):
    from weightbridge.adaptation.meta_controller import Action, AdmissionDecision, EvidenceState
    from weightbridge.adaptation.model_adapter import Capability
    from weightbridge.adaptation.universal_awl import (
        HostAuthorization, ModelEvidence, compile_awl_v3, parse_awl_v3, recommend_awl_v3,
    )

    started, cpu = time.perf_counter(), time.process_time()
    config = read_json(PROTOCOL)
    manifest = _read_material(material, config)
    manifest_sha = file_hash(material / "manifest.json")
    sources = _sources()
    out.mkdir(parents=True, exist_ok=False)
    intent = parse_awl_v3(config["intent"])
    engineering_method = config["engineering_method"]
    engineering_intent = parse_awl_v3({**config["intent"], "methods": [engineering_method]})
    practice_items = load_split(material, "practice", manifest, config["roster_counts_per_split"])
    practice = render_rows(practice_items)
    json_dump_atomic(out / "execution.json", {"schema": "weightbridge.phase_iiif.execution.v1",
                     "base_sha256": base_sha, "manifest_sha256": manifest_sha, "source_sha256": sources})
    with _open(base, config, base_sha, gradient_probe=True) as (adapter, runner, contract):
        print(f"{contract.family}: discovered {len(contract.editable_regions)} editable regions", flush=True)
        contract_record = contract.to_dict()
        latest_affine = max((region for region in contract.editable_regions if region.role == "ffn_output"),
                            key=lambda region: (region.layer_index, region.tensor_name), default=None)
        allowed_tensors = (latest_affine.tensor_name,) if latest_affine else ()
        host = HostAuthorization(base_sha, (*intent.methods, engineering_method), allowed_tensors,
                                 (intent.objective.family, *intent.preserve), intent.practice_data_id,
                                 intent.evaluation_data_id, intent.limits)
        practice_report = evaluate(runner, practice, config["inference"])
        qualification = qualify(practice_report["rows"], config["qualification"])
        if Capability.COMPUTE_GRADIENTS in contract.capabilities:
            plan = compile_awl_v3(intent, contract, host, method="constrained")
            state = EvidenceState(
                all(passed for name, passed in qualification["gates"].items() if name != "persistent_target_errors"),
                len(qualification["persistent_target_errors"]), intent.limits.max_forward_calls, None,
            )
            decision = recommend_awl_v3(plan, ModelEvidence(contract.family, base_sha, state))
        else:
            decision = AdmissionDecision(Action.HOLD, "gradient_capability_unavailable")
        json_dump_atomic(out / "practice.json", {"measurement": practice_report, "qualification": qualification,
                         "meta_decision": decision.record()})
        print(f"{contract.family}: practice {qualification['disposition']}", flush=True)
        records = {engineering_method: _fit(base, adapter, runner, contract, engineering_method,
                   engineering_intent, host, practice, config, out)}
        if qualification["disposition"] == "PASS" and Capability.COMPUTE_GRADIENTS in contract.capabilities:
            for method in intent.methods:
                records[method] = _fit(base, adapter, runner, contract, method, intent, host, practice, config, out)
        else:
            for method in intent.methods:
                records[method] = {"disposition": "HOLD", "reason": "practice_measurement_or_headroom", "candidate": None}
    del runner
    gc.collect()
    json_dump_atomic(out / "candidate-lock.json", {"schema": "weightbridge.phase_iiif.candidate_lock.v1",
                     "practice_sha256": file_hash(out / "practice.json"),
                     "candidates": records, "source_sha256": sources})
    if file_hash(material / "manifest.json") != manifest_sha or _sources() != sources:
        raise ValueError("Frozen protocol or code changed before independent evaluation")
    evaluation_items = load_split(material, "evaluation", manifest, config["roster_counts_per_split"])
    if {abs(item["value"]) for item in practice_items} & {abs(item["value"]) for item in evaluation_items}:
        raise ValueError("Practice and independent arithmetic values overlap")
    timeout = config["inference"]["worker_timeout_seconds"]
    verifier = out / "verification"
    verifier.mkdir()
    baseline = _independent(base, base_sha, material, manifest_sha, verifier / "base.json", timeout)
    comparisons = {}
    for method, record in records.items():
        if record["candidate"] is None:
            continue
        digest = record["candidate"]["checkpoint_sha256"]
        measured = _independent(out / "checkpoints" / method, digest, material, manifest_sha,
                                verifier / f"{method}.json", timeout)
        if measured["contract"]["family"] != baseline["contract"]["family"]:
            raise ValueError("Candidate reload reports a different model family")
        outcome = compare(baseline["measurement"]["rows"], measured["measurement"]["rows"],
                          intent.objective.min_gain, intent.risk.max_regressions)
        if method == engineering_method:
            outcome["behavior_gate"] = outcome["disposition"]
            outcome["disposition"] = "ENGINEERING_ONLY"
        comparisons[method] = {**outcome, "summary": measured["measurement"]["summary"],
                               "worker_resources": measured["worker_resources"]}
    rollback = _independent(base, base_sha, material, manifest_sha, verifier / "rollback.json", timeout)
    restored_rows = rollback["measurement"]["rows"] == baseline["measurement"]["rows"]
    if (load_checkpoint(base).checkpoint_sha256() != base_sha or _sources() != sources
            or file_hash(material / "manifest.json") != manifest_sha):
        raise ValueError("Original checkpoint or frozen material changed during research")
    engineering_ok = engineering_method in comparisons and restored_rows
    if not engineering_ok:
        engineering_status = "FAIL"
    else:
        engineering_status = "PASS"
    scientific = "HOLD"
    if qualification["disposition"] == "PASS" and all(method in comparisons for method in intent.methods):
        scientific = "PASS" if any(comparisons[method]["disposition"] == "PASS" for method in intent.methods) else "FAIL"
    report = {
        "schema": "weightbridge.phase_iiif.matrix_report.v1", "base_sha256": base_sha,
        "contract": contract_record, "intent": config["intent"], "manifest_sha256": manifest_sha,
        "practice": qualification, "candidates": records, "comparison": comparisons,
        "meta_decision": decision.record(),
        "baseline_summary": baseline["measurement"]["summary"],
        "baseline_resources": baseline["worker_resources"],
        "rollback": {"checkpoint_sha256": rollback["checkpoint_sha256"], "exact_replay": restored_rows,
                     "worker_resources": rollback["worker_resources"]},
        "engineering_disposition": engineering_status, "scientific_disposition": scientific,
        "sequential_disposition": "HOLD",
        "resources": {"host_cpu_seconds": time.process_time() - cpu, "wall_seconds": time.perf_counter() - started,
                      "host_peak_rss_bytes": peak_rss_bytes()},
        "artifacts": {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))
                      if "checkpoints" not in path.relative_to(out).parts},
    }
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.command == "worker":
        if not args.manifest_sha256:
            parser.error("worker requires --manifest-sha256")
        _worker(args.base, args.checkpoint_sha256, args.material, args.manifest_sha256, args.out)
    else:
        outcome = run(args.base, args.checkpoint_sha256, args.material, args.out)
        print(json.dumps({field: outcome[field] for field in ("engineering_disposition", "scientific_disposition",
                                                               "sequential_disposition")}, indent=2))
