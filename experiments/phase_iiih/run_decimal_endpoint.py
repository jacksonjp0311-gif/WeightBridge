"""Run the frozen endpoint-alignment study and separate checkpoint verifier."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from math import ceil, sqrt
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiig.run_signed_decimal import qualify_negative
from experiments.phase_iiih.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, build_decimal_fit_rows, compare_endpoint, decimal_endpoint,
    endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.endpoint_policy import EndpointEvidence, select_endpoint_action
from weightbridge.adaptation.meta_controller import EvidenceState
from weightbridge.adaptation.model_adapter import Capability
from weightbridge.adaptation.universal_awl import HostAuthorization, compile_awl_v3, parse_awl_v3
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64, apply_safetensors_patch
from weightbridge.provenance import peak_rss_bytes


def read_json(path):
    return json.loads(path.read_bytes())


def sources():
    paths = (
        "experiments/phase_iiih/run_decimal_endpoint.py", "experiments/phase_iiih/decimal_roster.py",
        "experiments/phase_iiih/decimal_protocol.json", "experiments/phase_iiif/run_matrix.py",
        "experiments/phase_iiig/run_signed_decimal.py", "experiments/phase_iiig/run_trial.py",
        "experiments/phase_iiig/endpoint.py", "experiments/phase_iiig/decimal_endpoint.py",
        "weightbridge/adaptation/behavior_endpoint.py", "weightbridge/adaptation/endpoint_update.py",
        "weightbridge/adaptation/endpoint_policy.py", "weightbridge/adaptation/content_scoring.py",
        "weightbridge/adaptation/universal_update.py", "weightbridge/adaptation/universal_awl.py",
        "weightbridge/adaptation/meta_controller.py", "weightbridge/adaptation/model_adapter.py",
        "weightbridge/inference/huggingface.py", "weightbridge/patches/huggingface.py",
    )
    return {name: file_hash(ROOT / name) for name in paths}


def endpoint(config):
    return decimal_endpoint(
        config["inference"], required_wordings=tuple(config["endpoint"]["wordings"]),
        max_candidate_evaluations=config["endpoint"]["max_candidate_evaluations"],
    )


def qualify(measurement, config):
    # Qualification counts a parsed comparison number only if it is one of the options.
    rows = [{**row, "exact": row["exact"] if row["valid"] else None} for row in measurement["rows"]]
    return qualify_negative(rows, config["target_family"], config["qualification"])


def _resources(started, cpu):
    return {"wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()}


def _worker(path, expected, material, split, output, manifest_sha, *, seal=None, hidden=None):
    started, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    if file_hash(material / "manifest.json") != manifest_sha or output.exists():
        raise ValueError("Verifier requires the frozen roster and a new output path")
    if split == "hidden":
        if seal is None or hidden is None or hidden.resolve().is_relative_to(ROOT):
            raise ValueError("Hidden verification requires a candidate seal and separate private material")
        lock = read_json(seal)
        if (lock["manifest_sha256"] != manifest_sha or lock["sources"] != sources()
                or lock["base_sha256"] != config["checkpoint_sha256"]
                or not lock["winner_sha256"] or expected not in {lock["base_sha256"], lock["winner_sha256"]}):
            raise ValueError("Hidden verification is outside the immutable candidate lock")
    elif seal is not None or hidden is not None:
        raise ValueError("Selection verification cannot access private material")
    rows = load_split(material, split, manifest, hidden_path=hidden)
    with _open(path, config, expected) as (_, runner, capabilities):
        measured = measure_endpoint(runner, rows, endpoint(config))
    json_dump_atomic(output, {
        "schema": "weightbridge.phase_iiih.verifier.v1", "checkpoint_sha256": expected,
        "split": split, "roster_sha256": manifest["splits"][split]["sha256"],
        "capabilities": capabilities.to_dict(), "measurement": measured,
        "resources": _resources(started, cpu),
    })


def independent(path, expected, material, split, output, manifest, config, *, seal=None, hidden=None):
    args = [sys.executable, str(Path(__file__)), "worker", "--base", str(path), "--sha256", expected,
            "--material", str(material), "--out", str(output), "--split", split,
            "--manifest-sha256", file_hash(material / "manifest.json")]
    if seal is not None:
        args.extend(["--seal", str(seal), "--hidden", str(hidden)])
    subprocess.run(args, check=True, timeout=config["inference"]["worker_timeout_seconds"])
    observed = read_json(output)
    if (observed["checkpoint_sha256"] != expected or observed["split"] != split
            or observed["roster_sha256"] != manifest["splits"][split]["sha256"]):
        raise ValueError("Verifier returned another checkpoint or roster")
    return observed


def fit_branches(adapter, runner, plan, practice_rows, practice, qualification, config, out):
    from weightbridge.adaptation.endpoint_update import fit_endpoint_probes

    fit_limits = config["fit_limits"]
    targets = list(qualification["repeatable_errors"])[:fit_limits["max_repeatable_error_items"]]
    solved = qualification["solved"]
    preserved = [*solved["target"][:fit_limits["max_solved_target_items"]], *solved["positive"], *solved["preserve"]]
    control_items = {solved["positive"][0], solved["preserve"][0]}
    control_ids = {row["id"] for row in practice_rows if row["item"] in control_items}
    contract = endpoint(config)
    budget = CandidateEvaluationBudget(contract)
    design = config["probe_design"]

    def evaluate():
        candidate = measure_endpoint(runner, practice_rows, contract, candidate_budget=budget)
        guard = compare_endpoint(practice, candidate, contract, min_repeatable_corrections=1)
        stable_negative_corrections = sorted(set(guard["corrected_repeatable_target_items"])
                                             & set(qualification["repeatable_errors"]))
        guard["corrected_repeatable_negative_items"] = stable_negative_corrections
        guard["passed"] = bool(guard["passed"] and stable_negative_corrections)
        return {"guard": guard, "measurement": candidate}

    results, candidates = {}, {}
    doses = tuple(design["doses"])
    generation_upper = len(doses) * ceil(len(practice_rows) / contract.batch_size) * contract.max_new_tokens
    for branch, mode in (("legacy", "legacy_decoy_control"), ("corrected", "semantic")):
        fit = build_decimal_fit_rows(runner, practice, contract, target_items=targets,
                                     preservation_items=preserved, max_rows=fit_limits["max_rows"], mode=mode)
        rows = [row.to_solver_row() for row in fit]
        prefix_indices = tuple(index for index, row in enumerate(rows) if row["id"] in control_ids)
        if len(prefix_indices) != 4:
            raise ValueError("Distribution probe requires the four frozen positive and copy prompts")
        numerical_upper = (1 + len(doses)) * (1 + ceil(2 * len(rows) / contract.batch_size))
        if generation_upper + numerical_upper > plan.limits.max_forward_calls:
            raise ValueError("Practice generation plus numerical probes exceed the host forward budget")
        json_dump_atomic(out / f"fit-{branch}.json", {"rows": [asdict(row) for row in fit],
                         "prefix_row_indices": prefix_indices})
        print(f"{branch}: fitting {len(rows)} presentations and screening {len(doses)} doses", flush=True)
        patches, result = fit_endpoint_probes(
            adapter, endpoint_runner(runner, contract), plan, rows, evaluate_candidate=evaluate,
            baseline_endpoint=practice, doses=doses, max_input_tokens=contract.max_input_tokens,
            batch_size=contract.batch_size, preservation_prefix_rows=prefix_indices,
            kl_weight=design["kl_weight"], norm_weight=design["norm_weight"],
        )
        resources = result["resources"]
        resources["candidate_decode_steps"] = sum(
            probe["endpoint"]["measurement"]["resources"]["batched_decode_steps"]
            for probe in result["probes"] if "endpoint" in probe)
        resources["total_forward_calls_including_decode"] = (
            resources["candidate_decode_steps"] + resources["total_numerical_forward_calls"])
        if resources["total_forward_calls_including_decode"] > plan.limits.max_forward_calls:
            raise ValueError("Measured practice probes exceeded the host forward budget")
        json_dump_atomic(out / f"probes-{branch}.json", result)
        print(f"{branch}: selected {result['selected']}", flush=True)
        results[branch] = result
        if branch == "corrected":
            for probe_id in {result["selected"][key] for key in ("endpoint", "distribution")} - {None}:
                candidates[probe_id] = patches[probe_id]
    return results, candidates


def surrogate_fidelity(branches):
    records = []
    for branch, result in branches.items():
        for probe in result["probes"]:
            if not probe.get("margin") or not probe.get("endpoint"):
                continue
            margin, guard = probe["margin"], probe["endpoint"]["guard"]
            records.append({"branch": branch, "probe": probe["id"], "surrogate_gain": margin["surrogate_gain"],
                            "target_item_gain": guard["target_item_gain"], "practice_pass": guard["passed"],
                            "regressions": len(guard["prohibited_regressions"]),
                            "lost_validity": len(guard["lost_valid_presentations"])})
    conditional = [record for record in records if record["surrogate_gain"] > 0]
    total = len(conditional)
    gains = sum(record["target_item_gain"] > 0 for record in conditional)
    interval = None
    if total:
        p, z = gains / total, 1.959963984540054
        center = (p + z**2 / (2 * total)) / (1 + z**2 / total)
        half = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / (1 + z**2 / total)
        interval = [max(0.0, center - half), min(1.0, center + half)]
    return {"pairs": records, "behavior_improvements_given_surrogate_improvement": [gains, total],
            "wilson_95_descriptive": interval,
            "limitation": "Doses share the parent, practice roster and two fitted directions. The interval assumes independence and is descriptive only."}


def materialize(base, out, plan, patches, corrected):
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": [plan.region.tensor_name],
              "max_entries": plan.limits.max_update_entries, "max_fraction": .001,
              "max_abs_delta": plan.limits.max_update_norm, "max_rank": plan.limits.max_rank}
    records = {}
    parent = load_checkpoint(base)
    region = plan.region
    original = _to_float64(parent.read_raw(region.tensor_name), region.dtype)
    expected = {probe["id"]: probe for probe in corrected["probes"]}
    for identifier, patch in sorted(patches.items()):
        probe = expected[identifier]
        if not probe["endpoint"]["guard"]["passed"] or not probe["restored_exactly"]:
            raise ValueError("Checkpoint materialization requires a passed and reversible endpoint probe")
        path = out / "checkpoints" / identifier
        receipt = apply_safetensors_patch(base, patch, policy, path)
        candidate = load_checkpoint(path)
        stored = _to_float64(candidate.read_raw(region.tensor_name), region.dtype)
        delta = stored - original
        count, norm = int(np.count_nonzero(delta)), float(np.linalg.norm(delta))
        tensor_sha = candidate.tensor_sha256(region.tensor_name)
        if (receipt["base_sha256"] != plan.checkpoint_sha256
                or receipt["candidate_sha256"] != candidate.checkpoint_sha256()
                or set(receipt["tensor_receipts"]) != {region.tensor_name}
                or tensor_sha != receipt["tensor_receipts"][region.tensor_name]["candidate_sha256"]
                or tensor_sha != probe["candidate_tensor_sha256"]
                or count != receipt["changed_entries"] or count != probe["changed_entries"]
                or not np.isclose(norm, probe["realized_update_norm"], rtol=0, atol=1e-10)):
            raise ValueError("Materialized checkpoint differs from the rounded live probe")
        json_dump_atomic(out / "patches" / f"{identifier}.json", patch)
        json_dump_atomic(out / "patches" / f"{identifier}-policy.json", policy)
        json_dump_atomic(out / "patches" / f"{identifier}-receipt.json", receipt)
        records[identifier] = {"checkpoint_sha256": receipt["candidate_sha256"], "tensor_sha256": tensor_sha,
                               "changed_entries": count, "realized_update_norm": norm}
    return records


def _paired(base, candidate, config, gate):
    result = compare_endpoint(base, candidate, endpoint(config), min_target_gain=gate["min_target_gain"])
    valid = sum(group["valid"] for group in result["candidate"].values())
    total = len(candidate["rows"])
    result["candidate_valid_rate"] = valid / total
    result["passed"] = bool(result["passed"] and valid >= total * config["endpoint"]["min_valid_rate"])
    return result


def _meta(qualification, fidelity, branches, *, independent_pass=None, verified=False):
    pairs = fidelity["pairs"]
    gains, surrogates = fidelity["behavior_improvements_given_surrogate_improvement"]
    all_probes = [probe for branch in branches.values() for probe in branch["probes"]]
    evidence = EndpointEvidence(
        measured=len(all_probes), effective=len(pairs), practice_passes=sum(row["practice_pass"] for row in pairs),
        surrogate_improvements=surrogates, behavior_gains_among_surrogate_improvements=gains,
        regressing_probes=sum(bool(row["regressions"] or row["lost_validity"]) for row in pairs),
        excessive_probes=sum(probe.get("reason") == "realized_norm_exceeds_budget" for probe in all_probes),
        independent_pass=independent_pass, hash_and_reload_verified=verified,
    )
    state = EvidenceState(
        all(value for key, value in qualification["gates"].items() if key != "repeatable_errors"),
        len(qualification["repeatable_errors"]), 0, None,
    )
    return {"inputs": asdict(evidence), "decision": select_endpoint_action(state, evidence).record()}


def _assert_frozen(base, base_sha, material, manifest_sha, source_sha):
    if (load_checkpoint(base).checkpoint_sha256() != base_sha
            or file_hash(material / "manifest.json") != manifest_sha or sources() != source_sha):
        raise ValueError("Original checkpoint, roster, or executable sources changed")


def _artifacts(out):
    return {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))
            if "checkpoints" not in path.relative_to(out).parts and path.name != "report.json"}


def run(base, material, out, selection_base_path=None):
    started, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    source_sha, manifest_sha = sources(), file_hash(material / "manifest.json")
    base_sha = config["checkpoint_sha256"]
    rows = load_split(material, "practice", manifest)
    out.mkdir(parents=True, exist_ok=False)
    branches, candidates = {}, {}
    plan = None
    reason = "practice_measurement_or_headroom_failed"
    with _open(base, config, base_sha, gradient_probe=True) as (adapter, runner, capabilities):
        measurement = measure_endpoint(runner, rows, endpoint(config))
        qualification = qualify(measurement, config)
        json_dump_atomic(out / "practice.json", {"measurement": measurement, "qualification": qualification})
        print(f"Qwen practice endpoint: {qualification['disposition']}, {len(qualification['repeatable_errors'])} repeated errors", flush=True)
        if qualification["disposition"] == "PASS":
            reason = "gradient_capability_unavailable"
            if Capability.COMPUTE_GRADIENTS in capabilities.capabilities:
                intent = parse_awl_v3(config["intent"])
                host = HostAuthorization(base_sha, intent.methods, tuple(r.tensor_name for r in capabilities.editable_regions),
                                         (intent.objective.family, *intent.preserve), intent.practice_data_id,
                                         intent.evaluation_data_id, intent.limits)
                plan = compile_awl_v3(intent, capabilities, host, method="constrained")
                branches, patches = fit_branches(adapter, runner, plan, rows, measurement, qualification, config, out)
                if not all(branch.get("parent_restored_exactly", not branch["probes"]) for branch in branches.values()):
                    raise ValueError("At least one direction did not restore the exact live parent")
                candidates = materialize(base, out, plan, patches, branches["corrected"])
                reason = "no_endpoint_qualified_probe" if not candidates else "no_independent_selection_gain"
    del runner
    gc.collect()
    return _finalize_run(base, material, out, selection_base_path, config, manifest, source_sha, manifest_sha,
                         base_sha, qualification, capabilities, plan, reason, branches, candidates, started, cpu)


def _finalize_run(base, material, out, selection_base_path, config, manifest, source_sha, manifest_sha,
                  base_sha, qualification, capabilities, plan, reason, branches, candidates, started, cpu):
    verification = out / "verification"
    paired, verifier_resources = {}, {}
    baseline = None
    if selection_base_path is not None:
        baseline = read_json(selection_base_path)
        if (baseline["checkpoint_sha256"] != base_sha or baseline["split"] != "selection"
                or baseline["roster_sha256"] != manifest["splits"]["selection"]["sha256"]
                or baseline["measurement"]["endpoint"] != endpoint(config).manifest()):
            raise ValueError("Separate parent verifier record does not match the frozen baseline")
        json_dump_atomic(verification / "selection-base.json", baseline)
    if candidates:
        if baseline is None:
            baseline = independent(base, base_sha, material, "selection", verification / "selection-base.json", manifest, config)
        for identifier, candidate in candidates.items():
            result = independent(out / "checkpoints" / identifier, candidate["checkpoint_sha256"], material,
                                 "selection", verification / f"selection-{identifier}.json", manifest, config)
            paired[identifier] = _paired(baseline["measurement"], result["measurement"], config, config["selection_gate"])
            verifier_resources[identifier] = result["resources"]
    if baseline is not None:
        verifier_resources["base"] = baseline["resources"]
    eligible = [identifier for identifier, guard in paired.items() if guard["passed"]]
    winner = min(eligible, key=lambda key: (-paired[key]["target_item_gain"], candidates[key]["realized_update_norm"], key), default=None)
    lock = {"schema": "weightbridge.phase_iiih.candidate_lock.v1", "manifest_sha256": manifest_sha,
            "sources": source_sha, "base_sha256": base_sha, "winner": winner,
            "winner_sha256": candidates[winner]["checkpoint_sha256"] if winner else None,
            "practice_sha256": file_hash(out / "practice.json"), "selection": paired, "candidates": candidates}
    json_dump_atomic(out / "candidate-lock.json", lock)
    _assert_frozen(base, base_sha, material, manifest_sha, source_sha)
    return _write_report(out, base_sha, source_sha, manifest_sha, qualification, baseline, capabilities, plan, reason,
                         branches, candidates, winner, verifier_resources, started, cpu, config)


def _write_report(out, base_sha, source_sha, manifest_sha, qualification, baseline, capabilities, plan, reason,
                  branches, candidates, winner, verifier_resources, started, cpu, config):
    fidelity = surrogate_fidelity(branches)
    report = {
        "schema": "weightbridge.phase_iiih.endpoint_report.v1", "behavioral_qualification": qualification["disposition"],
        "single_cycle": "FAIL" if candidates and not winner else "HOLD",
        "reason": "awaiting_host_hidden_verification" if winner else reason, "hidden_opened": False,
        "base_sha256": base_sha, "manifest_sha256": manifest_sha, "sources": source_sha,
        "capabilities": capabilities.to_dict(), "plan": plan.record() if plan else None, "practice": qualification,
        "selection_baseline_qualification": qualify(baseline["measurement"], config) if baseline else None,
        "arms": {"A": branches.get("legacy", {}).get("selected", {}).get("margin"),
                 "B": branches.get("corrected", {}).get("selected", {}).get("margin"),
                 "C": branches.get("corrected", {}).get("selected", {}).get("endpoint"),
                 "D": branches.get("corrected", {}).get("selected", {}).get("distribution")},
        "surrogate_fidelity": fidelity,
        "meta": _meta(qualification, fidelity, branches, independent_pass=False if candidates and not winner else None),
        "candidate_lock_sha256": file_hash(out / "candidate-lock.json"), "selection_winner": winner,
        "restoration": {"checkpoint_sha256": base_sha, "live_exact_by_branch":
                        {key: branch.get("parent_restored_exactly") for key, branch in branches.items()}},
        "resources": {"host": _resources(started, cpu), "independent_verifiers": verifier_resources,
                      "branches": {key: record["resources"] for key, record in branches.items()}},
        "artifacts": _artifacts(out),
    }
    json_dump_atomic(out / ("preparation.json" if winner else "report.json"), report)
    return {key: report[key] for key in ("behavioral_qualification", "single_cycle", "reason", "arms")}


def finish(base, material, out, hidden):
    from weightbridge.adaptation.endpoint_lineage import record_endpoint_lineage
    from weightbridge.adaptation.lineage import init_lineage

    started, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    report = read_json(out / "preparation.json")
    lock_path = out / "candidate-lock.json"
    lock = read_json(lock_path)
    winner = lock["winner"]
    if (out.resolve().is_relative_to(ROOT) or hidden.resolve().is_relative_to(ROOT) or not winner
            or report["candidate_lock_sha256"] != file_hash(lock_path)
            or report["selection_winner"] != winner or not lock["selection"][winner]["passed"]
            or file_hash(out / "practice.json") != lock["practice_sha256"] or (out / "report.json").exists()):
        raise ValueError("Hidden verification requires a fresh selection-locked private workspace")
    _assert_frozen(base, lock["base_sha256"], material, lock["manifest_sha256"], lock["sources"])
    candidate = out / "checkpoints" / winner
    verification = out / "verification"
    paths = {name: verification / f"hidden-{name}.json" for name in ("base", "candidate", "replay", "rollback")}
    records = {}

    def verify(label, path, digest):
        records[label] = independent(path, digest, material, "hidden", paths[label], manifest, config,
                                     seal=lock_path, hidden=hidden)
        return records[label]["measurement"]

    before = verify("base", base, lock["base_sha256"])
    after = verify("candidate", candidate, lock["winner_sha256"])
    gate = _paired(before, after, config, config["hidden_gate"])
    report.update(hidden_opened=True, single_cycle="FAIL", reason="independent_hidden_endpoint_failed")
    # Keep item-level hidden evidence in the host workspace. The public summary carries counts and digests.
    report["hidden"] = {"passed": gate["passed"], "target_item_gain": gate["target_item_gain"],
                        "regressions": len(gate["prohibited_regressions"]),
                        "candidate_valid_rate": gate["candidate_valid_rate"]}
    if gate["passed"]:
        replay = verify("replay", candidate, lock["winner_sha256"])
        rollback = verify("rollback", base, lock["base_sha256"])
        report["restart_exact"] = after["rows"] == replay["rows"]
        report["rollback_exact"] = before["rows"] == rollback["rows"]
        report["reason"] = "independent_reload_or_rollback_changed"
        if report["restart_exact"] and report["rollback_exact"]:
            lineage = out / "checkpoints" / "lineage"
            init_lineage(lineage, base)
            record_endpoint_lineage(lineage, base=base, candidate=candidate, contract=endpoint(config),
                hidden_roster=hidden, manifest=material / "manifest.json", base_report=paths["base"],
                candidate_report=paths["candidate"], replay_report=paths["replay"], rollback_report=paths["rollback"],
                selection_lock=lock_path, cycle_id="phase-iii-h-decimal-endpoint-v1",
                min_target_gain=config["hidden_gate"]["min_target_gain"])
            report.update(single_cycle="PASS", reason="independent_hidden_endpoint_and_reload_pass",
                          accepted_candidate_sha256=lock["winner_sha256"])
    _assert_frozen(base, lock["base_sha256"], material, lock["manifest_sha256"], lock["sources"])
    report["resources"]["hidden_host"] = _resources(started, cpu)
    report["resources"]["hidden_verifiers"] = {label: record["resources"] for label, record in records.items()}
    report["artifacts"] = _artifacts(out)
    json_dump_atomic(out / "report.json", report)
    return {key: report[key] for key in ("behavioral_qualification", "single_cycle", "reason", "arms")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "run", "finish"))
    for name in ("base", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--split", choices=("selection", "hidden"))
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--hidden", type=Path)
    parser.add_argument("--selection-base", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if not all((args.sha256, args.manifest_sha256, args.split)):
            parser.error("worker requires checkpoint and manifest hashes and a split")
        _worker(args.base, args.sha256, args.material, args.split, args.out, args.manifest_sha256,
                seal=args.seal, hidden=args.hidden)
    elif args.command == "run":
        if args.hidden is not None:
            parser.error("the fitting command never receives a hidden material path")
        print(json.dumps(run(args.base, args.material, args.out, args.selection_base)), flush=True)
    else:
        if args.hidden is None:
            parser.error("finish requires the host-held hidden path")
        print(json.dumps(finish(args.base, args.material, args.out, args.hidden)), flush=True)


if __name__ == "__main__":
    main()
