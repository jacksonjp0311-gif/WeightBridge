"""Fresh-process transfer, independent selection, and host-only hidden closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
import sys
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint
from experiments.phase_iiil.semantic_diagnostics import compare_semantic, summarize
from experiments.phase_iiil.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiil.source_manifest import ROOT, source_hashes
from weightbridge.adaptation.behavior_endpoint import _observation, measure_endpoint
from weightbridge.adaptation.endpoint_lineage import record_endpoint_lineage
from weightbridge.adaptation.fork_policy import ForkDecision
from weightbridge.adaptation.lineage import init_lineage
from weightbridge.adaptation.robust_jspace import JSpaceEvidence, jspace_recommendation
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


LOCK_SCHEMA = "weightbridge.phase_iiih.candidate_lock.v1"
WORKER_SCHEMA = "weightbridge.phase_iiih.verifier.v1"
TRANSFER = "transfer-validation"
PUBLIC_SPLITS = (TRANSFER, "selection")
_FAILURES = (subprocess.SubprocessError, OSError, ValueError, KeyError, RuntimeError)


def _read(path):
    return json.loads(path.read_bytes())


def _private(path):
    return not path.resolve().is_relative_to(ROOT)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_fit_report(out, material):
    report = _read(out / "fit-report.json")
    config = _read(PROTOCOL)
    read_manifest(material)
    if (report.get("schema") != "weightbridge.phase_iiil.fit.v1" or report["sources"] != source_hashes()
            or report["manifest_sha256"] != file_hash(material / "manifest.json")
            or report["checkpoint_sha256"] != config["checkpoint_sha256"]
            or report["optimization_splits"] != ["adaptation", "preservation-fit", "calibration"]
            or report["hidden_opened"] is not False):
        raise ValueError("Fit report does not bind the frozen source, material and base identity")
    for name, digest in report["artifacts"].items():
        path = (out / name).resolve()
        if (not name or not path.is_relative_to(out.resolve()) or Path(name).parts[0] in {"checkpoints", "verification"}
                or not _digest(digest) or file_hash(path) != digest):
            raise ValueError("Fit artifact does not match its frozen digest")
    candidates = report["candidates"]
    if len(candidates) > config["intent"]["limits"]["max_candidates"]:
        raise ValueError("Too many practice candidates for the frozen budget")
    states = set()
    for arm, candidate in candidates.items():
        if (re.fullmatch(r"[A-Z][1-9][0-9]*", arm) is None
                or not all(_digest(candidate[key]) for key in ("checkpoint_sha256", "tensor_sha256", "receipt_sha256"))
                or candidate["checkpoint_sha256"] == config["checkpoint_sha256"]
                or not isfinite(candidate["realized_update_norm"])
                or not 0 < candidate["realized_update_norm"] <= config["intent"]["limits"]["max_update_norm"]
                or type(candidate["changed_entries"]) is not int
                or not 0 < candidate["changed_entries"] <= config["intent"]["limits"]["max_update_entries"]
                or f"patches/{arm}-receipt.json" not in report["artifacts"]
                or f"arm-{arm}.json" not in report["artifacts"]
                or candidate["tensor_sha256"] in states):
            raise ValueError("Unbound, duplicate, or out-of-budget candidate in fit report")
        receipt = _read(out / "patches" / f"{arm}-receipt.json")
        # Native checkpoint receipts use explicit LF, while trial JSON uses host newlines.
        encoded = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        probe = _read(out / f"arm-{arm}.json")["probe"]
        if (hashlib.sha256(encoded).hexdigest() != candidate["receipt_sha256"]
                or probe.get("restored_exactly") is not True
                or probe.get("candidate_tensor_sha256") != candidate["tensor_sha256"]
                or probe.get("endpoint", {}).get("guard", {}).get("passed") is not True):
            raise ValueError("Candidate lacks an exact fit receipt and a passing restored practice probe")
        states.add(candidate["tensor_sha256"])
    return report


def _check_lock(path, digest, material, config, expected=None):
    if not _private(path) or file_hash(path) != digest:
        raise ValueError("Hidden verification requires a private hash-bound finalist lock")
    lock = _read(path)
    winner = lock.get("winner")
    if (lock.get("schema") != LOCK_SCHEMA or not winner
            or lock["base_sha256"] != config["checkpoint_sha256"]
            or lock["manifest_sha256"] != file_hash(material / "manifest.json")
            or lock["sources"] != source_hashes(verifier=True)
            or lock["transfer"].get(winner, {}).get("passed") is not True
            or lock["selection"].get(winner, {}).get("passed") is not True
            or lock["winner_sha256"] != lock["candidates"].get(winner, {}).get("checkpoint_sha256")
            or expected is not None and expected not in (lock["base_sha256"], lock["winner_sha256"])):
        raise ValueError("Finalist lock does not authorize this independent evaluation")
    return lock


def worker(checkpoint, sha, material, split, out, manifest_sha, *, lock=None, lock_sha=None, hidden=None):
    started, cpu = time.perf_counter(), time.process_time()
    manifest, config = read_manifest(material), _read(PROTOCOL)
    if out.exists() or file_hash(material / "manifest.json") != manifest_sha:
        raise ValueError("Verifier requires frozen material and a fresh output path")
    if split == "hidden":
        if not (lock and lock_sha and hidden and _private(hidden) and _private(out)
                and out.parent.resolve() == (lock.parent / "verification").resolve()):
            raise ValueError("Hidden data requires a private workspace and a finalist lock")
        verified = _check_lock(lock, lock_sha, material, config, sha)
        replay_lock(lock.parent, material, verified)
    elif split not in PUBLIC_SPLITS or any(value is not None for value in (lock, lock_sha, hidden)):
        raise ValueError("Public verification cannot receive hidden data")
    rows = load_split(material, split, manifest, hidden_path=hidden)
    with _open(checkpoint, config, sha) as (_, runner, capabilities):
        measured = measure_endpoint(runner, rows, endpoint(config))
    json_dump_atomic(out, {"schema": WORKER_SCHEMA, "split": split, "checkpoint_sha256": sha,
                          "roster_sha256": manifest["splits"][split]["sha256"], "manifest_sha256": manifest_sha,
                          "measurement": measured, "architecture": capabilities.architecture,
                          "resources": _resources(started, cpu)})


def _validate_report(path, sha, material, manifest, config, split, *, hidden=None):
    report = _read(path)
    expected = load_split(material, split, manifest, hidden_path=hidden)
    measured, contract = report["measurement"], endpoint(config)
    if (report.get("schema") != WORKER_SCHEMA or report.get("split") != split
            or report.get("checkpoint_sha256") != sha
            or report.get("manifest_sha256") != file_hash(material / "manifest.json")
            or report.get("roster_sha256") != manifest["splits"][split]["sha256"]
            or measured.get("endpoint") != contract.manifest()
            or len(expected) != len(measured["rows"])):
        raise ValueError("Verifier output differs from the frozen checkpoint, endpoint, or roster")
    for row, frozen in zip(measured["rows"], expected, strict=True):
        observed = _observation(row, contract.question(frozen), contract)
        if any(row.get(key) != value for key, value in {**frozen, **observed}.items()):
            raise ValueError("Verifier output altered frozen data or the deterministic parser classification")
    return report


def _launch(path, sha, material, split, out, manifest, config, *, lock=None, lock_sha=None, hidden=None):
    args = [sys.executable, "-m", "experiments.phase_iiil.verification", "worker", "--base", str(path.resolve()),
            "--sha256", sha, "--material", str(material.resolve()), "--split", split, "--out", str(out.resolve()),
            "--manifest-sha256", file_hash(material / "manifest.json")]
    if lock is not None:
        args += ["--lock", str(lock.resolve()), "--lock-sha256", lock_sha, "--hidden", str(hidden.resolve())]
    subprocess.run(args, cwd=ROOT, check=True, timeout=config["inference"]["worker_timeout_seconds"])
    return _validate_report(out, sha, material, manifest, config, split, hidden=hidden)


def transfer_gate(before, after, config):
    """Measure gains but advance solely by the frozen preservation requirements."""
    gate = compare_semantic(before, after)
    protected = sum(row["correct"] for row in before["rows"])
    return {**gate, "protected_presentations": protected,
            "candidate_valid_rate": sum(row["valid"] for row in after["rows"]) / len(after["rows"]),
            "passed": bool(protected and not gate["regressions"] and not gate["lost_valid"])}


def selection_gate(before, after, config, split="selection"):
    if split not in ("selection", "hidden") or config[f"{split}_gate"]["max_regressions"] != 0:
        raise ValueError("Independent semantic verification requires a frozen zero-regression gate")
    return compare_semantic(before, after, min_comparison_gain=config[f"{split}_gate"]["min_target_gain"])


def _rank(selection, candidates):
    return min((name for name, gate in selection.items() if gate["passed"]), default=None,
               key=lambda name: (-selection[name]["target_gain"], candidates[name]["realized_update_norm"], name))


def _checkpoint(out, arm, candidate, base_sha):
    directory = out / "checkpoints" / arm
    if not directory.resolve().is_relative_to(out.resolve()) or directory.is_symlink():
        raise ValueError("Candidate checkpoint must be stored in its private trial slot")
    path = directory / ".weightbridge_receipt.json"
    receipt = _read(path)
    tensors = receipt["tensor_receipts"]
    if (file_hash(path) != candidate["receipt_sha256"] or receipt["schema"] != "weightbridge.receipt.v2"
            or receipt != _read(out / "patches" / f"{arm}-receipt.json")
            or receipt["status"] != "candidate_only_not_promoted" or receipt["base_sha256"] != base_sha
            or receipt["candidate_sha256"] != candidate["checkpoint_sha256"]
            or receipt["changed_entries"] != candidate["changed_entries"] or len(tensors) != 1):
        raise ValueError("Candidate receipt differs from the materialized fit record")
    checkpoint = load_checkpoint(directory)
    tensor = next(iter(tensors))
    if (tensors[tensor]["candidate_sha256"] != candidate["tensor_sha256"]
            or checkpoint.tensor_sha256(tensor) != candidate["tensor_sha256"]
            or checkpoint.checkpoint_sha256() != candidate["checkpoint_sha256"]):
        raise ValueError("Materialized candidate does not reproduce the fit hash")
    return directory


def _advice(out, fit, transfer, selection, winner, errors):
    if errors:
        return ForkDecision("hold", "incomplete_independent_verifier").record()
    qualification = fit.get("qualification", {})
    persistent = len(qualification.get("repeatable_errors", ()))
    if fit["candidates"]:
        return jspace_recommendation(JSpaceEvidence(True, max(1, persistent), practice_pass=True,
            preservation_validation_required=True, preservation_validation_pass=bool(selection),
            selection_pass=bool(winner) if selection else None)).record()
    reason = fit.get("detail", {}).get("reason")
    if reason == "parent_measurement_did_not_qualify":
        gates = qualification.get("gates", {})
        enough_errors = gates.get("repeatable_errors")
        diagnostic = ("insufficient_persistent_errors" if enough_errors is False and all(
            passed for gate, passed in gates.items() if gate != "repeatable_errors") else "invalid_measurement")
        return ForkDecision("hold" if diagnostic == "insufficient_persistent_errors" else "measure", diagnostic).record()
    if reason == "insufficient_distinct_replay_matched_error_groups":
        return ForkDecision("measure", reason).record()
    if reason in {"incomplete_preservation_coverage", "incomplete_calibration_coverage"}:
        return ForkDecision("broaden_preservation", reason).record()
    if reason in {"no_calibrated_positive_slack", "preservation_secant_unavailable",
                  "incomplete_preservation_vjps", "incomplete_calibration_vjps"}:
        return ForkDecision("measure", reason).record()
    if reason != "comparison_complete":
        return ForkDecision("hold", reason or "no_practice_candidate").record()
    arms = [_read(out / name) for name in fit["artifacts"] if re.fullmatch(r"arm-[AB][12]\.json", name)]
    guards = [arm["probe"]["endpoint"] for arm in arms if "endpoint" in arm["probe"]]
    if not guards:
        cause = "no_feasible_supported_direction" if all(
            arm["probe"]["reason"] == "no_positive_feasible_direction" for arm in arms) else "no_stored_precision_probe"
    elif any(row["guard"]["target_gain"] > 0 and (
            row["guard"]["regressions"] or row["guard"]["lost_valid"]
            or row["preservation_fit_guard"]["prohibited_regressions"]
            or row["preservation_fit_guard"]["lost_valid_presentations"]) for row in guards):
        cause = "practice_correction_with_preservation_regressions"
    elif any(row["guard"]["semantic_behavior_passed"] and not row["guard"]["strict_measured_floor_passed"]
             for row in guards):
        cause = "measured_preservation_slack_failed"
    else:
        cause = "no_strict_semantic_practice_gain"
    return {**ForkDecision("hold", cause).record(), "complete_practice_probes": len(guards)}


def assess(base, material, out):
    """Fresh-load both parents and every eligible distinct checkpoint in gate order."""
    started, cpu = time.perf_counter(), time.process_time()
    manifest, config = read_manifest(material), _read(PROTOCOL)
    fit = validate_fit_report(out, material)
    fit_sha = file_hash(out / "fit-report.json")
    base_sha, candidates = config["checkpoint_sha256"], fit["candidates"]
    sources = source_hashes(verifier=True)
    if not _private(out) or load_checkpoint(base).checkpoint_sha256() != base_sha:
        raise ValueError("Assessment requires a private workspace and the frozen parent checkpoint")
    lock_path, result = out / "candidate-lock.json", out / "assessment.json"
    if lock_path.exists() or result.exists():
        raise FileExistsError("Assessment artifacts already exist")
    (out / "verification").mkdir()
    reports, resources, errors = {}, {}, {}

    def measure(label, path, sha, split):
        destination = out / "verification" / f"{split}-{label}.json"
        try:
            report = _launch(path, sha, material, split, destination, manifest, config)
        except _FAILURES as error:
            errors[destination.name] = {"reason": "incomplete_independent_verifier", "error_type": type(error).__name__}
            return None
        reports[destination.name] = file_hash(destination)
        resources[destination.name] = {**report["resources"], "endpoint": report["measurement"]["resources"]}
        return report["measurement"]

    parents = {split: measure("base", base, base_sha, split) for split in PUBLIC_SPLITS}
    transfer, selection = {}, {}
    for arm, info in sorted(candidates.items()):
        checkpoint = _checkpoint(out, arm, info, base_sha)
        if parents[TRANSFER] is None:
            transfer[arm] = {"passed": False, "reason": "missing_transfer_parent"}
            continue
        candidate = measure(arm, checkpoint, info["checkpoint_sha256"], TRANSFER)
        transfer[arm] = (transfer_gate(parents[TRANSFER], candidate, config) if candidate is not None else
                         {"passed": False, **errors[f"{TRANSFER}-{arm}.json"]})
        if not transfer[arm]["passed"]:
            continue
        if parents["selection"] is None:
            selection[arm] = {"passed": False, "reason": "missing_selection_parent"}
            continue
        candidate = measure(arm, checkpoint, info["checkpoint_sha256"], "selection")
        selection[arm] = (selection_gate(parents["selection"], candidate, config) if candidate is not None else
                          {"passed": False, **errors[f"selection-{arm}.json"]})
    winner = _rank(selection, candidates)
    if source_hashes(verifier=True) != sources or file_hash(out / "fit-report.json") != fit_sha:
        raise ValueError("Sources or fit evidence changed during verification")
    lock = {"schema": LOCK_SCHEMA, "base_sha256": base_sha, "sources": sources, "fit_report_sha256": fit_sha,
            "manifest_sha256": file_hash(material / "manifest.json"), "winner": winner,
            "winner_sha256": candidates[winner]["checkpoint_sha256"] if winner else None, "candidates": candidates,
            "transfer": transfer, "selection": selection, "reports": reports, "errors": errors}
    replay_lock(out, material, lock)
    json_dump_atomic(lock_path, lock)
    reason = ("hidden_verification_required" if winner else "incomplete_independent_verifier" if errors else
              "no_practice_candidate" if not candidates else "transfer_preservation_failure" if not selection else
              "selection_failure")
    decision = _advice(out, fit, transfer, selection, winner, errors)
    summary = {"schema": "weightbridge.phase_iiil.assessment.v1", "disposition": "HOLD" if winner or not candidates
               or errors else "FAIL", "reason": reason, "winner": winner, "hidden_opened": False,
               "transfer": transfer, "selection": selection, "candidate_lock_sha256": file_hash(lock_path),
               "worker_resources": resources, "errors": errors, "meta": decision,
               "parent": {split: summarize(measurement) for split, measurement in parents.items() if measurement is not None},
               "host_resources": _resources(started, cpu)}
    json_dump_atomic(result, summary)
    return summary


def replay_lock(out, material, lock):
    """Check all public evidence and independently recompute the finalist before hidden access."""
    manifest, config = read_manifest(material), _read(PROTOCOL)
    fit = validate_fit_report(out, material)
    base_sha, candidates = config["checkpoint_sha256"], fit["candidates"]
    if (lock.get("schema") != LOCK_SCHEMA or lock["sources"] != source_hashes(verifier=True)
            or lock["base_sha256"] != base_sha or lock["manifest_sha256"] != file_hash(material / "manifest.json")
            or lock["fit_report_sha256"] != file_hash(out / "fit-report.json") or lock["candidates"] != candidates):
        raise ValueError("Candidate lock differs from the frozen fit report")
    reports, failures = lock["reports"], lock["errors"]
    consumed = set()

    def retrieve(split, label, sha):
        name = f"{split}-{label}.json"
        consumed.add(name)
        if name in reports and name not in failures:
            path = out / "verification" / name
            if file_hash(path) != reports[name]:
                raise ValueError("Verifier receipt digest differs from the candidate lock")
            return _validate_report(path, sha, material, manifest, config, split)["measurement"]
        if name not in failures or name in reports or failures[name].get("reason") != "incomplete_independent_verifier":
            raise ValueError("Missing or contradictory verifier receipt")
        return None

    parents = {split: retrieve(split, "base", base_sha) for split in PUBLIC_SPLITS}
    transfer, selection = {}, {}
    for arm, info in sorted(candidates.items()):
        if parents[TRANSFER] is None:
            transfer[arm] = {"passed": False, "reason": "missing_transfer_parent"}
            continue
        candidate = retrieve(TRANSFER, arm, info["checkpoint_sha256"])
        transfer[arm] = (transfer_gate(parents[TRANSFER], candidate, config) if candidate is not None else
                         {"passed": False, **failures[f"{TRANSFER}-{arm}.json"]})
        if not transfer[arm]["passed"]:
            continue
        if parents["selection"] is None:
            selection[arm] = {"passed": False, "reason": "missing_selection_parent"}
            continue
        candidate = retrieve("selection", arm, info["checkpoint_sha256"])
        selection[arm] = (selection_gate(parents["selection"], candidate, config) if candidate is not None else
                          {"passed": False, **failures[f"selection-{arm}.json"]})
    winner = _rank(selection, candidates)
    if (set(reports) | set(failures) != consumed or transfer != lock["transfer"] or selection != lock["selection"]
            or winner != lock["winner"]
            or (candidates[winner]["checkpoint_sha256"] if winner else None) != lock["winner_sha256"]):
        raise ValueError("Transfer, independent selection, or finalist ranking could not be replayed")
    return {"winner": winner, "transfer": transfer, "selection": selection, "parents": parents}


def finish(base, material, out, hidden, lock_sha):
    """Open the host roster only after replaying frozen selection and finalist ranking."""
    config, manifest = _read(PROTOCOL), read_manifest(material)
    result, lock_path = out / "independent-outcome.json", out / "candidate-lock.json"
    if result.exists() or not (_private(out) and _private(hidden)):
        raise ValueError("Independent closure requires fresh private artifacts")
    lock = _check_lock(lock_path, lock_sha, material, config)
    replay_lock(out, material, lock)
    selected = _checkpoint(out, lock["winner"], lock["candidates"][lock["winner"]], lock["base_sha256"])
    if load_checkpoint(base).checkpoint_sha256() != lock["base_sha256"]:
        raise ValueError("Hidden closure parent is not the frozen original")
    records = {}

    def measure(label, path, sha):
        output = out / "verification" / f"hidden-{label}.json"
        report = _launch(path, sha, material, "hidden", output, manifest, config,
                         lock=lock_path, lock_sha=lock_sha, hidden=hidden)
        records[label] = output, report
        return report["measurement"]

    summary = {"schema": "weightbridge.phase_iiil.independent_outcome.v1", "hidden_verification_started": True,
               "disposition": "FAIL", "reason": "incomplete_independent_verifier", "candidate_lock_sha256": lock_sha}
    try:
        before = measure("base", base, lock["base_sha256"])
        after = measure("candidate", selected, lock["winner_sha256"])
        gate = selection_gate(before, after, config, "hidden")
        summary.update(reason="failed_independent_endpoint", hidden_gate_passed=gate["passed"],
                       target_item_gain=gate["target_gain"], regressions=len(gate["regressions"]),
                       lost_valid=len(gate["lost_valid"]), family_gain_interval=gate["family_gain_interval"])
        if gate["passed"]:
            replay = measure("replay", selected, lock["winner_sha256"])
            rollback = measure("rollback", base, lock["base_sha256"])
            summary.update(restart_exact=after["rows"] == replay["rows"], rollback_exact=before["rows"] == rollback["rows"],
                           reason="independent_reload_or_rollback_changed")
            if summary["restart_exact"] and summary["rollback_exact"]:
                _check_lock(lock_path, lock_sha, material, config)
                lineage = out / "checkpoints" / "lineage"
                init_lineage(lineage, base)
                state = record_endpoint_lineage(lineage, base=base, candidate=selected, contract=endpoint(config),
                    hidden_roster=hidden, manifest=material / "manifest.json", selection_lock=lock_path,
                    cycle_id="phase-iii-l-semantic-consensus-v1", min_target_gain=config["hidden_gate"]["min_target_gain"],
                    **{f"{name}_report": value[0] for name, value in records.items()})
                summary.update(disposition="PASS", reason="independent_endpoint_and_replay_pass",
                               accepted_candidate_sha256=state["head_sha256"], lineage_event_sha256=state["last_event_sha256"])
    except _FAILURES as error:
        summary.update(reason="incomplete_independent_verifier", error_type=type(error).__name__)
    summary["hidden_opened"] = True if records else None
    summary["meta"] = jspace_recommendation(JSpaceEvidence(True, 1, practice_pass=True,
        preservation_validation_required=True, preservation_validation_pass=True, selection_pass=True),
        independent=summary).record()
    summary["workers"] = {name: {"sha256": file_hash(path), "resources": report["resources"]}
                          for name, (path, report) in records.items()}
    json_dump_atomic(result, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "assess", "finish"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--split", choices=(*PUBLIC_SPLITS, "hidden"))
    parser.add_argument("--sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--lock-sha256")
    parser.add_argument("--hidden", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if not (args.split and args.sha256 and args.manifest_sha256):
            parser.error("worker requires split, checkpoint SHA-256 and manifest SHA-256")
        worker(args.base, args.sha256, args.material, args.split, args.out, args.manifest_sha256,
               lock=args.lock, lock_sha=args.lock_sha256, hidden=args.hidden)
    elif args.command == "assess":
        if any((args.hidden, args.lock, args.lock_sha256, args.split, args.sha256, args.manifest_sha256)):
            parser.error("assess requires only base, material and output")
        result = assess(args.base, args.material, args.out)
        print(json.dumps({key: result[key] for key in ("disposition", "reason", "winner", "candidate_lock_sha256")}))
    else:
        if not (args.hidden and args.lock_sha256):
            parser.error("finish requires the hidden roster path and finalist lock SHA-256")
        print(json.dumps(finish(args.base, args.material, args.out, args.hidden, args.lock_sha256), indent=2))
