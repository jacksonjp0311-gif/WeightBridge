"""Host-gated development transfer, preservation, selection and sealed verification."""
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
from experiments.phase_iiil.verification import _checkpoint, _digest, _private, _read
from experiments.phase_iiil.semantic_diagnostics import summarize
from experiments.phase_iiim.semantic_metrics import compare
from experiments.phase_iiim.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiim.source_manifest import ROOT, source_hashes
from weightbridge.adaptation.behavior_endpoint import _observation, measure_endpoint
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


WORKER_SCHEMA = "weightbridge.phase_iiih.verifier.v1"
LOCK_SCHEMA = "weightbridge.phase_iiim.stage_lock.v1"
DEVELOPMENT = "development-transfer"
PRESERVATION = "preservation-validation"
PUBLIC_SPLITS = (DEVELOPMENT, PRESERVATION, "selection")
_STAGES = ("development", "preservation", "selection")
_LOCK_NAMES = dict(zip(_STAGES, ("development-lock.json", "preservation-lock.json", "candidate-lock.json")))
_FAILURES = (subprocess.SubprocessError, OSError, ValueError, KeyError, RuntimeError)


def validate_fit_report(out, material):
    report, config = _read(out / "fit-report.json"), _read(PROTOCOL)
    read_manifest(material)
    if (report.get("schema") != "weightbridge.phase_iiim.fit.v1" or report["sources"] != source_hashes()
            or report["manifest_sha256"] != file_hash(material / "manifest.json")
            or report["checkpoint_sha256"] != config["checkpoint_sha256"]
            or report["optimization_splits"] != ["adaptation", "preservation-fit", "calibration"]
            or report["hidden_opened"] is not False):
        raise ValueError("Fit evidence does not bind source, development scope, model and roster")
    for name, digest in report["artifacts"].items():
        path = (out / name).resolve()
        if (not name or not path.is_relative_to(out.resolve())
                or Path(name).parts[0] in {"checkpoints", "verification"}
                or not _digest(digest) or file_hash(path) != digest):
            raise ValueError("Fitting artifact does not match its frozen hash")
    states = set()
    candidates = report["candidates"]
    if len(candidates) > config["intent"]["limits"]["max_candidates"]:
        raise ValueError("Candidate count exceeds the predeclared budget")
    for arm, info in candidates.items():
        if (re.fullmatch(r"[ABCD][12]", arm) is None
                or not all(_digest(info[key]) for key in ("checkpoint_sha256", "tensor_sha256", "receipt_sha256"))
                or info["checkpoint_sha256"] == config["checkpoint_sha256"]
                or not isfinite(info["realized_update_norm"])
                or not 0 < info["realized_update_norm"] <= config["intent"]["limits"]["max_update_norm"]
                or type(info["changed_entries"]) is not int
                or not 0 < info["changed_entries"] <= config["intent"]["limits"]["max_update_entries"]
                or info["tensor_sha256"] in states
                or f"arm-{arm}.json" not in report["artifacts"]
                or f"patches/{arm}-receipt.json" not in report["artifacts"]):
            raise ValueError("Practice candidate identity or numerical authorization differs")
        record = _read(out / f"arm-{arm}.json")
        receipt = _read(out / "patches" / f"{arm}-receipt.json")
        native = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        probe = record["probe"]
        if (hashlib.sha256(native).hexdigest() != info["receipt_sha256"]
                or record["norm_match"]["status"] != "MATCHED"
                or not probe.get("restored_exactly") or probe.get("candidate_tensor_sha256") != info["tensor_sha256"]
                or probe.get("endpoint", {}).get("guard", {}).get("passed") is not True):
            raise ValueError("Candidate lacks a matched, restored, practice-qualified stored checkpoint")
        states.add(info["tensor_sha256"])
    return report


def development_gate(before, after, config):
    gate = compare(before, after, min_comparison_gain=config["development_gate"]["min_target_gain"],
                   min_family_gain=config["development_gate"]["min_complete_family_gain"])
    return _valid_rate(gate, after, config)


def _valid_rate(gate, after, config):
    rate = sum(row["valid"] for row in after["rows"]) / len(after["rows"])
    return {**gate, "candidate_valid_rate": rate,
            "passed": bool(gate["passed"] and rate >= config["endpoint"]["min_valid_rate"])}


def preservation_gate(before, after, config):
    gate = compare(before, after)
    protected = sum(row["correct"] for row in before["rows"])
    return _valid_rate({**gate, "protected_presentations": protected,
            "passed": bool(protected >= config["preservation_gate"]["min_protected_presentations"]
                           and not gate["regressions"] and not gate["lost_valid"])}, after, config)


def independent_gate(before, after, config, split):
    if split not in ("selection", "hidden"):
        raise ValueError("Unknown independent gate")
    return _valid_rate(compare(before, after, min_comparison_gain=config[f"{split}_gate"]["min_target_gain"]), after, config)


def _rank(gates, candidates):
    return min((name for name, gate in gates.items() if gate["passed"]), default=None,
               key=lambda name: (-gates[name]["complete_family_gain"], -gates[name]["target_gain"],
                                 candidates[name]["realized_update_norm"], name))


def _validate_report(path, sha, material, manifest, config, split, *, hidden=None):
    record = _read(path)
    rows = load_split(material, split, manifest, hidden_path=hidden)
    measured, contract = record["measurement"], endpoint(config)
    if (record.get("schema") != WORKER_SCHEMA or record.get("split") != split
            or record.get("checkpoint_sha256") != sha
            or record.get("manifest_sha256") != file_hash(material / "manifest.json")
            or record.get("roster_sha256") != manifest["splits"][split]["sha256"]
            or measured.get("endpoint") != contract.manifest() or len(rows) != len(measured["rows"])):
        raise ValueError("Verifier receipt differs from its checkpoint, endpoint or committed partition")
    for frozen, observed in zip(rows, measured["rows"], strict=True):
        parsed = _observation(observed, contract.question(frozen), contract)
        if any(observed.get(key) != value for key, value in {**frozen, **parsed}.items()):
            raise ValueError("Verifier receipt altered frozen data or deterministic answer parsing")
    return record


def _check_lock(path, digest, material, config, stage, expected=None):
    if not _private(path) or file_hash(path) != digest:
        raise ValueError("Host verification requires the exact private stage lock")
    lock = _read(path)
    if (lock.get("schema") != LOCK_SCHEMA or lock.get("stage") != stage or not lock.get("eligible")
            or not lock.get("winner") or lock["sources"] != source_hashes(verifier=True)
            or lock["base_sha256"] != config["checkpoint_sha256"]
            or lock["manifest_sha256"] != file_hash(material / "manifest.json")
            or expected is not None and expected not in (lock["base_sha256"], lock["winner_sha256"])):
        raise ValueError("The required earlier verification stage did not authorize this worker")
    replay_lock(path.parent, material, lock)
    return lock


def worker(checkpoint, sha, material, split, out, manifest_sha, *, lock=None, lock_sha=None, hidden=None):
    wall, cpu = time.perf_counter(), time.process_time()
    manifest, config = read_manifest(material), _read(PROTOCOL)
    if out.exists() or file_hash(material / "manifest.json") != manifest_sha:
        raise ValueError("Verifier requires frozen input and a fresh output path")
    prior_stage = {PRESERVATION: "development", "selection": "preservation", "hidden": "selection"}
    if split in prior_stage:
        if not (lock and lock_sha and _private(out)
                and out.parent.resolve() == (lock.parent / "verification").resolve()):
            raise ValueError("This partition requires an earlier private verification lock")
        _check_lock(lock, lock_sha, material, config, prior_stage[split], sha)
    elif split != DEVELOPMENT or lock is not None or lock_sha is not None:
        raise ValueError("Unknown public verification or invalid stage authorization")
    if (split == "hidden") != (hidden is not None) or hidden is not None and not _private(hidden):
        raise ValueError("Only an authorized sealed worker can receive the host roster")
    rows = load_split(material, split, manifest, hidden_path=hidden)
    with _open(checkpoint, config, sha) as (_, runner, capabilities):
        measured = measure_endpoint(runner, rows, endpoint(config))
    json_dump_atomic(out, {"schema": WORKER_SCHEMA, "split": split, "checkpoint_sha256": sha,
        "roster_sha256": manifest["splits"][split]["sha256"], "manifest_sha256": manifest_sha,
        "architecture": capabilities.architecture, "measurement": measured, "resources": _resources(wall, cpu)})


def _launch(path, sha, material, split, out, manifest, config, *, lock=None, hidden=None):
    args = [sys.executable, "-m", "experiments.phase_iiim.verification", "worker", "--base", str(path.resolve()),
            "--sha256", sha, "--material", str(material.resolve()), "--split", split, "--out", str(out.resolve()),
            "--manifest-sha256", file_hash(material / "manifest.json")]
    if lock is not None:
        args += ["--lock", str(lock.resolve()), "--lock-sha256", file_hash(lock)]
    if hidden is not None:
        args += ["--hidden", str(hidden.resolve())]
    subprocess.run(args, cwd=ROOT, check=True, timeout=config["inference"]["worker_timeout_seconds"])
    return _validate_report(out, sha, material, manifest, config, split, hidden=hidden)


def replay_lock(out, material, lock):
    """Recompute each authorized public gate before a later partition can be opened."""
    config, manifest = _read(PROTOCOL), read_manifest(material)
    fit = validate_fit_report(out, material)
    stage = lock.get("stage")
    if (lock.get("schema") != LOCK_SCHEMA or stage not in _STAGES
            or lock["sources"] != source_hashes(verifier=True)
            or lock["fit_report_sha256"] != file_hash(out / "fit-report.json")
            or lock["manifest_sha256"] != file_hash(material / "manifest.json")
            or lock["base_sha256"] != config["checkpoint_sha256"] or lock["candidates"] != fit["candidates"]):
        raise ValueError("Stage lock differs from the source-bound fitting result")
    reports, errors, consumed = lock["reports"], lock["errors"], set()
    parents = {}

    def retrieve(split, label, sha):
        name = f"{split}-{label}.json"
        consumed.add(name)
        if name in reports and name not in errors:
            path = out / "verification" / name
            if file_hash(path) != reports[name]:
                raise ValueError("Worker receipt differs from its stage commitment")
            return _validate_report(path, sha, material, manifest, config, split)["measurement"]
        if name not in errors or name in reports or errors[name].get("reason") != "incomplete_independent_verifier":
            raise ValueError("A required independent worker receipt is missing")
        return None

    development, preservation, selection = {}, {}, {}
    candidates, base_sha = fit["candidates"], config["checkpoint_sha256"]
    if candidates:
        parents[DEVELOPMENT] = retrieve(DEVELOPMENT, "base", base_sha)
        for arm, info in sorted(candidates.items()):
            if parents[DEVELOPMENT] is None:
                development[arm] = {"passed": False, "reason": "missing_development_parent"}
            else:
                observed = retrieve(DEVELOPMENT, arm, info["checkpoint_sha256"])
                development[arm] = (development_gate(parents[DEVELOPMENT], observed, config) if observed is not None
                                    else {"passed": False, "reason": "incomplete_independent_verifier"})
    winner = _rank(development, candidates)
    eligible = winner is not None
    if stage in ("preservation", "selection"):
        if not winner:
            raise ValueError("No development-qualified candidate may advance")
        parents[PRESERVATION] = retrieve(PRESERVATION, "base", base_sha)
        observed = retrieve(PRESERVATION, winner, candidates[winner]["checkpoint_sha256"])
        preservation[winner] = (preservation_gate(parents[PRESERVATION], observed, config)
                               if parents[PRESERVATION] is not None and observed is not None else
                               {"passed": False, "reason": "incomplete_independent_verifier"})
        eligible = preservation[winner]["passed"]
    if stage == "selection":
        if not eligible:
            raise ValueError("Preservation failure cannot authorize independent selection")
        parents["selection"] = retrieve("selection", "base", base_sha)
        observed = retrieve("selection", winner, candidates[winner]["checkpoint_sha256"])
        selection[winner] = (independent_gate(parents["selection"], observed, config, "selection")
                             if parents["selection"] is not None and observed is not None else
                             {"passed": False, "reason": "incomplete_independent_verifier"})
        eligible = selection[winner]["passed"]
    if (set(reports) | set(errors) != consumed or development != lock["development"]
            or preservation != lock["preservation"] or selection != lock["selection"]
            or winner != lock["winner"] or eligible != lock["eligible"]
            or (candidates[winner]["checkpoint_sha256"] if winner else None) != lock["winner_sha256"]):
        raise ValueError("The stage gates and finalist ranking did not independently replay")
    return {"winner": winner, "eligible": eligible, "development": development,
            "preservation": preservation, "selection": selection, "parents": parents}


def assess(base, material, out):
    """Freeze a development finalist before unseen preservation and independent selection."""
    wall, cpu = time.perf_counter(), time.process_time()
    config, manifest, fit = _read(PROTOCOL), read_manifest(material), validate_fit_report(out, material)
    source, fit_sha = source_hashes(verifier=True), file_hash(out / "fit-report.json")
    if not _private(out) or load_checkpoint(base).checkpoint_sha256() != config["checkpoint_sha256"]:
        raise ValueError("Assessment requires a private host trial and the original checkpoint")
    if any((out / name).exists() for name in (*_LOCK_NAMES.values(), "assessment.json")):
        raise FileExistsError("Verification artifacts are write-once")
    (out / "verification").mkdir()
    reports, resources, errors = {}, {}, {}
    development, preservation, selection = {}, {}, {}
    parents = {}
    candidates, base_sha = fit["candidates"], config["checkpoint_sha256"]

    def measure(label, path, sha, split, prior_lock=None):
        destination = out / "verification" / f"{split}-{label}.json"
        try:
            receipt = _launch(path, sha, material, split, destination, manifest, config, lock=prior_lock)
        except _FAILURES as error:
            errors[destination.name] = {"reason": "incomplete_independent_verifier", "error_type": type(error).__name__}
            return None
        reports[destination.name] = file_hash(destination)
        resources[destination.name] = {**receipt["resources"], "endpoint": receipt["measurement"]["resources"]}
        return receipt["measurement"]

    def freeze(stage, eligible):
        if source_hashes(verifier=True) != source or file_hash(out / "fit-report.json") != fit_sha:
            raise ValueError("Source or fitting evidence changed during host verification")
        record = {"schema": LOCK_SCHEMA, "stage": stage, "sources": source, "fit_report_sha256": fit_sha,
                  "manifest_sha256": file_hash(material / "manifest.json"), "base_sha256": base_sha,
                  "candidates": candidates, "winner": winner,
                  "winner_sha256": candidates[winner]["checkpoint_sha256"] if winner else None,
                  "eligible": bool(eligible), "development": development, "preservation": preservation,
                  "selection": selection, "reports": reports, "errors": errors}
        replay_lock(out, material, record)
        destination = out / _LOCK_NAMES[stage]
        json_dump_atomic(destination, record)
        return destination

    if candidates:
        parents[DEVELOPMENT] = measure("base", base, base_sha, DEVELOPMENT)
        for arm, info in sorted(candidates.items()):
            if parents[DEVELOPMENT] is None:
                development[arm] = {"passed": False, "reason": "missing_development_parent"}
                continue
            checkpoint = _checkpoint(out, arm, info, base_sha)
            observed = measure(arm, checkpoint, info["checkpoint_sha256"], DEVELOPMENT)
            development[arm] = (development_gate(parents[DEVELOPMENT], observed, config) if observed is not None else
                                {"passed": False, "reason": "incomplete_independent_verifier"})
    winner = _rank(development, candidates)
    active_lock = freeze("development", winner is not None)
    if winner:
        info = candidates[winner]
        checkpoint = _checkpoint(out, winner, info, base_sha)
        parents[PRESERVATION] = measure("base", base, base_sha, PRESERVATION, active_lock)
        observed = measure(winner, checkpoint, info["checkpoint_sha256"], PRESERVATION, active_lock)
        preservation[winner] = (preservation_gate(parents[PRESERVATION], observed, config)
                                if parents[PRESERVATION] is not None and observed is not None else
                                {"passed": False, "reason": "incomplete_independent_verifier"})
        active_lock = freeze("preservation", preservation[winner]["passed"])
        if preservation[winner]["passed"]:
            parents["selection"] = measure("base", base, base_sha, "selection", active_lock)
            observed = measure(winner, checkpoint, info["checkpoint_sha256"], "selection", active_lock)
            selection[winner] = (independent_gate(parents["selection"], observed, config, "selection")
                                 if parents["selection"] is not None and observed is not None else
                                 {"passed": False, "reason": "incomplete_independent_verifier"})
            active_lock = freeze("selection", selection[winner]["passed"])
    eligible = bool(winner and selection.get(winner, {}).get("passed"))
    reason = ("incomplete_independent_verifier" if errors else "hidden_verification_required" if eligible else
              "no_development_transfer_candidate" if not winner else "preservation_validation_failure" if not selection else
              "selection_failure")
    result = {"schema": "weightbridge.phase_iiim.assessment.v1", "disposition": "HOLD" if not winner or eligible or errors else "FAIL",
              "reason": reason, "winner": winner if eligible else None, "development_winner": winner,
              "development": development, "preservation": preservation, "selection": selection,
              "hidden_opened": False, "active_lock": active_lock.name, "active_lock_sha256": file_hash(active_lock),
              "worker_resources": resources, "errors": errors,
              "parent": {name: summarize(measured) for name, measured in parents.items() if measured is not None},
              "meta": {"schema": "weightbridge.fork_decision.v1", "action": "host_hidden_verification" if eligible else "hold",
                       "reason": reason, "promotion_authority": False}, "host_resources": _resources(wall, cpu)}
    json_dump_atomic(out / "assessment.json", result)
    return result


def finish(base, material, out, hidden, lock_sha):
    """Score the sealed finalist and require review even after exact replay succeeds."""
    config, manifest = _read(PROTOCOL), read_manifest(material)
    result_path, lock_path = out / "independent-outcome.json", out / "candidate-lock.json"
    if result_path.exists() or not (_private(out) and _private(hidden)):
        raise ValueError("Sealed closure requires fresh private host artifacts")
    lock = _check_lock(lock_path, lock_sha, material, config, "selection")
    candidate = _checkpoint(out, lock["winner"], lock["candidates"][lock["winner"]], lock["base_sha256"])
    if load_checkpoint(base).checkpoint_sha256() != lock["base_sha256"]:
        raise ValueError("Parent checkpoint differs before sealed verification")
    records = {}

    def measure(label, path, sha):
        destination = out / "verification" / f"hidden-{label}.json"
        report = _launch(path, sha, material, "hidden", destination, manifest, config, lock=lock_path, hidden=hidden)
        records[label] = destination, report
        return report["measurement"]

    summary = {"schema": "weightbridge.phase_iiim.independent_outcome.v1", "hidden_verification_started": True,
               "disposition": "FAIL", "reason": "incomplete_independent_verifier", "candidate_lock_sha256": lock_sha,
               "lineage_accepted": False, "operator_review_required": True}
    try:
        before = measure("base", base, lock["base_sha256"])
        after = measure("candidate", candidate, lock["winner_sha256"])
        gate = independent_gate(before, after, config, "hidden")
        summary.update(hidden_gate_passed=gate["passed"], reason="failed_independent_endpoint",
                       target_item_gain=gate["target_gain"], complete_family_gain=gate["complete_family_gain"],
                       newly_completed_family_count=len(gate["newly_completed_families"]),
                       regressions=len(gate["regressions"]), lost_valid=len(gate["lost_valid"]),
                       family_gain_interval=gate["family_gain_interval"])
        if gate["passed"]:
            replay = measure("replay", candidate, lock["winner_sha256"])
            rollback = measure("rollback", base, lock["base_sha256"])
            summary.update(restart_exact=after["rows"] == replay["rows"], rollback_exact=before["rows"] == rollback["rows"],
                           reason="independent_reload_or_rollback_changed")
            if summary["restart_exact"] and summary["rollback_exact"]:
                _check_lock(lock_path, lock_sha, material, config, "selection")
                summary.update(disposition="PASS", reason="independent_endpoint_and_replay_pass_operator_review_required",
                               candidate_sha256=lock["winner_sha256"])
    except _FAILURES as error:
        summary.update(reason="incomplete_independent_verifier", error_type=type(error).__name__)
    summary["hidden_opened"] = True if records else None
    summary["meta"] = {"schema": "weightbridge.fork_decision.v1", "action": "operator_review" if summary["disposition"] == "PASS" else "hold",
                       "reason": summary["reason"], "promotion_authority": False}
    summary["workers"] = {name: {"sha256": file_hash(path), "resources": record["resources"]}
                          for name, (path, record) in records.items()}
    json_dump_atomic(result_path, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "assess", "finish"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--split", choices=(*PUBLIC_SPLITS, "hidden"))
    parser.add_argument("--sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--lock-sha256")
    parser.add_argument("--hidden", type=Path)
    args = parser.parse_args()
    if args.command == "worker":
        if not (args.split and args.sha256 and args.manifest_sha256):
            parser.error("worker requires its partition and immutable input hashes")
        worker(args.base, args.sha256, args.material, args.split, args.out, args.manifest_sha256,
               lock=args.lock, lock_sha=args.lock_sha256, hidden=args.hidden)
    elif args.command == "assess":
        if any((args.hidden, args.lock, args.lock_sha256, args.split, args.sha256, args.manifest_sha256)):
            parser.error("assess receives only the original checkpoint, material and private fitting trial")
        result = assess(args.base, args.material, args.out)
        print(json.dumps({key: result[key] for key in ("disposition", "reason", "winner", "active_lock", "active_lock_sha256")}))
    else:
        if not (args.hidden and args.lock_sha256):
            parser.error("finish requires a host-held roster and exact finalist-lock hash")
        print(json.dumps(finish(args.base, args.material, args.out, args.hidden, args.lock_sha256), indent=2))
