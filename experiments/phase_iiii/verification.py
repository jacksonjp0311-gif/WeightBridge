"""Host-only independent endpoint evaluation for the causal fork trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import endpoint, _paired
from experiments.phase_iiii.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.endpoint_lineage import record_endpoint_lineage
from weightbridge.adaptation.lineage import init_lineage
from weightbridge.adaptation.behavior_endpoint import measure_endpoint
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


ROOT = Path(__file__).resolve().parents[2]
LOCK_SCHEMA = "weightbridge.phase_iiih.candidate_lock.v1"
WORKER_SCHEMA = "weightbridge.phase_iiih.verifier.v1"
_SOURCES = (
    "experiments/phase_iiif/run_matrix.py", "experiments/phase_iiih/run_decimal_endpoint.py",
    "experiments/phase_iiig/roster_decimal_targeted.py", "weightbridge/core.py",
    "weightbridge/inference/huggingface.py", "weightbridge/inspection/safetensors.py",
    "weightbridge/patches/huggingface.py",
    *(f"experiments/phase_iiii/{name}" for name in
      ("baseline.py", "decimal_roster.py", "decimal_protocol.json", "run_causal_fork.py", "verification.py")),
    *(f"weightbridge/adaptation/{name}.py" for name in
      ("behavior_endpoint", "content_scoring", "endpoint_update", "endpoint_lineage", "lineage", "fork_policy",
       "fork_update", "generative_fork", "meta_controller", "model_adapter", "universal_awl", "universal_update")),
)


def _read(path):
    return json.loads(Path(path).read_bytes())


def source_hashes():
    return {name: file_hash(ROOT / name) for name in sorted(_SOURCES)}


def _private(path):
    return not Path(path).resolve().is_relative_to(ROOT)


def _check_lock(path, digest, material, config, expected=None):
    if not _private(path) or file_hash(path) != digest:
        raise ValueError("Hidden verification requires a private hash-bound finalist lock")
    lock = _read(path)
    winner = lock.get("winner")
    if (lock.get("schema") != LOCK_SCHEMA or not winner
            or lock["base_sha256"] != config["checkpoint_sha256"]
            or lock["manifest_sha256"] != file_hash(material / "manifest.json")
            or lock["sources"] != source_hashes()
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
        if not (lock and lock_sha and hidden and _private(hidden) and _private(out)):
            raise ValueError("Hidden data requires a private workspace and a finalist lock")
        _check_lock(lock, lock_sha, material, config, sha)
    elif split != "selection" or any(value is not None for value in (lock, lock_sha, hidden)):
        raise ValueError("Selection verification cannot receive hidden data")
    rows = load_split(material, split, manifest, hidden_path=hidden)
    with _open(checkpoint, config, sha) as (_, runner, capabilities):
        measured = measure_endpoint(runner, rows, endpoint(config))
    json_dump_atomic(out, {"schema": WORKER_SCHEMA, "split": split, "checkpoint_sha256": sha,
                          "roster_sha256": manifest["splits"][split]["sha256"], "manifest_sha256": manifest_sha,
                          "measurement": measured, "architecture": capabilities.architecture,
                          "resources": {"wall_seconds": time.perf_counter() - started,
                                        "cpu_seconds": time.process_time() - cpu,
                                        "peak_rss_bytes": peak_rss_bytes()}})


def _validate_report(path, sha, material, manifest, config, split, *, hidden=None):
    report = _read(path)
    expected = load_split(material, split, manifest, hidden_path=hidden)
    measured = report["measurement"]
    if (report.get("schema") != WORKER_SCHEMA or report.get("split") != split
            or report.get("checkpoint_sha256") != sha
            or report.get("manifest_sha256") != file_hash(material / "manifest.json")
            or report.get("roster_sha256") != manifest["splits"][split]["sha256"]
            or measured.get("endpoint") != endpoint(config).manifest()
            or len(expected) != len(measured["rows"])
            or any(any(row.get(key) != value for key, value in frozen.items())
                   for row, frozen in zip(measured["rows"], expected, strict=True))):
        raise ValueError("Verifier output differs from the frozen checkpoint, endpoint, or roster")
    return report


def _launch(path, sha, material, split, out, manifest, config, *, lock=None, lock_sha=None, hidden=None):
    args = [sys.executable, "-m", "experiments.phase_iiii.verification", "worker", "--base", str(path),
            "--sha256", sha, "--material", str(material), "--split", split, "--out", str(out),
            "--manifest-sha256", file_hash(material / "manifest.json")]
    if lock is not None:
        args += ["--lock", str(lock), "--lock-sha256", lock_sha, "--hidden", str(hidden)]
    subprocess.run(args, cwd=ROOT, check=True, timeout=config["inference"]["worker_timeout_seconds"])
    return _validate_report(out, sha, material, manifest, config, split, hidden=hidden)


def assess(base, material, out, candidates, parent_selection=None):
    """Score only saved practice winners and freeze one selection winner."""
    config, manifest = _read(PROTOCOL), read_manifest(material)
    source = source_hashes()
    base_sha = config["checkpoint_sha256"]
    if load_checkpoint(base).checkpoint_sha256() != base_sha:
        raise ValueError("Selection parent differs from the frozen checkpoint")
    if candidates and not _private(out):
        raise ValueError("Materialized adaptation requires a private workspace")
    lock_path = out / "candidate-lock.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    verification = out / "verification"
    verification.mkdir(exist_ok=True)
    selection, resources, receipts, reports = {}, {}, {}, {}
    parent = None
    if parent_selection is not None:
        parent = _validate_report(parent_selection, base_sha, material, manifest, config, "selection")
        json_dump_atomic(verification / "selection-base.json", parent)
    elif candidates:
        parent = _launch(base, base_sha, material, "selection", verification / "selection-base.json", manifest, config)
    if parent is not None:
        resources["base"] = parent["resources"]
        reports["base"] = file_hash(verification / "selection-base.json")
    for name, candidate in candidates.items():
        if not isinstance(name, str) or not name.isalnum():
            raise ValueError("Invalid candidate identifier")
        checkpoint = Path(candidate.get("checkpoint", out / "checkpoints" / name)).resolve(strict=True)
        if checkpoint != (out / "checkpoints" / name).resolve():
            raise ValueError("Candidate path is outside its private experiment slot")
        sha = candidate["checkpoint_sha256"]
        receipt_path = checkpoint / ".weightbridge_receipt.json"
        receipt = _read(receipt_path)
        if receipt.get("base_sha256") != base_sha or receipt.get("candidate_sha256") != sha:
            raise ValueError("Candidate receipt does not bind the source and materialized weights")
        receipts[name] = {"checkpoint_sha256": sha, "realized_update_norm": candidate["realized_update_norm"],
                          "receipt_sha256": file_hash(receipt_path)}
        result_path = verification / f"selection-{name}.json"
        try:
            result = _launch(checkpoint, sha, material, "selection", result_path, manifest, config)
            selection[name] = _paired(parent["measurement"], result["measurement"], config, config["selection_gate"])
            resources[name] = result["resources"]
            reports[name] = file_hash(result_path)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            selection[name] = {"passed": False, "reason": "incomplete_independent_verifier", "error_type": type(error).__name__}
    eligible = [name for name, gate in selection.items() if gate["passed"]]
    winner = min(eligible, key=lambda name: (-selection[name]["target_item_gain"],
                                             receipts[name]["realized_update_norm"], name), default=None)
    if source != source_hashes():
        raise ValueError("Experiment sources changed during verification")
    json_dump_atomic(lock_path, {"schema": LOCK_SCHEMA, "base_sha256": base_sha,
                                "manifest_sha256": file_hash(material / "manifest.json"), "sources": source,
                                "winner": winner, "winner_sha256": receipts[winner]["checkpoint_sha256"] if winner else None,
                                "selection": selection, "candidates": receipts, "reports": reports})
    return {"winner": winner, "selection": selection, "verifier_resources": resources,
            "candidate_lock_sha256": file_hash(lock_path)}


def finish(base, material, out, hidden, lock_sha):
    """Open sealed cases after a fresh selection replay and publish an aggregate receipt."""
    config, manifest = _read(PROTOCOL), read_manifest(material)
    result_path = out / "independent-outcome.json"
    if result_path.exists() or not (_private(out) and _private(hidden)):
        raise ValueError("Independent closure requires fresh private artifacts")
    lock_path = out / "candidate-lock.json"
    lock = _check_lock(lock_path, lock_sha, material, config)
    candidate = out / "checkpoints" / lock["winner"]
    receipt = candidate / ".weightbridge_receipt.json"
    if file_hash(receipt) != lock["candidates"][lock["winner"]]["receipt_sha256"]:
        raise ValueError("Locked candidate receipt differs")
    verification = out / "verification"
    records = {}

    def prior(name, sha):
        path = verification / f"selection-{name}.json"
        if file_hash(path) != lock["reports"][name]:
            raise ValueError("Locked selection receipt differs")
        return _validate_report(path, sha, material, manifest, config, "selection")["measurement"]

    expected = _paired(prior("base", lock["base_sha256"]), prior(lock["winner"], lock["winner_sha256"]),
                       config, config["selection_gate"])
    if expected != lock["selection"][lock["winner"]] or not expected["passed"]:
        raise ValueError("Selection gate could not be independently recomputed")

    def measure(label, path, sha):
        output = verification / f"hidden-{label}.json"
        records[label] = (output, _launch(path, sha, material, "hidden", output, manifest, config,
                                           lock=lock_path, lock_sha=lock_sha, hidden=hidden))
        return records[label][1]["measurement"]

    summary = {"schema": "weightbridge.phase_iiii.independent_outcome.v1", "hidden_verification_started": True,
               "disposition": "FAIL", "reason": "incomplete_independent_verifier", "candidate_lock_sha256": lock_sha}
    try:
        before = measure("base", base, lock["base_sha256"])
        after = measure("candidate", candidate, lock["winner_sha256"])
        gate = _paired(before, after, config, config["hidden_gate"])
        summary.update(reason="failed_independent_endpoint", hidden_gate_passed=gate["passed"],
                       target_item_gain=gate["target_item_gain"], regressions=len(gate["prohibited_regressions"]),
                       valid_rate=gate["candidate_valid_rate"])
        if gate["passed"]:
            summary["reason"] = "incomplete_independent_verifier"
            replay = measure("replay", candidate, lock["winner_sha256"])
            rollback = measure("rollback", base, lock["base_sha256"])
            summary.update(restart_exact=after["rows"] == replay["rows"], rollback_exact=before["rows"] == rollback["rows"],
                           reason="independent_reload_or_rollback_changed")
            if summary["restart_exact"] and summary["rollback_exact"]:
                _check_lock(lock_path, lock_sha, material, config)
                lineage = out / "checkpoints" / "lineage"
                init_lineage(lineage, base)
                state = record_endpoint_lineage(
                    lineage, base=base, candidate=candidate, contract=endpoint(config), hidden_roster=hidden,
                    manifest=material / "manifest.json", selection_lock=lock_path, cycle_id="phase-iii-i-causal-fork-v1",
                    min_target_gain=config["hidden_gate"]["min_target_gain"],
                    **{f"{name}_report": value[0] for name, value in records.items()},
                )
                summary.update(disposition="PASS", reason="independent_endpoint_and_replay_pass",
                               accepted_candidate_sha256=state["head_sha256"], lineage_event_sha256=state["last_event_sha256"])
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        summary.update(reason="incomplete_independent_verifier", error_type=type(error).__name__)
    summary["hidden_opened"] = True if records else None
    from weightbridge.adaptation.fork_policy import decide_fork_action

    summary["meta"] = decide_fork_action(measurement_valid=True, persistent_errors=1, practice_pass=True,
                                         selection_pass=True, independent=summary).record()
    summary["workers"] = {name: {"sha256": file_hash(path), "resources": result["resources"]}
                          for name, (path, result) in records.items()}
    json_dump_atomic(result_path, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "finish"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--split", choices=("selection", "hidden"))
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
    else:
        if not (args.hidden and args.lock_sha256):
            parser.error("finish requires the hidden roster path and finalist lock SHA-256")
        print(json.dumps(finish(args.base, args.material, args.out, args.hidden, args.lock_sha256), indent=2))
