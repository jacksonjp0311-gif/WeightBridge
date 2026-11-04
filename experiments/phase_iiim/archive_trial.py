"""Publish digest-checked public evidence from the factorial host trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments.phase_iiil.archive_trial import _private_fields, _public_bytes


FIT_FILES = re.compile(
    r"(?:parent-(?:adaptation|preservation-fit|calibration)|fork-diagnosis|basis|directions|"
    r"preservation-decisions|secants|sampling|preservation-sensitivities|calibration-(?:decisions|sensitivities)|"
    r"floor|probe-reference|token-class-parent|stored-pairwise|arm-[ABCD][12])\.json"
    r"|patches/[ABCD][12]-(?:patch|policy|receipt)\.json"
)
WORKER_FILES = re.compile(r"verification/(?:development-transfer|preservation-validation|selection)-(?:base|[ABCD][12])\.json")
STAGE_LOCKS = ("development-lock.json", "preservation-lock.json", "candidate-lock.json")


def public_records(source, material):
    from experiments.phase_iiil.semantic_diagnostics import summarize
    from experiments.phase_iiim.semantic_roster import PROTOCOL
    from experiments.phase_iiim.verification import replay_lock, validate_fit_report

    root = source.resolve(strict=True)
    fit = validate_fit_report(root, material)
    records = {name: _public_bytes(root, name) for name in ("fit-report.json", "assessment.json")}
    assessment = json.loads(records["assessment.json"])
    active_name = assessment.get("active_lock")
    if active_name not in STAGE_LOCKS:
        raise ValueError("Assessment references an unknown public verification stage")
    for name in STAGE_LOCKS[:STAGE_LOCKS.index(active_name) + 1]:
        records[name] = _public_bytes(root, name)
        replay_lock(root, material, json.loads(records[name]))
    active = json.loads(records[active_name])
    replay = replay_lock(root, material, active)
    expected_winner = replay["winner"] if active["stage"] == "selection" and replay["eligible"] else None
    expected_parents = {name: summarize(measured) for name, measured in replay["parents"].items() if measured is not None}
    if (assessment.get("schema") != "weightbridge.phase_iiim.assessment.v1"
            or assessment["active_lock_sha256"] != hashlib.sha256(records[active_name]).hexdigest()
            or assessment["hidden_opened"] is not False or assessment["winner"] != expected_winner
            or assessment["development_winner"] != replay["winner"] or assessment["parent"] != expected_parents
            or assessment["errors"] != active["errors"]
            or any(assessment[key] != replay[key] for key in ("development", "preservation", "selection"))):
        raise ValueError("Assessment does not independently replay from the frozen public stages")
    expected_reason = (
        "incomplete_independent_verifier" if active["errors"] else
        "hidden_verification_required" if expected_winner else
        "no_development_transfer_candidate" if not replay["winner"] else
        "preservation_validation_failure" if not replay["selection"] else "selection_failure")
    expected_disposition = "HOLD" if not replay["winner"] or expected_winner or active["errors"] else "FAIL"
    if assessment["reason"] != expected_reason or assessment["disposition"] != expected_disposition:
        raise ValueError("Assessment disposition differs from its independently replayed gate")
    for name, digest in fit["artifacts"].items():
        records[name] = _public_bytes(root, name, FIT_FILES, digest)
    work = {}
    for name, digest in active["reports"].items():
        filename = f"verification/{name}"
        records[filename] = _public_bytes(root, filename, WORKER_FILES, digest)
        receipt = json.loads(records[filename])
        work[name] = {**receipt["resources"], "endpoint": receipt["measurement"]["resources"]}
    if work != assessment["worker_resources"]:
        raise ValueError("Reported verifier computation differs from the source-bound workers")
    independent_path = root / "independent-outcome.json"
    if independent_path.exists():
        if not expected_winner:
            raise ValueError("A sealed outcome requires a selection-qualified locked finalist")
        raw = _public_bytes(root, "independent-outcome.json")
        outcome = json.loads(raw)
        if (outcome.get("schema") != "weightbridge.phase_iiim.independent_outcome.v1"
                or outcome["candidate_lock_sha256"] != assessment["active_lock_sha256"]
                or _private_fields(outcome) or outcome.get("lineage_accepted") is not False
                or outcome.get("operator_review_required") is not True):
            raise ValueError("Sealed evidence must be aggregate, operator-controlled, and bound to the finalist")
        if outcome["disposition"] == "PASS":
            config = json.loads(PROTOCOL.read_bytes())
            if (outcome.get("hidden_gate_passed") is not True or outcome.get("restart_exact") is not True
                    or outcome.get("rollback_exact") is not True or outcome.get("regressions") != 0
                    or outcome.get("lost_valid") != 0
                    or outcome.get("target_item_gain", -1) < config["hidden_gate"]["min_target_gain"]):
                raise ValueError("A passing sealed aggregate lacks its frozen gate and exact replay")
        records["independent-outcome.json"] = raw
    return records


def archive_trial(source, material, out):
    records = public_records(source, material)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in records.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    print(f"Archived {archive_trial(args.source, args.material, args.out)} public evidence files.", flush=True)
