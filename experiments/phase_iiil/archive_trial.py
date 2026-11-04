"""Archive only hash-bound public evidence from a Phase III-L host trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


_FIT_FILES = re.compile(
    r"(?:parent-(?:adaptation|preservation-fit|calibration|diagnostics)|fork-diagnosis|basis|"
    r"teacher-forced-parent|projected-agreement|preservation-decisions|secants|sampling|preservation-sensitivities|"
    r"calibration(?:-decisions|-sensitivities)?|probe-reference|geometry-[12]|arm-[AB][12])\.json"
    r"|patches/[AB][12]-(?:patch|policy|receipt)\.json"
)
_WORKER_FILES = re.compile(r"verification/(?:transfer-validation|selection)-(?:base|[AB][12])\.json")
_PRIVATE_FIELDS = frozenset({"rows", "prompt", "operands", "text", "generated_token_ids", "hidden_path",
                             "private_roster", "token_ids", "checkpoint_path", "model_path"})


def _public_bytes(root, name, pattern=None, digest=None):
    if pattern is not None and not pattern.fullmatch(name):
        raise ValueError(f"Trial references a file outside its public evidence allowlist: {name}")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Public evidence is outside the trial or is not a file: {name}")
    raw = path.read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Public evidence differs from its recorded hash: {name}")
    return raw


def _private_fields(value):
    if isinstance(value, dict):
        return (_PRIVATE_FIELDS & value.keys()) | {field for child in value.values() for field in _private_fields(child)}
    if isinstance(value, list):
        return {field for child in value for field in _private_fields(child)}
    return set()


def fit_attempt_records(source, material):
    """Keep a completed fitting HOLD before an independent verifier has been launched."""
    from experiments.phase_iiil.verification import validate_fit_report

    root = source.resolve(strict=True)
    records = {"fit-report.json": _public_bytes(root, "fit-report.json")}
    report = json.loads(records["fit-report.json"])
    if report["candidates"] or report["hidden_opened"] is not False:
        raise ValueError("Fit-only archive requires a fitting report without an eligible checkpoint")
    for name, digest in report["artifacts"].items():
        records[name] = _public_bytes(root, name, _FIT_FILES, digest)
    validate_fit_report(root, material)
    return records


def public_records(source, material):
    from experiments.phase_iiil.semantic_diagnostics import summarize
    from experiments.phase_iiil.source_manifest import source_hashes
    from experiments.phase_iiil.verification import replay_lock

    root = source.resolve(strict=True)
    records = {name: _public_bytes(root, name) for name in ("fit-report.json", "candidate-lock.json", "assessment.json")}
    report, lock, assessment = (json.loads(records[name]) for name in records)
    fit_sha = hashlib.sha256(records["fit-report.json"]).hexdigest()
    lock_sha = hashlib.sha256(records["candidate-lock.json"]).hexdigest()
    if (report["sources"] != source_hashes() or lock["fit_report_sha256"] != fit_sha
            or assessment.get("schema") != "weightbridge.phase_iiil.assessment.v1"
            or assessment["candidate_lock_sha256"] != lock_sha
            or any(assessment[field] != lock[field] for field in ("winner", "transfer", "selection"))):
        raise ValueError("Public trial assessment does not bind the frozen fitting report and lock")
    for name, digest in report["artifacts"].items():
        records[name] = _public_bytes(root, name, _FIT_FILES, digest)
    worker_resources = {}
    for name, digest in lock["reports"].items():
        path = name if name.startswith("verification/") else f"verification/{name}"
        records[path] = _public_bytes(root, path, _WORKER_FILES, digest)
        receipt = json.loads(records[path])
        worker_resources[Path(name).name] = {**receipt["resources"], "endpoint": receipt["measurement"]["resources"]}
    replay = replay_lock(root, material, lock)
    parents = {split: summarize(measurement) for split, measurement in replay["parents"].items() if measurement is not None}
    if (assessment["parent"] != parents or assessment["errors"] != lock["errors"]
            or assessment["worker_resources"] != worker_resources or assessment["hidden_opened"] is not False):
        raise ValueError("Public parent assessment does not replay from its verifier receipts")
    if (root / "independent-outcome.json").exists():
        if not lock["winner"]:
            raise ValueError("Independent hidden evidence requires a selection-qualified winner")
        raw = _public_bytes(root, "independent-outcome.json")
        outcome = json.loads(raw)
        if (outcome.get("schema") != "weightbridge.phase_iiil.independent_outcome.v1"
                or outcome.get("candidate_lock_sha256") != lock_sha or _private_fields(outcome)):
            raise ValueError("Independent outcome must be an aggregate bound to the locked candidate")
        records["independent-outcome.json"] = raw
    return records


def archive_trial(source, material, destination, *, fit_only=False):
    records = fit_attempt_records(source, material) if fit_only else public_records(source, material)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in records.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    print(f"Archived {archive_trial(args.source, args.material, args.out, fit_only=args.fit_only)} public evidence files.")
