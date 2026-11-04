"""Publish the hash-checked public evidence from a private preservation trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments.phase_iiij.verification import source_hashes


_ARM = r"(?:A|B|C|D[12]|E[1-4]|L17|LoRA)"
_PUBLIC = re.compile(
    r"(?:practice|diagnosis|parent-decisions|sampling|basis-L(?:17|23)|"
    r"sensitivities-(?:initial|D[12]|alternate)|risk-diagnostic|dose-formulation|candidate-lock)\.json"
    rf"|(?:arm|reference)-{_ARM}\.json"
    rf"|patches/{_ARM}-(?:receipt|policy|patch)\.json"
    rf"|verification/selection-(?:base|{_ARM})\.json"
)


def archive_trial(source, destination):
    root = source.resolve(strict=True)
    report_bytes = (root / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if (report.get("schema") != "weightbridge.phase_iiij.preservation_report.v1"
            or report["sources"] != source_hashes()):
        raise ValueError("Archive requires a completed report tied to the frozen source")
    records = {}
    for name, expected in report["artifacts"].items():
        path = root / name
        if not _PUBLIC.fullmatch(name) or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Report lists a file outside the public evidence allowlist: {name}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Evidence differs from its report hash: {name}")
        records[name] = raw
    records["report.json"] = report_bytes
    if report["selection"]["winner"]:
        path = (root / "independent-outcome.json").resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Independent outcome is outside its trial")
        raw = path.read_bytes()
        outcome = json.loads(raw)
        if (outcome.get("schema") != "weightbridge.phase_iiij.independent_outcome.v1"
                or outcome["candidate_lock_sha256"] != report["selection"]["candidate_lock_sha256"]):
            raise ValueError("Aggregate independent outcome does not bind the selected trial")
        records["independent-outcome.json"] = raw
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in records.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(f"Archived {archive_trial(args.source, args.out)} public evidence files.")
