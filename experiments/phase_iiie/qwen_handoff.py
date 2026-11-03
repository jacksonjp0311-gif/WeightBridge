"""Turn the archived Qwen measurement gate into a bounded host decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiie.qwen_roster import FAMILIES, PROTOCOL, SPLITS
from weightbridge.adaptation.meta_controller import EvidenceState, MethodSelector
from weightbridge.core import file_hash, json_dump_atomic


def decide(archive: Path, diagnostic_budget: float = 0.) -> dict:
    report_path = archive / "report.json"
    inventory = json.loads((archive / "artifact_manifest.json").read_bytes())
    report_hash = file_hash(report_path)
    protocol_hash = file_hash(PROTOCOL)
    if (inventory.get("schema") != "weightbridge.phase_iiie.qwen_archive.v1"
            or inventory["artifacts"].get("report.json") != report_hash
            or inventory["artifacts"].get("protocol.json") != protocol_hash
            or file_hash(archive / "protocol.json") != protocol_hash):
        raise ValueError("Archive measurement and protocol hashes differ")
    report = json.loads(report_path.read_bytes())
    config = json.loads(PROTOCOL.read_bytes())
    admission = report["admission"]
    selected = report["selected"]
    ranker_gates = {split: report["splits"][split]["rankers"][selected]["passed"] for split in SPLITS}
    if (report.get("schema") != "weightbridge.phase_iiie.qwen_report.v1"
            or report["protocol_sha256"] != protocol_hash
            or report["checkpoint_sha256"] != config["model_sha256"]
            or ranker_gates != admission["ranker_gates"]
            or any(type(value) is not bool for value in ranker_gates.values())
            or admission["candidate_attempts"] != 0 or admission["reserve_opened"] is not False):
        raise ValueError("Expected a pristine, frozen Qwen measurement report")
    measurement_valid = all(ranker_gates.values())
    limits = config["headroom_gate"]
    eligible = {}
    if measurement_valid:
        for family in FAMILIES:
            scope = admission["headroom"][family]
            if all(len(scope[split]["targets"]) >= limits["persistent_errors_per_split"]
                   and len(scope[split]["controls"]) >= limits["preserved_items_per_split"]
                   for split in ("practice", "selection")):
                eligible[family] = scope["practice"]["targets"]
    if list(eligible) != admission["eligible_families"]:
        raise ValueError("Headroom eligibility differs from frozen measurement report")
    state = EvidenceState(measurement_valid, sum(map(len, eligible.values())) if measurement_valid else None,
                          diagnostic_budget, first_order_reachable=None)
    recommendation = MethodSelector().choose(state)
    return {"schema": "weightbridge.phase_iiie.qwen_handoff.v1", "measurement_report_sha256": report_hash,
            "measurement_qualified": measurement_valid, "measurement_blocker": admission["hold_reason"],
            "diagnostic_budget": diagnostic_budget, "reachable_space_measured": False,
            "eligible_practice_targets": eligible, "decision": recommendation.record(),
            "candidate_attempts": 0, "reserve_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostic-budget", type=float, default=0.)
    args = parser.parse_args()
    json_dump_atomic(args.out, decide(args.archive, args.diagnostic_budget))
