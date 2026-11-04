"""Audit the coverage of the archived Phase III-I preservation anchors."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.phase_iiii.analyze import summarize
from weightbridge.core import file_hash


def _read(path):
    return json.loads(path.read_bytes())


def audit(root):
    trial = root / "trial"
    analysis = summarize(trial, root / "material")
    diagnosis, scan, arm = (_read(trial / name) for name in ("diagnosis.json", "scan.json", "arm-D.json"))
    practice = _read(trial / "practice.json")["measurement"]["rows"]
    by_id = {row["id"]: row for row in practice}
    sampled = {trajectory["prompt_id"] for trajectory in diagnosis["trajectories"]}
    anchors = []
    for anchor in scan["protected_anchors"]:
        trajectory = diagnosis["trajectories"][anchor["trajectory_index"]]
        row = by_id[trajectory["prompt_id"]]
        anchors.append({**anchor, "id": row["id"], "group": row["group"], "tier": row["tier"],
                        "wording": row["wording"], "item": row["item"],
                        "token_id": trajectory["token_ids"][anchor["position"]]})
    chosen = {anchor["id"] for anchor in anchors}
    regressions = [{"id": row_id, "group": by_id[row_id]["group"], "tier": by_id[row_id]["tier"],
                    "sampled_trajectory": row_id in sampled, "explicit_anchor": row_id in chosen}
                   for row_id in arm["probe"]["endpoint"]["guard"]["prohibited_regressions"]]
    return {"schema": "weightbridge.phase_iiij.phase_i_audit.v1",
            "report_sha256": analysis["report_sha256"], "scan_sha256": file_hash(trial / "scan.json"),
            "sampled_trajectory_groups": dict(Counter(by_id[row_id]["group"] for row_id in sorted(sampled))),
            "selected_anchor_groups": dict(Counter(anchor["group"] for anchor in anchors)),
            "anchors": anchors, "regressions": regressions, "disposition": analysis["single_cycle"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/phase-iii-i/causal-fork-v1"))
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), indent=2, sort_keys=True))
