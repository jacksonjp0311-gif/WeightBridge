"""Exploratory practice analysis of question score minus a neutral answer prior."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


def analyze(source):
    if source.get("schema") != "weightbridge.phase_iiic.practice_invariance.v1":
        raise ValueError("Expected the practice-only invariance probe")
    paired = []
    groups = defaultdict(list)
    for row in source["rows"]:
        task = np.asarray(row["answers_scores"])
        neutral = np.asarray(row["neutral_scores"])
        picks = {name: int(np.argmax(scores)) for name, scores in
                 (("raw", task), ("neutral", neutral), ("subtracted", task - neutral))}
        record = {"id": row["id"], "item_id": row["item_id"], "family": row["family"],
                  "order": row["order"], "correct_position": row["order"].index(0), "predictions": picks,
                  "task_margin": float(task[0] - task[1]), "neutral_margin": float(neutral[0] - neutral[1])}
        paired.append(record)
        groups[row["item_id"]].append(record)
    summary = {}
    for name in ("raw", "neutral", "subtracted"):
        correct = lambda row: row["predictions"][name] == row["correct_position"]
        summary[name] = {"correct_rows": sum(map(correct, paired)), "rows": len(paired),
                         "both_orders_correct": sum(all(map(correct, rows)) for rows in groups.values()),
                         "same_content_both_orders": sum(len({row["order"][row["predictions"][name]]
                                                              for row in rows}) == 1 for rows in groups.values()),
                         "items": len(groups)}
    item_deltas = [np.mean([int(row["predictions"]["subtracted"] == row["correct_position"])
                            - int(row["predictions"]["raw"] == row["correct_position"]) for row in rows])
                   for rows in groups.values()]
    samples = np.random.default_rng(2026110317).choice(item_deltas, size=(4000, len(groups)), replace=True).mean(axis=1)
    return {"summary": summary, "subtracted_minus_raw_paired_item_bootstrap_95":
            np.quantile(samples, [0.025, 0.975]).tolist(), "rows": paired}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.source.read_bytes()))
    json_dump_atomic(args.out, {"schema": "weightbridge.phase_iiic.neutral_prior_analysis.v1",
                                "practice_invariance_sha256": file_hash(args.source),
                                "scope": "Exploratory practice reanalysis. No inference and no change to frozen selection.",
                                **result})
    print(json.dumps(result["summary"], indent=2))
