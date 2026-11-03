"""Classify generated content responses without changing the frozen parser."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic.run_measurement import parse_response
from weightbridge.core import file_hash, json_dump_atomic


def diagnose(row):
    response = row["conditions"]["content"]["generation"]
    if response["stop_reason"] != "eos":
        return "TOKEN_BUDGET", None
    if response["eligible_position"] is not None:
        return "EXACT_CONTENT", response["eligible_position"]
    text = response["text"].strip()
    args = (row["answers"], row["labels"], row["aliases"])
    position = parse_response(text, *args, "letter")
    if position is not None:
        return "LABEL_ONLY", position
    match = re.fullmatch(r"([a-zA-Z])[.):]\s*(.+)", text, flags=re.S)
    if match:
        label = parse_response(match[1], *args, "letter")
        content = parse_response(match[2], *args, "content")
        if label is not None and content is not None:
            return ("LABEL_WITH_MATCHING_ANSWER", content) if label == content else ("LABEL_ANSWER_CONFLICT", None)
        if label is not None:
            return "LABEL_WITH_UNRECOGNIZED_TEXT", None
    return "OTHER_COMPLETED_TEXT", None


def analyze(rows):
    paired = []
    for row in rows:
        kind, recovered = diagnose(row)
        paired.append({"id": row["id"], "family": row["family"], "kind": kind,
                       "diagnostic_position": recovered, "correct_position": row["correct_position"]})
    return {"rows": paired, "counts": dict(Counter(row["kind"] for row in paired)),
            "diagnostic_correct": sum(row["diagnostic_position"] == row["correct_position"] for row in paired)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for split in ("practice", "selection", "qualification"):
        source = args.measurement / f"{split}-predictions.json"
        results[split] = {"source_sha256": file_hash(source), **analyze(json.loads(source.read_bytes())["rows"])}
    json_dump_atomic(args.out, {"schema": "weightbridge.phase_iiic.generation_diagnosis.v1",
                                "scope": "Post-hoc response-format diagnosis, not part of the gate.",
                                "splits": results})
    print(json.dumps({split: values["counts"] for split, values in results.items()}, indent=2))
