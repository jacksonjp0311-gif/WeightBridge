"""Compute question-clustered contrasts after the frozen measurement run closes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic.measurement_roster import FAMILIES, PROTOCOL
from experiments.phase_iiic.run_measurement import INSTRUCTIONS, _groups, _predict
from weightbridge.core import file_hash, json_dump_atomic


def _contrast(rows, terms, bootstrap):
    rng = np.random.default_rng(bootstrap["seed"])
    strata = []
    for family in FAMILIES:
        group_values = []
        for group in _groups([row for row in rows if row["family"] == family], across_paraphrases=True):
            effects = [sum(weight * int(_predict(row, mode, rule) == row["correct_position"])
                           for weight, mode, rule in terms) for row in group]
            group_values.append(np.mean(effects))
        strata.append(np.array(group_values))
    samples = np.concatenate([rng.choice(values, size=(bootstrap["replicates"], len(values)), replace=True)
                              for values in strata], axis=1).mean(axis=1)
    tail = (1 - bootstrap["interval"]) / 2
    return {"accuracy_difference": float(np.concatenate(strata).mean()),
            "paired_question_interval": [float(value) for value in np.quantile(samples, [tail, 1 - tail])],
            "underlying_items": sum(len(values) for values in strata)}


def analyze(rows, selected, bootstrap):
    ll, lc = ("letter", "label"), ("letter", selected)
    cl, cc = ("content", "label"), ("content", selected)
    contrasts = {
        "instruction_effect_with_label_score": ((1, *cl), (-1, *ll)),
        "instruction_effect_with_content_score": ((1, *cc), (-1, *lc)),
        "scoring_representation_effect_under_letter_instruction": ((1, *lc), (-1, *ll)),
        "scoring_representation_effect_under_content_instruction": ((1, *cc), (-1, *cl)),
        "factorial_interaction": ((1, *cc), (-1, *lc), (-1, *cl), (1, *ll)),
        "generation_instruction_difference": ((1, "content", "generation"), (-1, "letter", "generation")),
    }
    breakdown = {}
    for instruction in INSTRUCTIONS:
        length_groups = {"equal": [], "unequal": []}
        score_groups = []
        for row in rows:
            continuations = row["conditions"][instruction]["content_continuations"]
            score_groups.extend(continuations)
            lengths = [option["answer_tokens"] for option in continuations]
            length_groups["equal" if lengths[0] == lengths[1] else "unequal"].append(row)
        breakdown[instruction] = {"separate_encoding_mismatches": sum(not row["joint_tokenization_matches"]
                                                                       for row in score_groups),
                                  "length_groups": {kind: {
                                      "rows": len(group), **{f"{rule}_correct": sum(
                                          _predict(row, instruction, rule) == row["correct_position"] for row in group)
                                          for rule in ("sum", "mean", "eos_sum", "generation")},
                                  } for kind, group in length_groups.items()}}
    return {"factorial_accuracy_contrasts": {name: _contrast(rows, terms, bootstrap) for name, terms in contrasts.items()},
            "answer_tokenization": breakdown}


def run(measurement, output):
    report_path = measurement / "report.json"
    report, config = json.loads(report_path.read_bytes()), json.loads(PROTOCOL.read_bytes())
    if (report["schema"] != "weightbridge.phase_iiic.measurement_report.v1"
            or report["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("Expected a completed measurement report for the frozen protocol")
    results = {}
    for split in ("practice", "selection", "qualification"):
        path = measurement / f"{split}-predictions.json"
        raw = json.loads(path.read_bytes())
        if raw["summary"] != report["splits"][split]["summary"]:
            raise ValueError(f"{split} predictions do not match the measurement report")
        results[split] = {"predictions_sha256": file_hash(path),
                          **analyze(raw["rows"], report["selected_rule"], config["bootstrap"])}
    json_dump_atomic(output, {"schema": "weightbridge.phase_iiic.factorial_analysis.v1",
                              "report_sha256": file_hash(report_path), "source_sha256": file_hash(__file__),
                              "selected_rule": report["selected_rule"], "splits": results})
    print(json.dumps(results["qualification"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.measurement, args.out)
