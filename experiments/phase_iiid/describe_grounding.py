"""Summarize question-level uncertainty and presentation diagnostics without selecting an endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiid.analysis import DISPLAYS, _bootstrap, _letter, _positions, ranker_predictions
from experiments.phase_iiid.roster import PROTOCOL
from weightbridge.core import file_hash, json_dump_atomic


def _effect(rows, values, config):
    return {"estimate": sum(values) / len(values), "question_interval": _bootstrap(rows, values, config)}


def _ranker_summary(rows, predictions, config):
    correct = [p["task"] == row["correct_position"] for row, p in zip(rows, predictions, strict=True)]
    counterfactual = [p["counterfactual"] == row["counterfactual_position"]
                      for row, p in zip(rows, predictions, strict=True)]
    return {"task_accuracy": _effect(rows, correct, config),
            "counterfactual_accuracy": _effect(rows, counterfactual, config),
            "both_correct_fraction": _effect(rows, [a and b for a, b in zip(correct, counterfactual, strict=True)],
                                             config),
            "directional_margin_advantage": _effect(
                rows, [p["directional_margin_advantage"] for p in predictions], config)}


def describe_split(rows, decision, config):
    rule = decision["selected_ranker"]["rule"]
    predictions = ranker_predictions(rows, rule)
    families = {family: [i for i, row in enumerate(rows) if row["family"] == family]
                for family in sorted({row["family"] for row in rows})}
    labels = {}
    for label_set in sorted({"".join(sorted(row["labels"])) for row in rows}):
        subset = [row for row in rows if "".join(sorted(row["labels"])) == label_set]
        selected = ranker_predictions(subset, rule)
        labels[label_set] = {
            "rows": len(subset), "underlying_items": len({r["item_id"] for r in subset}),
            "letter_correct": sum(_letter(row) == row["correct_position"] for row in subset),
            "letter_first_position": sum(_letter(row) == 0 for row in subset),
            "letter_first_named_label": sum(row["labels"][_letter(row)] == label_set[0] for row in subset),
            "ranker_correct": sum(p["task"] == row["correct_position"] for row, p in zip(subset, selected, strict=True)),
            "ranker_first_position": sum(p["task"] == 0 for p in selected),
        }

    ranker_outcomes, generation_outcomes, controls = {}, {}, {}
    for display in DISPLAYS:
        raw_rule = {"display": display, "kind": "sum"}
        calibrated_rule = {"display": display, "kind": "prior_sum", **decision["factorial_prior"]}
        ranker_outcomes[display] = {kind: [p["task"] == row["correct_position"]
                                          for p, row in zip(ranker_predictions(rows, candidate), rows, strict=True)]
                                    for kind, candidate in (("raw", raw_rule), ("calibrated", calibrated_rule))}
        neutral = decision["factorial_prior"]["neutral"]
        controls[display] = {}
        for name in ("task", "counterfactual", "irrelevant", neutral):
            pairs = [_positions(row, display, name, raw_rule) for row in rows]
            controls[display][name] = {
                "left_preferences": sum(left > right for left, right in pairs),
                "right_preferences": sum(right > left for left, right in pairs),
                "ties": sum(left == right for left, right in pairs),
                "mean_absolute_margin": sum(abs(left - right) for left, right in pairs) / len(rows),
            }
        responses = [row["displays"][display]["generation"] for row in rows]
        generation_outcomes[display] = {}
        for parser in ("strict", "semantic"):
            task = [response["task"][f"{parser}_position"] for response in responses]
            changed = [response["counterfactual"][f"{parser}_position"] for response in responses]
            correct = [p == row["correct_position"] for p, row in zip(task, rows, strict=True)]
            changed_correct = [p == row["counterfactual_position"] for p, row in zip(changed, rows, strict=True)]
            generation_outcomes[display][parser] = {
                "parsed": [p is not None for p in task], "correct": correct,
                "counterfactual_correct": changed_correct,
                "both_correct": [a and b for a, b in zip(correct, changed_correct, strict=True)],
            }

    return {
        "selected_ranker": _ranker_summary(rows, predictions, config),
        "selected_ranker_by_family": {
            family: _ranker_summary([rows[i] for i in indices], [predictions[i] for i in indices], config)
            for family, indices in families.items()},
        "label_strata": labels,
        "raw_context_preferences": controls,
        "bullets_minus_labeled_accuracy": {
            key: _effect(rows, [int(b) - int(a) for a, b in zip(
                ranker_outcomes["labeled"][key], ranker_outcomes["bullets"][key], strict=True)], config)
            for key in ("raw", "calibrated")},
        "generation": {
            display: {parser: {key: _effect(rows, values, config) for key, values in outcomes.items()}
                      for parser, outcomes in parsers.items()}
            for display, parsers in generation_outcomes.items()},
        "generation_by_family": {
            family: {
                display: {parser: {key: _effect([rows[i] for i in indices], [values[i] for i in indices], config)
                                   for key, values in outcomes.items()} for parser, outcomes in parsers.items()}
                for display, parsers in generation_outcomes.items()}
            for family, indices in families.items()},
        "bullets_minus_labeled_generation": {
            parser: {key: _effect(rows, [int(b) - int(a) for a, b in zip(
                generation_outcomes["labeled"][parser][key], generation_outcomes["bullets"][parser][key],
                strict=True)], config) for key in generation_outcomes["labeled"][parser]}
            for parser in ("strict", "semantic")},
    }


def describe_run(run, output):
    report_path = run / "report.json"
    decision_path = run / "practice-decision.json"
    report = json.loads(report_path.read_bytes())
    if report["protocol_sha256"] != file_hash(PROTOCOL) or report["practice_decision_sha256"] != file_hash(decision_path):
        raise ValueError("Description requires the original protocol and locked decision")
    config, decision = (json.loads(path.read_bytes()) for path in (PROTOCOL, decision_path))
    summaries = {}
    for split in ("practice", "selection", "qualification"):
        predictions = run / f"{split}-predictions.json"
        if file_hash(predictions) != report["splits"][split]["predictions_sha256"]:
            raise ValueError(f"Raw evidence hash differs for {split}")
        summaries[split] = describe_split(json.loads(predictions.read_bytes())["rows"], decision, config)
    if output.exists():
        raise FileExistsError(output)
    json_dump_atomic(output, {
        "schema": "weightbridge.phase_iiid.description.v1", "report_sha256": file_hash(report_path),
        "description_source_sha256": file_hash(Path(__file__)), "splits": summaries,
        "scope": "Descriptive only. Does not choose an endpoint or change a gate.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    describe_run(args.run, args.out)
