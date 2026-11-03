"""Classify a closed ablation from measured evidence without selecting patches."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.bic_feedback import classify_candidate
from weightbridge.core import file_hash, json_dump_atomic


def _parent_margin_profile(rows):
    """Summarize low-margin predictions as a post-decision diagnostic."""
    threshold = 0.25
    target, sentinel = rows.get("target", []), rows.get("sentinel", [])
    correct = [row for row in target if row["base_prediction"] == row["label"]]
    fragile = [row for row in correct if row["base_signed_margin"] <= threshold]
    flips = [row for row in sentinel if row["base_prediction"] != row["candidate_prediction"]]
    return {"fragile_margin_threshold": threshold, "target_wrong": len(target) - len(correct),
            "target_correct_fragile": len(fragile), "target_correct_above_threshold": len(correct) - len(fragile),
            "preservation_flips_at_or_below_threshold": sum(abs(row["base_margin"]) <= threshold for row in flips),
            "preservation_flips_above_threshold": sum(abs(row["base_margin"]) > threshold for row in flips)}


def _verify_choice_counts(directory, name, cycle):
    reports = cycle.get("verification", {}).get("artifact_sha256", {})
    counts = {}
    for choice_format, hashes in reports.items():
        path = directory / name / cycle["cycle_id"] / f"verify-{choice_format}.json"
        if file_hash(path) != hashes["report"]:
            raise ValueError("Independent score file differs from its frozen receipt")
        rows = json.loads(path.read_bytes())["target"]
        before, after = [dict(Counter(row[key] for row in rows))
                         for key in ("base_prediction", "candidate_prediction")]
        reversal = len(before) == len(after) == 1 and before.keys() != after.keys()
        counts[choice_format] = {"before": before, "after": after, "unanimous_token_reversal": reversal}
    return counts


def classify(report, report_directory):
    if report.get("schema") != "weightbridge.phase_iiib.qwen_ablation.v1":
        raise ValueError("Expected a completed Phase III-B Qwen ablation")
    result = {}
    for name, arm in report["arms"].items():
        decisions = []
        for cycle in arm["cycles"]:
            attempt = cycle["attempt"]
            practice = attempt.get("practice", {})
            synthesis = attempt.get("synthesis", {})
            scope = attempt.get("scope", {})
            formats = cycle.get("verification", {}).get("formats", {})
            rows = practice.get("pairs", {})
            margin_shifts = np.array([row["candidate_margin"] - row["base_margin"]
                                      for group in ("target", "sentinel") for row in rows.get(group, [])])
            rms = float(np.sqrt(np.mean(margin_shifts**2))) if margin_shifts.size else 0.0
            constant_fraction = float(abs(margin_shifts.mean()) / rms) if rms else None
            origin = synthesis.get("origin_objective")
            predicted = synthesis.get("predicted_objective")
            measured = synthesis.get("observed_objective")
            codes = classify_candidate(
                changed_entries=scope.get("changed", synthesis.get("changed_entries", 0)),
                practice_gain=practice.get("target_gain"), heldout_gain=formats.get("ab", {}).get("target_gain"),
                heldout_regressions=sum(metrics["target_regressions"] for metrics in formats.values()),
                prior_prediction_flips=sum(metrics["prior_flips"] for metrics in formats.values()),
                sentinel_flips=sum(metrics["sentinel_flips"] for metrics in formats.values()),
                expected_objective_delta=predicted - origin if origin is not None and predicted is not None else None,
                measured_objective_delta=measured - origin if origin is not None and measured is not None else None,
                solver_disposition=synthesis.get("disposition"),
                budget_exhausted=attempt["candidate_attempts"] > 0 and not attempt["eligible"],
                behavior_unchanged=bool(margin_shifts.size) and rms == 0,
                global_bias_fraction=constant_fraction,
                objective_floor_violated=predicted is not None and predicted < -1e-9,
                prior_practice_prediction_flips=sum(row["family"] != "generic"
                                                    and row["base_prediction"] != row["candidate_prediction"]
                                                    for row in rows.get("sentinel", [])),
            )
            token_counts = _verify_choice_counts(report_directory, name, cycle)
            if any(format_counts["unanimous_token_reversal"] for format_counts in token_counts.values()):
                codes.append("UNANIMOUS_ANSWER_TOKEN_REVERSAL")
            decisions.append({"cycle_id": cycle["cycle_id"], "accepted": cycle["accepted"],
                              "practice_disposition": attempt["disposition"], "diagnostic_codes": codes,
                              "observed_common_practice_margin_rms": rms if margin_shifts.size else None,
                              "constant_margin_fraction": constant_fraction,
                              "independent_target_choice_counts": token_counts,
                              "common_practice_parent_profile": _parent_margin_profile(rows) if rows else None})
        result[name] = decisions
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = classify(json.loads(args.report.read_bytes()), args.report.parent)
    json_dump_atomic(args.out, {"schema": "weightbridge.phase_iiib.failure_analysis.v1",
                                "report_sha256": file_hash(args.report), "arms": result})
    print(json.dumps(result, indent=2))
