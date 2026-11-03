"""Classify a closed ablation from measured evidence without selecting patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.bic_feedback import classify_candidate
from weightbridge.core import file_hash, json_dump_atomic


def classify(report):
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
            )
            decisions.append({"cycle_id": cycle["cycle_id"], "accepted": cycle["accepted"],
                              "practice_disposition": attempt["disposition"], "diagnostic_codes": codes,
                              "observed_common_practice_margin_rms": rms if margin_shifts.size else None,
                              "constant_margin_fraction": constant_fraction})
        result[name] = decisions
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = classify(json.loads(args.report.read_bytes()))
    json_dump_atomic(args.out, {"schema": "weightbridge.phase_iiib.failure_analysis.v1",
                                "report_sha256": file_hash(args.report), "arms": result})
    print(json.dumps(result, indent=2))
