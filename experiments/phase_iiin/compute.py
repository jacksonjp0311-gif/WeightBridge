"""Reproduce recorded compute totals without inventing missing failure costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from weightbridge.core import file_hash, json_dump_atomic


def summarize(root):
    def read(name):
        return json.loads((root / name).read_bytes())

    analysis, execution = read("analysis.json"), read("execution.json")
    fit, assessment = read("trial/fit-report.json"), read("trial/assessment.json")
    if (execution["status"] != "COMPLETED" or execution["fit_report_sha256"] != file_hash(root / "trial/fit-report.json")
            or execution["runner_source_sha256"] != fit["sources"]["experiments/phase_iiin/run_paired.py"]):
        raise ValueError("Execution receipt does not bind the completed fitting implementation")
    failure = read("failed-attempt-v1/failure.json")
    diagnostic, reconstruction, bounds = (read(name) for name in (
        "replay-diagnostic.json", "reconstruction-B2.json", "local-pair-bounds.json"))
    components = {"completed_fitting": execution["resources"], "assessment_host": assessment["host_resources"],
        "parent_replay_diagnostic": diagnostic["resources"], "B2_reconstruction": reconstruction["resources"],
        "posthoc_linear_bounds": bounds["resources"],
        **{name: row["resources"] for name, row in (
            (name, read("trial/verification/" + name)) for name in assessment["worker_resources"])}}
    recorded_cpu = sum(row["cpu_seconds"] for row in components.values())
    prior_cpu = failure["resources"]["recorded_cpu_seconds_lower_bound"]
    work = analysis["resources"]["fitting_work_totals"]
    forward, backward = work["forward_calls"], work["backward_calls"]
    decode, responses = work["batched_decode_steps"], work["generated_examples"]
    generated_tokens = 0
    deltas, changed_events = [], 0

    def count_measurement(measured, *, include_work=False):
        nonlocal generated_tokens, decode, responses
        generated_tokens += sum(len(row["generated_token_ids"]) for row in measured["rows"])
        if include_work:
            decode += measured["resources"]["batched_decode_steps"]
            responses += len(measured["rows"])

    for split in fit["optimization_splits"]:
        count_measurement(read(f"trial/parent-{split}.json")["measurement"])
    for arm in analysis["arms"]:
        record = read(f"trial/arm-{arm}.json")
        if record["probe"].get("endpoint"):
            if record.get("duplicate_of"):
                continue
            for key in ("measurement", "preservation_fit_measurement"):
                count_measurement(record["probe"]["endpoint"][key])
            delta = np.asarray(record["rounded_delta"])
            deltas.append(delta != 0)
            changed_events += int(np.count_nonzero(delta))
    for name in assessment["worker_resources"]:
        count_measurement(read("trial/verification/" + name)["measurement"], include_work=True)
    forward += diagnostic["resources"]["forward_calls"]
    for split in fit["optimization_splits"]:
        count_measurement(read(f"failed-attempt-v1/parent-{split}.json")["measurement"], include_work=True)
    forward += read("failed-attempt-v1/fork-diagnosis.json")["resources"]["forward_calls"]
    return {"schema": "weightbridge.phase_iiin.compute.v1", "source_sha256": file_hash(__file__),
        "components": components,
        "completed_run_and_diagnostic_cpu_seconds": recorded_cpu,
        "initial_attempt_recorded_cpu_lower_bound": prior_cpu,
        "total_recorded_cpu_seconds_lower_bound": recorded_cpu + prior_cpu,
        "all_attempt_cpu_accounting_complete": False,
        "missing_scope": "Initial uncaught replay exception did not retain partial gradient or whole-process CPU. Known costs are included once; exact grand total is unavailable.",
        "largest_recorded_peak_rss_bytes": max(row.get("peak_rss_bytes", 0) for row in components.values()),
        "completed_fit_wall_seconds": execution["resources"]["wall_seconds"],
        "assessment_wall_seconds_including_workers": assessment["host_resources"]["wall_seconds"],
        "completed_fit_and_assessment_wall_seconds": execution["resources"]["wall_seconds"] + assessment["host_resources"]["wall_seconds"],
        "primary_fitting_explicit_forwards": work["forward_calls"],
        "primary_fitting_backwards": work["backward_calls"],
        "primary_fitting_batched_decode_steps": work["batched_decode_steps"],
        "all_recorded_model_forward_calls_lower_bound": forward + decode,
        "all_recorded_backward_calls_lower_bound": backward,
        "all_recorded_batched_decode_steps": decode,
        "all_generated_response_records": responses, "all_generated_native_tokens": generated_tokens,
        "unique_complete_candidate_practice_probes": analysis["unique_decoded_probes"],
        "decoded_candidate_changed_entry_events": changed_events,
        "decoded_candidate_distinct_changed_parameter_positions": int(np.count_nonzero(np.logical_or.reduce(deltas))) if deltas else 0,
        "verified_independent_improvements": 0, "compute_per_verified_improvement": None,
        "ai_token_tool_cost": "unavailable",
        "scope": "Fit process once, host CPU plus fresh worker CPU once, no nested CPU duplication. Host wall includes child wall. Initial repeated baselines are actual extra compute, not independent evidence. Tests, imports outside measurement regions, assistant cost and orchestration are excluded."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    json_dump_atomic(args.out, summarize(args.root))
