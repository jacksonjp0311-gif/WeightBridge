"""Account for recorded joint-study work, including numerical refusals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from weightbridge.core import file_hash, json_dump_atomic


def summarize(root):
    root = Path(root)

    def read(name):
        return json.loads((root / name).read_bytes())

    analysis, execution = read("analysis.json"), read("execution.json")
    fit, assessment = read("trial/fit-report.json"), read("trial/assessment.json")
    if (execution["status"] != "COMPLETED" or execution["fit_report_sha256"] != file_hash(root / "trial/fit-report.json")
            or execution["runner_source_sha256"] != fit["sources"]["experiments/phase_iiio/run_joint.py"]):
        raise ValueError("Execution receipt differs from the source-bound completed fit")
    components = {"completed_fitting": execution["resources"], "assessment_host": assessment["host_resources"]}
    work = analysis["resources"]["fitting_work_totals"]
    explicit, backward = work.get("forward_calls", 0), work.get("backward_calls", 0)
    decode, responses = work.get("batched_decode_steps", 0), work.get("generated_examples", 0)
    tokens, deltas, changed_events = 0, [], 0

    def count_measurement(measured, *, add_work=False):
        nonlocal tokens, decode, responses
        tokens += sum(len(row["generated_token_ids"]) for row in measured["rows"])
        if add_work:
            decode += measured["resources"]["batched_decode_steps"]
            responses += len(measured["rows"])

    for split in fit["optimization_splits"]:
        count_measurement(read(f"trial/parent-{split}.json")["measurement"])
    for name in assessment["worker_resources"]:
        worker = read("trial/verification/" + name)
        components[name] = worker["resources"]
        count_measurement(worker["measurement"], add_work=True)
    for arm in analysis["arms"]:
        record = read(f"trial/arm-{arm}.json")
        if record["probe"].get("endpoint") and not record.get("duplicate_of"):
            for key in ("measurement", "preservation_fit_measurement"):
                count_measurement(record["probe"]["endpoint"][key])
            delta = np.asarray(record["rounded_delta"])
            deltas.append(delta != 0)
            changed_events += int(np.count_nonzero(delta))
    for path in root.glob("reconstruction-*.json"):
        components[path.stem] = json.loads(path.read_bytes())["resources"]
    cpu = sum(value["cpu_seconds"] for value in components.values())
    return {"schema": "weightbridge.phase_iiio.compute.v1", "source_sha256": file_hash(__file__),
        "components": components, "completed_study_cpu_seconds": cpu,
        "completed_fit_wall_seconds": execution["resources"]["wall_seconds"],
        "assessment_wall_seconds_including_workers": assessment["host_resources"]["wall_seconds"],
        "completed_fit_and_assessment_wall_seconds": execution["resources"]["wall_seconds"] + assessment["host_resources"]["wall_seconds"],
        "largest_recorded_peak_rss_bytes": max(value.get("peak_rss_bytes", 0) for value in components.values()),
        "primary_fitting_explicit_forwards": explicit, "primary_fitting_backwards": backward,
        "primary_fitting_batched_decode_steps": work.get("batched_decode_steps", 0),
        "all_recorded_model_forward_calls": explicit + decode, "all_recorded_backward_calls": backward,
        "all_recorded_batched_decode_steps": decode, "all_generated_response_records": responses,
        "all_generated_native_tokens": tokens, "unique_complete_candidate_practice_probes": analysis["unique_decoded_probes"],
        "decoded_candidate_changed_entry_events": changed_events,
        "decoded_candidate_distinct_changed_parameter_positions": int(np.count_nonzero(np.logical_or.reduce(deltas))) if deltas else 0,
        "verified_independent_improvements": int(bool(analysis.get("independent", {}).get("disposition") == "PASS")) if analysis.get("independent") else 0,
        "compute_per_verified_improvement": None, "ai_token_tool_cost": "unavailable",
        "scope": "Completed fit process once, host CPU and independent workers once; host wall already includes worker wall. Numerical failures are included. Tests, orchestration, analyst replay CPU and unavailable assistant costs are outside these experimental measurements. Separate interrupted execution receipts remain separate evidence and are not silently treated as zero."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Compute evidence is write-once")
    json_dump_atomic(args.out, summarize(args.root))
