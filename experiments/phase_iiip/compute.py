"""Count Track A model work and whole-process costs without nested duplication."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from weightbridge.core import file_hash, json_dump_atomic


def summarize(root):
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_bytes())
    fit, execution, assessment = (read(name) for name in ("trial/fit-report.json", "execution.json", "trial/assessment.json"))
    if (execution["status"] != "COMPLETED" or execution["fit_report_sha256"] != file_hash(root / "trial/fit-report.json")
            or execution["runner_source_sha256"] != fit["sources"]["experiments/phase_iiip/run_right.py"]):
        raise ValueError("Compute requires a source-bound completed Track A execution")
    components = {"completed_fitting": execution["resources"], "assessment_host": assessment["host_resources"]}
    counts, stages, responses, tokens = Counter(), {}, 0, 0
    def add(name, resources):
        observed = {key: resources.get(key, 0) for key in
            ("forward_calls", "backward_calls", "batched_decode_steps", "generated_examples")}
        stages[name] = observed
        counts.update(observed)
    def endpoint_tokens(measurement):
        nonlocal responses, tokens
        responses += len(measurement["rows"])
        tokens += sum(len(row["generated_token_ids"]) for row in measurement["rows"])
    for split in fit["optimization_splits"]:
        parent = read(f"trial/parent-{split}.json")["measurement"]
        add(f"parent-{split}", parent["resources"])
        endpoint_tokens(parent)
    # The inherited runtime preflight executes one model gradient probe before
    # yielding an editable runner. A successful constrained plan requires it.
    # It was not serialized as a separate runtime artifact by this fitter; keep
    # that provenance distinction explicit rather than hiding its model work.
    discovery = {"forward_calls": 1, "backward_calls": 1,
        "evidence_type": "source_derived_from_successful_constrained_plan",
        "scope": "Inherited _open(gradient_probe=True) and adapter._runtime; finite nonzero gradient required by compiled plan. No separate runtime-probe timing retained; whole-fit CPU includes it."}
    if fit["qualification"]["disposition"] == "PASS" and "backward_calls" in fit["detail"]:
        add("runtime-capability-probe-source-derived", discovery)
    else:
        discovery = {"forward_calls": None, "backward_calls": None, "evidence_type": "not_separately_retained"}
    for name in ("fork-diagnosis", "target-block-metadata", "preservation-block-metadata", "calibration-block-metadata",
                 "native-parent-targets", "preservation-decisions"):
        path = root / f"trial/{name}.json"
        if path.exists():
            add(name, json.loads(path.read_bytes())["resources"])
    secants = read("trial/secants.json") if (root / "trial/secants.json").exists() else {}
    for name, record in secants.items():
        add("secant/" + name, record["resources"])
    for name in ("preservation-sensitivities", "calibration-sensitivities"):
        if (root / f"trial/{name}.json").exists():
            for i, attempt in enumerate(read(f"trial/{name}.json")["attempts"]):
                add(f"{name}/{i}", attempt["replay"]["resources"])
    if (root / "trial/calibration-decisions.json").exists():
        add("calibration-pool", read("trial/calibration-decisions.json")["pool"]["resources"])
    unique = set()
    for path in sorted((root / "trial").glob("arm-*.json")):
        arm = json.loads(path.read_bytes())
        for scope in ("numerical", "probe"):
            if "resources" in arm.get(scope, {}):
                add(f"{path.stem}/{scope}", arm[scope]["resources"])
        probe = arm["probe"]
        if probe.get("endpoint") and not arm.get("duplicate_of"):
            if probe["candidate_tensor_sha256"] in unique:
                raise ValueError("Repeated rounded generation state lacks a duplicate identity")
            unique.add(probe["candidate_tensor_sha256"])
            for key in ("measurement", "preservation_fit_measurement"):
                measured = probe["endpoint"][key]
                add(f"{path.stem}/{key}", measured["resources"])
                endpoint_tokens(measured)
    for name in assessment["worker_resources"]:
        worker = read("trial/verification/" + name)
        components[name] = worker["resources"]
        add(name, worker["measurement"]["resources"])
        endpoint_tokens(worker["measurement"])
    for path in root.glob("reconstruction-*.json"):
        components[path.stem] = json.loads(path.read_bytes())["resources"]
    return {"schema": "weightbridge.phase_iiip.compute.v1", "track": "A", "source_sha256": file_hash(__file__),
        "components": components, "recorded_cpu_seconds": sum(value["cpu_seconds"] for value in components.values()),
        "fit_and_assessment_wall_seconds": execution["resources"]["wall_seconds"] + assessment["host_resources"]["wall_seconds"],
        "maximum_peak_rss_bytes": max(value["peak_rss_bytes"] for value in components.values()),
        "model_work_by_stage": stages, "explicit_forward_calls": counts["forward_calls"],
        "runtime_discovery_work": discovery,
        "backward_calls": counts["backward_calls"], "greedy_batched_steps": counts["batched_decode_steps"],
        "total_model_forward_calls": counts["forward_calls"] + counts["batched_decode_steps"],
        "generated_response_records": responses, "generated_native_tokens": tokens,
        "unique_complete_candidate_practice_probes": len(unique), "saved_candidates": len(fit["candidates"]),
        "AI_token_tool_cost": "unavailable", "compute_per_verified_improvement": None,
        "scope": "Fit wrapper once plus assessment host CPU and independent worker CPU once. Host wall already includes its workers. Nested numerical resources and duplicate metadata are not extra CPU costs. Preflight, tests, model-free analysis, orchestrator and unavailable AI costs are separate. Interrupted execution receipts must be reported separately, never silently zeroed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    json_dump_atomic(args.out, summarize(args.root))
