"""Preflight and analyze the preregistered MCP LLM transfer cohort.

The protocol is host-owned. The agent receives MCP tools and a prompt, never
the cohort manifest, task assignment, practice/holdout files, or oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class CohortProtocolError(ValueError):
    """A cohort manifest or task binding is invalid or not ready to run."""


MECHANISMS = (
    "input_direction_reversal",
    "hidden_threshold_shift",
    "output_class_collapse",
    "output_intercept_shift",
)
REPLICATES_PER_MECHANISM = 5
EXPECTED_TASKS = len(MECHANISMS) * REPLICATES_PER_MECHANISM
ROOT = Path(__file__).resolve().parents[2]


def _canonical_hash(value: dict[str, Any]) -> str:
    material = {key: item for key, item in value.items() if key != "protocol_sha256"}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CohortProtocolError(f"Expected a JSON object in {path}")
    return value


def validate_protocol(protocol: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    """Return all reasons a frozen protocol cannot be run as preregistered."""
    errors: list[str] = []
    if protocol.get("schema") != "weightbridge.mcp_llm_cohort_protocol.v2":
        errors.append("unsupported protocol schema")
    if protocol.get("protocol_sha256") != _canonical_hash(protocol):
        errors.append("protocol hash mismatch")

    expected_hashes = protocol.get("source_hashes", {})
    if not isinstance(expected_hashes, dict):
        expected_hashes = {}
        errors.append("source_hashes must be an object")
    for key in ("task_generator", "mcp_tool_server", "agent_prompt", "cohort_evaluator"):
        item = expected_hashes.get(key)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            errors.append(f"missing source hash binding: {key}")
            continue
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing protocol source file: {key}")
        elif _file_sha256(path) != item.get("sha256"):
            errors.append(f"protocol source hash changed: {key}")
    prompt = expected_hashes.get("agent_prompt", {})
    if protocol.get("prompt_sha256") != prompt.get("sha256"):
        errors.append("prompt hash does not match the source hash binding")
    if protocol.get("prompt_path") != prompt.get("path"):
        errors.append("prompt path does not match the source hash binding")

    policy = protocol.get("host_policy", {})
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()
    if policy_hash != protocol.get("host_policy_sha256"):
        errors.append("host policy hash mismatch")
    expected_policy = {
        "schema": "weightbridge.policy.v1",
        "allow_tensors": ["input_weight", "hidden_bias", "output_weight", "output_bias"],
        "max_entries": 1,
        "max_fraction": 1 / 12,
        "max_abs_delta": 8.0,
    }
    if policy != expected_policy:
        errors.append("host policy differs from the preregistered one-entry policy")

    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS:
        errors.append(f"cohort must declare exactly {EXPECTED_TASKS} task slots")
    else:
        valid_tasks = [task for task in tasks if isinstance(task, dict)]
        ids = [task.get("task_id") for task in valid_tasks]
        seeds = [task.get("seed") for task in valid_tasks]
        counts = Counter(task.get("mechanism") for task in valid_tasks
                         if isinstance(task.get("mechanism"), str))
        if (len(ids) != EXPECTED_TASKS
                or any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != EXPECTED_TASKS):
            errors.append("task IDs must be present and unique")
        if (len(seeds) != EXPECTED_TASKS
                or any(type(item) is not int or item < 0 for item in seeds)
                or len(set(seeds)) != EXPECTED_TASKS):
            errors.append("task seeds must be present and unique")
        if counts != Counter({name: REPLICATES_PER_MECHANISM for name in MECHANISMS}):
            errors.append("task roster must balance five seeds across the four declared mechanisms")
        order = protocol.get("execution_order")
        if (not isinstance(order, list) or len(order) != EXPECTED_TASKS
                or set(order) != set(ids)):
            errors.append("execution_order must contain every task ID exactly once")
    model = protocol.get("model_binding", {})
    if not isinstance(model, dict):
        model = {}
        errors.append("model_binding must be an object")
    for key in ("provider", "model_id", "version", "identity_provenance"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            errors.append(f"LLM identity is not pinned: {key}")
    if model.get("identity_verified") is not True:
        errors.append("LLM identity has no host verification")
    sampling = model.get("sampling")
    if not isinstance(sampling, dict):
        errors.append("LLM sampling configuration is missing")
    evidence_path = model.get("identity_evidence_path")
    evidence_hash = model.get("identity_evidence_sha256")
    if not isinstance(evidence_path, str) or not evidence_path:
        errors.append("LLM identity evidence path is missing")
    elif not isinstance(evidence_hash, str) or len(evidence_hash) != 64:
        errors.append("LLM identity evidence hash is missing")
    else:
        path = Path(evidence_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append("LLM identity evidence file is missing")
        elif _file_sha256(path) != evidence_hash:
            errors.append("LLM identity evidence hash mismatch")
        else:
            try:
                receipt = _read_json(path)
            except (OSError, json.JSONDecodeError, CohortProtocolError):
                receipt = {}
            for field in ("provider", "model_id", "version", "sampling"):
                if receipt.get(field) != model.get(field):
                    errors.append(f"LLM identity evidence does not match {field}")
            if (receipt.get("schema") != "weightbridge.llm_identity_receipt.v1"
                    or not isinstance(receipt.get("host_call_id"), str)
                    or not receipt.get("host_call_id")):
                errors.append("LLM identity evidence receipt is incomplete")

    if protocol.get("max_candidates") != 8:
        errors.append("candidate budget must be frozen at eight")
    if protocol.get("max_failure_examples") != 48:
        errors.append("diagnostic example limit must be frozen at 48")
    task_design = protocol.get("task_design", {})
    if (not isinstance(task_design, dict)
            or task_design.get("practice_and_holdout_per_class") != 32
            or task_design.get("fresh_agent_context_per_task") is not True
            or task_design.get("hidden_labels_accessible_to_agent") is not False
            or task_design.get("agent_surface") != "MCP tools only; no filesystem, repository, or network tools"
            or task_design.get("tool_access_isolated") is not True
            or task_design.get("random_control") != (
                "one-entry random edit; same tensor and absolute magnitude when a candidate exists; fallback magnitude 4.0 otherwise")):
        errors.append("practice/holdout separation or fresh-session design is not frozen")
    gate = protocol.get("decision_gate", {})
    expected_gate = {
        "repair_definition": "candidate achieves 100 percent holdout accuracy and baseline is below 100 percent",
        "min_repair_rate": 0.7, "max_regression_rate": 0.05,
        "min_candidate_minus_random": 0.2, "ci_lower_strictly_above": 0.0,
    }
    if gate != expected_gate:
        errors.append("decision gate differs from the preregistered criteria")

    expected_analysis = {
        "bootstrap_seed": 918273,
        "confidence": 0.95,
        "method": "paired task-level accuracy differences; bootstrap seeds within each mechanism stratum",
        "resamples": 10_000,
    }
    if protocol.get("analysis") != expected_analysis:
        errors.append("analysis plan differs from the preregistered stratified bootstrap")

    if protocol.get("status") != "FROZEN":
        errors.append("protocol is a template and has not been frozen for a specific LLM")
    return errors


def load_cohort_task(path: str | Path, task_id: str, *, root: Path = ROOT) -> tuple[dict, dict]:
    """Load one host-owned task assignment, refusing an unpinned protocol."""
    protocol = _read_json(Path(path).resolve(strict=True))
    errors = validate_protocol(protocol, root=root)
    if errors:
        raise CohortProtocolError("; ".join(errors))
    matches = [task for task in protocol["tasks"] if task.get("task_id") == task_id]
    if len(matches) != 1:
        raise CohortProtocolError("task_id is not present exactly once in the cohort manifest")
    return protocol, matches[0]


def _correctness(metrics: dict[str, Any], label: str) -> np.ndarray:
    values = metrics.get("per_example_correct")
    if not isinstance(values, list) or not values or any(type(item) is not bool for item in values):
        raise CohortProtocolError(f"{label} has no valid per-example correctness array")
    return np.asarray(values, dtype=bool)


def _verify_transcript(record: dict[str, Any], *, root: Path) -> bool:
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list) or "agent_mcp_transcript.jsonl" not in artifacts:
        return False
    artifact_root = record.get("artifact_root")
    if not isinstance(artifact_root, str):
        return False
    root_path = Path(artifact_root)
    if not root_path.is_absolute():
        root_path = root / root_path
    transcript = root_path / "agent_mcp_transcript.jsonl"
    if not transcript.is_file():
        return False
    agent = record.get("agent", {})
    if not isinstance(agent, dict) or agent.get("mcp_transcript_sha256") != _file_sha256(transcript):
        return False
    try:
        events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return False
    if not events or any(not isinstance(event, dict)
                         or event.get("schema") != "weightbridge.mcp_exchange.v1"
                         or event.get("sequence") != index
                         for index, event in enumerate(events, 1)):
        return False
    resources = record.get("resources", {})
    return (isinstance(resources, dict)
            and resources.get("mcp_tool_calls") == len(events))


def _verify_identity_evidence(record: dict[str, Any], protocol: dict[str, Any], *, root: Path) -> bool:
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list) or "model_identity_receipt.json" not in artifacts:
        return False
    artifact_root = record.get("artifact_root")
    if not isinstance(artifact_root, str):
        return False
    root_path = Path(artifact_root)
    if not root_path.is_absolute():
        root_path = root / root_path
    receipt_path = root_path / "model_identity_receipt.json"
    if not receipt_path.is_file():
        return False
    model = protocol.get("model_binding", {})
    if not isinstance(model, dict) or _file_sha256(receipt_path) != model.get("identity_evidence_sha256"):
        return False
    try:
        receipt = _read_json(receipt_path)
    except (OSError, json.JSONDecodeError, CohortProtocolError):
        return False
    return (receipt.get("schema") == "weightbridge.llm_identity_receipt.v1"
            and all(receipt.get(field) == model.get(field)
                    for field in ("provider", "model_id", "version", "sampling")))


def _stratified_bootstrap(
    differences: dict[str, list[float]], *, seed: int, resamples: int,
    confidence: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mechanisms = [name for name in MECHANISMS if name in differences]
    if not mechanisms or any(not differences[name] for name in mechanisms):
        raise CohortProtocolError("paired candidate/random data is incomplete by mechanism")
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled: list[float] = []
        for mechanism in mechanisms:
            values = np.asarray(differences[mechanism], dtype=np.float64)
            sampled.extend(values[rng.integers(0, len(values), size=len(values))].tolist())
        draws[index] = float(np.mean(sampled))
    observed = float(np.mean([value for group in differences.values() for value in group]))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {"mean_candidate_minus_random": observed,
            "ci95_stratified_by_mechanism": [float(low), float(high)],
            "seed_clusters_per_mechanism": {key: len(value) for key, value in differences.items()},
            "bootstrap_seed": seed, "resamples": resamples,
            "confidence": confidence}


def summarize_cohort(
    protocol: dict[str, Any], records: list[dict[str, Any]], *, root: Path = ROOT,
) -> dict[str, Any]:
    """Audit cohort completeness and calculate the preregistered gates."""
    protocol_errors = validate_protocol(protocol, root=root)
    task_values = protocol.get("tasks", [])
    if not isinstance(task_values, list):
        task_values = []
    roster = {task.get("task_id"): task for task in task_values
              if isinstance(task, dict) and isinstance(task.get("task_id"), str)}
    protocol_model = protocol.get("model_binding", {})
    if not isinstance(protocol_model, dict):
        protocol_model = {}
    seen: dict[str, dict[str, Any]] = {}
    seen_sessions: set[str] = set()
    errors = list(protocol_errors)
    for record in records:
        if not isinstance(record, dict):
            errors.append("experiment record is not an object")
            continue
        binding = record.get("protocol_binding", {})
        if not isinstance(binding, dict):
            binding = {}
        task_id = binding.get("task_id")
        if not isinstance(task_id, str) or task_id not in roster:
            errors.append(f"unregistered or missing task binding: {task_id}")
            continue
        if task_id in seen:
            errors.append(f"duplicate task record: {task_id}")
            continue
        expected = roster[task_id]
        expected_execution_index = protocol.get("execution_order", []).index(task_id) + 1
        if binding.get("execution_index") != expected_execution_index:
            errors.append(f"execution order differs from manifest for {task_id}")
        if binding.get("protocol_sha256") != protocol.get("protocol_sha256"):
            errors.append(f"protocol hash differs for {task_id}")
        if binding.get("task_seed") != expected.get("seed"):
            errors.append(f"task seed differs from manifest for {task_id}")
        if binding.get("mechanism") != expected.get("mechanism"):
            errors.append(f"task mechanism differs from manifest for {task_id}")
        for key in ("provider", "model_id", "version", "prompt_sha256"):
            if binding.get(key) != (protocol_model.get(key)
                                    if key != "prompt_sha256" else protocol.get("prompt_sha256")):
                errors.append(f"{key} binding differs for {task_id}")
        if binding.get("identity_evidence_sha256") != protocol_model.get("identity_evidence_sha256"):
            errors.append(f"LLM identity evidence hash differs for {task_id}")
        if binding.get("identity_verified") is not True:
            errors.append(f"LLM identity is unverified for {task_id}")
        session_id = binding.get("agent_session_id")
        if not isinstance(session_id, str) or not session_id or session_id in seen_sessions:
            errors.append(f"agent session is missing or reused for {task_id}")
        else:
            seen_sessions.add(session_id)
        evaluation = record.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}
        if evaluation.get("holdout_accessed_by_agent") is not False:
            errors.append(f"holdout exposure is not ruled out for {task_id}")
        if evaluation.get("seed") != expected.get("seed"):
            errors.append(f"evaluation seed differs from manifest for {task_id}")
        oracle = record.get("task_oracle", {})
        if not isinstance(oracle, dict) or oracle.get("mechanism") != expected.get("mechanism"):
            errors.append(f"post-run oracle mechanism differs from manifest for {task_id}")
        candidate_metrics = evaluation.get("metrics", {}).get("ai_candidate") if isinstance(
            evaluation.get("metrics", {}), dict) else None
        control = record.get("control_modification")
        if not isinstance(control, dict) or control.get("parameter_count") != 1:
            errors.append(f"one-entry random control receipt is missing for {task_id}")
        elif candidate_metrics is None:
            if (control.get("magnitude_match_type") != "predeclared_fallback_magnitude"
                    or not np.isclose(abs(float(control.get("delta", np.nan))), 4.0)):
                errors.append(f"no-candidate fallback control is invalid for {task_id}")
        else:
            affected = record.get("affected_parameters", {})
            if (not isinstance(affected, dict)
                    or affected.get("count") != 1
                    or control.get("tensor") != affected.get("tensor")
                    or control.get("magnitude_match_type") != "same_tensor_and_absolute_magnitude"
                    or control.get("index") == affected.get("candidate_tensor_index")
                    or not np.isclose(abs(float(control.get("delta", np.nan))),
                                      float(affected.get("max_abs_delta", np.nan)))):
                errors.append(f"random control is not matched to the AI edit for {task_id}")
        if not _verify_transcript(record, root=root):
            errors.append(f"complete hashed MCP transcript is missing for {task_id}")
        if not _verify_identity_evidence(record, protocol, root=root):
            errors.append(f"hashed LLM identity receipt is missing for {task_id}")
        seen[task_id] = record

    missing = sorted(set(roster) - set(seen))
    if missing:
        errors.append(f"missing {len(missing)} of {len(roster)} planned task records")

    if len(seen) != EXPECTED_TASKS:
        return {"schema": "weightbridge.mcp_llm_cohort_summary.v2",
                "disposition": "INCONCLUSIVE", "planned_tasks": len(roster),
                "records_received": len(records), "records_bound_to_plan": len(seen),
                "missing_task_ids": missing, "reasons": sorted(set(errors)),
                "metrics": None}
    try:
        ordered_records = sorted(seen.values(),
                                 key=lambda item: item["protocol_binding"]["execution_index"])
        timestamps = [datetime.fromisoformat(record["timestamp_utc"].replace("Z", "+00:00"))
                      for record in ordered_records]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            errors.append("task timestamps do not follow the frozen execution order")
    except (KeyError, TypeError, ValueError):
        errors.append("a task record has no valid UTC timestamp for order verification")
    if errors:
        return {"schema": "weightbridge.mcp_llm_cohort_summary.v2",
                "disposition": "INCONCLUSIVE", "planned_tasks": len(roster),
                "records_received": len(records), "records_bound_to_plan": len(seen),
                "missing_task_ids": missing, "reasons": sorted(set(errors)),
                "metrics": None}

    repairs = 0
    regression_numerator = 0
    regression_denominator = 0
    paired_differences: dict[str, list[float]] = defaultdict(list)
    per_mechanism: dict[str, dict[str, Any]] = {}
    for mechanism in MECHANISMS:
        per_mechanism[mechanism] = {"n": 0, "repairs": 0,
                                    "baseline_accuracy": [], "candidate_accuracy": [],
                                    "random_accuracy": []}

    for task_id, record in seen.items():
        binding = record["protocol_binding"]
        mechanism = binding["mechanism"]
        record_evaluation = record.get("evaluation", {})
        if not isinstance(record_evaluation, dict):
            errors.append(f"evaluation record is malformed for {task_id}")
            continue
        metrics = record_evaluation.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"evaluation metrics are malformed for {task_id}")
            continue
        try:
            base = _correctness(metrics.get("baseline", {}), f"{task_id} baseline")
        except CohortProtocolError as exc:
            errors.append(str(exc))
            continue
        candidate_metrics = metrics.get("ai_candidate")
        if candidate_metrics is None:
            candidate = base.copy()
            candidate_accuracy = None
        else:
            try:
                candidate = _correctness(candidate_metrics, f"{task_id} AI candidate")
            except CohortProtocolError as exc:
                errors.append(str(exc))
                continue
            candidate_accuracy = float(candidate.mean())
            if candidate.shape != base.shape:
                errors.append(f"paired array lengths differ for {task_id}")
                continue
        random_metrics = metrics.get("random_control")
        if random_metrics is None:
            errors.append(f"matched random control missing for {task_id}")
            continue
        try:
            random = _correctness(random_metrics, f"{task_id} random control")
        except CohortProtocolError as exc:
            errors.append(str(exc))
            continue
        if random.shape != base.shape:
            errors.append(f"paired random array length differs for {task_id}")
            continue

        group = per_mechanism[mechanism]
        group["n"] += 1
        base_accuracy = float(base.mean())
        random_accuracy = float(random.mean())
        group["baseline_accuracy"].append(base_accuracy)
        group["random_accuracy"].append(random_accuracy)
        candidate_for_analysis = candidate if candidate_metrics is not None else base
        candidate_for_analysis_accuracy = float(candidate_for_analysis.mean())
        group["candidate_accuracy"].append(candidate_for_analysis_accuracy)
        if candidate_metrics is not None and base_accuracy < 1.0 and candidate_accuracy == 1.0:
            repairs += 1
            group["repairs"] += 1
        regression_numerator += int(np.count_nonzero(base & ~candidate_for_analysis))
        regression_denominator += int(np.count_nonzero(base))
        paired_differences[mechanism].append(candidate_for_analysis_accuracy - random_accuracy)

    if errors:
        return {"schema": "weightbridge.mcp_llm_cohort_summary.v2",
                "disposition": "INCONCLUSIVE", "planned_tasks": len(roster),
                "records_received": len(records), "records_bound_to_plan": len(seen),
                "missing_task_ids": missing, "reasons": sorted(set(errors)),
                "metrics": None}

    analysis = protocol["analysis"]
    paired = _stratified_bootstrap(
        paired_differences,
        seed=analysis["bootstrap_seed"],
        resamples=analysis["resamples"],
        confidence=analysis["confidence"],
    )
    repair_rate = repairs / EXPECTED_TASKS
    regression_rate = (regression_numerator / regression_denominator
                       if regression_denominator else 0.0)
    gate = protocol["decision_gate"]
    passed = (repair_rate >= float(gate["min_repair_rate"])
              and regression_rate <= float(gate["max_regression_rate"])
              and paired["mean_candidate_minus_random"] >= float(gate["min_candidate_minus_random"])
              and paired["ci95_stratified_by_mechanism"][0] > float(gate["ci_lower_strictly_above"]))
    disposition = "PASS" if passed else "FAIL"
    for name, group in per_mechanism.items():
        if group["n"]:
            group["repair_rate"] = group["repairs"] / group["n"]
            for key in ("baseline_accuracy", "candidate_accuracy", "random_accuracy"):
                values = group[key]
                group[key] = float(np.mean(values)) if values else None
    return {
        "schema": "weightbridge.mcp_llm_cohort_summary.v2",
        "cohort_id": protocol.get("cohort_id"), "disposition": disposition,
        "planned_tasks": EXPECTED_TASKS, "records_received": len(records),
        "repair_definition": "candidate scores 100% on holdout while baseline is below 100%",
        "analysis": analysis,
        "metrics": {"repairs": repairs, "repair_rate": repair_rate,
                    "regressions": regression_numerator,
                    "baseline_correct_items": regression_denominator,
                    "regression_rate": regression_rate,
                    "candidate_minus_matched_random": paired,
                    "by_mechanism": per_mechanism},
        "gate": gate, "reasons": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize a frozen WeightBridge MCP LLM cohort")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--records", nargs="*", type=Path, default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = _read_json(protocol_path)
    records = [_read_json(path.resolve(strict=True)) for path in args.records]
    summary = summarize_cohort(protocol, records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"summary": str(args.out), "disposition": summary["disposition"],
                      "records_received": summary["records_received"],
                      "reasons": summary.get("reasons", [])}, indent=2))


if __name__ == "__main__":
    main()
