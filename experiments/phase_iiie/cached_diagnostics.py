"""Measure the effect of reusing C diagnostics under an unchanged adaptive policy."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiie import run_synthetic as original
from experiments.phase_iiie.synthetic_tasks import feature_cost, make_tasks
from weightbridge.adaptation.meta_controller import Action, METHODS, MethodSelector, RiskEstimator
from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("cache_protocol.json")
ARCHIVE = ROOT / "results/phase-iii-e/synthetic"


def _add_curvature(task, observation, remaining):
    cost = feature_cost(task, level="D") - feature_cost(task, level="C")
    if cost.units > remaining:
        return None
    # Match the fixture formula using authorized practice derivatives only.
    curvature = float(np.max(np.abs(task.practice.mixed)) * task.radius**2 / 2)
    return {**observation, "interaction_signal": curvature}, cost


def _cached_policy(task, config, models, selector, curvature_value, diagnosis_value):
    task, observation, cost, actions = original._admit(task, "learned_adaptive", config, diagnosis_value)
    if not original._valid(observation):
        reason = "no persistent deficiency" if observation["measurement_valid"] else "measurement unresolved"
        return "hold", reason, actions, cost, None, None
    probed = original._probe(task, "C", config["budget_units"] - cost.units)
    if probed is None:
        return "hold", "diagnostic exceeded remaining budget", actions, cost, None, None
    observation, measured_cost = probed
    cost += measured_cost
    actions.append(Action.REACHABILITY.value)
    probabilities = original._predictions(models, "C", observation)
    bounds = original._bounds(task)
    decision = original._choose(selector, observation, bounds, config["budget_units"] - cost.units, probabilities)
    value = curvature_value.get(original._band(observation), {})
    request = (value.get("n", 0) >= config["minimum_diagnostic_samples"] and value["mean_incremental_utility"] > 0)
    if request:
        expanded = _add_curvature(task, observation, config["budget_units"] - cost.units)
        if expanded is not None:
            observation, measured_cost = expanded
            cost += measured_cost
            actions.append(Action.CURVATURE.value)
            probabilities = original._predictions(models, "D", observation)
            decision = original._choose(selector, observation, bounds, config["budget_units"] - cost.units, probabilities)
    method = decision.action.value
    if method not in METHODS:
        return "hold", decision.reason, actions, cost, None, probabilities
    outcome = original._candidate(task, method, 1.0, config["budget_units"] - cost.units)
    if outcome is None:
        return "hold", "method exceeded remaining budget", actions, cost, None, probabilities
    return method, decision.reason, actions, cost + outcome.cost, outcome, probabilities


def _record(task, result, config):
    method, reason, actions, cost, outcome, probabilities = result
    return {"task_id": task.id, "mechanism": task.mechanism, "method": method, "reason": reason,
            "diagnostics": actions, "cost": original._cost(cost), "outcome": original._outcome_record(outcome),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "utility": original._utility(outcome, cost, config)}


def _fingerprint(row):
    return {key: value for key, value in row.items() if key not in ("cost", "utility")}


def _summary(rows):
    total = {name: sum(row["cost"][name] for row in rows) for name in rows[0]["cost"]}
    return {"n": len(rows), "total_cost": total,
            "diagnostics": dict(Counter(action for row in rows for action in row["diagnostics"])),
            "improved": sum(bool(row["outcome"] and row["outcome"]["improved"]) for row in rows),
            "harmful": sum(bool(row["outcome"] and row["outcome"]["harmful"]) for row in rows),
            "mean_utility": sum(row["utility"] for row in rows) / len(rows)}


def run(output):
    protocol = json.loads(PROTOCOL.read_bytes())
    archive_report = ARCHIVE / "report.json"
    if file_hash(archive_report) != protocol["archived_selector_report_sha256"]:
        raise ValueError("The trained selector report differs from the frozen reference")
    trained = json.loads(archive_report.read_bytes())
    config = json.loads(original.PROTOCOL.read_bytes())
    if protocol["test_seed"] in config["seeds"].values():
        raise ValueError("The cache benchmark requires a fresh population seed")
    files = (PROTOCOL, Path(__file__), original.PROTOCOL, Path(original.__file__),
             Path(original.__file__).with_name("synthetic_tasks.py"), ROOT / "weightbridge/adaptation/meta_controller.py")
    source_hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}
    original_freeze = json.loads((ARCHIVE / "freeze.json").read_bytes())
    if any(source_hashes[name] != digest for name, digest in original_freeze["source_sha256"].items()):
        raise ValueError("Original decision or fixture sources differ from the frozen study")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    json_dump_atomic(output / "freeze.json", {"protocol": protocol, "source_sha256": source_hashes})
    models = {level: RiskEstimator(**{key: np.asarray(value) for key, value in parameters.items()})
              for level, parameters in trained["selectors"].items()}
    selector = MethodSelector(harm_limit=config["harm_limit"], harm_weight=config["harm_weight"],
                              cost_weight=config["cost_weight"], preservation_threshold=config["preservation_threshold"])
    curvature_value, diagnosis_value = trained["curvature_value_by_band"], trained["diagnosis_value"]
    tasks = [replace(task, id=f"cache-{protocol['test_seed']}/{task.id}") for task in
             make_tasks("test", seed=protocol["test_seed"], per_mechanism=protocol["rows_per_mechanism"])]
    rows = {}
    summaries = {}
    for arm in protocol["comparators"]:
        wall, cpu = perf_counter(), process_time()
        records = []
        for task in tasks:
            if arm == "cache_reusing_adaptive":
                result = _cached_policy(task, config, models, selector, curvature_value, diagnosis_value)
            else:
                policy = "learned_D" if arm == "all_diagnostics" else "learned_adaptive"
                result = original._policy(task, policy, config, models, selector, curvature_value, diagnosis_value, None)
            records.append(_record(task, result, config))
        elapsed_wall, elapsed_cpu = perf_counter() - wall, process_time() - cpu
        rows[arm] = records
        summaries[arm] = {**_summary(records), "wall_seconds": elapsed_wall, "cpu_seconds": elapsed_cpu}
    old, cached = rows["original_adaptive"], rows["cache_reusing_adaptive"]
    decisions_match = sum(_fingerprint(a) == _fingerprint(b) for a, b in zip(old, cached, strict=True))
    digests = {}
    for arm in ("original_adaptive", "cache_reusing_adaptive"):
        fingerprints = [_fingerprint(row) for row in rows[arm]]
        digests[arm] = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if any(file_hash(path) != source_hashes[path.relative_to(ROOT).as_posix()] for path in files):
        raise ValueError("A frozen source changed during measurement")
    raw = output / "outcomes.jsonl.gz"
    original._write_rows(raw, ({"arm": arm, **row} for arm, records in rows.items() for row in records))
    units_saved = summaries["original_adaptive"]["total_cost"]["units"] - summaries["cache_reusing_adaptive"]["total_cost"]["units"]
    report = {
        "schema": "weightbridge.anmc.cached_diagnostics_report.v1",
        "freeze_sha256": file_hash(output / "freeze.json"), "outcomes_sha256": file_hash(raw),
        "comparisons": summaries, "adaptive_matching_rows": decisions_match, "adaptive_row_digests": digests,
        "adaptive_work_units_saved": units_saved,
        "gate_passed": decisions_match == len(tasks) and units_saved > 0,
    }
    json_dump_atomic(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))
