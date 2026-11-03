"""Compare adaptation policies on frozen synthetic development and test populations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiie.synthetic_tasks import (
    Cost, diagnose, diagnosis_cost, feature_cost, features, make_tasks, method_cost_bound, run_method,
)
from weightbridge.adaptation.meta_controller import Action, EvidenceState, METHODS, MethodSelector, RiskEstimator
from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("synthetic_protocol.json")
FEATURE_NAMES = {
    "A": ("target_error_rate", "target_margin", "observation_agreement"),
    "B": ("preservation_risk", "preservation_margin"),
    "C": ("locally_reachable", "reachability_residual", "sensitivity_dispersion"),
    "D": ("interaction_signal",),
}
LEVELS = tuple(FEATURE_NAMES)
FIXED = {f"fixed_{method}": method for method in METHODS}
POLICIES = (*FIXED, "rules", "same_features_no_learning", "always_hold",
            *(f"learned_{level}" for level in LEVELS), "learned_adaptive", "learned_conservative", "oracle")


def _vector(observation, level):
    names = [name for tier in LEVELS if tier <= level for name in FEATURE_NAMES[tier]]
    return [observation[name] for name in names]


def _valid(observation):
    return bool(observation["measurement_valid"] and observation["persistent_deficiency"])


def _band(observation):
    return "linear_feasible" if observation["locally_reachable"] else "linear_infeasible"


def _cost(cost):
    return {**asdict(cost), "units": cost.units}


def _label(outcome):
    return [int(outcome.improved), int(outcome.harmful)]


def _utility(outcome, cost, config):
    gain, harm = _label(outcome) if outcome is not None else (0, 0)
    return gain - config["harm_weight"] * harm - config["cost_weight"] * cost.units


def _bounds(task):
    return tuple(method_cost_bound(task, method).units for method in METHODS)


def _candidate(task, method, radius, remaining):
    bound = method_cost_bound(task, method).units
    if bound > remaining:
        return None
    measured = run_method(task, method, radius_scale=radius)
    if measured.cost.units > bound + 1e-9:
        raise ValueError(f"Method exceeded its declared cost bound: {method}")
    return measured


def _probe(task, level, remaining):
    bound = feature_cost(task, level=level).units
    if bound > remaining:
        return None
    observation, measured = features(task, level=level)
    if measured.units > bound + 1e-9:
        raise ValueError(f"Diagnostic exceeded its declared cost bound: {level}")
    return observation, measured


def _development(tasks, config):
    """Profile completed development runs and account for every arm in that profiling."""
    records, diagnosis_trials, total = [], [], Cost()
    for task in tasks:
        probed = _probe(task, "D", config["budget_units"])
        if probed is None:
            continue
        observation, cost = probed
        total += cost
        diagnosis_paid = Cost()
        needs_diagnosis = not observation["measurement_valid"]
        if needs_diagnosis:
            if cost.units + diagnosis_cost(task).units + feature_cost(task, level="D").units > config["budget_units"]:
                continue
            task, diagnosis_paid = diagnose(task)
            observation, fresh_cost = features(task, level="D")
            cost += diagnosis_paid + fresh_cost
            total += diagnosis_paid + fresh_cost
        outcomes = {}
        if _valid(observation):
            for method in METHODS:
                outcome = _candidate(task, method, 1.0, config["budget_units"] - cost.units)
                if outcome is not None:
                    outcomes[method] = outcome
                    total += outcome.cost
        if needs_diagnosis:
            best = max((_utility(outcome, outcome.cost, config) for outcome in outcomes.values()), default=0.0)
            diagnosis_trials.append(max(0.0, best))
        if len(outcomes) == len(METHODS):
            records.append((task, observation, outcomes))
    return records, diagnosis_trials, total


def _arrays(records, level):
    x = np.asarray([_vector(observation, level) for _, observation, _ in records], dtype=float)
    y = np.asarray([[_label(outcomes[method]) for method in METHODS] for _, _, outcomes in records], dtype=float)
    return x, y


def _support(x, calibration, quantile):
    center = np.median(x, axis=0)
    limits = np.quantile(x, [0.25, 0.75], axis=0)
    scale = np.maximum(0.05, limits[1] - limits[0])
    distance = np.max(np.abs((calibration - center) / scale), axis=1)
    return center, scale, float(np.quantile(distance, quantile))


def _outside(observation, support):
    center, scale, limit = support
    return bool(np.max(np.abs((np.asarray(_vector(observation, "C")) - center) / scale)) > limit)


def _choose(selector, observation, bounds, remaining, probabilities=None, *, outside=False):
    state = EvidenceState(measurement_valid=bool(observation["measurement_valid"]),
                          persistent_errors=int(observation["persistent_deficiency"]),
                          remaining_budget=float(remaining),
                          first_order_reachable=bool(observation["locally_reachable"]),
                          preservation_risk=float(observation["preservation_risk"]),
                          curvature=observation.get("interaction_signal"),
                          sensitivity_disagreement=float(observation["sensitivity_dispersion"]),
                          method_costs=bounds, out_of_support=outside)
    return selector.choose(state, probabilities)


def _predictions(models, level, observation):
    return models[level].predict(np.asarray([_vector(observation, level)], dtype=float))[0]


def _value_of_curvature(records, models, selector, config):
    values = defaultdict(list)
    for task, observation, outcomes in records:
        utilities = {}
        for level in ("C", "D"):
            view = observation if level == "D" else {key: value for key, value in observation.items()
                                                     if key != "interaction_signal"}
            decision = _choose(selector, view, _bounds(task), config["budget_units"],
                               _predictions(models, level, view))
            outcome = outcomes.get(decision.action.value)
            additional = feature_cost(task, level="D") if level == "D" else Cost()
            utilities[level] = _utility(outcome, additional + (outcome.cost if outcome is not None else Cost()), config)
        values[_band(observation)].append(utilities["D"] - utilities["C"])
    return {band: {"n": len(entries), "mean_incremental_utility": float(np.mean(entries))}
            for band, entries in values.items()}


def _admit(task, policy, config, diagnosis_value):
    observation, cost = _probe(task, "A", config["budget_units"])
    actions = []
    if observation["measurement_valid"]:
        return task, observation, cost, actions
    bound = diagnosis_cost(task).units
    if policy.startswith("learned"):
        worth_measuring = diagnosis_value > config["cost_weight"] * (bound + feature_cost(task, level="A").units)
    else:
        worth_measuring = bound <= config["cheap_diagnosis_units"]
    if worth_measuring and cost.units + bound + feature_cost(task, level="A").units <= config["budget_units"]:
        task, diagnostic_cost = diagnose(task)
        if diagnostic_cost.units > bound + 1e-9:
            raise ValueError("Diagnosis exceeded its declared bound")
        observation, fresh_cost = features(task, level="A")
        cost += diagnostic_cost + fresh_cost
        actions.append(Action.MEASURE.value)
    return task, observation, cost, actions


def _policy(task, policy, config, models, selector, curvature_value, diagnosis_value, support):
    if policy == "always_hold":
        return "hold", "fixed abstention", [], Cost(), None, None
    task, observation, cost, actions = _admit(task, policy, config, diagnosis_value)
    if not _valid(observation):
        reason = "no persistent deficiency" if observation["measurement_valid"] else "measurement unresolved"
        return "hold", reason, actions, cost, None, None
    radius = config["conservative_radius_scale"] if policy == "learned_conservative" else 1.0
    remaining = config["budget_units"] - cost.units
    probabilities = None
    if policy in FIXED:
        method, reason = FIXED[policy], "fixed method after host measurement and headroom"
    else:
        level = policy[-1] if policy[-1] in LEVELS else "C"
        observed_level = "D" if policy == "same_features_no_learning" or level == "D" else "C"
        probed = _probe(task, observed_level, remaining)
        if probed is None:
            return "hold", "diagnostic exceeded remaining budget", actions, cost, None, None
        observation, measured_cost = probed
        cost += measured_cost
        actions.append(Action.REACHABILITY.value)
        if observed_level == "D":
            actions.append(Action.CURVATURE.value)
        outside = policy == "learned_conservative" and _outside(observation, support)
        if policy.startswith("learned"):
            probabilities = _predictions(models, level, observation)
        decision = _choose(selector, observation, _bounds(task), config["budget_units"] - cost.units,
                           probabilities, outside=outside)
        if policy in ("rules", "learned_adaptive") and observed_level == "C":
            value = curvature_value.get(_band(observation), {})
            # Obtaining D invokes the full fixture diagnostic and is charged as such.
            if policy == "rules":
                request = decision.action == Action.CURVATURE
            else:
                request = (value.get("n", 0) >= config["minimum_diagnostic_samples"]
                           and value["mean_incremental_utility"] > 0)
            probed = _probe(task, "D", config["budget_units"] - cost.units) if request else None
            if probed is not None:
                expanded, expanded_cost = probed
                cost += expanded_cost
                actions.append(Action.CURVATURE.value)
                if policy == "learned_adaptive":
                    probabilities = _predictions(models, "D", expanded)
                decision = _choose(selector, expanded, _bounds(task), config["budget_units"] - cost.units,
                                   probabilities)
        method, reason = decision.action.value, decision.reason
        remaining = config["budget_units"] - cost.units
    if method not in METHODS:
        return "hold", reason, actions, cost, None, probabilities
    outcome = _candidate(task, method, radius, remaining)
    if outcome is None:
        return "hold", "method exceeded remaining budget", actions, cost, None, probabilities
    return method, reason, actions, cost + outcome.cost, outcome, probabilities


def _truth(task, config):
    """Evaluator-only profile for false HOLD denominators and the unattainable oracle."""
    observation, base_cost = features(task, level="A")
    audit_cost, actions = base_cost, []
    if (not observation["measurement_valid"]
            and base_cost.units + diagnosis_cost(task).units + feature_cost(task, level="A").units <= config["budget_units"]):
        task, paid = diagnose(task)
        observation, fresh = features(task, level="A")
        base_cost += paid + fresh
        audit_cost = base_cost
        actions.append(Action.MEASURE.value)
    outcomes, conservative = {}, {}
    if _valid(observation):
        for method in METHODS:
            outcome = _candidate(task, method, 1.0, config["budget_units"] - base_cost.units)
            if outcome is not None:
                outcomes[method] = outcome
                audit_cost += outcome.cost
                smaller = _candidate(task, method, config["conservative_radius_scale"],
                                     config["budget_units"] - base_cost.units)
                conservative[method] = smaller
                audit_cost += smaller.cost
    return outcomes, conservative, base_cost, actions, audit_cost


def _oracle(truth, config):
    outcomes, conservative, base_cost, actions, _ = truth
    choices = [(method, outcome) for population in (outcomes, conservative)
               for method, outcome in population.items() if outcome.improved]
    if not choices:
        return "hold", "oracle found no verified method", [], Cost(), None, None
    method, outcome = max(choices, key=lambda pair: _utility(pair[1], pair[1].cost, config))
    return method, "oracle read independent outcomes", actions, base_cost + outcome.cost, outcome, None


def _outcome_record(outcome):
    return None if outcome is None else {**asdict(outcome), "cost": _cost(outcome.cost)}


def _record(task, result, truth, policy, config):
    method, reason, diagnostics, cost, outcome, probabilities = result
    success_available = any(candidate.improved for population in truth[:2] for candidate in population.values())
    improvement, harm = _label(outcome) if outcome is not None else (0, 0)
    return {
        "task_id": task.id, "mechanism": task.mechanism, "twin_id": task.twin_id, "policy": policy,
        "method": method, "reason": reason, "diagnostics": diagnostics, "cost": _cost(cost),
        "verified_path_available": success_available, "improved": bool(improvement), "harmful": bool(harm),
        "false_hold": method == "hold" and success_available,
        "false_admission": method != "hold" and not improvement,
        "unnecessary_admission": method != "hold" and not success_available,
        "risk_adjusted_utility": _utility(outcome, cost, config),
        "outcome": _outcome_record(outcome),
        "probabilities": probabilities.tolist() if probabilities is not None else None,
    }


def _aggregate(rows):
    count = len(rows)
    total_cost = {name: sum(row["cost"][name] for row in rows) for name in rows[0]["cost"]}
    improvement = sum(row["improved"] for row in rows)
    attempts = sum(row["method"] != "hold" for row in rows)
    verified_paths = sum(row["verified_path_available"] for row in rows)
    false_hold = sum(row["false_hold"] for row in rows)
    false_admission = sum(row["false_admission"] for row in rows)
    return {
        "n": count, "improved": improvement, "harmful": sum(row["harmful"] for row in rows),
        "accepted": sum(row["outcome"] is not None and row["outcome"]["accepted"] for row in rows),
        "practice_pass": sum(row["outcome"] is not None and row["outcome"]["practice_pass"] for row in rows),
        "attempts": attempts, "abstentions": count - attempts,
        "verified_paths_available": verified_paths, "false_hold": false_hold, "false_admission": false_admission,
        "unnecessary_admission": sum(row["unnecessary_admission"] for row in rows),
        "false_hold_rate_among_feasible": false_hold / verified_paths if verified_paths else None,
        "false_admission_rate_among_attempts": false_admission / attempts if attempts else None,
        "mean_risk_adjusted_utility": sum(row["risk_adjusted_utility"] for row in rows) / count,
        "total_cost": total_cost, "units_per_improvement": total_cost["units"] / improvement if improvement else None,
        "methods": dict(Counter(row["method"] for row in rows)),
        "diagnostics": dict(Counter(action for row in rows for action in row["diagnostics"])),
    }


def _twin_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["twin_id"] is not None:
            groups[row["twin_id"]].append(row)
    if not groups:
        return None
    pairs = list(groups.values())
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Expected complete observational twin pairs")
    return {
        "pairs": len(pairs),
        "same_decision_pairs": sum(a["method"] == b["method"] and a["diagnostics"] == b["diagnostics"]
                                   for a, b in pairs),
        "both_hold_pairs": sum(a["method"] == b["method"] == "hold" for a, b in pairs),
        "opposite_verified_outcome_pairs": sum((a["improved"] and b["harmful"]) or (b["improved"] and a["harmful"])
                                               for a, b in pairs),
    }


def _calibration(probabilities, labels, bins):
    result = {}
    for method_index, method in enumerate(METHODS):
        per_event = {}
        for index, event in enumerate(("improvement", "harm")):
            p, y = probabilities[:, method_index, index], labels[:, method_index, index]
            membership = np.minimum((bins * p).astype(int), bins - 1)
            reliability = []
            for bucket in np.unique(membership):
                keep = membership == bucket
                reliability.append({"bin": int(bucket), "n": int(keep.sum()),
                                    "predicted": float(p[keep].mean()), "observed": float(y[keep].mean())})
            ece = sum(row["n"] * abs(row["predicted"] - row["observed"]) for row in reliability) / len(p)
            per_event[event] = {"brier": float(np.mean((p - y) ** 2)), "ece": ece, "bins": reliability}
        result[method] = per_event
    return result


def _paired(rows, reference, config):
    if [row["task_id"] for row in rows] != [row["task_id"] for row in reference]:
        raise ValueError("Policy and reference populations are not paired")
    rng = np.random.default_rng(config["bootstrap_seed"])
    result = {}
    for metric in ("improved", "harmful", "risk_adjusted_utility"):
        groups = defaultdict(list)
        for row, base in zip(rows, reference, strict=True):
            groups[row["mechanism"]].append(float(row[metric]) - float(base[metric]))
        # The main familiar benchmark contains independent tasks. Shift twins
        # are reported as exact pairs and are excluded from this interval.
        boot = [rng.choice(values, size=(config["bootstrap_replicates"], len(values)), replace=True)
                for values in groups.values()]
        result[metric] = {"difference": float(np.mean([v for values in groups.values() for v in values])),
                          "interval": np.quantile(np.concatenate(boot, axis=1).mean(axis=1), [0.025, 0.975]).tolist()}
    return result


def _write_rows(path, rows):
    with path.open("xb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as archive:
        for row in rows:
            archive.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def run(output):
    started, cpu_started = perf_counter(), process_time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(PROTOCOL.read_bytes())
    files = [PROTOCOL, Path(__file__), Path(__file__).with_name("synthetic_tasks.py"),
             ROOT / "weightbridge/adaptation/meta_controller.py"]
    sources = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}
    json_dump_atomic(output / "freeze.json", {"schema": "weightbridge.anmc.synthetic_freeze.v1",
                                             "source_sha256": sources, "protocol": config})
    populations = {split: make_tasks(split, seed=seed, per_mechanism=config["rows_per_mechanism"][split])
                   for split, seed in config["seeds"].items()}
    task_ids = [task.id for tasks in populations.values() for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task populations overlap")
    selector = MethodSelector(harm_limit=config["harm_limit"], harm_weight=config["harm_weight"],
                              cost_weight=config["cost_weight"], preservation_threshold=config["preservation_threshold"])
    development = {}
    for split in ("train", "calibration"):
        wall, cpu = perf_counter(), process_time()
        records, diagnosis_trials, cost = _development(populations[split], config)
        development[split] = (records, diagnosis_trials, cost, perf_counter() - wall, process_time() - cpu)
    fit_wall, fit_cpu = perf_counter(), process_time()
    models = {}
    for level in LEVELS:
        x, y = _arrays(development["train"][0], level)
        cx, cy = _arrays(development["calibration"][0], level)
        models[level] = RiskEstimator.fit(x, y).calibrate(cx, cy)
    support = _support(_arrays(development["train"][0], "C")[0], _arrays(development["calibration"][0], "C")[0],
                       config["support_quantile"])
    curvature_value = _value_of_curvature(development["calibration"][0], models, selector, config)
    calibration_trials = development["calibration"][1]
    diagnosis_value = float(np.mean(calibration_trials)) if len(calibration_trials) >= config["minimum_diagnostic_samples"] else 0.0
    fit_time = {"wall_seconds": perf_counter() - fit_wall, "cpu_seconds": process_time() - fit_cpu}
    report = {
        "schema": "weightbridge.anmc.synthetic_report.v1", "freeze_sha256": file_hash(output / "freeze.json"),
        "feature_names": FEATURE_NAMES, "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "policies": POLICIES,
        "development": {split: {"profiled": len(records), "measurement_diagnosis_trials": len(trials),
                                "cost": _cost(cost), "wall_seconds": wall, "cpu_seconds": cpu}
                        for split, (records, trials, cost, wall, cpu) in development.items()},
        "selector_fit": fit_time, "curvature_value_by_band": curvature_value, "diagnosis_value": diagnosis_value,
        "selectors": {level: {name: value.tolist() if value is not None else None
                              for name, value in asdict(model).items()} for level, model in models.items()},
        "support": {"center": support[0].tolist(), "scale": support[1].tolist(), "limit": support[2]},
        "populations": {split: dict(Counter(task.mechanism for task in tasks)) for split, tasks in populations.items()},
        "test": {}, "shift": {},
    }
    all_rows = []
    for split in ("test", "shift"):
        audit_wall, audit_cpu = perf_counter(), process_time()
        tasks = populations[split]
        truths = {task.id: _truth(task, config) for task in tasks}
        audit_cost = sum((truth[-1] for truth in truths.values()), Cost())
        report[split]["evaluator_profiling"] = {"cost": _cost(audit_cost), "wall_seconds": perf_counter() - audit_wall,
                                                "cpu_seconds": process_time() - audit_cpu}
        policy_rows = {}
        for policy in POLICIES:
            wall, cpu = perf_counter(), process_time()
            rows = []
            for task in tasks:
                truth = truths[task.id]
                result = (_oracle(truth, config) if policy == "oracle" else
                          _policy(task, policy, config, models, selector, curvature_value, diagnosis_value, support))
                rows.append(_record(task, result, truth, policy, config))
            policy_rows[policy] = rows
            all_rows.extend({"split": split, **row} for row in rows)
            groups = defaultdict(list)
            for row in rows:
                groups[row["mechanism"]].append(row)
            report[split][policy] = {**_aggregate(rows), "wall_seconds": perf_counter() - wall,
                                     "cpu_seconds": process_time() - cpu,
                                     "by_mechanism": {group: _aggregate(items) for group, items in groups.items()},
                                     "twins": _twin_summary(rows)}
        if split == "test":
            report[split]["paired_against_rules"] = {name: _paired(rows, policy_rows["rules"], config)
                                                     for name, rows in policy_rows.items() if name != "rules"}
        # Score probabilities across all three methods on common eligible
        # independent cases. This audit is separate from deployed policy cost.
        wall, cpu = perf_counter(), process_time()
        audit_records, _, calibration_cost = _development(tasks, config)
        report[split]["calibration_audit"] = {"n": len(audit_records), "cost": _cost(calibration_cost),
                                                "wall_seconds": perf_counter() - wall,
                                                "cpu_seconds": process_time() - cpu,
                                                "levels": {}}
        for level in LEVELS:
            x, y = _arrays(audit_records, level)
            report[split]["calibration_audit"]["levels"][level] = _calibration(
                models[level].predict(x), y, config["calibration_bins"])
        for task in tasks:
            outcomes, conservative, base_cost, actions, cost = truths[task.id]
            all_rows.append({"split": split, "task_id": task.id, "mechanism": task.mechanism, "twin_id": task.twin_id,
                             "policy": "evaluator_counterfactual_profile", "diagnostics": actions,
                             "base_cost": _cost(base_cost), "cost": _cost(cost),
                             "full_radius": {name: _outcome_record(outcome) for name, outcome in outcomes.items()},
                             "conservative_radius": {name: _outcome_record(outcome)
                                                     for name, outcome in conservative.items()}})
    if any(file_hash(path) != sources[path.relative_to(ROOT).as_posix()] for path in files):
        raise ValueError("Frozen synthetic protocol or source changed during execution")
    raw = output / "outcomes.jsonl.gz"
    _write_rows(raw, all_rows)
    report["outcomes"] = {"rows": len(all_rows), "sha256": file_hash(raw)}
    report["total_runtime"] = {"wall_seconds": perf_counter() - started, "cpu_seconds": process_time() - cpu_started}
    json_dump_atomic(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.out)
    print(json.dumps({"populations": result["populations"], "runtime": result["total_runtime"]}, indent=2))
