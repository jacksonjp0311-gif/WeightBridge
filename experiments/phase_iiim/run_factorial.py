"""Run the frozen objective-by-scaling experiment on authorized fitting data."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from math import ceil
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan
from experiments.phase_iiij.run_preservation import _preservation_group
from experiments.phase_iiik.run_jspace import _active_measurements
from experiments.phase_iiil.run_semantic import GROUP_ORDER, _margins_record, _replay_stable_sample, diagnose
from experiments.phase_iiil.semantic_diagnostics import compare_semantic, summarize
from experiments.phase_iiim.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.counterexample_update import measure_parent_decisions
from weightbridge.adaptation.factorial_geometry import build_factorial_directions, match_stored_norm, stored_update_cosines
from weightbridge.adaptation.factorial_probe import StoredAffineSpace
from weightbridge.adaptation.factorial_update import fit_factorial_basis, measure_fork_token_classes
from weightbridge.adaptation.fork_update import AffineUpdate, prepare_probe_reference, probe_update
from weightbridge.adaptation.jspace_update import measure_directional_secant
from weightbridge.adaptation.robust_jspace import MarginObservation, calibrate_margin_floor, calibrate_residuals, solve_robust
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
FIT_SPLITS = ("adaptation", "preservation-fit", "calibration")


def _prepare(adapter, runner, plan, basis, directions, populations, config, out):
    design, contract = config["probe_design"], endpoint(config)
    batch, memory = contract.batch_size, plan.limits.max_memory_bytes
    settings = {"group_for": _preservation_group, "batch_size": batch, "max_memory_bytes": memory}
    protected = {"rows": populations["adaptation"][1]["rows"] + populations["preservation-fit"][1]["rows"]}
    pool, evidence = measure_parent_decisions(runner, protected, contract, **settings)
    json_dump_atomic(out / "preservation-decisions.json", evidence)
    secants, slopes = {}, {}
    for arm, direction in directions.items():
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            secants[arm] = {"reason": "zero_factorial_direction", "resources": {"forward_calls": 0}}
            continue
        preview = replace(basis, preferred_left=direction / norm)
        slopes[arm], secants[arm] = measure_directional_secant(adapter, runner, plan, preview, pool,
            radius=design["directional_secant_radius"], batch_size=batch)
    json_dump_atomic(out / "secants.json", secants)
    if not slopes or any(not values for values in slopes.values()):
        return None, {"reason": "missing_directional_evidence"}
    common = {anchor.key: min(values[anchor.key] for values in slopes.values()) for anchor in pool.eligible_anchors}
    sampling = {"required_groups": design["preservation_groups"], "required_wordings": contract.required_wordings}
    sample, active, attempts = _replay_stable_sample(adapter, runner, plan, basis, pool, batch=batch,
        rounds=design["maximum_sampling_rounds"], sampling={**sampling, "budget": design["anchors"],
        "mode": "stratified_directional", "slopes": common, "radius": design["directional_secant_radius"]})
    json_dump_atomic(out / "sampling.json", sample.record())
    json_dump_atomic(out / "preservation-sensitivities.json", {"attempts": attempts})
    if not sample.complete or any(anchor.key not in active for anchor in sample.anchors):
        return None, {"reason": "incomplete_replay_qualified_preservation"}
    cal_pool, cal_record = measure_parent_decisions(runner, populations["calibration"][1], contract, **settings)
    cal_sample, cal_gradients, cal_attempts = _replay_stable_sample(adapter, runner, plan, basis, cal_pool, batch=batch,
        rounds=design["maximum_sampling_rounds"], sampling={**sampling, "budget": design["calibration_anchors"],
                                                          "mode": "stratified"})
    json_dump_atomic(out / "calibration-decisions.json", {"pool": cal_record, "sample": cal_sample.record()})
    json_dump_atomic(out / "calibration-sensitivities.json", {"attempts": cal_attempts})
    if not cal_sample.complete or any(anchor.key not in cal_gradients for anchor in cal_sample.anchors):
        return None, {"reason": "incomplete_replay_qualified_calibration"}
    floor_record = calibrate_margin_floor([anchor.margin for anchor in cal_sample.anchors], design["positive_floor_grid"])
    json_dump_atomic(out / "floor.json", floor_record)
    if floor_record["selected_floor"] is None:
        return None, {"reason": "no_positive_calibration_floor"}
    cal_ids = {row["id"]: row.get("pair_id") or row["item"] for row in populations["calibration"][1]["rows"]}
    return (pool, sample, tuple(active[a.key] for a in sample.anchors), cal_pool, cal_sample,
            cal_gradients, cal_ids, floor_record["selected_floor"]), {"reason": "shared_constraints_ready"}


def _numeric_cell(space, runner, plan, basis, direction, shared, intended, config):
    wall, cpu = time.perf_counter(), time.process_time()
    pool, sample, selected, cal_pool, cal_sample, cal_gradients, cal_ids, floor = shared
    design = config["probe_design"]
    gradients = np.stack([value.gradient for value in selected])
    margins = np.array([value.margin for value in selected])
    batch, memory = config["inference"]["batch_size"], plan.limits.max_memory_bytes
    cached, passes, allowance = {}, [], 0.
    calls = 0
    if np.linalg.norm(direction) <= 1e-12:
        return None, {"norm_match": {"status": "UNMATCHED", "selected": None, "attempts": [],
                      "intended_norm": intended, "reason": "zero_factorial_direction"},
                      "calibration_passes": [], "direction_specific_allowance": 0.,
                      "calibration_sufficient": False, "floor": floor,
                      "resources": {**_resources(wall, cpu), "forward_calls": 0, "backward_calls": 0}}

    def active(preview):
        nonlocal calls
        digest = preview["tensor_sha256"]
        if digest not in cached:
            cached[digest] = space.margins(runner, pool, sample.anchors, preview,
                                          batch_size=batch, max_memory_bytes=memory)
            calls += cached[digest][1]["resources"]["forward_calls"]
        return cached[digest]

    def evaluate(amplitude):
        fitted = solve_robust(direction, gradients, margins, radius=amplitude, margin_floor=floor,
                              allowance=allowance)
        if fitted.update is None or np.linalg.norm(fitted.update) <= 1e-12:
            return {"realized_norm": 0., "feasible": False, "tensor_sha256": None,
                    "reason": fitted.status, "solver": fitted.record()}
        preview = space.preview(fitted.update)
        predicted = margins + gradients @ fitted.update
        eligible = bool(0 < preview["realized_norm"] <= plan.limits.max_update_norm and preview["changed_entries"])
        measurement = None
        if eligible:
            observed, measurement = active(preview)
            eligible = all(observed[a.key] > floor and row["replay_argmax_token_id"] == a.token_id
                           for a, row in zip(sample.anchors, measurement["rows"], strict=True))
        required = floor + allowance
        eligible = bool(eligible and np.min(predicted) >= required - 1e-7)
        return {key: preview[key] for key in ("realized_norm", "tensor_sha256", "changed_entries",
                "proposed_norm", "update", "delta", "stored_update_cosine")} | {
                    "feasible": eligible, "solver": fitted.record(), "active_measurement": measurement,
                    "active_constraint_count": int(np.count_nonzero(predicted - required <= 1e-6)),
                    "minimum_predicted_margin": float(np.min(predicted)), "required_margin": required}

    best, matching, certified = None, None, False
    for _ in range(2):
        matching = match_stored_norm(evaluate, intended_norm=intended, maximum_norm=plan.limits.max_update_norm,
            relative_tolerance=design["norm_relative_tolerance"], max_probes=design["norm_search_steps"])
        best = matching.selected
        if matching.record()["status"] != "MATCHED" or best is None:
            passes.append({"norm_match": matching.record(), "allowance_used": allowance})
            break
        preview = space.preview(best["update"])
        actual, cal_measurement = space.margins(runner, cal_pool, cal_sample.anchors, preview,
                                               batch_size=batch, max_memory_bytes=memory)
        calls += cal_measurement["resources"]["forward_calls"]
        observations = tuple(MarginObservation(cal_ids[a.prompt_id], a.prompt_id, a.position,
            cal_gradients[a.key].margin + float(cal_gradients[a.key].gradient @ best["update"]),
            actual[a.key], group=a.group) for a in cal_sample.anchors)
        calibration = calibrate_residuals(observations, quantile=design["adverse_item_quantile"])
        if calibration.allowance is None:
            passes.append({"norm_match": matching.record(), "allowance_used": allowance,
                           "calibration": calibration.record(), "calibration_measurement": cal_measurement,
                           "calibration_sufficient": False, "reason": "insufficient_calibration_items"})
            break
        certified = bool(calibration.allowance <= allowance + 1e-7)
        passes.append({"norm_match": matching.record(), "allowance_used": allowance,
                       "calibration": calibration.record(), "calibration_measurement": cal_measurement,
                       "calibration_sufficient": certified})
        if certified:
            break
        allowance = max(allowance, calibration.allowance)
    return best, {"norm_match": matching.record(), "calibration_passes": passes,
                  "direction_specific_allowance": allowance, "calibration_sufficient": certified,
                  "floor": floor, "resources": {**_resources(wall, cpu), "forward_calls": calls, "backward_calls": 0}}


def _fit(base, adapter, runner, capabilities, populations, config, out):
    contract = endpoint(config)
    runner = endpoint_runner(runner, contract)
    picked, diagnosis = diagnose(runner, populations["adaptation"][1], contract, config)
    json_dump_atomic(out / "fork-diagnosis.json", diagnosis)
    counts = Counter(group for _, group, _ in picked)
    if any(counts[group] != config["fit_limits"]["comparisons_per_group"] for group in GROUP_ORDER):
        return {"reason": "insufficient_balanced_replay_matched_comparisons", "groups": dict(counts)}, {}
    plan = _plan(config, capabilities, config["probe_design"]["fixed_layer"])
    forks = tuple(fork for _, _, members in picked for fork in members)
    protected_count = sum(row["correct"] for split in ("adaptation", "preservation-fit")
                          for row in populations[split][1]["rows"])
    calibration_count = sum(row["correct"] for row in populations["calibration"][1]["rows"])
    chunk = contract.batch_size
    fork_batches = ceil(len(forks) / chunk)
    # Bounds include all numerical adjustments and both calibration passes.
    forward_bound = (9 * ceil(protected_count / chunk) + ceil(calibration_count / chunk)
                     + 6 + 4 * fork_batches + (fork_batches + 1)
                     + 8 * (2 * config["probe_design"]["norm_search_steps"] + 2 + 2 * fork_batches + 1))
    if protected_count > 128 or forward_bound > plan.limits.max_forward_calls:
        return {"reason": "prospective_explicit_forward_budget", "forward_bound": forward_bound,
                "protected_presentations": protected_count}, {}
    lookup = {row["id"]: row for row in populations["adaptation"][1]["rows"]}
    metadata = {f.prompt_id: {key: lookup[f.prompt_id][key] for key in (
        "pair_id", "comparison_id", "order", "wording", "tier", "truth_precision")} for f in forks}
    fork_classes = {row["prompt_id"]: row for row in diagnosis["forks"]}
    for prompt, record in metadata.items():
        record["actual_token_class"] = fork_classes[prompt]["actual_category"]
        record["accepted_token_classes"] = fork_classes[prompt]["accepted_categories"]
    basis, ce, margin, evidence = fit_factorial_basis(adapter, runner, plan, forks,
        {item: group for item, group, _ in picked}, item_by_prompt={
            fork.prompt_id: item for item, _, members in picked for fork in members},
        metadata_by_prompt=metadata, batch_size=contract.batch_size, max_forks=config["fit_limits"]["max_forks"])
    json_dump_atomic(out / "basis.json", evidence)
    if basis is None:
        return {"reason": "no_shared_comparison_basis"}, {}
    factorial = build_factorial_directions(evidence["item_ids"], evidence["item_groups"], ce, margin)
    json_dump_atomic(out / "directions.json", factorial.record())
    shared, detail = _prepare(adapter, runner, plan, basis, factorial.directions, populations, config, out)
    if shared is None:
        return detail, {}
    result, saved = _compare(base, adapter, runner, plan, basis, forks, factorial, shared, populations, config, out)
    return {**result, "fit_items": evidence["item_ids"], "supported_groups": evidence["item_groups"],
            "columns": list(basis.columns), "right_factor": basis.right.tolist(),
            "prospective_explicit_forward_bound": forward_bound}, saved


def _compare(base, adapter, runner, plan, basis, forks, factorial, shared, populations, config, out):
    from experiments.phase_iiim.semantic_metrics import compare
    from weightbridge.adaptation.factorial_geometry import compare_realized_norms

    contract, design = endpoint(config), config["probe_design"]
    batch, memory = contract.batch_size, plan.limits.max_memory_bytes
    parent_rows, parent = populations["adaptation"]
    protected_rows, protected = populations["preservation-fit"]
    pool, sample, selected, _, _, _, _, floor = shared
    prompts = tuple(dict.fromkeys(anchor.prompt_id for anchor in sample.anchors))
    trajectories = tuple(pool.trajectories[prompt] for prompt in prompts)
    ref = prepare_probe_reference(adapter, runner, forks, trajectories, batch_size=batch, max_memory_bytes=memory)
    json_dump_atomic(out / "probe-reference.json", {"prompt_ids": prompts, "resources": ref.resources})
    classes = measure_fork_token_classes(runner, forks, batch_size=batch,
        max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=memory)
    json_dump_atomic(out / "token-class-parent.json", classes)
    space = StoredAffineSpace.open(adapter, runner, plan, basis)
    budget, seen, owners, arms, saved, deltas = CandidateEvaluationBudget(contract), set(), {}, {}, {}, {}

    def evaluate():
        measured = measure_endpoint(runner, parent_rows, contract, candidate_budget=budget)
        measured_fit = measure_endpoint(runner, protected_rows, contract)
        guard = compare(parent, measured, min_comparison_gain=1, min_repeatable_corrections=1)
        fit_guard = compare_endpoint(protected, measured_fit, contract)
        validity = sum(row["valid"] for row in measured["rows"]) / len(measured["rows"])
        guard.update(passed=bool(guard["passed"] and not fit_guard["prohibited_regressions"]
                                 and not fit_guard["lost_valid_presentations"]
                                 and validity >= config["endpoint"]["min_valid_rate"]),
                     target_item_gain=guard["target_gain"], prohibited_regressions=guard["regressions"],
                     lost_valid_presentations=guard["lost_valid"])
        token_classes = measure_fork_token_classes(runner, forks, batch_size=batch,
            max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=memory)
        return {"measurement": measured, "guard": guard, "preservation_fit_measurement": measured_fit,
                "preservation_fit_guard": fit_guard, "fork_token_classes": token_classes}

    for index, intended in enumerate(design["target_norms"], 1):
        for arm, direction in factorial.directions.items():
            name = f"{arm}{index}"
            best, numerical = _numeric_cell(space, runner, plan, basis, direction, shared, intended, config)
            record = {"arm": name, "intended_norm": intended, "plan": plan.record(), **numerical,
                      "probe": {"reason": "unmatched_or_uncertified_stored_candidate"}}
            if best is not None:
                record["rounded_delta"] = best["delta"].tolist()
                record["left_factor"] = best["update"].tolist()
                record["right_factor"] = basis.right.tolist()
                record["columns"] = list(basis.columns)
                deltas[name] = best["delta"]
            if (best is not None and numerical["norm_match"]["status"] == "MATCHED"
                    and numerical["calibration_sufficient"]):
                factor, allowance = best["update"], numerical["direction_specific_allowance"]
                geometry = _margins_record(basis, selected, factor, intended, floor, allowance)
                update = AffineUpdate(basis.columns, factor[:, None], basis.right[None, :], "inequality")
                patch, probe = probe_update(adapter, runner, plan, update, radius=None, seen=seen,
                    evaluate_candidate=evaluate, forks=forks, trajectories=trajectories, baseline_endpoint=parent,
                    batch_size=batch, reference=ref)
                if probe.get("candidate_tensor_sha256") != best["tensor_sha256"]:
                    raise ValueError("Decoded candidate differs from its norm-matched stored identity")
                record.update(probe=probe, geometry=geometry)
                digest = probe["candidate_tensor_sha256"]
                if digest in owners:
                    record["duplicate_of"] = owners[digest]
                elif probe.get("endpoint"):
                    owners[digest] = name
                    active = _active_measurements(sample.anchors, probe, prompts, geometry, floor=floor,
                                                  allowance=allowance)
                    record["active_measurements"] = active
                    guard = probe["endpoint"]["guard"]
                    guard["semantic_behavior_passed"] = guard["passed"]
                    guard["strict_measured_floor_passed"] = bool(active and all(
                        row["decision_preserved"] and row["measured_margin"] > floor for row in active))
                    guard["passed"] = bool(guard["passed"] and guard["strict_measured_floor_passed"])
                    if patch is not None and guard["passed"]:
                        candidate = _materialize(base, out, name, plan, patch, probe)
                        saved[name] = {key: value for key, value in candidate.items() if key != "checkpoint"}
                        saved[name]["receipt_sha256"] = file_hash(candidate["checkpoint"] / ".weightbridge_receipt.json")
            arms[name] = record
            json_dump_atomic(out / f"arm-{name}.json", record)
            guard = record["probe"].get("endpoint", {}).get("guard", {})
            print(f"{name}: {numerical['norm_match']['status']}, "
                  f"calibrated {numerical['calibration_sufficient']}, "
                  f"{record['probe']['reason']}, strict gain {guard.get('target_gain')}", flush=True)
    matches = {}
    for index, intended in enumerate(design["target_norms"], 1):
        states = {name: {"realized_norm": row["norm_match"]["selected"]["realized_norm"],
                         "tensor_sha256": row["norm_match"]["selected"]["tensor_sha256"]}
                  for name, row in arms.items() if name.endswith(str(index)) and row["norm_match"]["selected"]}
        matches[str(index)] = compare_realized_norms(states, intended_norm=intended,
            relative_tolerance=design["norm_relative_tolerance"]) if states else None
    json_dump_atomic(out / "stored-pairwise.json", {"norm_matching": matches,
                    "actual_delta_cosines": stored_update_cosines(deltas) if deltas else {}})
    return {"reason": "factorial_complete", "unique_complete_evaluations": budget.used,
            "practice_candidates": sorted(saved)}, saved


def run(base, material, out):
    from experiments.phase_iiim.source_manifest import source_hashes

    wall, cpu = time.perf_counter(), time.process_time()
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Run in an isolated host workspace")
    config, manifest, sources = read_json(PROTOCOL), read_manifest(material), source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    contract, populations, candidates = endpoint(config), {}, {}
    expected = config["checkpoint_sha256"]
    with _open(base, config, expected, gradient_probe=True) as (adapter, runner, capabilities):
        for split in FIT_SPLITS:
            frozen = load_split(material, split, manifest)
            start, start_cpu = time.perf_counter(), time.process_time()
            measured = measure_endpoint(runner, frozen, contract)
            populations[split] = frozen, measured
            json_dump_atomic(out / f"parent-{split}.json", {"measurement": measured,
                "summary": summarize(measured), "resources": _resources(start, start_cpu)})
        qualification = qualify(populations["adaptation"][1], config)
        if qualification["disposition"] == "PASS":
            detail, candidates = _fit(base, adapter, runner, capabilities, populations, config, out)
        else:
            detail = {"reason": "parent_behavior_did_not_qualify"}
    if sources != source_hashes() or load_checkpoint(base).checkpoint_sha256() != expected:
        raise ValueError("Original checkpoint or experiment source changed")
    artifacts = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))
                 if "checkpoints" not in path.relative_to(out).parts}
    result = {"schema": "weightbridge.phase_iiim.fit.v1", "sources": sources, "checkpoint_sha256": expected,
              "manifest_sha256": file_hash(material / "manifest.json"), "optimization_splits": list(FIT_SPLITS),
              "qualification": qualification, "detail": detail, "candidates": candidates,
              "artifacts": artifacts, "hidden_opened": False, "resources": _resources(wall, cpu)}
    json_dump_atomic(out / "fit-report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.base, args.material, args.out)["detail"], flush=True)
