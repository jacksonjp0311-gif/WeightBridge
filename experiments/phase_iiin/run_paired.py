"""Compare one-sided, paired-mean and pair-aware supervision in pinned geometry."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from math import ceil
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan
from experiments.phase_iiij.run_preservation import _preservation_group
from experiments.phase_iiik.run_jspace import _active_measurements
from experiments.phase_iiil.run_semantic import _margins_record, diagnose
from experiments.phase_iiil.semantic_diagnostics import summarize
from experiments.phase_iiim.run_factorial import _prepare
from experiments.phase_iiin.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.factorial_geometry import match_stored_norm, stored_update_cosines
from weightbridge.adaptation.factorial_probe import StoredAffineSpace
from weightbridge.adaptation.factorial_update import measure_fork_token_classes
from weightbridge.adaptation.generative_fork import GenerativeFork
from weightbridge.adaptation.paired_update import PairedReplayError, fit_paired_basis
from weightbridge.adaptation.paired_supervision import ComparisonEvidence, select_supervision, mean_direction, pair_gradient_diagnostics, solve_pair_maximin
from weightbridge.adaptation.fork_update import AffineUpdate, prepare_probe_reference, probe_update
from weightbridge.adaptation.robust_jspace import MarginObservation, calibrate_residuals, solve_robust
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
FIT_SPLITS = ("adaptation", "preservation-fit", "calibration")


def _numeric_cell(space, runner, plan, basis, direction, shared, intended, config, *, paired=None):
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
                      "intended_norm": intended, "reason": "zero_or_unavailable_arm_direction"},
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
        if paired is None:
            fitted = solve_robust(direction, gradients, margins, radius=amplitude, margin_floor=floor,
                                  allowance=allowance)
        else:
            fitted = solve_pair_maximin(paired["item_ids"], paired["family_ids"], paired["gradients"],
                paired["margins"], gradients, margins, radius=amplitude, floor=floor + allowance,
                **paired["solver_limits"])
        if fitted.update is None or np.linalg.norm(fitted.update) <= 1e-12:
            return {"realized_norm": 0., "feasible": False, "tensor_sha256": None,
                    "reason": fitted.record().get("status", "no_feasible_pair_update"), "solver": fitted.record()}
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


def _restore_fork(row):
    data = {field.name: row[field.name] for field in fields(GenerativeFork)}
    for name in ("prompt_token_ids", "generated_token_ids", "shared_token_ids", "accepted_token_ids",
                 "preferred_continuation", "unaligned_surfaces"):
        data[name] = tuple(data[name])
    data["accepted_continuations"] = tuple(tuple(value) for value in data["accepted_continuations"])
    return GenerativeFork(**data)


def _eligible_comparisons(parent, diagnosis, contract):
    """Keep replay-qualified errors together across the required wording pair."""
    from collections import defaultdict

    lookup = {row["id"]: row for row in parent["rows"]}
    by_order, eligible = defaultdict(list), defaultdict(list)
    forks = [_restore_fork(row) for row in diagnosis["forks"]]
    for fork in forks:
        by_order[fork.question_id].append(fork)
    for members in by_order.values():
        if (len(members) == len(contract.required_wordings)
                and {lookup[f.prompt_id]["wording"] for f in members} == set(contract.required_wordings)
                and all(f.status == "semantic" and f.replay_status == "matched" for f in members)):
            eligible[lookup[members[0].prompt_id]["comparison_id"]].extend(members)
    by_comparison = defaultdict(list)
    for row in parent["rows"]:
        if row["group"] == "target":
            by_comparison[row["comparison_id"]].append(row)
    records, groups, families = [], {}, {}
    for item, rows in by_comparison.items():
        row = rows[0]
        group = f"{row['tier'].removeprefix('negative-')}-" + {"shorter": "short", "longer": "long"}[row["truth_precision"]]
        members = sorted(eligible[item], key=lambda fork: fork.prompt_id)
        records.append(ComparisonEvidence(item, row["pair_id"], group,
            tuple(fork.prompt_id for fork in members), tuple(row["id"] for row in rows if row["correct"])))
        groups[item], families[item] = group, row["pair_id"]
    return records, eligible, groups, families


def _reference_geometry(config, plan):
    reference = config["fixed_geometry"]
    basis_path, report_path = ROOT / reference["basis_path"], ROOT / reference["fit_report_path"]
    if file_hash(basis_path) != reference["basis_sha256"] or file_hash(report_path) != reference["fit_report_sha256"]:
        raise ValueError("Pinned Phase III-M geometry evidence hash differs")
    previous, report = read_json(basis_path), read_json(report_path)
    archived_plan_path = report_path.with_name("arm-C2.json")
    if file_hash(archived_plan_path) != report["artifacts"]["arm-C2.json"]:
        raise ValueError("Audited reference affine plan differs from its immutable receipt")
    archived_plan = read_json(archived_plan_path)["plan"]
    columns, right = tuple(previous["columns"]), np.asarray(previous["right_factor"], dtype=np.float64)
    if (report["checkpoint_sha256"] != plan.checkpoint_sha256
            or list(columns) != report["detail"]["columns"]
            or right.tolist() != report["detail"]["right_factor"]
            or archived_plan["region"]["tensor_name"] != plan.region.tensor_name
            or archived_plan["region"]["shape"] != list(plan.region.shape)
            or archived_plan["region"]["dtype"] != plan.region.dtype):
        raise ValueError("Phase III-M geometry is not bound to the requested source checkpoint")
    # The inherited region resolver independently checks live shape, dtype and bytes.
    return columns, right, {**reference, "checkpoint_sha256": plan.checkpoint_sha256,
        "tensor_name": plan.region.tensor_name, "columns": list(columns), "right_factor": right.tolist(),
        "columns_reused_exactly": True, "right_factor_reused_exactly": True}


def _pair_preview(paired, dimension, radius):
    preview = solve_pair_maximin(paired["item_ids"], paired["family_ids"], paired["gradients"], paired["margins"],
        np.empty((0, dimension)), np.empty(0), radius=radius, floor=0., **paired["solver_limits"])
    update = preview.update
    if update is None or np.linalg.norm(update) <= 1e-12:
        return np.zeros(dimension), {**preview.record(), "preview_eligible": False,
                                     "reason": "no_pair_aware_preview_direction"}
    return update, {**preview.record(), "preview_eligible": True}


def _fit(base, adapter, runner, capabilities, populations, config, out):
    contract = endpoint(config)
    runner = endpoint_runner(runner, contract)
    _, diagnosis = diagnose(runner, populations["adaptation"][1], contract, config)
    json_dump_atomic(out / "fork-diagnosis.json", diagnosis)
    records, eligible, groups, families = _eligible_comparisons(populations["adaptation"][1], diagnosis, contract)
    selection = select_supervision(records)
    json_dump_atomic(out / "supervision-selection.json", selection.record())
    if selection.record()["status"] != "READY":
        return {"reason": selection.record()["reason"]}, {}
    names = tuple(dict.fromkeys(item for ids in selection.arm_items.values() for item in ids))
    forks = tuple(fork for item in names for fork in eligible[item])
    diagnostic_forks = tuple(_restore_fork(row) for row in diagnosis["forks"] if row["status"] == "semantic")
    replay_batches = tuple(diagnostic_forks[start:start + contract.batch_size]
                           for start in range(0, len(diagnostic_forks), contract.batch_size))
    gradient_forward_bound = sum(sum(any(f.prompt_id in {member.prompt_id for member in eligible[item]}
                                         for f in batch) for batch in replay_batches) for item in names)
    plan = _plan(config, capabilities, config["probe_design"]["fixed_layer"])
    columns, right, reference = _reference_geometry(config, plan)
    json_dump_atomic(out / "reference-geometry.json", reference)
    protected_count = sum(row["correct"] for split in ("adaptation", "preservation-fit")
                          for row in populations[split][1]["rows"])
    calibration_count = sum(row["correct"] for row in populations["calibration"][1]["rows"])
    chunk, design = contract.batch_size, config["probe_design"]
    fork_batches = max(ceil(sum(len(eligible[item]) for item in ids) / chunk) for ids in selection.arm_items.values())
    # Includes all direction scans, parent-prefix caches, native token diagnostics,
    # six candidate probes and both numerical matching/calibration passes.
    forward_bound = (7 * ceil(protected_count / chunk) + ceil(calibration_count / chunk)
        + 6 + diagnosis["resources"]["forward_calls"] + gradient_forward_bound + 3 * (2 * fork_batches + 1)
        + 6 * (2 * design["norm_search_steps"] + 2 + 2 * fork_batches + 1))
    if (protected_count > config["fit_limits"]["maximum_parent_correct_trajectories"]
            or len(names) > config["fit_limits"]["max_comparisons"]
            or len(forks) > config["fit_limits"]["max_forks"]
            or forward_bound > plan.limits.max_forward_calls
            or len(forks) + 2 * design["maximum_sampling_rounds"] * design["anchors"] > plan.limits.max_backward_calls):
        return {"reason": "prospective_compute_or_protection_budget", "forward_bound": forward_bound,
                "protected_presentations": protected_count, "fitted_comparisons": len(names), "forks": len(forks)}, {}
    lookup = {row["id"]: row for row in populations["adaptation"][1]["rows"]}
    metadata = {f.prompt_id: {key: lookup[f.prompt_id][key] for key in (
        "pair_id", "comparison_id", "order", "wording", "tier", "truth_precision")} for f in forks}
    fork_classes = {row["prompt_id"]: row for row in diagnosis["forks"]}
    for prompt, record in metadata.items():
        record["actual_token_class"] = fork_classes[prompt]["actual_category"]
        record["accepted_token_classes"] = fork_classes[prompt]["accepted_categories"]
    try:
        basis, gradients, evidence = fit_paired_basis(adapter, runner, plan, forks, groups,
            columns=columns, right_factor=right,
            item_by_prompt={fork.prompt_id: item for item in names for fork in eligible[item]},
            metadata_by_prompt=metadata, batch_size=contract.batch_size, max_forks=config["fit_limits"]["max_forks"],
            replay_batches=replay_batches)
    except PairedReplayError as error:
        json_dump_atomic(out / "gradient-failure.json", error.evidence)
        raise
    evidence["reference_geometry"] = reference
    json_dump_atomic(out / "basis.json", evidence)
    ids, margins = evidence["item_ids"], np.asarray(evidence["comparison_parent_margins"])
    indexes = {item: i for i, item in enumerate(ids)}
    selected_rows = {arm: np.asarray([indexes[item] for item in items]) for arm, items in selection.arm_items.items()}
    directions = {arm: mean_direction(selection.arm_items[arm], gradients[selected_rows[arm]]) for arm in ("A", "B")}
    limits = design["pair_solver"]
    solver_limits = {"max_search_steps": limits["bisection_steps"],
                     "max_projection_iterations": limits["maximum_iterations"], "tolerance": limits["tolerance"]}
    paired = {"item_ids": selection.arm_items["C"], "family_ids": [families[item] for item in selection.arm_items["C"]],
              "gradients": gradients[selected_rows["C"]], "margins": margins[selected_rows["C"]],
              "solver_limits": solver_limits}
    directions["C"], preview = _pair_preview(paired, gradients.shape[1], max(design["target_norms"]))
    direction_evidence = {"schema": "weightbridge.paired_directions.v1", "arm_items": selection.arm_items,
        "arms": {arm: {"direction": direction.tolist(), "norm": float(np.linalg.norm(direction)),
                       "comparison_slopes": (gradients[selected_rows[arm]] @ (direction / np.linalg.norm(direction))).tolist()
                            if np.linalg.norm(direction) > 1e-12 else None}
                 for arm, direction in directions.items()},
        "pair_aware_preview": preview,
        "pairs": pair_gradient_diagnostics(ids, [families[item] for item in ids], [groups[item] for item in ids],
                                            gradients, margins=margins)}
    json_dump_atomic(out / "directions.json", direction_evidence)
    shared, detail = _prepare(adapter, runner, plan, basis, directions, populations, config, out)
    if shared is None:
        return detail, {}
    result, saved = _compare(base, adapter, runner, plan, basis, forks, directions, selection.arm_items,
        paired, shared, populations, config, out)
    return {**result, "fit_items": list(ids), "supported_groups": evidence["item_groups"],
            "columns": list(basis.columns), "right_factor": basis.right.tolist(),
            "arm_items": {arm: list(items) for arm, items in selection.arm_items.items()},
            "prospective_explicit_forward_bound": forward_bound}, saved

def _compare(base, adapter, runner, plan, basis, forks, directions, arm_items, paired, shared, populations, config, out):
    from experiments.phase_iiin.semantic_metrics import compare
    from weightbridge.adaptation.factorial_geometry import compare_realized_norms

    contract, design = endpoint(config), config["probe_design"]
    batch, memory = contract.batch_size, plan.limits.max_memory_bytes
    parent_rows, parent = populations["adaptation"]
    protected_rows, protected = populations["preservation-fit"]
    pool, sample, selected, _, _, _, _, floor = shared
    prompts = tuple(dict.fromkeys(anchor.prompt_id for anchor in sample.anchors))
    trajectories = tuple(pool.trajectories[prompt] for prompt in prompts)
    reference_cache, references, parent_classes = {}, {}, {}
    arm_forks = {arm: tuple(fork for fork, item in zip(forks, basis.target_items, strict=True) if item in items)
                 for arm, items in arm_items.items()}
    # Fitter order is the de-duplicated comparison order; preserve that exact
    # native fork ordering across cache, token evidence and candidate comparison.
    for arm, members in arm_forks.items():
        key = tuple(fork.prompt_id for fork in members)
        if key not in reference_cache:
            ref = prepare_probe_reference(adapter, runner, members, trajectories, batch_size=batch, max_memory_bytes=memory)
            reference_cache[key] = ref
            parent_classes[arm] = measure_fork_token_classes(runner, members, batch_size=batch,
                max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=memory)
            json_dump_atomic(out / f"probe-reference-{arm}.json", {"prompt_ids": prompts,
                "fork_prompt_ids": list(key), "resources": ref.resources})
        else:
            parent_classes[arm] = {"shared_with": next(name for name in references if references[name] is reference_cache[key])}
        references[arm] = reference_cache[key]
    json_dump_atomic(out / "token-class-parent.json", {"arms": parent_classes})
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
        token_classes = measure_fork_token_classes(runner, current_forks, batch_size=batch,
            max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=memory)
        return {"measurement": measured, "guard": guard, "preservation_fit_measurement": measured_fit,
                "preservation_fit_guard": fit_guard, "fork_token_classes": token_classes}

    for index, intended in enumerate(design["target_norms"], 1):
        for arm, direction in directions.items():
            name = f"{arm}{index}"
            current_forks = arm_forks[arm]
            indices = [i for i, item in enumerate(basis.target_items) if item in arm_items[arm]]
            current_basis = replace(basis, target_ids=tuple(basis.target_ids[i] for i in indices),
                target_items=tuple(basis.target_items[i] for i in indices), target_margins=basis.target_margins[indices],
                target_gradients=basis.target_gradients[indices])
            best, numerical = _numeric_cell(space, runner, plan, current_basis, direction, shared, intended, config,
                                             paired=paired if arm == "C" else None)
            record = {"arm": name, "intended_norm": intended, "plan": plan.record(), **numerical,
                      "probe": {"reason": "unmatched_or_uncertified_stored_candidate"},
                      "fit_comparison_ids": list(arm_items[arm]), "fork_indices": indices,
                      "fork_prompt_ids": [fork.prompt_id for fork in current_forks]}
            if best is not None:
                record["rounded_delta"] = best["delta"].tolist()
                record["left_factor"] = best["update"].tolist()
                record["right_factor"] = basis.right.tolist()
                record["columns"] = list(basis.columns)
                deltas[name] = best["delta"]
            if (best is not None and numerical["norm_match"]["status"] == "MATCHED"
                    and numerical["calibration_sufficient"]):
                factor, allowance = best["update"], numerical["direction_specific_allowance"]
                geometry = _margins_record(current_basis, selected, factor, intended, floor, allowance)
                update = AffineUpdate(basis.columns, factor[:, None], basis.right[None, :], "inequality")
                patch, probe = probe_update(adapter, runner, plan, update, radius=None, seen=seen,
                    evaluate_candidate=evaluate, forks=current_forks, trajectories=trajectories, baseline_endpoint=parent,
                    batch_size=batch, reference=references[arm])
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
    return {"reason": "paired_supervision_complete", "unique_complete_evaluations": budget.used,
            "practice_candidates": sorted(saved)}, saved


def run(base, material, out):
    from experiments.phase_iiin.source_manifest import source_hashes

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
    result = {"schema": "weightbridge.phase_iiin.fit.v1", "sources": sources, "checkpoint_sha256": expected,
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
