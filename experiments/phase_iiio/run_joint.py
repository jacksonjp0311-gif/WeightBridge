"""Individual-fork joint feasibility in the audited fixed-right affine subspace."""
from __future__ import annotations

import argparse
from collections import defaultdict
from math import ceil
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan
from experiments.phase_iiil.run_semantic import _margins_record, diagnose
from experiments.phase_iiil.semantic_diagnostics import summarize
from experiments.phase_iiim.run_factorial import _prepare
from experiments.phase_iiin.run_paired import _eligible_comparisons, _reference_geometry, _restore_fork
from experiments.phase_iiio.semantic_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiio.semantic_metrics import compare
from weightbridge.adaptation.behavior_endpoint import CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint
from weightbridge.adaptation.factorial_geometry import compare_realized_norms, match_stored_norm, stored_update_cosines
from weightbridge.adaptation.factorial_probe import StoredAffineSpace
from weightbridge.adaptation.joint_feasibility import solve_joint_feasibility
from weightbridge.adaptation.joint_update import measure_native_forks, measure_stored_joint, probe_joint_generation
from weightbridge.adaptation.paired_update import PairedReplayError, fit_paired_basis
from weightbridge.adaptation.robust_jspace import MarginObservation, calibrate_residuals, solve_robust
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
FIT_SPLITS = ("adaptation", "preservation-fit", "calibration")


def _select_families(parent, diagnosis, contract):
    records, eligible, groups, families = _eligible_comparisons(parent, diagnosis, contract)
    by_family, parent_rows = defaultdict(list), defaultdict(list)
    for record in records:
        by_family[record.family_id].append(record)
    for row in parent["rows"]:
        if row["group"] == "target":
            parent_rows[row["pair_id"]].append(row)
    population, available = [], defaultdict(list)
    for family, members in by_family.items():
        rows = parent_rows[family]
        qualified_ids = {prompt for member in members for prompt in member.eligible_prompt_ids}
        wrong_ids = {row["id"] for row in rows if not row["correct"]}
        complete_coverage = wrong_ids <= qualified_ids
        qualifies = (len(rows) == 8 and all(row["valid"] for row in rows) and len(members) == 2
                     and all(member.eligible for member in members) and complete_coverage)
        distance = members[0].group.split("-")[0]
        population.append({"family_id": family, "distance": distance, "qualified": qualifies,
            "complete_error_fork_coverage": complete_coverage, "uncovered_error_prompt_ids": sorted(wrong_ids - qualified_ids),
            "comparison_ids": [member.comparison_id for member in members],
            "qualified_error_forks": [list(member.eligible_prompt_ids) for member in members],
            "parent_correct_obligations": [list(member.parent_correct_prompt_ids) for member in members]})
        if qualifies:
            available[distance].append(members)
    chosen = []
    for distance in ("close", "far"):
        options = sorted(available[distance], key=lambda rows: (
            -min(len(row.eligible_prompt_ids) for row in rows), rows[0].family_id))
        if options:
            chosen.extend(sorted(options[0], key=lambda row: row.comparison_id))
    ids = tuple(member.comparison_id for member in chosen)
    return ids, eligible, groups, families, {"schema": "weightbridge.joint_family_selection.v1",
        "status": "READY" if len(chosen) == 4 else "HOLD", "reason": "qualified_close_and_far_pairs" if len(chosen) == 4 else "insufficient_fully_replay_covered_families",
        "comparison_ids": list(ids), "family_ids": list(dict.fromkeys(member.family_id for member in chosen)),
        "comparison_count": len(ids), "population": population,
        "selection_rule": "one close and one far; greatest weakest member fork count; frozen family identity"}


def _system(rows, margins, plan, config, *, protected=None, protected_margins=None, floor=0., target_ids=None, preservation_ids=None):
    settings = config["probe_design"]["joint_solver"]
    return {"target_rows": rows.tolist(), "target_parent_margins": np.asarray(margins).tolist(),
        "target_floor": config["probe_design"]["target_margin"],
        "preservation_rows": np.empty((0, rows.shape[1])).tolist() if protected is None else protected.tolist(),
        "preservation_parent_margins": [] if protected_margins is None else np.asarray(protected_margins).tolist(),
        "preservation_floor": floor, "radius": plan.limits.max_update_norm,
        "max_sweeps": settings["max_sweeps"], "tolerance": settings["tolerance"],
        "target_ids": list(target_ids) if target_ids is not None else [f"target/{index}" for index in range(len(rows))],
        "preservation_ids": list(preservation_ids) if preservation_ids is not None else
            [f"preservation/{index}" for index in range(0 if protected is None else len(protected))]}


def _solve(system):
    return solve_joint_feasibility(**system)


def _joint_records(basis, comparison_rows, comparison_margins, evidence, selected, floor, plan, config):
    gradients = np.stack([value.gradient for value in selected])
    margins = np.array([value.margin for value in selected])
    protection = {"protected": gradients, "protected_margins": margins, "floor": floor,
                  "preservation_ids": [f"{value.anchor.prompt_id}:{value.anchor.position}" for value in selected]}
    systems = {"target_only": _system(basis.target_gradients, basis.target_margins, plan, config, target_ids=basis.target_ids),
        "with_preservation": _system(basis.target_gradients, basis.target_margins, plan, config, target_ids=basis.target_ids, **protection),
        "averaged_target_only": _system(comparison_rows, comparison_margins, plan, config, target_ids=evidence["item_ids"])}
    target, joint, averaged = (_solve(systems[key]) for key in ("target_only", "with_preservation", "averaged_target_only"))
    families = tuple(dict.fromkeys(row["metadata"]["pair_id"] for row in evidence["forks"]))
    per_family = {}
    for family in families:
        indices = [index for index, row in enumerate(evidence["forks"]) if row["metadata"]["pair_id"] == family]
        comparison_ids = tuple(dict.fromkeys(evidence["forks"][index]["comparison_id"] for index in indices))
        item_indices = [evidence["item_ids"].index(item) for item in comparison_ids]
        rows, intercepts = basis.target_gradients[indices], basis.target_margins[indices]
        scoped = {
            "target_only": _system(rows, intercepts, plan, config, target_ids=[basis.target_ids[index] for index in indices]),
            "with_preservation": _system(rows, intercepts, plan, config, target_ids=[basis.target_ids[index] for index in indices], **protection),
            "averaged_target_only": _system(comparison_rows[item_indices], comparison_margins[item_indices], plan, config, target_ids=comparison_ids)}
        result = {"fork_indices": indices}
        for name, system in scoped.items():
            systems[f"family/{family}/{name}"] = system
            result[name] = _solve(system).record()
        per_family[family] = result
    tau = config["probe_design"]["target_margin"]
    per_fork = [{"prompt_id": row["prompt_id"], "parent_margin": float(margin),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "target_only_individual_minimum_norm": max(0., tau - float(margin)) / float(np.linalg.norm(gradient)) if np.linalg.norm(gradient) else None,
        "reason": "cauchy_schwarz_single_halfspace" if np.linalg.norm(gradient) else "zero_gradient"}
        for row, gradient, margin in zip(evidence["forks"], basis.target_gradients, basis.target_margins, strict=True)]
    return target, joint, {"schema": "weightbridge.joint_feasibility_study.v1", "systems": systems,
        "averaged_target_only": averaged.record(), "target_only": target.record(), "with_preservation": joint.record(),
        "per_family": per_family, "per_fork": per_fork}


class _ForwardBudget:
    def __init__(self, limit, used):
        self.limit, self.used = limit, used
        if not 0 <= used <= limit:
            raise ValueError("Initial explicit forward work exceeds the frozen model budget")

    def reserve(self, count):
        if self.used + count > self.limit:
            raise _BudgetExhausted("Explicit forward budget exhausted before candidate measurement")
        self.used += count


class _BudgetExhausted(RuntimeError):
    pass


def _shared_forward_calls(out):
    records = {name: read_json(out / name) for name in ("preservation-decisions.json", "secants.json",
        "preservation-sensitivities.json", "calibration-decisions.json", "calibration-sensitivities.json")}
    return (records["preservation-decisions.json"]["resources"]["forward_calls"]
        + sum(row["resources"]["forward_calls"] for row in records["secants.json"].values())
        + sum(row["replay"]["resources"]["forward_calls"] for name in ("preservation-sensitivities.json", "calibration-sensitivities.json") for row in records[name]["attempts"])
        + records["calibration-decisions.json"]["pool"]["resources"]["forward_calls"])


def _fit(base, adapter, runner, capabilities, populations, config, out):
    contract, design = endpoint(config), config["probe_design"]
    runner = endpoint_runner(runner, contract)
    _, diagnosis = diagnose(runner, populations["adaptation"][1], contract, config)
    json_dump_atomic(out / "fork-diagnosis.json", diagnosis)
    names, eligible, groups, families, selection = _select_families(populations["adaptation"][1], diagnosis, contract)
    json_dump_atomic(out / "supervision-selection.json", selection)
    if selection["status"] != "READY":
        return {"reason": selection["reason"]}, {}
    forks = tuple(fork for item in names for fork in eligible[item])
    diagnostic = tuple(_restore_fork(row) for row in diagnosis["forks"] if row["status"] == "semantic")
    contexts = tuple(diagnostic[start:start + contract.batch_size] for start in range(0, len(diagnostic), contract.batch_size))
    plan = _plan(config, capabilities, design["fixed_layer"])
    columns, right, reference = _reference_geometry(config, plan)
    json_dump_atomic(out / "reference-geometry.json", reference)
    protected_count = sum(row["correct"] for split in ("adaptation", "preservation-fit") for row in populations[split][1]["rows"])
    if protected_count > 128 or len(forks) > 16:
        return {"reason": "prospective_protection_or_target_budget", "protected_presentations": protected_count, "forks": len(forks)}, {}
    lookup = {row["id"]: row for row in populations["adaptation"][1]["rows"]}
    context_for = {fork.prompt_id: [member.prompt_id for member in batch] for batch in contexts for fork in batch}
    metadata = {fork.prompt_id: {**{key: lookup[fork.prompt_id][key] for key in (
        "pair_id", "comparison_id", "order", "wording", "tier", "truth_precision")},
        "first_fork_position": len(fork.shared_token_ids), "original_context_prompt_ids": context_for[fork.prompt_id]}
        for fork in forks}
    try:
        basis, rows, evidence = fit_paired_basis(adapter, runner, plan, forks, groups, columns=columns,
            right_factor=right, item_by_prompt={fork.prompt_id: item for item in names for fork in eligible[item]},
            metadata_by_prompt=metadata, batch_size=contract.batch_size, max_forks=16, replay_batches=contexts)
    except PairedReplayError as error:
        json_dump_atomic(out / "gradient-failure.json", error.evidence)
        raise
    evidence["reference_geometry"] = reference
    native = measure_native_forks(runner, forks, contexts, max_forward_calls=plan.limits.max_forward_calls,
                                 max_memory_bytes=plan.limits.max_memory_bytes)
    json_dump_atomic(out / "native-parent-targets.json", native)
    for gradient, actual in zip(evidence["forks"], native["rows"], strict=True):
        gradient["metadata"]["rejected_top_token_id"] = actual["strongest_rejected_token_id"]
        gradient["metadata"]["native_parent_margin"] = actual["measured_margin"]
        gradient["metadata"]["native_gradient_margin_difference"] = actual["measured_margin"] - gradient["parent_margin"]
    json_dump_atomic(out / "basis.json", evidence)
    if any(abs(row["metadata"]["native_gradient_margin_difference"]) > 1e-8 for row in evidence["forks"]):
        return {"reason": "native_gradient_intercept_mismatch"}, {}
    target = _solve(_system(basis.target_gradients, basis.target_margins, plan, config, target_ids=basis.target_ids))
    preview = target.minimum_norm_update
    directions = {"A": rows.mean(axis=0)}
    if preview is not None and np.linalg.norm(preview) > 1e-12:
        directions["target_preview"] = preview
    json_dump_atomic(out / "directions.json", {"schema": "weightbridge.joint_directions.v1",
        "arms": {name: {"direction": direction.tolist(), "norm": float(np.linalg.norm(direction))} for name, direction in directions.items()},
        "target_only_preview": target.record(), "comparison_parent_margins": evidence["comparison_parent_margins"]})
    shared, detail = _prepare(adapter, runner, plan, basis, directions, populations, config, out)
    if shared is None:
        return detail, {}
    target, joint, mathematics = _joint_records(basis, rows, np.asarray(evidence["comparison_parent_margins"]),
        evidence, shared[2], shared[-1], plan, config)
    json_dump_atomic(out / "joint-feasibility.json", mathematics)
    used = diagnosis["resources"]["forward_calls"] + evidence["resources"]["forward_calls"] + native["resources"]["forward_calls"] + _shared_forward_calls(out)
    budget = _ForwardBudget(plan.limits.max_forward_calls, used)
    result, saved = _compare(base, adapter, runner, plan, basis, forks, contexts, directions["A"], shared,
        populations, config, out, budget, target_only_feasible=target.update is not None)
    return {**result, "fit_items": evidence["item_ids"], "supported_groups": evidence["item_groups"],
        "columns": list(basis.columns), "right_factor": basis.right.tolist(), "family_ids": selection["family_ids"],
        "explicit_forward_calls": budget.used, "explicit_forward_limit": budget.limit}, saved


def _failure_classification(record, *, arm, target_only_feasible, maximum_norm):
    certified = record.get("status") in {"CERTIFIED_NORM_BOUND_EXCEEDED", "CERTIFIED_RADIUS_SEPARATOR",
        "INFEASIBLE_ZERO_ROW", "INFEASIBLE_OPPOSED_CONSTRAINTS"}
    lower = record.get("certified_norm_lower_bound")
    certified = bool(certified or lower is not None and lower > maximum_norm)
    if certified:
        return "TARGET_PRESERVATION_CONFLICT" if arm == "C" and target_only_feasible else "NORM_BOUND_FAILURE"
    return "NUMERICAL_INCONCLUSIVE"


def _numeric_cell(space, runner, plan, basis, forks, contexts, direction, shared, intended, config, arm, budget, *, target_only_feasible=False):
    wall, cpu = time.perf_counter(), time.process_time()
    pool, sample, selected, cal_pool, cal_sample, cal_gradients, cal_ids, floor = shared
    design, batch = config["probe_design"], config["inference"]["batch_size"]
    tau, memory = design["target_margin"], plan.limits.max_memory_bytes
    gradients = np.stack([value.gradient for value in selected])
    margins = np.array([value.margin for value in selected])
    native_calls = sum(any(f.prompt_id in set(basis.target_ids) for f in context) for context in contexts)
    measurement_calls = native_calls + ceil(len(sample.anchors) / batch)
    cache, passes, allowance, best, matching, certified, calls = {}, [], 0., None, None, False, 0
    joint_fit = None

    def evaluate(amplitude):
        nonlocal calls
        fit = solve_robust(direction, gradients, margins, radius=amplitude, margin_floor=floor, allowance=allowance) if arm == "A" else joint_fit
        if fit.update is None or np.linalg.norm(fit.update) <= 1e-12:
            return {"realized_norm": 0., "feasible": False, "tensor_sha256": None,
                    "reason": "no_bounded_local_witness", "solver": fit.record(),
                    "failure_class": _failure_classification(fit.record(), arm=arm,
                        target_only_feasible=target_only_feasible, maximum_norm=plan.limits.max_update_norm)}
        update = fit.update
        if arm == "C":
            minimum = float(np.linalg.norm(update))
            if amplitude < minimum - 1e-10:
                return {"realized_norm": 0., "feasible": False, "tensor_sha256": None,
                        "reason": "requested_radius_below_minimum_witness", "solver": fit.record()}
            update = update * (amplitude / minimum)
        predicted_target = basis.target_margins + basis.target_gradients @ update
        predicted_protection = margins + gradients @ update
        if np.min(predicted_target) < tau or np.min(predicted_protection) < floor + allowance:
            return {"realized_norm": 0., "feasible": False, "tensor_sha256": None,
                    "reason": "individual_linear_margin_refusal", "solver": fit.record(),
                    "left_factor": update.tolist(), "target_predictions": predicted_target.tolist(),
                    "preservation_predictions": predicted_protection.tolist(),
                    "minimum_predicted_target": float(np.min(predicted_target)),
                    "minimum_predicted_preservation": float(np.min(predicted_protection))}
        preview = space.preview(update)
        eligible = bool(0 < preview["realized_norm"] <= plan.limits.max_update_norm and preview["changed_entries"])
        measured = None
        if eligible:
            digest = preview["tensor_sha256"]
            if digest not in cache:
                budget.reserve(measurement_calls)
                cache[digest] = measure_stored_joint(space, preview, runner, forks, contexts, pool, sample.anchors,
                    target_floor=tau, preserve_floor=floor, batch_size=batch,
                    max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=memory)
                calls += cache[digest]["resources"]["forward_calls"]
            measured = cache[digest]
            eligible = bool(measured["passed"])
        return {key: preview[key] for key in ("realized_norm", "tensor_sha256", "changed_entries", "proposed_norm", "update", "delta", "stored_update_cosine")} | {
            "feasible": eligible, "solver": fit.record(), "native_measurement": measured,
            "failure_class": None if eligible else "LINEAR_FEASIBLE_BF16_FAILURE",
            "left_factor": update.tolist(),
            "target_predictions": predicted_target.tolist(), "preservation_predictions": predicted_protection.tolist(),
            "active_constraint_count": int(np.count_nonzero(predicted_protection - floor - allowance <= 1e-6)),
            "minimum_predicted_margin": float(np.min(predicted_protection)), "required_margin": floor + allowance}

    try:
        for _ in range(2):
            if arm == "C":
                joint_fit = _solve(_system(basis.target_gradients, basis.target_margins, plan, config,
                    protected=gradients, protected_margins=margins, floor=floor + allowance, target_ids=basis.target_ids))
            matching = match_stored_norm(evaluate, intended_norm=intended, maximum_norm=plan.limits.max_update_norm,
                relative_tolerance=design["norm_relative_tolerance"], max_probes=design["norm_search_steps"])
            best = matching.selected
            if matching.record()["status"] != "MATCHED" or best is None:
                passes.append({"norm_match": matching.record(), "allowance_used": allowance})
                break
            preview = space.preview(best["update"])
            budget.reserve(ceil(len(cal_sample.anchors) / batch))
            observed, cal_measurement = space.margins(runner, cal_pool, cal_sample.anchors, preview,
                batch_size=batch, max_memory_bytes=memory)
            calls += cal_measurement["resources"]["forward_calls"]
            observations = tuple(MarginObservation(cal_ids[a.prompt_id], a.prompt_id, a.position,
                cal_gradients[a.key].margin + float(cal_gradients[a.key].gradient @ best["update"]), observed[a.key], group=a.group)
                for a in cal_sample.anchors)
            calibration = calibrate_residuals(observations, quantile=design["adverse_item_quantile"])
            if calibration.allowance is None:
                passes.append({"norm_match": matching.record(), "allowance_used": allowance,
                    "calibration": calibration.record(), "calibration_measurement": cal_measurement,
                    "calibration_sufficient": False, "reason": "insufficient_calibration_items"})
                break
            certified = calibration.allowance <= allowance + 1e-7
            passes.append({"norm_match": matching.record(), "allowance_used": allowance,
                "calibration": calibration.record(), "calibration_measurement": cal_measurement,
                "calibration_sufficient": certified})
            if certified:
                break
            allowance = max(allowance, calibration.allowance)
    except _BudgetExhausted:
        return None, {"norm_match": {"status": "UNMATCHED", "reason": "explicit_forward_budget_exhausted", "selected": None, "attempts": []},
            "calibration_passes": passes, "direction_specific_allowance": allowance, "calibration_sufficient": False,
            "floor": floor, "resources": {**_resources(wall, cpu), "forward_calls": calls, "backward_calls": 0}}
    return best, {"norm_match": matching.record(), "calibration_passes": passes,
        "direction_specific_allowance": allowance, "calibration_sufficient": bool(certified),
        "floor": floor, "resources": {**_resources(wall, cpu), "forward_calls": calls, "backward_calls": 0}}


def _compare(base, adapter, runner, plan, basis, forks, contexts, direction, shared, populations, config, out, forward_budget, *, target_only_feasible):
    contract, design = endpoint(config), config["probe_design"]
    parent_rows, parent = populations["adaptation"]
    protected_rows, protected = populations["preservation-fit"]
    pool, sample, selected, _, _, _, _, floor = shared
    batch = contract.batch_size
    space = StoredAffineSpace.open(adapter, runner, plan, basis)
    budget, seen, owners, arms, saved, deltas = CandidateEvaluationBudget(contract), set(), {}, {}, {}, {}

    def evaluate():
        measured = measure_endpoint(runner, parent_rows, contract, candidate_budget=budget)
        preserved = measure_endpoint(runner, protected_rows, contract)
        guard = compare(parent, measured, min_comparison_gain=1, min_repeatable_corrections=1)
        preserve_guard = compare_endpoint(protected, preserved, contract)
        rate = sum(row["valid"] for row in measured["rows"]) / len(measured["rows"])
        guard.update(passed=bool(guard["passed"] and not preserve_guard["prohibited_regressions"]
            and not preserve_guard["lost_valid_presentations"] and rate >= config["endpoint"]["min_valid_rate"]),
            target_item_gain=guard["target_gain"], prohibited_regressions=guard["regressions"],
            lost_valid_presentations=guard["lost_valid"], strict_measured_floor_passed=True,
            individual_target_floor_passed=True)
        return {"measurement": measured, "guard": guard, "preservation_fit_measurement": preserved,
                "preservation_fit_guard": preserve_guard}

    native_calls = sum(any(f.prompt_id in set(basis.target_ids) for f in context) for context in contexts)
    for index, intended in enumerate(design["target_norms"], 1):
        for arm in ("A", "C"):
            name = f"{arm}{index}"
            best, numerical = _numeric_cell(space, runner, plan, basis, forks, contexts, direction, shared,
                                             intended, config, arm, forward_budget, target_only_feasible=target_only_feasible)
            record = {"arm": name, "intended_norm": intended, "plan": plan.record(), **numerical,
                "fit_comparison_ids": list(dict.fromkeys(basis.target_items)), "fork_indices": list(range(len(forks))),
                "fork_prompt_ids": [fork.prompt_id for fork in forks],
                "probe": {"reason": "unmatched_or_uncertified_individual_fork_candidate"}}
            if best is not None:
                record.update(rounded_delta=best["delta"].tolist(), left_factor=best["update"].tolist(),
                    right_factor=basis.right.tolist(), columns=list(basis.columns),
                    target_predictions=best["target_predictions"], native_measurement=best.get("native_measurement"))
                deltas[name] = best["delta"]
            if best is not None and numerical["norm_match"]["status"] == "MATCHED" and numerical["calibration_sufficient"]:
                preview = space.preview(best["update"])
                geometry = _margins_record(basis, selected, best["update"], intended, floor, numerical["direction_specific_allowance"])
                record["geometry"] = geometry
                try:
                    forward_budget.reserve(native_calls + ceil(len(sample.anchors) / batch))
                    patch, probe = probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, sample.anchors,
                        target_floor=design["target_margin"], preserve_floor=floor, batch_size=batch, seen=seen,
                        evaluate_candidate=evaluate, baseline_endpoint=parent)
                except _BudgetExhausted:
                    patch, probe = None, {"reason": "explicit_forward_budget_exhausted_before_generation"}
                record["probe"] = probe
                digest = probe.get("candidate_tensor_sha256")
                if digest in owners:
                    record["duplicate_of"] = owners[digest]
                elif probe.get("endpoint"):
                    owners[digest] = name
                    protected_native = probe["native_measurement"]["protected"]["rows"]
                    active = [{"prompt_id": anchor.prompt_id, "position": anchor.position, "group": anchor.group,
                        "parent_margin": value.margin, "predicted_margin": prediction["predicted_margin"],
                        "measured_margin": row["margin"], "decision_preserved": row["replay_argmax_token_id"] == anchor.token_id,
                        "measured_slack": row["margin"] - floor}
                        for anchor, value, prediction, row in zip(sample.anchors, selected,
                            geometry["preservation_predictions"], protected_native, strict=True)]
                    record["active_measurements"] = active
                    if patch is not None and probe["endpoint"]["guard"]["passed"]:
                        candidate = _materialize(base, out, name, plan, patch, probe)
                        saved[name] = {key: value for key, value in candidate.items() if key != "checkpoint"}
                        saved[name]["receipt_sha256"] = file_hash(candidate["checkpoint"] / ".weightbridge_receipt.json")
            arms[name] = record
            json_dump_atomic(out / f"arm-{name}.json", record)
            print(f"{name}: {numerical['norm_match']['status']}, calibrated {numerical['calibration_sufficient']}, "
                  f"{record['probe']['reason']}", flush=True)
    matching = {}
    for index, intended in enumerate(design["target_norms"], 1):
        states = {name: {"realized_norm": row["norm_match"]["selected"]["realized_norm"],
                        "tensor_sha256": row["norm_match"]["selected"]["tensor_sha256"]}
            for name, row in arms.items() if name.endswith(str(index)) and row["norm_match"].get("selected")}
        matching[str(index)] = compare_realized_norms(states, intended_norm=intended,
            relative_tolerance=design["norm_relative_tolerance"]) if states else None
    json_dump_atomic(out / "stored-pairwise.json", {"norm_matching": matching,
                    "actual_delta_cosines": stored_update_cosines(deltas) if deltas else {}})
    return {"reason": "joint_feasibility_complete", "unique_complete_evaluations": budget.used,
            "practice_candidates": sorted(saved)}, saved


def run(base, material, out):
    from experiments.phase_iiio.source_manifest import source_hashes

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
    result = {"schema": "weightbridge.phase_iiio.fit.v1", "sources": sources, "checkpoint_sha256": expected,
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
