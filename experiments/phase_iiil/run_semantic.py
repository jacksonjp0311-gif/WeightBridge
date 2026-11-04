"""Fit the frozen shared-right semantic comparison on development data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from shutil import copyfile
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan
from experiments.phase_iiij.run_preservation import _preservation_group
from experiments.phase_iiik.run_jspace import _active_measurements
from experiments.phase_iiil.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.fork_update import AffineUpdate, prepare_probe_reference, probe_update
from weightbridge.adaptation.generative_fork import diagnose_fork, next_token_logits, with_fork_logits
from weightbridge.adaptation.robust_jspace import (
    calibrate_margin_floor, calibrate_residuals, estimate_window, select_preservation,
)
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
REPAIR = Path(__file__).with_name("replay_repair.json")
FIT_SPLITS = ("adaptation", "preservation-fit", "calibration")
GROUP_ORDER = ("close-short", "close-long", "far-short", "far-long")


def choose_forks(forks, rows, *, required_wordings, cap):
    """Keep matched wordings together and balance groups across distinct truth-flip families."""
    by_order, by_comparison = defaultdict(list), defaultdict(list)
    lookup = {row["id"]: row for row in rows}
    for fork in forks:
        by_order[fork.question_id].append(fork)
    for order in by_order.values():
        if (len(order) == len(required_wordings)
                and {lookup[fork.prompt_id]["wording"] for fork in order} == set(required_wordings)
                and all(fork.status == "semantic" and fork.replay_status == "matched" for fork in order)):
            by_comparison[lookup[order[0].prompt_id]["comparison_id"]].extend(order)
    groups = defaultdict(list)
    for comparison, members in by_comparison.items():
        row = lookup[members[0].prompt_id]
        precision = {"shorter": "short", "longer": "long"}[row["truth_precision"]]
        group = f"{row['tier'].removeprefix('negative-')}-{precision}"
        groups[group].append((comparison, row["pair_id"], members))
    for candidates in groups.values():
        candidates.sort(key=lambda item: (-len(item[2]), item[0]))
    picked, families = [], set()
    while len(picked) < cap:
        start = len(picked)
        for group in GROUP_ORDER:
            if len(picked) == cap:
                break
            available = next((entry for entry in groups[group] if entry[1] not in families), None)
            if available:
                comparison, family, members = available
                picked.append((comparison, group, members))
                families.add(family)
        if len(picked) == start:
            break
    return picked


def diagnose(runner, parent, contract, config):
    import torch
    from experiments.phase_iiil.semantic_diagnostics import describe_fork_tokens

    wall, cpu = time.perf_counter(), time.process_time()
    forks = [diagnose_fork(runner, row, contract, checkpoint_sha256=config["checkpoint_sha256"])
             for row in parent["rows"] if row["group"] == "target" and row["valid"] and not row["correct"]]
    semantic = [fork for fork in forks if fork.status == "semantic"]
    by_id = {}
    calls = 0
    with torch.inference_mode():
        for index in range(0, len(semantic), contract.batch_size):
            batch = semantic[index:index + contract.batch_size]
            calls += 1
            for fork, logits in zip(batch, next_token_logits(runner, [fork.prefix_ids for fork in batch]), strict=True):
                by_id[fork.prompt_id] = with_fork_logits(fork, logits, runner.tokenizer)
    forks = [by_id.get(fork.prompt_id, fork) for fork in forks]
    picked = choose_forks(forks, parent["rows"], required_wordings=contract.required_wordings,
                          cap=config["fit_limits"]["max_repeatable_error_items"])
    eos = runner.tokenizer.eos_token_id

    def category(token, surface):
        if token == eos:
            return "eos"
        if "\n" in surface:
            return "newline"
        if any(char in "0123456789" for char in surface):
            return "digit"
        return "other"

    evidence, rows_by_id = [], {row["id"]: row for row in parent["rows"]}
    for fork in forks:
        accepted = [runner.tokenizer.decode([token], skip_special_tokens=False) for token in fork.accepted_token_ids]
        actual = (runner.tokenizer.decode([fork.actual_token_id], skip_special_tokens=False)
                  if fork.actual_token_id is not None else None)
        evidence.append({**fork.to_dict(), "accepted_surfaces": accepted, "actual_surface": actual,
                         "accepted_categories": [category(t, s) for t, s in zip(fork.accepted_token_ids, accepted)],
                         "actual_category": category(fork.actual_token_id, actual) if actual is not None else None,
                         "fork_tokens": describe_fork_tokens(fork, runner.tokenizer, rows_by_id[fork.prompt_id])})
    return picked, {"forks": evidence, "chosen": [{"comparison_id": item, "group": group,
                    "prompts": [fork.prompt_id for fork in members]} for item, group, members in picked],
                    "resources": {**_resources(wall, cpu), "forward_calls": calls, "backward_calls": 0}}


def _margins_record(basis, sensitivities, factor, radius, floor, allowance):
    norm = float(np.linalg.norm(factor))
    unit = factor / norm
    gradients = np.stack([row.gradient for row in sensitivities])
    margins = np.array([row.margin for row in sensitivities])
    slopes = gradients @ unit
    return {
        "unrounded_norm": norm,
        "local_feasibility": estimate_window(basis.target_margins, basis.target_gradients @ unit,
            margins, slopes, radius=radius, margin_floor=floor, allowance=allowance, target_items=basis.target_items),
        "preservation_predictions": [{"prompt_id": row.anchor.prompt_id, "position": row.anchor.position,
                                      "parent_margin": row.margin, "slope": float(slope),
                                      "predicted_margin": float(row.margin + norm * slope)}
                                     for row, slope in zip(sensitivities, slopes, strict=True)],
    }


def _replay_stable_sample(adapter, runner, plan, basis, pool, *, batch, rounds, sampling):
    from weightbridge.adaptation.counterexample_update import fit_preservation

    verified, rejected, attempts = {}, set(), []
    cells = {}
    for _ in range(rounds):
        eligible = [anchor for anchor in pool.eligible_anchors
                    if anchor.key not in rejected
                    and cells.get((anchor.group, anchor.wording), anchor.key) == anchor.key]
        sample = select_preservation(eligible, **sampling)
        if not sample.complete:
            return sample, verified, attempts
        needed = [anchor for anchor in sample.anchors if anchor.key not in verified]
        found, evidence = fit_preservation(adapter, runner, plan, basis, pool, needed, batch_size=batch)
        attempts.append({"sample": sample.record(), "replay": evidence})
        verified.update(found)
        for anchor in needed:
            if anchor.key in found:
                cells[(anchor.group, anchor.wording)] = anchor.key
            else:
                rejected.add(anchor.key)
        if all(anchor.key in verified for anchor in sample.anchors):
            break
    return sample, verified, attempts


def _shared_constraints(adapter, runner, plan, basis, gradients, labels, populations, config, out):
    from weightbridge.adaptation.counterexample_update import measure_parent_decisions
    from weightbridge.adaptation.jspace_update import calibrate_direction_residuals, measure_directional_secant

    design, contract = config["probe_design"], endpoint(config)
    batch = contract.batch_size
    normalized = gradients / np.maximum(np.linalg.norm(gradients, axis=1, keepdims=True), 1e-12)
    group_means = np.stack([normalized[np.asarray(labels) == group].mean(axis=0) for group in dict.fromkeys(labels)])
    equal = group_means.mean(axis=0)
    if np.linalg.norm(equal) <= 1e-12:
        return "zero_equal_group_preview", None
    previews = {"historical": basis, "equal_group": replace(basis, preferred_left=equal / np.linalg.norm(equal))}
    settings = {"group_for": _preservation_group, "batch_size": batch,
                "max_memory_bytes": plan.limits.max_memory_bytes}
    rows = populations["adaptation"][1]["rows"] + populations["preservation-fit"][1]["rows"]
    pool, pool_record = measure_parent_decisions(runner, {"rows": rows}, contract, **settings)
    json_dump_atomic(out / "preservation-decisions.json", pool_record)
    all_slopes, secants = {}, {}
    for name, preview in previews.items():
        all_slopes[name], secants[name] = measure_directional_secant(adapter, runner, plan, preview, pool,
            radius=design["directional_secant_radius"], batch_size=batch)
    json_dump_atomic(out / "secants.json", secants)
    if any(not slope for slope in all_slopes.values()):
        return "preservation_secant_unavailable", None
    slopes = {anchor.key: min(row[anchor.key] for row in all_slopes.values()) for anchor in pool.eligible_anchors}
    sampling = {"required_groups": design["preservation_groups"], "required_wordings": contract.required_wordings}
    rounds = read_json(REPAIR)["maximum_sampling_rounds"]
    sample, active, active_attempts = _replay_stable_sample(adapter, runner, plan, basis, pool, batch=batch,
        rounds=rounds, sampling={**sampling, "budget": design["anchors"], "mode": "stratified_directional",
                                 "slopes": slopes, "radius": design["directional_secant_radius"]})
    json_dump_atomic(out / "sampling.json", sample.record())
    json_dump_atomic(out / "preservation-sensitivities.json", {"attempts": active_attempts})
    if not sample.complete:
        return "incomplete_preservation_coverage", None
    if any(anchor.key not in active for anchor in sample.anchors):
        return "incomplete_preservation_vjps", None
    selected = tuple(active[anchor.key] for anchor in sample.anchors)

    cal_pool, cal_pool_record = measure_parent_decisions(runner, populations["calibration"][1], contract, **settings)
    cal_sample, cal_vjps, cal_attempts = _replay_stable_sample(adapter, runner, plan, basis, cal_pool, batch=batch,
        rounds=rounds, sampling={**sampling, "budget": design["calibration_anchors"], "mode": "stratified"})
    json_dump_atomic(out / "calibration-decisions.json", {"pool": cal_pool_record, "sample": cal_sample.record()})
    json_dump_atomic(out / "calibration-sensitivities.json", {"attempts": cal_attempts})
    if not cal_sample.complete:
        return "incomplete_calibration_coverage", None
    if any(anchor.key not in cal_vjps for anchor in cal_sample.anchors):
        return "incomplete_calibration_vjps", None
    floor_record = calibrate_margin_floor([anchor.margin for anchor in cal_sample.anchors],
                                           design["positive_floor_grid"])
    floor = floor_record["selected_floor"]
    residuals, allowances = {}, []
    for name, preview in previews.items():
        for radius in design["doses"]:
            observations, _, evidence = calibrate_direction_residuals(adapter, runner, plan, preview, cal_pool,
                cal_sample.anchors, (), cal_vjps, radius=radius, batch_size=batch)
            summary = calibrate_residuals(observations, quantile=design["adverse_item_quantile"]) if observations else None
            residuals[f"{name}-{radius}"] = {"evidence": evidence, "summary": summary.record() if summary else None}
            if summary:
                allowances.append(summary.allowance)
    json_dump_atomic(out / "calibration.json", {"floor": floor_record, "probes": residuals,
                                               "common_allowance": max(allowances, default=None)})
    if floor is None or len(allowances) != len(residuals):
        return "no_calibrated_positive_slack", None
    return None, (sample, pool, selected, floor, max(allowances))


def _fit(base, adapter, runner, capabilities, populations, config, out):
    from weightbridge.adaptation.semantic_consensus import gradient_agreement
    from weightbridge.adaptation.semantic_update import fit_semantic_basis

    contract = endpoint(config)
    runner = endpoint_runner(runner, contract)
    picked, diagnosis = diagnose(runner, populations["adaptation"][1], contract, config)
    json_dump_atomic(out / "fork-diagnosis.json", diagnosis)
    if len({group for _, group, _ in picked}) < config["fit_limits"]["minimum_supported_groups"]:
        return {"reason": "insufficient_distinct_replay_matched_error_groups"}, {}
    plan = _plan(config, capabilities, config["probe_design"]["fixed_layer"])
    forks = tuple(fork for _, _, members in picked for fork in members)
    groups = {item: group for item, group, _ in picked}
    item_by_prompt = {fork.prompt_id: item for item, _, members in picked for fork in members}
    basis, gradients, fit_record = fit_semantic_basis(adapter, runner, plan, forks, groups,
        batch_size=contract.batch_size, max_forks=config["fit_limits"]["max_forks"], item_by_prompt=item_by_prompt)
    json_dump_atomic(out / "basis.json", fit_record)
    if basis is None:
        return {"reason": "no_shared_rank_one_basis"}, {}
    ids, labels = fit_record["item_ids"], fit_record["item_groups"]
    json_dump_atomic(out / "projected-agreement.json", gradient_agreement(ids, labels, gradients))
    reason, shared = _shared_constraints(adapter, runner, plan, basis, gradients, labels, populations, config, out)
    detail = {"fit_items": list(ids), "supported_groups": sorted(set(labels)), "columns": list(basis.columns),
              "shared_right_factor": basis.right.tolist()}
    if shared is None:
        return {**detail, "reason": reason}, {}
    arms, saved, count = _compare(base, adapter, runner, plan, populations, config, contract, out,
                                 basis, forks, ids, labels, gradients, shared)
    distinct = {str(i): bool(arms[f"A{i}"]["probe"].get("endpoint") and arms[f"B{i}"]["probe"].get("endpoint"))
                for i in range(1, len(config["probe_design"]["doses"]) + 1)}
    return {**detail, "reason": "comparison_complete", "unique_complete_evaluations": count,
            "floor": shared[3], "allowance": shared[4], "matched_pairs_distinct": distinct}, saved


def _compare(base, adapter, runner, plan, populations, config, contract, out, basis, forks,
             ids, labels, gradients, shared):
    from experiments.phase_iiil.semantic_diagnostics import compare_semantic
    from weightbridge.adaptation.semantic_consensus import solve_consensus

    rows, parent = populations["adaptation"]
    fit_rows, fit_parent = populations["preservation-fit"]
    batch, memory = contract.batch_size, plan.limits.max_memory_bytes
    sample, pool, selected, floor, allowance = shared
    prompts = tuple(dict.fromkeys(anchor.prompt_id for anchor in sample.anchors))
    trajectories = tuple(pool.trajectories[prompt] for prompt in prompts)
    ref = prepare_probe_reference(adapter, runner, forks, trajectories, batch_size=batch, max_memory_bytes=memory)
    json_dump_atomic(out / "probe-reference.json", {"resources": ref.resources, "prompt_ids": prompts})
    budget, seen, owners, saved, arms = CandidateEvaluationBudget(contract), set(), {}, {}, {}

    def evaluate():
        measured = measure_endpoint(runner, rows, contract, candidate_budget=budget)
        protected = measure_endpoint(runner, fit_rows, contract)
        guard = compare_semantic(parent, measured, min_comparison_gain=1, min_repeatable_corrections=1)
        fit_guard = compare_endpoint(fit_parent, protected, contract)
        guard["passed"] = bool(guard["passed"] and not fit_guard["prohibited_regressions"]
                               and not fit_guard["lost_valid_presentations"])
        guard.update(target_item_gain=guard["target_gain"], prohibited_regressions=guard["regressions"],
                     lost_valid_presentations=guard["lost_valid"])
        return {"measurement": measured, "guard": guard, "preservation_fit_measurement": protected,
                "preservation_fit_guard": fit_guard}

    for index, radius in enumerate(config["probe_design"]["doses"], 1):
        solve = solve_consensus(ids, labels, gradients, np.stack([row.gradient for row in selected]),
                                [row.margin for row in selected], radius=radius, floor=floor + allowance,
                                historical_direction=basis.preferred_left, required_groups=GROUP_ORDER)
        json_dump_atomic(out / f"geometry-{index}.json", solve.record())
        for label, fit in (("A", solve.historical), ("B", solve.maximin)):
            name = f"{label}{index}"
            record = {"arm": name, "nominal_radius": radius, "floor": floor, "allowance": allowance,
                      "plan": plan.record(), "probe": {"reason": "no_positive_feasible_direction"}}
            factor = fit.update
            if factor is not None and np.linalg.norm(factor) > 1e-12:
                update = AffineUpdate(basis.columns, factor[:, None], basis.right[None, :], "inequality")
                geometry = _margins_record(basis, selected, factor, radius, floor, allowance)
                patch, probe = probe_update(adapter, runner, plan, update, radius=None, seen=seen,
                    evaluate_candidate=evaluate, forks=forks, trajectories=trajectories, baseline_endpoint=parent,
                    batch_size=batch, reference=ref)
                record.update(probe=probe, geometry=geometry)
                digest = probe.get("candidate_tensor_sha256")
                if digest in owners:
                    record["duplicate_of"] = owners[digest]
                elif digest and probe.get("endpoint"):
                    owners[digest] = name
                    active_rows = _active_measurements(sample.anchors, probe, prompts, geometry,
                                                      floor=floor, allowance=allowance)
                    record["active_measurements"] = active_rows
                    guard = probe["endpoint"]["guard"]
                    guard["semantic_behavior_passed"] = guard["passed"]
                    guard["strict_measured_floor_passed"] = bool(active_rows and all(
                        row["decision_preserved"] and row["measured_margin"] > floor for row in active_rows))
                    guard["passed"] = bool(guard["passed"] and guard["strict_measured_floor_passed"])
                    if patch is not None and guard["passed"]:
                        candidate = _materialize(base, out, name, plan, patch, probe)
                        saved[name] = {key: value for key, value in candidate.items() if key != "checkpoint"}
                        saved[name]["receipt_sha256"] = file_hash(candidate["checkpoint"] / ".weightbridge_receipt.json")
            arms[name] = record
            json_dump_atomic(out / f"arm-{name}.json", record)
            print(f"{name}: {record['probe']['reason']}; "
                  f"{record.get('duplicate_of') or record['probe'].get('endpoint', {}).get('guard', {}).get('passed')}",
                  flush=True)
    return arms, saved, budget.used


def _initial_parent_receipts(directory, material, expected):
    path = directory / "fit-report.json"
    if file_hash(path) != read_json(REPAIR)["first_fit_report_sha256"]:
        raise ValueError("Initial fit report differs from the committed repair input")
    report = read_json(path)
    if (report["checkpoint_sha256"] != expected or report["hidden_opened"]
            or report["optimization_splits"] != list(FIT_SPLITS) or report["candidates"]
            or report["manifest_sha256"] != file_hash(material / "manifest.json")):
        raise ValueError("Initial fit report does not match the frozen development scope")
    result = {}
    for split in FIT_SPLITS:
        name = f"parent-{split}.json"
        if file_hash(directory / name) != report["artifacts"][name]:
            raise ValueError("Initial parent measurement differs from its committed digest")
        result[split] = directory / name
    return result


def run(base, material, out, *, reuse_initial=None):
    from experiments.phase_iiil.semantic_diagnostics import summarize
    from experiments.phase_iiil.source_manifest import source_hashes
    from weightbridge.adaptation.semantic_update import score_teacher_forced_options

    wall, cpu = time.perf_counter(), time.process_time()
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Run in an isolated host directory and archive only public evidence")
    config, manifest, source = read_json(PROTOCOL), read_manifest(material), source_hashes()
    expected = config["checkpoint_sha256"]
    prior = _initial_parent_receipts(reuse_initial, material, expected) if reuse_initial else {}
    out.mkdir(parents=True, exist_ok=False)
    contract, populations, candidates = endpoint(config), {}, {}
    with _open(base, config, expected, gradient_probe=True) as (adapter, runner, capabilities):
        for split in FIT_SPLITS:
            frozen = load_split(material, split, manifest)
            receipt = out / f"parent-{split}.json"
            if split in prior:
                copyfile(prior[split], receipt)
                measured = read_json(receipt)["measurement"]
            else:
                start, start_cpu = time.perf_counter(), time.process_time()
                measured = measure_endpoint(runner, frozen, contract)
                json_dump_atomic(receipt, {"measurement": measured,
                    "summary": summarize(measured), "resources": _resources(start, start_cpu)})
            populations[split] = frozen, measured
        parent = populations["adaptation"][1]
        qualification = qualify(parent, config)
        preferences, scores = score_teacher_forced_options(runner, parent["rows"], contract,
            max_forward_calls=config["intent"]["limits"]["max_forward_calls"],
            max_memory_bytes=config["intent"]["limits"]["max_memory_bytes"])
        json_dump_atomic(out / "teacher-forced-parent.json", scores)
        json_dump_atomic(out / "parent-diagnostics.json", summarize(parent, teacher_preferences=preferences))
        if qualification["disposition"] == "PASS":
            detail, candidates = _fit(base, adapter, runner, capabilities, populations, config, out)
        else:
            detail = {"reason": "parent_measurement_did_not_qualify"}
    if source_hashes() != source or load_checkpoint(base).checkpoint_sha256() != expected:
        raise ValueError("Fitting source or original checkpoint changed")
    artifacts = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))
                 if "checkpoints" not in path.relative_to(out).parts}
    report = {"schema": "weightbridge.phase_iiil.fit.v1", "checkpoint_sha256": expected,
              "sources": source, "manifest_sha256": file_hash(material / "manifest.json"),
              "qualification": qualification, "detail": detail, "candidates": candidates, "artifacts": artifacts,
              "reused_parent_receipts": sorted(prior),
              "optimization_splits": list(FIT_SPLITS), "hidden_opened": False, "resources": _resources(wall, cpu)}
    json_dump_atomic(out / "fit-report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--reuse-initial", type=Path)
    args = parser.parse_args()
    print(run(args.base, args.material, args.out, reuse_initial=args.reuse_initial)["detail"], flush=True)
