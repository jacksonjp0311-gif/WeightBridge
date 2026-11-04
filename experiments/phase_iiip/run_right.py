"""Track A: fixed versus optimized right factor in one authorized affine block."""
from __future__ import annotations

import argparse
from dataclasses import replace
from math import ceil
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan
from experiments.phase_iiil.run_semantic import diagnose
from experiments.phase_iiil.semantic_diagnostics import summarize
from experiments.phase_iiim.run_factorial import _prepare
from experiments.phase_iiin.run_paired import _reference_geometry, _restore_fork
from experiments.phase_iiio.run_joint import _select_families, _shared_forward_calls, _ForwardBudget, _BudgetExhausted
from experiments.phase_iiip.semantic_roster import PROTOCOL, read_manifest, load_split
from experiments.phase_iiip.independent_bounds import verify_relaxation_bound
from weightbridge.adaptation.behavior_endpoint import CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint
from weightbridge.adaptation.counterexample_update import ForkBasis
from weightbridge.adaptation.factorial_geometry import match_stored_norm, compare_realized_norms, stored_update_cosines
from weightbridge.adaptation.factorial_probe import StoredAffineSpace
from weightbridge.adaptation.fork_update import _resources as _model_resources
from weightbridge.adaptation.joint_feasibility import solve_minimum_norm
from weightbridge.adaptation.joint_update import measure_native_forks, measure_stored_joint, probe_joint_generation
from weightbridge.adaptation.right_factor_geometry import solve_block_relaxation, search_right_factor, RightFactorSearch, canonical_factors
from weightbridge.adaptation.robust_jspace import MarginObservation, calibrate_residuals
from weightbridge.adaptation.selected_block import collect_target_blocks, collect_preservation_blocks
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from experiments.phase_iiip.semantic_metrics import compare

ROOT = Path(__file__).resolve().parents[2]
FIT_SPLITS = ("adaptation", "preservation-fit", "calibration")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serial(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(child) for child in value]
    return value


def _fixed(blocks, rhs, right, plan, config, ids):
    return solve_minimum_norm(blocks @ right, rhs, radius=plan.limits.max_update_norm,
        max_sweeps=config["probe_design"]["joint_solver"]["max_sweeps"],
        tolerance=config["probe_design"]["joint_solver"]["tolerance"], row_ids=ids)


def _array_receipt(blocks):
    import hashlib
    values = np.ascontiguousarray(blocks, dtype="<f8")
    return {"shape": list(values.shape), "dtype": "float64", "array_sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def _relax_and_search(blocks, rhs, right, plan, config, ids, *, target_barrier=False):
    settings = config["probe_design"]["joint_solver"]
    relaxation = solve_block_relaxation(blocks, rhs, radius=plan.limits.max_update_norm,
        max_sweeps=settings["max_sweeps"], tolerance=settings["tolerance"], row_ids=ids)
    independent = verify_relaxation_bound(blocks, rhs, relaxation.record())
    fixed = _fixed(blocks, rhs, right, plan, config, ids)
    search = (RightFactorSearch(None, None, {"status": "TARGET_BLOCK_BOUND_REFUSAL", "inner_calls": 0,
        "reason": "Previously checked target-only bound excludes the entire protected rank-one system",
        "resources": {"forward_calls": 0, "backward_calls": 0}}) if target_barrier else
        search_right_factor(blocks, rhs, right, radius=plan.limits.max_update_norm,
        settings=config["probe_design"]["outer_search"]["settings"], row_ids=ids, relaxation=relaxation))
    return fixed, search, {"relaxation": relaxation.record(), "fixed_right": fixed.record(),
        "optimized_right": search.record(), "independent_bound_check": independent,
        "scope": "Unrounded local system only; relaxed matrix is diagnostic and never a patch."}


def _fit(base, adapter, runner, capabilities, populations, config, out):
    design, contract = config["probe_design"], endpoint(config)
    runner = endpoint_runner(runner, contract)
    _, diagnosis = diagnose(runner, populations["adaptation"][1], contract, config)
    json_dump_atomic(out / "fork-diagnosis.json", diagnosis)
    names, eligible, groups, _, selection = _select_families(populations["adaptation"][1], diagnosis, contract)
    json_dump_atomic(out / "supervision-selection.json", selection)
    if selection["status"] != "READY":
        return {"reason": selection["reason"], "unique_complete_evaluations": 0}, {}
    forks = tuple(fork for item in names for fork in eligible[item])
    diagnostic = tuple(_restore_fork(row) for row in diagnosis["forks"] if row["status"] == "semantic")
    contexts = tuple(diagnostic[start:start + contract.batch_size] for start in range(0, len(diagnostic), contract.batch_size))
    plan = _plan(config, capabilities, design["fixed_layer"])
    columns, right, reference = _reference_geometry(config, plan)
    json_dump_atomic(out / "reference-geometry.json", reference)
    protected_count = sum(row["correct"] for split in ("adaptation", "preservation-fit") for row in populations[split][1]["rows"])
    if len(forks) > 16 or protected_count > 128:
        return {"reason": "prospective_protection_or_target_budget", "unique_complete_evaluations": 0}, {}
    lookup = {row["id"]: row for row in populations["adaptation"][1]["rows"]}
    metadata = {fork.prompt_id: {key: lookup[fork.prompt_id][key] for key in
        ("pair_id", "comparison_id", "order", "wording", "tier", "truth_precision")} for fork in forks}
    target_blocks, target_evidence = collect_target_blocks(adapter, runner, plan, forks, columns=columns,
        replay_batches=contexts, metadata_by_prompt=metadata, batch_size=contract.batch_size)
    json_dump_atomic(out / "target-block-metadata.json", target_evidence)
    with (out / "target-blocks.npz").open("xb") as stream:
        np.savez(stream, targets=target_blocks)
    target_margins = np.asarray([row["parent_margin"] for row in target_evidence["rows"]])
    projected = target_blocks @ right
    item_ids = tuple(lookup[fork.prompt_id]["comparison_id"] for fork in forks)
    comparison_rows = np.stack([projected[[i for i, item in enumerate(item_ids) if item == name]].mean(axis=0) for name in names])
    preferred = comparison_rows.mean(axis=0)
    preferred /= max(np.linalg.norm(preferred), 1e-30)
    basis = ForkBasis(plan.checkpoint_sha256, plan.region.tensor_name, columns, preferred, right,
        tuple(fork.prompt_id for fork in forks), item_ids, target_margins, projected)
    evidence = {"schema": "weightbridge.phase_iiip.basis.v1", "columns": list(columns), "right_factor": right.tolist(),
        "item_ids": list(names), "item_groups": [groups[name] for name in names], "reference_geometry": reference,
        "forks": [{"prompt_id": fork.prompt_id, "comparison_id": item, "parent_margin": float(margin),
            "accepted_token_ids": list(fork.accepted_token_ids), "actual_token_id": fork.actual_token_id,
            "projected_margin_gradient": row.tolist(), "metadata": metadata[fork.prompt_id]}
            for fork, item, margin, row in zip(forks, item_ids, target_margins, projected, strict=True)]}
    json_dump_atomic(out / "basis.json", evidence)
    native = measure_native_forks(runner, forks, contexts, max_forward_calls=plan.limits.max_forward_calls,
        max_memory_bytes=plan.limits.max_memory_bytes)
    json_dump_atomic(out / "native-parent-targets.json", native)
    if any(row["measured_margin"] != margin for row, margin in zip(native["rows"], target_margins, strict=True)):
        raise ValueError("Native full-block derivative intercept differs from its original diagnostic context")
    tau = design["target_margin"]
    target_rhs = tau - target_margins
    # The selected-block relaxation is evaluated BEFORE any orientation search.
    target_relax = solve_block_relaxation(target_blocks, target_rhs, radius=plan.limits.max_update_norm,
        max_sweeps=design["joint_solver"]["max_sweeps"], tolerance=design["joint_solver"]["tolerance"], row_ids=basis.target_ids)
    target_bound_check = verify_relaxation_bound(target_blocks, target_rhs, target_relax.record())
    json_dump_atomic(out / "target-relaxation.json", target_relax.record())
    fixed_target = _fixed(target_blocks, target_rhs, right, plan, config, basis.target_ids)
    directions = {"A": preferred}
    if fixed_target.minimum_norm_update is not None and np.linalg.norm(fixed_target.minimum_norm_update):
        directions["target_preview"] = fixed_target.minimum_norm_update
    shared, detail = _prepare(adapter, runner, plan, basis, directions, populations, config, out)
    if shared is None:
        return {**detail, "unique_complete_evaluations": 0}, {}
    pool, sample, selected, cal_pool, cal_sample, _, cal_ids, floor = shared
    protected_blocks, protected_evidence = collect_preservation_blocks(adapter, runner, plan, pool, sample.anchors,
        columns=columns, batch_size=contract.batch_size)
    cal_blocks, cal_evidence = collect_preservation_blocks(adapter, runner, plan, cal_pool, cal_sample.anchors,
        columns=columns, batch_size=contract.batch_size)
    json_dump_atomic(out / "preservation-block-metadata.json", protected_evidence)
    json_dump_atomic(out / "calibration-block-metadata.json", cal_evidence)
    margins = np.asarray([row["parent_margin"] for row in protected_evidence["rows"]])
    cal_margins = np.asarray([row["parent_margin"] for row in cal_evidence["rows"]])
    protected_ids = [f"{anchor.prompt_id}:{anchor.position}" for anchor in sample.anchors]
    # No higher-rank relaxed witness can enter StoredAffineSpace or the patcher.
    with (out / "derivative-blocks.npz").open("xb") as stream:
        np.savez(stream, targets=target_blocks, preservation=protected_blocks, calibration=cal_blocks)
    json_dump_atomic(out / "selected-blocks.json", {"schema": "weightbridge.phase_iiip.blocks.v1",
        "file_sha256": file_hash(out / "derivative-blocks.npz"), "source_checkpoint_sha256": plan.checkpoint_sha256,
        "columns": list(columns), "tensor_name": plan.region.tensor_name,
        "arrays": {name: _array_receipt(value) for name, value in (("targets", target_blocks), ("preservation", protected_blocks), ("calibration", cal_blocks))}})
    all_blocks = np.concatenate((target_blocks, protected_blocks))
    rhs = np.concatenate((target_rhs, floor - margins))
    ids = [*basis.target_ids, *protected_ids]
    target_barrier = (target_relax.record().get("status") == "INFEASIBLE_ZERO_BLOCK" or
        (target_relax.record().get("certified_norm_lower_bound") or 0.) > plan.limits.max_update_norm + design["joint_solver"]["tolerance"])
    fixed, optimized, joint = _relax_and_search(all_blocks, rhs, right, plan, config, ids, target_barrier=target_barrier)
    target_search = search_right_factor(target_blocks, target_rhs, right, radius=plan.limits.max_update_norm,
        settings=design["outer_search"]["settings"], row_ids=basis.target_ids, relaxation=target_relax)
    json_dump_atomic(out / "feasibility.json", {"schema": "weightbridge.phase_iiip.feasibility.v1",
        "target_only": {"relaxation": target_relax.record(), "fixed_right": fixed_target.record(),
            "optimized_right": target_search.record(), "independent_bound_check": target_bound_check},
        "with_preservation": joint, "target_margins": target_margins.tolist(), "target_rhs": target_rhs.tolist(),
        "preservation_margins": margins.tolist(), "preservation_floor": floor, "row_ids": ids,
        "all_rhs": rhs.tolist(), "source_block_sha256": file_hash(out / "derivative-blocks.npz")})
    used = sum(rec["resources"]["forward_calls"] for rec in (diagnosis, target_evidence, protected_evidence, cal_evidence, native)) + _shared_forward_calls(out)
    backwards = sum(rec["resources"].get("backward_calls", 0) for rec in (target_evidence, protected_evidence, cal_evidence))
    for filename in ("preservation-sensitivities.json", "calibration-sensitivities.json"):
        backwards += sum(attempt["replay"]["resources"].get("backward_calls", 0)
            for attempt in read_json(out / filename)["attempts"])
    budget = _ForwardBudget(plan.limits.max_forward_calls, used)
    if backwards > plan.limits.max_backward_calls:
        raise ValueError("Selected block work exceeded the authorized backward budget")
    best_right = optimized.right
    arm_rights = {"A": right, "C": best_right}
    result, candidates = _candidates(base, adapter, runner, plan, basis, forks, contexts, arm_rights,
        target_blocks, protected_blocks, margins, cal_blocks, cal_margins, shared, populations, config, out, budget)
    return {**result, "family_ids": selection["family_ids"], "fit_items": list(names), "columns": list(columns),
        "right_factor": right.tolist(), "explicit_forward_calls": budget.used, "explicit_forward_limit": budget.limit,
        "backward_calls": backwards, "block_relaxation_target_status": target_relax.record()["status"]}, candidates


def _numeric_cell(space, runner, plan, basis, forks, contexts, blocks, margins, cal_blocks, cal_margins,
                  shared, config, budget, *, matched):
    wall, cpu = time.perf_counter(), time.process_time()
    pool, sample, _, cal_pool, cal_sample, _, cal_ids, floor = shared
    design, batch = config["probe_design"], config["inference"]["batch_size"]
    projected = blocks @ basis.right
    targets = basis.target_gradients
    all_rows = np.concatenate((targets, projected))
    ids = [*basis.target_ids, *(f"{a.prompt_id}:{a.position}" for a in sample.anchors)]
    native_calls = sum(any(f.prompt_id in set(basis.target_ids) for f in ctx) for ctx in contexts) + ceil(len(sample.anchors) / batch)
    allowance, passes, cache, best, calls = 0., [], {}, None, 0
    for _ in range(2):
        buffer = design["numerical_feasibility_buffer"]
        rhs = np.concatenate((design["target_margin"] + buffer - basis.target_margins, floor + allowance + buffer - margins))
        fit = _fixed(np.asarray(all_rows)[:, :, None], rhs, np.array([1.]), plan, config, ids)
        minimum = fit.update
        if minimum is None or not np.linalg.norm(minimum):
            return None, {"status": "LOCAL_WITNESS_REFUSED", "reason": fit.record()["status"],
                "solver": fit.record(), "calibration_passes": passes, "resources": _model_resources(wall, cpu, calls, 0)}

        def evaluate(amplitude):
            nonlocal calls
            u = minimum * (amplitude / np.linalg.norm(minimum))
            predicted_t = basis.target_margins + targets @ u
            predicted_p = margins + projected @ u
            if np.min(predicted_t) < design["target_margin"] or np.min(predicted_p) < floor + allowance:
                return {"feasible": False, "realized_norm": 0., "tensor_sha256": None, "reason": "individual_linear_margin_refusal"}
            preview = space.preview(u)
            numeric = bool(0 < preview["realized_norm"] <= plan.limits.max_update_norm and preview["changed_entries"])
            observed = None
            if numeric:
                digest = preview["tensor_sha256"]
                if digest not in cache:
                    budget.reserve(native_calls)
                    cache[digest] = measure_stored_joint(space, preview, runner, forks, contexts, pool, sample.anchors,
                        target_floor=design["target_margin"], preserve_floor=floor, batch_size=batch,
                        max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=plan.limits.max_memory_bytes)
                    calls += cache[digest]["resources"]["forward_calls"]
                observed = cache[digest]
                numeric = observed["passed"]
            return {key: preview[key] for key in ("realized_norm", "tensor_sha256", "changed_entries", "proposed_norm", "update", "delta", "stored_update_cosine")} | {
                "feasible": bool(numeric), "native_measurement": observed, "target_predictions": predicted_t.tolist(),
                "preservation_predictions": predicted_p.tolist()}

        if matched:
            match = match_stored_norm(evaluate, intended_norm=design["target_norms"][-1],
                maximum_norm=plan.limits.max_update_norm, relative_tolerance=design["norm_relative_tolerance"], max_probes=design["norm_search_steps"])
            best = match.selected if match.record()["status"] == "MATCHED" else None
            proposal = match.record()
        else:
            row = evaluate(float(np.linalg.norm(minimum)))
            best = row if row["feasible"] else None
            proposal = {"status": "MINIMUM_WITNESS" if best else "NATIVE_REFUSAL", "attempts": [serial(row)]}
        if best is None:
            passes.append({"proposal": proposal, "allowance": allowance, "solver": fit.record()})
            break
        preview = space.preview(best["update"])
        budget.reserve(ceil(len(cal_sample.anchors) / batch))
        actual, measured = space.margins(runner, cal_pool, cal_sample.anchors, preview,
            batch_size=batch, max_memory_bytes=plan.limits.max_memory_bytes)
        calls += measured["resources"]["forward_calls"]
        predicted = cal_margins + (cal_blocks @ basis.right) @ best["update"]
        residuals = tuple(MarginObservation(cal_ids[a.prompt_id], a.prompt_id, a.position, float(p), actual[a.key], group=a.group)
            for a, p in zip(cal_sample.anchors, predicted, strict=True))
        calibration = calibrate_residuals(residuals, quantile=design["adverse_item_quantile"])
        valid = calibration.allowance is not None and calibration.allowance <= allowance + 1e-7
        passes.append({"proposal": proposal, "solver": fit.record(), "allowance": allowance,
            "calibration": calibration.record(), "measurement": measured, "passed": bool(valid)})
        if valid:
            return best, {"status": "NATIVE_AND_CALIBRATION_PASS", "calibration_passes": passes,
                "direction_specific_allowance": allowance, "resources": _model_resources(wall, cpu, calls, 0)}
        if calibration.allowance is None:
            break
        allowance = max(allowance, calibration.allowance)
    return None, {"status": "NATIVE_OR_CALIBRATION_HOLD", "calibration_passes": passes,
        "direction_specific_allowance": allowance, "resources": _model_resources(wall, cpu, calls, 0)}


def _candidates(base, adapter, runner, plan, basis, forks, contexts, rights, target_blocks, protected_blocks,
                margins, cal_blocks, cal_margins, shared, populations, config, out, budget):
    contract = endpoint(config)
    generations, seen, owners, candidates, deltas, states = CandidateEvaluationBudget(contract), set(), {}, {}, {}, {}
    pool, sample, _, _, _, _, _, floor = shared
    def evaluate():
        measured = measure_endpoint(runner, populations["adaptation"][0], contract, candidate_budget=generations)
        protected = measure_endpoint(runner, populations["preservation-fit"][0], contract)
        guard = compare(populations["adaptation"][1], measured, min_comparison_gain=1, min_repeatable_corrections=1)
        preservation = compare_endpoint(populations["preservation-fit"][1], protected, contract)
        valid = sum(row["valid"] for row in measured["rows"]) / len(measured["rows"])
        guard.update(passed=bool(guard["passed"] and not preservation["prohibited_regressions"]
            and not preservation["lost_valid_presentations"] and valid >= config["endpoint"]["min_valid_rate"]),
            target_item_gain=guard["target_gain"], prohibited_regressions=guard["regressions"], lost_valid_presentations=guard["lost_valid"])
        return {"measurement": measured, "guard": guard, "preservation_fit_measurement": protected, "preservation_fit_guard": preservation}
    for suffix in (1, 2):
        for arm, right in rights.items():
            name = f"{arm}{suffix}"
            record = {"arm": name, "matched_cell": suffix == 2, "plan": plan.record(), "probe": {"reason": "no_admissible_local_witness"}}
            if right is not None:
                local = replace(basis, right=right, target_gradients=target_blocks @ right)
                space = StoredAffineSpace.open(adapter, runner, plan, local)
                try:
                    best, numeric = _numeric_cell(space, runner, plan, local, forks, contexts, protected_blocks, margins,
                        cal_blocks, cal_margins, shared, config, budget, matched=suffix == 2)
                except _BudgetExhausted:
                    best, numeric = None, {"status": "EXPLICIT_FORWARD_BUDGET_EXHAUSTED"}
                record["numerical"] = numeric
                if best is not None:
                    preview = space.preview(best["update"])
                    native_needed = sum(any(f.prompt_id in set(local.target_ids) for f in ctx) for ctx in contexts) + ceil(len(sample.anchors) / contract.batch_size)
                    try:
                        budget.reserve(native_needed)
                        patch, probe = probe_joint_generation(space, preview, runner, plan, forks, contexts, pool, sample.anchors,
                            target_floor=config["probe_design"]["target_margin"], preserve_floor=floor, batch_size=contract.batch_size,
                            seen=seen, evaluate_candidate=evaluate, baseline_endpoint=populations["adaptation"][1])
                    except _BudgetExhausted:
                        patch, probe = None, {"reason": "EXPLICIT_FORWARD_BUDGET_EXHAUSTED"}
                    stored_left, stored_right = canonical_factors(best["update"], right)
                    record.update(norm_match=numeric["calibration_passes"][-1]["proposal"],
                        left_factor=stored_left.tolist(), right_factor=stored_right.tolist(), columns=list(local.columns),
                        rounded_delta=preview["delta"].tolist(), target_predictions=best["target_predictions"], probe=probe,
                        proposed_norm=preview["proposed_norm"], realized_norm=preview["realized_norm"], tensor_sha256=preview["tensor_sha256"],
                        changed_entries=preview["changed_entries"], rounding_residual_norm=float(np.linalg.norm(preview["delta"] - np.outer(best["update"], right))))
                    singular = np.linalg.svd(preview["delta"], compute_uv=False)
                    record.update(stored_singular_values=singular.tolist(), stored_numerical_rank=int(np.count_nonzero(singular > max(preview["delta"].shape) * np.finfo(float).eps * singular[0])))
                    digest = preview["tensor_sha256"]
                    if digest in owners:
                        record["duplicate_of"] = owners[digest]
                    elif probe.get("endpoint"):
                        owners[digest] = name
                        deltas[name] = preview["delta"]
                        states[name] = {"realized_norm": preview["realized_norm"], "tensor_sha256": digest}
                        if patch is not None:
                            candidate = _materialize(base, out, name, plan, patch, probe)
                            candidates[name] = {key: value for key, value in candidate.items() if key != "checkpoint"}
                            candidates[name]["receipt_sha256"] = file_hash(candidate["checkpoint"] / ".weightbridge_receipt.json")
            else:
                record["numerical"] = {"status": "NO_WITHIN_CAP_ORIENTATION_WITNESS"}
            json_dump_atomic(out / f"arm-{name}.json", serial(record))
            print(name, record["numerical"]["status"], record["probe"]["reason"], flush=True)
    matched = {name: state for name, state in states.items() if name.endswith("2")}
    json_dump_atomic(out / "stored-pairwise.json", {"actual_delta_cosines": stored_update_cosines(deltas) if deltas else {},
        "matched_pair": compare_realized_norms(matched, intended_norm=config["probe_design"]["target_norms"][-1],
            relative_tolerance=config["probe_design"]["norm_relative_tolerance"]) if matched else None})
    return {"reason": "bounded_optimized_right_study_complete", "unique_complete_evaluations": generations.used,
        "practice_candidates": sorted(candidates)}, candidates


def run(base, material, out):
    from experiments.phase_iiip.source_manifest import source_hashes
    wall, cpu = time.perf_counter(), time.process_time()
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Model experiments require a private isolated workspace")
    config, manifest, sources = read_json(PROTOCOL), read_manifest(material), source_hashes()
    out.mkdir(parents=True, exist_ok=False)
    populations, candidates, contract = {}, {}, endpoint(config)
    with _open(base, config, config["checkpoint_sha256"], gradient_probe=True) as (adapter, runner, capabilities):
        for split in FIT_SPLITS:
            frozen = load_split(material, split, manifest)
            measured = measure_endpoint(runner, frozen, contract)
            populations[split] = frozen, measured
            json_dump_atomic(out / f"parent-{split}.json", {"measurement": measured, "summary": summarize(measured)})
        qualification = qualify(populations["adaptation"][1], config)
        detail = {"reason": "parent_behavior_did_not_qualify", "unique_complete_evaluations": 0}
        if qualification["disposition"] == "PASS":
            detail, candidates = _fit(base, adapter, runner, capabilities, populations, config, out)
    if sources != source_hashes() or load_checkpoint(base).checkpoint_sha256() != config["checkpoint_sha256"]:
        raise ValueError("Fitting source or immutable checkpoint changed during the study")
    artifacts = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".npz"} and "checkpoints" not in path.relative_to(out).parts}
    result = {"schema": "weightbridge.phase_iiip.fit.v1", "sources": sources, "checkpoint_sha256": config["checkpoint_sha256"],
        "manifest_sha256": file_hash(material / "manifest.json"), "optimization_splits": list(FIT_SPLITS),
        "qualification": qualification, "detail": detail, "candidates": candidates, "artifacts": artifacts,
        "hidden_opened": False, "resources": _resources(wall, cpu)}
    json_dump_atomic(out / "fit-report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.base, args.material, args.out)["detail"], flush=True)
