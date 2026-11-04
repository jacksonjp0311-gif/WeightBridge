"""Run the frozen robust J-space comparison, then gate one locked practice survivor."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan
from experiments.phase_iiij.run_preservation import _diagnose, _guard, _preservation_group
from experiments.phase_iiik.baseline import FITTING_SPLITS
from experiments.phase_iiik.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.fork_update import AffineUpdate
from weightbridge.adaptation.model_adapter import Capability
from weightbridge.adaptation.robust_jspace import (
    JSpaceEvidence, calibrate_margin_floor, directional_risk, estimate_window, interior_score, jspace_recommendation,
    solve_robust,
)
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def _read_parent(path, material, manifest, config, split):
    if split == "hidden":
        raise ValueError("The experiment runner cannot receive a hidden receipt")
    record = read_json(path)
    frozen = load_split(material, split, manifest)
    measured = record["measurement"]
    if (record.get("split") != split or record.get("checkpoint_sha256") != config["checkpoint_sha256"]
            or record.get("manifest_sha256") != file_hash(material / "manifest.json")
            or record.get("roster_sha256") != manifest["splits"][split]["sha256"]
            or measured.get("endpoint") != endpoint(config).manifest()
            or len(frozen) != len(measured["rows"])
            or any(any(row.get(key) != value for key, value in frozen_row.items())
                   for row, frozen_row in zip(measured["rows"], frozen, strict=True))
            or record.get("qualification") != qualify(measured, config)):
        raise ValueError("Parent receipt does not bind the frozen partition, checkpoint, and endpoint")
    return frozen, measured, record


def _read_fit(directory, material, manifest, config):
    return {split: _read_parent(directory / f"{split}.json", material, manifest, config, split)
            for split in FITTING_SPLITS}


def _choose_floor(calibration_anchors, grid):
    chosen = calibrate_margin_floor([anchor.margin for anchor in calibration_anchors], grid)
    return chosen["selected_floor"], chosen["cutoff"]


def _practice_guard(parent, candidate, fit_parent, fit_candidate, qualification, contract):
    adaptation = _guard(parent, candidate, qualification, contract)
    preservation = compare_endpoint(fit_parent, fit_candidate, contract)
    preservation["passed"] = not (preservation["prohibited_regressions"] or preservation["lost_valid_presentations"])
    return {**adaptation, "passed": bool(adaptation["passed"] and preservation["passed"]),
            "adaptation_passed": adaptation["passed"], "preservation_fit_passed": preservation["passed"]}, preservation


def _active_measurements(anchors, observed, prompt_ids, geometry, *, floor, allowance):
    trajectory = observed.get("trajectory")
    if not trajectory:
        return []
    paths = dict(zip(prompt_ids, trajectory["trajectories"], strict=True))
    predicted = {(row["prompt_id"], row["position"]): row["predicted_margin"]
                 for row in geometry.get("preservation_predictions", [])}
    rows = []
    for anchor in anchors:
        path = paths[anchor.prompt_id]
        margin = path["candidate_margins"][anchor.position]
        expectation = predicted.get(anchor.key)
        rows.append({**asdict(anchor), "parent_replay_margin": path["parent_margins"][anchor.position],
                     "predicted_margin": expectation, "measured_margin": margin,
                     "candidate_argmax": path["candidate_argmax_ids"][anchor.position],
                     "decision_preserved": path["candidate_argmax_ids"][anchor.position] == anchor.token_id,
                     "residual": None if expectation is None else margin - expectation,
                     "positive": margin > 0, "meets_floor": margin >= floor,
                     "meets_robust_floor": margin >= floor + allowance})
    return rows


def _max_width(geometry):
    candidates = geometry.get("local_feasibility", {}).get("per_target", [])
    widths = [row["safe_interval"]["upper"] - row["safe_interval"]["lower"]
              for row in candidates if row["safe_interval"] is not None]
    return max(widths, default=0.)


def _construct(basis, selected, *, radius, floor, allowance):
    gradients = np.stack([row.gradient for row in selected])
    margins = np.array([row.margin for row in selected])
    solution = solve_robust(basis.preferred_left, gradients, margins, radius=radius, margin_floor=floor,
                            allowance=allowance)
    evidence = {"schema": "weightbridge.robust_rank_one.v1", "method": "fixed_right_inequality_projection",
                "radius": radius, "floor": floor, "allowance": allowance, "solver": solution.record()}
    factor = solution.update
    if factor is None:
        return None, {**evidence, "reason": solution.status}
    norm = float(np.linalg.norm(factor))
    if norm <= 1e-12:
        return None, {**evidence, "reason": "no_nonzero_feasible_direction"}
    target_slopes = basis.target_gradients @ (factor / norm)
    preserve_slopes = gradients @ (factor / norm)
    window = estimate_window(basis.target_margins, target_slopes, margins, preserve_slopes, radius=radius,
                             margin_floor=floor, allowance=allowance, target_items=basis.target_items)
    return AffineUpdate(basis.columns, factor[:, None], basis.right[None, :], "inequality"), {
        **evidence, "reason": "local_candidate", "unrounded_norm": norm, "local_feasibility": window,
        "target_predictions": [{"prompt_id": prompt, "item": item, "parent_margin": float(margin),
                                "slope": float(slope), "predicted_margin": float(margin + slope * norm)}
                               for prompt, item, margin, slope in zip(basis.target_ids, basis.target_items,
                                                                    basis.target_margins, target_slopes, strict=True)],
        "preservation_predictions": [{"prompt_id": row.anchor.prompt_id, "position": row.anchor.position,
                                      "parent_margin": row.margin, "slope": float(slope),
                                      "predicted_margin": float(row.margin + slope * norm),
                                      "required_margin": floor + allowance}
                                     for row, slope in zip(selected, preserve_slopes, strict=True)],
    }


def _selection_keys(rows, observed):
    passed = []
    for name, record in rows.items():
        if not name.startswith("C"):
            continue
        probe = observed(name)
        guard = probe.get("endpoint", {}).get("guard", {})
        if guard.get("passed"):
            minimum = min((row["measured_margin"] for row in record["active_measurements"]), default=None)
            width = _max_width(record["geometry"])
            passed.append({"arm": name, "gain": guard["target_item_gain"],
                           "norm": probe["realized_update_norm"], "minimum_measured_margin": minimum,
                           "width": width, "floor": record["floor"]})
    max_gain = max(passed, key=lambda row: (row["gain"], -row["norm"]), default=None)
    ranked = []
    for row in passed:
        margin = row["minimum_measured_margin"]
        key = interior_score(practice_pass=True, practice_gain=row["gain"],
                             measured_slack=None if margin is None else margin - row["floor"],
                             measured_min_margin=margin, feasible_window_width=row["width"], realized_norm=row["norm"])
        if key is not None:
            ranked.append((key, row))
    best = max(ranked, key=lambda pair: pair[0], default=None)
    interior = None if best is None else best[1]
    return {"measured_practice_passers": passed, "max_gain": max_gain, "interior": interior}


def _fit(base, adapter, runner, capabilities, fitting, config, out):
    from weightbridge.adaptation.counterexample_update import fit_fork_basis, fit_preservation, measure_parent_decisions
    from weightbridge.adaptation.fork_update import prepare_probe_reference, probe_update
    from weightbridge.adaptation.jspace_update import calibrate_direction_residuals, measure_directional_secant
    from weightbridge.adaptation.robust_jspace import calibrate_residuals, select_preservation

    design = config["probe_design"]
    contract = endpoint(config)
    runner = endpoint_runner(runner, contract)
    adaptation_rows, adaptation, adaptation_record = fitting["adaptation"]
    preservation_rows, preservation, _ = fitting["preservation-fit"]
    _, calibration, _ = fitting["calibration"]
    qualification = adaptation_record["qualification"]
    forks, diagnosis = _diagnose(runner, contract, adaptation, qualification, config)
    json_dump_atomic(out / "diagnosis.json", diagnosis)
    if not forks:
        return {}, {}, {"reason": "no_aligned_matched_forks"}, None

    plan = _plan(config, capabilities, design["fixed_layer"])
    batch_size = contract.batch_size
    memory = plan.limits.max_memory_bytes
    groups, wordings = tuple(design["preservation_groups"]), contract.required_wordings
    settings = {"group_for": _preservation_group, "batch_size": batch_size, "max_memory_bytes": memory}
    pool, parent_pool = measure_parent_decisions(runner, preservation, contract, **settings)
    json_dump_atomic(out / "preservation-fit-decisions.json", parent_pool)
    basis, basis_evidence = fit_fork_basis(adapter, runner, plan, forks, batch_size=batch_size)
    json_dump_atomic(out / "basis.json", basis_evidence)
    if basis is None:
        return {}, {}, {"reason": "no_target_basis"}, None
    slopes, direction = measure_directional_secant(adapter, runner, plan, basis, pool,
                                                   radius=design["directional_secant_radius"], batch_size=batch_size)
    json_dump_atomic(out / "directional-secant.json", direction)
    selection_settings = {"budget": design["anchors"], "required_groups": groups, "required_wordings": wordings}
    if slopes:
        selection_settings.update(slopes=slopes, radius=design["directional_secant_radius"])
    historical = select_preservation(pool.eligible_anchors, mode="historical", **selection_settings)
    stratified = select_preservation(pool.eligible_anchors, mode="stratified", **selection_settings)
    sensitivity = diagnostic = None
    if slopes and all(anchor.key in slopes for anchor in pool.eligible_anchors):
        sensitivity = select_preservation(pool.eligible_anchors, mode="stratified_directional", **selection_settings)
        diagnostic = select_preservation(pool.eligible_anchors, mode="directional", **selection_settings)
    selections = {"historical": historical, "stratified_low_margin": stratified,
                  "stratified_directional": sensitivity, "directional": diagnostic}
    json_dump_atomic(out / "sampling.json", {name: sample.record() if sample is not None else None
                                            for name, sample in selections.items()})
    # Only A/B/C use measured preservation-fit VJPs. The two auxiliary samplers are diagnostic.
    anchor_by_key = {anchor.key: anchor for sample in (historical, sensitivity) if sample is not None
                     for anchor in sample.anchors}
    gradients, gradient_record = fit_preservation(adapter, runner, plan, basis, pool, tuple(anchor_by_key.values()),
                                                  batch_size=batch_size)
    json_dump_atomic(out / "fit-sensitivities.json", gradient_record)

    calibration_pool, calibration_parent = measure_parent_decisions(runner, calibration, contract, **settings)
    json_dump_atomic(out / "calibration-decisions.json", calibration_parent)
    calibration_sample = select_preservation(calibration_pool.eligible_anchors, budget=design["calibration_anchors"],
                                              mode="stratified", required_groups=groups, required_wordings=wordings)
    cal_sensitivities, cal_gradient_record = fit_preservation(adapter, runner, plan, basis, calibration_pool,
                                                             calibration_sample.anchors, batch_size=batch_size)
    json_dump_atomic(out / "calibration-sensitivities.json", cal_gradient_record)
    cal_anchors = tuple(anchor for anchor in calibration_sample.anchors if anchor.key in cal_sensitivities)
    calibration_obs = ()
    if cal_anchors:
        calibration_obs, _, cal_measurement = calibrate_direction_residuals(
            adapter, runner, plan, basis, calibration_pool, cal_anchors, (), cal_sensitivities,
            radius=design["directional_secant_radius"], batch_size=batch_size,
        )
        json_dump_atomic(out / "calibration-residuals.json", cal_measurement)
    floor, floor_cap = _choose_floor(cal_anchors, design["positive_floor_grid"])
    calibrated = calibrate_residuals(calibration_obs, quantile=design["adverse_item_quantile"]) if calibration_obs else None
    allowance = calibrated.allowance if calibrated is not None else None
    json_dump_atomic(out / "calibration-choice.json", {
        "sample": calibration_sample.record(), "floor_grid": design["positive_floor_grid"],
        "floor_cap": floor_cap, "floor": floor, "allowance": allowance,
        "calibration": calibrated.record() if calibrated is not None else None,
    })
    prompts = tuple(dict.fromkeys(anchor.prompt_id for anchor in anchor_by_key.values()))
    trajectories = tuple(pool.trajectories[prompt] for prompt in prompts)
    reference = prepare_probe_reference(adapter, runner, forks, trajectories, batch_size=batch_size,
                                        max_memory_bytes=memory)
    json_dump_atomic(out / "probe-reference.json", {"prompt_ids": prompts, "resources": reference.resources})
    budget = CandidateEvaluationBudget(contract)
    seen, owner, patches, rows = set(), {}, {}, {}

    def evaluate():
        measured = measure_endpoint(runner, adaptation_rows, contract, candidate_budget=budget)
        measured_fit = measure_endpoint(runner, preservation_rows, contract)
        guard, fit_guard = _practice_guard(adaptation, measured, preservation, measured_fit, qualification, contract)
        return {"measurement": measured, "guard": guard, "preservation_fit_measurement": measured_fit,
                "preservation_fit_guard": fit_guard}

    def canonical(name):
        return rows[name].get("duplicate_of", name)

    def observed(name):
        return rows[canonical(name)]["probe"]

    def attempt(name, sample, radius, epsilon, delta):
        anchors = () if sample is None else sample.anchors
        record = {"layer": plan.region.layer_index, "plan": plan.record(), "nominal_radius": radius,
                  "floor": epsilon, "allowance": delta, "anchors": [asdict(anchor) for anchor in anchors],
                  "trajectory_prompt_ids": prompts}
        geometry, probe, patch = {}, {"reason": "no_feasible_update"}, None
        if sample is None or not anchors or (not name.startswith("A") and not sample.complete):
            probe["reason"] = "incomplete_preservation_coverage"
        elif epsilon is None or delta is None:
            probe["reason"] = "no_calibrated_positive_slack"
        elif any(anchor.key not in gradients for anchor in anchors):
            probe["reason"] = "missing_anchor_sensitivity"
        else:
            update, geometry = _construct(basis, [gradients[anchor.key] for anchor in anchors],
                                          radius=radius, floor=epsilon, allowance=delta)
            if update is None:
                probe["reason"] = geometry.get("reason", probe["reason"])
            else:
                patch, probe = probe_update(adapter, runner, plan, update, radius=None, seen=seen,
                                            evaluate_candidate=evaluate, forks=forks, trajectories=trajectories,
                                            baseline_endpoint=adaptation, batch_size=batch_size, reference=reference)
        digest = probe.get("candidate_tensor_sha256")
        if digest and digest in owner:
            record["duplicate_of"] = owner[digest]
        elif digest and probe.get("endpoint"):
            owner[digest] = name
        if patch is not None:
            patches[name] = patch
        record.update(probe=probe, geometry=geometry)
        rows[name] = record
        record["active_measurements"] = _active_measurements(
            anchors, observed(name), prompts, geometry, floor=epsilon or 0., allowance=delta or 0.,
        )
        json_dump_atomic(out / f"arm-{name}.json", record)
        guard = observed(name).get("endpoint", {}).get("guard", {})
        print(f"Arm {name}: {probe['reason']}, practice gain {guard.get('target_item_gain')}", flush=True)

    for arm, sample, epsilon, delta in (("A", historical, 0., 0.), ("B", sensitivity, 0., 0.),
                                        ("C", sensitivity, floor, allowance)):
        for index, radius in enumerate(design["doses"], start=1):
            attempt(f"{arm}{index}", sample, radius, epsilon, delta)
    sampler_audit = {}
    for label, sample in selections.items():
        if sample is None:
            continue
        measured_sample = [anchor for anchor in sample.anchors if anchor.key in slopes]
        audit = {}
        for name, record in rows.items():
            if record.get("duplicate_of") or not record["probe"].get("endpoint"):
                continue
            fit_guard = record["probe"]["endpoint"]["preservation_fit_guard"]
            regressed = set(fit_guard["prohibited_regressions"])
            flagged = {anchor.prompt_id for anchor in measured_sample
                       if anchor.margin + record["nominal_radius"] * slopes[anchor.key] <= 0}
            audit[name] = {"observed_regressions": sorted(regressed), "predicted_risk": sorted(flagged),
                           "anticipated": sorted(flagged & regressed), "unanticipated": sorted(regressed - flagged),
                           "flagged_without_regression": sorted(flagged - regressed)}
        sampler_audit[label] = audit
    json_dump_atomic(out / "sampler-regression-audit.json", {
        "scope": "Preservation-fit presentations only. Raw preferred-direction BF16 secant extrapolated to nominal arm radius; "
                 "does not estimate a distinct arm-specific Jacobian for unsampled decisions.",
        "comparisons": sampler_audit,
    })
    choice = _selection_keys(rows, observed)
    candidates = {}
    if choice["interior"] is not None:
        name = choice["interior"]["arm"]
        candidate = _materialize(base, out, "D", plan, patches[canonical(name)], observed(name))
        candidates["D"] = candidate
        choice["checkpoint"] = {key: value for key, value in candidate.items() if key != "checkpoint"}
    json_dump_atomic(out / "d-choice.json", choice)
    return rows, candidates, {"reason": "practice_complete", "unique_complete_evaluations": budget.used,
                              "d_choice": choice, "validation_input_used_by_optimizer": False,
                              "sampling": sensitivity.record() if sensitivity is not None else None,
                              "calibration_sufficient": calibrated is not None and allowance is not None and floor is not None,
                              "max_raw_directional_risk": max((directional_risk(anchor.margin, slopes[anchor.key],
                                  radius=design["directional_secant_radius"])
                                  for anchor in pool.eligible_anchors if anchor.key in slopes), default=None)}, (basis, plan, calibrated)


def _held_residual_audit(adapter, runner, held, config, out, state):
    from weightbridge.adaptation.counterexample_update import fit_preservation, measure_parent_decisions
    from weightbridge.adaptation.jspace_update import calibrate_direction_residuals
    from weightbridge.adaptation.robust_jspace import select_preservation, validate_residuals

    basis, plan, calibrated = state
    design, contract = config["probe_design"], endpoint(config)
    runner = endpoint_runner(runner, contract)
    batch_size = contract.batch_size
    pool, pool_record = measure_parent_decisions(runner, held, contract, group_for=_preservation_group,
                                                 batch_size=batch_size, max_memory_bytes=plan.limits.max_memory_bytes)
    sample = select_preservation(pool.eligible_anchors, budget=design["validation_diagnostic_anchors"],
                                 mode="stratified", required_groups=design["preservation_groups"],
                                 required_wordings=contract.required_wordings)
    sensitivities, gradient_record = fit_preservation(adapter, runner, plan, basis, pool, sample.anchors,
                                                      batch_size=batch_size)
    anchors = tuple(anchor for anchor in sample.anchors if anchor.key in sensitivities)
    json_dump_atomic(out / "validation-decisions.json", pool_record)
    json_dump_atomic(out / "validation-sensitivities.json", gradient_record)
    if not anchors:
        return {"reason": "no_held_anchor_sensitivities", "sample": sample.record()}
    _, held_obs, residuals = calibrate_direction_residuals(
        adapter, runner, plan, basis, pool, (), anchors, sensitivities,
        radius=design["directional_secant_radius"], batch_size=batch_size,
    )
    json_dump_atomic(out / "validation-residuals.json", residuals)
    return {"reason": "audit_only_after_fit_lock", "sample": sample.record(),
            "validation": validate_residuals(calibrated, held_obs) if calibrated is not None and held_obs else None}


def run(base, material, out, fit_dir, parent_validation, parent_selection=None):
    import torch
    import transformers
    from experiments.phase_iiik.verification import assess, source_hashes

    start, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    sources, manifest_sha = source_hashes(), file_hash(material / "manifest.json")
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Run in a private host workspace and archive public evidence afterward")
    fitting = _read_fit(fit_dir, material, manifest, config)
    qualification = fitting["adaptation"][2]["qualification"]
    out.mkdir(parents=True, exist_ok=False)
    for split, (_, _, record) in fitting.items():
        json_dump_atomic(out / f"parent-{split}.json", record)
    baseline = config["checkpoint_sha256"]
    rows, candidates, state, held_audit = {}, {}, None, None
    detail = {"reason": "parent_adaptation_qualification"}
    if qualification["disposition"] == "PASS":
        with _open(base, config, baseline, gradient_probe=True) as (adapter, runner, capabilities):
            if Capability.COMPUTE_GRADIENTS in capabilities.capabilities:
                rows, candidates, detail, state = _fit(base, adapter, runner, capabilities, fitting, config, out)
            else:
                detail = {"reason": "gradient_capability_unavailable"}
            # The validation role is first loaded here, after all practice candidates and D choice are sealed.
            _, held, validation_record = _read_parent(parent_validation, material, manifest, config,
                                                      "preservation-validation")
            json_dump_atomic(out / "parent-preservation-validation.json", validation_record)
            if state is not None:
                held_audit = _held_residual_audit(adapter, runner, held, config, out, state)
                json_dump_atomic(out / "validation-audit.json", held_audit)
        del runner
    if sources != source_hashes() or manifest_sha != file_hash(material / "manifest.json"):
        raise ValueError("Experiment sources or frozen material changed during fitting")
    if load_checkpoint(base).checkpoint_sha256() != baseline:
        raise ValueError("Original checkpoint changed during fitting")
    assessment = assess(base, material, out, candidates, parent_selection, parent_validation)
    if assessment["winner"]:
        reason, disposition = "awaiting_host_hidden_verification", "HOLD"
    elif assessment["selection"]:
        reason, disposition = "independent_selection_gate", "FAIL"
    elif candidates:
        reason, disposition = "preservation_validation_gate", "FAIL"
    elif not rows:
        reason, disposition = detail["reason"], "HOLD"
    else:
        reason, disposition = "no_interior_practice_survivor", "HOLD"
    observed = [record["probe"] for record in rows.values() if record["probe"].get("endpoint")]
    protected = [value for record in rows.values() for value in record["active_measurements"]]
    calibrated_probes = [record["active_measurements"] for name, record in rows.items()
                         if name.startswith("C") and record["active_measurements"]]
    measured_validation = (held_audit or {}).get("validation") or {}
    sampling = detail.get("sampling") or {}
    validation = assessment["validation"].get("D")
    meta_evidence = JSpaceEvidence(
        measurement_valid=all(value for gate, value in qualification["gates"].items() if gate != "repeatable_errors"),
        persistent_errors=len(qualification["repeatable_errors"]), missing_groups=tuple(sampling.get("missing_groups", ())),
        missing_strata=tuple(sampling.get("missing_strata", ())),
        directional_vulnerability=detail.get("max_raw_directional_risk"),
        slack_satisfied=any(all(value["meets_robust_floor"] for value in group)
                            for group in calibrated_probes) if calibrated_probes else None,
        calibration_sufficient=detail.get("calibration_sufficient"),
        residual_validation_pass=measured_validation["item_coverage"] == 1. if measured_validation.get("item_coverage") is not None else None,
        max_prediction_error=max((abs(value["residual"]) for value in protected if value["residual"] is not None),
                                 default=None),
        rounded_noop=any(record["probe"]["reason"] == "no_stored_change" for record in rows.values()),
        precision_margin_disagreement=any(value["predicted_margin"] is not None and value["predicted_margin"] > 0
                                          and not value["positive"] for value in protected),
        constraint_conflict=any(record["geometry"].get("solver", {}).get("update_norm") is None
                                for record in rows.values() if record["geometry"]),
        window_width=max((_max_width(record["geometry"]) for record in rows.values() if record["geometry"]), default=None),
        corrected_with_regressions=any(probe["endpoint"]["guard"]["corrected_repeatable_negative_items"] and
            (probe["endpoint"]["guard"]["prohibited_regressions"]
             or probe["endpoint"]["preservation_fit_guard"]["prohibited_regressions"]) for probe in observed),
        practice_pass=bool(candidates), preservation_validation_required=True,
        preservation_validation_pass=None if validation is None else validation["passed"],
        selection_pass=bool(assessment["winner"]) if assessment["selection"] else None,
    )
    arms = {arm: {key: row[key] for key in ("layer", "duplicate_of", "nominal_radius") if key in row}
            | {"reason": row["probe"]["reason"]} for arm, row in rows.items()}
    d_choice = detail.get("d_choice", {}).get("interior")
    arms["D"] = {"reason": "locked_interior_candidate" if d_choice else "no_interior_practice_survivor",
                 "source_arm": d_choice["arm"] if d_choice else None}
    arms["E"] = {"reason": "not_eligible" if validation is None else "preservation_validation_pass"
                 if validation["passed"] else "preservation_validation_failed", "assessment": validation}
    report = {
        "schema": "weightbridge.phase_iiik.jspace_report.v1", "base_sha256": baseline,
        "manifest_sha256": manifest_sha, "sources": sources, "qualification": qualification,
        "single_cycle": disposition, "reason": reason, "hidden_opened": False,
        "selection": assessment, "practice_design": detail, "held_residual_audit": held_audit,
        "arms": arms, "meta": jspace_recommendation(meta_evidence).record(), "meta_evidence": asdict(meta_evidence),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "transformers": transformers.__version__},
        "resources": {"host": _resources(start, cpu),
                      "fitting_parent": {split: record["resources"] for split, (_, _, record) in fitting.items()}},
        "artifacts": {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))
                      if "checkpoints" not in path.relative_to(out).parts},
    }
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out", "fit-dir", "parent-validation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--selection-base", type=Path)
    args = parser.parse_args()
    result = run(args.base, args.material, args.out, args.fit_dir, args.parent_validation, args.selection_base)
    print(json.dumps({key: result[key] for key in ("single_cycle", "reason", "arms")}, indent=2), flush=True)
