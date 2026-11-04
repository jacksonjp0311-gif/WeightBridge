"""Run the frozen Phase III-J practice comparison, then hand off saved winners."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import platform
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.run_causal_fork import _materialize, _plan, _training_paths
from experiments.phase_iiij.decimal_roster import PROTOCOL, SECONDARY_PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, compare_endpoint, endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.generative_fork import diagnose_fork, next_token_logits, with_fork_logits
from weightbridge.adaptation.model_adapter import Capability
from weightbridge.adaptation.preservation_policy import PreservationEvidence, preservation_decision
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def _read_practice(path, material, manifest, config):
    record = read_json(path)
    frozen_rows = load_split(material, "practice", manifest)
    measured = record["measurement"]
    if (record.get("split") != "practice" or record.get("checkpoint_sha256") != config["checkpoint_sha256"]
            or record.get("manifest_sha256") != file_hash(material / "manifest.json")
            or record.get("roster_sha256") != manifest["splits"]["practice"]["sha256"]
            or measured.get("endpoint") != endpoint(config).manifest()
            or len(frozen_rows) != len(measured["rows"])
            or any(any(row.get(key) != value for key, value in frozen.items())
                   for row, frozen in zip(measured["rows"], frozen_rows, strict=True))
            or record.get("qualification") != qualify(measured, config)):
        raise ValueError("Practice receipt does not bind the parent, endpoint and frozen practice roster")
    return frozen_rows, measured, record


def _preservation_group(row):
    if row["group"] == "preserve":
        return "copy"
    if row["tier"] == "mixed-sign":
        return "mixed-sign"
    return row["group"]


def _diagnose(runner, contract, practice, qualification, config):
    import torch

    start, cpu = time.perf_counter(), time.process_time()
    rows = [row for row in practice["rows"] if row["item"] in qualification["repeatable_errors"]]
    forks = [diagnose_fork(runner, row, contract, checkpoint_sha256=config["checkpoint_sha256"]) for row in rows]
    semantic = [fork for fork in forks if fork.status == "semantic"]
    measured = {}
    calls = 0
    with torch.inference_mode():
        for i in range(0, len(semantic), contract.batch_size):
            batch = semantic[i:i + contract.batch_size]
            logits = next_token_logits(runner, [fork.prefix_ids for fork in batch])
            calls += 1
            for fork, value in zip(batch, logits, strict=True):
                measured[fork.prompt_id] = with_fork_logits(fork, value, runner.tokenizer)
    forks = [measured.get(fork.prompt_id, fork) for fork in forks]
    required = len(contract.required_wordings)
    targets = []
    for item in dict.fromkeys(row["item"] for row in rows):
        wordings = [fork for fork in forks if fork.question_id == item]
        if len(wordings) == required and all(fork.status == "semantic" and fork.replay_status == "matched"
                                             for fork in wordings):
            targets.append(item)
    targets = targets[:config["fit_limits"]["max_repeatable_error_items"]]
    return tuple(fork for fork in forks if fork.question_id in targets), {
        "forks": [fork.to_dict() for fork in forks], "target_items": targets,
        "resources": {**_resources(start, cpu), "forward_calls": calls, "backward_calls": 0},
    }


def _guard(parent, measurement, qualification, contract):
    guard = compare_endpoint(parent, measurement, contract, min_repeatable_corrections=1)
    corrections = sorted(set(guard["corrected_repeatable_target_items"]) & qualification["repeatable_errors"].keys())
    guard["corrected_repeatable_negative_items"] = corrections
    guard["passed"] = bool(guard["passed"] and corrections)
    return guard


def _rank_measurement(guard):
    return (bool(guard["passed"]), -len(guard["prohibited_regressions"]),
            len(guard["corrected_repeatable_negative_items"]), guard["target_item_gain"])


def _fit(base, adapter, runner, capabilities, practice_rows, practice, qualification, config, out):
    from weightbridge.adaptation.counterexample_update import (
        construct_update, fit_fork_basis, fit_preservation, fit_projected_lora,
        measure_parent_decisions, observe_counterexamples,
    )
    from weightbridge.adaptation.fork_update import prepare_probe_reference, probe_update
    from weightbridge.adaptation.preservation_geometry import sample_anchors

    design, secondary = config["probe_design"], read_json(SECONDARY_PROTOCOL)
    contract = endpoint(config)
    runner = endpoint_runner(runner, contract)
    batch_size = contract.batch_size
    forks, diagnosis = _diagnose(runner, contract, practice, qualification, config)
    json_dump_atomic(out / "diagnosis.json", diagnosis)
    if not forks:
        return {}, {}, {"reason": "no_aligned_matched_forks"}

    fixed = design["fixed_layer"]
    plan = _plan(config, capabilities, fixed)
    memory = plan.limits.max_memory_bytes
    pool, pool_evidence = measure_parent_decisions(runner, practice, contract, group_for=_preservation_group,
                                                   batch_size=batch_size, max_memory_bytes=memory)
    json_dump_atomic(out / "parent-decisions.json", pool_evidence)
    solved = qualification["solved"]
    historical_items = {*solved["target"][:design["historical_target_trajectory_items"]],
                        *solved["positive"][:1], *solved["preserve"][:1]}
    groups = tuple(design["preservation_groups"])
    wordings = contract.required_wordings
    global_sample = sample_anchors(tuple(anchor for anchor in pool.eligible_anchors if anchor.item in historical_items),
                                   budget=design["historical_anchor_count"], mode="global",
                                   required_groups=groups, required_wordings=wordings)
    stratified = sample_anchors(pool.eligible_anchors, budget=len(groups) * len(wordings), mode="stratified",
                                required_groups=groups, required_wordings=wordings)
    json_dump_atomic(out / "sampling.json", {"historical": global_sample.record(), "stratified": stratified.record()})
    basis, basis_evidence = fit_fork_basis(adapter, runner, plan, forks, batch_size=batch_size)
    json_dump_atomic(out / "basis-L23.json", basis_evidence)
    if basis is None:
        return {}, {}, {"reason": "no_target_basis", "sampling": stratified.record()}

    # One exact-parent VJP per selected decision. Reuse it across samplers and doses.
    by_key = {anchor.key: anchor for anchor in (*global_sample.anchors, *stratified.anchors)}
    sensitivities, evidence = fit_preservation(adapter, runner, plan, basis, pool, tuple(by_key.values()),
                                              batch_size=batch_size)
    json_dump_atomic(out / "sensitivities-initial.json", evidence)
    risk_anchors = tuple(value.anchor for value in sensitivities.values())
    slopes = {value.anchor.key: float(value.gradient @ basis.preferred_left) for value in sensitivities.values()}
    risk = sample_anchors(risk_anchors, budget=len(groups) * len(wordings), mode="risk",
                          required_groups=groups, required_wordings=wordings, slopes=slopes,
                          radius=design["fixed_dose"])
    json_dump_atomic(out / "risk-diagnostic.json", {"selection": risk.record(),
                     "scope": "initial VJP-measured decisions only", "direction": "raw rank-one preferred factor",
                     "slopes": [{"prompt_id": key[0], "position": key[1], "slope": value}
                                for key, value in slopes.items()]})
    arms, candidates, seen, owner, feedback, references, arm_anchors = {}, {}, set(), {}, {}, {}, {}
    budget = CandidateEvaluationBudget(contract)

    def record(arm, active_plan, probe, extra, patch=None):
        digest = probe.get("candidate_tensor_sha256")
        key = (active_plan.region.tensor_name, digest)
        row = {"layer": active_plan.region.layer_index, "plan": active_plan.record(), "probe": probe, **extra}
        if digest and key in owner:
            row["duplicate_of"] = owner[key]
        elif digest and probe.get("endpoint"):
            owner[key] = arm
        if patch is not None:
            candidates[arm] = _materialize(base, out, arm, active_plan, patch, probe)
            row["checkpoint"] = {key: value for key, value in candidates[arm].items() if key != "checkpoint"}
        arms[arm] = row
        json_dump_atomic(out / f"arm-{arm}.json", row)
        guard = probe.get("endpoint", {}).get("guard", {})
        print(f"Arm {arm}: {probe['reason']}, practice gain {guard.get('target_item_gain')}", flush=True)

    def canonical(arm):
        return arms[arm].get("duplicate_of", arm)

    def probe(arm, active_plan, update, anchors, extra, *, collect=False, main=True):
        arm_anchors[arm] = tuple(anchors)
        max_calls = design["max_main_evaluations"] if main else budget.limit
        if update is None or budget.used >= max_calls:
            reason = "decode_budget_exhausted" if budget.used >= max_calls else extra.get("reason", "no_feasible_update")
            record(arm, active_plan, {"reason": reason}, extra)
            return
        prompt_ids = tuple(dict.fromkeys(anchor.prompt_id for anchor in anchors))
        trajectories = tuple(pool.trajectories[prompt_id] for prompt_id in prompt_ids)
        if prompt_ids not in references:
            reference = prepare_probe_reference(adapter, runner, forks, trajectories,
                                                batch_size=batch_size, max_memory_bytes=memory)
            references[prompt_ids] = reference
            json_dump_atomic(out / f"reference-{arm}.json", {"prompt_ids": prompt_ids, "resources": reference.resources})
        reference = references[prompt_ids]
        captured = {}

        def evaluate():
            measurement = measure_endpoint(runner, practice_rows, contract, candidate_budget=budget)
            observation = {"measurement": measurement, "guard": _guard(practice, measurement, qualification, contract)}
            if collect and observation["guard"]["prohibited_regressions"]:
                counterexamples, ce_evidence = observe_counterexamples(
                    runner, pool, observation, limit=design["new_anchors_per_refinement"],
                    batch_size=batch_size, max_memory_bytes=memory, exclude={anchor.key for anchor in anchors},
                )
                feedback[arm] = tuple(counterexamples)
                captured["counterexamples"] = ce_evidence
            return observation

        patch, observed = probe_update(adapter, runner, active_plan, update, radius=None, seen=seen,
                                       evaluate_candidate=evaluate, forks=forks, trajectories=trajectories,
                                       baseline_endpoint=practice, batch_size=batch_size, reference=reference)
        record(arm, active_plan, observed, {**extra, **captured, "trajectory_prompt_ids": prompt_ids}, patch)

    def constrained(arm, active_plan, active_basis, anchors, gradients, method, radius, *, collect=False, main=True):
        anchors = tuple(anchors)
        if any(anchor.key not in gradients for anchor in anchors):
            record(arm, active_plan, {"reason": "missing_anchor_sensitivity"},
                   {"anchors": [asdict(anchor) for anchor in anchors], "nominal_radius": radius})
            return
        update, geometry = construct_update(active_basis, tuple(gradients[anchor.key] for anchor in anchors),
                                            radius=radius, floor=design["safety_floor"], method=method)
        probe(arm, active_plan, update, anchors,
              {"anchors": [asdict(anchor) for anchor in anchors], "nominal_radius": radius,
               "geometry": geometry, "reason": geometry.get("reason", "no_feasible_update")},
              collect=collect, main=main)

    fixed_dose = design["fixed_dose"]
    if len(global_sample.anchors) == design["historical_anchor_count"]:
        constrained("A", plan, basis, global_sample.anchors, sensitivities, "nullspace", fixed_dose, collect=True)
    else:
        record("A", plan, {"reason": "insufficient_historical_anchors"}, {})

    if stratified.complete:
        constrained("B", plan, basis, stratified.anchors, sensitivities, "nullspace", fixed_dose, collect=True)
        constrained("C", plan, basis, stratified.anchors, sensitivities, "inequality", fixed_dose, collect=True)
        active = list(stratified.anchors)
        past = "C"
        for i in range(1, design["refinements"] + 1):
            prior = canonical(past)
            guard = arms[prior]["probe"].get("endpoint", {}).get("guard", {})
            if not guard.get("prohibited_regressions"):
                break
            active_keys = {anchor.key for anchor in active}
            added = [anchor for anchor in feedback.get(prior, ()) if anchor.key not in active_keys]
            name = f"D{i}"
            if not added:
                record(name, plan, {"reason": "no_novel_verified_counterexample"}, {"refines": past})
                break
            fresh = [anchor for anchor in added if anchor.key not in sensitivities]
            if fresh:
                gradients, detail = fit_preservation(adapter, runner, plan, basis, pool, tuple(fresh),
                                                     batch_size=batch_size)
                sensitivities.update(gradients)
                json_dump_atomic(out / f"sensitivities-{name}.json", detail)
            active.extend(added)
            constrained(name, plan, basis, active, sensitivities, "inequality", fixed_dose, collect=True)
            arms[name]["refines"] = past
            arms[name]["added_anchors"] = [asdict(anchor) for anchor in added]
            json_dump_atomic(out / f"arm-{name}.json", arms[name])
            past = name
        formulation = "C"
        ranked = []
        for arm in (name for name in ("C", "D1", "D2") if name in arm_anchors):
            guard = arms[canonical(arm)]["probe"].get("endpoint", {}).get("guard")
            if guard is not None:
                ranked.append((_rank_measurement(guard), arm))
        if ranked:
            formulation = max(ranked, key=lambda pair: pair[0])[1]
        anchors = arm_anchors.get(formulation, tuple(stratified.anchors))
        json_dump_atomic(out / "dose-formulation.json", {"source_arm": formulation,
                         "anchors": [asdict(anchor) for anchor in anchors], "doses": design["doses"]})
        for index, radius in enumerate(design["doses"], start=1):
            constrained(f"E{index}", plan, basis, anchors, sensitivities, "inequality", radius)
    else:
        for arm in ("B", "C"):
            record(arm, plan, {"reason": "incomplete_stratified_coverage"}, {"sampling": stratified.record()})

    alternate = secondary["alternate"]
    alternate_plan = _plan(config, capabilities, alternate["layer"])
    if stratified.complete:
        alt_basis, detail = fit_fork_basis(adapter, runner, alternate_plan, forks, batch_size=batch_size)
        json_dump_atomic(out / f"basis-L{alternate['layer']}.json", detail)
        if alt_basis is not None:
            gradients, detail = fit_preservation(adapter, runner, alternate_plan, alt_basis, pool, stratified.anchors,
                                                 batch_size=batch_size)
            json_dump_atomic(out / "sensitivities-alternate.json", detail)
            constrained(alternate["arm"], alternate_plan, alt_basis, stratified.anchors, gradients, "inequality",
                        alternate["nominal_dose"], main=False)
        else:
            record(alternate["arm"], alternate_plan, {"reason": "no_target_basis"}, {})
    else:
        record(alternate["arm"], alternate_plan, {"reason": "incomplete_stratified_coverage"}, {})

    conventional = secondary["conventional"]
    conventional_config = {**deepcopy(config), "conventional": conventional}
    conventional_plan = _plan(conventional_config, capabilities, conventional["layer"], conventional=True)
    preserve_items = [*solved["target"][:config["fit_limits"]["max_solved_target_items"]],
                      *solved["positive"], *solved["preserve"]]
    training = _training_paths(runner, contract, practice, forks, preserve_items)
    settings = {key: conventional[key] for key in ("rank", "steps", "lr", "seed")}
    update, training_evidence = fit_projected_lora(adapter, runner, conventional_plan, training,
                                                   radius=conventional["float_training_norm"],
                                                   batch_size=batch_size, **settings)
    probe(conventional["arm"], conventional_plan, update, stratified.anchors,
          {"training": training_evidence, "reason": training_evidence.get("reason", "no_conventional_update")}, main=False)
    return arms, candidates, {"reason": "practice_complete", "sampling": stratified.record(),
                              "unique_complete_evaluations": budget.used}


def run(base, material, out, practice_path, selection_base=None):
    import torch
    import transformers
    from experiments.phase_iiij.verification import assess, source_hashes

    start, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    sources, manifest_sha = source_hashes(), file_hash(material / "manifest.json")
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Run in a private host workspace and archive public evidence afterward")
    practice_rows, practice, parent_record = _read_practice(practice_path, material, manifest, config)
    qualification = parent_record["qualification"]
    out.mkdir(parents=True, exist_ok=False)
    json_dump_atomic(out / "practice.json", parent_record)
    baseline = config["checkpoint_sha256"]
    arms, candidates, detail = {}, {}, {"reason": "parent_practice_qualification"}
    if qualification["disposition"] == "PASS":
        with _open(base, config, baseline, gradient_probe=True) as (adapter, runner, capabilities):
            if Capability.COMPUTE_GRADIENTS in capabilities.capabilities:
                arms, candidates, detail = _fit(base, adapter, runner, capabilities, practice_rows, practice,
                                                qualification, config, out)
            else:
                detail = {"reason": "gradient_capability_unavailable"}
        del runner
    if sources != source_hashes() or manifest_sha != file_hash(material / "manifest.json"):
        raise ValueError("Experiment sources or frozen material changed during fitting")
    if load_checkpoint(base).checkpoint_sha256() != baseline:
        raise ValueError("Original checkpoint changed during fitting")
    # The fit takes no selection argument. Only the host verifier receives this path.
    assessment = assess(base, material, out, candidates, selection_base)
    measured = [row["probe"] for row in arms.values() if row["probe"].get("endpoint")]
    intervals = [row["geometry"]["local_feasibility"] for row in arms.values()
                 if row.get("geometry", {}).get("local_feasibility")]
    decision = preservation_decision(PreservationEvidence(
        measurement_valid=all(value for gate, value in qualification["gates"].items() if gate != "repeatable_errors"),
        persistent_errors=len(qualification["repeatable_errors"]),
        missing_groups=tuple(detail.get("sampling", {}).get("missing_groups", ())),
        locally_feasible=any(item["safe_interval"] is not None for interval in intervals
                             for item in interval["per_target"]) if intervals else None,
        constraint_conflict=any(row["probe"]["reason"] == "constraint_conflict" for row in arms.values()),
        rounded_noop=any(row["probe"]["reason"] == "no_stored_change" for row in arms.values()),
        surrogate_mismatch=any((row.get("fork") or {}).get("cross_entropy_gain", 0) > 0 and
                               row["endpoint"]["guard"]["target_item_gain"] <= 0 for row in measured),
        corrected_with_regressions=any(row["endpoint"]["guard"]["corrected_repeatable_negative_items"] and
                                       row["endpoint"]["guard"]["prohibited_regressions"] for row in measured),
        practice_pass=bool(candidates), selection_pass=bool(assessment["winner"]) if candidates else None,
    )).record()
    if assessment["winner"]:
        reason, disposition = "awaiting_host_hidden_verification", "HOLD"
    elif candidates:
        reason, disposition = "independent_selection_gate", "FAIL"
    elif not arms:
        reason, disposition = detail["reason"], "HOLD"
    else:
        reason, disposition = "no_endpoint_qualified_probe", "HOLD"
    report = {
        "schema": "weightbridge.phase_iiij.preservation_report.v1", "base_sha256": baseline,
        "manifest_sha256": manifest_sha, "sources": sources, "qualification": qualification,
        "single_cycle": disposition, "reason": reason, "hidden_opened": False,
        "selection": assessment, "meta": decision, "practice_design": detail,
        "arms": {arm: {key: row[key] for key in ("layer", "duplicate_of", "checkpoint") if key in row}
                 | {"reason": row["probe"]["reason"]} for arm, row in arms.items()},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "transformers": transformers.__version__},
        "resources": {"host": _resources(start, cpu), "practice_parent": parent_record["resources"]},
        "artifacts": {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))
                      if "checkpoints" not in path.relative_to(out).parts},
    }
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out", "practice"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--selection-base", type=Path)
    args = parser.parse_args()
    result = run(args.base, args.material, args.out, args.practice, args.selection_base)
    print(json.dumps({key: result[key] for key in ("single_cycle", "reason", "arms")}, indent=2), flush=True)
