"""Run the frozen practice-only causal fork comparison, then hand off saved winners."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.decimal_roster import PROTOCOL, load_split, read_manifest
from experiments.phase_iiii.verification import assess, source_hashes
from weightbridge.adaptation.behavior_endpoint import (
    CandidateEvaluationBudget, build_decimal_fit_rows, compare_endpoint, endpoint_runner, measure_endpoint,
)
from weightbridge.adaptation.content_scoring import Continuation
from weightbridge.adaptation.endpoint_update import fit_endpoint_probes
from weightbridge.adaptation.fork_policy import decide_fork_action
from weightbridge.adaptation.generative_fork import (
    diagnose_fork, next_token_logits, protected_trajectory, trajectory_diagnostics, trajectory_logits,
    with_fork_logits,
)
from weightbridge.adaptation.model_adapter import Capability
from weightbridge.adaptation.universal_awl import HostAuthorization, compile_awl_v3, parse_awl_v3
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64, apply_safetensors_patch


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
        raise ValueError("Practice record does not bind the frozen parent, endpoint and practice roster")
    return frozen_rows, measured, record


def _plan(config, capabilities, layer, *, conventional=False, scanning=False):
    matches = [region for region in capabilities.editable_regions
               if region.layer_index == layer and region.role == "ffn_output"]
    if len(matches) != 1:
        raise ValueError(f"Frozen layer {layer} did not resolve to one authorized affine region")
    spec = deepcopy(config["intent"])
    method = "gradient" if conventional else "constrained"
    if conventional:
        spec["methods"] = [method]
        spec["limits"].update(max_update_entries=config["conventional"]["max_update_entries"],
                              max_update_norm=config["conventional"]["max_update_norm"],
                              max_rank=config["conventional"]["rank"],
                              max_backward_calls=4 * config["conventional"]["steps"])
    if scanning:
        spec["limits"]["max_backward_calls"] = 2 + config["probe_design"]["trajectory_constraints"]
    intent = parse_awl_v3(spec)
    host = HostAuthorization(config["checkpoint_sha256"], intent.methods, (matches[0].tensor_name,),
                             (intent.objective.family, *intent.preserve), intent.practice_data_id,
                             intent.evaluation_data_id, intent.limits)
    return compile_awl_v3(intent, capabilities, host, method=method)


def _batches(rows, size):
    return (rows[start:start + size] for start in range(0, len(rows), size))


def _diagnose(runner, contract, practice, qualification, config):
    import torch

    started, cpu = time.perf_counter(), time.process_time()
    forks = [diagnose_fork(runner, row, contract, checkpoint_sha256=config["checkpoint_sha256"])
             for row in practice["rows"] if row["item"] in qualification["repeatable_errors"]]
    measured = {}
    calls = 0
    with torch.inference_mode():
        for chunk in _batches([fork for fork in forks if fork.status == "semantic"], contract.batch_size):
            calls += 1
            values = next_token_logits(runner, [fork.prefix_ids for fork in chunk])
            for fork, logits in zip(chunk, values, strict=True):
                measured[fork.prompt_id] = with_fork_logits(fork, logits, runner.tokenizer)
    forks = tuple(measured.get(fork.prompt_id, fork) for fork in forks)
    eligible = [item for item in qualification["repeatable_errors"]
                if all(fork.status == "semantic" and fork.replay_status == "matched"
                       for fork in forks if fork.question_id == item)]
    targets = eligible[:config["fit_limits"]["max_repeatable_error_items"]]
    selected = tuple(fork for fork in forks if fork.question_id in targets)
    solved = qualification["solved"]
    preserved = [*solved["target"][:config["fit_limits"]["max_solved_target_items"]],
                 *solved["positive"], *solved["preserve"]]
    diagnostic_items = [solved["positive"][0], solved["preserve"][0], *solved["target"], *solved["positive"][1:],
                        *solved["preserve"][1:]][:config["probe_design"]["trajectory_items"]]
    trajectories = tuple(protected_trajectory(runner, row, contract)
                         for row in practice["rows"] if row["item"] in diagnostic_items)
    diagnostics = []
    with torch.inference_mode():
        for chunk in _batches(trajectories, contract.batch_size):
            calls += 1
            for trajectory, logits in zip(chunk, trajectory_logits(runner, chunk), strict=True):
                diagnostics.append(trajectory_diagnostics(trajectory, logits, runner.tokenizer))
    return selected, targets, preserved, trajectories, {
        "forks": [fork.to_dict() for fork in forks], "target_items": targets,
        "preservation_items": preserved, "sampled_trajectory_items": diagnostic_items,
        "trajectories": [row.to_dict() for row in trajectories], "parent_trajectory_diagnostics": diagnostics,
        "resources": {**_resources(started, cpu), "forward_calls": calls},
    }


def _training_paths(runner, contract, practice, forks, preserved):
    target = {fork.prompt_id: Continuation(fork.prompt_token_ids, fork.preferred_continuation, None, True)
              for fork in forks}
    paths = []
    for row in practice["rows"]:
        if row["id"] in target:
            paths.append(target[row["id"]])
        elif row["item"] in preserved:
            trace = protected_trajectory(runner, row, contract)
            paths.append(Continuation(trace.prompt_token_ids, trace.token_ids, None, True))
    return tuple(paths)


def _materialize(base, out, arm, plan, patch, probe):
    guard = probe.get("endpoint", {}).get("guard", {})
    if not guard.get("passed") or not probe.get("restored_exactly"):
        raise ValueError("Only an endpoint-qualified, exactly restored live probe can be saved")
    candidate_path = out / "checkpoints" / arm
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": [plan.region.tensor_name],
              "max_entries": plan.limits.max_update_entries, "max_fraction": .01,
              "max_abs_delta": plan.limits.max_update_norm, "max_rank": plan.limits.max_rank}
    receipt = apply_safetensors_patch(base, patch, policy, candidate_path)
    candidate = load_checkpoint(candidate_path)
    parent = load_checkpoint(base)
    tensor = plan.region.tensor_name
    delta = _to_float64(candidate.read_raw(tensor), plan.region.dtype) - _to_float64(
        parent.read_raw(tensor), plan.region.dtype)
    count, norm = int(np.count_nonzero(delta)), float(np.linalg.norm(delta))
    digest = candidate.tensor_sha256(tensor)
    if (receipt["base_sha256"] != plan.checkpoint_sha256
            or receipt["candidate_sha256"] != candidate.checkpoint_sha256()
            or set(receipt["tensor_receipts"]) != {tensor}
            or receipt["tensor_receipts"][tensor]["candidate_sha256"] != digest
            or digest != probe["candidate_tensor_sha256"] or count != receipt["changed_entries"]
            or count != probe["changed_entries"] or not np.isclose(norm, probe["realized_update_norm"], rtol=0, atol=1e-10)):
        raise ValueError("Materialized checkpoint differs from the stored-precision live probe")
    json_dump_atomic(out / "patches" / f"{arm}-receipt.json", receipt)
    json_dump_atomic(out / "patches" / f"{arm}-policy.json", policy)
    json_dump_atomic(out / "patches" / f"{arm}-patch.json", patch)
    return {"checkpoint": candidate_path, "checkpoint_sha256": receipt["candidate_sha256"],
            "realized_update_norm": norm, "changed_entries": count, "tensor_sha256": digest}


def _fidelity(arms):
    observations = []
    for arm, record in arms.items():
        probe = record.get("probe", {})
        guard = probe.get("endpoint", {}).get("guard")
        if guard is None:
            continue
        whole = probe.get("margin", {})
        fork = probe.get("fork", {})
        observations.append({
            "arm": arm, "target_item_gain": guard["target_item_gain"],
            "regressions": len(guard["prohibited_regressions"]),
            "lost_validity": len(guard["lost_valid_presentations"]), "practice_pass": guard["passed"],
            "whole_answer_loss_gain": whole.get("surrogate_gain"), "fork_measurement": fork,
        })
    return observations


def _fit(base, adapter, runner, capabilities, practice_rows, practice, qualification, config, out):
    from weightbridge.adaptation.fork_update import (
        fit_lora, measure_probe_reference, prepare_probe_reference, probe_update, scan_fork_regions,
    )

    contract = endpoint(config)
    design = config["probe_design"]
    forks, targets, preserved, trajectories, diagnosis = _diagnose(runner, contract, practice, qualification, config)
    runner = endpoint_runner(runner, contract)
    json_dump_atomic(out / "diagnosis.json", diagnosis)
    print(f"Diagnosis: {len(forks)} matched target forks across {len(targets)} repeatable errors", flush=True)
    if not targets:
        return {}, {}, diagnosis, None
    fit = build_decimal_fit_rows(runner, practice, contract, target_items=targets, preservation_items=preserved,
                                 max_rows=config["fit_limits"]["max_rows"], mode="semantic")
    json_dump_atomic(out / "fit-rows.json", {"rows": [asdict(row) for row in fit]})
    budget = CandidateEvaluationBudget(contract)
    arms, candidates, seen, by_digest = {}, {}, set(), {}
    fixed = design["fixed_layer"]
    plans = {layer: _plan(config, capabilities, layer) for layer in design["layers"]}
    scan_plans = [_plan(config, capabilities, layer, scanning=True) for layer in design["layers"]]
    dose = design["dose"]
    memory = plans[fixed].limits.max_memory_bytes
    reference = prepare_probe_reference(adapter, runner, forks, trajectories, batch_size=contract.batch_size,
                                        max_memory_bytes=memory)
    json_dump_atomic(out / "reference-resources.json", reference.resources)

    def evaluate():
        candidate = measure_endpoint(runner, practice_rows, contract, candidate_budget=budget)
        guard = compare_endpoint(practice, candidate, contract, min_repeatable_corrections=1)
        corrections = sorted(set(guard["corrected_repeatable_target_items"]) & qualification["repeatable_errors"].keys())
        guard["corrected_repeatable_negative_items"] = corrections
        guard["passed"] = bool(guard["passed"] and corrections)
        return {"guard": guard, "measurement": candidate}

    def record(arm, plan, patch, probe, diagnostics=None):
        result = {"layer": plan.region.layer_index, "plan": plan.record(), "probe": probe}
        digest = probe.get("candidate_tensor_sha256")
        key = plan.region.tensor_name, digest
        if digest and key in by_digest:
            result["duplicate_of"] = by_digest[key]
        elif digest and probe.get("endpoint"):
            by_digest[key] = arm
        if diagnostics is not None:
            result["diagnostics"] = diagnostics
        if patch is not None:
            candidates[arm] = _materialize(base, out, arm, plan, patch, probe)
            result["checkpoint"] = {key: value for key, value in candidates[arm].items() if key != "checkpoint"}
        arms[arm] = result
        json_dump_atomic(out / f"arm-{arm}.json", result)
        guard = probe.get("endpoint", {}).get("guard", {})
        print(f"Arm {arm}: {probe.get('reason')}, practice gain {guard.get('target_item_gain')}", flush=True)

    print("Arm A: fitting whole-answer comparator", flush=True)
    comparator_diagnostics = {}

    def evaluate_comparator():
        comparator_diagnostics.update(measure_probe_reference(runner, reference, batch_size=contract.batch_size,
                                                               max_memory_bytes=memory))
        return evaluate()

    patches, evidence = fit_endpoint_probes(
        adapter, runner, plans[fixed], [row.to_solver_row() for row in fit], evaluate_candidate=evaluate_comparator,
        baseline_endpoint=practice, doses=(dose,), max_input_tokens=contract.max_input_tokens,
        batch_size=contract.batch_size,
    )
    comparator = evidence["probes"][0] if evidence["probes"] else {"reason": evidence["reason"]}
    if comparator_diagnostics:
        comparator.update(fork=comparator_diagnostics["fork"], trajectory=comparator_diagnostics["trajectory"],
                          resources=comparator_diagnostics["resources"])
    if comparator.get("endpoint"):
        seen.add((plans[fixed].region.tensor_name, comparator["candidate_tensor_sha256"]))
    record("A", plans[fixed], patches.get("dose-1"), comparator,
           {key: value for key, value in evidence.items() if key != "probes"})

    print("Scanning three authorized affine regions", flush=True)
    selected, directions, scan = scan_fork_regions(
        adapter, runner, scan_plans, forks, trajectories, radius=dose, batch_size=contract.batch_size,
        margin_floor=design["trajectory_margin_floor"], max_constraints=design["trajectory_constraints"],
    )
    json_dump_atomic(out / "scan.json", scan)
    for arm, layer, direction in (("B", fixed, "fork"), ("C", selected, "fork"), ("D", selected, "protected")):
        if layer is None:
            record(arm, plans[fixed], None, {"reason": "no_reachable_region"})
            continue
        plan = plans[layer]
        update = directions.get(layer, {}).get(direction)
        if update is None:
            record(arm, plan, None, {"reason": "no_preservation_compatible_direction" if arm == "D" else "no_local_direction"})
            continue
        patch, probe = probe_update(adapter, runner, plan, update, radius=dose, seen=seen, evaluate_candidate=evaluate,
                                    forks=forks, trajectories=trajectories, baseline_endpoint=practice,
                                    batch_size=contract.batch_size, reference=reference)
        record(arm, plan, patch, probe)

    print("Arm E: training full-affine rank-four Adam baseline", flush=True)
    plan = _plan(config, capabilities, fixed, conventional=True)
    settings = {key: config["conventional"][key] for key in ("rank", "steps", "lr", "seed")}
    training = _training_paths(runner, contract, practice, forks, preserved)
    update, evidence = fit_lora(adapter, runner, plan, training, batch_size=contract.batch_size, **settings)
    if update is None:
        record("E", plan, None, {"reason": evidence["reason"]}, evidence)
    else:
        patch, probe = probe_update(adapter, runner, plan, update, radius=None, seen=seen, evaluate_candidate=evaluate,
                                    forks=forks, trajectories=trajectories, baseline_endpoint=practice,
                                    batch_size=contract.batch_size, reference=reference)
        record("E", plan, patch, probe, evidence)
    return arms, candidates, diagnosis, scan


def run(base, material, out, practice_path, selection_base=None):
    import torch
    import transformers

    started, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    sources, manifest_sha = source_hashes(), file_hash(material / "manifest.json")
    if out.resolve().is_relative_to(ROOT):
        raise ValueError("Run in a private host workspace and archive public evidence afterward")
    practice_rows, practice, parent_record = _read_practice(practice_path, material, manifest, config)
    qualification = parent_record["qualification"]
    out.mkdir(parents=True, exist_ok=False)
    json_dump_atomic(out / "practice.json", parent_record)
    baseline = config["checkpoint_sha256"]
    arms, candidates, diagnosis, scan = {}, {}, {}, None
    if qualification["disposition"] == "PASS":
        with _open(base, config, baseline, gradient_probe=True) as (adapter, runner, capabilities):
            if Capability.COMPUTE_GRADIENTS in capabilities.capabilities:
                arms, candidates, diagnosis, scan = _fit(base, adapter, runner, capabilities, practice_rows,
                                                          practice, qualification, config, out)
        del runner
    if sources != source_hashes() or manifest_sha != file_hash(material / "manifest.json"):
        raise ValueError("Experiment sources or frozen material changed during fitting")
    if load_checkpoint(base).checkpoint_sha256() != baseline:
        raise ValueError("Original checkpoint changed during fitting")
    # The practice fitter has finished before the verifier receives the optional selection-parent file.
    assessment = assess(base, material, out, candidates, selection_base)
    observations = _fidelity(arms)
    decision = decide_fork_action(
        measurement_valid=all(value for gate, value in qualification["gates"].items() if gate != "repeatable_errors"),
        persistent_errors=len(qualification["repeatable_errors"]),
        diagnosis_valid=bool(diagnosis.get("target_items")),
        reachable=scan["selected_layer"] is not None if scan else None,
        preservation_conflict=bool(observations) and all(row["regressions"] for row in observations),
        excessive_radius=any(row["probe"].get("reason") == "realized_norm_exceeds_budget" for row in arms.values()),
        surrogate_mismatch=any((row["whole_answer_loss_gain"] or 0) > 0 and row["target_item_gain"] <= 0
                               for row in observations),
        practice_pass=bool(candidates), selection_pass=bool(assessment["winner"]) if candidates else None,
    ).record()
    if assessment["winner"]:
        reason, disposition = "awaiting_host_hidden_verification", "HOLD"
    elif candidates:
        reason, disposition = "independent_selection_gate", "FAIL"
    elif qualification["disposition"] != "PASS":
        reason, disposition = "parent_practice_qualification", "HOLD"
    elif not diagnosis:
        reason, disposition = "gradient_capability_unavailable", "HOLD"
    elif not diagnosis["target_items"]:
        reason, disposition = "no_aligned_matched_forks", "HOLD"
    else:
        reason, disposition = "no_endpoint_qualified_probe", "HOLD"
    report = {
        "schema": "weightbridge.phase_iiii.causal_fork_report.v1", "base_sha256": baseline,
        "manifest_sha256": manifest_sha, "sources": sources, "qualification": qualification,
        "single_cycle": disposition, "reason": reason, "hidden_opened": False,
        "selection": assessment, "meta": decision, "fidelity": observations,
        "arms": {key: {name: record[name] for name in ("layer", "duplicate_of", "checkpoint") if name in record}
                 | {"reason": record["probe"].get("reason")} for key, record in arms.items()},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "torch": torch.__version__, "transformers": transformers.__version__},
        "resources": {"host": _resources(started, cpu), "practice_parent": parent_record["resources"]},
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
