"""Native-context remeasurement and endpoint guards for joint fork witnesses."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .endpoint_update import _live_probe
from .fork_update import _check_fork_identity, _check_memory, _first_divergences, _resources, _usable_forks
from .generative_fork import fork_statistics, next_token_logits
from .jspace_update import measure_update_margins
from .universal_update import _patch


def measure_native_forks(runner, forks, replay_batches, *, max_forward_calls, max_memory_bytes):
    """Measure selected native decisions in their exact original diagnostic batches.

    Nonselected rows reproduce the original padding/kernel context. They receive
    no target weight and are never silently substituted for a selected decision.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    forks = tuple(forks)
    contexts = tuple(tuple(batch) for batch in replay_batches)
    if (not forks or len(forks) > 32 or len(_usable_forks(forks)) != len(forks)
            or len({fork.prompt_id for fork in forks}) != len(forks)
            or not contexts or len(contexts) > 64
            or any(not 1 <= len(batch) <= 16 for batch in contexts)):
        raise ValueError("Joint native measurement requires bounded distinct replay-qualified forks")
    selected = {fork.prompt_id: fork for fork in forks}
    flat = tuple(fork for batch in contexts for fork in batch)
    if len({fork.prompt_id for fork in flat}) != len(flat):
        raise ValueError("Native diagnostic contexts must not duplicate presentations")
    lookup = {fork.prompt_id: fork for fork in flat}
    for fork in forks:
        context = lookup.get(fork.prompt_id)
        if (context is None or tuple(context.prefix_ids) != tuple(fork.prefix_ids)
                or tuple(context.accepted_token_ids) != tuple(fork.accepted_token_ids)):
            raise ValueError("Native context changed or omitted a required target decision")
    batches = [batch for batch in contexts if any(fork.prompt_id in selected for fork in batch)]
    if len(batches) > max_forward_calls:
        raise ValueError("Native fork remeasurement exceeds its explicit forward budget")
    rows, packing = {}, []
    with torch.inference_mode():
        for batch in batches:
            stats = fork_statistics(next_token_logits(runner, [fork.prefix_ids for fork in batch]),
                [fork.accepted_token_ids for fork in batch], actual_token_ids=[fork.actual_token_id for fork in batch])
            winners = stats.predicted_ids.detach().tolist()
            rivals = stats.strongest_rival_ids.detach().tolist()
            margins = stats.decision_margins.detach().tolist()
            packing.append({"prompt_ids": [fork.prompt_id for fork in batch],
                            "prefix_lengths": [len(fork.prefix_ids) for fork in batch]})
            for fork, winner, rival, margin in zip(batch, winners, rivals, margins, strict=True):
                if fork.prompt_id in selected:
                    rows[fork.prompt_id] = {"prompt_id": fork.prompt_id, "measured_margin": float(margin),
                        "greedy_token_id": winner, "strongest_rejected_token_id": rival,
                        "accepted_token_ids": list(fork.accepted_token_ids),
                        "accepted_token_selected": winner in fork.accepted_token_ids,
                        "parent_actual_token_id": fork.actual_token_id,
                        "first_fork_position": len(fork.shared_token_ids)}
            _check_memory(max_memory_bytes)
    return {"schema": "weightbridge.native_joint_margins.v1", "rows": [rows[f.prompt_id] for f in forks],
            "original_context_batches": packing,
            "context_only_rows_have_target_weight": False,
            "resources": _resources(wall, cpu, len(batches), 0)}


def measure_stored_joint(space, preview, runner, forks, replay_batches, pool, anchors, *,
                         target_floor, preserve_floor, batch_size, max_forward_calls, max_memory_bytes):
    """Remeasure both target and protection on one actual stored tensor state."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    with torch.inference_mode(), _live_probe(*space.probe_arguments(preview), torch):
        targets = measure_native_forks(runner, forks, replay_batches,
            max_forward_calls=max_forward_calls, max_memory_bytes=max_memory_bytes)
        observed, protected = measure_update_margins(runner, pool, anchors,
            batch_size=batch_size, max_memory_bytes=max_memory_bytes)
    target_pass = all(row["accepted_token_selected"] and row["measured_margin"] >= target_floor
                      for row in targets["rows"])
    preserve_pass = all(observed[anchor.key] > preserve_floor
                        and row["replay_argmax_token_id"] == anchor.token_id
                        for anchor, row in zip(anchors, protected["rows"], strict=True))
    return {"schema": "weightbridge.stored_joint_measurement.v1", "tensor_sha256": preview["tensor_sha256"],
            "target_floor": target_floor, "preservation_floor": preserve_floor,
            "targets": targets, "protected": protected, "target_pass": target_pass,
            "preservation_pass": preserve_pass, "passed": bool(target_pass and preserve_pass),
            "restored_exactly": True,
            "resources": _resources(wall, cpu,
                targets["resources"]["forward_calls"] + protected["resources"]["forward_calls"], 0)}


def probe_joint_generation(space, preview, runner, plan, forks, replay_batches, pool, anchors, *,
                           target_floor, preserve_floor, batch_size, seen, evaluate_candidate,
                           baseline_endpoint):
    """Generate only after the same stored state passes every individual margin.

    The full behavioral endpoint remains authoritative. This helper uses the
    established patch format and never admits or promotes a checkpoint itself.
    """
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    _check_fork_identity(forks, plan.checkpoint_sha256)
    key = (plan.region.tensor_name, preview["tensor_sha256"])
    evidence = {"schema": "weightbridge.joint_probe.v1", "method": "joint_fork_native_context", "rank": 1,
        "tensor_name": plan.region.tensor_name, "layer_index": plan.region.layer_index,
        "candidate_tensor_sha256": preview["tensor_sha256"], "restored_exactly": True,
        "realized_update_norm": preview["realized_norm"], "changed_entries": preview["changed_entries"],
        "proposed_norm": preview["proposed_norm"], "touched_entries": preview["delta"].size,
        "max_abs_delta": float(np.max(np.abs(preview["delta"]))) }
    if key in seen:
        return None, {**evidence, "reason": "duplicate_stored_candidate", "resources": _resources(wall, cpu, 0, 0)}
    if (not preview["changed_entries"] or not 0 < preview["realized_norm"] <= plan.limits.max_update_norm
            or preview["changed_entries"] > plan.limits.max_update_entries):
        raise ValueError("Joint generation proposal violates the stored numerical policy")
    measured = measure_stored_joint(space, preview, runner, forks, replay_batches, pool, anchors,
        target_floor=target_floor, preserve_floor=preserve_floor, batch_size=batch_size,
        max_forward_calls=plan.limits.max_forward_calls, max_memory_bytes=plan.limits.max_memory_bytes)
    if not measured["passed"]:
        return None, {**evidence, "reason": "stored_joint_margin_refusal", "native_measurement": measured,
                      "failure_class": "LINEAR_FEASIBLE_BF16_FAILURE",
                      "resources": _resources(wall, cpu, measured["resources"]["forward_calls"], 0)}
    with torch.inference_mode(), _live_probe(*space.probe_arguments(preview), torch):
        endpoint = evaluate_candidate()
        guard = endpoint.get("guard", {})
        if type(guard.get("passed")) is not bool:
            raise ValueError("Joint endpoint evaluator must return a boolean admission decision")
        if guard["passed"] and (type(guard.get("target_item_gain")) is not int
                or guard["target_item_gain"] < 1 or guard.get("prohibited_regressions") or guard.get("lost_valid_presentations")):
            raise ValueError("Joint endpoint admitted a candidate without positive gain and preservation")
    seen.add(key)
    patch = _patch(plan, hashlib.sha256(space.parent_bytes).hexdigest(), list(space.columns), preview["stored"]) if guard["passed"] else None
    fitted_ids = {fork.prompt_id for fork in forks}
    measured_rows = {row["id"]: row for row in endpoint["measurement"].get("rows", ())}
    incomplete = sorted(prompt for prompt in fitted_ids if prompt in measured_rows and not measured_rows[prompt]["correct"])
    mechanism = "FORK_CORRECTED_COMPLETION_WRONG" if incomplete else (
        "FITTED_ERROR_RESPONSES_COMPLETE" if fitted_ids <= set(measured_rows) else "NO_GENERATION_EVIDENCE")
    failure_class = ("FORK_CORRECTED_COMPLETION_WRONG" if incomplete else
        "INCOMPLETE_GENERATION_EVIDENCE" if mechanism == "NO_GENERATION_EVIDENCE" else
        "TARGET_PRESERVATION_CONFLICT" if guard.get("prohibited_regressions") else
        "VALID_OUTPUT_REGRESSION" if guard.get("lost_valid_presentations") else
        "PRACTICE_BEHAVIOR_PASS" if patch else "PRACTICE_ENDPOINT_GATE_FAILURE")
    return patch, {**evidence, "reason": "endpoint_pass" if patch else "endpoint_rejected",
        "failure_class": failure_class,
        "practice_admission": bool(guard["passed"]), "mechanism_status": mechanism,
        "fitted_error_responses_still_wrong": incomplete,
        "native_measurement": measured, "endpoint": endpoint,
        "first_divergences": _first_divergences(baseline_endpoint, endpoint["measurement"], runner.tokenizer),
        "resources": _resources(wall, cpu, measured["resources"]["forward_calls"], 0)}
