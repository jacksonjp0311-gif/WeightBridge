"""Stored-precision directional scans and independent residual measurements."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from math import isfinite
import time

import numpy as np

from ..patches.huggingface import _from_float64, _to_float64
from .counterexample_update import _prefix
from .endpoint_update import _live_probe
from .fork_update import _batches, _check_memory, _region, _resources
from .generative_fork import fork_statistics, next_token_logits, trajectory_logits


def _known_anchors(pool, anchors):
    anchors = tuple(anchors)
    known = {anchor.key: anchor for anchor in pool.eligible_anchors}
    if (len(known) != len(pool.eligible_anchors) or len({anchor.key for anchor in anchors}) != len(anchors)
            or any(known.get(anchor.key) != anchor for anchor in anchors)):
        raise ValueError("Expected distinct eligible parent trajectory decisions")
    return anchors


def _logit_records(anchors, values):
    stats = fork_statistics(values, [(anchor.token_id,) for anchor in anchors])
    return {anchor.key: {"margin": margin, "replay_argmax_token_id": winner, "strongest_rival_token_id": rival}
            for anchor, margin, winner, rival in zip(anchors, stats.decision_margins.tolist(),
                                                   stats.predicted_ids.tolist(), stats.strongest_rival_ids.tolist(), strict=True)}


def _all_trajectory_margins(runner, pool, anchors, *, batch_size, max_memory_bytes):
    """Replay all positions in batched trajectories using bounded native causal logits."""
    positions = defaultdict(list)
    for anchor in anchors:
        positions[anchor.prompt_id].append(anchor)
    paths = tuple(path for prompt, path in pool.trajectories.items() if prompt in positions)
    records, calls = {}, 0
    for batch in _batches(paths, batch_size):
        calls += 1
        for path, logits in zip(batch, trajectory_logits(runner, batch), strict=True):
            path_anchors = positions[path.prompt_id]
            if any(not 0 <= anchor.position < len(path.token_ids)
                   or path.token_ids[anchor.position] != anchor.token_id for anchor in path_anchors):
                raise ValueError("Decision does not match the original complete trajectory")
            records.update(_logit_records(path_anchors, logits[[anchor.position for anchor in path_anchors]]))
        _check_memory(max_memory_bytes)
    if len(records) != len(anchors):
        raise ValueError("Some preservation positions have no trajectory")
    return records, calls


def _position_margins(runner, pool, anchors, *, batch_size, max_memory_bytes):
    """Replay a small set of individual causal prefixes in the supplied order."""
    records, calls = {}, 0
    for batch in _batches(anchors, batch_size):
        calls += 1
        values = next_token_logits(runner, [_prefix(pool, anchor) for anchor in batch])
        records.update(_logit_records(batch, values))
        _check_memory(max_memory_bytes)
    return records, calls


def _round_preferred_direction(adapter, runner, plan, basis, radius, torch):
    limits = plan.limits
    if (plan.region is None or basis.checkpoint_sha256 != plan.checkpoint_sha256 or basis.tensor_name != plan.region.tensor_name
            or not isfinite(radius) or not 0 < radius <= limits.max_update_norm):
        raise ValueError("Directional measurement requires a bounded, checkpoint-matched affine basis")
    _check_memory(limits.max_memory_bytes)
    ref, module, raw = _region(adapter, runner, plan, torch)
    columns = basis.columns
    if (not columns or tuple(sorted(set(columns))) != columns or columns[0] < 0 or columns[-1] >= ref.shape[1]
            or len(columns) > plan.max_columns or ref.shape[0] * len(columns) > limits.max_update_entries
            or limits.max_rank < 1
            or basis.preferred_left.shape != (ref.shape[0],) or basis.right.shape != (len(columns),)
            or not np.isclose(np.linalg.norm(basis.preferred_left), 1.) or not np.isclose(np.linalg.norm(basis.right), 1.)):
        raise ValueError("Directional probe exceeds the authorized unit rank-one region")
    source = _to_float64(raw, ref.dtype).reshape(ref.shape)
    base_columns = source[:, columns]
    intended = radius * np.outer(basis.preferred_left, basis.right)
    stored = _to_float64(_from_float64(base_columns + intended, ref.dtype), ref.dtype).reshape(base_columns.shape)
    stored[stored == base_columns] = base_columns[stored == base_columns]
    change = stored - base_columns
    realized = float(np.linalg.norm(change))
    entries = int(np.count_nonzero(change))
    record = {"nominal_radius": radius, "realized_update_norm": realized, "changed_entries": entries,
              "touched_entries": int(change.size), "stored_direction_cosine": float((intended * change).sum() / (radius * realized))
              if realized > 0 and isfinite(realized) else None, "restored_exactly": True}
    if not entries or not isfinite(realized) or realized > limits.max_update_norm:
        return None, {**record, "reason": "no_stored_change" if not entries else "realized_norm_exceeds_budget"}
    source[:, columns] = stored
    expected = _from_float64(source, ref.dtype)
    return (module.weight, columns, stored, expected, raw), {
        **record, "candidate_tensor_sha256": hashlib.sha256(expected).hexdigest(), "reason": "measured"}


def _row(anchor):
    return {"item": anchor.item, "prompt_id": anchor.prompt_id, "position": anchor.position, "group": anchor.group,
            "wording": anchor.wording, "parent_token_id": anchor.token_id}


def measure_directional_secant(adapter, runner, plan, basis, pool, *, radius, batch_size=8):
    """Measure one BF16 finite secant across every eligible correct practice position."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    anchors = _known_anchors(pool, pool.eligible_anchors)
    if len(pool.trajectories) > 128:
        raise ValueError("Directional scan exceeds its preservation practice pool budget")
    if not anchors:
        return {}, {"schema": "weightbridge.jspace_directional_secant.v1", "reason": "no_eligible_positions",
                    "position_count": 0, "prompt_count": 0, "resources": _resources(wall, cpu, 0, 0)}
    prompt_ids = tuple(dict.fromkeys(anchor.prompt_id for anchor in anchors))
    batch_count = len(tuple(_batches(prompt_ids, batch_size)))
    if 2 * batch_count > plan.limits.max_forward_calls:
        raise ValueError("Directional scan exceeds its forward budget")
    probe, evidence = _round_preferred_direction(adapter, runner, plan, basis, radius, torch)
    evidence.update(schema="weightbridge.jspace_directional_secant.v1", slope_denominator="nominal_radius",
                    position_count=len(anchors), prompt_count=len(prompt_ids))
    if probe is None:
        return {}, {**evidence, "resources": _resources(wall, cpu, 0, 0)}
    settings = {"batch_size": batch_size, "max_memory_bytes": plan.limits.max_memory_bytes}
    with torch.inference_mode():
        before, before_calls = _all_trajectory_margins(runner, pool, anchors, **settings)
        with _live_probe(*probe, torch):
            after, after_calls = _all_trajectory_margins(runner, pool, anchors, **settings)
    slopes, rows = {}, []
    for anchor in anchors:
        prior, candidate = before[anchor.key], after[anchor.key]
        slope = (candidate["margin"] - prior["margin"]) / radius
        slopes[anchor.key] = slope
        rows.append({**_row(anchor), "parent_pool_margin": anchor.margin, "parent_replay": prior,
                     "candidate_replay": candidate, "slope": slope})
    return slopes, {**evidence, "rows": rows,
                    "parent_replay_mismatches": sum(row["parent_replay"]["replay_argmax_token_id"] != row["parent_token_id"]
                                                    for row in rows),
                    "candidate_changed_argmax": sum(row["candidate_replay"]["replay_argmax_token_id"]
                                                   != row["parent_replay"]["replay_argmax_token_id"] for row in rows),
                    "resources": _resources(wall, cpu, before_calls + after_calls, 0)}


def measure_update_margins(runner, pool, anchors, *, batch_size, max_memory_bytes):
    """Measure an active candidate on original prefixes without changing its weights."""
    import torch

    wall, cpu = time.perf_counter(), time.process_time()
    anchors = _known_anchors(pool, anchors)
    if len(anchors) > 64:
        raise ValueError("Candidate local margin probe exceeds its position budget")
    _check_memory(max_memory_bytes)
    with torch.inference_mode():
        records, calls = _position_margins(runner, pool, anchors, batch_size=batch_size, max_memory_bytes=max_memory_bytes)
    return {key: record["margin"] for key, record in records.items()}, {
        "schema": "weightbridge.jspace_candidate_margins.v1",
        "rows": [{**_row(anchor), **records[anchor.key]} for anchor in anchors],
        "resources": _resources(wall, cpu, calls, 0),
    }


def calibrate_direction_residuals(adapter, runner, plan, basis, pool, calibration, validation, sensitivities, *,
                                  radius, batch_size=8):
    """Measure individual-gradient residuals on disjoint calibration and validation items."""
    import torch
    from .robust_jspace import MarginObservation

    wall, cpu = time.perf_counter(), time.process_time()
    calibration, validation = tuple(calibration), tuple(validation)
    anchors = _known_anchors(pool, (*calibration, *validation))
    if (not anchors or {anchor.item for anchor in calibration} & {anchor.item for anchor in validation}
            or len(anchors) > 64 or any(anchor.key not in sensitivities or sensitivities[anchor.key].anchor != anchor for anchor in anchors)):
        raise ValueError("Residual partitions require distinct semantic items with individual sensitivities")
    # Validate the batch size before computing the runtime allowance.
    batches = tuple(_batches(anchors, batch_size))
    if 2 * len(batches) > plan.limits.max_forward_calls:
        raise ValueError("Residual measurement exceeds its forward budget")
    probe, evidence = _round_preferred_direction(adapter, runner, plan, basis, radius, torch)
    evidence.update(schema="weightbridge.jspace_directional_residuals.v1", calibration=[], validation=[])
    if probe is None:
        return (), (), {**evidence, "resources": _resources(wall, cpu, 0, 0)}
    settings = {"batch_size": batch_size, "max_memory_bytes": plan.limits.max_memory_bytes}
    with torch.inference_mode():
        before, before_calls = _position_margins(runner, pool, anchors, **settings)
        with _live_probe(*probe, torch):
            after, after_calls = _position_margins(runner, pool, anchors, **settings)
    observations = {}
    for name, partition in (("calibration", calibration), ("validation", validation)):
        rows = []
        for anchor in partition:
            sensitivity = sensitivities[anchor.key]
            prior, candidate = before[anchor.key], after[anchor.key]
            slope = float(sensitivity.gradient @ basis.preferred_left)
            predicted = prior["margin"] + radius * slope
            measured = candidate["margin"]
            rows.append(MarginObservation(anchor.item, anchor.prompt_id, anchor.position, predicted, measured, group=anchor.group))
            evidence[name].append({**_row(anchor), "gradient_parent_margin": sensitivity.margin, "slope": slope,
                                   "parent_replay": prior, "candidate_replay": candidate, "predicted_margin": predicted,
                                   "measured_margin": measured, "residual": measured - predicted})
        observations[name] = tuple(rows)
    return observations["calibration"], observations["validation"], {
        **evidence, "resources": _resources(wall, cpu, before_calls + after_calls, 0),
    }
