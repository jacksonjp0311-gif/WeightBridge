"""Objective and scaling comparisons in fixed projected rank-one coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from math import degrees, isfinite
from typing import Callable, Mapping

import numpy as np

from .semantic_consensus import gradient_agreement


def _cosine(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.clip(first @ second / denominator, -1., 1.)) if denominator else None


def _unit_rows(rows):
    norms = np.linalg.norm(rows, axis=1)
    result = np.zeros_like(rows)
    active = norms > 0
    result[active] = rows[active] / norms[active, None]
    return result, norms


@dataclass(frozen=True)
class FactorialDirections:
    directions: dict[str, np.ndarray]
    evidence: dict

    def record(self):
        return self.evidence


def build_factorial_directions(item_ids, groups, ce_gradients, margin_gradients):
    """Average each semantic comparison once, including explicitly zero gradients.

    Both supplied matrices are corrective gradients. CE rows already contain the
    negative loss gradient, and margin rows contain the positive margin gradient.
    Every row must average its own authorized presentations before this function.
    UNIT normalizes gradients in the fixed-v left-factor coordinates G_i v.
    """
    ids, labels = tuple(item_ids), tuple(groups)
    ce = np.asarray(ce_gradients, dtype=np.float64)
    margin = np.asarray(margin_gradients, dtype=np.float64)
    if (not ids or len(ids) != len(labels) or len(set(ids)) != len(ids)
            or not all(isinstance(value, str) and value for value in (*ids, *labels))
            or ce.ndim != 2 or ce.shape[0] != len(ids) or not ce.shape[1]
            or margin.shape != ce.shape or not np.isfinite(ce).all() or not np.isfinite(margin).all()):
        raise ValueError("Expected finite CE and margin rows for the same unique semantic comparisons")
    normalized_ce, ce_norms = _unit_rows(ce)
    normalized_margin, margin_norms = _unit_rows(margin)
    if not np.isfinite(ce_norms).all() or not np.isfinite(margin_norms).all():
        raise ValueError("Corrective gradient norms must be finite")
    rows = {"A": ce, "B": normalized_ce, "C": margin, "D": normalized_margin}
    directions = {name: values.mean(axis=0) for name, values in rows.items()}
    group_members = {label: np.asarray([group == label for group in labels]) for label in dict.fromkeys(labels)}
    counts = {label: int(members.sum()) for label, members in group_members.items()}
    arms = {}
    for name, values in rows.items():
        direction = directions[name]
        square = float(direction @ direction)
        contributions = values / len(ids)
        group_contributions = {label: contributions[members].sum(axis=0)
                               for label, members in group_members.items()}
        total_group_norm = sum(float(np.linalg.norm(value)) for value in group_contributions.values())
        equal_group = np.stack([values[members].mean(axis=0) for members in group_members.values()]).mean(axis=0)
        raw_norms = ce_norms if name in ("A", "B") else margin_norms
        scaling = np.ones(len(ids))
        if name in ("B", "D"):
            scaling = np.divide(1., raw_norms, out=np.zeros_like(raw_norms), where=raw_norms > 0)
        arms[name] = {
            "objective": "CE" if name in ("A", "B") else "MARGIN",
            "scaling": "RAW" if name in ("A", "C") else "UNIT",
            "aggregate": direction.tolist(),
            "aggregate_norm": float(np.linalg.norm(direction)),
            "item_weight": 1. / len(ids),
            "item_scale": dict(zip(ids, scaling.tolist(), strict=True)),
            "item_contribution_norm": dict(zip(ids, np.linalg.norm(contributions, axis=1).tolist(), strict=True)),
            "item_signed_projection_share": {item: float(row @ direction / square) if square else None
                                             for item, row in zip(ids, contributions, strict=True)},
            "group_contribution": {label: {
                "vector": value.tolist(), "norm": float(np.linalg.norm(value)),
                "norm_fraction_before_cancellation": float(np.linalg.norm(value)) / total_group_norm
                    if total_group_norm else None,
                "signed_projection_share": float(value @ direction / square) if square else None,
            } for label, value in group_contributions.items()},
            "equal_group_mean": equal_group.tolist(),
            "equal_group_vs_item_mean_distance": float(np.linalg.norm(equal_group - direction)),
            "group_reweighting_present": len(set(counts.values())) > 1,
            "zero_item_ids": [item for item, norm in zip(ids, raw_norms, strict=True) if norm == 0],
        }
    angles = {}
    names = tuple(directions)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            cosine = _cosine(directions[first], directions[second])
            angles[f"{first}:{second}"] = {
                "cosine": cosine, "angle_degrees": degrees(np.arccos(cosine)) if cosine is not None else None}
    return FactorialDirections(directions, {
        "schema": "weightbridge.factorial_directions.v1",
        "normalization_space": "per-comparison fixed-right left-factor gradient G_i v",
        "item_ids": list(ids), "item_groups": list(labels),
        "group_item_counts": counts, "equal_group_counts": len(set(counts.values())) == 1,
        "projected_item_gradients": {"CE": ce.tolist(), "MARGIN": margin.tolist()},
        "ce": gradient_agreement(ids, labels, ce),
        "margin": gradient_agreement(ids, labels, margin), "angles": angles, "arms": arms,
    })


@dataclass(frozen=True)
class NormMatch:
    selected: dict | None
    evidence: dict

    def record(self):
        return self.evidence


def match_stored_norm(evaluate: Callable[[float], dict], *, intended_norm: float, maximum_norm: float,
                      relative_tolerance: float = .02, max_probes: int = 10, amplitude_limit: float | None = None):
    """Measure a bounded scalar search and return the best admissible stored state.

    The callback must recheck its solver constraints and rounded tensor for every
    amplitude. Feasibility search and quantization can be nonmonotonic. This
    procedure reports its measured best state without certifying that every
    possible amplitude has been excluded.
    """
    limit = maximum_norm if amplitude_limit is None else amplitude_limit
    if (not all(isfinite(value) and value > 0 for value in (intended_norm, maximum_norm, limit))
            or intended_norm > maximum_norm or not isfinite(relative_tolerance)
            or not 0 <= relative_tolerance <= 1 or type(max_probes) is not int or not 1 <= max_probes <= 64):
        raise ValueError("Norm matching needs a positive authorized target and bounded search")
    attempts, candidates, amplitudes, digests = [], [], set(), set()
    lower, upper = 0., float(limit)
    amplitude = min(float(intended_norm), upper)
    best = None
    for _ in range(max_probes):
        if amplitude in amplitudes or amplitude <= 0:
            break
        amplitudes.add(amplitude)
        observed = {**evaluate(amplitude), "amplitude": amplitude}
        for payload in ("delta", "update"):
            if isinstance(observed.get(payload), np.ndarray):
                observed[payload] = observed[payload].copy()
        norm = observed.get("realized_norm")
        if type(observed.get("feasible")) is not bool or norm is not None and (
                isinstance(norm, bool) or not isinstance(norm, (int, float)) or not isfinite(norm) or norm < 0):
            raise ValueError("Norm callback requires a boolean feasibility decision and finite stored norm")
        digest = observed.get("tensor_sha256")
        if digest is not None and (not isinstance(digest, str) or not digest):
            raise ValueError("Stored tensor identity must be a nonempty string")
        admissible = bool(observed["feasible"] and norm is not None and 0 < norm <= maximum_norm and digest)
        error = abs(norm - intended_norm) if norm is not None else None
        relative = error / intended_norm if error is not None else None
        duplicate = digest in digests if digest else False
        row = {key: value for key, value in observed.items() if key not in ("delta", "update")}
        row.update(amplitude=amplitude, admissible=admissible, duplicate_stored_state=duplicate,
                   norm_matching_error=error, relative_norm_error=relative)
        attempts.append(row)
        if digest:
            digests.add(digest)
        if admissible:
            candidates.append((error, amplitude, observed))
            best = min(candidates, key=lambda item: (item[0], item[1]))
            if relative <= relative_tolerance:
                break
        # The bracket is a bounded proposal heuristic. Every returned state has
        # its own measured constraint and precision result.
        if norm is not None and norm > intended_norm:
            upper = min(upper, amplitude)
        else:
            lower = max(lower, amplitude)
        if norm is not None and norm > 0:
            proposal = amplitude * intended_norm / norm
        else:
            proposal = upper
        proposal = float(np.clip(proposal, lower, upper))
        if proposal in amplitudes or proposal <= lower or proposal >= upper and upper in amplitudes:
            proposal = (lower + upper) / 2
        amplitude = proposal
    selected = best[2] if best else None
    selected_row = next((row for row in attempts if best and row["amplitude"] == best[1]), None)
    matched = bool(selected_row and selected_row["relative_norm_error"] <= relative_tolerance)
    return NormMatch(selected, {
        "schema": "weightbridge.stored_norm_match.v1", "status": "MATCHED" if matched else "UNMATCHED",
        "reason": "measured_state_within_tolerance" if matched else
                  "no_admissible_stored_state" if selected is None else "no_measured_state_within_tolerance",
        "intended_norm": intended_norm, "maximum_norm": maximum_norm, "amplitude_limit": limit,
        "relative_tolerance": relative_tolerance, "probe_budget": max_probes, "attempts": attempts,
        "unique_stored_states": len(digests),
        "unique_admissible_states": len({row[2]["tensor_sha256"] for row in candidates}),
        "selected": selected_row,
    })


def stored_update_cosines(updates: Mapping[str, np.ndarray]):
    """Compare actual rounded deltas in common tensor coordinates."""
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in updates.items()}
    if (not arrays or any(not isinstance(name, str) or not name for name in arrays)
            or len({value.shape for value in arrays.values()}) != 1
            or any(not value.size or not np.isfinite(value).all() or not isfinite(float(np.linalg.norm(value)))
                   for value in arrays.values())):
        raise ValueError("Stored updates must be finite arrays with shared tensor coordinates")
    names, result = tuple(arrays), {}
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            a, b = arrays[first].ravel(), arrays[second].ravel()
            result[f"{first}:{second}"] = {
                "cosine": _cosine(a, b), "difference_norm": float(np.linalg.norm(a - b)),
                "different_entries": int(np.count_nonzero(a != b)), "identical_stored_update": bool(np.array_equal(a, b)),
            }
    return result


def compare_realized_norms(states: Mapping[str, Mapping], *, intended_norm: float, relative_tolerance: float = .02):
    """Report pairwise and cohort range matching against the same intended norm."""
    if (not states or not isfinite(intended_norm) or intended_norm <= 0
            or not isfinite(relative_tolerance) or not 0 <= relative_tolerance <= 1):
        raise ValueError("Norm comparisons need a positive common target and a bounded tolerance")
    norms = {}
    for name, state in states.items():
        norm = state.get("realized_norm")
        if (not isinstance(name, str) or not name or isinstance(norm, bool)
                or not isinstance(norm, (int, float)) or not isfinite(norm) or norm < 0):
            raise ValueError("Each arm requires a finite nonnegative realized norm")
        norms[name] = float(norm)
    individual = {name: {"realized_norm": norm, "absolute_target_error": abs(norm - intended_norm),
                         "relative_target_error": abs(norm - intended_norm) / intended_norm,
                         "within_target_tolerance": abs(norm - intended_norm) / intended_norm <= relative_tolerance}
                  for name, norm in norms.items()}
    names, pairs = tuple(states), {}
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            difference = abs(norms[first] - norms[second])
            pair_match = difference / intended_norm <= relative_tolerance
            targets_match = all(individual[name]["within_target_tolerance"] for name in (first, second))
            pairs[f"{first}:{second}"] = {
                "absolute_norm_difference": difference, "relative_norm_difference": difference / intended_norm,
                "matched": bool(pair_match and targets_match),
                "same_stored_tensor": bool(states[first].get("tensor_sha256")
                                          and states[first].get("tensor_sha256") == states[second].get("tensor_sha256")),
            }
    span = max(norms.values()) - min(norms.values())
    cohort_match = span / intended_norm <= relative_tolerance and all(
        row["within_target_tolerance"] for row in individual.values())
    return {"schema": "weightbridge.realized_norm_comparison.v1", "intended_norm": intended_norm,
            "relative_tolerance": relative_tolerance, "individual": individual, "pairs": pairs,
            "cohort_range": span, "relative_cohort_range": span / intended_norm,
            "cohort_status": "MATCHED" if cohort_match else "UNMATCHED"}
