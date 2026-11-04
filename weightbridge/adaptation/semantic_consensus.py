"""Bounded first-order consensus in a shared rank-one update coordinate system."""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, isfinite

import numpy as np

from .preservation_geometry import solve_inequalities


def _group_gradients(item_ids, groups, corrective_gradients, required_groups):
    ids, labels, expected = tuple(item_ids), tuple(groups), tuple(required_groups)
    rows = np.asarray(corrective_gradients, dtype=np.float64)
    if (not ids or len(ids) != len(labels)
            or not all(isinstance(value, str) and value for value in (*ids, *labels, *expected))
            or len(set(ids)) != len(ids)
            or rows.ndim != 2 or rows.shape[0] != len(ids) or not rows.shape[1] or not np.isfinite(rows).all()):
        raise ValueError("Expected one finite projected corrective gradient per unique underlying item")
    norms = np.linalg.norm(rows, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Projected corrective gradient norms must be finite")
    active = norms > 0
    unit_rows = np.zeros_like(rows)
    unit_rows[active] = rows[active] / norms[active, None]
    by_group = {group: np.array([label == group for label in labels]) for group in dict.fromkeys((*expected, *labels))}
    supported = tuple(group for group, members in by_group.items() if np.any(members & active))
    means = np.array([unit_rows[by_group[group]].mean(axis=0) for group in supported]).reshape(-1, rows.shape[1])
    return ids, labels, rows, norms, unit_rows, by_group, supported, means


def _cosine_matrix(rows):
    norms = np.linalg.norm(rows, axis=1)
    active = norms > 0
    normalized = np.zeros_like(rows)
    normalized[active] = rows[active] / norms[active, None]
    similarities = np.clip(normalized @ normalized.T, -1., 1.)
    return [[float(value) if active[i] and active[j] else None for j, value in enumerate(row)]
            for i, row in enumerate(similarities)]


def gradient_agreement(item_ids, groups, corrective_gradients, *, required_groups=()) -> dict:
    """Report cosines in the supplied projection; zero-vector cosines are undefined."""
    ids, labels, _, norms, unit_rows, by_group, supported, means = _group_gradients(
        item_ids, groups, corrective_gradients, required_groups)
    return {
        "schema": "weightbridge.projected_gradient_agreement.v1",
        "item_ids": list(ids),
        "item_groups": list(labels),
        "item_norms": norms.tolist(),
        "item_cosines": _cosine_matrix(unit_rows),
        "groups": list(by_group),
        "group_item_counts": {group: int(members.sum()) for group, members in by_group.items()},
        "group_nonzero_counts": {group: int(np.count_nonzero(members & (norms > 0)))
                                 for group, members in by_group.items()},
        "supported_groups": list(supported),
        "missing_groups": [group for group, members in by_group.items() if not members.any()],
        "zero_gradient_groups": [group for group, members in by_group.items() if members.any() and group not in supported],
        "group_cosine_order": list(supported),
        "group_cosines": _cosine_matrix(means),
        "group_mean_norms": dict(zip(supported, np.linalg.norm(means, axis=1).tolist(), strict=True)),
    }


@dataclass(frozen=True)
class ConsensusDirection:
    update: np.ndarray | None
    evidence: dict

    def record(self) -> dict:
        return self.evidence


@dataclass(frozen=True)
class ConsensusStudy:
    historical: ConsensusDirection
    weighted_mean: ConsensusDirection
    maximin: ConsensusDirection
    agreement: dict

    def record(self) -> dict:
        directions = {"historical": self.historical, "weighted_mean": self.weighted_mean, "maximin": self.maximin}
        angles = {}
        for first, second in (("historical", "weighted_mean"), ("historical", "maximin"), ("weighted_mean", "maximin")):
            a, b = directions[first].update, directions[second].update
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) if a is not None and b is not None else 0.
            cosine = float(np.clip(a @ b / denominator, -1., 1.)) if denominator > 0 else None
            angles[f"{first}:{second}"] = {"cosine": cosine, "angle_degrees": degrees(np.arccos(cosine))
                                           if cosine is not None else None}
        return {"schema": "weightbridge.gradient_consensus.v1", "agreement": self.agreement,
                "directions": {name: direction.record() for name, direction in directions.items()}, "angles": angles}


def _unconstrained_upper(groups, means, radius):
    # For any convex combination w of group rows, min_s a_s @ u <= w @ u <= radius * ||w||.
    combinations = [(row, {group: 1.}) for group, row in zip(groups, means, strict=True)]
    combinations.append((means.mean(axis=0), dict.fromkeys(groups, 1 / len(groups))))
    for index, a in enumerate(means):
        for other in range(index + 1, len(groups)):
            b = means[other]
            change = b - a
            square_norm = float(change @ change)
            if square_norm > 0:
                weight = float(np.clip(-(a @ change) / square_norm, 0., 1.))
                combinations.append((a + weight * change, {groups[index]: 1 - weight, groups[other]: weight}))
    row, weights = min(combinations, key=lambda candidate: np.linalg.norm(candidate[0]))
    return float(radius * np.linalg.norm(row)), {group: weight for group, weight in weights.items() if weight > 0}


def solve_consensus(item_ids, groups, corrective_gradients, preservation_gradients, protected_margins, *,
                    radius: float, floor=0.0, historical_direction=None, required_groups=(),
                    max_search_steps: int = 16, max_projection_iterations: int = 256,
                    tolerance: float = 1e-9) -> ConsensusStudy:
    """Compare shared-geometry directions under the same sampled margin constraints.

    Item rows are gradients of *increasing* the corrective margin. Normalize each item
    before taking group means. The historical preferred vector may instead come from
    the negative fork-loss gradient, with the same fixed right factor. Both mean
    directions use the inherited nearest-feasible projection. Maximin searches for a
    witness by adding one halfspace per supported group at a shared improvement level.
    A failed bounded projection is not a certificate of infeasibility.
    """
    required_groups = tuple(required_groups)
    ids, labels, raw, _, _, members, supported, means = _group_gradients(
        item_ids, groups, corrective_gradients, required_groups)
    if (not isfinite(radius) or radius <= 0 or not isfinite(tolerance) or tolerance <= 0
            or type(max_search_steps) is not int or not 1 <= max_search_steps <= 64
            or type(max_projection_iterations) is not int or not 1 <= max_projection_iterations <= 4096):
        raise ValueError("Consensus requires a positive finite radius and a bounded numerical search")
    prior = np.asarray(protected_margins, dtype=np.float64)
    protected = np.asarray(preservation_gradients, dtype=np.float64)
    if protected.shape == (0,):
        protected = protected.reshape(0, raw.shape[1])
    if protected.ndim != 2 or protected.shape[1] != raw.shape[1] or prior.shape != (len(protected),):
        raise ValueError("Preservation gradients and margins must share the fixed projection coordinates")
    bounds = np.asarray(floor, dtype=np.float64)
    if bounds.ndim == 0:
        bounds = np.full(len(protected), float(bounds))
    if (bounds.shape != prior.shape or not np.isfinite(protected).all() or not np.isfinite(prior).all()
            or not np.isfinite(bounds).all() or np.any(prior < 0) or np.any(bounds < 0)):
        raise ValueError("Preservation requires finite gradients, nonnegative margins, and nonnegative floors")
    historical = np.asarray(historical_direction, dtype=np.float64) if historical_direction is not None else raw.mean(axis=0)
    if historical.shape != (raw.shape[1],) or not np.isfinite(historical).all():
        raise ValueError("Historical direction must share the fixed projection coordinates")
    settings = {"radius": radius, "max_iterations": max_projection_iterations, "tolerance": tolerance}
    agreement = gradient_agreement(ids, labels, raw, required_groups=required_groups)
    margin_mean = raw.mean(axis=0)
    agreement["mean_gradient_cosine_order"] = ["historical", "raw_margin_mean", "equal_group_unit_margin_mean"]
    group_mean = means.mean(axis=0) if supported else np.zeros_like(historical)
    agreement["mean_gradient_cosines"] = _cosine_matrix(np.stack((historical, margin_mean, group_mean)))

    def describe(update, **evidence):
        if update is None:
            return ConsensusDirection(None, {**evidence, "update_norm": None})
        norm = float(np.linalg.norm(update))
        raw_improvements = raw @ update
        group_improvements = means @ update
        return ConsensusDirection(update, {**evidence, "update_norm": norm,
            "item_predicted_margin_change": dict(zip(ids, raw_improvements.tolist(), strict=True)),
            "item_directional_slope": dict(zip(ids, (raw_improvements / norm).tolist(), strict=True)) if norm else None,
            "group_raw_mean_improvement": {group: float(raw_improvements[mask].mean())
                                           for group, mask in members.items() if mask.any()},
            "group_normalized_improvement": dict(zip(supported, group_improvements.tolist(), strict=True)),
            "worst_group_normalized_improvement": float(group_improvements.min()) if supported else None,
            "limiting_groups": [group for group, value in zip(supported, group_improvements, strict=True)
                                if value <= group_improvements.min() + tolerance],
            "protected_predicted_margins": (prior + protected @ update).tolist(),
        })

    def fit_mean(direction):
        if not np.any(direction):
            return describe(None, status="zero_preferred_direction")
        fit = solve_inequalities(direction, protected, prior, floor=bounds, **settings)
        return describe(fit.update, status=fit.status, projection=fit.record())

    ordinary = fit_mean(historical)
    weighted = fit_mean(group_mean) if supported else describe(None, status="no_supported_groups")
    if not supported:
        return ConsensusStudy(ordinary, weighted, describe(None, status="no_supported_groups"), agreement)

    upper, upper_weights = _unconstrained_upper(supported, means, radius)
    evidence = {"analytic_upper_bound": upper, "analytic_upper_bound_group_weights": upper_weights,
                "search_steps": [], "witnessed_lower_bound": None}
    if upper <= tolerance:
        status = "no_positive_common_direction" if upper == 0. else "common_improvement_upper_below_tolerance"
        return ConsensusStudy(ordinary, weighted, describe(None, status=status, **evidence), agreement)
    zero = np.zeros(raw.shape[1])
    baseline = solve_inequalities(zero, protected, prior, floor=bounds, **settings)
    if baseline.update is None:
        return ConsensusStudy(ordinary, weighted, describe(None, status="no_preservation_witness",
                             preservation_projection=baseline.record(), **evidence), agreement)

    initial_witnesses = [fit.update for fit in (baseline, ordinary, weighted) if fit.update is not None]
    best = max(initial_witnesses, key=lambda point: float((means @ point).min()))
    lower = float((means @ best).min())
    search_upper = upper
    joint_rows = np.vstack((protected, means))
    joint_margins = np.concatenate((prior, np.zeros(len(means))))
    for _ in range(max_search_steps):
        if search_upper - lower <= tolerance:
            break
        level = (lower + search_upper) / 2
        joint_floor = np.concatenate((bounds, np.full(len(means), level)))
        fit = solve_inequalities(zero, joint_rows, joint_margins, floor=joint_floor, **settings)
        evidence["search_steps"].append({"level": level, "status": fit.status, "iterations": fit.iterations})
        actual = float((means @ fit.update).min()) if fit.update is not None else None
        if actual is not None and actual > lower + tolerance:
            best, lower = fit.update, actual
        else:
            search_upper = level

    evidence.update(witnessed_lower_bound=lower, numerical_search_ceiling=search_upper)
    consensus = describe(best, status="positive_common_witness", **evidence) if lower > tolerance else describe(
        None, status="no_positive_common_witness", **evidence)
    return ConsensusStudy(ordinary, weighted, consensus, agreement)
