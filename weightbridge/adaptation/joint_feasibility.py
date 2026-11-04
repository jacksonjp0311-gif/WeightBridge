"""Bounded minimum-norm witnesses for jointly specified local behavior margins.

The constraints live in a host-authorized, low-dimensional update space. These
numerical results concern its linear approximation, not BF16 storage, complete
generation, generalization, or checkpoint promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from math import isfinite
import time

import numpy as np


SCOPE = ("Numerical minimum-norm analysis of the supplied finite local linear inequalities in "
         "fixed update coordinates. No BF16, nonlinear behavior, hidden evaluation or promotion authority. "
         "Numerical certificates use floating-point residuals and conservative roundoff padding, "
         "not formally verified interval arithmetic.")


@dataclass(frozen=True)
class JointFeasibility:
    update: np.ndarray | None
    minimum_norm_update: np.ndarray | None
    evidence: dict

    def record(self):
        return self.evidence


def _array_hash(values):
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode("ascii") + b"\0" + array.tobytes()).hexdigest()


def _floors(values, count, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(count, float(result))
    if result.shape != (count,) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} requires finite nonnegative floors for every decision")
    return result


def _exact_opposed_certificate(first, second, first_bound, second_bound):
    """Check an exact Farkas pair for the supplied binary floating-point values.

    Rounded normalization alone cannot establish exact proportionality. Fraction
    arithmetic is used only for the rare putatively opposite row pair.
    """
    pivot = next((index for index, value in enumerate(first) if value != 0), None)
    if pivot is None or second[pivot] == 0 or np.signbit(first[pivot]) == np.signbit(second[pivot]):
        return None
    left, right = Fraction(float(first[pivot])), Fraction(float(second[pivot]))
    if any(Fraction(float(a)) * right != Fraction(float(b)) * left
           for a, b in zip(first, second, strict=True)):
        return None
    weights = (abs(right), abs(left))
    weighted_bound = weights[0] * Fraction(float(first_bound)) + weights[1] * Fraction(float(second_bound))
    if weighted_bound <= 0:
        return None
    return {"exact_raw_binary_float_opposition": True, "exact_positive_weighted_rhs": True,
            "nonnegative_original_row_weights": [float(value) for value in weights],
            "pivot_coordinate": pivot}


def solve_minimum_norm(rows, rhs, *, radius, max_sweeps=8192, tolerance=1e-9,
                       row_ids=None, kinds=None):
    """Minimize .5*||u||^2 subject to ``rows @ u >= rhs`` in bounded arithmetic.

    Hildreth's cyclic dual coordinate ascent uses a row-normalized small Gram
    matrix. Every nonnegative dual iterate supplies a norm lower bound, even if
    its corresponding primal vector has not satisfied all constraints. A failed
    bounded search is therefore distinct from a measured dual bound above the
    authorized radius. ``update`` contains only a within-radius primal witness;
    ``minimum_norm_update`` is an unrestricted diagnostic and may exceed it.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    a, b = np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)
    if (a.ndim != 2 or not 1 <= a.shape[0] <= 64 or not 1 <= a.shape[1] <= 1024
            or b.shape != (a.shape[0],) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not isfinite(radius) or radius <= 0 or not isfinite(tolerance) or tolerance <= 0
            or type(max_sweeps) is not int or not 1 <= max_sweeps <= 65536):
        raise ValueError("Joint feasibility needs bounded finite rows, positive radius and numerical budgets")
    ids = tuple(f"decision/{index}" for index in range(len(a))) if row_ids is None else tuple(row_ids)
    labels = tuple("target" for _ in a) if kinds is None else tuple(kinds)
    if (len(ids) != len(a) or len(set(ids)) != len(ids) or len(labels) != len(a)
            or any(not isinstance(value, str) or not value for value in (*ids, *labels))):
        raise ValueError("Every local inequality needs a unique identity and nonempty kind")
    # Scaling prevents a nonzero tiny row from underflowing to an alleged exact
    # zero constraint. Unrepresentably large norms are refused below.
    scales = np.max(np.abs(a), axis=1)
    scaled = np.divide(a, scales[:, None], out=np.zeros_like(a), where=scales[:, None] > 0)
    norms = scales * np.sqrt(np.sum(scaled * scaled, axis=1))
    if not np.isfinite(norms).all():
        raise ValueError("Joint constraint row norms overflowed finite arithmetic")
    nonzero = norms > 0
    axes = np.divide(a, norms[:, None], out=np.zeros_like(a), where=nonzero[:, None])
    levels = np.divide(b, norms, out=np.zeros_like(b), where=nonzero)
    if not np.isfinite(levels).all():
        raise ValueError("Normalized joint constraints overflowed finite arithmetic")
    gram = axes @ axes.T
    eps = np.finfo(np.float64).eps
    base = {
        "schema": "weightbridge.joint_linear_feasibility.v1", "scope": SCOPE,
        "objective": "minimize one half squared unrounded update norm subject to every supplied inequality",
        "algorithm": "Hildreth cyclic nonnegative dual coordinate ascent on row-normalized Gram matrix",
        "radius": float(radius), "tolerance": float(tolerance), "maximum_sweeps": max_sweeps,
        "row_count": len(a), "dimension": a.shape[1], "row_ids": list(ids), "row_kinds": list(labels),
        "input_sha256": {"rows": _array_hash(a), "rhs": _array_hash(b)},
        "row_norms": norms.tolist(), "required_changes": b.tolist(),
        "individual_minimum_norms": [float(max(0., level)) if active else
                                     (0. if bound <= 0 else None)
                                     for active, bound, level in zip(nonzero, b, levels, strict=True)],
        "individual_norm_bound_failures": [item for item, active, level in zip(ids, nonzero, levels, strict=True)
                                            if active and level > radius],
        "certificates_are_numerical_not_formal_intervals": True,
    }

    def resources():
        return {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                "forward_calls": 0, "backward_calls": 0, "generated_response_count": 0}

    # An exact zero row with a strictly positive right side is contradictory;
    # tolerance must not turn an impossible constant requirement into a witness.
    impossible = np.flatnonzero(~nonzero & (b > 0))
    if len(impossible):
        index = int(impossible[0])
        return JointFeasibility(None, None, {**base, "status": "INFEASIBLE_ZERO_ROW", "sweeps": 0,
            "within_radius": False, "primal_feasible": False, "minimum_norm": None,
            "certified_norm_lower_bound": None,
            "contradiction_certificate": {"row_id": ids[index], "zero_row": True,
                                          "positive_required_change": float(b[index])},
            "resources": resources()})

    # Summing two nonnegative normalized constraints yields a radius separator.
    # Only an exactly zero summed axis is labeled an opposed-row contradiction.
    # A nearly opposed pair can merely exclude this radius, not all radii.
    pair_separators = []
    for first in range(len(a)):
        for second in range(first + 1, len(a)):
            if not nonzero[first] or not nonzero[second]:
                continue
            axis = axes[first] + axes[second]
            norm = float(np.linalg.norm(axis))
            needed = float(levels[first] + levels[second])
            guard = 64 * eps * (1 + abs(levels[first]) + abs(levels[second]) + radius * norm)
            exact = (_exact_opposed_certificate(a[first], a[second], b[first], b[second])
                     if np.count_nonzero(axis) == 0 else None)
            if exact is not None:
                return JointFeasibility(None, None, {**base, "status": "INFEASIBLE_OPPOSED_CONSTRAINTS",
                    "sweeps": 0, "within_radius": False, "primal_feasible": False, "minimum_norm": None,
                    "certified_norm_lower_bound": None,
                    "contradiction_certificate": {"row_ids": [ids[first], ids[second]],
                        "summed_axis_norm": norm, "summed_required_change": needed, **exact},
                    "resources": resources()})
            if needed > radius * norm + guard:
                record = {"row_ids": [ids[first], ids[second]], "nonnegative_normalized_weights": [1., 1.],
                          "summed_axis_norm": norm, "summed_required_change": needed,
                          "radius_support_upper": radius * norm, "roundoff_padding": guard,
                          "separating_excess": needed - radius * norm}
                pair_separators.append(record)

    dual = np.zeros(len(a))
    products = np.zeros(len(a))
    best_primal, best_primal_record = None, None
    best_dual_record = None
    converged = False
    numerical_failure = None

    def inspect():
        point = axes.T @ dual
        norm = float(np.linalg.norm(point))
        slack = a @ point - b
        normalized_slack = axes @ point - levels
        objective = .5 * norm * norm
        original_dual = np.divide(dual, norms, out=np.zeros_like(dual), where=nonzero)
        dual_point = a.T @ original_dual
        dual_square = float(dual_point @ dual_point)
        dual_value = float(b @ original_dual - .5 * dual_square)
        # The guard deliberately dominates roundoff at the small local norms in
        # this study. It is conservative numerical padding, not an interval proof.
        padding = float(64 * eps * (1 + np.abs(b) @ original_dual + .5 * dual_square))
        objective_lower = float(np.sqrt(max(0., 2 * (dual_value - padding))))
        numerator = float(b @ original_dual)
        numerator_padding = float(64 * eps * (1 + np.abs(b) @ original_dual))
        denominator = float(np.linalg.norm(dual_point))
        denominator_padding = float(64 * eps * (1 + denominator))
        directional_lower = float(max(0., numerator - numerator_padding) / (denominator + denominator_padding))
        lower = max(objective_lower, directional_lower)
        primal_violation = float(max(0., -np.min(slack)))
        positive = dual > 0
        projected = np.where(positive, np.abs(normalized_slack), np.maximum(0., -normalized_slack))
        complementarity = float(np.max(np.abs(dual * normalized_slack)))
        gap = objective - dual_value
        row = {"primal_norm": norm, "primal_objective": objective, "dual_objective": dual_value,
               "duality_gap": float(gap), "dual_roundoff_padding": padding,
               "certified_norm_lower_bound": lower,
               "dual_objective_norm_lower_bound": objective_lower,
               "dual_direction_norm_lower_bound": directional_lower,
               "dual_direction_numerator": numerator,
               "dual_direction_denominator": denominator,
               "dual_direction_roundoff_padding": {"numerator": numerator_padding, "denominator": denominator_padding},
               "maximum_primal_violation": primal_violation,
               "maximum_normalized_primal_violation": float(max(0., -np.min(normalized_slack))),
               "maximum_complementarity_residual": complementarity,
               "projected_dual_residual": float(np.max(projected)),
               "stationarity_residual_norm": float(np.linalg.norm(point - dual_point)),
               "dual_nonnegative": bool(np.all(dual >= 0)),
               "normalized_dual_multipliers": dual.tolist(),
               "original_dual_multipliers": original_dual.tolist(),
               "constraint_slacks": slack.tolist(),
               "active_constraint_ids": [item for item, value in zip(ids, slack, strict=True)
                                         if abs(value) <= tolerance],
               "active_dual_constraint_ids": [item for item, value in zip(ids, dual, strict=True) if value > 0]}
        return point, row

    for sweep in range(max_sweeps + 1):
        if sweep:
            for index in range(len(a)):
                if not nonzero[index]:
                    continue
                previous = dual[index]
                next_value = max(0., previous + (levels[index] - products[index]) / gram[index, index])
                change = next_value - previous
                dual[index] = next_value
                products += change * gram[:, index]
            # Recompute rather than allowing incremental Gram error to drift.
            products = gram @ dual
        if not np.isfinite(dual).all() or not np.isfinite(products).all():
            numerical_failure = "nonfinite_dual_iterate"
            break
        point, observation = inspect()
        if not all(isfinite(observation[key]) for key in ("primal_norm", "dual_objective", "duality_gap",
                "maximum_primal_violation", "certified_norm_lower_bound")):
            numerical_failure = "nonfinite_primal_or_dual_measurement"
            break
        if best_dual_record is None or observation["certified_norm_lower_bound"] > best_dual_record["certified_norm_lower_bound"]:
            best_dual_record = observation
        primal_feasible = (observation["maximum_primal_violation"] <= tolerance
                           and observation["maximum_normalized_primal_violation"] <= tolerance)
        if primal_feasible:
            if best_primal_record is None or observation["primal_norm"] < best_primal_record["primal_norm"]:
                best_primal, best_primal_record = point.copy(), observation
        if (primal_feasible
                and observation["projected_dual_residual"] <= tolerance
                and observation["maximum_complementarity_residual"] <= tolerance * (1 + observation["primal_norm"] ** 2)
                and abs(observation["duality_gap"]) <= tolerance * (1 + observation["primal_objective"])):
            converged = True
            best_primal, best_primal_record = point.copy(), observation
            break
    lower = best_dual_record["certified_norm_lower_bound"] if best_dual_record else None
    norm = best_primal_record["primal_norm"] if best_primal_record else None
    within = best_primal is not None and norm <= radius
    certified_excess = lower is not None and lower > radius + tolerance
    if certified_excess:
        status = "CERTIFIED_NORM_BOUND_EXCEEDED"
        within = False
    elif within and converged:
        status = "LOCAL_LINEAR_FEASIBLE"
    elif within:
        status = "FEASIBLE_BOUNDED_WITNESS"
    else:
        status = "BOUNDED_NO_WITNESS"
    # Strong pair separators may diagnose cap failure even when coordinate
    # ascent has not yet supplied a sufficient norm lower bound.
    if pair_separators and not within and not certified_excess:
        status = "CERTIFIED_RADIUS_SEPARATOR"
    detail = best_primal_record or best_dual_record or {}
    return JointFeasibility(best_primal.copy() if within else None,
        best_primal.copy() if best_primal is not None else None,
        {**base, "status": status, "sweeps": sweep, "converged": converged,
         "primal_feasible": best_primal is not None, "within_radius": bool(within),
         "minimum_norm": norm if converged else None, "best_primal_norm": norm,
         "certified_norm_lower_bound": lower,
         "minimum_norm_certified_to_numerical_tolerance": converged,
         "certificate": best_dual_record,
         "primal_witness": best_primal_record,
         "constraint_slacks": detail.get("constraint_slacks"),
         "active_constraint_ids": detail.get("active_constraint_ids", []),
         "pair_radius_separators": pair_separators,
         "numerical_failure": numerical_failure,
         "bounded_search_failure_proves_infeasibility": False,
         "resources": resources()})


def solve_joint_feasibility(target_rows, target_parent_margins, *, target_floor=.0625,
                            preservation_rows=(), preservation_parent_margins=(), preservation_floor=.125,
                            radius=.025, max_sweeps=8192, tolerance=1e-9,
                            target_ids=None, preservation_ids=None):
    """Translate individual target and preservation margins into common halfspaces."""
    target = np.asarray(target_rows, dtype=np.float64)
    parent = np.asarray(target_parent_margins, dtype=np.float64)
    if (target.ndim != 2 or not len(target) or parent.shape != (len(target),)
            or not np.isfinite(parent).all()):
        raise ValueError("Every target decision requires a finite row and parent margin")
    protected = np.asarray(preservation_rows, dtype=np.float64)
    if protected.shape == (0,):
        protected = protected.reshape(0, target.shape[1])
    prior = np.asarray(preservation_parent_margins, dtype=np.float64)
    if (protected.ndim != 2 or protected.shape[1] != target.shape[1]
            or prior.shape != (len(protected),) or not np.isfinite(prior).all() or np.any(prior < 0)):
        raise ValueError("Preservation decisions require compatible rows and replayed nonnegative parent margins")
    floors = _floors(target_floor, len(target), "Targets")
    if np.any(floors <= 0):
        raise ValueError("Individual corrective decisions require a strictly positive target floor")
    protected_floors = _floors(preservation_floor, len(protected), "Preservation")
    ids = tuple(f"target/{index}" for index in range(len(target))) if target_ids is None else tuple(target_ids)
    prior_ids = tuple(f"preservation/{index}" for index in range(len(protected))) if preservation_ids is None else tuple(preservation_ids)
    if len(ids) != len(target) or len(prior_ids) != len(protected):
        raise ValueError("Every target and preservation decision requires a matching identity")
    margins = np.concatenate((parent, prior))
    required = np.concatenate((floors, protected_floors))
    result = solve_minimum_norm(np.vstack((target, protected)), required - margins,
        radius=radius, max_sweeps=max_sweeps, tolerance=tolerance,
        row_ids=(*ids, *prior_ids), kinds=("target",) * len(target) + ("preservation",) * len(protected))
    evidence = {**result.record(), "target_count": len(target), "preservation_count": len(protected),
                "parent_margins": margins.tolist(), "required_margin_floors": required.tolist()}
    if evidence.get("constraint_slacks") is not None:
        evidence["predicted_margins"] = (required + np.asarray(evidence["constraint_slacks"])).tolist()
    return JointFeasibility(result.update, result.minimum_norm_update, evidence)
