"""Bounded local feasibility for target equations and signed preservation margins."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class Reachability:
    witness: tuple[float, ...]
    feasible: bool
    target_residual: float
    min_preservation_slack: float | None
    cycles: int
    projections: int
    status: str


def local_reachability(target_jacobian, desired, preservation_jacobian, margins, signs, min_margins, *,
                       radius, tolerance=1e-4, max_cycles=200) -> Reachability:
    """Seek a feasible point in at most 64 authorized coordinates.

    Dykstra projections intersect target tolerance slabs, preservation
    halfspaces, and a norm ball. An exhausted search is indeterminate.
    """
    target = np.asarray(target_jacobian, dtype=np.float64)
    preserve = np.asarray(preservation_jacobian, dtype=np.float64)
    desired, margins, signs, floor = [np.asarray(value, dtype=np.float64)
                                      for value in (desired, margins, signs, min_margins)]
    if (target.ndim != 2 or not 1 <= target.shape[0] <= 256 or not 1 <= target.shape[1] <= 64
            or preserve.ndim != 2 or preserve.shape[1] != target.shape[1] or preserve.shape[0] > 256
            or desired.shape != (target.shape[0],)
            or any(value.shape != (preserve.shape[0],) for value in (margins, signs, floor))
            or not all(np.isfinite(value).all() for value in (target, preserve, desired, margins, signs, floor))
            or np.any((signs != 1) & (signs != -1)) or np.any(floor < 0)
            or not isfinite(radius) or radius < 0 or not isfinite(tolerance) or tolerance < 0
            or type(max_cycles) is not int or not 1 <= max_cycles <= 2_000):
        raise ValueError("Invalid bounded local reachability problem")

    constraints = np.concatenate((target, -target, -signs[:, None] * preserve))
    bounds = np.concatenate((desired + tolerance, -desired + tolerance, signs * margins - floor))
    norms = np.linalg.norm(constraints, axis=1)
    if not np.isfinite(norms).all() or not np.isfinite(bounds).all():
        raise ValueError("Nonfinite local constraint scale")
    nonzero = norms > 0
    if np.any((~nonzero) & (bounds < 0)):
        impossible = True
    else:
        impossible = bool(np.any(bounds[nonzero] < -radius * norms[nonzero] - 1e-12))
    normals = constraints[nonzero] / norms[nonzero, None]
    limits = bounds[nonzero] / norms[nonzero]
    if not np.isfinite(limits).all():
        raise ValueError("Nonfinite local constraint limits")

    vector = np.zeros(target.shape[1])
    adjustments = np.zeros((len(normals) + 1, len(vector)))

    def result(cycles, projections, status):
        residual = float(np.max(np.abs(target @ vector - desired)))
        slacks = signs * (margins + preserve @ vector) - floor
        min_slack = float(np.min(slacks)) if len(slacks) else None
        feasible = (residual <= tolerance + 1e-10 and (min_slack is None or min_slack >= -1e-10)
                    and np.linalg.norm(vector) <= radius + 1e-10)
        return Reachability(tuple(float(value) for value in vector), bool(feasible), residual, min_slack,
                            cycles, projections, "feasible" if feasible else status)

    state = result(0, 0, "infeasible_bound" if impossible else "searching")
    if state.feasible or impossible:
        return state
    for cycle in range(1, max_cycles + 1):
        for index, (normal, limit) in enumerate(zip(normals, limits, strict=True)):
            point = vector + adjustments[index]
            excess = max(0., float(normal @ point) - limit)
            adjustments[index] = excess * normal
            vector = point - adjustments[index]
        point = vector + adjustments[-1]
        length = np.linalg.norm(point)
        vector = point * (radius / length) if length > radius else point
        adjustments[-1] = point - vector
        state = result(cycle, cycle * len(adjustments), "indeterminate")
        if state.feasible:
            return state
    return state
