"""Practice-only preservation sampling and bounded rank-one update geometry."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class DecisionAnchor:
    item: str
    prompt_id: str
    group: str
    wording: int
    position: int
    margin: float
    token_id: int

    def __post_init__(self):
        if (not all(isinstance(value, str) and value for value in (self.item, self.prompt_id, self.group))
                or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                           for value in (self.wording, self.position, self.token_id))
                or not isfinite(self.margin) or self.margin < 0):
            raise ValueError("An anchor requires a replayed parent decision and a nonnegative margin")

    @property
    def key(self) -> tuple[str, int]:
        return self.prompt_id, self.position


@dataclass(frozen=True)
class AnchorSample:
    mode: str
    anchors: tuple[DecisionAnchor, ...]
    represented_strata: tuple[tuple[str, int], ...]
    missing_strata: tuple[tuple[str, int], ...]
    missing_groups: tuple[str, ...]
    unavailable_strata: tuple[tuple[str, int], ...]
    unavailable_groups: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_strata and not self.missing_groups

    def record(self) -> dict:
        return {**asdict(self), "complete": self.complete}


def sample_anchors(anchors: Sequence[DecisionAnchor], *, budget: int, mode: str = "stratified",
                   required_groups: Sequence[str] = (), required_wordings: Sequence[int] = (),
                   slopes: Mapping[tuple[str, int], float] | None = None, radius: float | None = None,
                   group_quotas: Mapping[str, int] | None = None) -> AnchorSample:
    """Allocate coverage before filling spare slots. The historical mode ignores quotas."""
    anchors = tuple(anchors)
    if (not isinstance(budget, int) or isinstance(budget, bool) or budget < 0
            or mode not in ("global", "stratified", "risk")):
        raise ValueError("Invalid anchor mode or budget")
    if len({anchor.key for anchor in anchors}) != len(anchors):
        raise ValueError("Duplicate practice anchor identity")
    groups = tuple(dict.fromkeys((*required_groups, *(anchor.group for anchor in anchors))))
    wordings = tuple(dict.fromkeys(required_wordings))
    if (not all(isinstance(group, str) and group for group in groups)
            or not all(isinstance(wording, int) and not isinstance(wording, bool) and wording >= 0
                       for wording in wordings)):
        raise ValueError("Invalid required preservation strata")
    required = tuple((group, wording) for group in groups for wording in wordings)
    quotas = dict(group_quotas or {})
    if (mode == "global" and quotas or any(group not in groups or not isinstance(value, int)
            or isinstance(value, bool) or value < 0 for group, value in quotas.items())):
        raise ValueError("Invalid preservation group quotas")
    if mode == "risk" and (slopes is None or radius is None or not isfinite(radius) or radius <= 0
            or any(anchor.key not in slopes or not isfinite(slopes[anchor.key]) for anchor in anchors)):
        raise ValueError("Risk sampling requires a measured direction slope for every input anchor and a radius")

    def priority(anchor):
        risk = (radius * max(0.0, -slopes[anchor.key]) / max(1e-12, anchor.margin)) if mode == "risk" else 0.0
        return -risk, anchor.margin, anchor.position

    selected: list[DecisionAnchor] = []
    if mode == "global":
        # Phase III-I picked the weakest eligible token in each presentation first.
        # Ties between presentations retain the caller's frozen presentation order.
        weakest = {}
        for anchor in anchors:
            current = weakest.get(anchor.prompt_id)
            if current is None or (anchor.margin, anchor.position) < (current.margin, current.position):
                weakest[anchor.prompt_id] = anchor
        selected = sorted(weakest.values(), key=lambda anchor: anchor.margin)[:budget]
    else:
        ordered = sorted(anchors, key=priority)
        used = set()
        counts: Counter[str] = Counter()

        def take(group, wording=None):
            if len(selected) == budget or counts[group] >= quotas.get(group, budget):
                return
            for anchor in ordered:
                if anchor.group == group and (wording is None or wording == anchor.wording) and anchor.key not in used:
                    selected.append(anchor)
                    used.add(anchor.key)
                    counts[group] += 1
                    return

        for group in groups:
            eligible = next((anchor for anchor in ordered if anchor.group == group
                             and (not wordings or anchor.wording in wordings)), None)
            take(group, eligible.wording if eligible is not None else None)
        for group, wording in required:
            if not any(anchor.group == group and anchor.wording == wording for anchor in selected):
                take(group, wording)
        while len(selected) < budget:
            previous = len(selected)
            if mode == "risk":
                # With required coverage allocated, the strongest measured risk gets the spare slot.
                remaining = next((anchor for anchor in ordered if anchor.key not in used
                                  and counts[anchor.group] < quotas.get(anchor.group, budget)), None)
                if remaining is not None:
                    take(remaining.group)
            else:
                for group in sorted(groups, key=lambda group: counts[group]):
                    take(group)
            if len(selected) == previous:
                break

    available_strata = {(anchor.group, anchor.wording) for anchor in anchors}
    represented = tuple(dict.fromkeys((anchor.group, anchor.wording) for anchor in selected))
    selected_groups = {anchor.group for anchor in selected}
    available_groups = {anchor.group for anchor in anchors}
    return AnchorSample(mode, tuple(selected), represented, tuple(cell for cell in required if cell not in represented),
                        tuple(group for group in groups if group not in selected_groups),
                        tuple(cell for cell in required if cell not in available_strata),
                        tuple(group for group in groups if group not in available_groups))


@dataclass(frozen=True)
class _Interval:
    lower: float
    upper: float
    lower_open: bool = False
    upper_open: bool = False

    def intersect(self, other: _Interval) -> _Interval | None:
        lower, upper = max(self.lower, other.lower), min(self.upper, other.upper)
        lower_open = ((lower == self.lower and self.lower_open) or (lower == other.lower and other.lower_open))
        upper_open = ((upper == self.upper and self.upper_open) or (upper == other.upper and other.upper_open))
        if lower > upper or (lower == upper and (lower_open or upper_open)):
            return None
        return _Interval(lower, upper, lower_open, upper_open)


def _margin_interval(margin: float, slope: float, floor: float, domain: _Interval, *, strict: bool):
    if slope == 0.0:
        return domain if (margin > floor if strict else margin >= floor) else None
    crossing = (floor - margin) / slope
    side = (_Interval(crossing, np.inf, strict, False) if slope > 0
            else _Interval(-np.inf, crossing, False, strict))
    return domain.intersect(side)


def _finite_vector(values, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector")
    return vector


def _floors(floor, count: int) -> np.ndarray:
    values = np.asarray(floor, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Floor must be a finite scalar or one value per decision")
    if values.ndim == 0:
        values = np.full(count, float(values))
    if values.shape != (count,):
        raise ValueError("Floor must be a finite scalar or one value per decision")
    return values


def feasible_intervals(target_margins, target_slopes, protected_margins, protected_slopes, *,
                       radius: float, floor=0.0, target_floor=0.0) -> dict:
    """Intersect local decision intervals along one unit direction. A target tie is not a correction."""
    if not isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be finite and positive")
    target = _finite_vector(target_margins, "Target margins")
    target_direction = _finite_vector(target_slopes, "Target slopes")
    protected = _finite_vector(protected_margins, "Protected margins")
    protected_direction = _finite_vector(protected_slopes, "Protected slopes")
    if target.shape != target_direction.shape or protected.shape != protected_direction.shape:
        raise ValueError("Every decision must have a matching margin and directional slope")
    domain = _Interval(0.0, radius)
    protected_interval = domain
    individual_protected = []
    for margin, slope, bound in zip(protected.tolist(), protected_direction.tolist(), _floors(floor, len(protected)).tolist(), strict=True):
        interval = _margin_interval(margin, slope, bound, domain, strict=False)
        individual_protected.append(None if interval is None else asdict(interval))
        protected_interval = (protected_interval.intersect(interval)
                              if protected_interval is not None and interval is not None else None)
    all_targets = protected_interval
    individual_targets = []
    for margin, slope, bound in zip(target.tolist(), target_direction.tolist(), _floors(target_floor, len(target)).tolist(), strict=True):
        interval = _margin_interval(margin, slope, bound, domain, strict=True)
        safe = protected_interval.intersect(interval) if protected_interval is not None and interval is not None else None
        all_targets = all_targets.intersect(interval) if all_targets is not None and interval is not None else None
        threshold = (bound - margin) / slope if margin <= bound and slope > 0 else (0.0 if margin > bound else None)
        individual_targets.append({"parent_margin": margin, "slope": slope, "floor": bound,
                                   "alpha_target_min": threshold if threshold is not None and isfinite(threshold) else None,
                                   "target_interval": None if interval is None else asdict(interval),
                                   "safe_interval": None if safe is None else asdict(safe),
                                   "reason": ("no_target_crossing_within_radius" if interval is None else
                                              "preservation_conflict" if safe is None else "local_witness")})
    return {"protected_interval": None if protected_interval is None else asdict(protected_interval),
            "alpha_preserve_max": None if protected_interval is None else protected_interval.upper,
            "per_protected": individual_protected, "per_target": individual_targets,
            "all_target_interval": None if all_targets is None else asdict(all_targets)}


def _direction_and_rows(direction, preservation_gradients, radius):
    vector = _finite_vector(direction, "Direction")
    rows = np.asarray(preservation_gradients, dtype=np.float64)
    if rows.shape == (0,):
        rows = rows.reshape(0, len(vector))
    if (not vector.size or rows.ndim != 2 or rows.shape[1] != len(vector) or not np.isfinite(rows).all()
            or not isfinite(radius) or radius <= 0):
        raise ValueError("Invalid preservation gradients or radius")
    return vector, rows


def project_nullspace(direction, preservation_gradients, *, radius: float) -> np.ndarray | None:
    """Historical rigid projection of the output factor, then rescale to the dose."""
    vector, rows = _direction_and_rows(direction, preservation_gradients, radius)
    norms = np.linalg.norm(rows, axis=1)
    vector_norm = float(np.linalg.norm(vector))
    if not np.isfinite(norms).all() or not isfinite(vector_norm):
        raise ValueError("Direction or preservation gradient norm is not finite")
    active = norms > 1e-12
    if active.any():
        _, singular, basis = np.linalg.svd(rows[active] / norms[active, None], full_matrices=False)
        basis = basis[singular > singular[0] * 1e-8]
        projected = vector - basis.T @ (basis @ vector)
    else:
        projected = vector
    norm = float(np.linalg.norm(projected))
    if norm <= 1e-9 * max(1.0, vector_norm):
        return None
    return (radius / norm) * projected


@dataclass(frozen=True)
class InequalitySolution:
    update: np.ndarray | None
    status: str
    iterations: int
    max_violation: float
    max_normalized_violation: float | None
    radius_violation: float
    update_norm: float | None
    target_alignment: float | None

    def record(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "update"}


def solve_inequalities(direction, preservation_gradients, protected_margins, *, radius: float,
                       floor=0.0, max_iterations: int = 512, tolerance: float = 1e-9) -> InequalitySolution:
    """Project a preferred output factor onto measured halfspaces and a radius ball.

    Dykstra's algorithm minimizes squared distance to the preferred radius-sized factor.
    A bounded iteration failure is reported as a missing witness, not a proof of infeasibility.
    """
    vector, rows = _direction_and_rows(direction, preservation_gradients, radius)
    margins = _finite_vector(protected_margins, "Protected margins")
    if len(margins) != len(rows) or not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("Expected one margin per preservation row and a positive iteration budget")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    bounds = _floors(floor, len(rows)) - margins
    norms = np.linalg.norm(rows, axis=1)
    if not np.isfinite(norms).all() or not np.isfinite(bounds).all():
        raise ValueError("Preservation constraint bound or gradient norm is not finite")
    active = norms > 0
    axes = rows[active] / norms[active, None]
    with np.errstate(over="ignore", invalid="raise"):
        levels = bounds[active] / norms[active]
    # A halfspace below the minimum possible dot product cannot bind inside the ball.
    binding = levels >= -radius
    axes, levels = axes[binding], levels[binding]
    direction_norm = float(np.linalg.norm(vector))
    if not isfinite(direction_norm):
        raise ValueError("Preferred direction norm is not finite")
    preferred = (radius / direction_norm) * vector if direction_norm > 0 else np.zeros_like(vector)

    def residual(point):
        raw = max(0.0, float(np.max(bounds - rows @ point, initial=0.0)))
        normalized = max(0.0, float(np.max(levels - axes @ point, initial=0.0)))
        return raw, normalized, max(0.0, float(np.linalg.norm(point)) - radius)

    def result(status, steps, point, *, usable):
        raw, normalized, excess = residual(point)
        return InequalitySolution(point.copy() if usable else None, status, steps, raw,
                                  normalized if isfinite(normalized) else None, excess,
                                  float(np.linalg.norm(point)) if usable else None,
                                  float((preferred / radius) @ point) if usable else None)

    if (bounds[~active] > tolerance).any():
        return result("impossible_constant_constraint", 0, np.zeros_like(preferred), usable=False)
    if (levels > radius + tolerance).any():
        return result("halfspace_outside_radius", 0, np.zeros_like(preferred), usable=False)
    if max(residual(preferred)) <= tolerance:
        return result("preferred_feasible", 0, preferred, usable=True)

    point = preferred.copy()
    corrections = np.zeros(len(axes))
    ball_correction = np.zeros_like(point)
    best = None
    best_distance = np.inf
    for iteration in range(1, max_iterations + 1):
        previous = point.copy()
        previous_corrections = corrections.copy()
        previous_ball_correction = ball_correction.copy()
        for index, (axis, level) in enumerate(zip(axes, levels, strict=True)):
            shifted = point + corrections[index] * axis
            deficit = max(0.0, float(level - axis @ shifted))
            point = shifted + deficit * axis
            corrections[index] = -deficit
        shifted = point + ball_correction
        norm = float(np.linalg.norm(shifted))
        point = (radius / norm) * shifted if norm > radius else shifted
        ball_correction = shifted - point
        if max(residual(point)) <= tolerance:
            distance = float(np.linalg.norm(point - preferred))
            if distance < best_distance:
                best, best_distance = point.copy(), distance
            if max(float(np.linalg.norm(point - previous)),
                   float(np.max(np.abs(corrections - previous_corrections), initial=0.0)),
                   float(np.linalg.norm(ball_correction - previous_ball_correction))) <= tolerance:
                return result("projected_feasible", iteration, point, usable=True)
    if best is not None:
        return result("feasible_iteration_limit", max_iterations, best, usable=True)
    return result("no_feasible_witness", max_iterations, point, usable=False)
