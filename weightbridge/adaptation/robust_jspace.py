"""Practice-only sensitivity ranking and local robust update evidence."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from .fork_policy import ForkDecision, decide_fork_action
from .preservation_geometry import DecisionAnchor, feasible_intervals, sample_anchors, solve_inequalities


@dataclass(frozen=True)
class PreservationSelection:
    mode: str
    anchors: tuple[DecisionAnchor, ...]
    represented_strata: tuple[tuple[str, int], ...]
    missing_strata: tuple[tuple[str, int], ...]
    missing_groups: tuple[str, ...]
    unavailable_strata: tuple[tuple[str, int], ...]
    item_counts: tuple[tuple[str, int], ...]
    eligible_decisions: int
    measured_decisions: int
    selected_risks: tuple[float | None, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_strata and not self.missing_groups

    def record(self) -> dict:
        return {**asdict(self), "complete": self.complete, "represented_items": len(self.item_counts),
                "unmeasured_decisions": self.eligible_decisions - self.measured_decisions}


def directional_risk(margin: float, slope: float, *, radius: float, margin_floor: float = 0.0) -> float:
    """Adverse directional movement relative to available slack. This is not a probability."""
    if not all(isfinite(value) for value in (margin, slope, radius, margin_floor)) or radius <= 0 or margin_floor < 0:
        raise ValueError("Directional risk needs finite margins and a positive radius")
    result = radius * max(0.0, -slope) / max(margin - margin_floor, 1e-12)
    if not isfinite(result):
        raise ValueError("Directional risk is not representable")
    return result


def select_preservation(anchors: Sequence[DecisionAnchor], *, budget: int, mode: str = "stratified_directional",
                        required_groups: Sequence[str] = (), required_wordings: Sequence[int] = (),
                        slopes: Mapping[tuple[str, int], float] | None = None, radius: float | None = None,
                        margin_floor: float = 0.0) -> PreservationSelection:
    """Compare four samplers on a common pool and report unmeasured coverage explicitly."""
    anchors = tuple(anchors)
    if (type(budget) is not int or budget < 0 or mode not in
            ("historical", "stratified", "directional", "stratified_directional")):
        raise ValueError("Invalid preservation sampling mode or budget")
    by_key = {anchor.key: anchor for anchor in anchors}
    if len(by_key) != len(anchors):
        raise ValueError("Duplicate practice anchor identity")
    groups = tuple(dict.fromkeys((*required_groups, *(anchor.group for anchor in anchors))))
    wordings = tuple(dict.fromkeys(required_wordings))
    if (not all(isinstance(group, str) and group for group in groups)
            or not all(type(wording) is int and wording >= 0 for wording in wordings)):
        raise ValueError("Invalid required preservation cells")
    required = tuple((group, wording) for group in groups for wording in wordings)
    directional = mode in ("directional", "stratified_directional")
    if (directional or slopes is not None) and (slopes is None or radius is None or not isfinite(radius) or radius <= 0):
        raise ValueError("A directional sample needs measured slopes and a positive radius")
    if not isfinite(margin_floor) or margin_floor < 0:
        raise ValueError("Preservation margin floor must be finite and nonnegative")
    slopes = dict(slopes or {})
    if any(key not in by_key for key in slopes):
        raise ValueError("A measured slope does not belong to the supplied practice pool")
    risks = {key: directional_risk(by_key[key].margin, slope, radius=radius, margin_floor=margin_floor)
             for key, slope in slopes.items()}
    if mode == "historical":
        selected = list(sample_anchors(anchors, budget=budget, mode="global").anchors)
    else:
        ordered = sorted((anchor for anchor in anchors if not directional or anchor.key in risks),
                         key=lambda anchor: (-risks[anchor.key] if directional else 0.0, anchor.margin))
        selected = []
        selected_keys = set()

        def take(matches):
            if len(selected) < budget:
                for anchor in matches:
                    if anchor.key not in selected_keys:
                        selected.append(anchor)
                        selected_keys.add(anchor.key)
                        break

        if mode != "directional":
            # Give every available group a slot before spending slots on its other wordings.
            for group in groups:
                take(anchor for anchor in ordered if anchor.group == group and (not wordings or anchor.wording in wordings))
            for cell in required:
                if not any((anchor.group, anchor.wording) == cell for anchor in selected):
                    take(anchor for anchor in ordered if (anchor.group, anchor.wording) == cell)
        items = {anchor.item for anchor in selected}
        # For spare slots first take one decision per unrepresented underlying item.
        for anchor in ordered:
            if anchor.item not in items and len(selected) < budget:
                take((anchor,))
                items.add(anchor.item)
        for anchor in ordered:
            if len(selected) == budget:
                break
            take((anchor,))

    represented = tuple(dict.fromkeys((anchor.group, anchor.wording) for anchor in selected))
    selected_groups = {anchor.group for anchor in selected}
    available = {(anchor.group, anchor.wording) for anchor in anchors if not directional or anchor.key in risks}
    return PreservationSelection(mode, tuple(selected), represented, tuple(cell for cell in required if cell not in represented),
                                 tuple(group for group in groups if group not in selected_groups),
                                 tuple(cell for cell in required if cell not in available),
                                 tuple(sorted(Counter(anchor.item for anchor in selected).items())),
                                 len(anchors), len(risks), tuple(risks.get(anchor.key) for anchor in selected))


@dataclass(frozen=True)
class MarginObservation:
    item: str
    prompt_id: str
    position: int
    predicted_margin: float
    measured_margin: float
    group: str | None = None

    def __post_init__(self):
        if (not self.item or not self.prompt_id or type(self.position) is not int or self.position < 0
                or not all(isfinite(value) for value in (self.predicted_margin, self.measured_margin, self.residual))):
            raise ValueError("Residual evidence requires a finite prediction and measurement tied to a practice item")

    @property
    def residual(self) -> float:
        return float(self.measured_margin - self.predicted_margin)

    def record(self) -> dict:
        return {**asdict(self), "residual": self.residual}


@dataclass(frozen=True)
class ItemResidual:
    item: str
    observations: int
    worst_shortfall: float
    max_abs_residual: float


def _item_residuals(observations: Sequence[MarginObservation]) -> tuple[ItemResidual, ...]:
    by_item: dict[str, list[float]] = {}
    observed = set()
    for observation in observations:
        key = observation.item, observation.prompt_id, observation.position
        if key in observed:
            raise ValueError("Residual observations must identify a decision once per partition")
        observed.add(key)
        by_item.setdefault(observation.item, []).append(observation.residual)
    return tuple(ItemResidual(item, len(values), max(0.0, -min(values)), max(abs(value) for value in values))
                 for item, values in sorted(by_item.items()))


@dataclass(frozen=True)
class ResidualCalibration:
    allowance: float | None
    quantile: float
    minimum_items: int
    item_residuals: tuple[ItemResidual, ...]

    @property
    def status(self) -> str:
        return "estimated" if self.allowance is not None else "insufficient_calibration_items"

    @property
    def items(self) -> tuple[str, ...]:
        return tuple(record.item for record in self.item_residuals)

    def record(self) -> dict:
        return {**asdict(self), "status": self.status, "item_count": len(self.items),
                "quantile_unit": "worst adverse residual per underlying item"}


def calibrate_residuals(observations: Sequence[MarginObservation], *, quantile: float = 0.9,
                        minimum_items: int = 3) -> ResidualCalibration:
    """Set an allowance from the higher empirical order statistic of adverse item residuals."""
    if not isfinite(quantile) or not 0 < quantile <= 1 or type(minimum_items) is not int or minimum_items < 1:
        raise ValueError("Invalid residual allowance calibration policy")
    items = _item_residuals(observations)
    values = [record.worst_shortfall for record in items]
    allowance = float(np.quantile(values, quantile, method="higher")) if len(values) >= minimum_items else None
    return ResidualCalibration(allowance, quantile, minimum_items, items)


def validate_residuals(calibration: ResidualCalibration, observations: Sequence[MarginObservation]) -> dict:
    """Assess empirical allowance coverage on different underlying items without refitting."""
    items = _item_residuals(observations)
    if set(calibration.items).intersection(record.item for record in items):
        raise ValueError("Calibration and residual validation must use disjoint underlying items")
    allowance = calibration.allowance
    covered = None if allowance is None else sum(record.worst_shortfall <= allowance for record in items)
    return {"status": "assessed" if allowance is not None and items else
            calibration.status if allowance is None else "no_validation_items", "allowance": allowance,
            "covered_items": covered, "item_count": len(items), "item_residuals": [asdict(record) for record in items],
            "item_coverage": None if covered is None or not items else covered / len(items)}


def calibrate_margin_floor(parent_margins: Sequence[float], candidate_floors: Sequence[float], *,
                           quantile: float = 0.25, fraction: float = 0.5) -> dict:
    """Choose a predeclared floor using only the lower empirical parent-margin quantile."""
    margins = tuple(parent_margins)
    floors = tuple(candidate_floors)
    if (not isfinite(quantile) or not 0 <= quantile <= 1 or not isfinite(fraction) or not 0 < fraction <= 1
            or not floors or not all(isfinite(value) and value > 0 for value in floors)
            or not all(isfinite(value) and value >= 0 for value in margins)):
        raise ValueError("Invalid frozen margin floor calibration")
    parent_quantile = float(np.quantile(margins, quantile, method="lower")) if margins else None
    cutoff = parent_quantile * fraction if parent_quantile is not None else None
    selected = max((floor for floor in floors if cutoff is not None and floor <= cutoff), default=None)
    return {"selected_floor": selected, "candidate_floors": floors, "parent_margin_quantile": parent_quantile,
            "quantile": quantile, "fraction": fraction, "cutoff": cutoff, "observation_count": len(margins),
            "status": "selected" if selected is not None else "no_supported_positive_floor"}


def _robust_floor(margin_floor, allowance, count):
    floor, reserve = np.asarray(margin_floor, dtype=np.float64), np.asarray(allowance, dtype=np.float64)
    for value in (floor, reserve):
        if (value.ndim != 0 and value.shape != (count,)) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Preservation floors and allowances must be finite and nonnegative")
    total = floor + reserve
    if not np.isfinite(total).all():
        raise ValueError("Combined preservation floor is not finite")
    return total


def solve_robust(direction, preservation_gradients, protected_margins, *, radius: float,
                 margin_floor=0.0, allowance=0.0, max_iterations: int = 512):
    """Solve in the fixed-right factor space with explicit floor and empirical allowance."""
    return solve_inequalities(direction, preservation_gradients, protected_margins, radius=radius,
                              floor=_robust_floor(margin_floor, allowance, len(protected_margins)),
                              max_iterations=max_iterations)


def estimate_window(target_margins, target_slopes, protected_margins, protected_slopes, *, radius: float,
                    margin_floor=0.0, allowance=0.0, target_items: Sequence[str] | None = None) -> dict:
    """Report per-fork and paired-item directional intersections, with strict target crossing."""
    result = feasible_intervals(target_margins, target_slopes, protected_margins, protected_slopes, radius=radius,
                                floor=_robust_floor(margin_floor, allowance, len(protected_margins)))

    def width(interval):
        return 0.0 if interval is None else interval["upper"] - interval["lower"]

    for target in result["per_target"]:
        target["safe_width"] = width(target["safe_interval"])
    result["joint_safe_width"] = width(result["all_target_interval"])
    if target_items is not None:
        if len(target_items) != len(result["per_target"]) or any(not item for item in target_items):
            raise ValueError("A target needs one underlying item identity per fork")
        by_item = {}
        for item, target in zip(target_items, result["per_target"], strict=True):
            by_item.setdefault(item, []).append(target["safe_interval"])
        windows = []
        for item, intervals in by_item.items():
            intersection = None
            if all(interval is not None for interval in intervals):
                lower = max(interval["lower"] for interval in intervals)
                upper = min(interval["upper"] for interval in intervals)
                lower_open = any(interval["lower"] == lower and interval["lower_open"] for interval in intervals)
                upper_open = any(interval["upper"] == upper and interval["upper_open"] for interval in intervals)
                if lower < upper or lower == upper and not (lower_open or upper_open):
                    intersection = {"lower": lower, "upper": upper, "lower_open": lower_open, "upper_open": upper_open}
            windows.append({"item": item, "fork_count": len(intervals), "safe_interval": intersection,
                            "safe_width": width(intersection)})
        result["paired_item_windows"] = windows
        result["max_paired_safe_width"] = max((window["safe_width"] for window in windows), default=0.0)
    return result


def interior_score(*, practice_pass: bool, practice_gain: int, measured_slack: float | None,
                   feasible_window_width: float, realized_norm: float,
                   measured_min_margin: float | None = None) -> tuple[float, ...] | None:
    """Rank practice passers with a positive measured margin and a positive local window."""
    nonnegative = feasible_window_width, realized_norm
    if not all(isfinite(value) and value >= 0 for value in nonnegative):
        raise ValueError("Interior ranking requires finite, nonnegative window and norm")
    if measured_min_margin is None:
        measured_min_margin = measured_slack
    if (not practice_pass or practice_gain <= 0 or not realized_norm or not feasible_window_width
            or measured_slack is None or not isfinite(measured_slack)
            or measured_min_margin is None or not isfinite(measured_min_margin) or measured_min_margin <= 0):
        return None
    return measured_slack, feasible_window_width, practice_gain, -realized_norm


@dataclass(frozen=True)
class JSpaceEvidence:
    measurement_valid: bool
    persistent_errors: int
    missing_groups: tuple[str, ...] = ()
    missing_strata: tuple[tuple[str, int], ...] = ()
    directional_vulnerability: float | None = None
    slack_satisfied: bool | None = None
    calibration_sufficient: bool | None = None
    residual_validation_pass: bool | None = None
    max_prediction_error: float | None = None
    rounded_noop: bool = False
    precision_margin_disagreement: bool = False
    constraint_conflict: bool = False
    window_width: float | None = None
    minimum_window_width: float = 0.0
    corrected_with_regressions: bool = False
    practice_pass: bool = False
    preservation_validation_required: bool = False
    preservation_validation_pass: bool | None = None
    selection_pass: bool | None = None
    smaller_radius_available: bool = False
    alternate_layer_available: bool = False
    adapter_available: bool = False


def jspace_recommendation(evidence: JSpaceEvidence, *, independent: dict | None = None) -> ForkDecision:
    """Route staged evidence without granting authority over selection, lineage or deployment."""
    if not evidence.measurement_valid or not evidence.persistent_errors:
        return decide_fork_action(measurement_valid=evidence.measurement_valid, persistent_errors=evidence.persistent_errors)
    if evidence.preservation_validation_required and evidence.preservation_validation_pass is not True:
        if evidence.preservation_validation_pass is False:
            return ForkDecision("hold", "preservation_validation_failed")
        if evidence.practice_pass:
            return ForkDecision("preservation_validate", "preservation_validation_unmeasured")
    if independent is not None or evidence.selection_pass is not None or evidence.practice_pass:
        return decide_fork_action(measurement_valid=True, persistent_errors=evidence.persistent_errors,
                                  practice_pass=evidence.practice_pass, selection_pass=evidence.selection_pass,
                                  independent=independent)
    if evidence.missing_groups or evidence.missing_strata:
        return ForkDecision("broaden_preservation", "missing_preservation_coverage")
    if evidence.calibration_sufficient is False or evidence.residual_validation_pass is False:
        return ForkDecision("measure", "residual_allowance_unreliable")
    if evidence.rounded_noop or evidence.constraint_conflict or evidence.window_width == 0:
        reason = ("bf16_noop" if evidence.rounded_noop else "constraint_conflict" if evidence.constraint_conflict
                  else "empty_local_window")
        if evidence.alternate_layer_available:
            return ForkDecision("alternate_layer", reason)
        if evidence.adapter_available:
            return ForkDecision("conventional_fit", reason)
        return ForkDecision("hold", reason)
    if evidence.slack_satisfied is False or (evidence.directional_vulnerability or 0.0) >= 1.0:
        return ForkDecision("smaller_radius" if evidence.smaller_radius_available else "hold", "preservation_slack_deficient")
    if evidence.precision_margin_disagreement:
        return ForkDecision("measure", "stored_precision_margin_disagreement")
    if evidence.corrected_with_regressions:
        return ForkDecision("smaller_radius" if evidence.smaller_radius_available else "hold", "practice_correction_with_regressions")
    if evidence.window_width is None:
        return ForkDecision("measure", "local_window_unmeasured")
    if evidence.window_width <= evidence.minimum_window_width:
        return ForkDecision("hold", "narrow_local_window")
    return ForkDecision("hold", "probe_budget_exhausted")
