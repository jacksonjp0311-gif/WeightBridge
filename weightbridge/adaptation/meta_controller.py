"""Small, host-gated experiment selector with no checkpoint authority."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import numpy as np


METHODS = ("first_order", "constrained", "interaction")


class Action(str, Enum):
    MEASURE = "measure"
    REACHABILITY = "reachability"
    CURVATURE = "curvature"
    FIRST_ORDER = "first_order"
    CONSTRAINED = "constrained"
    INTERACTION = "interaction"
    HOLD = "hold"


@dataclass(frozen=True)
class EvidenceState:
    measurement_valid: bool | None
    persistent_errors: int | None
    remaining_budget: float
    first_order_reachable: bool | None
    preservation_risk: float | None = None
    curvature: float | None = None
    sensitivity_disagreement: float | None = None
    method_costs: tuple[float, float, float] = (1., 1., 1.)
    out_of_support: bool = False
    diagnostic_costs: tuple[float, float, float] = (1., 1., 1.)
    allowed_methods: tuple[str, ...] = METHODS

    def __post_init__(self):
        for value in (self.measurement_valid, self.first_order_reachable):
            if value is not None and type(value) is not bool:
                raise ValueError("Qualification and reachability must be booleans or unknown")
        if type(self.out_of_support) is not bool:
            raise ValueError("Support status must be a boolean")
        if self.persistent_errors is not None and (type(self.persistent_errors) is not int or self.persistent_errors < 0):
            raise ValueError("Persistent error count must be a nonnegative integer or unknown")
        if (len(self.method_costs) != 3 or len(self.diagnostic_costs) != 3
                or not all(isfinite(value) and value >= 0 for value in
                           (self.remaining_budget, *self.method_costs, *self.diagnostic_costs))
                or any(method not in METHODS for method in self.allowed_methods)):
            raise ValueError("Invalid budget, action cost or method authorization")
        if not all(value is None or isfinite(value) and value >= 0
                   for value in (self.preservation_risk, self.curvature, self.sensitivity_disagreement)):
            raise ValueError("Diagnostic magnitudes must be finite and nonnegative")


@dataclass(frozen=True)
class AdmissionDecision:
    action: Action
    reason: str
    utility: float | None = None

    def record(self) -> dict:
        return {"schema": "weightbridge.meta_decision.v1", "action": self.action.value,
                "reason": self.reason, "utility": self.utility}


@dataclass(frozen=True)
class MethodSelector:
    harm_limit: float = .12
    harm_weight: float = 2.
    cost_weight: float = .01
    curvature_threshold: float = .1
    disagreement_limit: float | None = None

    def __post_init__(self):
        if (not 0 <= self.harm_limit <= 1
                or not all(isfinite(value) and value >= 0 for value in
                           (self.harm_weight, self.cost_weight, self.curvature_threshold))
                or self.disagreement_limit is not None
                and (not isfinite(self.disagreement_limit) or self.disagreement_limit < 0)):
            raise ValueError("Invalid selector thresholds")

    def choose(self, evidence: EvidenceState, predictions=None) -> AdmissionDecision:
        """Recommend a diagnostic, bounded solver or HOLD from practice evidence only."""
        def diagnostic(action, index, reason):
            if evidence.diagnostic_costs[index] > evidence.remaining_budget:
                return AdmissionDecision(Action.HOLD, "diagnostic_budget")
            return AdmissionDecision(action, reason)

        if evidence.measurement_valid is not True:
            return diagnostic(Action.MEASURE, 0, "measurement_unqualified")
        if evidence.persistent_errors is None:
            return diagnostic(Action.MEASURE, 0, "headroom_unmeasured")
        if evidence.persistent_errors == 0:
            return AdmissionDecision(Action.HOLD, "no_persistent_headroom")
        if evidence.out_of_support:
            return AdmissionDecision(Action.HOLD, "outside_calibrated_support")
        if (self.disagreement_limit is not None and evidence.sensitivity_disagreement is not None
                and evidence.sensitivity_disagreement > self.disagreement_limit):
            return AdmissionDecision(Action.HOLD, "sensitivity_disagreement")
        if evidence.first_order_reachable is None:
            return diagnostic(Action.REACHABILITY, 1, "reachability_unmeasured")

        affordable = [index for index, method in enumerate(METHODS) if method in evidence.allowed_methods
                      and evidence.method_costs[index] <= evidence.remaining_budget]
        interaction_available = 2 in affordable
        interaction_supported = evidence.curvature is not None and evidence.curvature > self.curvature_threshold
        if not evidence.first_order_reachable and interaction_available and evidence.curvature is None:
            return diagnostic(Action.CURVATURE, 2, "linear_correction_unreachable")
        eligible = [index for index in affordable if
                    (interaction_supported if index == 2 else evidence.first_order_reachable)]
        if not eligible:
            return AdmissionDecision(Action.HOLD, "no_supported_method_within_budget")
        if predictions is None:
            preferred = 1 if evidence.preservation_risk else 0
            index = preferred if preferred in eligible else eligible[0]
            return AdmissionDecision(Action(METHODS[index]), "practice_rule")

        probabilities = np.asarray(predictions, dtype=float)
        if probabilities.shape != (3, 2) or not np.isfinite(probabilities).all() or np.any(
                (probabilities < 0) | (probabilities > 1)):
            raise ValueError("Expected three method pairs of improvement and harm probabilities")
        estimates = [(float(probabilities[index, 0] - self.harm_weight * probabilities[index, 1]
                            - self.cost_weight * evidence.method_costs[index]), index)
                     for index in eligible if probabilities[index, 1] <= self.harm_limit]
        utility, index = max(estimates, key=lambda value: value[0], default=(0., -1))
        if utility <= 0:
            return AdmissionDecision(Action.HOLD, "risk_or_cost", utility)
        return AdmissionDecision(Action(METHODS[index]), "calibrated_utility", utility)


def _matrix(features):
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Features must be a finite nonempty matrix")
    return values


def _outcomes(outcomes, length):
    values = np.asarray(outcomes, dtype=float)
    if values.shape != (length, 3, 2) or np.any((values != 0) & (values != 1)):
        raise ValueError("Expected binary improvement/harm outcomes for all methods")
    return values.reshape(length, 6)


def _sigmoid(values):
    return np.exp(-np.logaddexp(0., -values))


@dataclass(frozen=True)
class RiskEstimator:
    """Six regularized logistic models, with separate optional Platt calibration."""

    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    calibration_slope: np.ndarray | None = None
    calibration_intercept: np.ndarray | None = None

    @classmethod
    def fit(cls, features, outcomes):
        x = _matrix(features)
        y = _outcomes(outcomes, len(x))
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.
        x = np.clip((x - center) / scale, -8., 8.)
        weights = np.zeros((x.shape[1], y.shape[1]))
        prevalence = (y.sum(axis=0) + .5) / (len(x) + 1.)
        intercept = np.log(prevalence / (1. - prevalence))
        step = 1. / (1. + .25 * np.linalg.norm(x, ord=2)**2 / len(x))
        for _ in range(240):
            residual = _sigmoid(x @ weights + intercept) - y
            weights -= step * (x.T @ residual / len(x) + .015 * weights)
            intercept -= step * residual.mean(axis=0)
        return cls(center, scale, weights, intercept)

    def _logits(self, features):
        x = _matrix(features)
        if x.shape[1] != self.center.size:
            raise ValueError("Feature width differs from training")
        return np.clip((x - self.center) / self.scale, -8., 8.) @ self.weights + self.intercept

    def calibrate(self, features, outcomes):
        logits = self._logits(features)
        y = _outcomes(outcomes, len(logits))
        slope = np.ones(logits.shape[1])
        bias = np.zeros(logits.shape[1])
        step = 1. / (1. + .25 * np.square(logits).mean(axis=0))
        for _ in range(200):
            residual = _sigmoid(logits * slope + bias) - y
            slope = np.maximum(0., slope - step * ((logits * residual).mean(axis=0) + .01 * (slope - 1)))
            bias -= step * (residual.mean(axis=0) + .01 * bias)
        return RiskEstimator(self.center, self.scale, self.weights, self.intercept, slope, bias)

    def predict(self, features):
        logits = self._logits(features)
        if self.calibration_slope is not None:
            logits = logits * self.calibration_slope + self.calibration_intercept
        return _sigmoid(logits).reshape(-1, 3, 2)
