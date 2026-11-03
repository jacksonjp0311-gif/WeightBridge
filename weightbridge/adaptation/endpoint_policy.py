"""Endpoint evidence for the Adaptive Neural Meta-Compiler.

The result is a recommendation to the host. It cannot grant permissions,
authorize a checkpoint, or waive independent verification.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .meta_controller import Action, EvidenceState, MethodSelector


class EndpointAction(str, Enum):
    MEASURE = "measure"
    HOLD = "hold"
    BOUNDED_PROBE = "bounded_probe"
    SMALLER_PROBE = "smaller_probe"
    CHANGE_OBJECTIVE = "change_objective"
    INDEPENDENT_VERIFY = "independent_verify"
    RECORD_VERIFIED = "record_verified"


@dataclass(frozen=True)
class EndpointEvidence:
    measured: int = 0
    effective: int = 0
    practice_passes: int = 0
    surrogate_improvements: int = 0
    behavior_gains_among_surrogate_improvements: int = 0
    regressing_probes: int = 0
    excessive_probes: int = 0
    smaller_probe_authorized: bool = False
    independent_pass: bool | None = None
    hash_and_reload_verified: bool = False

    def __post_init__(self):
        counts = (self.measured, self.effective, self.practice_passes, self.surrogate_improvements,
                  self.behavior_gains_among_surrogate_improvements, self.regressing_probes,
                  self.excessive_probes)
        if (any(type(value) is not int or value < 0 for value in counts)
                or self.effective > self.measured
                or max(self.practice_passes, self.surrogate_improvements, self.regressing_probes) > self.effective
                or self.practice_passes + self.regressing_probes > self.effective
                or self.behavior_gains_among_surrogate_improvements > self.surrogate_improvements
                or self.excessive_probes > self.measured):
            raise ValueError("Inconsistent endpoint probe counts")
        if (type(self.smaller_probe_authorized) is not bool or type(self.hash_and_reload_verified) is not bool
                or (self.independent_pass is not None and type(self.independent_pass) is not bool)
                or (self.independent_pass is not None and self.practice_passes == 0)):
            raise ValueError("Independent verification requires a practice-qualified candidate")


@dataclass(frozen=True)
class EndpointDecision:
    action: EndpointAction
    reason: str
    method: str | None = None

    def record(self) -> dict:
        return {"schema": "weightbridge.endpoint_meta_decision.v1", "action": self.action.value,
                "reason": self.reason, "method": self.method, "promotion_authority": False}


def select_endpoint_action(state: EvidenceState, evidence: EndpointEvidence,
                           selector: MethodSelector | None = None) -> EndpointDecision:
    """Combine ANMC's capability/budget routing with actual endpoint evidence."""
    method = (selector or MethodSelector()).choose(state)
    if state.measurement_valid is not True or state.persistent_errors is None:
        action = EndpointAction.MEASURE if method.action is Action.MEASURE else EndpointAction.HOLD
        return EndpointDecision(action, method.reason)
    if state.persistent_errors == 0 or state.out_of_support:
        return EndpointDecision(EndpointAction.HOLD, method.reason)
    if evidence.independent_pass is False:
        return EndpointDecision(EndpointAction.HOLD, "failed_independent_endpoint")
    if evidence.independent_pass is True:
        return (EndpointDecision(EndpointAction.RECORD_VERIFIED, "independent_endpoint_and_reload_pass")
                if evidence.hash_and_reload_verified else
                EndpointDecision(EndpointAction.HOLD, "checkpoint_identity_or_reload_unverified"))
    if evidence.practice_passes:
        return EndpointDecision(EndpointAction.INDEPENDENT_VERIFY, "actual_practice_endpoint_pass")
    if evidence.measured:
        if (evidence.surrogate_improvements >= 2
                and evidence.behavior_gains_among_surrogate_improvements == 0):
            action = (EndpointAction.CHANGE_OBJECTIVE if state.remaining_budget > 0 else EndpointAction.HOLD)
            return EndpointDecision(action, "surrogate_endpoint_mismatch")
        if evidence.excessive_probes or evidence.regressing_probes:
            smaller = (evidence.smaller_probe_authorized and state.remaining_budget > 0
                       and method.action in (Action.FIRST_ORDER, Action.CONSTRAINED, Action.INTERACTION))
            reason = "preservation_regression" if evidence.regressing_probes else "excessive_update_magnitude"
            return EndpointDecision(EndpointAction.SMALLER_PROBE if smaller else EndpointAction.HOLD, reason,
                                    method.action.value if smaller else None)
        return EndpointDecision(EndpointAction.HOLD, "no_endpoint_gain" if evidence.effective else
                                "ineffective_stored_precision_direction")
    if method.action in (Action.FIRST_ORDER, Action.CONSTRAINED, Action.INTERACTION):
        return EndpointDecision(EndpointAction.BOUNDED_PROBE, method.reason, method.action.value)
    if method.action in (Action.MEASURE, Action.REACHABILITY, Action.CURVATURE):
        return EndpointDecision(EndpointAction.MEASURE, method.reason, method.action.value)
    return EndpointDecision(EndpointAction.HOLD, method.reason)
