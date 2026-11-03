"""Deterministic advice for the host running a causal fork adaptation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForkDecision:
    action: str
    reason: str

    def record(self) -> dict:
        return {"schema": "weightbridge.fork_decision.v1", "action": self.action,
                "reason": self.reason, "promotion_authority": False}


def decide_fork_action(*, measurement_valid: bool, persistent_errors: int, diagnosis_valid: bool | None = None,
                       reachable: bool | None = None, preservation_conflict: bool = False,
                       excessive_radius: bool = False, surrogate_mismatch: bool = False,
                       practice_pass: bool = False, selection_pass: bool | None = None,
                       independent: dict | None = None, remaining_probe_budget: bool = False,
                       conventional_authorized: bool = False) -> ForkDecision:
    if independent is not None:
        if (practice_pass and selection_pass is True and independent.get("disposition") == "PASS"
                and independent.get("hidden_gate_passed") is True and independent.get("restart_exact") is True
                and independent.get("rollback_exact") is True and independent.get("lineage_event_sha256")):
            return ForkDecision("record_verified", "independent_endpoint_and_replay_pass")
        return ForkDecision("hold", independent.get("reason", "incomplete_independent_verifier"))
    if not measurement_valid:
        return ForkDecision("measure", "invalid_measurement")
    if persistent_errors < 1:
        return ForkDecision("hold", "insufficient_persistent_errors")
    if selection_pass is not None:
        if not practice_pass or not selection_pass:
            return ForkDecision("hold", "selection_failure")
        return ForkDecision("independent_verify", "selection_pass")
    if practice_pass:
        return ForkDecision("independent_select", "corrected_practice_behavior")
    if diagnosis_valid is not True:
        return ForkDecision("measure" if diagnosis_valid is False else "diagnose_fork",
                            "incorrect_fork_diagnosis" if diagnosis_valid is False else "fork_unmeasured")
    if reachable is None:
        return ForkDecision("scan_regions", "reachability_unmeasured")
    if not reachable:
        return ForkDecision("conventional_fit" if conventional_authorized and remaining_probe_budget else "hold",
                            "insufficient_parameter_reachability")
    if preservation_conflict:
        return ForkDecision("constrained_fit" if remaining_probe_budget else "hold", "preservation_conflict")
    if excessive_radius:
        return ForkDecision("hold", "excessive_update_radius")
    if surrogate_mismatch:
        return ForkDecision("measure", "surrogate_to_generation_mismatch")
    return ForkDecision("first_order_fit" if remaining_probe_budget else "hold",
                        "bounded_probe" if remaining_probe_budget else "probe_budget_exhausted")
