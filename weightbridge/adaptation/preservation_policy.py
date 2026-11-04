"""Evidence routing for bounded behavioral preservation trials."""
from __future__ import annotations

from dataclasses import dataclass

from .fork_policy import ForkDecision, decide_fork_action


@dataclass(frozen=True)
class PreservationEvidence:
    measurement_valid: bool
    persistent_errors: int
    missing_groups: tuple[str, ...] = ()
    locally_feasible: bool | None = None
    constraint_conflict: bool = False
    rounded_noop: bool = False
    surrogate_mismatch: bool = False
    corrected_with_regressions: bool = False
    practice_pass: bool = False
    selection_pass: bool | None = None
    refinement_available: bool = False
    smaller_radius_available: bool = False
    alternate_layer_available: bool = False
    adapter_available: bool = False


def preservation_decision(evidence: PreservationEvidence, *, independent: dict | None = None) -> ForkDecision:
    """Recommend one bounded action without granting host authority."""
    if independent is not None or evidence.selection_pass is not None or evidence.practice_pass:
        return decide_fork_action(measurement_valid=evidence.measurement_valid,
                                  persistent_errors=evidence.persistent_errors, practice_pass=evidence.practice_pass,
                                  selection_pass=evidence.selection_pass, independent=independent)
    if not evidence.measurement_valid or not evidence.persistent_errors:
        return decide_fork_action(measurement_valid=evidence.measurement_valid,
                                  persistent_errors=evidence.persistent_errors)
    if evidence.missing_groups:
        return ForkDecision("broaden_preservation", "missing_preservation_coverage")
    if evidence.corrected_with_regressions:
        if evidence.refinement_available:
            return ForkDecision("refine_constraints", "practice_correction_with_regressions")
        if evidence.smaller_radius_available:
            return ForkDecision("smaller_radius", "practice_correction_with_regressions")
        return ForkDecision("hold", "practice_correction_with_regressions")
    if evidence.constraint_conflict or evidence.locally_feasible is False or evidence.rounded_noop:
        reason = ("bf16_noop" if evidence.rounded_noop else
                  "constraint_conflict" if evidence.constraint_conflict else "empty_local_interval")
        if evidence.alternate_layer_available:
            return ForkDecision("alternate_layer", reason)
        if evidence.adapter_available:
            return ForkDecision("conventional_fit", reason)
        return ForkDecision("hold", reason)
    if evidence.surrogate_mismatch:
        return ForkDecision("measure", "surrogate_to_generation_mismatch")
    if evidence.locally_feasible is None:
        return ForkDecision("measure", "local_feasibility_unmeasured")
    return ForkDecision("hold", "probe_budget_exhausted")
