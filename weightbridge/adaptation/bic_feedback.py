"""Describe measured outcomes after a candidate's decision is frozen."""


def classify_candidate(
    *, changed_entries, practice_gain=None, heldout_gain=None, heldout_regressions=None,
    prior_prediction_flips=None, sentinel_flips=None, expected_objective_delta=None,
    measured_objective_delta=None, solver_disposition=None, budget_exhausted=False,
    norm_cap_exceeded=False, behavior_unchanged=False, global_bias_fraction=None,
):
    """Return evidence labels, without making a causal diagnosis from a score."""
    codes = []
    if solver_disposition == "PRECISION_LOSS" and changed_entries == 0:
        codes.append("PRECISION_INDUCED_UPDATE_LOSS")
    if changed_entries and behavior_unchanged:
        codes.append("NO_MEASURED_FUNCTIONAL_EFFECT")
    if practice_gain is not None and practice_gain < 0:
        codes.append("PRACTICE_TARGET_REGRESSION")
    if (practice_gain is not None and practice_gain > 0 and heldout_gain is not None
            and (heldout_gain <= 0 or (heldout_regressions or 0) > 0)):
        codes.append("POOR_GENERALIZATION")
    if (prior_prediction_flips or 0) > 0:
        codes.append("CROSS_TASK_INTERFERENCE")
    if (sentinel_flips or 0) > 0:
        codes.append("UNRELATED_BEHAVIOR_CHANGE")
    if norm_cap_exceeded:
        codes.append("PARAMETER_DISTURBANCE_CAP")
    if (expected_objective_delta is not None and measured_objective_delta is not None
            and expected_objective_delta < -1e-9 and measured_objective_delta > 1e-9):
        codes.append("MISLEADING_LOCAL_APPROXIMATION")
    if global_bias_fraction is not None and global_bias_fraction >= 0.9:
        codes.append("DOMINANT_CONSTANT_MARGIN_SHIFT")
    if budget_exhausted and (practice_gain is None or practice_gain <= 0):
        codes.append("BUDGET_EXHAUSTED_WITHOUT_PRACTICE_GAIN")
    return codes
