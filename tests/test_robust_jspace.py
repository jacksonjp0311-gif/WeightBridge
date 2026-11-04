import numpy as np
import pytest

from weightbridge.adaptation.preservation_geometry import DecisionAnchor
from weightbridge.adaptation.robust_jspace import (
    JSpaceEvidence,
    MarginObservation,
    calibrate_margin_floor,
    calibrate_residuals,
    estimate_window,
    interior_score,
    jspace_recommendation,
    select_preservation,
    solve_robust,
    validate_residuals,
)
from weightbridge.patches.huggingface import _from_float64, _to_float64


def decision(item, group, wording, position, margin):
    return DecisionAnchor(item, f"{item}/{wording}", group, wording, position, margin, 19)


def test_sensitivity_selects_high_margin_risk_and_diversifies_spare_slots():
    low = decision("near-tie", "target", 0, 0, 0.01)
    dangerous = decision("adverse", "target", 0, 1, 2.0)
    same = decision("adverse", "target", 0, 2, 3.0)
    paired = decision("adverse", "target", 1, 1, 2.0)
    positive_0 = decision("control", "positive", 0, 0, 0.5)
    positive_1 = decision("control", "positive", 1, 0, 0.5)
    anchors = (low, dangerous, same, paired, positive_0, positive_1)
    slopes = {row.key: value for row, value in zip(anchors, (2, -100, -130, -30, -1, -1), strict=True)}
    minimum = select_preservation(anchors, budget=1, mode="historical")
    directional = select_preservation(anchors, budget=1, mode="directional", slopes=slopes, radius=0.05)
    assert minimum.anchors == (low,)
    assert directional.anchors == (dangerous,)
    assert directional.selected_risks == (2.5,)
    stratified = select_preservation(anchors, budget=5, required_groups=("target", "positive"),
                                     required_wordings=(0, 1), slopes=slopes, radius=0.05)
    assert stratified.complete
    assert stratified.anchors == (dangerous, positive_0, paired, positive_1, low)
    assert stratified.record()["represented_items"] == 3


def test_unmeasured_decisions_cannot_supply_directional_coverage():
    measured = decision("target", "target", 0, 0, 1.0)
    unmeasured = decision("control", "copy", 1, 0, 0.05)
    sampled = select_preservation((measured, unmeasured), budget=4, required_wordings=(0, 1),
                                  slopes={measured.key: 0.0}, radius=0.05)
    assert sampled.anchors == (measured,)
    assert sampled.missing_groups == ("copy",)
    assert ("copy", 1) in sampled.unavailable_strata
    assert sampled.record()["unmeasured_decisions"] == 1


def test_residual_allowance_counts_underlying_items_once_and_uses_held_items():
    many_tokens = [MarginObservation("a", "a/0", position, 1.0, 0.9) for position in range(10)]
    observations = [*many_tokens, MarginObservation("b", "b/0", 1, 1.0, 0.7),
                    MarginObservation("b", "b/1", 2, 1.0, 0.6), MarginObservation("c", "c/0", 0, 1.0, 1.1)]
    calibration = calibrate_residuals(observations, quantile=0.8)
    assert calibration.allowance == pytest.approx(0.4)
    assert calibration.items == ("a", "b", "c")
    assert calibration.record()["item_count"] == 3
    assessed = validate_residuals(calibration, [MarginObservation("d", "d/0", 0, 1.0, 0.8),
                                                MarginObservation("e", "e/1", 0, 1.0, 0.4)])
    assert assessed["item_coverage"] == 0.5
    assert calibrate_residuals(many_tokens).allowance is None
    with pytest.raises(ValueError, match="disjoint underlying items"):
        validate_residuals(calibration, [MarginObservation("b", "unseen-prompt", 8, 1.0, 1.0)])


def test_frozen_higher_residual_quantile_at_five_items():
    # At q=.8 a nearest-rank rule would return 3; the frozen higher method returns 4.
    points = [MarginObservation(str(i), str(i), 0, 4.0, 4.0 - i) for i in range(5)]
    assert calibrate_residuals(points, quantile=0.8).allowance == 4.0


def test_predeclared_floor_uses_lower_parent_quantile_and_can_abstain():
    result = calibrate_margin_floor([0.125, 0.5, 2.0, 4.0], [0.025, 0.05, 0.125])
    assert result["parent_margin_quantile"] == 0.125
    assert result["selected_floor"] == 0.05
    unsupported = calibrate_margin_floor([0.0, 0.5], [0.05])
    assert unsupported["selected_floor"] is None


def test_robust_floor_tightens_paired_window_and_excludes_tie():
    window = estimate_window([-0.125, -0.25, -0.125], [1, 1, 1], [0.5], [-1], radius=0.75,
                             margin_floor=0.125, allowance=0.125, target_items=("a", "a", "b"))
    assert window["alpha_preserve_max"] == 0.25
    assert window["per_target"][0]["safe_width"] == 0.125
    assert window["paired_item_windows"][0]["safe_interval"] is None
    assert window["paired_item_windows"][1]["safe_width"] == 0.125
    assert window["joint_safe_width"] == 0.0
    solution = solve_robust([1, 0], [[-1, 0]], [0.5], radius=0.75, margin_floor=0.125, allowance=0.125)
    np.testing.assert_allclose(solution.update, [0.25, 0.0], atol=1e-9)
    impossible = solve_robust([1, 0], [[1, 0]], [0.0], radius=0.1, margin_floor=0.125, allowance=0.125)
    assert impossible.update is None and impossible.status == "halfspace_outside_radius"


def test_bf16_residual_can_flip_predicted_positive_margin():
    source = np.array([1.0, 0.0])
    proposed = np.array([0.003, -0.0025])
    rounded = _to_float64(_from_float64(source + proposed, "BF16"), "BF16")
    parent_margin = 0.002
    measured = MarginObservation("bf16-item", "bf16-item/0", 0, parent_margin + proposed.sum(),
                                 parent_margin + (rounded - source).sum())
    assert measured.predicted_margin > 0 > measured.measured_margin
    allowance = calibrate_residuals([measured], minimum_items=1).allowance
    assert allowance == pytest.approx(-measured.residual)


def test_interior_prefers_more_measured_slack_only_after_real_practice_gate():
    settings = {"practice_pass": True, "practice_gain": 1, "measured_min_margin": 0.02,
                "feasible_window_width": 0.001, "realized_norm": 0.01}
    small_slack = interior_score(measured_slack=-0.02, **settings)
    more_slack = interior_score(measured_slack=-0.01, **settings)
    assert more_slack > small_slack
    assert interior_score(measured_slack=1.0, **{**settings, "practice_pass": False}) is None
    assert interior_score(measured_slack=1.0, **{**settings, "measured_min_margin": 0.0}) is None
    assert interior_score(measured_slack=1.0, **{**settings, "feasible_window_width": 0.0}) is None


def test_meta_preservation_validation_cannot_be_bypassed_by_a_feasible_window():
    evidence = {"measurement_valid": True, "persistent_errors": 2, "window_width": 0.01, "practice_pass": True,
                "preservation_validation_required": True}
    assert jspace_recommendation(JSpaceEvidence(**evidence)).action == "preservation_validate"
    failed = jspace_recommendation(JSpaceEvidence(**evidence, preservation_validation_pass=False, selection_pass=True))
    assert failed.reason == "preservation_validation_failed"
    qualified = jspace_recommendation(JSpaceEvidence(**evidence, preservation_validation_pass=True))
    assert qualified.action == "independent_select"
    assert qualified.record()["promotion_authority"] is False
