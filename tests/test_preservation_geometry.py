import json
from pathlib import Path

import numpy as np
import pytest

from weightbridge.adaptation.preservation_geometry import (
    DecisionAnchor,
    feasible_intervals,
    project_nullspace,
    sample_anchors,
    solve_inequalities,
)


GROUPS = ("target", "positive", "mixed-sign", "copy")


def anchor(group, wording, margin, *, item=None, position=0):
    item = item or group
    return DecisionAnchor(item, f"{item}/{wording}", group, wording, position, margin, 17)


def test_historical_global_crowds_out_groups_and_stratified_represents_all_wordings():
    points = tuple(anchor(group, wording, group_index + 0.1 * wording)
                   for group_index, group in enumerate(GROUPS, 1) for wording in (0, 1))
    extra = anchor("target", 0, 0.2, position=2)
    global_sample = sample_anchors((*points, extra), budget=4, mode="global",
                                   required_groups=GROUPS, required_wordings=(0, 1))
    assert global_sample.anchors[0] == extra
    assert len({point.prompt_id for point in global_sample.anchors}) == 4
    assert global_sample.missing_groups == ("mixed-sign", "copy")
    assert not global_sample.complete
    sample = sample_anchors((*points, extra), budget=8, required_groups=GROUPS, required_wordings=(0, 1))
    assert sample.complete
    assert tuple(point.group for point in sample.anchors[:4]) == GROUPS
    assert set(sample.represented_strata) == {(group, wording) for group in GROUPS for wording in (0, 1)}


def test_global_sampler_replays_frozen_phase_i_anchor_positions():
    trial = Path(__file__).resolve().parents[1] / "results/phase-iii-i/causal-fork-v1/trial"
    scan = json.loads((trial / "scan.json").read_bytes())
    trajectories = json.loads((trial / "diagnosis.json").read_bytes())["trajectories"]
    practice = {row["id"]: row for row in json.loads((trial / "practice.json").read_bytes())["measurement"]["rows"]}
    anchors = []
    for path, trajectory in zip(scan["parent_trajectories"], trajectories, strict=True):
        row = practice[trajectory["prompt_id"]]
        for position, (margin, observed, replayed) in enumerate(zip(
                path["margins"], path["token_ids"], path["replay_argmax_ids"], strict=True)):
            if margin >= 0 and observed == replayed:
                anchors.append(DecisionAnchor(row["item"], row["id"], row["group"], row["wording"], position, margin, observed))
    chosen = sample_anchors(anchors, budget=4, mode="global")
    assert [anchor.key for anchor in chosen.anchors] == [
        (trajectories[entry["trajectory_index"]]["prompt_id"], entry["position"])
        for entry in scan["protected_anchors"]
    ]
    assert {anchor.group for anchor in chosen.anchors} == {"target"}


def test_unavailable_group_and_unrepresented_wording_are_explicit():
    points = (anchor("target", 0, 0.2), anchor("target", 1, 0.3), anchor("positive", 0, 0.1))
    sample = sample_anchors(points, budget=2, required_groups=("target", "positive", "copy"), required_wordings=(0, 1))
    assert sample.missing_groups == ("copy",)
    assert sample.unavailable_groups == ("copy",)
    assert sample.missing_strata == (("target", 1), ("positive", 1), ("copy", 0), ("copy", 1))
    assert sample.unavailable_strata == (("positive", 1), ("copy", 0), ("copy", 1))
    assert not sample.complete


def test_risk_uses_direction_sensitivity_and_keeps_group_coverage():
    vulnerable = anchor("target", 0, 2.0, item="sensitive")
    safe = anchor("target", 0, 0.001, item="tiny-margin")
    control = anchor("positive", 0, 1.0)
    points = (safe, vulnerable, control)
    slopes = {point.key: slope for point, slope in zip(points, (1.0, -30.0, -0.2), strict=True)}
    sample = sample_anchors(points, budget=2, mode="risk", required_groups=("target", "positive"),
                            required_wordings=(0,), slopes=slopes, radius=0.1)
    assert sample.complete and sample.anchors == (vulnerable, control)
    with pytest.raises(ValueError, match="measured direction slope"):
        sample_anchors(points, budget=2, mode="risk", slopes={}, radius=0.1)


def test_feasible_interval_reports_each_target_and_excludes_a_boundary_tie():
    result = feasible_intervals([-0.1, -0.25, -0.05], [1, 1, -1], [0.25], [-1], radius=0.5)
    assert result["alpha_preserve_max"] == 0.25
    assert result["per_target"][0]["safe_interval"] == {
        "lower": 0.1, "upper": 0.25, "lower_open": True, "upper_open": False,
    }
    assert result["per_target"][1]["safe_interval"] is None
    assert result["per_target"][1]["reason"] == "preservation_conflict"
    assert result["per_target"][2]["alpha_target_min"] is None
    assert result["all_target_interval"] is None


def test_positive_slope_can_restore_a_floor_and_conflicting_intervals_report_empty():
    result = feasible_intervals([-0.05], [1], [0.0, 0.3], [1, -1], radius=1, floor=[0.1, 0.1])
    assert result["protected_interval"]["lower"] == 0.1
    assert result["protected_interval"]["upper"] == pytest.approx(0.2)
    assert result["all_target_interval"] == result["protected_interval"]
    impossible = feasible_intervals([-0.05], [1], [0.0, 0.15], [1, -1], radius=1, floor=[0.1, 0.1])
    assert impossible["protected_interval"] is None


def test_inequality_uses_available_margin_headroom_when_nullspace_has_no_useful_direction():
    assert project_nullspace([1.0, 0.0], [[-1.0, 0.0]], radius=0.5) is None
    result = solve_inequalities([1.0, 0.0], [[-1.0, 0.0]], [0.3], radius=0.5, floor=0.1)
    assert result.status == "projected_feasible"
    np.testing.assert_allclose(result.update, [0.2, 0.0], atol=1e-9)
    assert result.target_alignment == pytest.approx(0.2)
    assert result.max_violation <= 1e-9 and result.radius_violation <= 1e-9
    assert "update" not in result.record()


def test_solver_reports_only_provable_single_conflicts_and_no_witness_for_other_exhaustion():
    constant = solve_inequalities([1.0, 0.0], [[0.0, 0.0]], [0.0], radius=0.5, floor=0.1)
    outside = solve_inequalities([1.0, 0.0], [[1.0, 0.0]], [-0.6], radius=0.5)
    contradictory = solve_inequalities([1.0, 0.0], [[1.0, 0.0], [-1.0, 0.0]], [-0.2, -0.2],
                                        radius=0.5, max_iterations=3)
    assert (constant.status, outside.status, contradictory.status) == (
        "impossible_constant_constraint", "halfspace_outside_radius", "no_feasible_witness",
    )
    assert all(result.update is None for result in (constant, outside, contradictory))
    assert contradictory.iterations == 3 and contradictory.max_violation > 0


def test_projection_handles_interacting_halfspaces_and_gradient_scales():
    result = solve_inequalities([1, 1], [[-3, 0], [-1, -1], [0, 1]], [0.9, 0.9, -0.2], radius=1.0)
    assert result.status == "projected_feasible"
    np.testing.assert_allclose(result.update, [0.3, 0.6], atol=1e-8)
