import json

import numpy as np
import pytest

from weightbridge.adaptation.joint_feasibility import solve_joint_feasibility, solve_minimum_norm


def test_orthogonal_joint_minimum_norm_matches_analytic_primal_and_dual_bounds():
    rows = np.array([[3., 0.], [0., 4.]])
    rhs = np.array([.03, .08])
    solved = solve_minimum_norm(rows, rhs, radius=.025, row_ids=["first", "second"])
    np.testing.assert_allclose(solved.update, [.01, .02], atol=1e-12)
    np.testing.assert_array_equal(solved.minimum_norm_update, solved.update)
    evidence = solved.record()
    assert evidence["status"] == "LOCAL_LINEAR_FEASIBLE"
    assert evidence["minimum_norm"] == pytest.approx(np.sqrt(.0005), abs=1e-12)
    assert evidence["certified_norm_lower_bound"] <= evidence["minimum_norm"] + 1e-14
    assert evidence["minimum_norm"] - evidence["certified_norm_lower_bound"] < 1e-10
    assert evidence["active_constraint_ids"] == ["first", "second"]
    assert evidence["certificate"]["dual_nonnegative"]
    assert evidence["certificate"]["maximum_primal_violation"] < 1e-12
    assert abs(evidence["certificate"]["duality_gap"]) < 1e-14
    assert evidence["resources"]["forward_calls"] == evidence["resources"]["backward_calls"] == 0
    json.dumps(evidence, allow_nan=False)


def test_individually_reachable_rows_can_jointly_exceed_the_radius():
    solved = solve_minimum_norm([[1., 0.], [0., 1.]], [.02, .02], radius=.025)
    evidence = solved.record()
    assert evidence["individual_norm_bound_failures"] == []
    assert evidence["individual_minimum_norms"] == [.02, .02]
    assert evidence["status"] == "CERTIFIED_NORM_BOUND_EXCEEDED"
    assert solved.update is None
    np.testing.assert_allclose(solved.minimum_norm_update, [.02, .02])
    assert evidence["certified_norm_lower_bound"] > .025
    assert evidence["minimum_norm"] == pytest.approx(np.sqrt(2) * .02)
    assert not evidence["within_radius"]


def test_dual_certificate_recomputes_from_original_rows_and_nonnegative_multipliers():
    rows = np.array([[1000., 0.], [-300., 400.]])
    rhs = np.array([.7, .9])
    solved = solve_minimum_norm(rows, rhs, radius=.001, tolerance=1e-12)
    evidence = solved.record()
    certificate = evidence["certificate"]
    dual = np.asarray(certificate["original_dual_multipliers"])
    assert np.all(dual >= 0)
    vector = rows.T @ dual
    objective = rhs @ dual - .5 * (vector @ vector)
    assert certificate["dual_objective"] == pytest.approx(objective, abs=1e-17)
    assert certificate["certified_norm_lower_bound"] >= certificate["dual_objective_norm_lower_bound"]
    expected_root = np.sqrt(max(0, 2 * (objective - certificate["dual_roundoff_padding"])))
    assert certificate["dual_objective_norm_lower_bound"] == pytest.approx(expected_root)
    assert evidence["status"] == "CERTIFIED_NORM_BOUND_EXCEEDED"
    assert not evidence["bounded_search_failure_proves_infeasibility"]


def test_positive_target_floor_and_preservation_are_separate_constraints():
    target = [[1., 0.]]
    parent = [-.019]
    alone = solve_joint_feasibility(target, parent, target_floor=.001, radius=.025, target_ids=["fork"])
    assert alone.record()["status"] == "LOCAL_LINEAR_FEASIBLE"
    assert alone.record()["predicted_margins"] == pytest.approx([.001])
    assert alone.record()["target_count"] == 1 and alone.record()["preservation_count"] == 0
    joint = solve_joint_feasibility(target, parent, target_floor=.001,
        preservation_rows=[[0., 1.]], preservation_parent_margins=[0.], preservation_floor=.022,
        radius=.025, target_ids=["fork"], preservation_ids=["protected"])
    assert joint.record()["status"] == "CERTIFIED_NORM_BOUND_EXCEEDED"
    assert joint.record()["row_kinds"] == ["target", "preservation"]
    assert joint.record()["predicted_margins"] == pytest.approx([.001, .022])
    assert joint.record()["certified_norm_lower_bound"] > .025
    assert joint.update is None


def test_exact_zero_and_opposed_requirements_have_explicit_local_contradictions():
    constant = solve_minimum_norm([[0., 0.], [1., 0.]], [1e-15, .01], radius=.025)
    assert constant.record()["status"] == "INFEASIBLE_ZERO_ROW"
    assert constant.update is constant.minimum_norm_update is None
    assert constant.record()["contradiction_certificate"]["positive_required_change"] == 1e-15
    opposed = solve_joint_feasibility([[1., 0.]], [-.01], target_floor=.001,
        preservation_rows=[[-1., 0.]], preservation_parent_margins=[.001], preservation_floor=.002)
    assert opposed.record()["status"] == "INFEASIBLE_OPPOSED_CONSTRAINTS"
    assert opposed.record()["contradiction_certificate"]["summed_axis_norm"] == 0
    assert opposed.update is None
    assert "predicted_margins" not in opposed.record()
    json.dumps(opposed.record(), allow_nan=False)


def test_nearly_opposed_rows_exclude_a_radius_without_claiming_global_infeasibility():
    rows = np.array([[1., 0.], [-1., 1e-7]])
    rhs = np.array([.01, .01])
    # An explicit very large vector satisfies the same linear constraints.
    np.testing.assert_allclose(rows @ np.array([.01, 200000.]), rhs, atol=1e-15)
    solved = solve_minimum_norm(rows, rhs, radius=.025, max_sweeps=2)
    assert solved.record()["status"] in {"CERTIFIED_NORM_BOUND_EXCEEDED", "CERTIFIED_RADIUS_SEPARATOR"}
    assert not solved.record()["status"].startswith("INFEASIBLE")
    assert solved.record()["pair_radius_separators"][0]["summed_axis_norm"] > 0
    assert solved.update is None
    assert not solved.record()["bounded_search_failure_proves_infeasibility"]


def test_rounded_normalization_cannot_turn_nonproportional_raw_rows_into_exact_infeasibility():
    # These normalized float rows cancel exactly, although the supplied raw
    # binary-float values are not exactly proportional as rational numbers.
    from fractions import Fraction
    rows = np.array([[8.478938298659354, .9510489111618077],
                     [-23.055806853822705, -2.5860784961428536]])
    axes = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    np.testing.assert_array_equal(axes[0] + axes[1], [0., 0.])
    assert Fraction(float(rows[0, 0])) * Fraction(float(rows[1, 1])) != (
        Fraction(float(rows[1, 0])) * Fraction(float(rows[0, 1])))
    solved = solve_minimum_norm(rows, [.01, .01], radius=.025, max_sweeps=2)
    assert solved.record()["status"] in {"CERTIFIED_NORM_BOUND_EXCEEDED", "CERTIFIED_RADIUS_SEPARATOR"}
    assert "contradiction_certificate" not in solved.record()
    assert not solved.record()["bounded_search_failure_proves_infeasibility"]


def test_bounded_failure_does_not_manufacture_a_minimum_or_a_norm_certificate():
    solved = solve_minimum_norm([[1., 0.], [-.5, np.sqrt(.75)]], [1., 1.], radius=3., max_sweeps=1)
    evidence = solved.record()
    assert evidence["status"] == "BOUNDED_NO_WITNESS"
    assert solved.update is solved.minimum_norm_update is None
    assert evidence["minimum_norm"] is None and not evidence["converged"]
    assert evidence["certified_norm_lower_bound"] < 3
    assert evidence["sweeps"] == 1
    json.dumps(evidence, allow_nan=False)


def test_bounded_primal_witness_is_distinct_from_converged_minimum_norm():
    solved = solve_minimum_norm([[1., 0.], [.5, np.sqrt(.75)]], [1., 1.], radius=2., max_sweeps=1)
    evidence = solved.record()
    assert evidence["status"] == "FEASIBLE_BOUNDED_WITNESS"
    assert solved.update is not None
    assert evidence["best_primal_norm"] <= 2
    assert evidence["minimum_norm"] is None
    assert not evidence["minimum_norm_certified_to_numerical_tolerance"]
    assert np.min(np.array([[1., 0.], [.5, np.sqrt(.75)]]) @ solved.update - 1.) >= -1e-9


def test_duplicate_and_zero_satisfied_rows_remain_valid_and_do_not_require_inverses():
    solved = solve_minimum_norm([[1., 0.], [1., 0.], [0., 0.]], [.01, .01, -1.], radius=.025)
    np.testing.assert_allclose(solved.update, [.01, 0.])
    assert solved.record()["status"] == "LOCAL_LINEAR_FEASIBLE"
    assert solved.record()["individual_minimum_norms"] == [.01, .01, 0.]
    assert solved.record()["constraint_slacks"][2] == 1.
    trivial = solve_minimum_norm([[1., 0.], [0., 1.]], [-1., -2.], radius=.025)
    np.testing.assert_array_equal(trivial.update, [0., 0.])
    assert trivial.record()["minimum_norm"] == 0.


def test_input_rescaling_preserves_minimum_and_source_arrays_are_unchanged():
    rows = np.array([[3., 0.], [1., 2.]])
    rhs = np.array([.03, .04])
    before = rows.copy(), rhs.copy()
    first = solve_minimum_norm(rows, rhs, radius=.025)
    scaled = solve_minimum_norm(rows * np.array([[100.], [.001]]), rhs * [100., .001], radius=.025)
    np.testing.assert_allclose(first.update, scaled.update, atol=1e-10)
    np.testing.assert_array_equal(rows, before[0])
    np.testing.assert_array_equal(rhs, before[1])
    assert first.record()["input_sha256"] != scaled.record()["input_sha256"]


def test_nonzero_tiny_row_is_not_underflowed_into_an_exact_zero_contradiction():
    solved = solve_minimum_norm([[1e-200]], [1e-202], radius=.025)
    assert solved.record()["status"] == "LOCAL_LINEAR_FEASIBLE"
    np.testing.assert_allclose(solved.update, [.01], rtol=0, atol=1e-12)
    assert solved.record()["row_norms"] == [1e-200]
    assert "contradiction_certificate" not in solved.record()


@pytest.mark.parametrize("rows,rhs,kwargs", [
    ([], [], {}), ([[1.]], [float("nan")], {}), ([[float("inf")]], [1.], {}),
    ([[1.]], [1.], {"max_sweeps": 0}), ([[1.]], [1.], {"tolerance": 0}),
    ([[1.], [1.]], [1., 1.], {"row_ids": ["duplicate", "duplicate"]}),
    (np.ones((65, 1)), np.ones(65), {}), (np.ones((1, 1025)), [1.], {}),
])
def test_malformed_or_unbounded_requests_are_refused(rows, rhs, kwargs):
    with pytest.raises(ValueError):
        solve_minimum_norm(rows, rhs, radius=.025, **kwargs)


def test_wrapper_refuses_nonpositive_target_floor_or_unreplayed_preservation():
    with pytest.raises(ValueError, match="strictly positive"):
        solve_joint_feasibility([[1.]], [-1.], target_floor=0.)
    with pytest.raises(ValueError, match="nonnegative parent"):
        solve_joint_feasibility([[1.]], [-1.], preservation_rows=[[1.]], preservation_parent_margins=[-.1])
