"""Algebra fixtures only; no synthetic result constitutes model evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from weightbridge.adaptation.joint_feasibility import solve_minimum_norm
from weightbridge.adaptation.right_factor_geometry import (
    DEFAULT_SEARCH, canonical_factors, outer_value_gradient, search_right_factor,
    solve_block_relaxation, verify_block_certificate,
)


def diagonal_blocks(scale=1.):
    return scale*np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]])


def test_block_gram_relaxation_matches_analytic_matrix_minimum_and_dual():
    g, b = diagonal_blocks(), np.array([.01, .02])
    solved = solve_block_relaxation(g, b, radius=.025, tolerance=1e-12)
    np.testing.assert_allclose(solved.update, np.diag(b), atol=1e-12)
    record = solved.record()
    assert record["minimum_norm"] == pytest.approx(np.linalg.norm(b))
    assert record["status"] == "LOCAL_LINEAR_FEASIBLE"
    assert record["matrix_installation_authorized"] is False
    assert record["resources"]["forward_calls"] == record["resources"]["backward_calls"] == 0
    replay = verify_block_certificate(g, b, record)
    assert replay["certified_norm_lower_bound"] <= np.linalg.norm(b)+1e-12
    assert replay["rank_one_norm_lower_bound"] == pytest.approx(.025, abs=1e-10)
    json.dumps(record, allow_nan=False)


def test_fixed_right_fails_but_free_right_finds_within_cap_rank_one_witness():
    g, b = diagonal_blocks(), np.array([.01, .01])
    fixed = solve_minimum_norm(g@np.array([1., 0.]), b, radius=.025)
    assert fixed.update is None and fixed.record()["status"] == "INFEASIBLE_ZERO_ROW"
    result = search_right_factor(g, b, [1., 0.], radius=.025)
    assert result.record()["status"] == "LOCAL_RANK_ONE_WITNESS"
    assert np.linalg.norm(result.right) == pytest.approx(1.)
    assert np.linalg.norm(result.left) == pytest.approx(.02, abs=1e-8)
    assert np.min(np.einsum("ijk,j,k->i", g, result.left, result.right)-b) >= -1e-8
    assert result.record()["selected_start"] == "relaxed-balanced"
    assert result.record()["inner_calls"] <= result.record()["maximum_inner_calls"] <= 132


def test_relaxed_matrix_can_fit_budget_while_rank_one_bound_excludes_it():
    # Frobenius(diag(1,1))=sqrt(2), but anyrank-one update with both
    # diagonal entries>=1 has norm>=2, even without the off-diagonal zeros.
    g, b = diagonal_blocks(), np.ones(2)
    relaxed = solve_block_relaxation(g, b, radius=1.5)
    assert relaxed.update is not None
    assert np.linalg.matrix_rank(relaxed.update) == 2
    assert relaxed.record()["certificate"]["rank_one_norm_lower_bound"] > 1.5
    result = search_right_factor(g, b, [1., 1.], radius=1.5, relaxation=relaxed)
    assert result.left is result.right is None
    assert result.record()["status"] == "RANK_ONE_NORM_BOUND_EXCEEDED"
    assert result.record()["inner_calls"] == 0
    assert not result.record()["global_rank_one_optimum_claimed"]


def test_factor_normalization_and_sign_preserve_exact_outer_product():
    original_u, original_v = np.array([2., -3.]), np.array([-4., 1.])
    u, v = canonical_factors(original_u, original_v)
    np.testing.assert_allclose(np.outer(u, v), np.outer(original_u, original_v), atol=1e-14)
    assert np.linalg.norm(v) == pytest.approx(1.)
    assert v[np.argmax(np.abs(v))] > 0
    again_u, again_v = canonical_factors(u, v)
    np.testing.assert_allclose(again_u, u, atol=1e-14)
    np.testing.assert_allclose(again_v, v, atol=1e-14)
    np.testing.assert_allclose(canonical_factors(-original_u, -original_v)[0], u)
    with pytest.raises(ValueError, match="nonzero"):
        canonical_factors([1.], [0.])


def test_outer_original_row_dual_derivative_matches_finite_differences():
    # Single active row, non-unit physical scaling. Normalized multipliers
    # give the wrong envelope derivative by the row-norm factor.
    g = np.array([[[3., 1.], [2., -.5]]])
    b, theta = np.array([.7]), .4
    v = np.array([np.cos(theta), np.sin(theta)])
    inner = solve_minimum_norm(g@v, b, radius=1., tolerance=1e-12)
    value, gradient, evidence = outer_value_gradient(g, b, v, inner, tolerance=1e-10)
    assert evidence["qualified"]
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    step = 1e-5
    def objective(angle):
        factor = np.array([np.cos(angle), np.sin(angle)])
        solution = solve_minimum_norm(g@factor, b, radius=1., tolerance=1e-12)
        return .5*float(solution.minimum_norm_update@solution.minimum_norm_update)
    finite = (objective(theta+step)-objective(theta-step))/(2*step)
    assert float(gradient@tangent) == pytest.approx(finite, rel=1e-7, abs=1e-9)
    assert value == pytest.approx(objective(theta), abs=1e-12)
    dual = np.asarray(evidence["original_dual_multipliers"])
    np.testing.assert_allclose(gradient, -np.einsum("i,ijk,j->k", dual, g, inner.minimum_norm_update))
    normalized = np.asarray(inner.record()["primal_witness"]["normalized_dual_multipliers"])
    assert not np.allclose(dual, normalized)


def test_numerical_lower_bound_stops_outer_before_orientations():
    g, b = diagonal_blocks(), np.array([.02, .02])
    result = search_right_factor(g, b, [1., 1.], radius=.025)
    assert result.record()["status"] == "BLOCK_RELAXATION_REFUSAL"
    assert result.record()["relaxation"]["certified_norm_lower_bound"] > .025
    assert result.record()["inner_calls"] == 0
    assert result.left is None


@pytest.mark.parametrize("component", ["certified_norm_lower_bound", "dual_objective",
    "dual_cauchy_norm_lower_bound", "rank_one_norm_lower_bound", "dual_matrix_spectral_norm"])
def test_independent_dual_checker_rejects_tampered_bound_components(component):
    g, b = diagonal_blocks(), [.01, .02]
    record = deepcopy(solve_block_relaxation(g, b, radius=.025).record())
    record["certificate"][component] += .1
    with pytest.raises(ValueError, match="does not follow"):
        verify_block_certificate(g, b, record)


def test_independent_checker_rejects_multipliers_and_raw_block_tampering():
    g, b = diagonal_blocks(), [.01, .02]
    record = deepcopy(solve_block_relaxation(g, b, radius=.025).record())
    record["certificate"]["original_dual_multipliers"][0] = -1.
    with pytest.raises(ValueError, match="nonnegative"):
        verify_block_certificate(g, b, record)
    record = solve_block_relaxation(g, b, radius=.025).record()
    with pytest.raises(ValueError, match="input identity"):
        verify_block_certificate(g*2, b, record)


def test_zero_blocks_and_zero_denominator_never_create_false_finite_witness():
    zero = solve_block_relaxation(np.zeros((1, 2, 2)), [.01], radius=.025)
    assert zero.update is None and zero.record()["status"] == "INFEASIBLE_ZERO_BLOCK"
    assert verify_block_certificate(np.zeros((1, 2, 2)), [.01], zero.record())["zero_block_verified"]
    trivial = solve_block_relaxation(np.zeros((1, 2, 2)), [-.01], radius=.025)
    np.testing.assert_array_equal(trivial.update, np.zeros((2, 2)))
    assert trivial.record()["certificate"]["zero_spectral_denominator"]
    assert trivial.record()["certificate"]["rank_one_norm_lower_bound"] == 0.
    json.dumps(trivial.record(), allow_nan=False)


def test_bounded_no_witness_is_not_global_infeasibility():
    g = np.array([[[1., 0.]], [[-.5, np.sqrt(.75)]]])
    result = solve_block_relaxation(g, [1., 1.], radius=3., max_sweeps=1)
    assert result.record()["status"] == "BOUNDED_NO_WITNESS"
    assert not result.record()["converged"]
    assert result.record()["minimum_norm"] is None
    assert not result.record()["bounded_search_failure_proves_infeasibility"]


def test_outer_refuses_unqualified_kkt_derivative():
    g = np.array([[[1., 0.], [0., 0.]], [[-.5, 0.], [np.sqrt(.75), 0.]]])
    inner = solve_minimum_norm(g@np.array([1., 0.]), [1., 1.], radius=3., max_sweeps=1)
    value, gradient, evidence = outer_value_gradient(g, [1., 1.], [1., 0.], inner)
    assert value is gradient is None and not evidence["qualified"]


def test_block_dimension_and_outer_budgets_are_enforced_without_flattening():
    # A large selectedblock is allowed atthisexplicitGraminterface; the
    # ordinary vector solver must still refuse dimensionsabove1024.
    g = np.zeros((1, 1024, 16)); g[0, 0, 0] = 1.
    result = solve_block_relaxation(g, [.01], radius=.025)
    assert result.update.shape == (1024, 16)
    with pytest.raises(ValueError, match="bounded finite rows"):
        solve_minimum_norm(g.reshape(1, -1), [.01], radius=.025)
    with pytest.raises(ValueError, match="at most64"):
        solve_block_relaxation(np.zeros((1, 1025, 16)), [.01], radius=.025)
    with pytest.raises(ValueError, match="outer budget"):
        search_right_factor(diagonal_blocks(), [.01, .01], [1., 1.], radius=.025, settings={"starts": 5})


def test_search_is_deterministic_and_deduplicates_canonical_start_states():
    g, b = diagonal_blocks(), [.01, .01]
    first = search_right_factor(g, b, [1., 0.], radius=.025)
    second = search_right_factor(g, b, [-1., 0.], radius=.025)
    np.testing.assert_array_equal(first.left, second.left)
    np.testing.assert_array_equal(first.right, second.right)
    assert first.record()["duplicate_starts"]
    assert first.record()["inner_calls"] == second.record()["inner_calls"]
    assert first.record()["settings"] == DEFAULT_SEARCH


def test_supplied_relaxation_must_match_cap_and_decision_order():
    g, b = diagonal_blocks(), [.01, .01]
    result = solve_block_relaxation(g, b, radius=.03)
    with pytest.raises(ValueError, match="different authorization"):
        search_right_factor(g, b, [1., 1.], radius=.025, relaxation=result)


def test_over_cap_inner_minimum_remains_usable_diagnostic_for_outer_descent():
    g, b = np.array([[[1., 4.]]]), [.02]
    result = search_right_factor(g, b, [1., 0.], radius=.015)
    inherited = next(row for row in result.record()["starts"] if row["name"] == "inherited")
    assert inherited["steps"][0]["inner"]["minimum_norm"] == pytest.approx(.02)
    assert not inherited["steps"][0]["inner"]["within_radius"]
    assert inherited["steps"][0]["derivative"]["qualified"]
    assert len(inherited["steps"]) > 1
    assert inherited["steps"][-1]["inner"]["minimum_norm"] < .015
    assert result.left is not None and np.linalg.norm(result.left) < .005


def test_bounded_outer_failure_does_not_claim_global_rank_one_failure():
    g, b = diagonal_blocks(), [.01, .01]
    result = search_right_factor(g, b, [1., 0.], radius=.025,
        settings={"starts": 1, "iterations": 0})
    assert result.record()["status"] == "BOUNDED_NO_RANK_ONE_WITNESS"
    assert not result.record()["global_rank_one_optimum_claimed"]
    assert result.record()["relaxation"]["within_radius"]
    assert result.left is None and result.record()["inner_calls"] == 1


def test_small_gram_result_matches_existing_vector_solver_without_changing_its_limits():
    g = np.array([[[3., 0.], [1., 0.]], [[1., 2.], [0., 1.]]])
    b = [.02, .03]
    relaxed = solve_block_relaxation(g, b, radius=.025, tolerance=1e-12)
    vector = solve_minimum_norm(g.reshape(2, 4), b, radius=.025, tolerance=1e-12)
    np.testing.assert_allclose(relaxed.minimum_norm_update, vector.minimum_norm_update.reshape(2, 2), atol=1e-12)
    np.testing.assert_allclose(relaxed.record()["primal_witness"]["original_dual_multipliers"],
                               vector.record()["primal_witness"]["original_dual_multipliers"], atol=1e-12)
