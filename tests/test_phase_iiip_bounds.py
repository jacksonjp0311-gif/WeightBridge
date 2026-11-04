"""Independent reporting checks; these fixtures are not real-model evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.phase_iiip.independent_bounds import verify_relaxation_bound
from weightbridge.adaptation.right_factor_geometry import solve_block_relaxation


def blocks():
    return np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]])


def test_second_implementation_checks_frobenius_and_rank_one_spectral_bounds():
    data = blocks()
    receipt = solve_block_relaxation(data, [1., 1.], radius=1.5).record()
    checked = verify_relaxation_bound(data, [1., 1.], receipt)
    assert checked["certified_norm_lower_bound"] == pytest.approx(np.sqrt(2), abs=1e-10)
    assert checked["rank_one_norm_lower_bound"] == pytest.approx(2., abs=1e-10)
    assert not checked["block_radius_excluded"] and checked["rank_one_radius_excluded"]
    assert not checked["formal_interval_proof"]
    assert checked["model_forward_calls"] == checked["model_backward_calls"] == 0


@pytest.mark.parametrize("field", ["dual_objective", "dual_numerator", "dual_matrix_frobenius_norm",
    "dual_matrix_spectral_norm", "rank_one_norm_lower_bound", "certified_norm_lower_bound",
    "dual_cauchy_norm_lower_bound", "dual_objective_norm_lower_bound", "dual_spectral_padding"])
def test_independent_reporter_rejects_each_tampered_arithmetic_component(field):
    data = blocks()
    receipt = deepcopy(solve_block_relaxation(data, [.01, .02], radius=.025).record())
    receipt["certificate"][field] += .1
    with pytest.raises(ValueError, match="component"):
        verify_relaxation_bound(data, [.01, .02], receipt)


def test_rescaling_original_rows_preserves_independently_checked_bounds():
    data, rhs = blocks(), np.array([.01, .02])
    original = verify_relaxation_bound(data, rhs, solve_block_relaxation(data, rhs, radius=.025).record())
    scale = np.array([1000., .003])
    scaled = data*scale[:, None, None]
    result = verify_relaxation_bound(scaled, rhs*scale,
        solve_block_relaxation(scaled, rhs*scale, radius=.025).record())
    assert result["certified_norm_lower_bound"] == pytest.approx(original["certified_norm_lower_bound"], abs=1e-10)
    assert result["rank_one_norm_lower_bound"] == pytest.approx(original["rank_one_norm_lower_bound"], abs=1e-10)


def test_zero_spectral_denominator_and_exact_zero_contradiction_are_explicit():
    zero = np.zeros((1, 2, 2))
    satisfied = solve_block_relaxation(zero, [-1.], radius=.025).record()
    checked = verify_relaxation_bound(zero, [-1.], satisfied)
    assert checked["rank_one_norm_lower_bound"] == 0.
    assert not checked["block_radius_excluded"]
    impossible = solve_block_relaxation(zero, [1.], radius=.025).record()
    assert verify_relaxation_bound(zero, [1.], impossible)["status"] == "EXACT_ZERO_BLOCK_CONTRADICTION_CHECKED"
    impossible["contradiction_row_id"] = "invented"
    with pytest.raises(ValueError, match="contradiction"):
        verify_relaxation_bound(zero, [1.], impossible)


def test_input_tampering_negative_duals_and_unearned_cap_exclusion_are_refused():
    data, rhs = blocks(), [.01, .02]
    receipt = solve_block_relaxation(data, rhs, radius=.025).record()
    with pytest.raises(ValueError, match="input identity"):
        verify_relaxation_bound(data*2, rhs, receipt)
    changed = deepcopy(receipt)
    changed["certificate"]["original_dual_multipliers"][0] = -1.
    with pytest.raises(ValueError, match="nonnegative"):
        verify_relaxation_bound(data, rhs, changed)
    changed = deepcopy(receipt)
    changed["status"] = "CERTIFIED_NORM_BOUND_EXCEEDED"
    with pytest.raises(ValueError, match="radius exclusion"):
        verify_relaxation_bound(data, rhs, changed)
