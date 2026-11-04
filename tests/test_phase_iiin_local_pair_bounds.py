import numpy as np
import pytest

from experiments.phase_iiin.local_pair_bounds import minimum_pair_norm


def test_orthogonal_pair_exact_threshold_and_nonnegative_kkt_certificate():
    result = minimum_pair_norm([[3., 0.], [0., 4.]], [-.03, -.08])
    assert result["status"] == "KKT_CERTIFIED_LOCAL_LINEAR_BOUND"
    assert result["minimum_norm"] == pytest.approx(np.sqrt(.01 ** 2 + .02 ** 2))
    certificate = result["kkt_certificate"]
    assert certificate["dual_nonnegative"]
    assert certificate["certified_norm_lower_bound"] == pytest.approx(result["minimum_norm"])
    assert abs(certificate["duality_gap"]) < 1e-15
    assert certificate["maximum_primal_violation"] < 1e-12


def test_single_active_constraint_and_already_correct_pair():
    single = minimum_pair_norm([[1., 0.], [2., 0.]], [-1., -1.])
    assert single["minimum_norm"] == pytest.approx(1.)
    assert single["active_set"] == [0]
    both = minimum_pair_norm([[1., 0.], [0., 1.]], [1., 2.])
    assert both["minimum_norm"] == 0 and both["active_set"] == []


def test_degenerate_conflicting_rows_are_conservative_not_global_infeasibility():
    result = minimum_pair_norm([[1., 0.], [-1., 0.]], [-1., -1.])
    assert result["status"] == "INCONCLUSIVE_NO_CERTIFIED_WITNESS"
    assert result["minimum_norm"] is None
    assert "not an exact infeasibility certificate" in result["interpretation"]
    zero = minimum_pair_norm([[0., 0.], [1., 0.]], [-1., -1.])
    assert zero["status"] == "LOCAL_LINEAR_INFEASIBLE_ZERO_ROW"
    assert zero["model_forward_calls"] == zero["model_backward_calls"] == 0


def test_negative_cosine_can_still_have_a_certified_positive_local_solution():
    result = minimum_pair_norm([[1., 0.], [-.5, np.sqrt(.75)]], [-1., -1.])
    assert result["status"] == "KKT_CERTIFIED_LOCAL_LINEAR_BOUND"
    assert result["minimum_norm"] == pytest.approx(2.)
    assert min(result["dual_multipliers"]) > 0


def test_near_singular_ambiguous_bound_does_not_claim_a_solution():
    result = minimum_pair_norm([[1., 0.], [-1., 1e-9]], [-1., -1.])
    assert result["status"] == "INCONCLUSIVE_NO_CERTIFIED_WITNESS"


@pytest.mark.parametrize("rows,margins", [([[1., 0.]], [-1.]),
    ([[1., 0.], [0., float('nan')]], [-1., -1.]), ([[1., 0.], [0., 1.]], [-1.])])
def test_malformed_or_nonfinite_inputs_refused(rows, margins):
    with pytest.raises(ValueError):
        minimum_pair_norm(rows, margins)
