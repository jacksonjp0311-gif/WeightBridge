from dataclasses import replace

import pytest

from weightbridge.adaptation.bic_guard import MeasuredProbe, select_measured_probe


def test_guard_rejects_lower_loss_with_row_regressions():
    parent = MeasuredProbe((0.0, 0.0), 0.8, (True, False, False), (True, False))
    earlier_flip = MeasuredProbe((0.2, 0.2), 0.3, (True, True, True), (False, False))
    net_gain_with_regression = MeasuredProbe((0.2, -0.2), 0.4, (False, True, True), (True, False))
    safe = MeasuredProbe((0.2, 0.0), 0.6, (True, True, False), (True, False))

    assert select_measured_probe(parent, [earlier_flip, net_gain_with_regression, safe]) is safe
    assert select_measured_probe(parent, [earlier_flip, net_gain_with_regression]) is None


def test_guard_can_require_target_gain_in_addition_to_objective_improvement():
    parent = MeasuredProbe((0.0, 0.0), 0.8, (True, False), (True, False))
    same_predictions = replace(parent, coefficients=(0.1, 0.0), objective=0.6)
    objective_increase = replace(same_predictions, objective=0.9, target_correct=(True, True))

    assert select_measured_probe(parent, [same_predictions]) is same_predictions
    assert select_measured_probe(parent, [same_predictions, objective_increase], require_target_gain=True) is None
    assert select_measured_probe(parent, [replace(parent, objective=0.2)]) is None


@pytest.mark.parametrize("invalid", [{"objective": float("nan")}, {"preservation_first_choice": (True,)}])
def test_guard_rejects_incomplete_or_nonfinite_measurements(invalid):
    parent = MeasuredProbe((0.0, 0.0), 0.8, (True, False), (True, False))
    probe = MeasuredProbe((0.1, 0.0), 0.6, (True, True), (True, False))

    with pytest.raises(ValueError):
        select_measured_probe(parent, [probe, replace(probe, **invalid)])
