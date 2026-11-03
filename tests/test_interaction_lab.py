import json
from pathlib import Path

import numpy as np

from experiments.phase_iiib import interaction_lab


PROTOCOL = Path(interaction_lab.__file__).with_name("synthetic_protocol.json")


def test_scalar_derivatives_match_finite_differences_away_from_stationarity():
    point = np.array([0.31, -0.17])
    target, weight = -0.4, 0.506

    def loss(values):
        return weight * (values[0] * values[1] - target) ** 2

    expected_gradient, expected_hessian = interaction_lab.product_derivatives(point, target, weight)
    gradient, hessian = interaction_lab.finite_derivatives(loss, point, 1e-4)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-7, rtol=0)
    np.testing.assert_allclose(hessian, expected_hessian, atol=1e-7, rtol=0)


def test_controlled_interaction_protocol_exercises_positive_and_negative_cases():
    result = interaction_lab.run(json.loads(PROTOCOL.read_bytes()))
    assert result["exploratory_gate_passed"], result["gates"]
    dormant = result["scalar"]["two_weight"]["shared_compiler"]
    assert dormant["interaction"]["actual_loss"] < dormant["first_order"]["actual_loss"]
    active = result["scalar"]["active_first_order"]["shared_compiler"]
    assert active["first_order"]["actual_loss"] <= active["interaction"]["actual_loss"]
    harmful = result["scalar"]["harmful_interaction"]["shared_compiler"]
    assert harmful["interaction"]["predicted_loss"] < harmful["first_order"]["predicted_loss"]
    assert harmful["interaction"]["actual_loss"] > harmful["first_order"]["actual_loss"]
    assert not harmful["interaction"]["accepted"]
