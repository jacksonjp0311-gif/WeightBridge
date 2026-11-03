from dataclasses import replace

import numpy as np
import pytest

from weightbridge.adaptation.counterfactual import align_margin, paired_components, paired_loss
from weightbridge.adaptation.meta_controller import Action, EvidenceState, MethodSelector, RiskEstimator
from weightbridge.adaptation.reachability import local_reachability


def test_pair_alignment_ignores_printed_position_and_shared_answer_bias():
    assert align_margin(-1., -4., (0, 1)) == align_margin(-4., -1., (1, 0)) == 3.
    assert paired_components(5., -3.) == (4., 1.)
    assert paired_components(12., 4.) == (4., 8.)
    assert paired_components(-3., 5.) == (-4., 1.)
    assert paired_components(6., 6.) == (0., 6.)
    with pytest.raises(ValueError):
        align_margin(-4., -1., (0, 0))


def test_pair_loss_gradients_and_standalone_direction_penalty():
    point = np.array([3., -.7])
    _, grad = paired_loss(*point, preference_weight=.8)
    for i in range(2):
        step = np.eye(2)[i] * 1e-5
        high = paired_loss(*(point + step), preference_weight=.8)[0]
        low = paired_loss(*(point - step), preference_weight=.8)[0]
        assert grad[i] == pytest.approx((high - low) / 2e-5, abs=1e-8)
    assert paired_loss(5., 1.)[0] > paired_loss(2., -2.)[0]
    assert paired_loss(800., -800.)[0] == 0.


def test_reachability_witness_preserves_signed_behavior_and_norm():
    problem = dict(target_jacobian=[[1., 1.]], desired=[1.], preservation_jacobian=[[1., 0.]],
                   margins=[-.2], signs=[-1.], min_margins=[.1], tolerance=1e-5)
    reachable = local_reachability(**problem, radius=1.1)
    x, y = reachable.witness
    assert reachable.feasible and reachable.status == "feasible"
    assert x <= .1 + 1e-10 and abs(x + y - 1.) < 1.01e-5
    assert np.linalg.norm(reachable.witness) <= 1.1
    outside = local_reachability(**problem, radius=.5)
    assert not outside.feasible and outside.status == "infeasible_bound"
    assert not local_reachability(**problem, radius=.72, max_cycles=1).feasible


def test_host_gates_override_attractive_learned_predictions():
    state = EvidenceState(True, 2, 5., True, preservation_risk=.5, curvature=.3,
                          method_costs=(1., 2., 3.), allowed_methods=("first_order", "constrained"))
    selector = MethodSelector(disagreement_limit=.25)
    p = [[.8, .04], [.85, .15], [.95, .01]]
    assert selector.choose(state, p).action is Action.FIRST_ORDER
    assert selector.choose(replace(state, measurement_valid=False), p).action is Action.MEASURE
    assert selector.choose(replace(state, persistent_errors=0), p).action is Action.HOLD
    assert selector.choose(replace(state, first_order_reachable=None), p).action is Action.REACHABILITY
    assert selector.choose(replace(state, remaining_budget=.5), p).action is Action.HOLD
    assert selector.choose(replace(state, sensitivity_disagreement=.3), p).action is Action.HOLD
    assert selector.choose(replace(state, out_of_support=True), p).action is Action.HOLD
    assert selector.choose(replace(state, first_order_reachable=False, curvature=None,
                                   allowed_methods=("interaction",)), p).action is Action.CURVATURE
    assert selector.choose(state).record()["schema"] == "weightbridge.meta_decision.v1"


def test_logistic_estimator_learns_and_calibrates_on_separate_samples():
    x = np.linspace(-3., 3., 100).reshape(-1, 1)
    labels = np.zeros((100, 3, 2), dtype=int)
    labels[:, 0, 0] = x[:, 0] > 0
    labels[:, 0, 1] = x[:, 0] < 0
    model = RiskEstimator.fit(x[::2], labels[::2]).calibrate(x[1::2], labels[1::2])
    probabilities = model.predict([[-2.], [2.]])
    assert probabilities.shape == (2, 3, 2)
    assert probabilities[1, 0, 0] > .8 and probabilities[0, 0, 1] > .8
    assert probabilities[0, 0, 0] < .2 and probabilities[1, 0, 1] < .2
