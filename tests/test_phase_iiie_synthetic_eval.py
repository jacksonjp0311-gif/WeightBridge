import json

import numpy as np
import pytest

from experiments.phase_iiie import run_synthetic as study
from experiments.phase_iiie.synthetic_tasks import feature_cost, make_tasks, method_cost_bound


CONFIG = json.loads(study.PROTOCOL.read_bytes())


def test_diagnostic_and_candidate_budget_refuse_before_execution(monkeypatch):
    task = make_tasks("train", seed=2, per_mechanism=2)[0]

    def forbidden(*args, **kwargs):
        pytest.fail("a refused action was executed")

    monkeypatch.setattr(study, "features", forbidden)
    monkeypatch.setattr(study, "run_method", forbidden)
    assert study._probe(task, "D", feature_cost(task, level="D").units - 1) is None
    assert study._candidate(task, "interaction", 1.0, method_cost_bound(task, "interaction").units - 1) is None


def test_hidden_twins_contribute_distinct_outcomes_to_the_same_policy_decision():
    tasks = [task for task in make_tasks("shift", seed=2, per_mechanism=4) if task.twin_id]
    rows = [study._record(task, study._policy(task, "fixed_first_order", CONFIG, None, None, None, None, None),
                          study._truth(task, CONFIG), "fixed_first_order", CONFIG) for task in tasks]
    assert study._twin_summary(rows) == {"pairs": 2, "same_decision_pairs": 2,
                                         "both_hold_pairs": 0, "opposite_verified_outcome_pairs": 2}
    aggregate = study._aggregate(rows)
    assert aggregate["improved"] == aggregate["harmful"] == aggregate["false_admission"] == 2
    assert aggregate["accepted"] == 2
    assert all(row["cost"]["units"] <= CONFIG["budget_units"] for row in rows)


def test_calibration_checks_event_reliability_for_each_method():
    probabilities = np.broadcast_to(np.array([0.2, 0.8]), (2, 3, 2))
    labels = np.broadcast_to(np.array([0, 1]), probabilities.shape)
    result = study._calibration(probabilities, labels, bins=10)
    for method in study.METHODS:
        assert result[method]["improvement"]["brier"] == pytest.approx(.04)
        assert result[method]["harm"]["ece"] == pytest.approx(.2)


def test_fixed_hold_spends_no_diagnostic_or_update_compute(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("fixed HOLD invoked a diagnostic")

    monkeypatch.setattr(study, "_admit", forbidden)
    result = study._policy(None, "always_hold", CONFIG, None, None, None, None, None)
    assert result[0] == "hold" and result[3].units == 0
