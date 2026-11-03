from dataclasses import astuple, replace

import numpy as np
import pytest

from experiments.phase_iiie.synthetic_tasks import (
    Cost,
    FAMILIAR,
    HELD,
    METHODS,
    Margins,
    Task,
    _project,
    diagnose,
    diagnosis_cost,
    feature_cost,
    features,
    make_tasks,
    method_cost_bound,
    run_method,
)


def test_populations_have_disjoint_ids_and_held_mechanisms():
    splits = {name: make_tasks(name, seed=31, per_mechanism=6)
              for name in ("train", "calibration", "test", "shift")}
    ids = [task.id for tasks in splits.values() for task in tasks]
    assert len(ids) == len(set(ids)) == 120
    for name in ("train", "calibration", "test"):
        assert {task.mechanism for task in splits[name]} == set(FAMILIAR)
    assert {task.mechanism for task in splits["shift"]} == set(HELD)
    expected = [features(task, "D") for task in splits["test"]]
    assert [features(task, "D") for task in make_tasks("test", seed=31, per_mechanism=6)] == expected


def test_reversal_twins_share_all_evidence_and_the_same_rounded_candidate():
    pairs = [task for task in make_tasks("shift", seed=47, per_mechanism=8) if task.twin_id]
    for first, second in zip(pairs[::2], pairs[1::2], strict=True):
        assert first.twin_id == second.twin_id
        assert first.parent_checkpoint == second.parent_checkpoint
        assert features(first, "D") == features(second, "D")
        for method in METHODS:
            a, b = (run_method(task, method, radius_scale=0.5) for task in (first, second))
            assert a.checkpoint_sha256 == b.checkpoint_sha256
            assert a.practice_pass and b.practice_pass
            assert sorted((a.improved, b.improved)) == [False, True]
            assert sorted((a.harmful, b.harmful)) == [False, True]
            assert a.cost == b.cost


def test_diagnosis_can_find_absent_headroom_and_expensive_probes_are_charged():
    population = make_tasks("train", seed=23, per_mechanism=20)
    artifacts = [task for task in population if task.mechanism == "measurement_artifact"]
    assert all(run_method(task, "first_order").reason == "MEASUREMENT_REQUIRED" for task in artifacts)
    after = [diagnose(task) for task in artifacts]
    assert {features(task)[0]["persistent_deficiency"] for task, _ in after} == {0, 1}
    assert all(cost.forwards == 4 for _, cost in after)
    expensive = make_tasks("shift", seed=23, per_mechanism=2)[-1]
    diagnosed, cost = diagnose(expensive)
    assert cost == diagnosis_cost(expensive)
    assert cost.forwards == 96
    assert features(diagnosed)[0]["persistent_deficiency"] == 1


def _task(margins):
    return Task("manual", "unit", 1.0, np.zeros(2, dtype="<f2").tobytes(), margins, margins, True, 1.0)


def test_interaction_repairs_a_dormant_pair_with_zero_first_derivative():
    margins = Margins(np.array([-0.4, -0.3, 0.2]), np.array([0.4]),
                      np.zeros((3, 2)), np.zeros((1, 2)), np.full(3, 2.0))
    task = _task(margins)
    assert features(task, "C")[0]["locally_reachable"] == 0
    assert run_method(task, "first_order").reason == "NO_DIRECTION"
    assert run_method(task, "constrained").reason == "NO_DIRECTION"
    interaction = run_method(task, "interaction")
    assert interaction.accepted and interaction.target_gain == 2
    assert 0 < interaction.update_norm <= task.radius
    assert interaction.checkpoint_sha256 is not None
    delta = 1e-4
    pp = _project(margins, np.array([delta, delta]), quadratic=True)[0]
    pn = _project(margins, np.array([delta, -delta]), quadratic=True)[0]
    np_ = _project(margins, np.array([-delta, delta]), quadratic=True)[0]
    nn = _project(margins, np.array([-delta, -delta]), quadratic=True)[0]
    assert (pp - pn - np_ + nn) / (4 * delta**2) == pytest.approx(margins.mixed)


def test_constrained_repair_preserves_a_fragile_margin_and_honors_cost_bound():
    margins = Margins(np.array([-0.4, -0.3, 0.2]), np.array([0.13]),
                      np.tile([1.0, 0.7], (3, 1)), np.array([[-0.95, 0.0]]), np.zeros(3))
    task = _task(margins)
    assert run_method(task, "first_order").reason == "PRACTICE_REJECTED"
    constrained = run_method(task, "constrained")
    assert constrained.accepted and constrained.preservation_regressions == 0
    for level in "ABCD":
        assert features(task, level)[1] == feature_cost(task, level=level)
    for method in METHODS:
        actual = run_method(task, method).cost
        limit = method_cost_bound(task, method)
        assert all(a <= b for a, b in zip(astuple(actual), astuple(limit), strict=True))
    no_headroom = replace(task, measurement_valid=False)
    assert run_method(no_headroom, "constrained").cost == Cost()
