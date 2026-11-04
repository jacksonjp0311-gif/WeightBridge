"""Mocked runner stage boundaries; fixtures are not real-model evidence."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiip import run_right
from weightbridge.adaptation.counterexample_update import ForkBasis


@pytest.fixture
def numerical_fixture():
    config = run_right.read_json(run_right.PROTOCOL)
    plan = SimpleNamespace(limits=SimpleNamespace(max_update_norm=.025, max_forward_calls=480,
        max_memory_bytes=1 << 40), record=lambda: {})
    right = np.array([1., 0.])
    basis = ForkBasis("parent", "weight", (0, 1), np.array([1., 0.]), right,
        ("wrong",), ("comparison",), np.array([-.001]), np.array([[10., 0.]]))
    anchor = SimpleNamespace(prompt_id="protected", position=0, key=("protected", 0), token_id=0)
    cal = tuple(SimpleNamespace(prompt_id=f"cal{i}", position=0, key=(f"cal{i}", 0),
        group="target") for i in range(3))
    shared = (object(), SimpleNamespace(anchors=(anchor,)), (), object(),
        SimpleNamespace(anchors=cal), {}, {a.prompt_id: f"family{i}" for i, a in enumerate(cal)}, .125)
    blocks, margins = np.zeros((1, 2, 2)), np.array([.5])
    cal_blocks, cal_margins = np.zeros((3, 2, 2)), np.full(3, .5)
    forks = (SimpleNamespace(prompt_id="wrong"),)

    class Space:
        preview_calls = 0
        calibration_calls = 0
        measured_calibration_margin = .5

        def preview(self, u):
            self.preview_calls += 1
            delta = np.outer(u, right)
            return {"realized_norm": float(np.linalg.norm(delta)), "tensor_sha256": hashlib.sha256(delta.tobytes()).hexdigest(),
                "changed_entries": int(np.count_nonzero(delta)), "proposed_norm": float(np.linalg.norm(u)),
                "update": u.copy(), "delta": delta, "stored_update_cosine": 1.}

        def margins(self, runner, pool, anchors, preview, **kwargs):
            self.calibration_calls += 1
            return {a.key: self.measured_calibration_margin for a in anchors}, {
                "rows": [], "resources": {"forward_calls": 1, "backward_calls": 0}}

    return SimpleNamespace(config=config, plan=plan, basis=basis, shared=shared, blocks=blocks,
        margins=margins, cal_blocks=cal_blocks, cal_margins=cal_margins, forks=forks,
        contexts=(forks,), space=Space())


def cell(f, budget=None, *, matched=False):
    if budget is None:
        budget = run_right._ForwardBudget(480, 0)
    return run_right._numeric_cell(f.space, None, f.plan, f.basis, f.forks, f.contexts,
        f.blocks, f.margins, f.cal_blocks, f.cal_margins, f.shared, f.config, budget,
        matched=matched)


def test_fixed_projection_matches_full_block_and_free_right_search_uses_frozen_configuration(numerical_fixture):
    f = numerical_fixture
    blocks = np.array([[[0., 10.], [0., 0.]]])
    rhs = np.array([.08])
    fixed, search, evidence = run_right._relax_and_search(blocks, rhs, np.array([1., 0.]),
        f.plan, f.config, ["wrong"])
    assert fixed.update is None and fixed.record()["status"] == "INFEASIBLE_ZERO_ROW"
    assert search.right is not None and search.left is not None
    assert np.linalg.norm(search.left) == pytest.approx(.008)
    assert np.einsum('ijk,j,k->i', blocks, search.left, search.right)[0] >= rhs[0] - 1e-8
    assert evidence["optimized_right"]["settings"] == f.config["probe_design"]["outer_search"]["settings"]
    assert evidence["relaxation"]["resources"]["forward_calls"] == 0


def test_below_floor_prediction_is_refused_before_native_materialization(numerical_fixture, monkeypatch):
    f = numerical_fixture
    too_small = np.array([(f.config['probe_design']['target_margin'] + .001 - 5e-9) / 10., 0.])
    monkeypatch.setattr(run_right, "_fixed", lambda *a, **k:
        SimpleNamespace(update=too_small, record=lambda: {"status": "tolerance_only_fixture"}))
    monkeypatch.setattr(f.space, "preview", lambda *a: pytest.fail("Below-floor predictions cannot materialize"))
    monkeypatch.setattr(run_right, "measure_stored_joint", lambda *a, **k: pytest.fail("No native inference"))
    best, evidence = cell(f)
    assert best is None and evidence["status"] == "NATIVE_OR_CALIBRATION_HOLD"
    assert evidence["resources"]["forward_calls"] == 0
    assert f.space.calibration_calls == 0


def test_native_refusal_stops_before_calibration_or_generation(numerical_fixture, monkeypatch):
    f = numerical_fixture
    calls = []

    def native(*args, **kwargs):
        calls.append(kwargs)
        return {"passed": False, "resources": {"forward_calls": 2, "backward_calls": 0}}

    monkeypatch.setattr(run_right, "measure_stored_joint", native)
    monkeypatch.setattr(run_right, "probe_joint_generation", lambda *a, **k: pytest.fail("No decode at numeric stage"))
    budget = run_right._ForwardBudget(480, 0)
    best, evidence = cell(f, budget)
    assert best is None and len(calls) == 1
    assert evidence["resources"]["forward_calls"] == budget.used == 2
    assert f.space.calibration_calls == 0
    assert calls[0]["target_floor"] == f.config["probe_design"]["target_margin"]


def test_numerical_minimum_native_and_calibration_scopes_count_once(numerical_fixture, monkeypatch):
    f = numerical_fixture
    monkeypatch.setattr(run_right, "measure_stored_joint", lambda *a, **k:
        {"passed": True, "resources": {"forward_calls": 2, "backward_calls": 0}})
    budget = run_right._ForwardBudget(480, 0)
    best, evidence = cell(f, budget)
    assert best is not None and evidence["status"] == "NATIVE_AND_CALIBRATION_PASS"
    assert best["realized_norm"] == pytest.approx(.0063501)
    assert evidence["direction_specific_allowance"] == 0.
    assert evidence["resources"]["forward_calls"] == budget.used == 3
    assert f.space.calibration_calls == 1
    assert evidence["calibration_passes"][0]["proposal"]["status"] == "MINIMUM_WITNESS"


def test_repeated_stored_state_reuses_native_measurement_but_keeps_calibration_cost(numerical_fixture, monkeypatch):
    f = numerical_fixture
    f.space.measured_calibration_margin = .45
    native_calls = []
    monkeypatch.setattr(run_right, "measure_stored_joint", lambda *a, **k:
        native_calls.append(1) or {"passed": True, "resources": {"forward_calls": 2, "backward_calls": 0}})
    budget = run_right._ForwardBudget(480, 0)
    best, evidence = cell(f, budget)
    assert best is not None and evidence["status"] == "NATIVE_AND_CALIBRATION_PASS"
    assert evidence["direction_specific_allowance"] == pytest.approx(.05)
    assert len(native_calls) == 1 and f.space.calibration_calls == 2
    assert evidence["resources"]["forward_calls"] == budget.used == 4


def test_matched_cell_uses_actual_stored_norm_and_frozen_target(numerical_fixture, monkeypatch):
    f = numerical_fixture
    monkeypatch.setattr(run_right, "measure_stored_joint", lambda *a, **k:
        {"passed": True, "resources": {"forward_calls": 2, "backward_calls": 0}})
    best, evidence = cell(f, matched=True)
    assert best is not None
    assert best["realized_norm"] == pytest.approx(f.config["probe_design"]["target_norms"][-1])
    assert evidence["calibration_passes"][0]["proposal"]["status"] == "MATCHED"


def test_numerical_budget_refuses_before_an_extra_native_call(numerical_fixture, monkeypatch):
    f = numerical_fixture
    monkeypatch.setattr(run_right, "measure_stored_joint", lambda *a, **k: pytest.fail("Budget must refuse first"))
    with pytest.raises(run_right._BudgetExhausted):
        cell(f, run_right._ForwardBudget(1, 0))


def test_absent_orientation_witness_records_unmeasured_cells_without_data_access(numerical_fixture, monkeypatch, tmp_path):
    f = numerical_fixture
    monkeypatch.setattr(run_right.StoredAffineSpace, "open", lambda *a, **k: pytest.fail("No witness means no preview"))
    monkeypatch.setattr(run_right, "measure_endpoint", lambda *a, **k: pytest.fail("No witness means no generation"))
    result, candidates = run_right._candidates(None, None, None, f.plan, f.basis, f.forks, f.contexts,
        {"A": None, "C": None}, np.zeros((1, 2, 2)), f.blocks, f.margins, f.cal_blocks,
        f.cal_margins, f.shared, {}, f.config, tmp_path, run_right._ForwardBudget(480, 0))
    assert not candidates and result["unique_complete_evaluations"] == 0
    for name in ('A1', 'A2', 'C1', 'C2'):
        record = json.loads((tmp_path / f'arm-{name}.json').read_bytes())
        assert record['numerical']['status'] == 'NO_WITHIN_CAP_ORIENTATION_WITNESS'
        assert 'endpoint' not in record['probe']
