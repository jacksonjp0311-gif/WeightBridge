"""Exercise actual rounded J-Space margin diagnostics on a small BF16 model."""

from dataclasses import replace

import pytest

from test_fork_update import fixture
from weightbridge.adaptation import jspace_update
from weightbridge.adaptation.counterexample_update import DecisionPool, fit_fork_basis, fit_preservation
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.generative_fork import ProtectedTrajectory
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


torch = pytest.importorskip("torch")


def _geometry():
    adapter, runner, plan, fork, _, raw = fixture()
    fork.prompt_id, fork.question_id = "target-prompt", "target"
    basis, _ = fit_fork_basis(adapter, runner, plan, [fork])
    vulnerable = DecisionAnchor("calibration", "form-cal", "target", 0, 0, .25, 3)
    safe = DecisionAnchor("validation", "form-val", "copy", 1, 0, .625, 2)
    pool = DecisionPool((vulnerable, safe), {
        vulnerable.prompt_id: ProtectedTrajectory(vulnerable.prompt_id, (1,), (3,), "eos"),
        safe.prompt_id: ProtectedTrajectory(safe.prompt_id, (4,), (2,), "eos"),
    })
    return adapter, runner, plan, basis, pool, raw


def test_stored_directional_scan_includes_high_and_low_margin_decisions_and_restores():
    adapter, runner, plan, basis, pool, raw = _geometry()
    slopes, receipt = jspace_update.measure_directional_secant(adapter, runner, plan, basis, pool, radius=.3)
    vulnerable, safe = pool.anchors
    assert slopes[vulnerable.key] < 0
    assert slopes[safe.key] == 0
    assert receipt["position_count"] == 2 and receipt["changed_entries"] > 0
    assert receipt["nominal_radius"] != receipt["realized_update_norm"]
    assert receipt["resources"]["forward_calls"] == 2 and receipt["restored_exactly"]
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
    slopes, no_op = jspace_update.measure_directional_secant(adapter, runner, plan, basis, pool, radius=1e-12)
    assert not slopes and no_op["reason"] == "no_stored_change"


def test_failed_directional_measurement_restores_exact_parent(monkeypatch):
    adapter, runner, plan, basis, pool, raw = _geometry()
    trajectory_logits = jspace_update.trajectory_logits
    calls = 0

    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert _tensor_bytes(runner.model.affine.weight, torch) != raw
            raise RuntimeError("interrupted active measurement")
        return trajectory_logits(*args)

    monkeypatch.setattr(jspace_update, "trajectory_logits", interrupt)
    with pytest.raises(RuntimeError, match="interrupted active measurement"):
        jspace_update.measure_directional_secant(adapter, runner, plan, basis, pool, radius=.3)
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw


def test_calibration_and_held_residuals_replay_separately_and_bind_semantic_items():
    adapter, runner, plan, basis, pool, raw = _geometry()
    vulnerable, safe = pool.anchors
    sensitivity, _ = fit_preservation(adapter, runner, plan, basis, pool, pool.anchors)
    cal, held, report = jspace_update.calibrate_direction_residuals(
        adapter, runner, plan, basis, pool, (vulnerable,), (), sensitivity, radius=.3)
    assert len(cal) == 1 and not held and report["validation"] == []
    row = report["calibration"][0]
    assert row["residual"] == pytest.approx(row["measured_margin"] - row["predicted_margin"])
    assert abs(row["residual"]) > 0
    assert row["candidate_replay"]["replay_argmax_token_id"] != vulnerable.token_id
    cal, held, report = jspace_update.calibrate_direction_residuals(
        adapter, runner, plan, basis, pool, (), (safe,), sensitivity, radius=.3)
    assert not cal and len(held) == 1 and report["validation"][0]["residual"] == 0

    same_item = replace(safe, item=vulnerable.item)
    overlap = DecisionPool((vulnerable, same_item), pool.trajectories)
    with pytest.raises(ValueError, match="distinct semantic items"):
        jspace_update.calibrate_direction_residuals(adapter, runner, plan, basis, overlap, (vulnerable,),
                                                     (same_item,), sensitivity, radius=.3)
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
