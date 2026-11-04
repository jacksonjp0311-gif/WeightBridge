"""Check practice counterexamples and realized stored norm on the tiny BF16 model."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_fork_update import fixture
from weightbridge.adaptation.content_scoring import Continuation
from weightbridge.adaptation.counterexample_update import (
    DecisionPool, construct_update, fit_fork_basis, fit_preservation, fit_projected_lora, observe_counterexamples,
)
from weightbridge.adaptation.endpoint_update import _tensor_bytes
from weightbridge.adaptation.fork_update import AffineUpdate, probe_update
from weightbridge.adaptation.generative_fork import ProtectedTrajectory
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


torch = pytest.importorskip("torch")


def test_real_bf16_fork_vjps_and_authorized_regression_refit():
    adapter, runner, plan, fork, _, raw = fixture()
    fork.prompt_id, fork.question_id = "target-prompt", "target"
    anchor = DecisionAnchor("copy", "copy-0", "copy", 0, 0, .625, 2)
    path = ProtectedTrajectory(anchor.prompt_id, (4,), (2, 0), "eos")
    pool = DecisionPool((anchor,), {anchor.prompt_id: path})
    basis, measured = fit_fork_basis(adapter, runner, plan, [fork])
    assert measured["resources"]["backward_calls"] == 2
    assert basis.target_margins.tolist() == pytest.approx([-.25])
    assert basis.target_gradients @ basis.preferred_left > 0
    sensitive, _ = fit_preservation(adapter, runner, plan, basis, pool, [anchor])
    assert list(sensitive) == [anchor.key]
    assert sensitive[anchor.key].margin == pytest.approx(.625)
    with pytest.raises(ValueError, match="parent practice"):
        fit_preservation(adapter, runner, plan, basis, pool, [replace(anchor, group="heldout")])
    update, report = construct_update(basis, sensitive, radius=.3, method="inequality")
    assert update.rank == 1 and report["target_predictions"][0]["predicted_margin"] > 0
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw

    wrong = np.zeros((6, 1))
    wrong[3, 0] = 1.
    intentionally_regressing = AffineUpdate((4,), wrong, np.ones((1, 1)), "fixture")

    def candidate():
        result = {"guard": {"passed": False, "target_item_gain": 1, "prohibited_regressions": [anchor.prompt_id]},
                  "measurement": {"rows": [{"id": anchor.prompt_id, "correct": False, "status": "incorrect",
                                           "generated_token_ids": [3, 0]}]}}
        selected, evidence = observe_counterexamples(runner, pool, result, limit=1, batch_size=2, max_memory_bytes=2**40)
        assert selected == (anchor,)
        counterexample = evidence["counterexamples"][0]
        assert counterexample["candidate_token_id"] == 3
        assert counterexample["candidate_margin"] < 0 < counterexample["parent_margin"]
        skipped, budget_evidence = observe_counterexamples(runner, pool, result, limit=0, batch_size=2, max_memory_bytes=2**40)
        assert not skipped and budget_evidence["counterexamples"][0]["reason"] == "refinement_limit"
        return result

    patch, observed = probe_update(adapter, runner, plan, intentionally_regressing, radius=None, seen=set(),
                                   evaluate_candidate=candidate)
    assert patch is None and observed["reason"] == "endpoint_rejected"
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw


def test_projected_lora_enforces_the_rounded_envelope_at_each_step():
    adapter, runner, plan, _, _, raw = fixture()
    plan.limits.max_update_norm = .08
    paths = (Continuation((1,), (2,), 0, True), Continuation((4,), (2,), 0, True))
    update, report = fit_projected_lora(adapter, runner, plan, paths, rank=2, steps=2, lr=.5, radius=.07)
    assert update.rank == 2
    assert report["trainable_parameters"] == 24
    assert all(step["unrounded_norm"] <= .07 for step in report["step_history"])
    assert all(step["realized_update_norm"] <= .08 for step in report["step_history"])
    assert any(step["projection_steps"] > 0 for step in report["step_history"])
    patch, probe = probe_update(adapter, runner, plan, update, radius=None, seen=set(), evaluate_candidate=lambda:
                                {"guard": {"passed": False, "target_item_gain": 0}, "measurement": {}})
    assert patch is None and 0 < probe["realized_update_norm"] <= .08
    assert probe["realized_update_norm"] == pytest.approx(report["step_history"][-1]["realized_update_norm"])
    assert _tensor_bytes(runner.model.affine.weight, torch) == raw
