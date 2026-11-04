"""Exercise the refinement loop with deterministic practice-only probe receipts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiij import run_preservation as trial
from weightbridge.adaptation import counterexample_update as backend
from weightbridge.adaptation import fork_update as probes
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


@pytest.mark.parametrize("main_budget", [5, 9])
def test_refinements_append_novel_practice_constraints_and_write_each_arm_once(tmp_path, monkeypatch, main_budget):
    config = json.loads(trial.PROTOCOL.read_bytes())
    config["probe_design"]["max_main_evaluations"] = main_budget
    groups = config["probe_design"]["preservation_groups"]
    initial = tuple(DecisionAnchor(group, f"{group}/{wording}", group, wording, 0, 1.0, 20)
                    for group in groups for wording in (0, 1))
    counterexamples = tuple(DecisionAnchor("target", f"target/{wording}", "target", wording, position, 2.0, 21)
                            for position in (1, 2, 3) for wording in (0, 1))
    pool = backend.DecisionPool((*initial, *counterexamples), {point.prompt_id: None for point in initial})
    basis = SimpleNamespace(preferred_left=np.ones(1))
    runner = object()
    fit_batches = []

    def fake_plan(_config, _capabilities, layer, **_kwargs):
        return SimpleNamespace(region=SimpleNamespace(layer_index=layer, tensor_name=f"layer-{layer}"),
                               limits=SimpleNamespace(max_memory_bytes=1), record=lambda: {"layer": layer})

    def fit_constraints(_adapter, live_runner, plan, source_basis, parent_pool, anchors, **_kwargs):
        assert live_runner is runner and source_basis is basis and parent_pool is pool
        keys = tuple(point.key for point in anchors)
        fit_batches.append((plan.region.layer_index, keys))
        return {point.key: SimpleNamespace(anchor=point, gradient=np.ones(1)) for point in anchors}, {}

    def measure(_runner, _rows, _contract, *, candidate_budget):
        candidate_budget.charge()
        return {"probe_number": candidate_budget.used}

    def guard(_parent, observed, _qualification, _contract):
        number = observed["probe_number"]
        return {"passed": False, "prohibited_regressions": ["target/0"] if 3 <= number <= 5 else [],
                "target_item_gain": 0, "corrected_repeatable_negative_items": []}

    def feedback(_runner, parent_pool, observed, *, exclude, limit, **_kwargs):
        assert parent_pool is pool and limit == 2
        number = observed["measurement"]["probe_number"]
        position = number - 2
        assert {point.key for point in initial} <= exclude
        assert {point.key for point in counterexamples if point.position < position} <= exclude
        offered = tuple(point for point in counterexamples if point.position == position)
        assert all(point.key not in exclude for point in offered)
        return offered, {"offered": [point.key for point in offered]}

    def probe(_adapter, _runner, _plan, _update, *, evaluate_candidate, **_kwargs):
        observed = evaluate_candidate()
        digest = str(observed["measurement"]["probe_number"])
        return None, {"reason": "endpoint_rejected", "candidate_tensor_sha256": digest, "endpoint": observed}

    monkeypatch.setattr(trial, "endpoint_runner", lambda runner, _contract: runner)
    monkeypatch.setattr(trial, "_diagnose", lambda *_args: ((object(),), {}))
    monkeypatch.setattr(trial, "_plan", fake_plan)
    monkeypatch.setattr(trial, "_training_paths", lambda *_args: ())
    monkeypatch.setattr(trial, "measure_endpoint", measure)
    monkeypatch.setattr(trial, "_guard", guard)
    monkeypatch.setattr(backend, "measure_parent_decisions", lambda *_args, **_kwargs: (pool, {}))
    monkeypatch.setattr(backend, "fit_fork_basis", lambda *_args, **_kwargs: (basis, {}))
    monkeypatch.setattr(backend, "fit_preservation", fit_constraints)
    monkeypatch.setattr(backend, "construct_update", lambda *_args, **_kwargs: (object(), {}))
    monkeypatch.setattr(backend, "observe_counterexamples", feedback)
    monkeypatch.setattr(backend, "fit_projected_lora", lambda *_args, **_kwargs: (None, {}))
    monkeypatch.setattr(probes, "prepare_probe_reference", lambda *_args, **_kwargs: SimpleNamespace(resources={}))
    monkeypatch.setattr(probes, "probe_update", probe)
    qualification = {"solved": {"target": ["target"], "positive": ["positive"], "preserve": ["copy"]}}

    arms, candidates, summary = trial._fit(None, None, runner, None, (), {}, qualification, config, tmp_path)

    assert not candidates and "D3" not in arms
    assert summary["unique_complete_evaluations"] == main_budget + 1  # Alternate uses its own slot.
    for number, parent in ((1, "C"), (2, "D1")):
        receipt = json.loads((tmp_path / f"arm-D{number}.json").read_bytes())
        expected = {point.key for point in counterexamples if point.position == number}
        assert receipt["refines"] == parent
        assert {(row["prompt_id"], row["position"]) for row in receipt["added_anchors"]} == expected
        assert len(receipt["anchors"]) == len(initial) + 2 * number
        assert (23, tuple(point.key for point in counterexamples if point.position == number)) in fit_batches
    if main_budget == 5:
        assert {arms[f"E{index}"]["probe"]["reason"] for index in range(1, 5)} == {"decode_budget_exhausted"}
