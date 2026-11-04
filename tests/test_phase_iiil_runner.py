from types import SimpleNamespace

import numpy as np

from experiments.phase_iiil import run_semantic, semantic_diagnostics
from weightbridge.adaptation import semantic_consensus


def test_fork_selection_balances_groups_without_reusing_truth_flip_families():
    forks, rows = [], []

    def add(comparison, pair, tier, precision, orders, replay=None, wordings=(0, 1)):
        for order in range(orders):
            item = f"{comparison}/{order}"
            for wording in wordings:
                prompt = f"{item}/{wording}"
                rows.append({"id": prompt, "wording": wording, "comparison_id": comparison, "pair_id": pair,
                             "tier": tier, "truth_precision": precision})
                forks.append(SimpleNamespace(question_id=item, prompt_id=prompt, status="semantic",
                                             replay_status=replay if wording == 1 and replay else "matched"))

    add("a-close-short", "flip-one", "negative-close", "shorter", 2)
    add("a-close-long-relative", "flip-one", "negative-close", "longer", 2)
    add("b-close-long", "flip-two", "negative-close", "longer", 1)
    add("a-far-short-incomplete", "flip-three", "negative-far", "shorter", 2, wordings=(0,))
    add("b-far-short-replay-mismatch", "flip-four", "negative-far", "shorter", 2, replay="greedy_replay_mismatch")
    add("c-far-short", "flip-five", "negative-far", "shorter", 1)
    add("a-far-long", "flip-six", "negative-far", "longer", 2)

    picked = run_semantic.choose_forks(forks, rows, required_wordings=(0, 1), cap=8)

    assert [(item, group, len(members)) for item, group, members in picked] == [
        ("a-close-short", "close-short", 4), ("b-close-long", "close-long", 2),
        ("c-far-short", "far-short", 2), ("a-far-long", "far-long", 4),
    ]
    assert [item for item, _, _ in run_semantic.choose_forks(forks, rows, required_wordings=(0, 1), cap=2)] == [
        "a-close-short", "b-close-long",
    ]


def test_practice_guard_bridges_fork_api_and_deduplicates_stored_candidates(monkeypatch, tmp_path):
    contract = SimpleNamespace(batch_size=8, max_candidate_evaluations=4)
    plan = SimpleNamespace(limits=SimpleNamespace(max_memory_bytes=4096), record=lambda: {})
    basis = SimpleNamespace(columns=(0,), right=np.array([1.]), preferred_left=np.array([1.]))
    populations = {"adaptation": ("fit rows", "fit parent"), "preservation-fit": ("protected rows", "protected parent")}
    anchor = SimpleNamespace(prompt_id="protected-prompt")
    shared = (SimpleNamespace(anchors=(anchor,)), SimpleNamespace(trajectories={"protected-prompt": object()}),
              [SimpleNamespace(gradient=np.array([1.]), margin=.5)], .125, .01)
    solution = SimpleNamespace(update=np.array([.01]))
    fit = SimpleNamespace(historical=solution, maximin=solution, record=lambda: {})
    monkeypatch.setattr(semantic_consensus, "solve_consensus", lambda *args, **kwargs: fit)
    monkeypatch.setattr(run_semantic, "prepare_probe_reference", lambda *args, **kwargs: SimpleNamespace(resources={}))
    monkeypatch.setattr(run_semantic, "_margins_record", lambda *args: {})
    monkeypatch.setattr(run_semantic, "_active_measurements",
                        lambda *args, **kwargs: [{"decision_preserved": True, "measured_margin": .125}])
    monkeypatch.setattr(semantic_diagnostics, "compare_semantic",
                        lambda *args, **kwargs: {"passed": True, "target_gain": 1, "regressions": [], "lost_valid": []})
    monkeypatch.setattr(run_semantic, "compare_endpoint",
                        lambda *args: {"prohibited_regressions": [], "lost_valid_presentations": []})
    evaluation_calls = []

    def measure(runner, rows, contract, *, candidate_budget=None):
        if candidate_budget is not None:
            candidate_budget.charge()
        evaluation_calls.append(rows)
        return "measured " + rows

    def probe(*args, seen, evaluate_candidate, **kwargs):
        state = "same rounded hash"
        if state in seen:
            return None, {"candidate_tensor_sha256": state, "reason": "duplicate_stored_candidate"}
        endpoint = evaluate_candidate()
        assert endpoint["guard"]["passed"] is True
        assert endpoint["guard"]["target_item_gain"] == 1
        assert endpoint["guard"]["prohibited_regressions"] == endpoint["guard"]["lost_valid_presentations"] == []
        seen.add(state)
        return {}, {"candidate_tensor_sha256": state, "endpoint": endpoint, "reason": "endpoint_pass"}

    def reject_materialize(*args):
        raise AssertionError("An anchor at the floor does not qualify as positive measured slack")

    monkeypatch.setattr(run_semantic, "measure_endpoint", measure)
    monkeypatch.setattr(run_semantic, "probe_update", probe)
    monkeypatch.setattr(run_semantic, "_materialize", reject_materialize)
    config = {"probe_design": {"doses": [.0125, .0225]}}

    arms, saved, count = run_semantic._compare(None, None, None, plan, populations, config, contract, tmp_path,
                                               basis, (), ["i"], ["close-short"], np.array([[1.]]), shared)

    assert count == 1 and evaluation_calls == ["fit rows", "protected rows"]
    assert not saved
    assert arms["A1"]["probe"]["endpoint"]["guard"]["semantic_behavior_passed"]
    assert not arms["A1"]["probe"]["endpoint"]["guard"]["passed"]
    assert all(arms[arm]["duplicate_of"] == "A1" for arm in ("A2", "B1", "B2"))
