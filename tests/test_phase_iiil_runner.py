import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiil import run_semantic, semantic_diagnostics
from weightbridge.adaptation import counterexample_update, semantic_consensus
from weightbridge.adaptation.preservation_geometry import DecisionAnchor
from weightbridge.core import file_hash


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


@pytest.mark.parametrize("last_replacement_replays", [True, False])
def test_directional_anchor_retries_freeze_passing_cells_and_obey_round_cap(monkeypatch, last_replacement_replays):
    targets = tuple(DecisionAnchor(f"item-{i}", f"target-{i}", "target", 0, 0, .1 + i / 10, 1)
                    for i in range(1, 5))
    controls = (DecisionAnchor("control-item", "control", "control", 0, 0, .2, 2),
                DecisionAnchor("other-control-item", "other-control", "control", 0, 0, .3, 2))
    anchors = targets + controls
    pool = SimpleNamespace(eligible_anchors=anchors)
    attempts = []

    def fake_fit(adapter, runner, plan, basis, pool, needed, *, batch_size):
        attempts.append(tuple(anchor.prompt_id for anchor in needed))
        passing = {"control", "target-4"} | ({"target-3"} if last_replacement_replays else set())
        return {anchor.key: anchor for anchor in needed if anchor.prompt_id in passing}, {}

    monkeypatch.setattr(counterexample_update, "fit_preservation", fake_fit)
    rounds = run_semantic.read_json(run_semantic.REPAIR)["maximum_sampling_rounds"]
    sample, verified, receipts = run_semantic._replay_stable_sample(
        None, None, None, None, pool, batch=8, rounds=rounds,
        sampling={"mode": "stratified_directional", "required_groups": ("target", "control"),
                  "required_wordings": (0,), "budget": 2, "radius": .0125,
                  "slopes": {anchor.key: -1. for anchor in anchors}},
    )

    assert rounds == 3
    assert attempts == [("target-1", "control"), ("target-2",), ("target-3",)]
    assert len(receipts) == 3
    assert sample.complete and sample.anchors == (targets[2], controls[0])
    assert set(verified) == ({targets[2].key, controls[0].key} if last_replacement_replays else {controls[0].key})


def test_reuse_accepts_only_committed_parent_receipts(monkeypatch, tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    manifest = material / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    initial = tmp_path / "initial"
    initial.mkdir()
    receipts = {split: initial / f"parent-{split}.json" for split in run_semantic.FIT_SPLITS}
    for path in receipts.values():
        path.write_text("{}", encoding="utf-8")
    report_path = initial / "fit-report.json"
    report = {"checkpoint_sha256": "verified-parent", "hidden_opened": False,
              "optimization_splits": list(run_semantic.FIT_SPLITS), "candidates": {},
              "manifest_sha256": file_hash(manifest),
              "artifacts": {path.name: file_hash(path) for path in receipts.values()}}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    repair = tmp_path / "replay_repair.json"
    repair.write_text(json.dumps({"first_fit_report_sha256": file_hash(report_path)}), encoding="utf-8")
    monkeypatch.setattr(run_semantic, "REPAIR", repair)

    assert run_semantic._initial_parent_receipts(initial, material, "verified-parent") == receipts
    with pytest.raises(ValueError, match="frozen development scope"):
        run_semantic._initial_parent_receipts(initial, material, "different-parent")
    receipts["adaptation"].write_text('{"replaced": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="parent measurement differs"):
        run_semantic._initial_parent_receipts(initial, material, "verified-parent")
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="committed repair input"):
        run_semantic._initial_parent_receipts(initial, material, "verified-parent")
