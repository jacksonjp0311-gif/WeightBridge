from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.phase_iiim import run_factorial, semantic_metrics, source_manifest
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


@pytest.mark.parametrize("residuals,certified,calibration_items", [((.02, .01), True, 3), ((.02, .03), False, 3),
                                                                  ((.02, .01), False, 1)])
def test_numeric_cell_recalibrates_exact_direction_at_most_twice(monkeypatch, residuals, certified, calibration_items):
    anchor = DecisionAnchor("item", "p", "target", 0, 0, 1., 1)
    sample = SimpleNamespace(anchors=(anchor,))
    active_pool, calibration_pool = object(), object()
    sensitivity = SimpleNamespace(gradient=np.array([1.]), margin=1.)
    calibration = tuple(DecisionAnchor(f"cal-item-{i}", f"cal-{i}", "target", 0, 0, 1., 1)
                        for i in range(calibration_items))
    shared = (active_pool, sample, (sensitivity,), calibration_pool, SimpleNamespace(anchors=calibration),
              {a.key: sensitivity for a in calibration}, {a.prompt_id: f"family-{i}" for i, a in enumerate(calibration)}, .125)
    plan = SimpleNamespace(limits=SimpleNamespace(max_memory_bytes=4096, max_update_norm=.025))
    allowances, calibration_count, active_count = [], 0, 0

    def solve(direction, gradients, margins, *, radius, margin_floor, allowance):
        allowances.append(allowance)
        return SimpleNamespace(update=np.array([radius]), record=lambda: {"status": "preferred_feasible"})

    class Space:
        def preview(self, update):
            amplitude = float(update[0])
            return {"realized_norm": amplitude, "tensor_sha256": f"stored-{amplitude}", "changed_entries": 1,
                    "proposed_norm": amplitude, "update": update, "delta": update[:, None],
                    "stored_update_cosine": 1.}

        def margins(self, runner, pool, anchors, preview, **kwargs):
            nonlocal calibration_count, active_count
            margin = 1. + preview["update"][0]
            if pool is calibration_pool:
                margin -= residuals[calibration_count]
                calibration_count += 1
            else:
                assert pool is active_pool
                active_count += 1
            return {a.key: float(margin) for a in anchors}, {
                "rows": [{"replay_argmax_token_id": 1} for a in anchors], "resources": {"forward_calls": 1}}

    monkeypatch.setattr(run_factorial, "solve_robust", solve)
    config = {"probe_design": {"norm_relative_tolerance": .02, "norm_search_steps": 4,
                                "adverse_item_quantile": .8}, "inference": {"batch_size": 8}}
    best, evidence = run_factorial._numeric_cell(Space(), None, plan, None, np.array([1.]), shared, .01, config)

    assert best is not None and best["realized_norm"] == .01
    assert evidence["calibration_sufficient"] is certified
    if calibration_items >= 3:
        assert allowances == pytest.approx([0., .02])
        assert calibration_count == 2 and active_count == 1
        assert evidence["resources"]["forward_calls"] == 3
        assert len(evidence["calibration_passes"]) == 2
        assert evidence["direction_specific_allowance"] == pytest.approx(max(residuals))
    else:
        assert allowances == [0.]
        assert calibration_count == active_count == 1
        assert evidence["resources"]["forward_calls"] == 2
        assert len(evidence["calibration_passes"]) == 1


def _comparison_fixture(monkeypatch):
    contract = SimpleNamespace(batch_size=8, max_candidate_evaluations=8)
    plan = SimpleNamespace(limits=SimpleNamespace(max_memory_bytes=4096, max_forward_calls=100), record=lambda: {})
    basis = SimpleNamespace(columns=(0,), right=np.array([1.]))
    anchor = SimpleNamespace(prompt_id="protected-prompt")
    sample = SimpleNamespace(anchors=(anchor,))
    pool = SimpleNamespace(trajectories={"protected-prompt": object()})
    shared = (pool, sample, (), None, None, None, None, .125)
    populations = {"adaptation": ("fit rows", "fit parent"),
                   "preservation-fit": ("protected rows", "protected parent")}
    factorial = SimpleNamespace(directions={arm: np.array([1.]) for arm in "ABCD"})
    config = {"probe_design": {"target_norms": [.01], "norm_relative_tolerance": .02},
              "endpoint": {"min_valid_rate": .9}}
    monkeypatch.setattr(run_factorial, "endpoint", lambda config: contract)
    monkeypatch.setattr(run_factorial.StoredAffineSpace, "open", lambda *args: object())
    monkeypatch.setattr(run_factorial, "prepare_probe_reference", lambda *args, **kwargs: SimpleNamespace(resources={}))
    monkeypatch.setattr(run_factorial, "measure_fork_token_classes", lambda *args, **kwargs: {"rows": []})
    return contract, plan, basis, shared, populations, factorial, config


def test_practice_gate_aliases_deduplicate_and_measure_floor_before_saving(monkeypatch, tmp_path):
    contract, plan, basis, shared, populations, factorial, config = _comparison_fixture(monkeypatch)
    state = "same-stored-hash"
    best = {"update": np.array([.01]), "delta": np.array([[.01]]), "tensor_sha256": state}
    numerical = {"norm_match": {"status": "MATCHED", "selected": {"realized_norm": .01,
                   "tensor_sha256": state}}, "calibration_sufficient": True, "direction_specific_allowance": .01}
    monkeypatch.setattr(run_factorial, "_numeric_cell", lambda *args: (best, numerical))
    monkeypatch.setattr(run_factorial, "_margins_record", lambda *args: {})
    monkeypatch.setattr(run_factorial, "_active_measurements", lambda *args, **kwargs: [
        {"decision_preserved": True, "measured_margin": .125}])
    monkeypatch.setattr(semantic_metrics, "compare", lambda *args, **kwargs: {
        "passed": True, "target_gain": 1, "regressions": [], "lost_valid": []})
    monkeypatch.setattr(run_factorial, "compare_endpoint", lambda *args: {
        "prohibited_regressions": [], "lost_valid_presentations": []})
    evaluations = []

    def measure(runner, rows, contract, *, candidate_budget=None):
        if candidate_budget:
            candidate_budget.charge()
        evaluations.append(rows)
        return {"rows": [{"valid": True}]}

    def probe(*args, seen, evaluate_candidate, **kwargs):
        if state in seen:
            return None, {"candidate_tensor_sha256": state, "reason": "duplicate_stored_candidate"}
        result = evaluate_candidate()
        assert result["guard"]["passed"]
        assert result["guard"]["target_item_gain"] == 1
        assert result["guard"]["prohibited_regressions"] == result["guard"]["lost_valid_presentations"] == []
        seen.add(state)
        return {}, {"candidate_tensor_sha256": state, "reason": "endpoint_pass", "endpoint": result}

    def refuse_save(*args):
        raise AssertionError("A measured margin equal to the floor must not materialize")

    monkeypatch.setattr(run_factorial, "measure_endpoint", measure)
    monkeypatch.setattr(run_factorial, "probe_update", probe)
    monkeypatch.setattr(run_factorial, "_materialize", refuse_save)
    detail, saved = run_factorial._compare(None, None, None, plan, basis, (), factorial,
                                          shared, populations, config, tmp_path)
    assert detail["unique_complete_evaluations"] == 1
    assert evaluations == ["fit rows", "protected rows"] and not saved
    import json
    arms = {arm: json.loads((tmp_path / f"arm-{arm}1.json").read_bytes()) for arm in "ABCD"}
    assert not arms["A"]["probe"]["endpoint"]["guard"]["passed"]
    assert all(arms[arm]["duplicate_of"] == "A1" for arm in "BCD")


@pytest.mark.parametrize("status,certified", [("UNMATCHED", True), ("MATCHED", False)])
def test_no_generation_for_unmatched_or_uncertified_numeric_state(monkeypatch, tmp_path, status, certified):
    _, plan, basis, shared, populations, factorial, config = _comparison_fixture(monkeypatch)
    best = {"update": np.array([.01]), "delta": np.array([[.01]]), "tensor_sha256": "state"}
    evidence = {"norm_match": {"status": status, "selected": {"realized_norm": .01,
                  "tensor_sha256": "state"}}, "calibration_sufficient": certified}
    monkeypatch.setattr(run_factorial, "_numeric_cell", lambda *args: (best, evidence))

    def reject_decode(*args, **kwargs):
        raise AssertionError("A refused numerical state cannot execute candidate generation")

    monkeypatch.setattr(run_factorial, "probe_update", reject_decode)
    detail, saved = run_factorial._compare(None, None, None, plan, basis, (), factorial,
                                          shared, populations, config, tmp_path)
    assert detail["unique_complete_evaluations"] == 0 and not saved


def test_run_reads_only_the_three_authorized_fitting_splits(monkeypatch, tmp_path):
    config = {"checkpoint_sha256": "parent"}
    material = tmp_path / "material"
    material.mkdir()
    (material / "manifest.json").write_text("{}", encoding="utf-8")
    loaded = []

    def load(material, split, manifest):
        assert split in ("adaptation", "preservation-fit", "calibration")
        loaded.append(split)
        return []

    @contextmanager
    def runtime(*args, **kwargs):
        yield None, None, None

    def fit(base, adapter, runner, capabilities, populations, config, out):
        assert set(populations) == set(run_factorial.FIT_SPLITS)
        return {"reason": "fixture_complete"}, {}

    monkeypatch.setattr(run_factorial, "read_json", lambda *args: config)
    monkeypatch.setattr(run_factorial, "read_manifest", lambda *args: {})
    monkeypatch.setattr(run_factorial, "load_split", load)
    monkeypatch.setattr(run_factorial, "_open", runtime)
    monkeypatch.setattr(run_factorial, "endpoint", lambda config: object())
    monkeypatch.setattr(run_factorial, "measure_endpoint", lambda *args: {"rows": []})
    monkeypatch.setattr(run_factorial, "summarize", lambda *args: {})
    monkeypatch.setattr(run_factorial, "qualify", lambda *args: {"disposition": "PASS"})
    monkeypatch.setattr(run_factorial, "_fit", fit)
    monkeypatch.setattr(run_factorial, "load_checkpoint", lambda *args: SimpleNamespace(checkpoint_sha256=lambda: "parent"))
    monkeypatch.setattr(source_manifest, "source_hashes", lambda *args: {})
    result = run_factorial.run(tmp_path / "base", material, tmp_path / "trial")
    assert loaded == list(run_factorial.FIT_SPLITS)
    assert result["hidden_opened"] is False
    with pytest.raises(FileExistsError):
        run_factorial.run(tmp_path / "base", material, tmp_path / "trial")


@pytest.mark.parametrize("counts", [(2, 2, 2, 1), (3, 1, 2, 2)])
def test_fit_refuses_a_missing_or_imbalanced_semantic_group_before_gradients(monkeypatch, tmp_path, counts):
    picked = [(f"{group}-{i}", group, ()) for group, count in zip(run_factorial.GROUP_ORDER, counts, strict=True)
              for i in range(count)]
    monkeypatch.setattr(run_factorial, "endpoint", lambda config: object())
    monkeypatch.setattr(run_factorial, "endpoint_runner", lambda runner, contract: runner)
    monkeypatch.setattr(run_factorial, "diagnose", lambda *args: (picked, {}))

    def reject_gradients(*args, **kwargs):
        raise AssertionError("Unbalanced fitting evidence must not construct a factorial direction")

    monkeypatch.setattr(run_factorial, "fit_factorial_basis", reject_gradients)
    detail, candidates = run_factorial._fit(None, None, None, None, {"adaptation": ([], {})},
                                             {"fit_limits": {"comparisons_per_group": 2}}, tmp_path)
    assert detail["reason"] == "insufficient_balanced_replay_matched_comparisons"
    assert not candidates
