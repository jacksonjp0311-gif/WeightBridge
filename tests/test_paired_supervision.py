import json

import numpy as np
import pytest

from weightbridge.adaptation.paired_supervision import (
    ComparisonEvidence, mean_direction, pair_gradient_diagnostics,
    select_supervision, solve_pair_maximin,
)


def records(count=4):
    result = []
    for distance in ("close", "far"):
        for index in range(count):
            family = f"{distance}-{index}"
            for length in ("short", "long"):
                item = f"{family}-{length}"
                result.append(ComparisonEvidence(item, family, f"{distance}-{length}",
                    (item + "-w0", item + "-w1"), (item + "-correct",)))
    return result


def test_supervision_allocations_match_comparison_budgets_but_report_different_family_diversity():
    selected = select_supervision(records())
    assert selected.record()["status"] == "READY"
    assert selected.arm_items["B"] == selected.arm_items["C"]
    assert len(selected.arm_items["A"]) == len(selected.arm_items["B"]) == 8
    for name, arm in selected.record()["arms"].items():
        assert arm["comparison_weight"] == 1 / 8
        assert arm["group_counts"] == {"close-short": 2, "close-long": 2, "far-short": 2, "far-long": 2}
        assert arm["family_count"] == (8 if name == "A" else 4)
    assert selected.arm_items["A"][:4] == (
        "close-0-short", "close-1-long", "far-0-short", "far-1-long")
    assert set(selected.arm_items["B"]) == {
        f"{distance}-{index}-{length}" for distance in ("close", "far")
        for index in (0, 1) for length in ("short", "long")}
    assert "not pairing alone" in selected.record()["causal_limitation"]
    json.dumps(selected.record(), allow_nan=False)


def test_insufficient_dual_error_families_hold_without_relabeling_parent_correct_comparisons():
    source = records()
    changed = [ComparisonEvidence(row.comparison_id, row.family_id, row.group,
                row.eligible_prompt_ids if row.group.endswith("short") or row.family_id.endswith("0") else (),
                row.parent_correct_prompt_ids) for row in source]
    result = select_supervision(changed)
    assert result.record()["status"] == "HOLD"
    assert result.record()["reason"] in {
        "insufficient_one_sided_family_diversity", "insufficient_dual_error_families"}
    population = {row["comparison_id"]: row for row in result.record()["population"]}
    assert not population["close-1-long"]["eligible"]
    assert population["close-1-long"]["parent_correct_preservation_prompt_ids"] == ["close-1-long-correct"]
    assert "close-1-long" not in result.arm_items["B"]


def test_allocation_is_order_independent_and_cannot_manufacture_family_diversity():
    source = records()
    assert select_supervision(source).arm_items == select_supervision(list(reversed(source))).arm_items
    too_few_families = select_supervision(records(count=2))
    assert too_few_families.record()["status"] == "HOLD"
    assert too_few_families.record()["reason"] == "insufficient_one_sided_family_diversity"
    assert len(too_few_families.arm_items["A"]) == 4
    with pytest.raises(ValueError, match="balanced equal comparison"):
        select_supervision(source, one_sided_count=4, paired_family_count=4)
    with pytest.raises(ValueError, match="unique comparison"):
        select_supervision(source + source[:1])


def test_replayqualified_error_and_parentcorrect_identity_must_be_disjoint():
    with pytest.raises(ValueError, match="disjoint qualified errors"):
        ComparisonEvidence("one", "family", "close-short", ("w0", "w1"), ("w0",))
    with pytest.raises(ValueError, match="multiple comparison"):
        select_supervision([ComparisonEvidence("one", "f0", "close-short", ("w0", "w1")),
                            ComparisonEvidence("two", "f1", "far-short", ("w0", "w1"))])
    with pytest.raises(ValueError, match="both answer-length"):
        select_supervision([ComparisonEvidence("one", "f", "close-short", ("x", "y")),
                            ComparisonEvidence("two", "f", "close-short", ("a", "b"))])


def test_raw_mean_weights_each_comparison_once_and_preserves_corrective_sign():
    # The first row can already average four qualified wording/order forks, while
    # the second averages two. Their total weights remain equal here.
    np.testing.assert_allclose(mean_direction(["first", "second"], [[6, 2], [0, 2]]), [3, 2])
    assert float(np.array([6, 2]) @ mean_direction(["first", "second"], [[6, 2], [0, 2]])) > 0
    np.testing.assert_array_equal(mean_direction(["first", "second"], [[0, 0], [0, 0]]), [0, 0])
    with pytest.raises(ValueError, match="unique comparison"):
        mean_direction(["a", "a"], [[1, 0], [0, 1]])


def test_pair_gradient_agreement_includes_norm_imbalance_and_individual_slopes():
    result = pair_gradient_diagnostics(["a", "b", "c", "d"], ["one", "one", "two", "two"],
        ["close-short", "close-long", "far-short", "far-long"], [[10, 0], [1, 0], [0, 1], [0, -1]],
        margins=[-1, -2, -3, -4])
    first, second = result["families"]["one"], result["families"]["two"]
    assert first["cosine"] == 1
    assert first["norm_imbalance_ratio"] == first["norm_ratio_first_to_second"] == 10
    assert first["relationship"] == "compatible" and first["both_mean_slopes_positive"]
    assert second["cosine"] == -1 and second["relationship"] == "conflicting"
    assert second["mean_direction_slopes"] == [0, 0]
    assert second["parent_comparison_margins"] == [-3, -4]
    json.dumps(result, allow_nan=False)


def test_pair_diagnostics_do_not_invent_cosines_for_zero_or_missing_opposing_rows():
    result = pair_gradient_diagnostics(["a", "b", "c"], ["one", "one", "solo"],
        ["short", "long", "short"], [[0, 0], [1, 0], [0, 1]])
    pair = result["families"]["one"]
    assert pair["cosine"] is None and pair["norm_imbalance_ratio"] is None
    assert pair["relationship"] == "zero_gradient"
    assert not result["families"]["solo"]["paired"]
    assert "cosine" not in result["families"]["solo"]


def test_pair_maximin_recovers_symmetric_analytic_witness_without_dense_hessians():
    solved = solve_pair_maximin(["a", "b"], ["family", "family"], [[1, 0], [0, 1]], [-1, -1],
                                [], [], radius=2, max_search_steps=24)
    np.testing.assert_allclose(solved.update, [np.sqrt(2), np.sqrt(2)], atol=1e-6)
    assert solved.record()["witnessed_minimum_margin"] == pytest.approx(np.sqrt(2) - 1, abs=1e-6)
    assert solved.record()["status"] == "positive_joint_margin_witness"
    assert solved.record()["all_predicted_comparisons_correct"]
    assert not solved.record()["bounded_search_is_infeasibility_certificate"]
    assert np.linalg.norm(solved.update) <= 2 + 1e-9
    json.dumps(solved.record(), allow_nan=False)


def test_absolute_margin_objective_accounts_for_unequal_parent_deficits():
    solved = solve_pair_maximin(["a", "b"], ["f", "f"], [[1, 0], [0, 1]], [-.1, -.9],
                                [], [], radius=1, max_search_steps=24)
    changes = solved.record()["predicted_margin_changes"]
    assert changes["b"] > changes["a"]
    assert solved.record()["predicted_margins"]["a"] == pytest.approx(
        solved.record()["predicted_margins"]["b"], abs=2e-6)


def test_conflicting_pair_returns_nonpositive_witness_instead_of_claiming_rule_correction():
    solved = solve_pair_maximin(["a", "b"], ["f", "f"], [[1, 0], [-1, 0]], [-1, -1],
                                [], [], radius=.025)
    np.testing.assert_array_equal(solved.update, [0, 0])
    assert solved.record()["status"] == "bounded_nonpositive_margin_witness"
    assert solved.record()["witnessed_minimum_margin"] == -1
    assert not solved.record()["all_predicted_comparisons_correct"]


def test_preservation_halfspaces_are_enforced_without_forcing_exact_old_margins():
    solved = solve_pair_maximin(["a", "b"], ["f", "f"], [[1, 0], [0, 1]], [-1, -1],
                                [[-1, 0]], [.3], radius=2, floor=.2)
    assert solved.update[0] <= .1 + 1e-8
    assert solved.record()["protected_predicted_margins"][0] >= .2 - 1e-8
    assert not solved.record()["all_predicted_comparisons_correct"]


def test_impossible_or_budget_limited_preservation_never_returns_unsafe_update():
    impossible = solve_pair_maximin(["a", "b"], ["f", "f"], [[1], [1]], [-1, -1],
                                    [[0]], [0], radius=.1, floor=.2)
    assert impossible.update is None and impossible.record()["status"] == "no_preservation_witness"
    assert impossible.record()["preservation_baseline"]["status"] == "impossible_constant_constraint"
    with pytest.raises(ValueError, match="bounded budgets"):
        solve_pair_maximin(["a", "b"], ["f", "f"], [[1], [1]], [-1, -1],
                           [], [], radius=.1, max_search_steps=0)
    with pytest.raises(ValueError, match="paired finite rows"):
        solve_pair_maximin(["a", "b"], ["f", "other"], [[1], [1]], [-1, -1], [], [], radius=.1)
    limited = solve_pair_maximin(["a", "b"], ["f", "f"], [[1, 0], [0, 1]], [-1, -1],
                                [], [], radius=2, max_search_steps=1, max_projection_iterations=1)
    assert len(limited.record()["search_steps"]) <= 1
    assert not limited.record()["bounded_search_is_infeasibility_certificate"]
    if limited.update is not None:
        assert np.linalg.norm(limited.update) <= 2 + 1e-9


@pytest.mark.parametrize("preview_update", [None, np.zeros(2)])
def test_pair_preview_failure_keeps_other_allocations_available(preview_update, monkeypatch, tmp_path):
    """A legitimate C cancellation may not suppress the A/B scientific contrast."""
    from types import SimpleNamespace
    from experiments.phase_iiin import run_paired as experiment
    from weightbridge.adaptation.paired_supervision import PairDirection

    config = experiment.read_json(experiment.PROTOCOL)
    forks = {name: [SimpleNamespace(prompt_id=prompt)] for name, prompt in (("a", "p0"), ("b", "p1"))}
    rows = [{"id": prompt, "correct": False, "pair_id": family, "comparison_id": item,
             "order": 0, "wording": 0, "tier": "negative-close", "truth_precision": length}
            for prompt, family, item, length in (("p0", "fa", "a", "shorter"), ("p1", "fb", "b", "longer"))]
    populations = {"adaptation": (rows, {"rows": rows}),
                   "preservation-fit": ([], {"rows": []}), "calibration": ([], {"rows": []})}
    selection = SimpleNamespace(arm_items={"A": ("a",), "B": ("b",), "C": ("b",)},
                                record=lambda: {"status": "READY"})
    plan = SimpleNamespace(limits=SimpleNamespace(max_forward_calls=480, max_backward_calls=96))
    basis = SimpleNamespace(columns=(0,), right=np.array([1.]))
    evidence = {"item_ids": ["a", "b"], "item_groups": ["close-short", "close-long"],
                "comparison_parent_margins": [-1, -1]}
    monkeypatch.setattr(experiment, "endpoint_runner", lambda runner, contract: runner)
    monkeypatch.setattr(experiment, "diagnose", lambda *args: ([], {"forks": [
        {"prompt_id": prompt, "actual_category": "digit", "accepted_categories": ["digit"]}
        for prompt in ("p0", "p1")]}))
    monkeypatch.setattr(experiment, "_eligible_comparisons", lambda *args: (
        [], forks, {"a": "close-short", "b": "close-long"}, {"a": "fa", "b": "fb"}))
    monkeypatch.setattr(experiment, "select_supervision", lambda *args: selection)
    monkeypatch.setattr(experiment, "_plan", lambda *args: plan)
    monkeypatch.setattr(experiment, "_reference_geometry", lambda *args: ((0,), np.array([1.]), {}))
    monkeypatch.setattr(experiment, "fit_paired_basis", lambda *args, **kwargs: (
        basis, np.eye(2), evidence))
    monkeypatch.setattr(experiment, "solve_pair_maximin", lambda *args, **kwargs: PairDirection(
        preview_update, {"status": "fixture_no_common_direction"}))
    observed = {}

    def prepare(*args):
        observed.update(args[4])
        return "shared", {}

    monkeypatch.setattr(experiment, "_prepare", prepare)
    monkeypatch.setattr(experiment, "_compare", lambda *args: ({"reason": "A_B_continued"}, {}))
    detail, _ = experiment._fit(None, None, None, None, populations, config, tmp_path)
    assert detail["reason"] == "A_B_continued"
    np.testing.assert_array_equal(observed["A"], [1., 0.])
    np.testing.assert_array_equal(observed["B"], [0., 1.])
    np.testing.assert_array_equal(observed["C"], [0., 0.])
    recorded = json.loads((tmp_path / "directions.json").read_bytes())
    assert not recorded["pair_aware_preview"]["preview_eligible"]
    json.dumps(recorded, allow_nan=False)
