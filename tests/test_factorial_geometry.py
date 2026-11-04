import json

import numpy as np
import pytest

from weightbridge.adaptation.factorial_geometry import (
    build_factorial_directions, compare_realized_norms, match_stored_norm, stored_update_cosines,
)


def test_four_conditions_separate_objective_and_unit_scaling_with_equal_item_weights():
    ce = np.array([[100., 0], [1, 0], [0, 2], [0, 2]])
    margin = np.array([[0., 10], [0, 1], [3, 0], [3, 0]])
    result = build_factorial_directions(["i0", "i1", "i2", "i3"], ["short", "short", "long", "long"], ce, margin)
    np.testing.assert_allclose(result.directions["A"], [25.25, 1])
    np.testing.assert_allclose(result.directions["B"], [.5, .5])
    np.testing.assert_allclose(result.directions["C"], [1.5, 2.75])
    np.testing.assert_allclose(result.directions["D"], [.5, .5])
    evidence = result.record()
    assert evidence["equal_group_counts"]
    assert set(evidence["angles"]) == {"A:B", "A:C", "A:D", "B:C", "B:D", "C:D"}
    for arm in evidence["arms"].values():
        assert arm["item_weight"] == .25
        assert not arm["group_reweighting_present"]
        assert arm["equal_group_vs_item_mean_distance"] == pytest.approx(0)
        assert sum(arm["item_signed_projection_share"].values()) == pytest.approx(1)
        assert sum(group["signed_projection_share"] for group in arm["group_contribution"].values()) == pytest.approx(1)
    assert evidence["arms"]["A"]["item_contribution_norm"]["i0"] == 25
    assert evidence["arms"]["B"]["item_contribution_norm"]["i0"] == .25
    assert evidence["angles"]["B:D"]["cosine"] == pytest.approx(1)
    json.dumps(evidence, allow_nan=False)


def test_zero_rows_remain_in_item_denominator_and_direction_cosines_are_undefined_when_zero():
    result = build_factorial_directions(["a", "b"], ["one", "two"], [[0, 0], [0, 3]], [[0, 0], [0, 0]])
    np.testing.assert_allclose(result.directions["B"], [0, .5])
    np.testing.assert_allclose(result.directions["D"], [0, 0])
    assert result.record()["arms"]["B"]["zero_item_ids"] == ["a"]
    assert result.record()["arms"]["B"]["item_scale"]["a"] == 0
    assert result.record()["angles"]["B:D"]["cosine"] is None
    assert result.record()["arms"]["D"]["item_signed_projection_share"] == {"a": None, "b": None}


def test_unequal_group_counts_are_reported_as_distinct_from_global_item_averaging():
    result = build_factorial_directions(["a", "b", "c"], ["one", "one", "two"],
                                       [[1, 0], [1, 0], [0, 1]], [[1, 0], [1, 0], [0, 1]])
    arm = result.record()["arms"]["D"]
    np.testing.assert_allclose(result.directions["D"], [2 / 3, 1 / 3])
    np.testing.assert_allclose(arm["equal_group_mean"], [.5, .5])
    assert arm["group_reweighting_present"] and arm["equal_group_vs_item_mean_distance"] > 0
    with pytest.raises(ValueError, match="unique semantic"):
        build_factorial_directions(["a", "a"], ["one", "one"], [[1], [1]], [[1], [1]])
    with pytest.raises(ValueError, match="same unique"):
        build_factorial_directions(["a"], ["one"], [[1, 2]], [[1]])


def test_measured_norm_search_corrects_amplitude_and_never_selects_an_over_budget_state():
    calls = []

    def measure(amplitude):
        calls.append(amplitude)
        actual = 2 * amplitude
        return {"realized_norm": actual, "feasible": True, "tensor_sha256": f"{actual:.12f}",
                "changed_entries": 2, "delta": np.array([actual]), "update": np.array([amplitude])}

    matched = match_stored_norm(measure, intended_norm=.02, maximum_norm=.025)
    assert matched.record()["status"] == "MATCHED"
    assert matched.selected["realized_norm"] == pytest.approx(.02)
    assert calls[0] == .02 and all(amplitude <= .025 for amplitude in calls)
    assert not matched.record()["attempts"][0]["admissible"]
    assert matched.selected["amplitude"] == .01
    assert len(calls) <= 10
    assert matched.record()["selected"]["relative_norm_error"] <= .02
    json.dumps(matched.record(), allow_nan=False)


def test_quantized_plateau_is_unmatched_and_identical_stored_states_are_recorded():
    def plateau(amplitude):
        actual = min(amplitude, .006)
        return {"realized_norm": actual, "feasible": True, "tensor_sha256": str(actual), "changed_entries": 1}

    result = match_stored_norm(plateau, intended_norm=.02, maximum_norm=.025, max_probes=7)
    assert result.record()["status"] == "UNMATCHED"
    assert result.selected["realized_norm"] == .006
    assert result.record()["selected"]["norm_matching_error"] == pytest.approx(.014)
    assert any(attempt["duplicate_stored_state"] for attempt in result.record()["attempts"])
    assert len(result.record()["attempts"]) <= 7
    assert result.record()["unique_stored_states"] == 1


def test_no_constraint_witness_and_zero_stored_update_cannot_match():
    rejected = match_stored_norm(lambda _: {"realized_norm": None, "feasible": False,
                                           "tensor_sha256": None, "reason": "no_feasible_witness"},
                                 intended_norm=.01, maximum_norm=.025, max_probes=3)
    assert rejected.selected is None and rejected.record()["status"] == "UNMATCHED"
    assert 1 <= len(rejected.record()["attempts"]) <= 3
    noop = match_stored_norm(lambda _: {"realized_norm": 0., "feasible": True, "tensor_sha256": "parent"},
                             intended_norm=.01, maximum_norm=.025)
    assert noop.selected is None and noop.record()["reason"] == "no_admissible_stored_state"
    with pytest.raises(ValueError, match="finite stored norm"):
        match_stored_norm(lambda _: {"realized_norm": float("nan"), "feasible": True},
                          intended_norm=.01, maximum_norm=.025)
    with pytest.raises(ValueError, match="positive authorized"):
        match_stored_norm(lambda _: {}, intended_norm=.03, maximum_norm=.025)


def test_actual_bfloat16_rounding_can_remove_or_distort_a_float_update():
    source = np.array([1., -.5], dtype=np.float32)
    direction = np.array([.6, .8], dtype=np.float32)

    def bfloat_round(values):
        bits = values.view(np.uint32)
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return rounded.view(np.float32)

    def measure(amplitude):
        rounded = bfloat_round(source + amplitude * direction)
        delta = rounded.astype(np.float64) - source
        return {"realized_norm": float(np.linalg.norm(delta)), "feasible": True,
                "tensor_sha256": rounded.tobytes().hex(), "changed_entries": int(np.count_nonzero(delta)), "delta": delta}

    tiny = match_stored_norm(measure, intended_norm=.0001, maximum_norm=.0002)
    assert tiny.record()["status"] == "UNMATCHED" and tiny.selected is None
    target = measure(.0125)["realized_norm"]
    result = match_stored_norm(measure, intended_norm=target, maximum_norm=.025)
    assert result.selected["realized_norm"] <= .025
    assert all(row["realized_norm"] == 0 or row["changed_entries"] > 0 for row in result.record()["attempts"])


def test_stored_delta_comparison_reports_exact_identity_and_coordinate_differences():
    result = stored_update_cosines({"A": np.array([[1., 0.]]), "B": np.array([[1., 0.]]),
                                   "C": np.array([[0., 2.]])})
    assert result["A:B"] == {"cosine": 1., "difference_norm": 0., "different_entries": 0,
                             "identical_stored_update": True}
    assert result["A:C"]["cosine"] == 0
    assert result["A:C"]["difference_norm"] == pytest.approx(np.sqrt(5))
    assert result["A:C"]["different_entries"] == 2
    with pytest.raises(ValueError, match="shared tensor coordinates"):
        stored_update_cosines({"A": [[1, 2]], "B": [[1]]})


def test_individual_target_tolerance_does_not_imply_pairwise_or_cohort_matching():
    result = compare_realized_norms({"A": {"realized_norm": .00981, "tensor_sha256": "a"},
                                    "B": {"realized_norm": .01019, "tensor_sha256": "b"},
                                    "C": {"realized_norm": .01, "tensor_sha256": "c"}},
                                   intended_norm=.01)
    assert all(row["within_target_tolerance"] for row in result["individual"].values())
    assert not result["pairs"]["A:B"]["matched"]
    assert result["pairs"]["A:C"]["matched"] and result["pairs"]["B:C"]["matched"]
    assert result["relative_cohort_range"] == pytest.approx(.038)
    assert result["cohort_status"] == "UNMATCHED"
    same = compare_realized_norms({"A": {"realized_norm": .0099, "tensor_sha256": "same"},
                                  "B": {"realized_norm": .0099, "tensor_sha256": "same"}},
                                 intended_norm=.01)
    assert same["cohort_status"] == "MATCHED" and same["pairs"]["A:B"]["same_stored_tensor"]
    outside = compare_realized_norms({"A": {"realized_norm": .02}, "B": {"realized_norm": .02}}, intended_norm=.01)
    assert not outside["pairs"]["A:B"]["matched"] and outside["cohort_status"] == "UNMATCHED"
    with pytest.raises(ValueError, match="finite nonnegative"):
        compare_realized_norms({"A": {"realized_norm": float("nan")}}, intended_norm=.01)
