import json

import numpy as np
import pytest

from weightbridge.adaptation.semantic_consensus import gradient_agreement, solve_consensus


def test_unit_item_and_equal_group_weights_limit_dominant_magnitude_and_population():
    result = solve_consensus(["p1", "p2", "p3"], ["close", "close", "far"], [[100, 0], [50, 0], [0, 1]],
                             [], [], radius=1, required_groups=("close", "far", "unseen"))
    assert result.agreement["missing_groups"] == ["unseen"]
    np.testing.assert_allclose(result.agreement["group_cosines"], np.eye(2))
    np.testing.assert_allclose(result.weighted_mean.update, np.sqrt(.5) * np.ones(2))
    np.testing.assert_allclose(result.maximin.update, result.weighted_mean.update)
    assert result.historical.record()["item_directional_slope"]["p3"] < .01
    assert result.record()["angles"]["historical:maximin"]["angle_degrees"] > 40
    json.dumps(result.record(), allow_nan=False)


def test_maximin_recovers_positive_common_direction_while_equal_mean_favors_opposing_groups():
    # Two distinct subgroup means can have unequal norm when members disagree.
    result = solve_consensus(["p1", "p2", "p3"], ["close", "close", "far"],
                             [[1, 0], [-1, 1], [1, 0]], [], [], radius=1)
    directions = result.record()["directions"]
    optimum = result.maximin.update
    assert np.linalg.norm(optimum) <= 1 + 1e-9
    assert directions["maximin"]["worst_group_normalized_improvement"] > .37
    assert (directions["maximin"]["worst_group_normalized_improvement"]
            > directions["weighted_mean"]["worst_group_normalized_improvement"] + .08)
    assert directions["maximin"]["witnessed_lower_bound"] <= directions["maximin"]["analytic_upper_bound"] + 1e-9


def test_maximin_obeys_binding_positive_preservation_floor():
    result = solve_consensus(["p1", "p2"], ["close", "far"], [[1, 0], [0, 1]], [[-1, 0]], [.5],
                             radius=1, floor=.25)
    assert result.maximin.update is not None
    row = result.maximin.record()
    assert row["protected_predicted_margins"][0] >= .25 - 1e-9
    assert row["worst_group_normalized_improvement"] == pytest.approx(.25, abs=1e-5)
    assert row["limiting_groups"]


def test_opposition_and_disconnected_groups_are_reported_separately():
    result = solve_consensus(["p1", "p2", "p3"], ["close", "far", "disconnected"], [[1, 0], [-1, 0], [0, 0]],
                             [], [], radius=1)
    assert result.agreement["item_cosines"] == [[1., -1., None], [-1., 1., None], [None, None, None]]
    assert result.agreement["zero_gradient_groups"] == ["disconnected"]
    assert result.maximin.update is None
    assert result.maximin.record()["status"] == "no_positive_common_direction"
    assert result.maximin.record()["analytic_upper_bound"] == 0
    assert result.maximin.record()["analytic_upper_bound_group_weights"] == {"close": .5, "far": .5}


def test_preservation_outside_ball_produces_no_witness():
    result = solve_consensus(["p1"], ["close"], [[1, 0]], [[1, 0]], [0.], radius=.1, floor=.5)
    assert result.maximin.update is None
    assert result.maximin.record()["preservation_projection"]["status"] == "halfspace_outside_radius"


def test_rejects_duplicate_items_and_unshared_historical_geometry():
    with pytest.raises(ValueError, match="unique underlying item"):
        gradient_agreement(["p1", "p1"], ["close", "far"], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="fixed projection"):
        solve_consensus(["p1"], ["close"], [[1, 0]], [], [], radius=1, historical_direction=[1])
