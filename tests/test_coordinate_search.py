from collections import defaultdict

from weightbridge.diagnostics.coordinate_search import disconnected_coordinate, stratified_proposals


def test_frozen_coordinate_schedule_uses_both_signs_and_skips_blocked_coordinates():
    tensors = {"input_weight": {"shape": [2, 2]}, "hidden_bias": {"shape": [2]},
               "output_weight": {"shape": [2, 2]}, "output_bias": {"shape": [2]}}
    blocked = {"input_weight": [[0, 1], [1, 1]], "hidden_bias": [[1]]}
    policy = {"allow_tensors": list(tensors), "max_abs_delta": 8.0}
    patches = list(stratified_proposals(tensors, blocked, policy, seed=122008))
    assert patches == list(stratified_proposals(tensors, blocked, policy, seed=122008))
    assert len(patches) == 8
    by_tensor = defaultdict(list)
    for patch in patches:
        assert not disconnected_coordinate(blocked, patch["tensor"], patch["index"])
        by_tensor[patch["tensor"]].append(patch)
    for tensor in tensors:
        negative, positive = by_tensor[tensor]
        assert negative["index"] == positive["index"]
        assert [negative["delta"], positive["delta"]] == [-3.0, 3.0]


def test_connectivity_gate_leaves_malformed_proposals_to_patch_validation():
    blocked = {"hidden_bias": [[1]]}
    assert disconnected_coordinate(blocked, "hidden_bias", [1])
    assert not disconnected_coordinate(blocked, "hidden_bias", [True])
    assert not disconnected_coordinate(blocked, ["hidden_bias"], [1])
    assert not disconnected_coordinate(blocked, "hidden_bias", 1)
