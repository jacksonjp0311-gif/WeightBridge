from copy import deepcopy
import hashlib

import numpy as np
import pytest

from experiments.phase_iiil.archive_trial import _private_fields, _public_bytes
from experiments.phase_iiim import analyze
from experiments.phase_iiim.analyze import _behavior_agreement, _factorial_effects, _numerical_rejections, _preferred_geometry, _stored_record, _termination_diagnostics
from experiments.phase_iiim.archive_trial import FIT_FILES, WORKER_FILES
from weightbridge.adaptation.factorial_geometry import compare_realized_norms
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


def test_archive_allowlists_reject_host_hidden_rows_and_unbound_or_traversal_paths(tmp_path):
    path = tmp_path / "arm-D2.json"
    path.write_bytes(b"{}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _public_bytes(tmp_path, path.name, FIT_FILES, digest) == b"{}"
    for forbidden in ("../arm-D2.json", "hidden.json", "verification/hidden-base.json",
                      "checkpoints/D2/config.json", "stdout.txt"):
        with pytest.raises(ValueError, match="allowlist"):
            _public_bytes(tmp_path, forbidden, FIT_FILES, digest)
    assert WORKER_FILES.fullmatch("verification/development-transfer-D2.json")
    assert WORKER_FILES.fullmatch("verification/preservation-validation-base.json")
    assert not WORKER_FILES.fullmatch("verification/hidden-base.json")
    assert _private_fields({"workers": {"base": {"sha256": digest}}, "regressions": 1}) == set()
    assert _private_fields({"payload": {"rows": [{"text": "sealed"}]}}) == {"rows", "text"}
    path.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="recorded hash"):
        _public_bytes(tmp_path, path.name, FIT_FILES, digest)


def _numeric_record():
    delta = np.array([[.0078125, 0], [0, 0]])
    norm = float(np.linalg.norm(delta))
    return {
        "rounded_delta": delta.tolist(), "left_factor": [.0078125, 0], "right_factor": [1, 0],
        "columns": [0, 1], "intended_norm": norm, "probe": {},
        "norm_match": {"status": "MATCHED", "selected": {
            "realized_norm": norm, "changed_entries": 1, "admissible": True,
            "stored_update_cosine": 1., "tensor_sha256": "state"}},
    }, {"intent": {"limits": {"max_update_entries": 4, "max_update_norm": .025}},
        "probe_design": {"norm_relative_tolerance": .02}}


def test_stored_update_provenance_recomputes_norm_support_alignment_and_label():
    record, config = _numeric_record()
    np.testing.assert_array_equal(_stored_record(record, config), record["rounded_delta"])
    altered = deepcopy(record)
    altered["rounded_delta"][1][0] = .00390625
    with pytest.raises(ValueError, match="support or realized norm"):
        _stored_record(altered, config)
    altered = deepcopy(record)
    altered["norm_match"]["status"] = "UNMATCHED"
    with pytest.raises(ValueError, match="Norm-match label"):
        _stored_record(altered, config)
    altered = deepcopy(record)
    altered["probe"] = {"endpoint": {}, "candidate_tensor_sha256": "other",
                        "realized_update_norm": .0078125, "changed_entries": 1}
    # An empty endpoint is not a decoded result and is deliberately not authorized.
    assert _stored_record(altered, config) is not None
    altered["probe"]["endpoint"] = {"measurement": {}}
    with pytest.raises(ValueError, match="Decoded checkpoint"):
        _stored_record(altered, config)


def _arms():
    gains = [0, 1, 2, 4]
    changes = [[0, 0], [1, 0], [1, 1], [2, 2]]
    return {f"{name}1": {"tensor_sha256": name, "duplicate_of": None,
                        "practice_gate": {"target_gain": gain, "local_corrected_target_presentations": list(range(gain)),
                                          "complete_family_gain": 0, "regressions": []},
                        "fit_regressions": [], "family_comparison_changes": dict(zip(("f0", "f1"), family))}
            for name, gain, family in zip("ABCD", gains, changes, strict=True)}


def test_factorial_effects_require_all_matched_distinct_decoded_states():
    arms = _arms()
    norms = compare_realized_norms({name: {"realized_norm": .01, "tensor_sha256": row["tensor_sha256"]}
                                   for name, row in arms.items()}, intended_norm=.01)
    result = _factorial_effects(arms, 1, norms)
    assert result["informative"]
    assert result["effects_on_counts"]["strict_comparison_gain"] == {
        "objective": 2.5, "normalization": 1.5, "interaction": 1.}
    assert result["family_cluster_intervals"]["interaction"]["clusters"] == 2
    unmatched = {**norms, "cohort_status": "UNMATCHED"}
    assert not _factorial_effects(arms, 1, unmatched)["informative"]
    missing = deepcopy(arms)
    del missing["B1"]["practice_gate"]
    assert not _factorial_effects(missing, 1, norms)["informative"]
    alias = deepcopy(arms)
    alias["B1"]["tensor_sha256"] = alias["A1"]["tensor_sha256"]
    assert not _factorial_effects(alias, 1, norms)["informative"]


def test_matching_stored_norm_can_fail_strict_preservation_without_claiming_unreachable_geometry():
    attempt = {
        "realized_norm": .01, "relative_norm_error": 0., "norm_matching_error": 0., "amplitude": .01,
        "tensor_sha256": "rounded", "changed_entries": 4, "feasible": False, "admissible": False,
        "minimum_predicted_margin": .3, "required_margin": .125,
        "active_measurement": {"rows": [{"prompt_id": "p", "position": 2, "group": "target", "wording": 1,
                                          "margin": .125, "replay_argmax_token_id": 3, "parent_token_id": 3}]},
    }
    record = {"floor": .125, "calibration_passes": [{"norm_match": {"attempts": [attempt]}}]}
    config = {"probe_design": {"norm_relative_tolerance": .02}, "intent": {"limits": {"max_update_norm": .025}}}
    result = _numerical_rejections(record, config)
    assert result["stored_norm_reachable_in_measured_proposals"]
    assert result["within_target_tolerance_before_preservation"] == 1
    assert result["admissible_attempts"] == 0 and result["greedy_token_refusal_attempts"] == 0
    assert result["measured_margin_refusal_attempts"] == 1 and result["linear_constraint_refusal_attempts"] == 0
    assert result["closest_measured"]["realized_norm"] == .01
    assert result["interpretation"] == "stored_norm_reachable_but_no_preservation_admissible_state"


def test_raw_direction_window_exposes_individual_conflict_without_a_global_infeasibility_claim():
    record = {"arm": "A1", "intended_norm": .2, "floor": .1, "direction_specific_allowance": 0.}
    basis = {"forks": [
        {"prompt_id": "f0", "comparison_id": "i0", "parent_margin": -.05, "projected_margin_gradient": [1., 0.]},
        {"prompt_id": "f1", "comparison_id": "i1", "parent_margin": -.05, "projected_margin_gradient": [-1., 0.]},
    ], "projected_gradients": {"MARGIN": [[1., 0.], [-1., 0.]]}}
    directions = {"arms": {"A": {"aggregate": [1., 0.]}}}
    secants = {"A": {"rows": [{"prompt_id": "protected", "position": 0, "slope": -1.,
                               "parent_replay": {"margin": .5}}]}}
    anchor = DecisionAnchor("saved", "protected", "target", 0, 0, .5, 2)
    result = _preferred_geometry(record, basis, directions, secants, [anchor])
    assert result["available"] and result["kind"] == "raw_preferred_direction_gradient_and_measured_BF16_finite_secant"
    assert result["positive_target_fork_slopes"] == 1 and result["target_fork_count"] == 2
    assert result["local_feasible_region"]["all_target_interval"] is None
    assert result["adverse_protected_scan_positions"] == result["sampled_adverse_positions"] == 1
    assert "does not establish global" in result["assumptions"][-1]


def test_behavior_agreement_counts_valid_answers_and_tokens_separately():
    first = {"rows": [
        {"id": "p0", "valid": True, "exact": "-1", "correct": True, "generated_token_ids": [1]},
        {"id": "p1", "valid": True, "exact": "-2", "correct": False, "generated_token_ids": [2]},
        {"id": "p2", "valid": False, "exact": None, "correct": False, "generated_token_ids": [3]},
    ]}
    second = deepcopy(first)
    second["rows"][0]["generated_token_ids"] = [4]  # Different valid representation of the same answer.
    second["rows"][1].update(exact="-1", correct=True, generated_token_ids=[1])
    result = _behavior_agreement(first, second)
    assert result["both_valid"] == 2 and result["same_valid_semantic_answer"] == 1
    assert result["same_correctness"] == 2 and result["same_generated_token_sequence"] == 1
    assert result["different_correctness_ids"] == ["p1"]


def test_fork_kind_join_distinguishes_recorded_prefix_from_the_actual_candidate_path(monkeypatch):
    metadata = {"pair_id": "family"}
    basis = {"forks": [
        {"prompt_id": "p0", "comparison_id": "i0", "actual_token_id": 5,
         "accepted_token_ids": [2], "metadata": metadata},
        {"prompt_id": "p1", "comparison_id": "i0", "actual_token_id": 5,
         "accepted_token_ids": [9], "metadata": metadata},
    ]}
    prior = {"numeric_surface_tokenizations": {"0": [2], ".": [3]}, "newline_tokenization": [9], "rows": [
        {"prompt_id": prompt, "accepted_token_ids": accepted, "accepted_probability": .1,
         "numeric_subset_probability": .8, "termination_subset_probability": .1}
        for prompt, accepted in (("p0", [2]), ("p1", [9]))]}
    current = {"rows": [{**row, "accepted_probability": .7, "numeric_subset_probability": .5,
                         "termination_subset_probability": .4, "greedy_token_id": token}
                        for row, token in zip(prior["rows"], (2, 9), strict=True)]}
    record = {"probe": {"fork": {"candidate_predicted_ids": [2, 9]},
                       "first_divergences": [{"parent_phase": "inside_answer", "candidate_token_id": 2}],
                       "endpoint": {"measurement": {"rows": [
                           {"id": "p0", "correct": True, "generated_token_ids": [1, 2]},
                           {"id": "p1", "correct": True, "generated_token_ids": [7, 9]},
                       ]}, "fork_token_classes": current}}}
    diagnosis = {"forks": [{"prompt_id": prompt, "decoding": {"eos_token_id": 10}, "shared_token_ids": [1]}
                            for prompt in ("p0", "p1")]}
    monkeypatch.setattr(analyze, "semantic_summary", lambda _: {
        "families": {"family": {"original": "i0", "counterfactual": "i1", "status": "only_original_correct"}},
        "comparisons": {"i1": {"all_correct": False}}})
    result = _termination_diagnostics(record, basis, prior, diagnosis)
    assert result["accepted_changed_prefix_tokens_by_kind"]["digit"] == {
        "accepted_changes": 1, "completed_correct": 1, "visited_parent_prefix": 1}
    assert result["accepted_changed_prefix_tokens_by_kind"]["newline"]["visited_parent_prefix"] == 0
    assert not result["fork_rows"][1]["full_generation_visited_parent_prefix"]
    assert not result["fork_rows"][0]["counterfactual_comparison_correct"]
    assert result["first_divergence_parent_phases"] == {"inside_answer": 1}


def test_development_factorial_uses_fresh_family_clusters_and_its_own_declared_scope():
    gains, changes = [3, 1, 3, 1], [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]]
    arms = {f"{name}2": {
        "tensor_sha256": name, "fit_regressions": [],
        "development_gate": {"target_gain": gain, "local_corrected_target_presentations": list(range(gain)),
                             "complete_family_gain": 0, "regressions": []},
        "family_comparison_changes": dict(zip(("f0", "f1", "f2", "f3"), family)),
    } for name, gain, family in zip("ABCD", gains, changes, strict=True)}
    norms = compare_realized_norms({name: {"realized_norm": .02, "tensor_sha256": row["tensor_sha256"]}
                                   for name, row in arms.items()}, intended_norm=.02)
    result = _factorial_effects(arms, 2, norms, gate_key="development_gate", scope="Fresh development-transfer.")
    assert result["informative"] and result["scope"] == "Fresh development-transfer."
    assert result["effects_on_counts"]["strict_comparison_gain"] == {
        "objective": 0., "normalization": -2., "interaction": 0.}
    assert result["family_cluster_intervals"]["normalization"]["clusters"] == 4
