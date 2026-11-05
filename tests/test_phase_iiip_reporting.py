"""Reporting fixtures are synthetic validation cases, never model evidence."""
from copy import deepcopy
import hashlib
import io
import json

import numpy as np
import pytest

from experiments.phase_iiip import analyze
from experiments.phase_iiip.archive_trial import FIT_FILES, WORKER_FILES
from experiments.phase_iiil.archive_trial import _public_bytes
from weightbridge.adaptation.joint_feasibility import solve_minimum_norm
from weightbridge.adaptation.right_factor_geometry import solve_block_relaxation


def _diagonal():
    return np.array([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]]), np.array([.01, .02])


def test_independent_block_bound_checks_frobenius_and_spectral_arithmetic():
    blocks, rhs = _diagonal()
    result = solve_block_relaxation(blocks, rhs, radius=.025, tolerance=1e-12)
    checked = analyze.verify_block_bound(blocks, rhs, result.record())
    assert checked["independently_checked"]
    assert checked["certified_norm_lower_bound"] == pytest.approx(np.linalg.norm(rhs), abs=1e-10)
    assert checked["rank_one_norm_lower_bound"] == pytest.approx(.025, abs=1e-10)


@pytest.mark.parametrize("field", ["dual_objective", "dual_cauchy_norm_lower_bound", "certified_norm_lower_bound",
                                  "rank_one_norm_lower_bound", "dual_matrix_spectral_norm"])
def test_independent_reporting_rejects_tampered_bound_without_solver_replay(field):
    blocks, rhs = _diagonal()
    record = deepcopy(solve_block_relaxation(blocks, rhs, radius=.025).record())
    record["certificate"][field] += .1
    with pytest.raises(ValueError, match="independently"):
        analyze.verify_block_bound(blocks, rhs, record)


def test_independent_block_arithmetic_rejects_rows_and_negative_multipliers():
    blocks, rhs = _diagonal()
    record = solve_block_relaxation(blocks, rhs, radius=.025).record()
    with pytest.raises(ValueError, match="input identity"):
        analyze.verify_block_bound(blocks * 2, rhs, record)
    damaged = deepcopy(record)
    damaged["certificate"]["original_dual_multipliers"][0] = -.1
    with pytest.raises(ValueError, match="nonnegative"):
        analyze.verify_block_bound(blocks, rhs, damaged)


def test_zero_block_proof_is_not_a_failed_bounded_search():
    blocks, rhs = np.zeros((1, 2, 2)), np.array([.01])
    evidence = solve_block_relaxation(blocks, rhs, radius=.025).record()
    assert analyze.verify_block_bound(blocks, rhs, evidence) == {"zero_block_verified": True}
    evidence["status"] = "BOUNDED_NO_WITNESS"
    with pytest.raises(ValueError, match="zero-block"):
        analyze.verify_block_bound(blocks, rhs, evidence)


def test_vector_dual_is_checked_in_original_not_normalized_coordinates():
    rows, rhs = np.array([[3., 1.], [0., 2.]]), np.array([.03, .01])
    record = solve_minimum_norm(rows, rhs, radius=.025).record()
    assert analyze.verify_vector_bound(rows, rhs, record)["independently_checked_dual"]
    altered = deepcopy(record)
    altered["certificate"]["original_dual_multipliers"] = altered["certificate"]["normalized_dual_multipliers"]
    with pytest.raises(ValueError, match="objective differs"):
        analyze.verify_vector_bound(rows, rhs, altered)


def test_native_reporting_projects_the_actual_candidate_right_factor():
    basis = {"forks": [{"prompt_id": "wrong", "parent_margin": -.01,
                        "accepted_token_ids": [1], "projected_margin_gradient": [0.]}]}
    block = np.array([[[0., 10.]]])
    native = {"targets": {"rows": [{"prompt_id": "wrong", "measured_margin": .125,
        "accepted_token_ids": [1], "greedy_token_id": 1, "accepted_token_selected": True}]}}
    useful = analyze._projected_basis(basis, block, np.array([0., 1.]))
    useless = analyze._projected_basis(basis, block, np.array([1., 0.]))
    assert analyze.validate_native_targets(native, useful, np.array([.01]), .0625)["passed"]
    assert not analyze.validate_native_targets(native, useless, np.array([.01]), .0625)["passed"]


def test_unmeasured_eight_response_family_is_null_not_behavioral_failure():
    rows = [{"id": f"{variant}-{order}-{wording}", "comparison_id": f"c{variant}", "counterfactual": variant,
        "order": order, "wording": wording, "group": "target", "pair_id": "family", "correct": False,
        "exact": "-2", "answer": "-1"} for variant in (0, 1) for order in (0, 1) for wording in (0, 1)]
    matrix = analyze.response_matrix({"rows": rows}, None, {}, None, None, ["family"])
    assert matrix["family"]["candidate_complete"] is None
    assert len(matrix["family"]["rows"]) == 8
    assert all(row["candidate_correct"] is None and not row["behavior_measured"] for row in matrix["family"]["rows"])


def _stored():
    left, right = np.array([.005, .007]), np.array([.8, .6])
    proposed = np.outer(left, right)
    delta = proposed + np.array([[1e-5, -2e-5], [-1e-5, 1e-5]])
    singular = np.linalg.svd(delta, compute_uv=False)
    record = {"left_factor": left.tolist(), "right_factor": right.tolist(), "rounded_delta": delta.tolist(),
        "columns": [2, 3], "proposed_norm": float(np.linalg.norm(proposed)), "realized_norm": float(np.linalg.norm(delta)),
        "changed_entries": 4, "rounding_residual_norm": float(np.linalg.norm(delta-proposed)),
        "stored_singular_values": singular.tolist(), "stored_numerical_rank": 2}
    config = {"intent": {"limits": {"max_update_norm": .025, "max_update_entries": 16384}}}
    return record, {"columns": [2, 3]}, config


def test_stored_precision_rank_and_residual_are_reported_without_expanding_proposed_geometry():
    record, basis, config = _stored()
    result = analyze._stored_state(record, basis, config)
    assert np.linalg.matrix_rank(np.outer(record["left_factor"], record["right_factor"])) == 1
    assert np.linalg.matrix_rank(result) == 2
    record["stored_numerical_rank"] = 1
    with pytest.raises(ValueError, match="rounding"):
        analyze._stored_state(record, basis, config)


def test_binary_archive_hash_allowlist_and_fresh_destination_boundaries(tmp_path):
    raw = io.BytesIO()
    np.savez(raw, targets=np.ones((1, 2, 2)), preservation=np.zeros((0, 2, 2)))
    payload = raw.getvalue()
    path = tmp_path / "derivative-blocks.npz"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert _public_bytes(tmp_path, path.name, FIT_FILES, expected) == payload
    assert analyze._arrays(payload)["targets"].shape == (1, 2, 2)
    with pytest.raises(ValueError, match="recorded hash"):
        _public_bytes(tmp_path, path.name, FIT_FILES, "a" * 64)
    assert FIT_FILES.fullmatch("target-blocks.npz")
    assert FIT_FILES.fullmatch("target-block-metadata.json")
    assert not FIT_FILES.fullmatch("selection.json")
    assert not FIT_FILES.fullmatch("hidden.json")
    assert not WORKER_FILES.fullmatch("verification/hidden-base.json")
    with pytest.raises(ValueError, match="allowlist"):
        _public_bytes(tmp_path, "hidden.json", FIT_FILES)


def test_numerical_archive_never_unpickles_or_accepts_unknown_keys():
    raw = io.BytesIO()
    np.savez(raw, targets=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        analyze._arrays(raw.getvalue())
    raw = io.BytesIO()
    np.savez(raw, private_roster=np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="unknown"):
        analyze._arrays(raw.getvalue())


def _provenance():
    blocks = np.array([[[0., 10.]]])
    prefix = [4, 5]
    fork = {"prompt_id": "wrong", "parent_margin": -.01, "accepted_token_ids": [1],
            "actual_token_id": 2, "metadata": {"family": "synthetic"}, "projected_margin_gradient": [0.]}
    metadata = {"checkpoint_sha256": "parent", "tensor_name": "affine", "source_tensor_sha256": "source",
        "source_dtype": "BF16", "columns": [0, 1], "block_shape": [1, 2], "parent_restored_exactly": True,
        "block_array_sha256": analyze._hash(blocks), "context_only_rows_have_supervision_weight": False,
        "packing": [{"prompt_ids": ["wrong", "context-only"], "native_prefix_ids": [prefix, [6]],
            "original_token_ids": [2, 3], "replay_token_ids": [2, 3], "selected_prompt_ids": ["wrong"]}],
        "rows": [{**fork, "native_prefix_ids": prefix, "replay_argmax_token_id": 2, "original_context_index": 0,
            "native_prefix_sha256": hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode("ascii")).hexdigest(),
            "gradient_block_sha256": analyze._hash(blocks[0])}]}
    return metadata, blocks, {"columns": [0, 1], "right_factor": [1., 0.], "forks": [fork]}


def test_derivative_provenance_binds_original_native_context_and_parent():
    metadata, blocks, basis = _provenance()
    analyze.verify_block_provenance(metadata, blocks, name="targets", basis=basis,
        checkpoint_sha256="parent", tensor_name="affine", source_tensor_sha256="source")


@pytest.mark.parametrize("field", ["prefix", "context", "parent", "projection"])
def test_derivative_provenance_rejects_changed_context_or_unearned_projection(field):
    metadata, blocks, basis = _provenance()
    if field == "prefix":
        metadata["rows"][0]["native_prefix_ids"] = [4, 6]
    elif field == "context":
        metadata["packing"][0]["selected_prompt_ids"] = []
    elif field == "parent":
        metadata["source_tensor_sha256"] = "different-source"
    else:
        basis["forks"][0]["projected_margin_gradient"] = [1.]
    with pytest.raises(ValueError, match="differs|differ"):
        analyze.verify_block_provenance(metadata, blocks, name="targets", basis=basis,
            checkpoint_sha256="parent", tensor_name="affine", source_tensor_sha256="source")
