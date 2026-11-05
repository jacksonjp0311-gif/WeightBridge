"""Replay Track A numerical evidence and permitted complete-generation outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import endpoint, qualify
from experiments.phase_iiil.analyze import _checked_measurement, _subgroup_outcomes
from experiments.phase_iiil.semantic_diagnostics import summarize as semantic_summary
from experiments.phase_iiim.analyze import _work_counts
from experiments.phase_iiin.analyze import _cluster_difference, _distribution, _verify_geometry
from experiments.phase_iiio.analyze import response_matrix, validate_native_targets
from experiments.phase_iiip.archive_trial import public_records
from experiments.phase_iiip.semantic_metrics import compare
from experiments.phase_iiip.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import compare_endpoint
from weightbridge.adaptation.factorial_geometry import compare_realized_norms, stored_update_cosines
from weightbridge.core import json_dump_atomic


def _hash(values):
    value = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(value.shape).encode("ascii") + b"\0" + value.tobytes()).hexdigest()


def _arrays(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if sum(member.file_size for member in archive.infolist()) > 32 * 1024 * 1024:
            raise ValueError("Numerical archive exceeds its expansion bound")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if not archive.files or set(archive.files) - {"targets", "preservation", "calibration"}:
            raise ValueError("Numerical archive contains unknown selected-block arrays")
        result = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    if any(value.ndim != 3 or not 0 <= len(value) <= 64 or not 1 <= value.shape[1] <= 1024
           or not 1 <= value.shape[2] <= 16 or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Numerical archive violates bounded finite selected-block dimensions")
    return result


def _same(recorded, expected, message, *, atol=1e-12):
    if recorded is None or not np.isfinite(recorded) or not np.isclose(recorded, expected, rtol=1e-10, atol=atol):
        raise ValueError(message)


def verify_block_bound(blocks, rhs, record):
    """Independent arithmetic; do not call the fitting solver's bound helper."""
    blocks, rhs = np.asarray(blocks, dtype=float), np.asarray(rhs, dtype=float)
    if record["input_sha256"] != {"blocks": _hash(blocks), "rhs": _hash(rhs)}:
        raise ValueError("Relaxed block bound input identity differs")
    certificate = record.get("certificate")
    if certificate is None:
        if (record["status"] != "INFEASIBLE_ZERO_BLOCK"
                or not np.any((np.max(np.abs(blocks), axis=(1, 2)) == 0) & (rhs > 0))):
            raise ValueError("Relaxation lacks a valid zero-block contradiction or numerical certificate")
        return {"zero_block_verified": True}
    dual = np.asarray(certificate["original_dual_multipliers"], dtype=float)
    if dual.shape != rhs.shape or not np.isfinite(dual).all() or np.any(dual < 0):
        raise ValueError("Relaxed bound requires original-row nonnegative finite multipliers")
    matrix = np.tensordot(dual, blocks, axes=(0, 0))
    norm, square = float(np.linalg.norm(matrix)), float(np.sum(matrix * matrix))
    numerator = float(dual @ rhs)
    eps, scale = np.finfo(float).eps, float(np.abs(rhs) @ dual)
    padding = float(64 * eps * (1 + scale + .5 * square))
    numerator_padding = float(64 * eps * (1 + scale))
    norm_padding = float(64 * eps * (1 + norm))
    objective = numerator - .5 * square
    objective_bound = float(np.sqrt(max(0., 2 * (objective - padding))))
    ratio_bound = float(max(0., numerator - numerator_padding) / (norm + norm_padding))
    spectral = float(np.linalg.svd(matrix, compute_uv=False)[0])
    spectral_padding = float(64 * eps * max(matrix.shape) * (1 + norm))
    rank_one_bound = float(max(0., numerator - numerator_padding) / (spectral + spectral_padding))
    expected = {"dual_objective": objective, "dual_roundoff_padding": padding,
        "dual_numerator": numerator, "dual_numerator_padding": numerator_padding,
        "dual_matrix_frobenius_norm": norm, "dual_matrix_norm_padding": norm_padding,
        "dual_objective_norm_lower_bound": objective_bound, "dual_cauchy_norm_lower_bound": ratio_bound,
        "certified_norm_lower_bound": max(objective_bound, ratio_bound),
        "dual_matrix_spectral_norm": spectral, "dual_spectral_padding": spectral_padding,
        "rank_one_norm_lower_bound": rank_one_bound}
    for key, value in expected.items():
        _same(certificate.get(key), value, "Relaxed bound does not follow independently from its original-row dual", atol=1e-14)
    if certificate.get("zero_spectral_denominator") is not (spectral == 0):
        raise ValueError("Spectral zero-denominator classification differs")
    _same(record["certified_norm_lower_bound"], expected["certified_norm_lower_bound"], "Top-level relaxed norm bound differs")
    witness = record.get("primal_witness")
    if witness:
        weights = np.asarray(witness["original_dual_multipliers"], dtype=float)
        if weights.shape != rhs.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Relaxed primal reconstruction has invalid multipliers")
        point = np.tensordot(weights, blocks, axes=(0, 0))
        slack = np.einsum("ijk,jk->i", blocks, point, optimize=True) - rhs
        if not np.allclose(slack, witness["constraint_slacks"], rtol=1e-10, atol=1e-10):
            raise ValueError("Relaxed primal slack does not reconstruct")
        _same(record["best_primal_norm"], float(np.linalg.norm(point)), "Relaxed primal norm differs")
        if record.get("minimum_norm") is not None:
            _same(record["minimum_norm"], float(np.linalg.norm(point)), "Relaxed approximate minimum differs")
    return {**expected, "independently_checked": True,
        "scope": "Numerical unrounded local-system bounds, not formal intervals or behavioral evidence."}


def verify_vector_bound(rows, rhs, record):
    """Check fixed/orientation-specific convex certificates in original coordinates."""
    rows, rhs = np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)
    if record["input_sha256"] != {"rows": _hash(rows), "rhs": _hash(rhs)}:
        raise ValueError("Rank-one inner certificate input identity differs")
    certificate = record.get("certificate")
    if certificate is None:
        # Degenerate cases retain their solver identity; a bounded search failure
        # supplies no invented positive norm bound.
        return {"independently_checked_dual": False, "status": record["status"]}
    dual = np.asarray(certificate["original_dual_multipliers"], dtype=float)
    if dual.shape != rhs.shape or not np.isfinite(dual).all() or np.any(dual < 0):
        raise ValueError("Inner certificate has invalid original-row multipliers")
    point = rows.T @ dual
    square, numerator = float(point @ point), float(rhs @ dual)
    eps, scale, norm = np.finfo(float).eps, float(np.abs(rhs) @ dual), float(np.linalg.norm(point))
    padding = float(64 * eps * (1 + scale + .5 * square))
    numerator_padding = float(64 * eps * (1 + scale))
    norm_padding = float(64 * eps * (1 + norm))
    objective = numerator - .5 * square
    lower = max(float(np.sqrt(max(0., 2 * (objective - padding)))),
                float(max(0., numerator - numerator_padding) / (norm + norm_padding)))
    _same(certificate.get("dual_objective"), objective, "Inner dual objective differs")
    _same(record["certified_norm_lower_bound"], lower, "Inner norm bound does not independently replay")
    witness = record.get("primal_witness")
    if witness:
        value = rows.T @ np.asarray(witness["original_dual_multipliers"])
        _same(record["best_primal_norm"], float(np.linalg.norm(value)), "Inner primal norm differs")
        if not np.allclose(rows @ value - rhs, witness["constraint_slacks"], rtol=1e-10, atol=1e-10):
            raise ValueError("Inner primal slack differs")
    return {"independently_checked_dual": True, "norm_lower_bound": lower}


def numerical_evidence(records, arrays, basis, config):
    if not arrays:
        return {}
    targets = arrays["targets"]
    margins = np.asarray([row["parent_margin"] for row in basis["forks"]])
    rhs = config["probe_design"]["target_margin"] - margins
    result = {}
    if "target-relaxation.json" in records:
        result["target_relaxation"] = verify_block_bound(targets, rhs, records["target-relaxation.json"])
    feasibility = records.get("feasibility.json")
    if not feasibility:
        return result
    if (not np.array_equal(feasibility["target_margins"], margins)
            or not np.array_equal(feasibility["target_rhs"], rhs)):
        raise ValueError("Target feasibility system differs from per-fork native evidence")
    protected = records["preservation-block-metadata.json"]["rows"]
    prior = np.asarray([row["parent_margin"] for row in protected])
    floor = records["floor.json"]["selected_floor"]
    combined = np.concatenate((targets, arrays["preservation"]))
    combined_rhs = np.concatenate((rhs, floor - prior))
    if (not np.array_equal(feasibility["preservation_margins"], prior)
            or feasibility["preservation_floor"] != floor or not np.array_equal(feasibility["all_rhs"], combined_rhs)):
        raise ValueError("Protected feasibility system differs from full-block parent evidence")
    initial = np.asarray(basis["right_factor"])
    for scope, blocks, changes in (("target_only", targets, rhs), ("with_preservation", combined, combined_rhs)):
        source = feasibility[scope]
        checked = verify_block_bound(blocks, changes, source["relaxation"])
        fixed = source["fixed_right"]
        fixed_checked = verify_vector_bound(blocks @ initial, changes, fixed)
        optimized = source["optimized_right"]
        best_right = optimized.get("best_right")
        if best_right is not None:
            right, left = np.asarray(best_right), np.asarray(optimized["best_left"])
            if (right.shape != (blocks.shape[2],) or left.shape != (blocks.shape[1],)
                    or not np.isclose(np.linalg.norm(right), 1., rtol=1e-10, atol=1e-10)):
                raise ValueError("Optimized diagnostic witness changes rank-one coordinates")
            verify_vector_bound(blocks @ right, changes, optimized["best_inner"])
            _same(optimized["best_norm"], float(np.linalg.norm(left)), "Optimized diagnostic norm differs")
            if np.min(np.einsum("ijk,j,k->i", blocks, left, right) - changes) < -1.1e-8:
                raise ValueError("Optimized diagnostic witness does not satisfy individual inequalities to solver tolerance")
        result[scope] = {"relaxed_status": source["relaxation"]["status"],
            "relaxed_minimum_norm": source["relaxation"].get("minimum_norm"),
            "relaxed_norm_lower_bound": checked.get("certified_norm_lower_bound"),
            "rank_one_norm_lower_bound": checked.get("rank_one_norm_lower_bound"),
            "fixed_status": fixed["status"], "fixed_minimum_norm": fixed.get("minimum_norm"),
            "fixed_norm_lower_bound": fixed.get("certified_norm_lower_bound"),
            "optimized_status": optimized["status"], "optimized_best_norm": optimized.get("best_norm"),
            "optimized_inner_calls": optimized.get("inner_calls"),
            "block_certificate": checked, "fixed_certificate": fixed_checked,
            "global_rank_one_minimum_claimed": False}
    return result


def _stored_state(record, basis, config):
    if "rounded_delta" not in record:
        return None
    left, right, delta = (np.asarray(record[name], dtype=float) for name in ("left_factor", "right_factor", "rounded_delta"))
    maximum = config["intent"]["limits"]["max_update_norm"]
    if (left.ndim != 1 or right.ndim != 1 or delta.shape != (len(left), len(right))
            or record["columns"] != basis["columns"] or len(right) != len(basis["columns"])
            or not all(np.isfinite(value).all() for value in (left, right, delta))
            or not np.isclose(np.linalg.norm(right), 1., rtol=1e-10, atol=1e-10)):
        raise ValueError("Stored state changes the authorized rank-one columns or finite dimensions")
    proposed, norm = np.outer(left, right), float(np.linalg.norm(delta))
    support = int(np.count_nonzero(delta))
    _same(record["proposed_norm"], float(np.linalg.norm(proposed)), "Proposed norm differs from factors")
    _same(record["realized_norm"], norm, "Stored norm differs from rounded tensor delta")
    if (not 0 < norm <= maximum or np.linalg.norm(proposed) > maximum
            or support != record["changed_entries"] or support > config["intent"]["limits"]["max_update_entries"]):
        raise ValueError("Stored state violates support or proposed/native norm authorization")
    _same(record["rounding_residual_norm"], float(np.linalg.norm(delta-proposed)), "BF16 rounding residual differs")
    singular = np.linalg.svd(delta, compute_uv=False)
    if not np.allclose(singular, record["stored_singular_values"], rtol=1e-10, atol=1e-12):
        raise ValueError("Stored delta singular values differ")
    rank = int(np.count_nonzero(singular > max(delta.shape)*np.finfo(float).eps*singular[0]))
    if rank != record["stored_numerical_rank"]:
        raise ValueError("Stored numerical rank differs; rounding is not extra optimized capacity")
    return delta


def _projected_basis(basis, blocks, right):
    return {**basis, "forks": [{**row, "projected_margin_gradient": gradient.tolist()}
        for row, gradient in zip(basis["forks"], blocks @ right, strict=True)]}


def verify_block_provenance(metadata, blocks, *, name, basis, checkpoint_sha256, tensor_name, source_tensor_sha256):
    """Bind each retained derivative to its original row and native context."""
    if (metadata["checkpoint_sha256"] != checkpoint_sha256 or metadata["tensor_name"] != tensor_name
            or metadata["source_tensor_sha256"] != source_tensor_sha256 or metadata["columns"] != basis["columns"]
            or metadata["block_shape"] != list(blocks.shape[1:]) or metadata["source_dtype"] != "BF16"
            or metadata.get("parent_restored_exactly") is not True
            or metadata["block_array_sha256"] != _hash(blocks) or len(metadata["rows"]) != len(blocks)):
        raise ValueError("Full-block provenance differs from the immutable native parent or retained arrays")
    if name == "targets":
        if (metadata.get("context_only_rows_have_supervision_weight") is not False
                or [row["prompt_id"] for row in metadata["rows"]] != [row["prompt_id"] for row in basis["forks"]]):
            raise ValueError("Target block rows or context-only supervision differ")
        projected = blocks @ np.asarray(basis["right_factor"])
    for index, (row, block) in enumerate(zip(metadata["rows"], blocks, strict=True)):
        if row["gradient_block_sha256"] != _hash(block):
            raise ValueError("Individual derivative provenance hash differs")
        context = metadata["packing"][row["original_context_index"]]
        matches = [i for i, prompt in enumerate(context["prompt_ids"]) if prompt == row["prompt_id"]
                   and (name == "targets" or context["positions"][i] == row["position"])]
        if len(matches) != 1:
            raise ValueError("Derivative row has no unique original native context")
        position = matches[0]
        prefix = context["native_prefix_ids"][position]
        prefix_hash = hashlib.sha256(json.dumps(prefix, separators=(",", ":")).encode("ascii")).hexdigest()
        if (row["native_prefix_ids"] != prefix or row["native_prefix_sha256"] != prefix_hash
                or row["actual_token_id"] != context["original_token_ids"][position]
                or row["replay_argmax_token_id"] != context["replay_token_ids"][position]
                or row["actual_token_id"] != row["replay_argmax_token_id"]):
            raise ValueError("Derivative native prefix or replay-token identity differs")
        if name == "targets":
            fork = basis["forks"][index]
            if (row["prompt_id"] not in context["selected_prompt_ids"]
                    or any(row[key] != fork[key] for key in ("parent_margin", "accepted_token_ids", "actual_token_id", "metadata"))
                    or not np.allclose(projected[index], fork["projected_margin_gradient"], rtol=1e-12, atol=1e-12)):
                raise ValueError("Individual target identity or inherited gradient projection differs")


def summarize(trial, material):
    raw = public_records(trial, material)
    records = {name: json.loads(value) for name, value in raw.items() if name.endswith(".json")}
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    report, assessment = records["fit-report.json"], records["assessment.json"]
    _verify_geometry(config, records)
    contract = endpoint(config)
    frozen = {name: load_split(material, name, manifest) for name in report["optimization_splits"]}
    parents = {name: _checked_measurement(records[f"parent-{name}.json"]["measurement"], rows, contract)
               for name, rows in frozen.items()}
    if qualify(parents["adaptation"], config) != report["qualification"]:
        raise ValueError("Parent admission differs from the frozen endpoint")
    for name, measured in parents.items():
        if records[f"parent-{name}.json"]["summary"] != semantic_summary(measured):
            raise ValueError("Parent complete-answer summary does not replay")
    arrays = _arrays(raw["target-blocks.npz"]) if "target-blocks.npz" in raw else {}
    if "derivative-blocks.npz" in raw:
        full = _arrays(raw["derivative-blocks.npz"])
        if arrays and not np.array_equal(arrays["targets"], full["targets"]):
            raise ValueError("Retained derivative targets differ from the original target-only blocks")
        arrays = full
    basis, selection = records.get("basis.json", {}), records.get("supervision-selection.json", {})
    if arrays:
        for name, metadata_name in (("targets", "target-block-metadata.json"), ("preservation", "preservation-block-metadata.json"), ("calibration", "calibration-block-metadata.json")):
            if name in arrays:
                metadata = records[metadata_name]
                verify_block_provenance(metadata, arrays[name], name=name, basis=basis,
                    checkpoint_sha256=config["checkpoint_sha256"], tensor_name=records["reference-geometry.json"]["tensor_name"],
                    source_tensor_sha256=records["target-block-metadata.json"]["source_tensor_sha256"])
        if "selected-blocks.json" in records:
            binding = records["selected-blocks.json"]
            if binding["file_sha256"] != hashlib.sha256(raw["derivative-blocks.npz"]).hexdigest():
                raise ValueError("Selected-block receipt does not bind its NPZ bytes")
            for name, array in arrays.items():
                expected = {"shape": list(array.shape), "dtype": "float64", "array_sha256": hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()}
                if binding["arrays"][name] != expected:
                    raise ValueError("Selected-block receipt array identity differs")
    numeric = numerical_evidence(records, arrays, basis, config)
    parent, families = parents["adaptation"], selection.get("family_ids", ())
    arms, deltas, generations, owners, work, decoded_deltas, decoded_states = {}, {}, {}, {}, {}, {}, {}
    for name in report["artifacts"]:
        if name.endswith(".json") and not name.startswith("arm-"):
            scopes = _work_counts(records[name])
            if scopes:
                work[name] = scopes
    for suffix in (1, 2):
        for label in "AC":
            name = f"{label}{suffix}"
            if f"arm-{name}.json" not in records:
                continue
            source = records[f"arm-{name}.json"]
            probe, numerical = source["probe"], source["numerical"]
            native, alias = probe.get("native_measurement"), source.get("duplicate_of")
            digest = source.get("tensor_sha256")
            delta = _stored_state(source, basis, config)
            if delta is not None:
                deltas[name] = delta
            left = np.asarray(source["left_factor"]) if "left_factor" in source else None
            right = np.asarray(source["right_factor"]) if "right_factor" in source else None
            local = _projected_basis(basis, arrays["targets"], right) if right is not None else basis
            row = {"arm": name, "method": "fixed_right" if label == "A" else "optimized_right",
                "cell": "minimum_witness" if suffix == 1 else "matched_stored_norm",
                "numerical_status": numerical["status"], "reason": probe["reason"],
                "realized_norm": source.get("realized_norm"), "proposed_norm": source.get("proposed_norm"),
                "changed_entries": source.get("changed_entries"), "tensor_sha256": digest,
                "duplicate_of": alias, "behavior_measured": "endpoint" in probe,
                "rounding_residual_norm": source.get("rounding_residual_norm"),
                "stored_numerical_rank": source.get("stored_numerical_rank"),
                "calibration_sufficient": numerical["status"] == "NATIVE_AND_CALIBRATION_PASS",
                "norm_match": source.get("norm_match", {}).get("status"), "numerical": numerical,
                "practice_gate": None, "fit_regressions": None, "fit_validity_losses": None}
            if native and left is not None:
                row["native_targets"] = validate_native_targets(native, local, left, config["probe_design"]["target_margin"])
            if alias:
                if alias not in generations or owners.get(digest) != alias:
                    raise ValueError("Alias lacks an earlier identical decoded checkpoint")
                for key in ("practice_gate", "subgroups", "distribution", "fit_regressions", "fit_validity_losses"):
                    row[key] = arms[alias][key]
                generations[name] = generations[alias]
            elif "endpoint" in probe:
                if (digest in owners or not row["calibration_sufficient"] or not native.get("passed")
                        or not row.get("native_targets", {}).get("passed") or not probe.get("restored_exactly")):
                    raise ValueError("Generation lacked a unique native-qualified restored tensor")
                protected = native["protected"]["rows"]
                if (len(protected) != config["probe_design"]["anchors"]
                        or any(row["margin"] <= native["preservation_floor"] or row["replay_argmax_token_id"] != row["parent_token_id"] for row in protected)):
                    raise ValueError("Decoded candidate lost a sampled protected native decision")
                measured = _checked_measurement(probe["endpoint"]["measurement"], frozen["adaptation"], contract)
                preserved = _checked_measurement(probe["endpoint"]["preservation_fit_measurement"], frozen["preservation-fit"], contract)
                gate = compare(parent, measured, min_comparison_gain=1, min_repeatable_corrections=1)
                fit_guard = compare_endpoint(parents["preservation-fit"], preserved, contract)
                valid = sum(item["valid"] for item in measured["rows"]) / len(measured["rows"])
                gate.update(passed=bool(gate["passed"] and not fit_guard["prohibited_regressions"]
                    and not fit_guard["lost_valid_presentations"] and valid >= config["endpoint"]["min_valid_rate"]),
                    target_item_gain=gate["target_gain"], prohibited_regressions=gate["regressions"], lost_valid_presentations=gate["lost_valid"])
                if gate != probe["endpoint"]["guard"] or fit_guard != probe["endpoint"]["preservation_fit_guard"]:
                    raise ValueError("Track A complete-generation practice gate does not independently replay")
                row.update(practice_gate=gate, subgroups=_subgroup_outcomes(parent, measured, gate),
                    distribution=_distribution(measured), fit_regressions=fit_guard["prohibited_regressions"], fit_validity_losses=fit_guard["lost_valid_presentations"])
                generations[name], owners[digest] = measured, name
                decoded_deltas[name] = delta
                decoded_states[name] = {"realized_norm": source["realized_norm"], "tensor_sha256": digest}
            row["response_matrix"] = response_matrix(parent, generations.get(name), local, left, native, families, probe.get("first_divergences", ())) if families else {}
            arms[name] = row
            scopes = {"numeric": {"resources": numerical.get("resources", {})}, "probe": {"resources": probe.get("resources", {})}}
            if "endpoint" in probe:
                scopes["generation"] = {key: {"resources": probe["endpoint"][key]["resources"]} for key in ("measurement", "preservation_fit_measurement")}
            work[f"arm-{name}.json"] = _work_counts(scopes)
    unique = len(owners)
    if unique != report["detail"].get("unique_complete_evaluations", 0) or unique > 4:
        raise ValueError("Decoded candidate count differs from frozen four-state budget")
    matched = {name: value for name, value in decoded_states.items() if name.endswith("2")}
    matching = compare_realized_norms(matched, intended_norm=config["probe_design"]["target_norms"][-1], relative_tolerance=config["probe_design"]["norm_relative_tolerance"]) if matched else None
    actual_pairwise = stored_update_cosines(decoded_deltas) if decoded_deltas else {}
    if "stored-pairwise.json" in records and records["stored-pairwise.json"] != {"actual_delta_cosines": actual_pairwise, "matched_pair": matching}:
        raise ValueError("Decoded stored-state comparison differs")
    development, development_generations = {}, {}
    before_dev = records.get("verification/development-transfer-base.json", {}).get("measurement")
    for name, gate in assessment["development"].items():
        worker = records.get(f"verification/development-transfer-{name}.json")
        if worker:
            measured = worker["measurement"]
            development_generations[name] = measured
            development[name] = {"gate": gate, "distribution": _distribution(measured), "subgroups": _subgroup_outcomes(before_dev, measured, gate)}
    contrasts = {}
    for population, observations, baseline in (("practice", generations, parent), ("development", development_generations, before_dev)):
        usable = (matching and matching["cohort_status"] == "MATCHED" and "A2" in observations and "C2" in observations
                  and arms["A2"]["tensor_sha256"] != arms["C2"]["tensor_sha256"] and baseline)
        contrasts[population] = _cluster_difference(baseline, observations["A2"], observations["C2"]) if usable else {
            "informative": False, "reason": "Requires distinct, matched actual stored states and both complete-generation observations."}
    totals = Counter()
    for scopes in work.values():
        for counts in scopes.values():
            totals.update(counts)
    independent = records.get("independent-outcome.json")
    return {"schema": "weightbridge.phase_iiip.analysis.v1", "track": "A",
        "fit_report_sha256": hashlib.sha256(raw["fit-report.json"]).hexdigest(),
        "assessment_sha256": hashlib.sha256(raw["assessment.json"]).hexdigest(),
        "disposition": independent["disposition"] if independent else assessment["disposition"],
        "reason": independent["reason"] if independent else assessment["reason"], "hidden_opened": bool(independent),
        "lineage_accepted": False, "parent_qualification": report["qualification"], "parent_diagnostics": _distribution(parent),
        "fit_detail": report["detail"], "basis": basis, "supervision_selection": selection, "numerical": numeric,
        "arms": arms, "unique_decoded_probes": unique, "stored_norm_matching": matching,
        "stored_update_comparison": actual_pairwise, "matched_generation_contrasts": contrasts,
        "development": development, "preservation": assessment["preservation"], "independent_selection": assessment["selection"], "independent": independent,
        "resources": {"fitting": report["resources"], "fitting_stage_counts": work, "fitting_work_totals": dict(sorted(totals.items())),
            "verification": assessment["worker_resources"], "verification_host": assessment["host_resources"]},
        "scope": "Track A only. Relaxed matrices are diagnostic; null behavior is unmeasured. Minimum-witness doses and matched-strength direction comparisons are separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    json_dump_atomic(args.out, summarize(args.trial, args.material))
