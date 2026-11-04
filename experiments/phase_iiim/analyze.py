"""Replay the factorial study and report only measured, matched causal contrasts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import endpoint, qualify
from experiments.phase_iiik.run_jspace import _active_measurements
from experiments.phase_iiil.analyze import _checked_measurement, _subgroup_outcomes
from experiments.phase_iiil.semantic_diagnostics import summarize as semantic_summary
from experiments.phase_iiim.archive_trial import public_records
from experiments.phase_iiim.semantic_metrics import compare
from experiments.phase_iiim.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import compare_endpoint
from weightbridge.adaptation.factorial_geometry import build_factorial_directions, compare_realized_norms, stored_update_cosines
from weightbridge.adaptation.preservation_geometry import DecisionAnchor
from weightbridge.adaptation.robust_jspace import estimate_window


def _family_changes(parent, candidate):
    prior, after = semantic_summary(parent), semantic_summary(candidate)
    return {
        family: sum(int(after["comparisons"][item]["all_correct"]) - int(prior["comparisons"][item]["all_correct"])
                    for item in (members["original"], members["counterfactual"]))
        for family, members in prior["families"].items()
    }


def _practice(record, parent, protected, frozen, contract, config, anchors, prompts):
    probe = record["probe"]
    scored = probe["endpoint"]
    measured = _checked_measurement(scored["measurement"], frozen["adaptation"], contract)
    measured_fit = _checked_measurement(scored["preservation_fit_measurement"], frozen["preservation-fit"], contract)
    semantic = compare(parent, measured, min_comparison_gain=1, min_repeatable_corrections=1)
    preservation = compare_endpoint(protected, measured_fit, contract)
    valid_rate = sum(row["valid"] for row in measured["rows"]) / len(measured["rows"])
    behavior = bool(semantic["passed"] and not preservation["prohibited_regressions"]
                    and not preservation["lost_valid_presentations"] and valid_rate >= config["endpoint"]["min_valid_rate"])
    active = _active_measurements(anchors, probe, prompts, record["geometry"],
                                  floor=record["floor"], allowance=record["direction_specific_allowance"])
    floor_pass = bool(active and all(row["decision_preserved"] and row["measured_margin"] > record["floor"]
                                    for row in active))
    gate = {**semantic, "passed": behavior and floor_pass, "semantic_behavior_passed": behavior,
            "strict_measured_floor_passed": floor_pass,
            "target_item_gain": semantic["target_gain"], "prohibited_regressions": semantic["regressions"],
            "lost_valid_presentations": semantic["lost_valid"]}
    if (not probe.get("restored_exactly") or scored["guard"] != gate or preservation != scored["preservation_fit_guard"]
            or active != record["active_measurements"]):
        raise ValueError("Practice gate does not replay from exact endpoint and active-margin evidence")
    after = semantic_summary(measured)
    return {
        "practice_gate": gate, "subgroups": _subgroup_outcomes(parent, measured, gate),
        "truth_flip_matrix": after["truth_flip_matrix"], "family_comparison_changes": _family_changes(parent, measured),
        "fit_regressions": preservation["prohibited_regressions"],
        "fit_validity_losses": preservation["lost_valid_presentations"],
        "minimum_measured_margin": min(row["measured_margin"] for row in active),
        "active_constraint_count": len(active),
        "first_divergences": probe.get("first_divergences"),
        "fork": probe.get("fork"), "fork_token_classes": scored.get("fork_token_classes"),
    }


def _stored_record(record, config):
    """Recompute selected slice norm, support, and alignment from archived deltas."""
    selected = record["norm_match"].get("selected")
    if selected is None:
        if record["norm_match"]["status"] != "UNMATCHED" or "rounded_delta" in record:
            raise ValueError("A missing numerical state cannot be called matched")
        return None
    delta = np.asarray(record["rounded_delta"], dtype=np.float64)
    left, right = np.asarray(record["left_factor"], dtype=np.float64), np.asarray(record["right_factor"], dtype=np.float64)
    if (delta.shape != (left.size, right.size) or len(record["columns"]) != right.size
            or not np.isfinite(delta).all() or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("Rounded update does not share the fixed rank-one tensor coordinates")
    norm, support = float(np.linalg.norm(delta)), int(np.count_nonzero(delta))
    if (not np.isclose(norm, selected["realized_norm"], rtol=0, atol=1e-10)
            or support != selected["changed_entries"] or support > config["intent"]["limits"]["max_update_entries"]
            or norm > config["intent"]["limits"]["max_update_norm"]):
        raise ValueError("Stored update support or realized norm differs from its numerical record")
    target, tolerance = record["intended_norm"], config["probe_design"]["norm_relative_tolerance"]
    error = abs(norm - target)
    matched = bool(selected["admissible"] and error / target <= tolerance)
    if (record["norm_match"]["status"] == "MATCHED") != matched:
        raise ValueError("Norm-match label differs from the measured intervention")
    proposed = np.outer(left, right)
    denominator = norm * float(np.linalg.norm(proposed))
    cosine = float(np.sum(delta * proposed) / denominator) if denominator else None
    if cosine is not None and not np.isclose(cosine, selected["stored_update_cosine"], atol=1e-10, rtol=0):
        raise ValueError("Stored-update alignment differs from the rounded delta")
    probe = record["probe"]
    if probe.get("endpoint") and (
            probe["candidate_tensor_sha256"] != selected["tensor_sha256"]
            or not np.isclose(probe["realized_update_norm"], norm, atol=1e-10, rtol=0)
            or probe["changed_entries"] != support):
        raise ValueError("Decoded checkpoint is different from the norm-matched tensor")
    return delta


def _numerical_rejections(record, config):
    """Separate a reachable stored norm from a refused preservation decision."""
    attempts = [attempt for stage in record["calibration_passes"] for attempt in stage["norm_match"]["attempts"]]
    tolerance = config["probe_design"]["norm_relative_tolerance"]
    maximum = config["intent"]["limits"]["max_update_norm"]
    closest = min((row for row in attempts if row.get("realized_norm") is not None and row.get("tensor_sha256")),
                  key=lambda row: row["norm_matching_error"], default=None)
    floor, failed_positions = record["floor"], {}
    margin_rejections = token_rejections = linear_rejections = 0
    for attempt in attempts:
        rows = (attempt.get("active_measurement") or {}).get("rows", ())
        failed = [row for row in rows if row["margin"] <= floor or row["replay_argmax_token_id"] != row["parent_token_id"]]
        margin_rejections += any(row["margin"] <= floor for row in rows)
        token_rejections += any(row["replay_argmax_token_id"] != row["parent_token_id"] for row in rows)
        linear_rejections += attempt.get("minimum_predicted_margin", float("inf")) < attempt.get("required_margin", 0) - 1e-7
        for row in failed:
            key = (row["prompt_id"], row["position"])
            entry = failed_positions.setdefault(key, {
                "prompt_id": row["prompt_id"], "position": row["position"], "group": row["group"], "wording": row["wording"],
                "refusal_count": 0, "minimum_measured_margin": row["margin"], "floor": floor})
            entry["refusal_count"] += 1
            entry["minimum_measured_margin"] = min(entry["minimum_measured_margin"], row["margin"])
    matching = [row for row in attempts if row.get("relative_norm_error") is not None
                and row["relative_norm_error"] <= tolerance and 0 < row["realized_norm"] <= maximum]
    return {
        "attempt_count": len(attempts),
        "unique_stored_states": len({row["tensor_sha256"] for row in attempts if row.get("tensor_sha256")}),
        "within_target_tolerance_before_preservation": len(matching),
        "stored_norm_reachable_in_measured_proposals": bool(matching),
        "admissible_attempts": sum(bool(row["admissible"]) for row in attempts),
        "measured_margin_refusal_attempts": margin_rejections, "greedy_token_refusal_attempts": token_rejections,
        "linear_constraint_refusal_attempts": linear_rejections,
        "failed_prefixes": list(failed_positions.values()),
        "closest_measured": {key: closest[key] for key in (
            "amplitude", "realized_norm", "norm_matching_error", "relative_norm_error", "tensor_sha256",
            "changed_entries", "feasible", "admissible")} if closest else None,
        "interpretation": "stored_norm_reachable_but_no_preservation_admissible_state"
            if matching and not any(row["admissible"] for row in attempts)
            else "see_measured_norm_preservation_and_calibration_separately",
    }


def _preferred_geometry(record, basis, directions, secants, anchors):
    """Combine measured raw-direction secants with individual target gradient slopes."""
    label = record["arm"][0]
    vector = np.asarray(directions["arms"][label]["aggregate"], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return {"available": False, "reason": "zero_raw_direction"}
    unit = vector / norm
    targets = basis["forks"]
    target_gradients = np.asarray([row["projected_margin_gradient"] for row in targets], dtype=np.float64)
    target_slopes = target_gradients @ unit
    rows = {(row["prompt_id"], row["position"]): row for row in secants[label]["rows"]}
    if any(anchor.key not in rows for anchor in anchors):
        return {"available": False, "reason": "selected_prefix_missing_from_preferred_direction_secant"}
    protected = [rows[anchor.key] for anchor in anchors]
    mean = np.asarray(basis["projected_gradients"]["MARGIN"], dtype=np.float64).mean(axis=0)
    alignment = float(mean @ unit / np.linalg.norm(mean)) if np.linalg.norm(mean) else None
    window = estimate_window(
        [row["parent_margin"] for row in targets], target_slopes,
        [row["parent_replay"]["margin"] for row in protected], [row["slope"] for row in protected],
        radius=record["intended_norm"], margin_floor=record["floor"],
        allowance=record["direction_specific_allowance"], target_items=[row["comparison_id"] for row in targets])
    return {
        "available": True, "kind": "raw_preferred_direction_gradient_and_measured_BF16_finite_secant",
        "target_gradient_alignment": alignment, "positive_target_fork_slopes": int(np.count_nonzero(target_slopes > 0)),
        "target_fork_count": len(targets),
        "target_forks": [{"prompt_id": row["prompt_id"], "comparison_id": row["comparison_id"],
                         "parent_margin": row["parent_margin"], "slope": float(slope)}
                        for row, slope in zip(targets, target_slopes, strict=True)],
        "protected_scan_positions": len(rows), "adverse_protected_scan_positions": sum(row["slope"] < 0 for row in rows.values()),
        "sampled_adverse_positions": sum(row["slope"] < 0 for row in protected),
        "local_feasible_region": window,
        "assumptions": [
            "Target slopes are local projected VJPs along the unconstrained normalized arm direction.",
            "Protected slopes are measured finite BF16 secants on complete parent trajectories at the frozen scan radius.",
            "The scalar radius uses the intended stored norm as a nominal left-factor bound.",
            "The constrained candidate may follow a different direction, and this diagnostic does not authorize application.",
            "A missing local witness does not establish global or nonlinear infeasibility.",
        ],
    }


def _factorial_effects(arms, dose, norms, *, gate_key="practice_gate",
                      scope="Frozen practice factorial. Independent generalization is reported by separate gates."):
    names = [f"{arm}{dose}" for arm in "ABCD"]
    ready = all(name in arms and gate_key in arms[name] for name in names)
    identities = [arms.get(name, {}).get("tensor_sha256") for name in names]
    distinct = all(identities) and len(set(identities)) == 4
    matched = bool(norms and norms["cohort_status"] == "MATCHED")
    if not (ready and distinct and matched):
        return {"informative": False, "all_decoded": ready, "four_distinct_states": bool(distinct),
                "cohort_matched": matched, "reason": "incomplete_unmatched_or_aliased_factorial"}
    coefficients = {"objective": [-.5, -.5, .5, .5], "normalization": [-.5, .5, -.5, .5],
                    "interaction": [1., -1., -1., 1.]}
    metrics = {
        "strict_comparison_gain": [arms[name][gate_key]["target_gain"] for name in names],
        "local_response_corrections": [len(arms[name][gate_key]["local_corrected_target_presentations"]) for name in names],
        "complete_family_gain": [arms[name][gate_key]["complete_family_gain"] for name in names],
        "prohibited_regressions": [len(arms[name][gate_key]["regressions"]) + len(arms[name]["fit_regressions"])
                                  for name in names],
    }
    effects = {metric: {effect: float(np.asarray(weights) @ values) for effect, weights in coefficients.items()}
               for metric, values in metrics.items()}
    families = list(arms[names[0]]["family_comparison_changes"])
    changes = np.array([[arms[name]["family_comparison_changes"][family] for family in families] for name in names])
    rng = np.random.default_rng(20261103)
    indices = rng.integers(0, len(families), size=(4000, len(families)))
    intervals = {}
    for effect, weights in coefficients.items():
        family_effects = np.asarray(weights) @ changes / 2
        samples = family_effects[indices].mean(axis=1)
        intervals[effect] = {"estimate": float(family_effects.mean()),
                             "low": float(np.quantile(samples, .025)), "high": float(np.quantile(samples, .975)),
                             "clusters": len(families), "resamples": 4000,
                             "estimand": "paired strict-comparison accuracy effect"}
    return {"informative": True, "all_decoded": True, "four_distinct_states": True,
            "cohort_matched": True, "effects_on_counts": effects, "family_cluster_intervals": intervals,
            "scope": scope}


def _pair_contrasts(arms, dose, norms):
    result = {}
    if norms is None:
        return result
    for first, second, factor in (("A", "B", "normalization_at_CE"), ("C", "D", "normalization_at_MARGIN"),
                                  ("A", "C", "objective_at_RAW"), ("B", "D", "objective_at_UNIT")):
        a, b = f"{first}{dose}", f"{second}{dose}"
        match = norms["pairs"].get(f"{a}:{b}")
        informative = bool(match and match["matched"] and not match["same_stored_tensor"]
                           and "practice_gate" in arms.get(a, {}) and "practice_gate" in arms.get(b, {}))
        row = {"informative": informative, "first": a, "second": b, "norm_match": match}
        if informative:
            row["strict_gain_difference"] = arms[b]["practice_gate"]["target_gain"] - arms[a]["practice_gate"]["target_gain"]
            row["complete_family_gain_difference"] = arms[b]["practice_gate"]["complete_family_gain"] - arms[a]["practice_gate"]["complete_family_gain"]
            row["prohibited_regression_difference"] = (
                len(arms[b]["practice_gate"]["regressions"]) + len(arms[b]["fit_regressions"])
                - len(arms[a]["practice_gate"]["regressions"]) - len(arms[a]["fit_regressions"]))
        result[factor] = row
    return result


def _behavior_agreement(first, second):
    """Compare generated behavior separately from numerical direction agreement."""
    before, after = first["rows"], second["rows"]
    if [row["id"] for row in before] != [row["id"] for row in after]:
        raise ValueError("Behavioral agreement requires the same frozen presentations")
    both_valid = [(a, b) for a, b in zip(before, after, strict=True) if a["valid"] and b["valid"]]
    return {
        "presentations": len(before), "both_valid": len(both_valid),
        "same_valid_semantic_answer": sum(a["exact"] == b["exact"] for a, b in both_valid),
        "same_correctness": sum(a["correct"] == b["correct"] for a, b in zip(before, after, strict=True)),
        "same_generated_token_sequence": sum(a["generated_token_ids"] == b["generated_token_ids"]
                                             for a, b in zip(before, after, strict=True)),
        "different_correctness_ids": [a["id"] for a, b in zip(before, after, strict=True) if a["correct"] != b["correct"]],
    }


def _termination_diagnostics(record, basis, parent_classes, diagnosis):
    """Join recorded-parent-prefix decisions to complete candidate generations."""
    probe = record["probe"]
    endpoint_rows = {row["id"]: row for row in probe["endpoint"]["measurement"]["rows"]}
    meanings = semantic_summary(probe["endpoint"]["measurement"])
    originals = {row["prompt_id"]: row for row in diagnosis["forks"]}
    initial = {row["prompt_id"]: row for row in parent_classes["rows"]}
    after = {row["prompt_id"]: row for row in probe["endpoint"]["fork_token_classes"]["rows"]}
    tokens = parent_classes["numeric_surface_tokenizations"]
    digits = {sequence[0] for text, sequence in tokens.items() if text.isdigit() and len(sequence) == 1}
    newline = set(parent_classes["newline_tokenization"])
    eos = set()
    for fork in diagnosis["forks"]:
        token = fork["decoding"]["eos_token_id"]
        eos.update(token if isinstance(token, list) else (token,))

    def kind(token):
        return "eos" if token in eos else "newline" if token in newline else "digit" if token in digits else "other"

    predicted = probe["fork"]["candidate_predicted_ids"]
    if len(predicted) != len(basis["forks"]):
        raise ValueError("Fork decision order differs from the frozen basis")
    rows, counts = [], {name: {"accepted_changes": 0, "completed_correct": 0, "visited_parent_prefix": 0}
                        for name in ("digit", "newline", "eos", "other")}
    for fork, token in zip(basis["forks"], predicted, strict=True):
        prompt_id = fork["prompt_id"]
        original, before, current, output = originals[prompt_id], initial[prompt_id], after[prompt_id], endpoint_rows[prompt_id]
        if (before["accepted_token_ids"] != fork["accepted_token_ids"]
                or current["accepted_token_ids"] != fork["accepted_token_ids"]):
            raise ValueError("Token-class probabilities do not share the diagnosed accepted continuations")
        accepted = token in fork["accepted_token_ids"]
        changed = token != fork["actual_token_id"]
        category = kind(token)
        shared = original["shared_token_ids"]
        generated = output["generated_token_ids"]
        visited = bool(len(generated) > len(shared) and generated[:len(shared)] == shared)
        family = meanings["families"][fork["metadata"]["pair_id"]]
        counterpart = family["counterfactual"] if fork["comparison_id"] == family["original"] else family["original"]
        if accepted and changed:
            counts[category]["accepted_changes"] += 1
            counts[category]["completed_correct"] += bool(output["correct"])
            counts[category]["visited_parent_prefix"] += visited
        rows.append({
            "prompt_id": prompt_id, "comparison_id": fork["comparison_id"], "pair_id": fork["metadata"]["pair_id"],
            "parent_token_id": fork["actual_token_id"], "parent_kind": kind(fork["actual_token_id"]),
            "candidate_prefix_token_id": token, "candidate_prefix_kind": category,
            "accepted_prefix_token": accepted, "changed_prefix_token": changed,
            "full_generation_visited_parent_prefix": visited,
            "full_generation_token_at_original_fork": generated[len(shared)] if visited else None,
            "completed_correct": bool(output["correct"]),
            "counterfactual_comparison_correct": bool(meanings["comparisons"][counterpart]["all_correct"]),
            "complete_truth_flip_family": family["status"] == "both_correct",
            "probability_changes": {name: current[name] - before[name] for name in (
                "accepted_probability", "numeric_subset_probability", "termination_subset_probability")},
            "parent_probabilities": {name: before[name] for name in (
                "accepted_probability", "numeric_subset_probability", "termination_subset_probability")},
            "candidate_probabilities": {name: current[name] for name in (
                "accepted_probability", "numeric_subset_probability", "termination_subset_probability")},
            "prefix_replay_argmax_agreement": current["greedy_token_id"] == token,
        })
    first = probe.get("first_divergences", ())
    return {
        "accepted_changed_prefix_tokens_by_kind": counts, "fork_rows": rows,
        "first_divergence_parent_phases": dict(Counter(row.get("parent_phase", "unclassified") for row in first)),
        "first_divergence_candidate_token_kinds": dict(Counter(kind(row.get("candidate_token_id")) for row in first)),
        "mean_probability_changes": {name: float(np.mean([row["probability_changes"][name] for row in rows]))
                                     for name in ("accepted_probability", "numeric_subset_probability", "termination_subset_probability")},
        "scope": (
            "Prefix decisions and probability changes use the recorded parent fork. A candidate can visit another path. "
            "The numeric subset consists of native standalone digit and decimal-point tokens, "
            "and the termination subset consists of native newline/EOS. These subsets do not exhaust all number continuations."),
    }


def _work_counts(value, path=""):
    counts = {}
    keys = {"forward_calls", "backward_calls", "batched_decode_steps", "generate_batches", "generated_examples"}
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{path}/{key}" if path else key
            if key == "resources":
                work = {name: number for name, number in child.items() if name in keys}
                if work:
                    counts[location] = work
            elif isinstance(child, (dict, list)):
                counts.update(_work_counts(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            counts.update(_work_counts(child, f"{path}/{index}"))
    return counts


def summarize(trial, material):
    raw = public_records(trial, material)
    records = {name: json.loads(value) for name, value in raw.items()}
    report, assessment = records["fit-report.json"], records["assessment.json"]
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    contract = endpoint(config)
    frozen = {name: load_split(material, name, manifest) for name in report["optimization_splits"]}
    parents = {name: _checked_measurement(records[f"parent-{name}.json"]["measurement"], rows, contract)
               for name, rows in frozen.items()}
    for name, measured in parents.items():
        if semantic_summary(measured) != records[f"parent-{name}.json"]["summary"]:
            raise ValueError("Parent semantic summary differs from completed generation")
    parent, protected = parents["adaptation"], parents["preservation-fit"]
    if qualify(parent, config) != report["qualification"]:
        raise ValueError("Frozen parent qualification differs from measured responses")
    basis = records.get("basis.json")
    if records.get("directions.json"):
        calculated = build_factorial_directions(basis["item_ids"], basis["item_groups"],
            basis["projected_gradients"]["CE"], basis["projected_gradients"]["MARGIN"]).record()
        if calculated != records["directions.json"]:
            raise ValueError("Factorial directions differ from their measured comparison gradients")
    sampling = records.get("sampling.json", {})
    anchors = tuple(DecisionAnchor(**row) for row in sampling.get("anchors", ()))
    prompts = records.get("probe-reference.json", {}).get("prompt_ids", ())
    arms, owners, deltas, generations = {}, {}, {}, {}
    stage_work = {}
    for filename in report["artifacts"]:
        if not filename.startswith("arm-"):
            measured_work = _work_counts(records[filename])
            if measured_work:
                stage_work[filename] = measured_work
    for dose in range(1, len(config["probe_design"]["target_norms"]) + 1):
        for label in "ABCD":
            name = f"{label}{dose}"
            record = records.get(f"arm-{name}.json")
            if record is None:
                continue
            probe, alias = record["probe"], record.get("duplicate_of")
            selected = record["norm_match"].get("selected")
            digest = selected["tensor_sha256"] if selected else None
            delta = _stored_record(record, config)
            if delta is not None:
                if record["columns"] != basis["columns"] or record["right_factor"] != basis["right_factor"]:
                    raise ValueError("Factorial arms do not share authorized columns and right factor")
                deltas[name] = delta
            row = {
                "arm": name, "objective": "CE" if label in "AB" else "MARGIN",
                "normalization": "RAW" if label in "AC" else "UNIT",
                "intended_norm": record["intended_norm"], "norm_match": record["norm_match"],
                "reason": probe["reason"], "duplicate_of": alias, "tensor_sha256": digest,
                "realized_norm": selected["realized_norm"] if selected else None,
                "changed_entries": selected["changed_entries"] if selected else None,
                "calibration_sufficient": record["calibration_sufficient"],
                "direction_specific_allowance": record["direction_specific_allowance"],
                "floor": record["floor"], "restored_exactly": probe.get("restored_exactly"),
                "local_feasibility": record.get("geometry", {}).get("local_feasibility"),
                "numeric_resources": record["resources"], "probe_resources": probe.get("resources"),
                "numerical_rejections": _numerical_rejections(record, config),
                "raw_preferred_geometry": _preferred_geometry(record, basis, records["directions.json"],
                                                               records["secants.json"], anchors),
            }
            if alias:
                if alias not in arms or owners.get(digest) != alias:
                    raise ValueError("Candidate alias does not bind the identical stored state")
                for key in ("practice_gate", "subgroups", "truth_flip_matrix", "family_comparison_changes",
                            "fit_regressions", "fit_validity_losses", "minimum_measured_margin", "fork", "fork_token_classes",
                            "termination_diagnostics"):
                    row[key] = arms[alias][key]
                generations[name] = generations[alias]
            elif "endpoint" in probe:
                if (not digest or digest in owners or record["norm_match"]["status"] != "MATCHED"
                        or not record["calibration_sufficient"]):
                    raise ValueError("Decoded candidate is duplicated, unmatched, or lacks directional calibration")
                owners[digest] = name
                row.update(_practice(record, parent, protected, frozen, contract, config, anchors, prompts))
                row["termination_diagnostics"] = _termination_diagnostics(
                    record, basis, records["token-class-parent.json"], records["fork-diagnosis.json"])
                generations[name] = probe["endpoint"]["measurement"]
            arms[name] = row
            # The numeric cell counts its cached prefix probes exactly once.
            # Candidate diagnostics and actual generation have separate counters.
            work = {"numeric": {"resources": record["resources"]}}
            if probe.get("resources"):
                work["probe"] = {"resources": probe["resources"]}
            if "endpoint" in probe:
                work["generation"] = {"adaptation": {"resources": probe["endpoint"]["measurement"]["resources"]},
                    "preservation_fit": {"resources": probe["endpoint"]["preservation_fit_measurement"]["resources"]},
                    "fork_token_classes": {"resources": probe["endpoint"]["fork_token_classes"]["resources"]}}
            stage_work[f"arm-{name}.json"] = _work_counts(work)
    unique_decodes = sum("practice_gate" in row and not row["duplicate_of"] for row in arms.values())
    if unique_decodes != report["detail"].get("unique_complete_evaluations", 0):
        raise ValueError("Decoded candidates differ from the frozen practice budget")
    pairwise, effects, pair_contrasts = {"norm_matching": {}, "actual_delta_cosines": stored_update_cosines(deltas) if deltas else {}}, {}, {}
    for dose, target in enumerate(config["probe_design"]["target_norms"], 1):
        selected = {name: {"realized_norm": row["realized_norm"], "tensor_sha256": row["tensor_sha256"]}
                    for name, row in arms.items() if name.endswith(str(dose)) and row["realized_norm"] is not None}
        comparison = compare_realized_norms(selected, intended_norm=target,
            relative_tolerance=config["probe_design"]["norm_relative_tolerance"]) if selected else None
        pairwise["norm_matching"][str(dose)] = comparison
        effects[str(dose)] = _factorial_effects(arms, dose, comparison)
        pair_contrasts[str(dose)] = _pair_contrasts(arms, dose, comparison)
    if records.get("stored-pairwise.json", pairwise) != pairwise:
        raise ValueError("Stored pairwise intervention differences do not reconstruct from archived deltas")
    behavior_pairwise = {}
    names = tuple(generations)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            behavior_pairwise[f"{first}:{second}"] = _behavior_agreement(generations[first], generations[second])
    development_arms, development_subgroups = {}, {}
    development_parent = records.get("verification/development-transfer-base.json")
    if development_parent is not None:
        before = development_parent["measurement"]
        for name, gate in assessment["development"].items():
            observed = records.get(f"verification/development-transfer-{name}.json")
            if observed is None:
                continue
            measured = observed["measurement"]
            development_arms[name] = {
                "tensor_sha256": arms[name]["tensor_sha256"], "development_gate": gate,
                "fit_regressions": [], "family_comparison_changes": _family_changes(before, measured)}
            development_subgroups[name] = _subgroup_outcomes(before, measured, gate)
    development_effects = {
        str(dose): _factorial_effects(development_arms, dose, pairwise["norm_matching"][str(dose)],
            gate_key="development_gate",
            scope="Frozen development-transfer factorial. This population can rank finalists and is separate from independent selection.")
        for dose in range(1, len(config["probe_design"]["target_norms"]) + 1)}
    total_work = Counter()
    for scopes in stage_work.values():
        for values in scopes.values():
            total_work.update(values)
    independent = records.get("independent-outcome.json")
    baseline = semantic_summary(parent, fork_tokens={
        row["prompt_id"]: row["fork_tokens"] for row in records.get("fork-diagnosis.json", {}).get("forks", ())
        if row.get("fork_tokens")})
    return {
        "schema": "weightbridge.phase_iiim.analysis.v1",
        "fit_report_sha256": hashlib.sha256(raw["fit-report.json"]).hexdigest(),
        "assessment_sha256": hashlib.sha256(raw["assessment.json"]).hexdigest(),
        "disposition": independent["disposition"] if independent else assessment["disposition"],
        "reason": independent["reason"] if independent else assessment["reason"],
        "hidden_opened": independent.get("hidden_opened") if independent else False,
        "lineage_accepted": False, "winner": assessment["winner"], "parent_qualification": report["qualification"],
        "parent_diagnostics": baseline, "fit_detail": report["detail"], "basis": records.get("basis.json"),
        "directions": records.get("directions.json"), "sampling": sampling, "arms": arms,
        "unique_decoded_probes": unique_decodes,
        "practice_passers": [name for name, row in arms.items() if row.get("practice_gate", {}).get("passed")
                             and not row["duplicate_of"]],
        "stored_pairwise": pairwise, "behavior_pairwise": behavior_pairwise,
        "factorial_effects": effects, "pair_contrasts": pair_contrasts,
        "development_factorial_effects": development_effects, "development_subgroups": development_subgroups,
        "development": assessment["development"], "preservation": assessment["preservation"],
        "selection": assessment["selection"], "independent": independent,
        "resources": {"fitting": report["resources"], "fitting_stage_counts": stage_work,
                      "fitting_work_totals": dict(sorted(total_work.items())),
                      "verification": assessment["worker_resources"], "verification_host": assessment["host_resources"]},
    }


def plot(summary, out):
    """Plot recorded directions, complete generation, and actual intervention sizes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if not summary["directions"] and not summary["arms"]:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    agreement, behavior, dose, subgroup = axes.ravel()
    names = list("ABCD")
    if summary["directions"]:
        values = np.eye(4)
        for pair, record in summary["directions"]["angles"].items():
            first, second = pair.split(":")
            values[names.index(first), names.index(second)] = record["cosine"] if record["cosine"] is not None else np.nan
            values[names.index(second), names.index(first)] = values[names.index(first), names.index(second)]
        image = agreement.imshow(values, vmin=-1, vmax=1, cmap="RdBu")
        labels = ["A CE RAW", "B CE UNIT", "C MARGIN RAW", "D MARGIN UNIT"]
        agreement.set_xticks(range(4), labels, rotation=22, ha="right")
        agreement.set_yticks(range(4), labels)
        for i in range(4):
            for j in range(4):
                value = values[i, j]
                text = f"{value:.4f}" if i != j and abs(value) > .995 else f"{value:.2f}"
                agreement.text(j, i, text if np.isfinite(value) else "undefined",
                               ha="center", va="center", color="white" if abs(value) > .65 else "black")
        fig.colorbar(image, ax=agreement, fraction=.045)
        agreement.set_title("Unconstrained factorial direction cosine")
    else:
        agreement.set_axis_off()
    measured = [(name, row) for name, row in summary["arms"].items() if "practice_gate" in row]
    if measured:
        labels = [name + (" alias" if row["duplicate_of"] else "") for name, row in measured]
        x = np.arange(len(measured))
        behavior.bar(x - .2, [row["practice_gate"]["target_gain"] for _, row in measured], width=.38,
                     color="#35724c", label="Strict comparison gain")
        behavior.bar(x + .2, [len(row["practice_gate"]["regressions"]) + len(row["fit_regressions"])
                              for _, row in measured], width=.38, color="#ab563f", label="Correct responses lost")
        behavior.set_xticks(x, labels, rotation=20)
        behavior.yaxis.set_major_locator(MaxNLocator(integer=True))
        behavior.axhline(0, color="#777777", linewidth=.7)
        behavior.set_title("Complete frozen practice generation")
        excluded = [name for name, row in summary["arms"].items() if "practice_gate" not in row]
        if excluded:
            behavior.text(.03, .95, "No generation probe for " + ", ".join(excluded),
                          transform=behavior.transAxes, va="top", fontsize=9)
        behavior.legend(loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False)
        groups = list(measured[0][1]["subgroups"])
        changes = [[row["subgroups"][group]["candidate_correct_presentations"]
                    - row["subgroups"][group]["parent_correct_presentations"] for group in groups]
                   for _, row in measured]
        span = max(1, max(abs(value) for row in changes for value in row))
        subgroup.imshow(changes, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")
        subgroup.set_xticks(range(len(groups)), [name.removeprefix("negative-").replace("/", "\n") for name in groups])
        subgroup.set_yticks(range(len(labels)), labels)
        for i, values in enumerate(changes):
            for j, value in enumerate(values):
                subgroup.text(j, i, f"{value:+d}" if value else "0", ha="center", va="center",
                              color="white" if abs(value) > .55 * span else "black")
        subgroup.set_title("Net correct presentations by semantic group")
    else:
        behavior.text(.5, .5, "No complete candidate practice decoding", ha="center", va="center")
        behavior.set_axis_off()
        subgroup.set_axis_off()
    numeric = [(name, row) for name, row in summary["arms"].items()
               if row["realized_norm"] is not None or row["numerical_rejections"]["closest_measured"] is not None]
    if numeric:
        x = np.arange(len(numeric))
        dose.plot(x, [row["intended_norm"] for _, row in numeric], linestyle="none", marker="_", markersize=19,
                  color="black", label="Intended stored norm")
        for index, (_, row) in enumerate(numeric):
            matched = row["norm_match"]["status"] == "MATCHED"
            actual = row["realized_norm"] if row["realized_norm"] is not None else row["numerical_rejections"]["closest_measured"]["realized_norm"]
            dose.scatter(index, actual, color="#3277a8" if matched else "#b47438",
                         marker="o" if matched else "x")
        dose.set_xticks(x, [name for name, _ in numeric], rotation=20)
        dose.set_ylabel("Actual BF16 update norm")
        dose.set_title("Measured norms and separate admissibility")
        dose.legend(loc="upper left", frameon=False)
    else:
        dose.set_axis_off()
    fig.savefig(out, dpi=165, metadata={"Description": "Measured Phase III-M gradients, norms, and complete decoding."})
    plt.close(fig)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("trial", "material"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = summarize(args.trial, args.material)
    if args.plot:
        plot(result, args.plot)
    print(json.dumps(result, indent=2))
