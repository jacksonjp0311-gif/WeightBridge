"""Replay numerical witnesses, native fork checks and complete joint outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import endpoint, qualify
from experiments.phase_iiil.analyze import _checked_measurement, _subgroup_outcomes
from experiments.phase_iiil.semantic_diagnostics import summarize as semantic_summary
from experiments.phase_iiim.analyze import _stored_record, _work_counts
from experiments.phase_iiin.analyze import _cluster_difference, _distribution, _family_changes, _verify_geometry
from experiments.phase_iiio.archive_trial import public_records
from experiments.phase_iiio.semantic_metrics import compare
from experiments.phase_iiio.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import compare_endpoint
from weightbridge.adaptation.factorial_geometry import compare_realized_norms, stored_update_cosines
from weightbridge.adaptation.joint_feasibility import solve_joint_feasibility
from weightbridge.core import json_dump_atomic


def _without_resources(value):
    if isinstance(value, dict):
        return {key: _without_resources(child) for key, child in value.items() if key != "resources"}
    if isinstance(value, list):
        return [_without_resources(child) for child in value]
    return value


def verify_joint_systems(joint):
    """Replay bound inputs and independently recompute their nonnegative duals."""
    for key, system in joint.get("systems", {}).items():
        if key.startswith("family/"):
            family, formulation = key.removeprefix("family/").rsplit("/", 1)
            recorded = joint["per_family"][family][formulation]
        else:
            recorded = joint[key]
        replay = solve_joint_feasibility(**system).record()
        if _without_resources(replay) != _without_resources(recorded):
            raise ValueError("Joint local system does not replay from its archived individual rows")
        certificate = recorded.get("certificate")
        if certificate is None:
            continue
        target = np.asarray(system["target_rows"], dtype=float)
        protected = np.asarray(system["preservation_rows"], dtype=float).reshape(-1, target.shape[1])
        rows = np.vstack((target, protected))
        prior = np.concatenate((system["target_parent_margins"], system["preservation_parent_margins"]))
        target_floors = np.broadcast_to(system["target_floor"], len(target))
        protected_floors = np.broadcast_to(system["preservation_floor"], len(protected))
        rhs = np.concatenate((target_floors, protected_floors)) - prior
        dual = np.asarray(certificate["original_dual_multipliers"])
        if dual.shape != rhs.shape or not np.isfinite(dual).all() or np.any(dual < 0):
            raise ValueError("Joint numerical lower bound lacks a nonnegative finite dual witness")
        point = rows.T @ dual
        square = float(point @ point)
        objective = float(rhs @ dual - .5 * square)
        padding = float(64 * np.finfo(float).eps * (1 + np.abs(rhs) @ dual + .5 * square))
        objective_lower = float(np.sqrt(max(0., 2 * (objective - padding))))
        numerator = float(rhs @ dual)
        numerator_padding = float(64 * np.finfo(float).eps * (1 + np.abs(rhs) @ dual))
        denominator = float(np.linalg.norm(point))
        denominator_padding = float(64 * np.finfo(float).eps * (1 + denominator))
        directional_lower = float(max(0., numerator - numerator_padding) / (denominator + denominator_padding))
        lower = max(objective_lower, directional_lower)
        components = {
            "dual_objective": objective, "dual_objective_norm_lower_bound": objective_lower,
            "dual_direction_norm_lower_bound": directional_lower,
            "dual_direction_numerator": numerator, "dual_direction_denominator": denominator,
            "certified_norm_lower_bound": lower,
        }
        if (not np.isclose(objective, certificate["dual_objective"], rtol=1e-10, atol=1e-12)
                or not np.isclose(lower, recorded["certified_norm_lower_bound"], rtol=1e-10, atol=1e-12)
                or any(not np.isclose(value, certificate[name], rtol=1e-10, atol=1e-12)
                       for name, value in components.items())
                or not np.isclose(padding, certificate["dual_roundoff_padding"], rtol=1e-10, atol=1e-18)
                or not np.isclose(numerator_padding, certificate["dual_direction_roundoff_padding"]["numerator"],
                                  rtol=1e-10, atol=1e-18)
                or not np.isclose(denominator_padding, certificate["dual_direction_roundoff_padding"]["denominator"],
                                  rtol=1e-10, atol=1e-18)):
            raise ValueError("Joint numerical norm lower bound does not follow from the recorded dual")


def validate_native_targets(native, basis, left, target_floor):
    forks = basis["forks"]
    rows = native["targets"]["rows"]
    lookup = {row["prompt_id"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != {row["prompt_id"] for row in forks}:
        raise ValueError("Native fork measurement omitted or duplicated an individual target")
    predictions, measured = {}, {}
    for fork in forks:
        name, current = fork["prompt_id"], lookup[fork["prompt_id"]]
        prediction = float(fork["parent_margin"] + np.asarray(fork["projected_margin_gradient"]) @ left)
        value = current["measured_margin"]
        if (not np.isfinite(value) or current["accepted_token_ids"] != fork["accepted_token_ids"]
                or current["accepted_token_selected"] != (current["greedy_token_id"] in fork["accepted_token_ids"])):
            raise ValueError("Native target token identity or acceptance flag differs")
        predictions[name], measured[name] = prediction, value
    passed = bool(all(value >= target_floor for value in predictions.values())
                  and all(row["measured_margin"] >= target_floor and row["accepted_token_selected"] for row in rows))
    return {"passed": passed, "predicted_margins": predictions, "measured_margins": measured,
            "minimum_predicted_margin": min(predictions.values()), "minimum_measured_margin": min(measured.values()),
            "maximum_absolute_prediction_error": max(abs(predictions[name] - measured[name]) for name in predictions),
            "accepted_native_forks": sum(row["accepted_token_selected"] for row in rows), "required_forks": len(forks)}


def response_matrix(parent, candidate, basis, left, native, selected_families, first_divergences=()):
    """All eight presentations remain visible even when no candidate was decoded."""
    parent_rows = {row["id"]: row for row in parent["rows"]}
    after = {row["id"]: row for row in candidate["rows"]} if candidate else {}
    forks = {row["prompt_id"]: row for row in basis.get("forks", ())}
    native_rows = {row["prompt_id"]: row for row in native.get("targets", {}).get("rows", ())} if native else {}
    divergences = {row["id"]: row for row in first_divergences if "id" in row}
    result = {}
    for family in selected_families:
        rows = [row for row in parent_rows.values() if row["group"] == "target" and row["pair_id"] == family]
        if len(rows) != 8 or len({(row["counterfactual"], row["order"], row["wording"]) for row in rows}) != 8:
            raise ValueError("A selected semantic family does not contain all eight required outcomes")
        outputs = []
        for row in sorted(rows, key=lambda r: (r["counterfactual"], r["order"], r["wording"])):
            name, fork, current, generated = row["id"], forks.get(row["id"]), native_rows.get(row["id"]), after.get(row["id"])
            predicted = (float(fork["parent_margin"] + np.asarray(fork["projected_margin_gradient"]) @ left)
                         if fork is not None and left is not None else None)
            outputs.append({"prompt_id": name, "comparison_id": row["comparison_id"],
                "counterfactual": row["counterfactual"], "operand_order": row["order"], "wording": row["wording"],
                "parent_correct": row["correct"], "candidate_correct": generated["correct"] if generated else None,
                "parent_semantic_answer": row["exact"], "correct_semantic_answer": row["answer"],
                "parent_fork_margin": fork["parent_margin"] if fork else None,
                "predicted_candidate_margin": predicted,
                "measured_candidate_margin": current["measured_margin"] if current else None,
                "native_fork_greedy_token": current["greedy_token_id"] if current else None,
                "accepted_native_fork": current["accepted_token_selected"] if current else None,
                "first_divergent_token": divergences.get(name),
                "completed_semantic_answer": generated["exact"] if generated and generated["completed"] else None,
                "generated_response_valid": generated["valid"] if generated else None,
                "behavior_measured": generated is not None,
                "role": "preservation_obligation" if row["correct"] else "validated_target_fork" if fork else "unqualified_error"})
        result[family] = {"rows": outputs,
            "parent_complete": all(row["parent_correct"] for row in outputs),
            "candidate_complete": all(row["candidate_correct"] for row in outputs) if candidate else None}
    return result


def capacity_decision(joint, arms, development):
    target = joint.get("target_only", {})
    protected = joint.get("with_preservation", {})
    certified = {"CERTIFIED_NORM_BOUND_EXCEEDED", "CERTIFIED_RADIUS_SEPARATOR", "INFEASIBLE_ZERO_ROW", "INFEASIBLE_OPPOSED_CONSTRAINTS"}
    if target.get("status") in certified:
        action, reason = "optimized_right_factor_rank_one", "fixed_right_individual_target_local_cap_failure"
    elif protected.get("status") in certified and target.get("within_radius"):
        action, reason = "optimized_right_factor_rank_one", "target_preservation_selectivity_conflict"
    elif any(row.get("native_targets", {}).get("passed") is False for row in arms.values()):
        action, reason = "precision_aware_synthesis", "linear_to_stored_native_disagreement"
    elif any(row.get("native_targets", {}).get("passed") and not row.get("practice_gate", {}).get("passed")
             for row in arms.values() if row.get("behavior_measured")):
        action, reason = "sequence_level_semantic_objective", "forks_correct_but_complete_endpoint_failed"
    elif any(row.get("gate", {}).get("passed") for row in development.values()):
        action, reason = "independent_verification", "fresh_complete_family_requires_existing_host_gates"
    elif any(row.get("practice_gate", {}).get("complete_family_gain", 0) > 0 for row in arms.values()):
        action, reason = "representation_probe", "practice_complete_family_without_qualified_fresh_transfer"
    else:
        action, reason = "additional_measurement", "bounded_numerical_or_behavioral_evidence_inconclusive"
    return {"schema": "weightbridge.phase_iiio.capacity_decision.v1", "next_experiment": action, "reason": reason,
        "production_authority": False, "lineage_authority": False,
        "scope": "Evidence-based recommendation, not implementation or proof that other rank-one right factors can solve the task."}


def summarize(trial, material):
    raw = public_records(trial, material)
    records = {name: json.loads(value) for name, value in raw.items()}
    report, assessment = records["fit-report.json"], records["assessment.json"]
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    _verify_geometry(config, records)
    contract = endpoint(config)
    frozen = {name: load_split(material, name, manifest) for name in report["optimization_splits"]}
    parents = {name: _checked_measurement(records[f"parent-{name}.json"]["measurement"], rows, contract)
               for name, rows in frozen.items()}
    if qualify(parents["adaptation"], config) != report["qualification"]:
        raise ValueError("Parent qualification differs from actual complete responses")
    for name, parent in parents.items():
        if records[f"parent-{name}.json"]["summary"] != semantic_summary(parent):
            raise ValueError("Parent summary does not replay")
    parent, basis = parents["adaptation"], records.get("basis.json", {})
    selection = records.get("supervision-selection.json", {})
    family_ids = selection.get("family_ids", selection.get("selected_family_ids", ()))
    if not family_ids and selection.get("arms"):
        family_ids = selection["arms"].get("B", selection["arms"].get("A", {})).get("family_ids", ())
    arms, deltas, generations, owners, work = {}, {}, {}, {}, {}
    for name in report["artifacts"]:
        if not name.startswith("arm-"):
            scopes = _work_counts(records[name])
            if scopes:
                work[name] = scopes
    for dose in range(1, len(config["probe_design"]["target_norms"]) + 1):
        for label in "AC":
            name = f"{label}{dose}"
            record = records.get(f"arm-{name}.json")
            if record is None:
                continue
            selected, probe = record["norm_match"].get("selected"), record["probe"]
            digest, alias = selected["tensor_sha256"] if selected else None, record.get("duplicate_of")
            delta = _stored_record(record, config)
            if delta is not None:
                if record["columns"] != basis["columns"] or record["right_factor"] != basis["right_factor"]:
                    raise ValueError("Joint candidate coordinates differ from the audited fixed geometry")
                deltas[name] = delta
            native = probe.get("native_measurement") or record.get("native_measurement")
            if native is None and selected:
                native = selected.get("native_measurement") or selected.get("active_measurement")
            row = {"arm": name, "formulation": config["probe_design"]["arms"][label],
                "intended_norm": record["intended_norm"], "realized_norm": selected["realized_norm"] if selected else None,
                "changed_entries": selected["changed_entries"] if selected else None,
                "tensor_sha256": digest, "norm_match": record["norm_match"]["status"], "reason": probe["reason"],
                "duplicate_of": alias, "behavior_measured": "endpoint" in probe,
                "calibration_sufficient": record["calibration_sufficient"],
                "direction_specific_allowance": record["direction_specific_allowance"], "floor": record["floor"],
                "numeric_resources": record["resources"], "probe_resources": probe.get("resources")}
            left = np.asarray(record["left_factor"]) if "left_factor" in record else None
            if native and native.get("targets") and left is not None:
                row["native_targets"] = validate_native_targets(native, basis, left, config["probe_design"]["target_margin"])
            if alias:
                if alias not in generations or owners.get(digest) != alias:
                    raise ValueError("Joint alias does not bind an earlier identical decoded tensor")
                for key in ("practice_gate", "subgroups", "distribution", "fit_regressions", "fit_validity_losses"):
                    row[key] = arms[alias][key]
                generations[name] = generations[alias]
            elif "endpoint" in probe:
                if digest in owners or record["norm_match"]["status"] != "MATCHED" or not record["calibration_sufficient"]:
                    raise ValueError("Decoded joint candidate duplicated, unmatched or uncalibrated")
                if not row.get("native_targets", {}).get("passed") or not native.get("passed") or not probe.get("restored_exactly"):
                    raise ValueError("Generation ran without all required native target decisions and exact restoration")
                protected_rows = native["protected"]["rows"]
                if (len(protected_rows) != len(records["sampling.json"]["anchors"])
                        or any(item["margin"] <= record["floor"]
                               or item["replay_argmax_token_id"] != item["parent_token_id"] for item in protected_rows)):
                    raise ValueError("Decoded candidate lost a protected native decision or positive margin floor")
                owners[digest] = name
                measured = _checked_measurement(probe["endpoint"]["measurement"], frozen["adaptation"], contract)
                preserved = _checked_measurement(probe["endpoint"]["preservation_fit_measurement"], frozen["preservation-fit"], contract)
                gate = compare(parent, measured, min_comparison_gain=1, min_repeatable_corrections=1)
                fit_guard = compare_endpoint(parents["preservation-fit"], preserved, contract)
                valid = sum(item["valid"] for item in measured["rows"]) / len(measured["rows"])
                gate["passed"] = bool(gate["passed"] and not fit_guard["prohibited_regressions"]
                    and not fit_guard["lost_valid_presentations"] and valid >= config["endpoint"]["min_valid_rate"])
                if gate["passed"] != probe["endpoint"]["guard"]["passed"] or fit_guard != probe["endpoint"]["preservation_fit_guard"]:
                    raise ValueError("Joint full-generation guard does not replay independently")
                row.update(practice_gate=gate, subgroups=_subgroup_outcomes(parent, measured, gate),
                    distribution=_distribution(measured), fit_regressions=fit_guard["prohibited_regressions"],
                    fit_validity_losses=fit_guard["lost_valid_presentations"])
                generations[name] = measured
            row["response_matrix"] = response_matrix(parent, generations.get(name), basis, left, native, family_ids,
                                                       probe.get("first_divergences", ())) if family_ids else {}
            arms[name] = row
            scopes = {"numeric": {"resources": record["resources"]}}
            if probe.get("resources"):
                scopes["probe"] = {"resources": probe["resources"]}
            if "endpoint" in probe:
                scopes["generation"] = {key: {"resources": probe["endpoint"][key]["resources"]}
                    for key in ("measurement", "preservation_fit_measurement")}
            work[f"arm-{name}.json"] = _work_counts(scopes)
    unique = sum(row["behavior_measured"] and not row["duplicate_of"] for row in arms.values())
    if unique != report["detail"].get("unique_complete_evaluations", 0) or unique > 4:
        raise ValueError("Actual generation count differs from the frozen four-candidate budget")
    matching = {}
    for dose, norm in enumerate(config["probe_design"]["target_norms"], 1):
        states = {name: {"realized_norm": row["realized_norm"], "tensor_sha256": row["tensor_sha256"]}
                  for name, row in arms.items() if name.endswith(str(dose)) and row["realized_norm"] is not None}
        matching[str(dose)] = compare_realized_norms(states, intended_norm=norm,
            relative_tolerance=config["probe_design"]["norm_relative_tolerance"]) if states else None
    actual_pairwise = stored_update_cosines(deltas) if deltas else {}
    if "stored-pairwise.json" in records and records["stored-pairwise.json"] != {
            "norm_matching": matching, "actual_delta_cosines": actual_pairwise}:
        raise ValueError("Rounded stored-state comparison does not replay")
    development, dev_generations = {}, {}
    before_dev = records.get("verification/development-transfer-base.json", {}).get("measurement")
    if before_dev:
        for name, gate in assessment["development"].items():
            worker = records.get(f"verification/development-transfer-{name}.json")
            if worker:
                measured = worker["measurement"]
                dev_generations[name] = measured
                development[name] = {"gate": gate, "distribution": _distribution(measured),
                    "subgroups": _subgroup_outcomes(before_dev, measured, gate)}
    contrasts = {}
    for dose in range(1, 3):
        first, second = f"A{dose}", f"C{dose}"
        norms = matching[str(dose)]
        distinct = first in arms and second in arms and arms[first]["tensor_sha256"] != arms[second]["tensor_sha256"]
        row = {"distinct_states": bool(distinct), "matched_norms": norms}
        for population, observations, baseline in (("practice", generations, parent), ("development", dev_generations, before_dev)):
            row[population] = (_cluster_difference(baseline, observations[first], observations[second])
                if distinct and norms and norms["cohort_status"] == "MATCHED" and first in observations and second in observations and baseline else
                {"informative": False, "reason": "Requires distinct, matched, numerically qualified complete-generation observations; no artificial behavioral zero."})
        contrasts[f"{first}:{second}"] = row
    totals = Counter()
    for scopes in work.values():
        for counts in scopes.values():
            totals.update(counts)
    joint = records.get("joint-feasibility.json", {})
    verify_joint_systems(joint)
    independent = records.get("independent-outcome.json")
    return {"schema": "weightbridge.phase_iiio.analysis.v1",
        "fit_report_sha256": hashlib.sha256(raw["fit-report.json"]).hexdigest(),
        "assessment_sha256": hashlib.sha256(raw["assessment.json"]).hexdigest(),
        "disposition": independent["disposition"] if independent else assessment["disposition"],
        "reason": independent["reason"] if independent else assessment["reason"], "hidden_opened": bool(independent),
        "lineage_accepted": False, "parent_qualification": report["qualification"],
        "parent_diagnostics": _distribution(parent), "fit_detail": report["detail"], "basis": basis,
        "supervision_selection": selection, "joint_feasibility": joint, "sampling": records.get("sampling.json"),
        "arms": arms, "unique_decoded_probes": unique, "stored_norm_matching": matching,
        "stored_update_comparison": actual_pairwise, "matched_generation_contrasts": contrasts,
        "development": development, "preservation": assessment["preservation"],
        "independent_selection": assessment["selection"], "independent": independent,
        "capacity_decision": capacity_decision(joint, arms, development),
        "resources": {"fitting": report["resources"], "fitting_stage_counts": work,
            "fitting_work_totals": dict(sorted(totals.items())), "verification": assessment["worker_resources"],
            "verification_host": assessment["host_resources"]}}


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    joint = summary["joint_feasibility"]
    keys = [key for key in ("averaged_target_only", "target_only", "with_preservation") if key in joint]
    if keys:
        x = np.arange(len(keys))
        axes[0].bar(x - .18, [joint[k].get("certified_norm_lower_bound")
                             if joint[k].get("certified_norm_lower_bound") is not None else np.nan for k in keys], width=.34,
                    label="Numerical dual lower bound")
        axes[0].bar(x + .18, [joint[k].get("best_primal_norm") if joint[k].get("best_primal_norm") is not None else np.nan
                             for k in keys], width=.34, label="Approximate minimum norm")
        axes[0].set_xticks(x, [key.replace("_", "\n") for key in keys], fontsize=9)
        axes[0].axhline(.025, color="red", linestyle="--", label="Authorized cap")
        axes[0].legend(frameon=False, fontsize=8)
    else:
        axes[0].text(.5, .5, "No qualified numerical systems", ha="center")
    axes[0].set_title("Recorded local linear bounds")
    axes[0].set_ylabel("Unrounded update Frobenius norm")
    measured = [(name, row) for name, row in summary["arms"].items() if row.get("practice_gate")]
    if measured:
        x = np.arange(len(measured))
        axes[1].bar(x - .18, [row["practice_gate"]["target_gain"] for _, row in measured], width=.34, label="Strict comparison gain")
        axes[1].bar(x + .18, [row["practice_gate"]["complete_family_gain"] for _, row in measured], width=.34, label="Complete family gain")
        axes[1].set_xticks(x, [name for name, _ in measured])
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[1].text(.5, .5, "No numerically qualified generation candidates\nBehavioral outcomes remain unmeasured", ha="center", va="center")
        axes[1].set_axis_off()
    axes[1].set_title("Actual complete practice generation")
    fig.savefig(out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = summarize(args.trial, args.material)
    json_dump_atomic(args.out, result)
    if args.figure:
        plot(result, args.figure)
