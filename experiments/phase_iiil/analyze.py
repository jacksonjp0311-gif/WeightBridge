"""Replay Phase III-L public decisions and summarize the matched decoded updates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.phase_iiih.run_decimal_endpoint import endpoint, qualify
from experiments.phase_iiik.run_jspace import _active_measurements
from experiments.phase_iiil.archive_trial import public_records
from experiments.phase_iiil.semantic_diagnostics import compare_semantic, summarize as summarize_semantics
from experiments.phase_iiil.semantic_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import _observation, _roster, compare_endpoint
from weightbridge.adaptation.preservation_geometry import DecisionAnchor


def _checked_measurement(measurement, rows, contract):
    if measurement["endpoint"] != contract.manifest():
        raise ValueError("Generation evidence uses a different behavioral endpoint")
    questions, _ = _roster(rows, contract)
    for frozen, observed in zip(rows, measurement["rows"], strict=True):
        if any(observed[key] != value for key, value in frozen.items()):
            raise ValueError("Generation evidence differs from its frozen presentation")
        status = _observation(observed, questions[frozen["id"]], contract)
        if any(observed[key] != value for key, value in status.items()):
            raise ValueError("Generation evidence has inconsistent parsing or correctness")
    return measurement


def _subgroup_outcomes(parent, candidate, gate):
    before, after = summarize_semantics(parent), summarize_semantics(candidate)
    results = {}
    for name, group in before["subgroups"].items():
        items = [item for item, row in before["comparisons"].items() if row["group"] == name]
        prior = {row["id"] for row in before["target_observations"] if row["group"] == name}
        results[name] = {"presentations": group["presentations"], "comparisons": len(items),
            "parent_correct_presentations": group["correct"],
            "candidate_correct_presentations": after["subgroups"][name]["correct"],
            "parent_strict_correct": sum(before["comparisons"][item]["all_correct"] for item in items),
            "candidate_strict_correct": sum(after["comparisons"][item]["all_correct"] for item in items),
            "regressed_presentations": [row for row in gate["regressions"] if row in prior],
            "new_strict_corrections": [item for item in gate["corrected_target_comparisons"] if item in items]}
    return results


def _practice_replay(record, probe, parent, protected, contract, anchors, prompts, frozen):
    scored = probe["endpoint"]
    candidate = _checked_measurement(scored["measurement"], frozen["adaptation"], contract)
    candidate_fit = _checked_measurement(scored["preservation_fit_measurement"], frozen["preservation-fit"], contract)
    semantic = compare_semantic(parent, candidate, min_comparison_gain=1, min_repeatable_corrections=1)
    fit = compare_endpoint(protected, candidate_fit, contract)
    behavior = semantic["passed"] and not fit["prohibited_regressions"] and not fit["lost_valid_presentations"]
    active = _active_measurements(anchors, probe, prompts, record["geometry"],
                                  floor=record["floor"], allowance=record["allowance"])
    floor_passed = bool(active and all(row["decision_preserved"] and row["measured_margin"] > record["floor"]
                                       for row in active))
    recorded = scored["guard"]
    fields = {**semantic, "passed": bool(behavior and floor_passed), "semantic_behavior_passed": bool(behavior),
              "strict_measured_floor_passed": floor_passed}
    if (not probe["restored_exactly"] or any(recorded[key] != value for key, value in fields.items())
            or fit != scored["preservation_fit_guard"] or active != record["active_measurements"]):
        raise ValueError("Public practice result does not reproduce its semantic, preservation, or rounded-margin gate")
    return {"practice_gate": fields, "adaptation_subgroups": _subgroup_outcomes(parent, candidate, fields),
            "fit_regressions": fit["prohibited_regressions"],
            "fit_validity_losses": fit["lost_valid_presentations"],
            "minimum_measured_margin": min((row["measured_margin"] for row in active), default=None),
            "measured_robust_floor_count": sum(row["meets_robust_floor"] for row in active),
            "active_anchor_count": len(active), "endpoint_resources": {"adaptation": candidate["resources"],
                                                                         "preservation_fit": candidate_fit["resources"]}}


def _measured_work(value, path=""):
    work = {}
    count_keys = {"forward_calls", "backward_calls", "batched_decode_steps", "generate_batches", "generated_examples"}
    if not isinstance(value, dict):
        return work
    for key, child in value.items():
        name = f"{path}/{key}" if path else key
        if key == "resources" and isinstance(child, dict):
            counts = {k: v for k, v in child.items() if k in count_keys}
            if counts:
                work[name] = counts
        elif isinstance(child, dict):
            work.update(_measured_work(child, name))
        elif isinstance(child, list) and key in ("rounds", "attempts"):
            for index, attempt in enumerate(child):
                work.update(_measured_work(attempt, f"{name}/{index}"))
    return work


def summarize(trial, material):
    raw = public_records(trial, material)
    records = {name: json.loads(value) for name, value in raw.items()}
    report, assessment = records["fit-report.json"], records["assessment.json"]
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    contract = endpoint(config)
    frozen = {name: load_split(material, name, manifest) for name in report["optimization_splits"]}
    parents = {name: _checked_measurement(records[f"parent-{name}.json"]["measurement"], rows, contract)
               for name, rows in frozen.items()}
    for name, parent in parents.items():
        if summarize_semantics(parent) != records[f"parent-{name}.json"]["summary"]:
            raise ValueError("Archived parent summary differs from completed generations")
    adaptation = parents["adaptation"]
    if qualify(adaptation, config) != report["qualification"]:
        raise ValueError("Parent qualification differs from its endpoint observations")
    teacher = records["teacher-forced-parent.json"]
    preferences = {row["prompt_id"]: row["sum_preference"] for row in teacher["rows"]}
    baseline = summarize_semantics(adaptation, teacher_preferences=preferences)
    if baseline != records["parent-diagnostics.json"]:
        raise ValueError("Archived parent teacher-forced comparison differs")
    fork_data = records.get("fork-diagnosis.json", {})
    forks = {row["prompt_id"]: row.get("fork_tokens") for row in fork_data.get("forks", []) if row.get("fork_tokens")}
    if forks:
        baseline = summarize_semantics(adaptation, teacher_preferences=preferences, fork_tokens=forks)
    directions = {str(i): records[f"geometry-{i}.json"] for i in range(1, 3) if f"geometry-{i}.json" in records}
    sampling = records.get("sampling.json", {})
    anchors = tuple(DecisionAnchor(**row) for row in sampling.get("anchors", ()))
    prompts = records.get("probe-reference.json", {}).get("prompt_ids", ())
    arms, owners = {}, {}
    for name in ("A1", "B1", "A2", "B2"):
        record = records.get(f"arm-{name}.json")
        if record is None:
            continue
        probe, duplicate = record["probe"], record.get("duplicate_of")
        digest = probe.get("candidate_tensor_sha256")
        if duplicate and (not digest or owners.get(digest) != duplicate):
            raise ValueError("Rounded candidate alias does not name the identical measured tensor")
        row = {"reason": probe["reason"], "duplicate_of": duplicate, "candidate_tensor_sha256": digest,
               "nominal_radius": record["nominal_radius"], "proposed_norm": probe.get("proposed_norm"),
               "realized_norm": probe.get("realized_update_norm"), "changed_entries": probe.get("changed_entries"),
               "restored_exactly": probe.get("restored_exactly"), "resources": probe.get("resources"),
               "local_feasibility": record.get("geometry", {}).get("local_feasibility")}
        if "endpoint" in probe:
            if not digest or digest in owners:
                raise ValueError("Unique practice decoding does not bind a distinct rounded tensor")
            owners[digest] = name
            row.update(_practice_replay(record, probe, adaptation, parents["preservation-fit"], contract,
                                        anchors, prompts, frozen))
            row["fork"] = probe.get("fork")
        elif duplicate:
            for key in ("practice_gate", "adaptation_subgroups", "fit_regressions", "fit_validity_losses", "fork"):
                row[key] = arms[duplicate][key]
        arms[name] = row
    decoded = [row for row in arms.values() if "practice_gate" in row and not row["duplicate_of"]]
    if len(decoded) != report["detail"].get("unique_complete_evaluations", 0):
        raise ValueError("Archived decoded-candidate count differs from the fitting budget receipt")
    matched = {}
    for index, direction in directions.items():
        a, b = arms.get(f"A{index}", {}), arms.get(f"B{index}", {})
        matched[index] = {"both_have_behavior_measurement": all("practice_gate" in row for row in (a, b)),
                          "both_decoded": all("practice_gate" in row and not row["duplicate_of"] for row in (a, b)),
                          "distinct_stored_tensors": bool(a.get("candidate_tensor_sha256") and b.get("candidate_tensor_sha256")
                                                          and a["candidate_tensor_sha256"] != b["candidate_tensor_sha256"]),
                          **direction["angles"]["historical:maximin"]}
    fork_count = Counter(row.get("actual_category") for row in fork_data.get("forks", []) if row.get("actual_category"))
    fork_by_group = {}
    for observation in baseline["target_observations"]:
        fork = observation["fork"]
        if fork is None:
            continue
        group = fork_by_group.setdefault(observation["group"], dict.fromkeys(
            ("forks", "replay_matched", "at_first_distinguishing_digit", "at_answer_length_boundary",
             "accepted_termination"), 0))
        group["forks"] += 1
        group["replay_matched"] += fork["replay_status"] == "matched"
        for key in ("at_first_distinguishing_digit", "at_answer_length_boundary"):
            group[key] += fork[key]
        group["accepted_termination"] += any(token["kind"] in ("newline", "eos") for token in fork["accepted_tokens"])
    independent = records.get("independent-outcome.json")
    reused = {f"parent-{name}.json" for name in report.get("reused_parent_receipts", [])}
    counted = {record: _measured_work(records[record]) for record in report["artifacts"] if record not in reused}
    counted = {record: counts for record, counts in counted.items() if counts}
    total_work = Counter()
    for scopes in counted.values():
        for counts in scopes.values():
            total_work.update(counts)
    verified_subgroups = {}
    for gate, split in (("transfer", "transfer-validation"), ("selection", "selection")):
        outcomes = {}
        for arm, comparison in assessment[gate].items():
            parent = records.get(f"verification/{split}-base.json")
            candidate = records.get(f"verification/{split}-{arm}.json")
            if parent and candidate:
                outcomes[arm] = _subgroup_outcomes(parent["measurement"], candidate["measurement"], comparison)
        verified_subgroups[gate] = outcomes
    return {"schema": "weightbridge.phase_iiil.analysis.v1",
            "fit_report_sha256": hashlib.sha256(raw["fit-report.json"]).hexdigest(),
            "assessment_sha256": hashlib.sha256(raw["assessment.json"]).hexdigest(),
            "disposition": independent["disposition"] if independent else assessment["disposition"],
            "reason": independent["reason"] if independent else assessment["reason"],
            "hidden_opened": independent.get("hidden_opened") if independent else False, "winner": assessment["winner"],
            "parent_qualification": report["qualification"], "parent_diagnostics": baseline,
            "fork_counts": dict(sorted(fork_count.items())), "fork_types_by_group": fork_by_group, "fit_detail": report["detail"],
            "basis": records.get("basis.json"), "projected_agreement": records.get("projected-agreement.json"),
            "directions": directions, "matched_pairs": matched, "sampling": sampling or None,
            "calibration": records.get("calibration.json"), "arms": arms, "unique_decoded_probes": len(decoded),
            "practice_passers": [name for name, row in arms.items()
                                 if not row["duplicate_of"] and row.get("practice_gate", {}).get("passed")],
            "transfer": assessment["transfer"], "selection": assessment["selection"],
            "verified_subgroups": verified_subgroups, "independent": independent,
            "resources": {"fitting": report["resources"], "teacher_forced": teacher["resources"],
                          "fork_diagnosis": fork_data.get("resources"), "fitting_stage_counts": counted,
                          "fitting_work_totals": dict(sorted(total_work.items())),
                          "reused_parent_measurements": sorted(reused),
                          "verification": assessment["worker_resources"]}}


def plot(summary, output):
    """Plot recorded strict gain, regressions, and observed subgroup agreement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    measured = [(name, row) for name, row in summary["arms"].items() if "practice_gate" in row]
    agreement = summary["projected_agreement"]
    if not measured and not (agreement and agreement["group_cosines"]):
        return False
    fig, (matrix, behavior, subgroup) = plt.subplots(1, 3, figsize=(13.7, 4), layout="constrained")
    if agreement and agreement["group_cosines"]:
        labels = agreement["group_cosine_order"]
        values = np.array(agreement["group_cosines"], dtype=float)
        picture = matrix.imshow(values, cmap="RdBu", vmin=-1, vmax=1)
        matrix.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        matrix.set_yticks(range(len(labels)), labels)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                matrix.text(j, i, f"{value:.2f}", ha="center", va="center",
                            color="white" if abs(value) > .6 else "black")
        fig.colorbar(picture, ax=matrix, fraction=.045)
        matrix.set_title("Projected subgroup gradient agreement")
    else:
        matrix.set_axis_off()
    if measured:
        names = [f"{name} ({row['duplicate_of']})" if row["duplicate_of"] else name for name, row in measured]
        x = np.arange(len(names))
        behavior.bar(x - .2, [row["practice_gate"]["target_gain"] for _, row in measured], width=.38,
                     label="Strict target gain", color="#35724c")
        behavior.bar(x + .2, [len(row["practice_gate"]["regressions"]) + len(row["fit_regressions"])
                              for _, row in measured], width=.38, label="Previously correct responses lost", color="#ab563f")
        behavior.set_xticks(x, names)
        behavior.set_title("Complete practice generation")
        behavior.yaxis.set_major_locator(MaxNLocator(integer=True))
        behavior.axhline(0, color="#777777", linewidth=.7)
        behavior.legend(loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False)
        groups = list(measured[0][1]["adaptation_subgroups"])
        gains = [[row["adaptation_subgroups"][group]["candidate_correct_presentations"]
                  - row["adaptation_subgroups"][group]["parent_correct_presentations"] for group in groups]
                 for _, row in measured]
        span = max(1, max(abs(value) for row in gains for value in row))
        subgroup.imshow(gains, cmap="RdBu", vmin=-span, vmax=span, aspect="auto")
        subgroup.set_xticks(range(len(groups)), [name.removeprefix("negative-").replace("/", "\n") for name in groups])
        subgroup.set_yticks(range(len(names)), names)
        for i, row in enumerate(gains):
            for j, value in enumerate(row):
                subgroup.text(j, i, f"{value:+d}" if value else "0", ha="center", va="center",
                              color="white" if abs(value) > .55 * span else "black")
        subgroup.set_title("Net correct presentations by subgroup")
    else:
        behavior.set_axis_off()
        subgroup.set_axis_off()
    fig.savefig(output, dpi=165, metadata={"Description": "Phase III-L measured agreement and full greedy outcomes."})
    plt.close(fig)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("trial", "material"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    summary = summarize(args.trial, args.material)
    if args.plot:
        plot(summary, args.plot)
    print(json.dumps(summary, indent=2))
