"""Replay the public Phase III-K gates and compare actual rounded probes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from experiments.phase_iiih.run_decimal_endpoint import _paired, endpoint, qualify
from experiments.phase_iiik.decimal_roster import PROTOCOL, read_manifest
from experiments.phase_iiik.run_jspace import _active_measurements, _practice_guard, _selection_keys
from experiments.phase_iiik.verification import preservation_gate, source_hashes
from weightbridge.core import file_hash


def _read(path):
    return json.loads(path.read_bytes())


def _tier_summary(before, after, contract):
    grouped = defaultdict(list)
    for a, b in zip(before["rows"], after["rows"], strict=True):
        question = contract.question(a)
        key = a["group"], a.get("tier") or "copy", a["item"]
        grouped[key].append((contract.parse(a["text"], a["stop_reason"]).answer == question.answer,
                             contract.parse(b["text"], b["stop_reason"]).answer == question.answer))
    results = {}
    for (group, tier, _), values in grouped.items():
        row = results.setdefault(f"{group}/{tier}", {"items": 0, "parent_correct_items": 0,
                                                    "candidate_correct_items": 0, "regressed_presentations": 0})
        row["items"] += 1
        row["parent_correct_items"] += all(a for a, _ in values)
        row["candidate_correct_items"] += all(b for _, b in values)
        row["regressed_presentations"] += sum(a and not b for a, b in values)
    return results


def _validation_replay(trial, config, report, lock):
    assessment = report["selection"]
    if (file_hash(trial / "candidate-lock.json") != assessment["candidate_lock_sha256"]
            or any(lock[key] != assessment[key] for key in ("winner", "selection", "validation"))
            or lock["sources"] != report["sources"] or lock["manifest_sha256"] != report["manifest_sha256"]):
        raise ValueError("Finalist lock differs")
    for stage, prefix, compare in (
        ("validation", "preservation-validation", preservation_gate),
        ("selection", "selection", lambda before, after, config: _paired(before, after, config, config["selection_gate"])),
    ):
        reports = lock["validation_reports" if stage == "validation" else "reports"]
        base = None
        for label, digest in sorted(reports.items(), key=lambda pair: (pair[0] != "base", pair[0])):
            path = trial / "verification" / f"{prefix}-{label}.json"
            if file_hash(path) != digest:
                raise ValueError(f"Independent receipt differs for {prefix}-{label}")
            measured = _read(path)["measurement"]
            if label == "base":
                base = measured
            elif compare(base, measured, config) != lock[stage][label]:
                raise ValueError(f"Independent gate differs for {prefix}-{label}")


def summarize(trial, material):
    report, config = _read(trial / "report.json"), _read(PROTOCOL)
    read_manifest(material)
    if report["sources"] != source_hashes() or report["manifest_sha256"] != file_hash(material / "manifest.json"):
        raise ValueError("Phase III-K sources or material commitment differs")
    for name, digest in report["artifacts"].items():
        path = (trial / name).resolve()
        if not path.is_relative_to(trial.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Evidence hash differs for {name}")
    lock = _read(trial / "candidate-lock.json")
    _validation_replay(trial, config, report, lock)
    contract = endpoint(config)
    adaptation = _read(trial / "parent-adaptation.json")["measurement"]
    fit = _read(trial / "parent-preservation-fit.json")["measurement"]
    if qualify(adaptation, config) != report["qualification"]:
        raise ValueError("Parent adaptation qualification differs")
    raw = {name: _read(trial / f"arm-{name}.json") for name in report["arms"] if name not in ("D", "E")}
    samples = _read(trial / "sampling.json") if raw else {}
    secant = _read(trial / "directional-secant.json") if raw else {}
    forecast = {(row["prompt_id"], row["position"]) for row in secant.get("rows", [])
                if row["candidate_replay"]["margin"] <= 0}

    def observed(name):
        return raw[raw[name].get("duplicate_of", name)]["probe"]

    arms = {}
    from weightbridge.adaptation.preservation_geometry import DecisionAnchor

    for name, record in raw.items():
        probe, used = record["probe"], observed(name)
        if record.get("duplicate_of") and probe["candidate_tensor_sha256"] != used["candidate_tensor_sha256"]:
            raise ValueError(f"Deduplicated candidate differs for {name}")
        anchors = tuple(DecisionAnchor(**anchor) for anchor in record["anchors"])
        active = _active_measurements(anchors, used, record["trajectory_prompt_ids"], record["geometry"],
                                      floor=record["floor"] or 0., allowance=record["allowance"] or 0.)
        if active != record["active_measurements"]:
            raise ValueError(f"Rounded active-margin measurements differ for {name}")
        row = {"reason": probe["reason"], "duplicate_of": record.get("duplicate_of"),
               "nominal_radius": record["nominal_radius"], "realized_norm": probe.get("realized_update_norm"),
               "changed_entries": probe.get("changed_entries"), "floor": record["floor"], "allowance": record["allowance"],
               "anchor_groups": dict(Counter(anchor.group for anchor in anchors)), "active_count": len(active),
               "active_positive_count": sum(point["positive"] for point in active),
               "linear_binding_count": sum(point["predicted_margin"] is not None and
                                           abs(point["predicted_margin"] - ((record["floor"] or 0.) +
                                               (record["allowance"] or 0.))) <= 1e-6 for point in active),
               "active_measured_robust_floor_count": sum(point["meets_robust_floor"] for point in active),
               "minimum_measured_margin": min((point["measured_margin"] for point in active), default=None),
               "resources": probe.get("resources")}
        local = record["geometry"].get("local_feasibility")
        if local is not None:
            row.update(individual_positive_windows=sum(item["safe_width"] > 0 for item in local["per_target"]),
                       paired_positive_windows=sum(item["safe_width"] > 0 for item in local["paired_item_windows"]),
                       max_paired_safe_width=local["max_paired_safe_width"], joint_safe_width=local["joint_safe_width"],
                       solver=record["geometry"]["solver"])
        if "endpoint" in used:
            result = used["endpoint"]
            measured, measured_fit = result["measurement"], result["preservation_fit_measurement"]
            guard, fit_guard = _practice_guard(adaptation, measured, fit, measured_fit, report["qualification"], contract)
            if guard != result["guard"] or fit_guard != result["preservation_fit_guard"] or not used["restored_exactly"]:
                raise ValueError(f"Archived full practice gate differs for {name}")
            before_by_id = {row["id"]: row["generated_token_ids"] for row in fit["rows"]}
            after_by_id = {row["id"]: row["generated_token_ids"] for row in measured_fit["rows"]}
            failures = []
            for prompt_id in fit_guard["prohibited_regressions"]:
                before_ids, after_ids = before_by_id[prompt_id], after_by_id[prompt_id]
                failures.append((prompt_id, next((i for i, (a, b) in enumerate(zip(before_ids, after_ids)) if a != b),
                                                min(len(before_ids), len(after_ids)))))
            row.update(practice_pass=guard["passed"], adaptation_gain=guard["target_item_gain"],
                       repeatable_corrections=guard["corrected_repeatable_negative_items"],
                       adaptation_regressions=guard["prohibited_regressions"], fit_regressions=fit_guard["prohibited_regressions"],
                       fit_regression_first_differences=failures,
                       secant_nonpositive_at_fit_regression=sum(key in forecast for key in failures),
                       sampler_exact_fit_regression_anchor_hits={key: sum(f in {(a["prompt_id"], a["position"]) for a in sample["anchors"]}
                                                                         for f in failures) for key, sample in samples.items() if sample},
                       adaptation_by_tier=_tier_summary(adaptation, measured, contract),
                       fit_by_tier=_tier_summary(fit, measured_fit, contract),
                       decode_steps=measured["resources"]["batched_decode_steps"] + measured_fit["resources"]["batched_decode_steps"])
        arms[name] = row
    decoded = [row for row in arms.values() if not row["duplicate_of"] and "practice_pass" in row]
    if len(decoded) != report["practice_design"].get("unique_complete_evaluations", 0):
        raise ValueError("Rounded decode count differs from the trial budget receipt")
    choice = report["practice_design"].get("d_choice")
    if choice is not None and any(choice[key] != value for key, value in _selection_keys(raw, observed).items()):
        raise ValueError("Interior choice differs from locked practice ranking")
    best_gain = max((row["adaptation_gain"] for row in decoded), default=None)
    for row in arms.values():
        if "adaptation_gain" in row:
            row["descriptive_unconstrained_gain_regret"] = best_gain - row["adaptation_gain"]
    arms.update({name: report["arms"][name] for name in ("D", "E") if name in report["arms"]})
    hidden_result = trial / "independent-outcome.json"
    independent = _read(hidden_result) if hidden_result.exists() else None
    if independent and independent["candidate_lock_sha256"] != report["selection"]["candidate_lock_sha256"]:
        raise ValueError("Hidden outcome does not bind the locked candidate")
    return {"schema": "weightbridge.phase_iiik.analysis.v1", "report_sha256": file_hash(trial / "report.json"),
            "disposition": independent["disposition"] if independent else report["single_cycle"], "arms": arms,
            "d_choice": choice, "validation": lock["validation"], "selection": lock["selection"], "independent": independent,
            "unique_decoded_probes": len(decoded), "practice_passing_probes": sum(row["practice_pass"] for row in decoded),
            "secant_positions": secant.get("position_count"), "secant_nonpositive_positions": len(forecast),
            "held_residual_audit": report["held_residual_audit"],
            "resources": {**report["resources"], "verifier": report["selection"]["verifier_resources"]}}


def plot_arms(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    arms = [(name, row) for name, row in summary["arms"].items() if "practice_pass" in row]
    if not arms:
        return
    labels = [f"{name} ({row['duplicate_of']})" if row["duplicate_of"] else name for name, row in arms]
    fig, (gain, preservation) = plt.subplots(1, 2, figsize=(8.6, 3.5), layout="constrained")
    colors = {"A": "#72828D", "B": "#2D6383", "C": "#35724C"}
    gain.bar(labels, [row["adaptation_gain"] for _, row in arms], color=[colors[name[0]] for name, _ in arms])
    gain.axhline(0, color="#888888", linewidth=.8)
    gain.set_ylabel("Net gain in complete target items")
    adapt = [len(row["adaptation_regressions"]) for _, row in arms]
    preservation.bar(labels, adapt, color="#AA5E3F", label="Adaptation")
    preservation.bar(labels, [len(row["fit_regressions"]) for _, row in arms], bottom=adapt,
                     color="#5B4A73", label="Preservation fit")
    preservation.set_ylabel("Previously correct presentations lost")
    preservation.legend(frameon=False)
    preservation.set_ylim(0, max(1, max(a + len(row["fit_regressions"]) for a, (_, row) in zip(adapt, arms, strict=True))) * 1.35)
    for axis in (gain, preservation):
        axis.tick_params(axis="x", labelrotation=25)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(output, dpi=170, metadata={"Description": "Phase III-K actual BF16 practice outcomes. Parentheses identify reused exact states."})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("trial", "material"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = summarize(args.trial, args.material)
    if args.plot:
        plot_arms(result, args.plot)
    print(json.dumps(result, indent=2))
