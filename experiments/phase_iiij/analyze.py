"""Replay public Phase III-J gates and summarize local and measured adaptation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import _paired, endpoint, qualify
from experiments.phase_iiij.decimal_roster import PROTOCOL, read_manifest
from experiments.phase_iiij.run_preservation import _guard
from experiments.phase_iiij.verification import source_hashes
from weightbridge.core import file_hash


def _read(path):
    return json.loads(path.read_bytes())


def _target_items(measurement, contract):
    by_item = defaultdict(list)
    for row in measurement["rows"]:
        if row["group"] == "target":
            answer = contract.parse(row["text"], row["stop_reason"])
            by_item[row["item"]].append(answer.completed and answer.answer == contract.question(row).answer)
    return {item: all(values) for item, values in by_item.items()}


def _paired_item_delta(parent, candidate, contract):
    baseline, after = _target_items(parent, contract), _target_items(candidate, contract)
    if baseline.keys() != after.keys():
        raise ValueError("Underlying target items differ across paired measurements")
    differences = np.asarray([int(after[item]) - int(was_correct) for item, was_correct in baseline.items()])
    # Descriptive paired bootstrap. Resample entire underlying items, never prompt variants.
    generator = np.random.default_rng(329)
    resampled = differences[generator.integers(0, len(differences), size=(4000, len(differences)))].mean(axis=1)
    return {"parent": sum(baseline.values()), "candidate": sum(after.values()), "items": len(baseline),
            "corrected_items": [item for item, value in baseline.items() if not value and after[item]],
            "lost_items": [item for item, value in baseline.items() if value and not after[item]],
            "paired_item_bootstrap_95_descriptive": np.quantile(resampled, [.025, .975]).tolist()}


def _replay_selection(trial, report, lock, config):
    if (file_hash(trial / "candidate-lock.json") != report["selection"]["candidate_lock_sha256"]
            or lock["sources"] != report["sources"] or lock["manifest_sha256"] != report["manifest_sha256"]
            or lock["selection"] != report["selection"]["selection"] or lock["winner"] != report["selection"]["winner"]):
        raise ValueError("Finalist lock differs from the main report")
    parent, pairs = None, {}
    for label in sorted(lock["reports"], key=lambda key: (key != "base", key)):
        path = trial / "verification" / f"selection-{label}.json"
        if file_hash(path) != lock["reports"][label]:
            raise ValueError(f"Independent selection receipt differs for {label}")
        record = _read(path)
        if label == "base":
            parent = record["measurement"]
        else:
            if _paired(parent, record["measurement"], config, config["selection_gate"]) != lock["selection"][label]:
                raise ValueError(f"Independent selection gate differs for {label}")
            pairs[label] = _paired_item_delta(parent, record["measurement"], endpoint(config))
    return pairs


def _replay_interrupted(trial, material):
    root = material.parent
    receipt_path = root / "interrupted-attempt.json"
    if not receipt_path.exists():
        return None
    receipt = _read(receipt_path)
    if receipt["frozen_protocol_sha256"] != file_hash(PROTOCOL):
        raise ValueError("Interrupted attempt references a different protocol")
    checks, resources = {}, []
    for name, digest in receipt["artifact_hashes"].items():
        path = (root / "interrupted" / name).resolve()
        if path.parent != (root / "interrupted").resolve() or file_hash(path) != digest:
            raise ValueError(f"Interrupted artifact hash differs for {name}")
        original = _read(path)
        if name.startswith("arm-"):
            earlier, replayed = original["probe"], _read(trial / name)["probe"]
            checks[name] = {"candidate_tensor_exact": earlier["candidate_tensor_sha256"] == replayed["candidate_tensor_sha256"],
                            "response_rows_exact": earlier["endpoint"]["measurement"]["rows"] ==
                                                   replayed["endpoint"]["measurement"]["rows"]}
            resources.append(earlier["resources"])
        elif name.startswith(("basis-", "reference-", "sensitivities-")) or name in ("diagnosis.json", "parent-decisions.json"):
            resources.append(original["resources"])
    return {"failed_execution": receipt["reason"], "replayed_probes": checks,
            "recorded_stage_wall_seconds": sum(row["wall_seconds"] for row in resources),
            "recorded_stage_cpu_seconds": sum(row["cpu_seconds"] for row in resources)}


def summarize(trial, material):
    report, config = _read(trial / "report.json"), _read(PROTOCOL)
    read_manifest(material)
    if report["sources"] != source_hashes() or report["manifest_sha256"] != file_hash(material / "manifest.json"):
        raise ValueError("Phase III-J sources or roster commitment differ")
    for name, digest in report["artifacts"].items():
        path = (trial / name).resolve()
        if not path.is_relative_to(trial.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Evidence hash differs for {name}")
    lock = _read(trial / "candidate-lock.json")
    selection_pairs = _replay_selection(trial, report, lock, config)
    parent = _read(trial / "practice.json")["measurement"]
    qualification = qualify(parent, config)
    if qualification != report["qualification"]:
        raise ValueError("Practice qualification differs")
    contract = endpoint(config)
    rows = {row["id"]: row for row in parent["rows"]}
    diagnosis_path = trial / "diagnosis.json"
    diagnosis = _read(diagnosis_path) if diagnosis_path.exists() else {}
    fitted_forks = [fork for fork in diagnosis.get("forks", []) if fork["question_id"] in diagnosis["target_items"]]
    arms = {}
    for arm in report["arms"]:
        record = _read(trial / f"arm-{arm}.json")
        probe = record["probe"]
        row = {"reason": probe["reason"], "layer": record["layer"], "nominal_radius": record.get("nominal_radius"),
               "realized_norm": probe.get("realized_update_norm"), "changed_entries": probe.get("changed_entries"),
               "duplicate_of": record.get("duplicate_of"), "probe_resources": probe.get("resources"),
               "candidate_tensor_sha256": probe.get("candidate_tensor_sha256"),
               "counterexample_resources": record.get("counterexamples", {}).get("resources"),
               "added_anchors": record.get("added_anchors", []),
               "anchor_group_counts": dict(Counter(anchor["group"] for anchor in record.get("anchors", []))),
               "training_resources": record.get("training", {}).get("resources")}
        local = record.get("geometry", {}).get("local_feasibility")
        if local is not None:
            row.update(local_crossings=sum(item["safe_interval"] is not None for item in local["per_target"]),
                       local_preservation_max=local["alpha_preserve_max"], local_all_target_interval=local["all_target_interval"],
                       solver=record["geometry"]["solver"])
        observed = probe.get("endpoint")
        if observed is not None:
            measurement = observed["measurement"]
            guard = _guard(parent, measurement, qualification, contract)
            if guard != observed["guard"] or probe.get("restored_exactly") is not True:
                raise ValueError(f"Practice gate or parent restoration differs for {arm}")
            fork = probe.get("fork") or {}
            responses = {value["id"]: value for value in measurement["rows"]}
            fork_paths = []
            for fitted in fitted_forks:
                response = responses[fitted["prompt_id"]]
                prefix, ids = fitted["shared_token_ids"], response["generated_token_ids"]
                position = len(prefix)
                retained = ids[:position] == prefix
                accepted = retained and len(ids) > position and ids[position] in fitted["accepted_token_ids"]
                parsed = contract.parse(response["text"], response["stop_reason"])
                fork_paths.append({"prompt_id": fitted["prompt_id"], "retained": retained, "accepted": accepted,
                                   "completed_correct": parsed.completed and parsed.answer == contract.question(response).answer})
            row.update(endpoint_pass=guard["passed"], target_gain=guard["target_item_gain"],
                       repeatable_corrections=guard["corrected_repeatable_negative_items"],
                       lost_validity=guard["lost_valid_presentations"], regressions=guard["prohibited_regressions"],
                       regression_counts_by_roster_group=dict(Counter(rows[key]["group"] for key in guard["prohibited_regressions"])),
                       fork_ce_gain=fork.get("cross_entropy_gain"), accepted_fork_argmaxes=fork.get("accepted_next_token_count"),
                       practice_targets=_paired_item_delta(parent, measurement, contract),
                       completed=sum(group["completed"] for group in guard["candidate"].values()),
                       valid=sum(group["valid"] for group in guard["candidate"].values()),
                       decode_steps=measurement["resources"]["batched_decode_steps"], actual_fork_paths=fork_paths)
        arms[arm] = row
    for arm, row in arms.items():
        alias = row["duplicate_of"]
        if alias and (row["candidate_tensor_sha256"] != arms[alias]["candidate_tensor_sha256"]
                      or row["layer"] != arms[alias]["layer"] or "endpoint_pass" not in arms[alias]):
            raise ValueError(f"Deduplicated candidate identity differs for {arm}")
    measured = [row for row in arms.values() if "endpoint_pass" in row]
    if len(measured) != report["practice_design"].get("unique_complete_evaluations", 0):
        raise ValueError("Endpoint count differs from the frozen budget receipt")
    independent_path = trial / "independent-outcome.json"
    independent = _read(independent_path) if independent_path.exists() else None
    if independent and independent["candidate_lock_sha256"] != report["selection"]["candidate_lock_sha256"]:
        raise ValueError("Independent outcome references a different finalist lock")
    accepted = bool(independent and independent["disposition"] == "PASS")
    return {"schema": "weightbridge.phase_iiij.analysis.v1", "report_sha256": file_hash(trial / "report.json"),
            "single_cycle": independent["disposition"] if independent else report["single_cycle"],
            "practice_correction": "PASS" if any(row["endpoint_pass"] for row in measured) else "HOLD",
            "independent_selection": "PASS" if lock["winner"] else "FAIL" if lock["selection"] else "HOLD",
            "persistent_improvement": "PASS" if accepted else "HOLD", "sequential_readiness": "HOLD",
            "arms": arms, "independent_outcome": independent, "winner": lock["winner"],
            "unique_decoded_probes": len(measured), "passing_practice_probes": sum(row["endpoint_pass"] for row in measured),
            "selection_pairs": selection_pairs,
            "interrupted_replay": _replay_interrupted(trial, material),
            "resources": {**report["resources"], "selection": report["selection"]["verifier_resources"]}}


def plot_doses(analysis, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = analysis["arms"]
    doses = [(row["realized_norm"], arms[row["duplicate_of"]] if row["duplicate_of"] else row)
             for name, row in arms.items() if name.startswith("E") and row["realized_norm"] is not None]
    doses = sorted((norm, row) for norm, row in doses if "endpoint_pass" in row)
    if not doses:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), layout="constrained")
    norms, rows = zip(*doses, strict=True)
    axes[0].plot(norms, [row["target_gain"] for row in rows], color="#165E80", marker="o", label="Net target gain")
    axes[0].plot(norms, [len(row["repeatable_corrections"]) for row in rows], color="#436642", marker="s",
                 label="Repeatable errors corrected")
    axes[1].plot(norms, [len(row["regressions"]) for row in rows], color="#963F28", marker="o",
                 label="Previously correct presentations lost")
    for ax in axes:
        ax.axhline(0, color="#8B8B8B", linewidth=.8, zorder=0)
        ax.set_xlabel("Realized BF16 update norm")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Underlying target items")
    axes[1].set_ylabel("Responses")
    fig.savefig(out, dpi=170, metadata={"Description": "Phase III-J measured Qwen practice doses"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("trial", "material"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    analysis = summarize(args.trial, args.material)
    if args.plot:
        plot_doses(analysis, args.plot)
    print(json.dumps(analysis, indent=2))
