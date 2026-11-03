"""Audit and replay a paired connectivity ablation from saved evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from summarize_local_mcp_runs import (
    ROOT, ToyMLP, _file_sha256, _read_json, read_lines, summarize_run,
)
from run_local_mcp_cohort import SOURCE, diagnostic_view, write_json


LOGIT_THRESHOLD = 1e-9
ARMS = ("control", "probe")


def verify_inputs(control, probe):
    if _file_sha256(control / "base.npz") != _file_sha256(probe / "base.npz"):
        raise ValueError(f"Base checkpoints differ for {control.name}")
    for name in ("practice.npz", "hidden_holdout.npz"):
        with np.load(control / name, allow_pickle=False) as a, np.load(probe / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError(f"Split arrays differ for {control.name}/{name}")
    connectivity = ToyMLP.load(probe / "base.npz").logit_connectivity()
    expected = None
    for arm, directory in zip(ARMS, (control, probe)):
        transcript = read_lines(directory / "agent_mcp_transcript.jsonl")
        report = transcript[0]["response"]["result"]
        view = diagnostic_view(report)
        if arm == "control":
            expected = view
        elif view != expected:
            raise ValueError(f"Practice views differ for {control.name}")
        if arm == "probe":
            view["logit_connectivity"] = connectivity
        if view != _read_json(directory / "diagnostics_sent.json"):
            raise ValueError(f"Saved diagnostic view differs for {directory}")
    return connectivity["single_edits_with_no_logit_path"]


def indicators(trial, disconnected):
    edits = trial["accepted_practice_edits"]
    active = sum(item["max_absolute_practice_logit_delta"] > LOGIT_THRESHOLD for item in edits)
    return {
        "any_logit_change": active > 0,
        "any_practice_class_change": any(item["practice_class_changes"] > 0 for item in edits),
        "any_practice_accuracy_gain": any(item["practice_correct_delta"] > 0 for item in edits),
        "logit_changing_submissions": active,
        "accepted_submissions": len(edits),
        "distinct_accepted_checkpoints": len({item["candidate_sha256"] for item in edits}),
        "disconnected_upstream_submissions": sum(
            any(change["index"] in disconnected.get(change["tensor"], []) for change in item["patch"])
            for item in edits
        ),
        "finalized_candidate": bool(trial["counts"]["finalized_candidates"]),
        "full_holdout_repair": bool(trial["counts"]["full_repairs"]),
    }


def replay(root):
    plan = _read_json(root / "paired_plan.json")
    runs = {}
    trials = {}
    identities = {}
    protocols = {}
    for arm in ARMS:
        directory = root / arm
        protocol = protocols[arm] = _read_json(directory / "frozen_protocol.json")
        if protocol["protocol_sha256"] != plan["protocol_sha256"][arm]:
            raise ValueError(f"Protocol differs from frozen plan for {arm}")
        identity = _read_json(directory / "frozen_protocol.identity_receipt.json")
        if (identity.pop("connectivity_probe") is not (arm == "probe")
                or identity["host_source_sha256"] != _file_sha256(SOURCE)
                or identity["toy_model_source_sha256"] != _file_sha256(ROOT / "weightbridge/diagnostics/toy_mlp.py")):
            raise ValueError(f"Host receipt differs from the replay source for {arm}")
        identity.pop("host_call_id")
        identities[arm] = identity
        run = runs[arm] = summarize_run(directory, include_class_changes=True)
        missing = len(protocol["tasks"]) - identity["planned_task_prefix"]
        expected_reasons = [f"missing {missing} of {len(protocol['tasks'])} planned task records"] if missing else []
        if run["cohort_reasons"] != expected_reasons:
            raise ValueError(f"Cohort evidence failed audit for {arm}")
        trials[arm] = {trial["task_id"]: trial for trial in run["trials"]}
    if identities["control"] != identities["probe"]:
        raise ValueError("Model, host or budget differs across arms")
    for key in ("tasks", "execution_order", "task_design", "host_policy", "prompt_sha256", "source_hashes"):
        if protocols["control"][key] != protocols["probe"][key]:
            raise ValueError(f"Paired protocol setting differs: {key}")
    prefix = identities["control"]["planned_task_prefix"]
    task_order = protocols["control"]["execution_order"][:prefix]
    expected_schedule = [
        {"task": task_id, "arm": arm}
        for index, task_id in enumerate(task_order)
        for arm in (ARMS if index % 2 == 0 else ARMS[::-1])
    ]
    if plan["schedule"] != expected_schedule or any(set(trials[arm]) != set(task_order) for arm in ARMS):
        raise ValueError("Planned pairs are incomplete or differ from the frozen order")
    timestamps = [datetime.fromisoformat(_read_json(
        root / step["arm"] / step["task"] / "experiment_record.json")["timestamp_utc"])
        for step in expected_schedule]
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Trial timestamps differ from the frozen counterbalanced order")
    pairs = []
    for task_id in task_order:
        disconnected = verify_inputs(root / "control" / task_id, root / "probe" / task_id)
        pairs.append({
            "task_id": task_id,
            "mechanism": trials["control"][task_id]["mechanism"],
            **{arm: indicators(trials[arm][task_id], disconnected) for arm in ARMS},
        })
    effect = [int(pair["probe"]["any_logit_change"]) - int(pair["control"]["any_logit_change"])
              for pair in pairs]
    return {
        "schema": "weightbridge.connectivity_ablation_summary.v1",
        "disposition": "INCONCLUSIVE",
        "plan_sha256": _file_sha256(root / "paired_plan.json"),
        "pairs": pairs, "verified_pairs": len(pairs),
        "primary": {
            "logit_threshold": LOGIT_THRESHOLD,
            "active_tasks": {arm: sum(pair[arm]["any_logit_change"] for pair in pairs) for arm in ARMS},
            "probe_minus_control_fraction": sum(effect) / len(effect),
            "observed_increase": sum(effect) > 0,
            "probe_wins": effect.count(1), "control_wins": effect.count(-1), "ties": effect.count(0),
        },
        "runs": runs, "host_resources": _read_json(root / "host_resources.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.out, replay(args.study))


if __name__ == "__main__":
    main()
