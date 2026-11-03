"""Replay the practice breakpoint menu and paired AI selection experiment."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from summarize_local_mcp_runs import ROOT, ToyMLP, _file_sha256, _read_json, read_lines, summarize_run
from summarize_search_comparison import check_splits
from run_local_mcp_cohort import SOLVER_ARMS, SOURCE, write_json
from weightbridge.diagnostics.breakpoint_search import breakpoint_menu, rank_key


def verify_trial(directory, arm, trial, policy):
    saved = _read_json(directory / "solver_menu.json")
    with np.load(directory / "practice.npz", allow_pickle=False) as practice:
        menu = breakpoint_menu(ToyMLP.load(directory / "base.npz"),
                               practice["features"], practice["labels"], policy)
    order = np.random.default_rng(trial["seed"]).permutation(len(menu["candidates"]))
    options = {chr(65 + i): menu["candidates"][int(j)] for i, j in enumerate(order)}
    for key in ("baseline_correct", "practice_n", "analytical_deltas_scored"):
        if menu[key] != saved[key]:
            raise ValueError(f"Menu replay differs for {directory}/{key}")
    if options != saved["options"]:
        raise ValueError(f"Candidate options differ for {directory}")
    if options and arm == "guided":
        logits = saved["label_logits"]
        if set(logits) != set(options) or saved["model_forwards"] != 1:
            raise ValueError(f"Model choice evidence is invalid for {directory}")
        label = max(options, key=logits.__getitem__)
    else:
        if saved["model_forwards"] != 0:
            raise ValueError(f"Unexpected model forward for {directory}")
        label = min(options, key=lambda item: rank_key(options[item])) if options else None
    if label != saved["selected_label"]:
        raise ValueError(f"Recorded selection differs for {directory}")
    calls = read_lines(directory / "agent_mcp_transcript.jsonl")
    edits = trial["accepted_practice_edits"]
    evaluations = [row for row in calls if row["request"]["tool"] == "evaluate_practice"]
    if len(edits) != int(label is not None) or len(evaluations) != len(edits):
        raise ValueError(f"Selection exceeded one practice evaluation for {directory}")
    if label is not None:
        candidate = options[label]
        change = {"tensor": candidate["tensor"], "index": candidate["index"],
                  "op": "add", "value": candidate["delta"]}
        if edits[0]["patch"] != [change]:
            raise ValueError(f"Accepted patch differs from selected candidate for {directory}")
        practice_correct = evaluations[0]["response"]["result"]["result"]["correct"]
        if practice_correct != saved["baseline_correct"] + edits[0]["practice_correct_delta"]:
            raise ValueError(f"Practice checkpoint replay differs for {directory}")
        should_finalize = (practice_correct == candidate["practice_correct"]
                           and practice_correct > saved["baseline_correct"])
    else:
        practice_correct = None
        should_finalize = False
    if should_finalize != (trial["holdout"]["ai_candidate"] is not None):
        raise ValueError(f"Candidate finalization gate differs for {directory}")
    return {"menu": saved, "menu_sha256": _file_sha256(directory / "solver_menu.json"),
            "practice_correct": practice_correct, "practice_evaluations": len(evaluations),
            "holdout": trial["holdout"]}


def replay(root):
    plan = _read_json(root / "paired_plan.json")
    if plan["schema"] != "weightbridge.solver_selection_comparison.v1" or plan["practice_evaluation_cap"] != 1:
        raise ValueError("Unrecognized selection plan")
    runs, protocols, trials = {}, {}, {}
    for arm in SOLVER_ARMS:
        path = root / arm
        protocol = protocols[arm] = _read_json(path / "frozen_protocol.json")
        receipt = _read_json(path / "frozen_protocol.identity_receipt.json")
        if (plan["protocol_sha256"][arm] != protocol["protocol_sha256"]
                or receipt["strategy"] != arm or receipt["practice_evaluation_cap"] != 1):
            raise ValueError(f"Incorrect frozen protocol for {arm}")
        for key, source in (
            ("host_source_sha256", SOURCE),
            ("toy_model_source_sha256", ROOT / "weightbridge/diagnostics/toy_mlp.py"),
            ("breakpoint_source_sha256", ROOT / "weightbridge/diagnostics/breakpoint_search.py"),
            ("search_source_sha256", ROOT / "weightbridge/diagnostics/coordinate_search.py"),
        ):
            if receipt[key] != _file_sha256(source):
                raise ValueError(f"Incorrect frozen source for {arm}/{key}")
        run = runs[arm] = summarize_run(path, include_class_changes=True)
        missing = len(protocol["tasks"]) - receipt["planned_task_prefix"]
        if run["cohort_reasons"] != ([f"missing {missing} of {len(protocol['tasks'])} planned task records"] if missing else []):
            raise ValueError(f"Cohort evidence failed audit for {arm}")
        trials[arm] = {trial["task_id"]: trial for trial in run["trials"]}
    control = protocols["guided"]
    for key in ("tasks", "execution_order", "task_design", "host_policy", "prompt_sha256", "source_hashes"):
        if protocols["solver"][key] != control[key]:
            raise ValueError(f"Shared protocol setting differs for {key}")
    prefix = _read_json(root / "guided" / "frozen_protocol.identity_receipt.json")["planned_task_prefix"]
    order = control["execution_order"][:prefix]
    schedule = [{"arm": arm, "task": task} for i, task in enumerate(order)
                for arm in (SOLVER_ARMS if i % 2 == 0 else SOLVER_ARMS[::-1])]
    if schedule != plan["schedule"] or any(set(trials[arm]) != set(order) for arm in SOLVER_ARMS):
        raise ValueError("Incomplete or changed paired selection schedule")
    timestamps = [datetime.fromisoformat(_read_json(
        root / row["arm"] / row["task"] / "experiment_record.json")["timestamp_utc"]) for row in schedule]
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Execution order differs from the frozen schedule")
    pairs = []
    for task in order:
        check_splits(root / "guided" / task, root / "solver" / task)
        pair = {"task": task, "mechanism": trials["guided"][task]["mechanism"]}
        for arm in SOLVER_ARMS:
            pair[arm] = verify_trial(root / arm / task, arm, trials[arm][task], control["host_policy"])
        if pair["guided"]["menu"]["options"] != pair["solver"]["menu"]["options"]:
            raise ValueError(f"Arms received different menus for {task}")
        pairs.append(pair)
    gains = {}
    for arm in SOLVER_ARMS:
        gains[arm] = sum(
            (holdout := pair[arm]["holdout"])["ai_candidate"] is not None
            and holdout["ai_candidate"]["correct"] > holdout["baseline"]["correct"]
            for pair in pairs
        )
    return {
        "schema": "weightbridge.solver_selection_summary.v1", "disposition": "INCONCLUSIVE",
        "plan_sha256": _file_sha256(root / "paired_plan.json"), "verified_pairs": len(pairs),
        "tasks_with_finalized_holdout_gain": gains,
        "guided_tasks_with_multiple_options": sum(len(pair["guided"]["menu"]["options"]) > 1 for pair in pairs),
        "selections_differ": sum(pair["guided"]["menu"]["selected_label"] != pair["solver"]["menu"]["selected_label"]
                                 for pair in pairs),
        "pairs": pairs, "runs": runs, "host_resources": _read_json(root / "host_resources.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, replay(args.study))


if __name__ == "__main__":
    main()
