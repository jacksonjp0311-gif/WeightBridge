"""Replay the paired advice, rejection and coordinate search comparison."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from summarize_local_mcp_runs import ROOT, ToyMLP, _file_sha256, _read_json, read_lines, summarize_run
from run_local_mcp_cohort import SEARCH_ARMS, SOURCE, diagnostic_view, form_request, write_json
from weightbridge.diagnostics.coordinate_search import disconnected_coordinate, stratified_proposals


def check_splits(reference, target):
    if _file_sha256(reference / "base.npz") != _file_sha256(target / "base.npz"):
        raise ValueError(f"Base checkpoint differs for {target}")
    for name in ("practice.npz", "hidden_holdout.npz"):
        with np.load(reference / name, allow_pickle=False) as a, np.load(target / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError(f"Split arrays differ for {target}/{name}")


def audit_search(calls, evaluations, view, policy, seed):
    submissions = [row["request"]["arguments"]["patch"]["changes"][0] for row in calls
                   if row["request"].get("tool") == "submit_patch"]
    planned = list(stratified_proposals(
        view["tensors"], view["logit_connectivity"]["single_edits_with_no_logit_path"], policy, seed))
    expected = [{"tensor": patch["tensor"], "index": patch["index"],
                 "op": "add", "value": patch["delta"]} for patch in planned]
    if submissions != expected or len(evaluations) != len(planned):
        raise ValueError("Search did not execute the frozen proposal schedule")
    best = max(evaluations, key=lambda row: row["response"]["result"]["result"]["correct"])
    final = calls[-1]["request"]
    if best["response"]["result"]["result"]["correct"] > view["practice"]["correct"]:
        if (final["tool"] != "finalize_candidate" or
                final["arguments"]["candidate_id"] != best["request"]["arguments"]["candidate_id"]):
            raise ValueError("Search did not finalize the first best practice improvement")
    elif final["tool"] != "stop_without_candidate":
        raise ValueError("Search finalized without a practice improvement")


def trial_metrics(arm, directory, trial, policy, budget):
    calls = read_lines(directory / "agent_mcp_transcript.jsonl")
    turn_path = directory.parent / f"{directory.name}.model_turns.jsonl"
    diagnostics_path = directory / "diagnostics_sent.json"
    turns = read_lines(turn_path)
    checkpoint = ToyMLP.load(directory / "base.npz")
    view = diagnostic_view(calls[0]["response"]["result"])
    view["logit_connectivity"] = checkpoint.logit_connectivity()
    if view != _read_json(diagnostics_path):
        raise ValueError(f"Diagnostic view differs for {directory}")
    blocked = view["logit_connectivity"]["single_edits_with_no_logit_path"]
    evaluations = [row for row in calls if row["request"].get("tool") == "evaluate_practice"]
    if len(evaluations) > budget or not all(row["response"]["ok"] for row in evaluations):
        raise ValueError(f"Practice evaluation budget or validity check failed for {directory}")
    rejections = [event for event in turns if "host_rejection" in event]
    if any(arm != "reject" or not disconnected_coordinate(blocked, event["tensor"], event["index"])
           for event in rejections):
        raise ValueError(f"Unexpected host rejection for {directory}")
    model_outputs = {event["turn"]: event["raw_output"] for event in turns if "raw_output" in event}
    for event in rejections:
        rejected = form_request(model_outputs[event["turn"]])
        if (rejected["tool"] != "submit_patch"
                or any(event[key] != rejected["arguments"][key] for key in ("tensor", "index"))):
            raise ValueError(f"Rejection does not match the model proposal for {directory}")
    edits = trial["accepted_practice_edits"]
    replayed_correct = {edit["candidate_sha256"]: view["practice"]["correct"] + edit["practice_correct_delta"]
                        for edit in edits}
    for evaluation in evaluations:
        result = evaluation["response"]["result"]
        if replayed_correct.get(result["candidate_sha256"]) != result["result"]["correct"]:
            raise ValueError(f"Practice score differs from checkpoint replay for {directory}")
    disconnected = sum(any(disconnected_coordinate(blocked, patch["tensor"], patch["index"])
                           for patch in edit["patch"]) for edit in edits)
    if arm in ("reject", "search") and disconnected:
        raise ValueError(f"Disconnected proposal passed the gate for {directory}")
    if arm == "search":
        audit_search(calls, evaluations, view, policy, trial["seed"])
        attempts = sum("search_proposal" in event for event in turns)
        if attempts != budget:
            raise ValueError("Search did not use its frozen attempt count")
    else:
        attempts = sum("raw_output" in event for event in turns)
    if attempts > budget:
        raise ValueError(f"Attempt budget exceeded for {directory}")
    parsed = 0
    for event in turns:
        if "raw_output" not in event:
            continue
        try:
            parsed += form_request(event["raw_output"])["tool"] == "submit_patch"
        except ValueError:
            pass
    holdout = trial["holdout"]
    broker = _read_json(directory / "experiment_record.json")["resources"]
    return {
        "model_turn_log_sha256": _file_sha256(turn_path),
        "diagnostics_sent_sha256": _file_sha256(diagnostics_path),
        "decision_attempts": attempts,
        "parsed_patch_attempts": attempts if arm == "search" else parsed,
        "host_rejections": len(rejections),
        "practice_evaluations": len(evaluations),
        "practice_correct_in_evaluation_order": [
            row["response"]["result"]["result"]["correct"] for row in evaluations
        ],
        "accepted_submissions": len(edits),
        "disconnected_submissions": disconnected,
        "any_practice_logit_change": any(edit["max_absolute_practice_logit_delta"] > 1e-9 for edit in edits),
        "any_practice_prediction_change": any(edit["practice_class_changes"] > 0 for edit in edits),
        "any_practice_accuracy_gain": any(edit["practice_correct_delta"] > 0 for edit in edits),
        "finalized_candidate": holdout["ai_candidate"] is not None,
        "finalized_holdout_correct_delta": (
            None if holdout["ai_candidate"] is None else
            holdout["ai_candidate"]["correct"] - holdout["baseline"]["correct"]),
        "full_holdout_repair": bool(trial["counts"]["full_repairs"]),
        "broker_cpu_seconds": broker["cpu_seconds"],
    }, view


def replay(root):
    plan = _read_json(root / "paired_plan.json")
    budget = plan["max_practice_evaluations"]
    runs, protocols, identities = {}, {}, {}
    for arm in SEARCH_ARMS:
        path = root / arm
        protocol = protocols[arm] = _read_json(path / "frozen_protocol.json")
        if protocol["protocol_sha256"] != plan["protocol_sha256"][arm]:
            raise ValueError(f"Protocol differs from plan for {arm}")
        receipt = _read_json(path / "frozen_protocol.identity_receipt.json")
        for field, source in (
            ("host_source_sha256", SOURCE),
            ("toy_model_source_sha256", ROOT / "weightbridge/diagnostics/toy_mlp.py"),
            ("search_source_sha256", ROOT / "weightbridge/diagnostics/coordinate_search.py"),
        ):
            if receipt[field] != _file_sha256(source):
                raise ValueError(f"Source differs from frozen receipt for {arm}/{field}")
        if receipt.pop("strategy") != arm or receipt.get("connectivity_probe") is not True:
            raise ValueError(f"Incorrect arm receipt for {arm}")
        receipt.pop("host_call_id")
        identities[arm] = receipt
        run = runs[arm] = summarize_run(path, include_class_changes=True)
        missing = len(protocol["tasks"]) - receipt["planned_task_prefix"]
        if run["cohort_reasons"] != ([f"missing {missing} of {len(protocol['tasks'])} planned task records"] if missing else []):
            raise ValueError(f"Cohort audit failed for {arm}")
    if identities["advice"] != identities["reject"]:
        raise ValueError("The AI arms used different models or settings")
    if (identities["search"]["provider"] != "weightbridge-algorithm"
            or identities["search"]["sampling"]["practice_evaluation_cap"] != budget
            or identities["advice"]["sampling"]["max_model_turns"] != budget):
        raise ValueError("Search or AI budget differs from the plan")
    control = protocols["advice"]
    for arm in SEARCH_ARMS[1:]:
        for key in ("tasks", "execution_order", "task_design", "host_policy", "prompt_sha256", "source_hashes"):
            if protocols[arm][key] != control[key]:
                raise ValueError(f"Shared protocol setting differs for {arm}/{key}")
    order = control["execution_order"][:identities["advice"]["planned_task_prefix"]]
    schedule = [{"task": task, "arm": arm} for i, task in enumerate(order)
                for arm in SEARCH_ARMS[i % 3:] + SEARCH_ARMS[:i % 3]]
    if schedule != plan["schedule"]:
        raise ValueError("Schedule differs from frozen plan")
    timestamps = [datetime.fromisoformat(_read_json(
        root / step["arm"] / step["task"] / "experiment_record.json")["timestamp_utc"])
        for step in schedule]
    if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Trial order differs from the frozen schedule")
    trials = {arm: {trial["task_id"]: trial for trial in run["trials"]}
              for arm, run in runs.items()}
    if any(set(items) != set(order) for items in trials.values()):
        raise ValueError("One or more paired tasks are missing")
    pairs = []
    for task in order:
        pair = {"task_id": task, "mechanism": trials["advice"][task]["mechanism"]}
        expected_view = None
        for arm in SEARCH_ARMS:
            directory = root / arm / task
            if arm != "advice":
                check_splits(root / "advice" / task, directory)
            pair[arm], view = trial_metrics(arm, directory, trials[arm][task], control["host_policy"], budget)
            if arm == "advice":
                expected_view = view
            elif view != expected_view:
                raise ValueError(f"Practice evidence differs for {task}/{arm}")
        count = pair["reject"]["practice_evaluations"]
        search_prefix = pair["search"]["practice_correct_in_evaluation_order"][:count]
        baseline_correct = expected_view["practice"]["correct"]
        pair["search_prefix_matched_to_reject_evaluations"] = {
            "evaluations": len(search_prefix),
            "any_practice_accuracy_gain": any(correct > baseline_correct for correct in search_prefix),
        }
        pairs.append(pair)
    gains = {arm: sum(pair[arm]["any_practice_accuracy_gain"] for pair in pairs) for arm in SEARCH_ARMS}
    holdout_gains = {
        arm: sum((pair[arm]["finalized_holdout_correct_delta"] or 0) > 0 for pair in pairs)
        for arm in SEARCH_ARMS
    }
    return {
        "schema": "weightbridge.search_comparison_summary.v1",
        "disposition": "INCONCLUSIVE", "plan_sha256": _file_sha256(root / "paired_plan.json"),
        "verified_pairs": len(pairs), "tasks_with_practice_accuracy_gain": gains,
        "tasks_with_finalized_holdout_gain": holdout_gains,
        "search_tasks_with_practice_gain_at_reject_evaluation_count": sum(
            pair["search_prefix_matched_to_reject_evaluations"]["any_practice_accuracy_gain"] for pair in pairs),
        "rejection_beat_both_on_primary": gains["reject"] > max(gains["advice"], gains["search"]),
        "pairs": pairs, "runs": runs, "host_resources": _read_json(root / "host_resources.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.out, replay(args.study))


if __name__ == "__main__":
    main()
