"""Summarize bounded local MCP trials without pooling separate protocols."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.evaluation.mcp_cohort import _file_sha256, _read_json, summarize_cohort
from weightbridge.diagnostics.toy_mlp import ToyMLP


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_scores(directory, record, calls):
    metrics = record["evaluation"]["metrics"]
    models = {
        "baseline": (directory / "base.npz", record["source_model"]["sha256"]),
        "random_control": (directory / "random_control.npz",
                           record["control_modification"]["candidate_sha256"]),
    }
    if metrics["ai_candidate"] is not None:
        final = calls[-1]["response"]["result"]
        candidate = directory / "candidates" / "sessions" / final["session_id"] / (
            f"candidate-{final['candidate_id']}.npz")
        models["ai_candidate"] = (candidate, final["candidate_sha256"])
    with np.load(directory / "hidden_holdout.npz", allow_pickle=False) as holdout:
        features, labels = holdout["features"], holdout["labels"]
    for key, (checkpoint, digest) in models.items():
        if _file_sha256(checkpoint) != digest:
            raise ValueError(f"Checkpoint hash mismatch for {directory.name}/{key}")
        correct = ToyMLP.load(checkpoint).forward(features).argmax(axis=1) == labels
        reported = metrics[key]
        if (correct.tolist() != reported["per_example_correct"]
                or correct.size != reported["n"] or int(correct.sum()) != reported["correct"]
                or float(correct.mean()) != reported["accuracy"]):
            raise ValueError(f"Holdout replay differs for {directory.name}/{key}")


def accepted_practice_edits(directory, record, calls, *, include_class_changes=False):
    accepted = [call for call in calls if call["request"].get("tool") == "submit_patch"
                and call["response"].get("ok") is True]
    if not accepted:
        return []
    session_id = record["protocol_binding"]["agent_session_id"]
    base = ToyMLP.load(directory / "base.npz")
    with np.load(directory / "practice.npz", allow_pickle=False) as practice:
        features = practice["features"]
        if include_class_changes:
            labels = practice["labels"]
    baseline_logits = base.forward(features)
    edits = []
    for call in accepted:
        receipt = call["response"]["result"]
        path = directory / "candidates" / "sessions" / session_id / (
            f"candidate-{receipt['candidate_id']}.npz")
        if _file_sha256(path) != receipt["candidate_sha256"]:
            raise ValueError(f"Practice candidate hash mismatch for {path.name}")
        candidate = ToyMLP.load(path)
        changed = sum(int(np.count_nonzero(weights != candidate.tensors[name]))
                      for name, weights in base.tensors.items())
        if changed != receipt["changed_entries"]:
            raise ValueError(f"Practice candidate change count mismatch for {path.name}")
        candidate_logits = candidate.forward(features)
        edit = {
            "candidate_sha256": receipt["candidate_sha256"], "changed_entries": changed,
            "patch": call["request"]["arguments"]["patch"]["changes"],
            "max_absolute_practice_logit_delta": float(
                np.max(np.abs(candidate_logits - baseline_logits))),
        }
        if include_class_changes:
            candidate_classes = candidate_logits.argmax(axis=1)
            baseline_classes = baseline_logits.argmax(axis=1)
            edit["practice_class_changes"] = int(np.count_nonzero(
                candidate_classes != baseline_classes))
            edit["practice_correct_delta"] = int(
                np.count_nonzero(candidate_classes == labels) - np.count_nonzero(baseline_classes == labels))
        edits.append(edit)
    return edits


def summarize_run(path, *, include_class_changes=False):
    protocol = _read_json(path / "frozen_protocol.json")
    record_paths = sorted(path.glob("trial-*/experiment_record.json"))
    records = [_read_json(record) for record in record_paths]
    cohort = summarize_cohort(protocol, records)
    if cohort != _read_json(path / "cohort_summary.json"):
        raise ValueError(f"Cohort replay differs from recorded summary for {path.name}")
    trials = []
    totals = Counter()
    for record_path, record in zip(record_paths, records):
        task_id = record["protocol_binding"]["task_id"]
        turns = read_lines(path / f"{task_id}.model_turns.jsonl")
        model_turns = [row for row in turns if "raw_output" in row]
        calls = read_lines(record_path.parent / "agent_mcp_transcript.jsonl")
        verify_scores(record_path.parent, record, calls)
        counts = Counter({
            "model_turns": len(model_turns),
            "model_input_tokens": sum(row["input_tokens"] for row in model_turns),
            "model_output_tokens": sum(row["output_tokens"] for row in model_turns),
            "model_token_cap_turns": sum(row["at_token_cap"] for row in model_turns),
            "repeated_model_outputs": len(model_turns) - len({row["raw_output"] for row in model_turns}),
            "transport_errors": sum("transport_error" in row for row in turns),
            "mcp_calls": len(calls),
            "mcp_rejections": sum(not call["response"].get("ok", False) for call in calls),
            "accepted_patches": sum(call["request"].get("tool") == "submit_patch"
                                    and call["response"].get("ok") is True for call in calls),
            "host_budget_stops": sum("host_stop" in row for row in turns),
        })
        metrics = record["evaluation"]["metrics"]
        candidate = metrics["ai_candidate"]
        counts["finalized_candidates"] = int(candidate is not None)
        counts["full_repairs"] = int(candidate is not None and candidate["accuracy"] == 1
                                    and metrics["baseline"]["accuracy"] < 1)
        totals.update(counts)
        trials.append({
            "task_id": task_id, "mechanism": record["protocol_binding"]["mechanism"],
            "seed": record["evaluation"]["seed"],
            "record": record_path.resolve().relative_to(ROOT).as_posix(),
            "record_sha256": _file_sha256(record_path),
            "holdout": {key: (None if result is None else {"correct": result["correct"], "n": result["n"]})
                        for key, result in metrics.items()},
            "random_control_match": record["control_modification"]["magnitude_match_type"],
            "accepted_practice_edits": accepted_practice_edits(
                record_path.parent, record, calls, include_class_changes=include_class_changes),
            "counts": dict(counts),
        })
    return {
        "cohort_id": protocol["cohort_id"], "protocol_sha256": protocol["protocol_sha256"],
        "cohort_disposition": cohort["disposition"],
        "cohort_reasons": cohort["reasons"],
        "holdout_replay_verified": True,
        "resources": _read_json(path / "host_resources.json"),
        "counts": dict(totals), "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = {"schema": "weightbridge.local_mcp_smoke_comparison.v1",
               "runs": [summarize_run(path) for path in args.runs]}
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
