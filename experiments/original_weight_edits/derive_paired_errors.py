"""Derive paired rescue/regression counts from frozen toy artifacts.

This post-hoc audit reads the already frozen holdout and immutable checkpoint
files. It never feeds results back to the diagnostic worker and is recorded as
an analysis addendum, not an independent experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from weightbridge.core import file_hash
from weightbridge.diagnostics import ToyMLP
from weightbridge.provenance import append_experiment


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def derive(roots: list[Path], output: Path) -> dict:
    if not 2 <= len(roots) <= 32:
        raise ValueError("This derivation expects 2..32 isolated discovery runs")
    runs = []
    for root in roots:
        root = root.resolve()
        record = _read(root / "experiment_record.json")
        holdout_path = root / "hidden_holdout.json"
        holdout = _read(holdout_path)
        features = np.asarray(holdout["features"], dtype=np.float64)
        labels = np.asarray(holdout["labels"], dtype=np.int64)
        candidates = {
            "baseline": root / "base.npz",
            "ai_candidate": root / "ai_candidate.npz",
            "random_control": root / "random_control.npz",
        }
        correctness = {}
        for name, checkpoint in candidates.items():
            predicted = ToyMLP.load(checkpoint).forward(features).argmax(axis=1)
            correctness[name] = (predicted == labels).astype(bool)
        baseline = correctness["baseline"]
        metrics = {}
        for name, values in correctness.items():
            rescues = int(np.count_nonzero(~baseline & values))
            regressions = int(np.count_nonzero(baseline & ~values))
            baseline_correct = int(np.count_nonzero(baseline))
            metrics[name] = {
                "accuracy": float(values.mean()), "correct": int(values.sum()), "n": int(values.size),
                "rescues_vs_baseline": rescues, "regressions_vs_baseline": regressions,
                "regression_rate_among_baseline_correct": (regressions / baseline_correct if baseline_correct else None),
                "per_example_correct": values.tolist(),
                "checkpoint_sha256": file_hash(candidates[name]),
            }
        runs.append({
            "experiment_id": record["experiment_id"],
            "task_seed": record["evaluation"].get("seed", record["evaluation"]["paired_accuracy_difference_ci"]["seed"]),
            "holdout_sha256": file_hash(holdout_path),
            "holdout_n": int(labels.size),
            "metrics": metrics,
            "source_artifacts": {name: str(path.relative_to(ROOT)) for name, path in candidates.items()} | {
                "hidden_holdout": str(holdout_path.relative_to(ROOT)),
            },
        })
    output = output.resolve()
    report = {
        "schema": "weightbridge.derived_paired_correctness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "independent_measurements_added": 0,
        "method": "Deterministically reload each frozen NPZ and score the already evaluated hidden holdout; no patch selection or agent interaction occurs.",
        "runs": runs,
    }
    _save(output, report)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    correction = {
        "schema": "weightbridge.ledger_correction.v1",
        "experiment_id": "toy-discovery-paired-correctness-" + output.stem.replace("_", "-") + "-20261103",
        "disposition": "METRIC_DERIVATION_ADDED_NOT_INDEPENDENT",
        "referenced_experiment_ids": [item["experiment_id"] for item in runs],
        "observation": "Added paired per-example correctness and exact rescue/regression counts reconstructed from frozen model and holdout artifacts; this is not an additional independent experiment.",
        "artifact": str(output.relative_to(ROOT)),
        "artifact_sha256": digest,
        "metrics": {item["experiment_id"]: {
            name: {key: value for key, value in result.items() if key != "per_example_correct"}
            for name, result in item["metrics"].items()
        } for item in runs},
    }
    append_experiment(ROOT / "results" / "experiment_ledger.jsonl", correction)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True,
                        help="two or more frozen synthetic run directories (maximum 32)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.roots, args.out)
    print(json.dumps({"schema": result["schema"], "runs": len(result["runs"]),
                      "independent_measurements_added": result["independent_measurements_added"]}, indent=2))


if __name__ == "__main__":
    main()
