"""Run a practice-only AI-directed patch search on the injected-defect MLP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightbridge.core import apply_patch, file_hash
from weightbridge.diagnostics import make_defect_benchmark, ToyMLP
from weightbridge.evaluation.metrics import paired_bootstrap_interval
from weightbridge.provenance import append_experiment, load_ledger, peak_rss_bytes


def evaluate(checkpoint: Path, features: np.ndarray, labels: np.ndarray) -> dict:
    model = ToyMLP.load(checkpoint)
    logits = model.forward(features)
    prediction = logits.argmax(axis=1)
    correct = prediction == labels
    return {
        "n": int(labels.size), "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "per_example_correct": correct.astype(bool).tolist(),
        "class_errors": {str(label): int(np.count_nonzero((labels == label) & ~correct)) for label in (0, 1)},
        "mean_output_norm": float(np.linalg.norm(logits, axis=1).mean()),
        "completion_rate": 1.0,
        "token_limit_rate": None,
        "reasoning_loop_rate": None,
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(output: Path, *, seed: int = 731, per_class: int = 32) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    benchmark = make_defect_benchmark(seed=seed, per_class=per_class)
    model: ToyMLP = benchmark["model"]
    practice = benchmark["practice"]
    holdout = benchmark["holdout"]
    base_path = output / "base.npz"
    model.save(base_path)
    base_hash = file_hash(base_path)
    measurement_key = hashlib.sha256(json.dumps({
        "dataset": benchmark["dataset"], "seed": seed, "per_class": per_class,
        "source_model_sha256": base_hash,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if any(item.get("measurement_key") == measurement_key for item in load_ledger(ROOT / "results" / "experiment_ledger.jsonl")):
        raise ValueError("This synthetic seed and source model are already recorded")
    practice_path = output / "agent_visible_practice.json"
    _write_json(practice_path, {"dataset": benchmark["dataset"], "seed": seed,
                                "features": practice["features"].tolist(),
                                "labels": practice["labels"].tolist()})
    # The worker receives a task file with practice data only. The holdout is
    # still only in this parent process and is not written until the worker exits.
    agent_dir = output / "agent_trial_candidates"
    agent_dir.mkdir()
    request = {
        "schema": "weightbridge.agent_task.v1", "base_path": str(base_path),
        "practice_features": practice["features"].tolist(), "practice_labels": practice["labels"].tolist(),
        "candidate_dir": str(agent_dir), "result_path": str(output / "agent_result.json"),
        "iteration_budget": 8, "allowed_tensors": ["output_bias"],
    }
    request_path = output / "agent_task_practice_only.json"
    _write_json(request_path, request)
    worker = subprocess.run([sys.executable, "-m", "weightbridge.agents.worker", "--request", str(request_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=300)
    if worker.returncode != 0:
        raise RuntimeError(f"Agent worker failed: {worker.stderr[-2000:]}")
    agent_result = json.loads((output / "agent_result.json").read_text(encoding="utf-8"))

    base_holdout = evaluate(base_path, holdout["features"], holdout["labels"])
    _write_json(output / "hidden_holdout.json", {
        "schema": "weightbridge.synthetic_holdout.v1", "seed": seed + 1,
        "features": holdout["features"].tolist(), "labels": holdout["labels"].tolist(),
        "note": "Never passed to the diagnostic agent; evaluated once after finalization.",
    })

    if agent_result["chosen_patch"] is None:
        ai_result = None
        ai_patch_hash = None
        ai_candidate_hash = None
    else:
        patch = agent_result["chosen_patch"]
        _write_json(output / "ai_patch.json", patch)
        ai_candidate_path = output / "ai_candidate.npz"
        ai_receipt = apply_patch(base_path, patch,
                                {"schema": "weightbridge.policy.v1", "allow_tensors": ["output_bias"],
                                 "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 5.0},
                                ai_candidate_path)
        ai_result = evaluate(ai_candidate_path, holdout["features"], holdout["labels"])
        ai_patch_hash = ai_receipt["patch_sha256"]
        ai_candidate_hash = ai_receipt["candidate_sha256"]
        _write_json(output / "ai_receipt.json", ai_receipt)

    # One seeded random sign and one randomly selected output-bias coordinate,
    # matched to the AI edit in parameter count and absolute magnitude.
    rng = np.random.default_rng(20261102)
    control_coordinate = int(rng.integers(0, 2))
    control_sign = int(rng.choice(np.array([-1, 1])))
    control_delta = float(4 * control_sign)
    control_patch = {
        "schema": "weightbridge.patch.v1", "base_sha256": base_hash,
        "reason": "seeded random one-entry control matched by tensor, count and edit magnitude",
        "evidence_ids": ["random-control-seed-20261102"],
        "changes": [{"tensor": "output_bias", "index": [control_coordinate],
                     "op": "add", "value": control_delta}], "projections": [],
    }
    control_path = output / "random_control.npz"
    control_receipt = apply_patch(base_path, control_patch,
                                  {"schema": "weightbridge.policy.v1", "allow_tensors": ["output_bias"],
                                   "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 5.0},
                                  control_path)
    control_result = evaluate(control_path, holdout["features"], holdout["labels"])
    _write_json(output / "random_control_patch.json", control_patch)

    # Persist the holdout-scoring receipt after candidate identity is fixed.
    raw_results = {"baseline": base_holdout, "ai_candidate": ai_result, "random_control": control_result}
    _write_json(output / "evaluation.json", raw_results)
    paired_ci = None
    if ai_result is not None:
        base_correct = ToyMLP.load(base_path).forward(holdout["features"]).argmax(axis=1) == holdout["labels"]
        ai_correct = ToyMLP.load(output / "ai_candidate.npz").forward(holdout["features"]).argmax(axis=1) == holdout["labels"]
        paired_ci = paired_bootstrap_interval(base_correct.astype(float), ai_correct.astype(float), seed=seed)

    experiment_id = f"toy-discovery-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    worker_resources = json.loads((output / "agent_result.resources.json").read_text(encoding="utf-8"))
    wall = time.perf_counter() - started_wall
    cpu = time.process_time() - started_cpu + float(worker_resources["cpu_seconds"])
    record = {
        "schema": "weightbridge.experiment.v1", "experiment_id": experiment_id,
        "measurement_key": measurement_key, "replicate_role": "single_seed_discovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": agent_result["hypothesis"],
        "source_model": {"id": "synthetic-toy-mlp-v1", "sha256": base_hash,
                         "tensor_sha256": _tensor_hashes(model)},
        "proposed_modification": agent_result["chosen_patch"],
        "affected_parameters": {"tensor": "output_bias", "count": 1,
                                 "fraction": 1 / sum(value.size for value in model.tensors.values()),
                                 "candidate_tensor_index": [1] if agent_result["chosen_patch"] else None},
        "control_modification": {"tensor": "output_bias", "index": [control_coordinate],
                                 "delta": control_delta, "parameter_count": 1,
                                 "candidate_sha256": control_receipt["candidate_sha256"]},
        "evaluation": {"dataset": benchmark["dataset"], "version": "1", "seed": seed,
                       "practice_n": int(len(practice["labels"])), "holdout_n": int(len(holdout["labels"])),
                       "practice_test_separated": True, "holdout_accessed_by_agent": False,
                       "metrics": raw_results, "paired_accuracy_difference_ci": paired_ci},
        "resources": {"wall_seconds": wall, "cpu_seconds": cpu,
                      "peak_rss_bytes": max((value for value in (peak_rss_bytes(), worker_resources["peak_rss_bytes"])
                                              if value is not None), default=None),
                      "peak_vram_bytes": None, "inference_tokens": None, "ai_tokens": 0,
                      "python": platform.python_version(), "platform": platform.platform()},
        "agent": {"type": "deterministic_diagnostic_search_algorithm_not_llm",
                  **agent_result},
        "artifact_root": output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output),
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()) + ["experiment_record.json"],
        "disposition": "PASS" if ai_result and paired_ci and paired_ci["ci_low"] > 0 else "INCONCLUSIVE",
        "scientific_scope": "one synthetic injected-bias task; no claim of general learning or intelligence",
    }
    _write_json(output / "experiment_record.json", record)
    append_experiment(ROOT / "results" / "experiment_ledger.jsonl", record)
    return record


def _tensor_hashes(model: ToyMLP) -> dict[str, str]:
    return {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in model.tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--per-class", type=int, default=32)
    args = parser.parse_args()
    target = Path(args.out) if args.out else ROOT / "results" / "original_weight_edits" / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = run(target, seed=args.seed, per_class=args.per_class)
    print(json.dumps({"experiment_id": result["experiment_id"], "disposition": result["disposition"],
                      "artifacts": result["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
