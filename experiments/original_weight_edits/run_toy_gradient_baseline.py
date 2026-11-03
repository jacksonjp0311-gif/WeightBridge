"""Conventional full-batch SGD baseline on the synthetic MLP only.

This explicitly gradient-trained baseline is separate from WeightBridge's
gradient-free candidate search. It trains on practice rows only and evaluates
once on a disjoint holdout after the candidate is frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from weightbridge.core import file_hash
from weightbridge.diagnostics import ToyMLP, make_defect_benchmark
from weightbridge.evaluation.metrics import paired_bootstrap_interval
from weightbridge.provenance import append_experiment, load_ledger, peak_rss_bytes


def _loss_and_gradients(model: ToyMLP, features: np.ndarray, labels: np.ndarray):
    logits, activations = model.forward(features, return_activations=True)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    rows = np.arange(len(labels))
    loss = float(-np.log(np.maximum(probs[rows, labels], 1e-15)).mean())
    dlogits = probs
    dlogits[rows, labels] -= 1
    dlogits /= len(labels)
    hidden = activations["hidden"]
    output_weight = model.tensors["output_weight"]
    dhidden = dlogits @ output_weight.T
    dpreactivation = dhidden * (1 - hidden * hidden)
    gradients = {
        "input_weight": features.T @ dpreactivation,
        "hidden_bias": dpreactivation.sum(axis=0),
        "output_weight": hidden.T @ dlogits,
        "output_bias": dlogits.sum(axis=0),
    }
    return loss, gradients


def train(model: ToyMLP, features: np.ndarray, labels: np.ndarray, *,
          steps: int = 400, learning_rate: float = 0.05) -> dict:
    if type(steps) is not int or not 1 <= steps <= 10_000 or not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("invalid SGD steps or learning rate")
    loss_start, _ = _loss_and_gradients(model, features, labels)
    for _ in range(steps):
        _, gradients = _loss_and_gradients(model, features, labels)
        for name, gradient in gradients.items():
            model.tensors[name] -= learning_rate * gradient
    loss_end, _ = _loss_and_gradients(model, features, labels)
    return {"optimizer": "full_batch_SGD", "objective": "cross_entropy", "steps": steps,
            "learning_rate": learning_rate, "practice_loss_start": loss_start,
            "practice_loss_end": loss_end}


def _accuracy(model_path: Path, features: np.ndarray, labels: np.ndarray) -> dict:
    logits = ToyMLP.load(model_path).forward(features)
    predicted = logits.argmax(axis=1)
    correct = predicted == labels
    return {"n": int(len(labels)), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "per_example_correct": correct.astype(bool).tolist(), "checkpoint_sha256": file_hash(model_path),
            "candidate_reloaded_from_file": True}


def _save_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(output: Path, *, seeds: tuple[int, ...] = (1201, 1202, 1203, 1204, 1205),
        steps: int = 400, learning_rate: float = 0.05) -> dict:
    if not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain 1..32 unique values")
    measurement_key = hashlib.sha256(json.dumps({"suite": "synthetic-separated-threshold-v1",
                                                  "seeds": sorted(seeds), "optimizer": "full_batch_SGD",
                                                  "steps": steps, "learning_rate": learning_rate},
                                                 sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if any(item.get("measurement_key") == measurement_key for item in
           load_ledger(ROOT / "results" / "experiment_ledger.jsonl")):
        raise ValueError("This gradient baseline seed set and training configuration are already recorded")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    replicates = []
    base_scores, trained_scores = [], []
    for index, seed in enumerate(seeds):
        folder = output / f"replicate-{index:02d}-seed-{seed}"
        folder.mkdir()
        benchmark = make_defect_benchmark(seed=seed, per_class=32)
        base = benchmark["model"]
        base_path = folder / "base.npz"
        base.save(base_path)
        train_path = folder / "gradient_candidate.npz"
        trained = ToyMLP.load(base_path)
        config = train(trained, benchmark["practice"]["features"], benchmark["practice"]["labels"],
                       steps=steps, learning_rate=learning_rate)
        trained.save(train_path)
        base_result = _accuracy(base_path, benchmark["holdout"]["features"], benchmark["holdout"]["labels"])
        trained_result = _accuracy(train_path, benchmark["holdout"]["features"], benchmark["holdout"]["labels"])
        base_scores.extend(base_result["per_example_correct"])
        trained_scores.extend(trained_result["per_example_correct"])
        holdout_path = folder / "holdout_after_freeze.json"
        _save_json(holdout_path, {"seed": seed, "features": benchmark["holdout"]["features"].tolist(),
                                  "labels": benchmark["holdout"]["labels"].tolist(),
                                  "note": "Created after SGD candidate was frozen; never used for updates."})
        _save_json(folder / "evaluation.json", {"baseline": base_result, "gradient_candidate": trained_result})
        changed = sum(int(np.count_nonzero(base.tensors[name] != trained.tensors[name])) for name in base.tensor_names)
        total = sum(value.size for value in base.tensors.values())
        replicates.append({"seed": seed, "practice_n": int(len(benchmark["practice"]["labels"])),
                           "holdout_n": int(len(benchmark["holdout"]["labels"])), "training": config,
                           "baseline": {k: v for k, v in base_result.items() if k != "per_example_correct"},
                           "gradient_candidate": {k: v for k, v in trained_result.items() if k != "per_example_correct"},
                           "changed_parameter_count": changed, "parameter_fraction_changed": changed / total,
                           "artifacts": [f"{folder.name}/base.npz", f"{folder.name}/gradient_candidate.npz",
                                         f"{folder.name}/evaluation.json", f"{folder.name}/holdout_after_freeze.json"]})
    paired = paired_bootstrap_interval(np.asarray(base_scores, dtype=float),
                                       np.asarray(trained_scores, dtype=float), seed=120501)
    experiment_id = f"toy-gradient-baseline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    record = {"schema": "weightbridge.experiment.v1", "experiment_id": experiment_id,
              "measurement_key": measurement_key,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "hypothesis": "Full-batch gradient descent on labeled practice rows can correct the injected output-bias defect, at a distinct training and parameter-change cost.",
              "source_model": {"id": "synthetic-toy-mlp-v1", "sha256": replicates[0]["baseline"]["checkpoint_sha256"]},
              "proposed_modification": {"type": "conventional_gradient_training", "optimizer": "full_batch_SGD",
                                        "objective": "cross_entropy", "steps": steps, "learning_rate": learning_rate},
              "affected_parameters": {"parameter_counts_by_seed": [item["changed_parameter_count"] for item in replicates],
                                      "fractions_by_seed": [item["parameter_fraction_changed"] for item in replicates]},
              "evaluation": {"dataset": "synthetic-separated-threshold-v1", "seeds": list(seeds),
                             "practice_test_separated": True, "holdout_accessed_during_training": False,
                             "baseline_accuracy": float(np.mean(base_scores)),
                             "candidate_accuracy": float(np.mean(trained_scores)),
                             "paired_difference_ci95": paired, "replicates": replicates},
              "resources": {"wall_seconds": time.perf_counter() - wall_start,
                            "cpu_seconds": time.process_time() - cpu_start,
                            "peak_rss_bytes": peak_rss_bytes(), "peak_vram_bytes": None,
                            "ai_tokens": 0, "inference_tokens": 0,
                            "python": platform.python_version(), "platform": platform.platform()},
              "artifact_root": output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output),
              "artifacts": [path for item in replicates for path in item["artifacts"]] + ["experiment_record.json"],
              "disposition": "PASS" if paired["ci_low"] > 0 else "INCONCLUSIVE",
              "scientific_scope": "synthetic optimizer baseline only; not a Qwen or language-model fine-tuning comparison"}
    _save_json(output / "experiment_record.json", record)
    append_experiment(ROOT / "results" / "experiment_ledger.jsonl", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1201, 1202, 1203, 1204, 1205])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()
    target = args.out or ROOT / "results" / "original_weight_edits" / f"gradient-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = run(target, seeds=tuple(args.seeds), steps=args.steps, learning_rate=args.learning_rate)
    print(json.dumps({"experiment_id": result["experiment_id"], "disposition": result["disposition"],
                      "baseline_accuracy": result["evaluation"]["baseline_accuracy"],
                      "candidate_accuracy": result["evaluation"]["candidate_accuracy"],
                      "paired_difference_ci95": result["evaluation"]["paired_difference_ci95"]}, indent=2))


if __name__ == "__main__":
    main()
