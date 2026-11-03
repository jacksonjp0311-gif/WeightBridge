"""Run the four-condition synthetic Memory-to-Weights ablation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
from weightbridge.diagnostics import ToyMLP, make_defect_benchmark
from weightbridge.evaluation.metrics import paired_bootstrap_interval
from weightbridge.memory import ExperienceStore, MemoryToWeightsConsolidation
from weightbridge.provenance import append_experiment, load_ledger, peak_rss_bytes


WEIGHT_ONLY_DELTAS = [1, -1, 2, -2, 3, -3, -4, -5]


def evaluate_checkpoint(path: Path, features: np.ndarray, labels: np.ndarray,
                        *, decision_bias: tuple[int, float] | None = None) -> dict:
    logits = ToyMLP.load(path).forward(features)
    if decision_bias is not None:
        index, delta = decision_bias
        logits[:, index] += delta
    predicted = logits.argmax(axis=1)
    correct = predicted == labels
    return {"n": int(labels.size), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "per_example_correct": correct.astype(bool).tolist(),
            "class_errors": {str(label): int(np.count_nonzero((labels == label) & ~correct)) for label in (0, 1)},
            "completion_rate": 1.0, "checkpoint_sha256": file_hash(path),
            "candidate_reloaded_from_file": True}


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_agent(base: Path, practice: dict, run_dir: Path, *, name: str,
               deltas: list[float], memory_context: str = "",
               memory_evidence_ids: list[str] | None = None) -> dict:
    candidate_dir = run_dir / f"{name}_trial_candidates"
    candidate_dir.mkdir()
    result_path = run_dir / f"{name}_agent.json"
    request = {
        "schema": "weightbridge.agent_task.v1", "base_path": str(base),
        "practice_features": practice["features"].tolist(), "practice_labels": practice["labels"].tolist(),
        "candidate_dir": str(candidate_dir), "result_path": str(result_path),
        "iteration_budget": 8, "allowed_tensors": ["output_bias"],
        "candidate_deltas": deltas,
    }
    if memory_context:
        request["memory_context"] = memory_context
        request["memory_evidence_ids"] = memory_evidence_ids or []
    request_path = run_dir / f"{name}_practice_only_task.json"
    _write_json(request_path, request)
    proc_start = time.perf_counter()
    process = subprocess.run([sys.executable, "-m", "weightbridge.agents.worker", "--request", str(request_path)],
                             cwd=ROOT, capture_output=True, text=True, timeout=300)
    if process.returncode != 0:
        raise RuntimeError(f"{name} agent worker failed: {process.stderr[-2000:]}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    resources = json.loads(result_path.with_suffix(".resources.json").read_text(encoding="utf-8"))
    result["worker_elapsed_seconds"] = time.perf_counter() - proc_start
    result["worker_resources"] = resources
    return result


def _find_source_run(source_run: Path | None) -> Path:
    if source_run is not None:
        return source_run.resolve()
    candidates = []
    for folder in (ROOT / "results" / "original_weight_edits").glob("run-*"):
        record = folder / "experiment_record.json"
        if (folder / "ai_patch.json").is_file() and (folder / "ai_receipt.json").is_file() and record.is_file():
            parsed = json.loads(record.read_text(encoding="utf-8"))
            if parsed.get("disposition") == "PASS" and parsed.get("evaluation", {}).get("holdout_accessed_by_agent") is False:
                candidates.append((record.stat().st_mtime, folder))
    if not candidates:
        raise FileNotFoundError("No prior successful isolated synthetic experiment; pass --source-run")
    return max(candidates)[1]


def run(output: Path, *, source_run: Path | None = None,
        seeds: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005)) -> dict:
    if (not seeds or len(seeds) > 32 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("seeds must contain 1..32 unique nonnegative integers")
    output = output.resolve()
    source = _find_source_run(source_run)
    source_record = json.loads((source / "experiment_record.json").read_text(encoding="utf-8"))
    source_patch = json.loads((source / "ai_patch.json").read_text(encoding="utf-8"))
    source_receipt = json.loads((source / "ai_receipt.json").read_text(encoding="utf-8"))
    intervention = source_patch["changes"][0]
    if (source_record["disposition"] != "PASS" or source_record["source_model"]["sha256"] != source_patch["base_sha256"]
            or intervention["tensor"] != "output_bias" or intervention["index"] != [1]
            or intervention["op"] != "add"):
        raise ValueError("Source experience did not verify a compatible synthetic candidate")
    source_model_hash = source_record["source_model"]["sha256"]
    measurement_key = hashlib.sha256(json.dumps({"suite": "synthetic-separated-threshold-v1",
                                                  "source_model_sha256": source_model_hash,
                                                  "seeds": sorted(seeds)}, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()
    if any(item.get("measurement_key") == measurement_key for item in load_ledger(ROOT / "results" / "experiment_ledger.jsonl")):
        raise ValueError("This MWC seed set and source experience are already recorded")
    output.mkdir(parents=True, exist_ok=False)
    if source_receipt["candidate_sha256"] != file_hash(source / "ai_candidate.npz"):
        raise ValueError("Source candidate no longer matches its immutable patch receipt")

    store = ExperienceStore(output / "memory_store")
    consolidation = MemoryToWeightsConsolidation(store)
    source_experience_id = consolidation.consolidate(
        failure_fingerprint="toy-mlp-v1:input-x0-threshold-0.5:one-sided-output-errors",
        observation_id=source_record["experiment_id"],
        diagnosis="One-sided practice failures were corrected by a -4.0 output_bias[1] shift.",
        patch_sha256=source_receipt["patch_sha256"],
        intervention=intervention,
        source_model_sha256=source_model_hash,
        evaluation={"source_holdout_accuracy": source_record["evaluation"]["metrics"]["ai_candidate"]["accuracy"],
                    "source_holdout_n": source_record["evaluation"]["metrics"]["ai_candidate"]["n"],
                    "source_experiment_id": source_record["experiment_id"]},
    )
    matching = [item for item in store.retrieve("toy-mlp-v1:input-x0-threshold-0.5:one-sided-output-errors")
                if item.get("source_model_sha256") == source_model_hash]
    if len(matching) != 1 or matching[0]["intervention"] != intervention:
        raise ValueError("Memory retrieval failed exact fingerprint and model identity check")
    memory = matching[0]
    memory_record_hash = source_experience_id
    memory_delta = float(memory["intervention"]["value"])
    if abs(memory_delta) > 5:
        raise ValueError("Retrieved synthetic edit exceeds host policy")

    started = time.perf_counter()
    cpu_started = time.process_time()
    replicates = []
    paired_arrays: dict[str, list[float]] = {name: [] for name in
                                              ("no_memory_no_edit", "external_memory_only",
                                               "weight_edit_only", "memory_plus_weight_edit")}
    total_worker_cpu = 0.0
    all_worker_rss: list[int] = []
    for replicate_index, seed in enumerate(seeds):
        replicate = output / f"replicate-{replicate_index:02d}-seed-{seed}"
        replicate.mkdir()
        benchmark = make_defect_benchmark(seed=seed, per_class=32)
        model: ToyMLP = benchmark["model"]
        practice, holdout = benchmark["practice"], benchmark["holdout"]
        base = replicate / "base.npz"
        model.save(base)
        if file_hash(base) != source_model_hash:
            raise ValueError("Synthetic source weights drifted since stored experience")
        _write_json(replicate / "practice.json", {"features": practice["features"].tolist(),
                                                   "labels": practice["labels"].tolist(),
                                                   "note": "Only this file is sent to diagnostic workers."})

        # Agent worker processes see practice examples and the authorized prior
        # memory hint only. The holdout remains in this parent and is not written
        # until both workers exit and each candidate identity is fixed.
        weight_only = _run_agent(base, practice, replicate, name="weight_only",
                                 deltas=WEIGHT_ONLY_DELTAS)
        memory_context = (f"Verified record {memory_record_hash[:16]} for the same synthetic model hash "
                          f"found a one-entry intervention on output_bias[1], delta {memory_delta:+.1f}.")
        memory_weight = _run_agent(base, practice, replicate, name="memory_weight",
                                   deltas=[memory_delta, *WEIGHT_ONLY_DELTAS],
                                   memory_context=memory_context,
                                   memory_evidence_ids=[memory_record_hash])
        total_worker_cpu += float(weight_only["worker_resources"]["cpu_seconds"])
        total_worker_cpu += float(memory_weight["worker_resources"]["cpu_seconds"])
        all_worker_rss.extend(value for value in (weight_only["worker_resources"]["peak_rss_bytes"],
                                                  memory_weight["worker_resources"]["peak_rss_bytes"])
                              if value is not None)

        if not weight_only["chosen_patch"] or not memory_weight["chosen_patch"]:
            raise ValueError("At least one diagnostic condition failed to choose a practice-improving edit")
        weight_patch = weight_only["chosen_patch"]
        memory_patch = memory_weight["chosen_patch"]
        weight_candidate = replicate / "weight_only_candidate.npz"
        memory_candidate = replicate / "memory_weight_candidate.npz"
        policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["output_bias"],
                  "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 5.0}
        weight_receipt = apply_patch(base, weight_patch, policy, weight_candidate)
        memory_receipt = apply_patch(base, memory_patch, policy, memory_candidate)

        # Bounded holdout evaluations happen once, after both selected candidates
        # are frozen. Model loaders read candidate files from disk (restart test).
        raw_baseline = evaluate_checkpoint(base, holdout["features"], holdout["labels"])
        memory_only = evaluate_checkpoint(base, holdout["features"], holdout["labels"],
                                          decision_bias=(int(intervention["index"][0]), memory_delta))
        weight_result = evaluate_checkpoint(weight_candidate, holdout["features"], holdout["labels"])
        memory_weight_result = evaluate_checkpoint(memory_candidate, holdout["features"], holdout["labels"])
        memory_only["checkpoint_unchanged"] = memory_only["checkpoint_sha256"] == source_model_hash
        memory_only["intervention_location"] = "external decision-time logit bias; checkpoint unmodified"
        result_set = {"no_memory_no_edit": raw_baseline, "external_memory_only": memory_only,
                      "weight_edit_only": weight_result, "memory_plus_weight_edit": memory_weight_result}
        for name, result in result_set.items():
            paired_arrays[name].extend(float(value) for value in result["per_example_correct"])
        _write_json(replicate / "holdout.json", {"seed": seed, "features": holdout["features"].tolist(),
                                                  "labels": holdout["labels"].tolist(),
                                                  "note": "Written after both agents finalized; never sent to either worker."})
        _write_json(replicate / "selected_patches.json", {"weight_only_patch": weight_patch,
                                                            "weight_only_receipt": weight_receipt,
                                                            "memory_weight_patch": memory_patch,
                                                            "memory_weight_receipt": memory_receipt})
        _write_json(replicate / "evaluation.json", result_set)
        replicates.append({"seed": seed, "practice_n": int(len(practice["labels"])),
                           "holdout_n": int(len(holdout["labels"])), "holdout_agent_access": False,
                           "results": result_set,
                           "practice_evaluator_calls": {"weight_edit_only": weight_only["evaluator_calls"],
                                                        "memory_plus_weight_edit": memory_weight["evaluator_calls"],
                                                        "external_memory_only": 0},
                           "practice_forward_examples": {"weight_edit_only": weight_only["evaluator_calls"] * len(practice["labels"]),
                                                         "memory_plus_weight_edit": memory_weight["evaluator_calls"] * len(practice["labels"]),
                                                         "external_memory_only": 0},
                           "memory_record_id": memory_record_hash,
                           "agent_results": {"weight_edit_only": weight_only,
                                             "memory_plus_weight_edit": memory_weight},
                           "candidate_sha256": {"weight_edit_only": weight_receipt["candidate_sha256"],
                                                "memory_plus_weight_edit": memory_receipt["candidate_sha256"]}})

    overall: dict[str, Any] = {}
    baseline_vector = np.asarray(paired_arrays["no_memory_no_edit"], dtype=float)
    for name, values in paired_arrays.items():
        vector = np.asarray(values, dtype=float)
        overall[name] = {"n": int(vector.size), "correct": int(vector.sum()),
                         "accuracy": float(vector.mean()), "completion_rate": 1.0}
        if name != "no_memory_no_edit":
            overall[name]["paired_difference_vs_no_memory_no_edit"] = paired_bootstrap_interval(
                baseline_vector, vector, seed=77123)
    final_experiment_id = f"mwc-ablation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    cpu = time.process_time() - cpu_started + total_worker_cpu
    wall = time.perf_counter() - started
    summary = {"schema": "weightbridge.mwc_ablation.v1", "experiment_id": final_experiment_id,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "prior_verified_experience": {"observation_id": source_record["experiment_id"],
                                              "memory_record_id": memory_record_hash,
                                              "model_sha256": source_model_hash,
                                              "patch_sha256": source_receipt["patch_sha256"],
                                              "intervention": intervention},
               "conditions": overall, "replicates": replicates,
               "practice_holdout_separated": True, "holdout_accessed_by_agents": False,
               "policy": {"candidate_limit": 8, "max_entries": 1, "max_abs_delta": 5.0,
                          "dataset_seeds": list(seeds)},
               "resources": {"wall_seconds": wall, "cpu_seconds": cpu,
                             "worker_peak_rss_bytes": max(all_worker_rss, default=None),
                             "parent_peak_rss_bytes": peak_rss_bytes(),
                             "peak_vram_bytes": None, "agent_tokens": 0,
                             "practice_forward_examples": {name: sum(item["practice_forward_examples"].get(name, 0)
                                                                     for item in replicates)
                                                            for name in ("weight_edit_only", "memory_plus_weight_edit",
                                                                         "external_memory_only")},
                             "platform": platform.platform(), "python": platform.python_version()},
               "disposition": "PASS" if all(overall[name]["accuracy"] > 0.9
                                              for name in ("external_memory_only", "weight_edit_only",
                                                           "memory_plus_weight_edit")) else "INCONCLUSIVE",
               "limitations": ["Synthetic deterministic failure family only.",
                               "External-memory-only condition uses a retrieval-time output-logit rule, not a weight edit.",
                               "One source experience and five synthetic task seeds do not establish general MWC benefit.",
                               "No conventional optimizer baseline or natural-language-model memory ablation."],
               "artifacts": ["mwc_ablation.json", "memory_store", *[f"replicate-{i:02d}-seed-{seed}"
                                                                     for i, seed in enumerate(seeds)]]}
    _write_json(output / "mwc_ablation.json", summary)
    _write_json(output / "benchmark_summary.json", {"experiment_id": final_experiment_id,
                                                      "conditions": overall,
                                                      "raw_results": "mwc_ablation.json"})
    # Persist the newly verified memory-plus-weight outcome only after its
    # independent holdout score is known. It is not available to this run's agents.
    for index, replicate in enumerate(replicates):
        eval_m = replicate["results"]["memory_plus_weight_edit"]
        if eval_m["accuracy"] > replicate["results"]["no_memory_no_edit"]["accuracy"]:
            consolidation.consolidate(
                failure_fingerprint="toy-mlp-v1:input-x0-threshold-0.5:one-sided-output-errors",
                observation_id=f"{final_experiment_id}-replicate-{index}",
                diagnosis="Prior verified output-bias hint was validated against a fresh disjoint synthetic holdout.",
                patch_sha256=replicate["agent_results"]["memory_plus_weight_edit"]["candidate_trials"][0]["patch_sha256"],
                intervention=intervention,
                source_model_sha256=source_model_hash,
                evaluation={"holdout_accuracy": eval_m["accuracy"], "holdout_n": eval_m["n"],
                            "parent_experiment": final_experiment_id, "replicate": index},
            )
    ledger_record = {
        "schema": "weightbridge.experiment.v1", "experiment_id": final_experiment_id,
        "measurement_key": measurement_key,
        "timestamp_utc": summary["timestamp_utc"], "hypothesis": "A verified synthetic experience should reduce diagnostic evaluator calls while an external-memory-only retrieval rule and a weight edit may both correct the same sparse output-bias defect.",
        "source_model": {"id": "synthetic-toy-mlp-v1", "sha256": source_model_hash},
        "proposed_modification": intervention,
        "affected_parameters": {"count": 1, "fraction": 1 / 12, "tensor": "output_bias[1]"},
        "evaluation": {"suite": "synthetic-separated-threshold-v1", "seeds": list(seeds),
                       "practice_test_separated": True, "holdout_accessed_by_agent": False,
                       "conditions": overall},
        "resources": summary["resources"], "memory_source_experiment": source_record["experiment_id"],
        "artifact_root": output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output),
        "artifacts": summary["artifacts"], "disposition": summary["disposition"],
        "limitations": summary["limitations"],
    }
    _write_json(output / "experiment_record.json", ledger_record)
    append_experiment(ROOT / "results" / "experiment_ledger.jsonl", ledger_record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--source-run", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003, 1004, 1005])
    args = parser.parse_args()
    target = Path(args.out) if args.out else ROOT / "results" / "memory_ablation" / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    summary = run(target, source_run=Path(args.source_run) if args.source_run else None,
                  seeds=tuple(args.seeds))
    print(json.dumps({"experiment_id": summary["experiment_id"], "disposition": summary["disposition"],
                      "conditions": summary["conditions"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
