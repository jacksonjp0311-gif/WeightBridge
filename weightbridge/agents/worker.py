"""Isolated process worker: receives only practice data, never holdout data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from .diagnostic import DiagnosticAgent
from ..diagnostics.toy_mlp import ToyMLP
from ..provenance.resources import peak_rss_bytes


def evaluate(path: Path, features: np.ndarray, labels: np.ndarray) -> dict:
    logits = ToyMLP.load(path).forward(features)
    predictions = logits.argmax(axis=1)
    correct = predictions == labels
    return {"n": int(labels.size), "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "class_errors": {str(label): int(np.count_nonzero((labels == label) & ~correct)) for label in (0, 1)}}


def run(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    required = {"schema", "base_path", "practice_features", "practice_labels", "candidate_dir",
                "result_path", "iteration_budget", "allowed_tensors"}
    optional = {"candidate_deltas", "memory_context", "memory_evidence_ids"}
    if not required <= set(request) or set(request) - required - optional or request["schema"] != "weightbridge.agent_task.v1":
        raise ValueError("Invalid agent worker request")
    features = np.asarray(request["practice_features"], dtype=np.float64)
    labels = np.asarray(request["practice_labels"], dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),) or len(features) > 10000:
        raise ValueError("Invalid practice dataset shape or size")
    model = ToyMLP.load(request["base_path"])
    deltas = tuple(request.get("candidate_deltas", (-1, 1, -2, 2, -3, 3, -4, -5)))
    agent = DiagnosticAgent(iteration_budget=request["iteration_budget"], deltas=deltas,
                            memory_context=request.get("memory_context", ""),
                            memory_evidence_ids=tuple(request.get("memory_evidence_ids", [])))
    result = agent.run(
        base_path=request["base_path"], model=model,
        practice_features=features, practice_labels=labels,
        evaluate_practice=lambda path: evaluate(path, features, labels),
        candidate_dir=request["candidate_dir"], allowed_tensors=request["allowed_tensors"],
    )
    Path(request["result_path"]).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    resource = {"wall_seconds": None, "cpu_seconds": None, "peak_rss_bytes": peak_rss_bytes(),
                "peak_vram_bytes": None, "python": platform.python_version(),
                "platform": platform.platform(), "recorded_at": datetime.now(timezone.utc).isoformat()}
    resource["wall_seconds"] = time.perf_counter() - _START
    resource["cpu_seconds"] = time.process_time() - _CPU_START
    Path(request["result_path"]).with_suffix(".resources.json").write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_START = time.perf_counter()
_CPU_START = time.process_time()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    run(Path(args.request))


if __name__ == "__main__":
    main()
