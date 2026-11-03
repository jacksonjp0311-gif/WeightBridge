"""Replay Phase III-I fork diagnostics on the unchanged SmolLM2 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import endpoint
from experiments.phase_iiii.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import compare_endpoint, measure_endpoint
from weightbridge.adaptation.generative_fork import (
    diagnose_fork, next_token_logits, protected_trajectory, trajectory_diagnostics,
    trajectory_logits, with_fork_logits,
)
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes


ROOT = Path(__file__).resolve().parents[2]
MATERIAL = ROOT / "results/phase-iii-i/causal-fork-v1/material"
ORIGINAL_SMOLLM2 = "4e4984f36a1a8566aceb2d0baf01e267042dc34d087fc4609f64b80de249624a"


def run(base: Path, material: Path, out: Path) -> dict:
    import torch

    if out.exists():
        raise FileExistsError(out)
    started, cpu = time.perf_counter(), time.process_time()
    config = json.loads(PROTOCOL.read_bytes())
    manifest = read_manifest(material)
    practice = load_split(material, "practice", manifest)
    contract = endpoint(config)
    with _open(base, config, ORIGINAL_SMOLLM2) as (adapter, runner, capabilities):
        measured = measure_endpoint(runner, practice, contract)
        observed = measured["rows"]
        # A fixed roster order determines the sample independently of replay agreement.
        wrong = [row for row in observed if row["group"] == "target" and row["status"] == "incorrect"][:4]
        correct = [row for row in observed if row["correct"]][:2]
        forks = [diagnose_fork(runner, row, contract, checkpoint_sha256=ORIGINAL_SMOLLM2,
                               model_id="SmolLM2-135M-Instruct") for row in wrong]
        actionable = [fork for fork in forks if fork.status == "semantic"]
        protected = [protected_trajectory(runner, row, contract) for row in correct]
        replayed_forks, trajectory_reports = {}, []
        with torch.inference_mode():
            if actionable:
                logits = next_token_logits(runner, [fork.prefix_ids for fork in actionable])
                replayed_forks = {fork.prompt_id: with_fork_logits(fork, vector, runner.tokenizer)
                                  for fork, vector in zip(actionable, logits, strict=True)}
            if protected:
                logits = trajectory_logits(runner, protected)
                trajectory_reports = [trajectory_diagnostics(path, vectors, runner.tokenizer)
                                      for path, vectors in zip(protected, logits, strict=True)]
        if adapter.checkpoint.checkpoint_sha256() != ORIGINAL_SMOLLM2:
            raise ValueError("SmolLM2 source checkpoint changed during portability diagnostics")
    compared = compare_endpoint(measured, measured, contract)
    forks = [replayed_forks.get(fork.prompt_id, fork) for fork in forks]
    report = {
        "schema": "weightbridge.phase_iiii.cross_model.v1",
        "engineering_disposition": "PASS" if len(actionable) >= 2 else "HOLD",
        "behavioral_adaptation_disposition": "NOT_TESTED",
        "checkpoint_sha256": ORIGINAL_SMOLLM2,
        "protocol_sha256": file_hash(PROTOCOL),
        "practice_sha256": manifest["splits"]["practice"]["sha256"],
        "source_sha256": {path: file_hash(ROOT / path) for path in (
            "experiments/phase_iiii/cross_model.py", "weightbridge/adaptation/generative_fork.py",
            "weightbridge/adaptation/behavior_endpoint.py", "weightbridge/adaptation/content_scoring.py",
        )},
        "capabilities": capabilities.to_dict(),
        "parent_metrics": compared["parent"],
        "measurement": measured,
        "fork_sample_policy": "First four wrong valid target presentations in frozen practice order",
        "forks": [fork.to_dict() for fork in forks],
        "fork_replay_counts": {state: sum(fork.replay_status == state for fork in forks) for state in
                               ("matched", "accepted_already_preferred", "greedy_replay_mismatch", "unmeasured")},
        "protected_trajectory_sample_policy": "First two correct presentations in frozen practice order",
        "protected_trajectories": trajectory_reports,
        "resources": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
                      "peak_rss_bytes": peak_rss_bytes(), "decode_forward_calls": measured["resources"]["batched_decode_steps"],
                      "diagnostic_forward_calls": int(bool(actionable)) + int(bool(protected)), "backward_calls": 0},
    }
    json_dump_atomic(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--material", type=Path, default=MATERIAL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.material, args.out)
    print(json.dumps({key: result[key] for key in ("engineering_disposition", "parent_metrics", "fork_replay_counts", "resources")}))
