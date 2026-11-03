"""Append each Phase III-H trial once to the research evidence ledger."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from experiments.phase_iiih.analyze import replay
from experiments.phase_iiih.run_decimal_endpoint import PROTOCOL, read_json
from weightbridge.core import file_hash
from weightbridge.provenance.ledger import append_experiment


def record(root: Path) -> None:
    results = root / "results/phase-iii-h"
    trial = results / "decimal-endpoint-v1/trial"
    qwen_path = trial / "report.json"
    smol_path = results / "cross-model-smollm2/report.json"
    qwen, smol, config = read_json(qwen_path), read_json(smol_path), read_json(PROTOCOL)
    trials = replay(trial)
    relative = lambda path: path.relative_to(results).as_posix()
    now = datetime.now(timezone.utc).isoformat()
    common = {"schema": "weightbridge.experiment.v1", "timestamp_utc": now,
              "artifact_root": results.relative_to(root).as_posix()}
    ledger = root / "results/experiment_ledger.jsonl"
    append_experiment(ledger, {
        **common, "experiment_id": "phase-iii-h-qwen-endpoint-v1", "measurement_key": file_hash(qwen_path),
        "hypothesis": "Using actual task rivals and screening rounded doses on completed greedy responses may find a repeatable correction without sacrificing previously correct responses.",
        "source_model": {"qwen3_5": qwen["base_sha256"]},
        "proposed_modification": {"type": "bounded native rank-one affine doses", "arms": config["probe_design"],
                                  "layer_tensor": (qwen.get("plan") or {}).get("region")},
        "affected_parameters": {"probe_count": len(trials), "probes": trials,
                                "accepted_checkpoint": qwen.get("accepted_candidate_sha256")},
        "evaluation": {"suite": config["schema"], "practice_test_separated": True, "seeds": config["seeds"],
                       "inference": config["inference"], "qualification": qwen["behavioral_qualification"],
                       "arms_selected": qwen["arms"], "hidden_opened": qwen["hidden_opened"],
                       "surrogate_fidelity": qwen["surrogate_fidelity"], "reason": qwen["reason"]},
        "resources": qwen["resources"], "artifacts": [relative(qwen_path), relative(trial / "practice.json"),
                                                      "v6-audit.json"], "disposition": qwen["single_cycle"],
    })
    append_experiment(ledger, {
        **common, "experiment_id": "phase-iii-h-smollm2-endpoint-replay", "measurement_key": file_hash(smol_path),
        "hypothesis": "The same behavior contract can score and reproduce endpoint observations from a distinct native model after stored patch application and rollback.",
        "source_model": {"llama": smol["patch_receipt"]["base_sha256"]},
        "proposed_modification": {"type": "exact replay of a historical engineering patch"},
        "affected_parameters": {"changed_entries": smol["patch_receipt"]["changed_entries"],
                                "parameter_fraction": smol["patch_receipt"]["parameter_fraction_changed"]},
        "evaluation": {"suite": config["schema"], "split": "practice", "seed": config["seeds"]["practice"],
                       "engineering_disposition": smol["engineering_disposition"],
                       "adaptation_disposition": smol["behavioral_adaptation_disposition"],
                       "rollback_exact": smol["rollback_exact"],
                       "diagnostic_endpoint_comparison": smol["diagnostic_endpoint_comparison"]},
        "resources": smol["resources"], "artifacts": [relative(smol_path)], "disposition": "ENGINEERING_PASS",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    record(parser.parse_args().root)
