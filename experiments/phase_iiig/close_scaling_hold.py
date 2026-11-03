"""Close the Qwen-only v3 HOLD without treating a checkpoint pin as a participant."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig import run_scaling as scaling, run_trial as trial
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


def close(base, material, out):
    config = trial._json(scaling.PROTOCOL)
    participants = set(config["eligible_families"])
    if participants != {"qwen"} or set(config["checkpoints"]) != {"qwen", "smollm2"}:
        raise ValueError("Expected only the frozen v3 participant/checkpoint inventory mismatch")
    manifest = scaling._material(material)
    manifest_sha = file_hash(material / "manifest.json")
    preparation = out / "preparation.json"
    lock_path = out / "candidate-lock.json"
    report, lock = map(trial._json, (preparation, lock_path))
    sources = scaling._sources()
    base_sha = config["checkpoints"]["qwen"]
    if (report["model"] != "qwen" or lock["model"] != "qwen"
            or report["manifest_sha256"] != manifest_sha or lock["manifest_sha256"] != manifest_sha
            or report["source_sha256"] != sources or lock["source_sha256"] != sources
            or report["base_sha256"] != base_sha or lock["base_sha256"] != base_sha
            or load_checkpoint(base).checkpoint_sha256() != base_sha
            or report["behavioral_qualification"] != "HOLD" or report["single_cycle"] != "HOLD"
            or report["candidate_attempts"] != 0 or report["hidden_opened"] is not False
            or report["candidates"] or lock["candidates"] or lock["eligible"]
            or report["winner"] is not None or lock["winner"] is not None
            or list((out / "verification").iterdir()) or (out / "independent-lock.json").exists()):
        raise ValueError("A v3 closure may only record an already locked no-candidate HOLD")
    family = report["selected_family"]
    if family not in config["eligible_families"]["qwen"] or family != lock["family"]:
        raise ValueError("Prepared family differs from the frozen Qwen family")
    selection = trial._json(out / "selection.json")["rows"]
    verified_selection = trial.qualify(selection, family, config["qualification"])
    if verified_selection != report["selection"] or verified_selection["disposition"] != "HOLD":
        raise ValueError("Selection does not independently replay to HOLD")
    receipt_path = out / "closure.json"
    json_dump_atomic(receipt_path, {
        "schema": "weightbridge.phase_iiig.singleton_hold_closure.v1",
        "reason": "v3 authorized only Qwen under eligible_families but also retained a SmolLM2 checkpoint pin",
        "participants": sorted(participants), "preparation_sha256": file_hash(preparation),
        "candidate_lock_sha256": file_hash(lock_path), "manifest_sha256": manifest_sha,
        "selection_roster_sha256": manifest["splits"]["selection"]["sha256"],
        "closure_source_sha256": file_hash(Path(__file__)), "hidden_opened": False,
    })
    report["closure"] = {"type": "singleton_hold", "sha256": file_hash(receipt_path)}
    report["artifacts"] = {path.relative_to(out).as_posix(): file_hash(path)
                           for path in sorted(out.rglob("*.json"))}
    json_dump_atomic(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    outcome = close(args.base, args.material, args.out)
    print(outcome["behavioral_qualification"], flush=True)
