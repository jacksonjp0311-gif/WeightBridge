"""Fit preregistered host controls before opening the completed study's holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.controls import build_matched_control, matched_random_patch
from weightbridge.core import EditRejected, file_hash, json_dump_atomic
from weightbridge.evaluation.hf_edit import evaluate_hf_edit
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64, apply_safetensors_patch
from weightbridge.provenance import peak_rss_bytes


def read(path):
    return json.loads(path.read_bytes())


def measure(operation):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        result = {"status": "COMPLETE", "result": operation()}
    except EditRejected as exc:
        result = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        result = {"status": "HOLD", "reason": f"{type(exc).__name__}: {exc}"}
    return {**result, "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
            "process_peak_rss_so_far_bytes": peak_rss_bytes(), "peak_vram_bytes": None}


def rounded_column_delta(base, candidate, patch, policy):
    edit = patch["slices"][0]
    name = edit["tensor"]
    checkpoints = [load_checkpoint(path) for path in (base, candidate)]

    def selected(checkpoint):
        ref = checkpoint.tensors[name]
        raw = checkpoint.read_raw(name, max_bytes=policy["max_tensor_bytes"])
        return _to_float64(raw, ref.dtype).reshape(ref.shape)[:, edit["indices"]]

    delta = selected(checkpoints[1]) - selected(checkpoints[0])
    return {"requested_frobenius_norm": float(np.linalg.norm(edit["values"])),
            "realized_frobenius_norm": float(np.linalg.norm(delta)),
            "changed_entries": int(np.count_nonzero(delta))}


def run(base, materialized, study_run, frozen_config, out):
    protocol_path = materialized / "protocol.json"
    protocol_sha = file_hash(protocol_path)
    protocol, config = read(protocol_path), read(frozen_config)
    completion_path = study_run / "summary.json"
    completion = read(completion_path)
    if protocol_sha != config["protocol_sha256"] or protocol_sha != completion["protocol_sha256"]:
        raise ValueError("Study and control configuration do not bind the same protocol")
    if completion["final"]["suite_sha256"] != protocol["final_sha256"]:
        raise ValueError("The frozen final evaluation must complete before running controls")
    if load_checkpoint(base).checkpoint_sha256() != protocol["base_sha256"]:
        raise ValueError("Controls require the exact original checkpoint")

    spec = protocol["cycles"][0]
    cycle_id = spec["id"]
    practice = materialized / "practice" / f"{cycle_id}.json"
    if file_hash(practice) != spec["practice_sha256"]:
        raise ValueError("Practice data differ from the frozen protocol")
    first = study_run / "state" / "cycles" / cycle_id / "session" / "attempts" / "candidate-0001"
    first_result = read(first / "result.json")
    reference = config["fixed_deterministic_comparator"]
    if any(first_result.get(key) != reference[key] for key in ("tensor", "strength")):
        raise ValueError("The first proposal differs from the frozen comparator recipe")
    patch_path = first / "patch.json"
    if file_hash(patch_path) != first_result["patch_file_sha256"]:
        raise ValueError("The frozen comparator patch changed")
    patch = read(patch_path)
    reference_model = first / "model"
    original_hash = protocol["base_sha256"]
    reference_hash = first_result["candidate_sha256"]
    if patch["base_sha256"] != original_hash or load_checkpoint(reference_model).checkpoint_sha256() != reference_hash:
        raise ValueError("Comparator provenance differs from the frozen first attempt")
    if original_hash == reference_hash:
        raise ValueError("Comparator reference has no rounded weight change")

    out.mkdir(parents=True)
    random_dir, gradient_dir = out / "random", out / "gradient"
    random_dir.mkdir()
    gradient_dir.mkdir()
    random_model, gradient_model = random_dir / "model", gradient_dir / "model"
    seed = config["random_comparator"]["seed"]

    def random_control():
        control = matched_random_patch(patch, seed=seed)
        json_dump_atomic(random_dir / "patch.json", control)
        result = build_matched_control(base, patch_path, protocol["policy"], random_model, seed=seed)
        if result["proposal"]["candidate_sha256"] != reference_hash:
            raise ValueError("Control replay did not reproduce the frozen proposal")
        return result

    def gradient_control():
        from weightbridge.adaptation.gradient_control import fit_gradient_control

        recipe = config["gradient_comparator"]
        control = fit_gradient_control(base, read(practice), patch, protocol["inference"],
                                       **{key: recipe[key] for key in ("seed", "steps", "learning_rate")})
        json_dump_atomic(gradient_dir / "patch.json", control)
        receipt = apply_safetensors_patch(base, control, protocol["policy"], gradient_model)
        rounded = rounded_column_delta(base, gradient_model, control, protocol["policy"])
        if rounded["changed_entries"] != receipt["changed_entries"]:
            raise ValueError("Rounded gradient changes differ from the patch receipt")
        return {**receipt, "rounded_scope": rounded}

    builds = {"random": measure(random_control), "gradient": measure(gradient_control)}
    json_dump_atomic(out / "builds.json", builds)

    holdout = materialized / "sealed" / f"{cycle_id}.json"
    if file_hash(holdout) != spec["holdout_sha256"]:
        raise ValueError("Sealed comparator data differ from the frozen protocol")
    options = {**protocol["inference"], "min_target_gain": protocol["min_target_gain"],
               "max_sentinel_flips": protocol["max_sentinel_flips"]}
    arms = {"deterministic": reference_model, "random": random_model, "gradient": gradient_model}
    scores = {}
    for arm, model in arms.items():
        if arm in builds and builds[arm]["status"] != "COMPLETE":
            scores[arm] = {"status": builds[arm]["status"], "reason": builds[arm]["reason"]}
            continue
        scores[arm] = {}
        for split, suite in (("practice", practice), ("holdout", holdout)):
            report = out / f"{arm}-{split}.json"
            scores[arm][split] = measure(lambda: evaluate_hf_edit(base, model, suite, report, **options))
            json_dump_atomic(out / f"{arm}-{split}.resources.json", scores[arm][split])
    summary = {"schema": "weightbridge.phase_iii.control_run.v1", "protocol_sha256": protocol_sha,
               "control_config_sha256": file_hash(frozen_config), "study_summary_sha256": file_hash(completion_path),
               "cycle_id": cycle_id, "family": spec["family"], "base_sha256": original_hash,
               "reference_candidate_sha256": reference_hash, "reference_patch_sha256": file_hash(patch_path),
               "builds": builds, "scores": scores}
    json_dump_atomic(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("base", "materialized", "study-run", "frozen-config", "out"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.base, args.materialized, args.study_run, args.frozen_config, args.out)
    print(json.dumps({"scores": {arm: {split: result["status"] for split, result in splits.items()}
                                          if "status" not in splits else {"build": splits["status"]}
                                 for arm, splits in summary["scores"].items()}}, sort_keys=True))
