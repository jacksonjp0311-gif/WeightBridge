"""Independently rebuild a practice-qualified Phase III-O checkpoint without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import _resources
from experiments.phase_iiio.verification import ROOT, validate_fit_report
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64, apply_safetensors_patch


def reconstruct(base, trial, material, scratch, receipt_path, *, arm="C2"):
    wall, cpu = time.perf_counter(), time.process_time()
    reconstruction_source_sha = file_hash(__file__)
    if scratch.exists() or scratch.resolve().is_relative_to(ROOT):
        raise ValueError("Reconstruction requires a fresh checkpoint path outside the repository")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    report = validate_fit_report(trial, material)
    source_report_sha = file_hash(trial / "fit-report.json")
    if arm not in report["candidates"]:
        raise ValueError("Reconstruction arm must identify a saved practice-qualified candidate")
    candidate_record = report["candidates"][arm]
    inputs = {}
    hashes = {}
    for suffix in ("patch", "policy", "receipt"):
        relative = f"patches/{arm}-{suffix}.json"
        path = trial / relative
        hashes[suffix] = file_hash(path)
        if hashes[suffix] != report["artifacts"].get(relative):
            raise ValueError("Reconstruction input differs from the immutable fitting evidence")
        inputs[suffix] = json.loads(path.read_bytes())
    source = load_checkpoint(base)
    parent_sha = source.checkpoint_sha256()
    saved = inputs["receipt"]
    if parent_sha != report["checkpoint_sha256"] or parent_sha != saved["base_sha256"]:
        raise ValueError("Reconstruction parent differs from the verified source checkpoint")

    replay = apply_safetensors_patch(base, inputs["patch"], inputs["policy"], scratch)
    reconstructed = load_checkpoint(scratch)
    candidate_sha = reconstructed.checkpoint_sha256()
    tensor_hashes, changed, squared_norm = {}, 0, 0.
    for name in replay["tensor_receipts"]:
        tensor_hashes[name] = reconstructed.tensor_sha256(name)
        ref = source.tensors[name]
        delta = (_to_float64(reconstructed.read_raw(name), ref.dtype)
                 - _to_float64(source.read_raw(name), ref.dtype))
        changed += int(np.count_nonzero(delta))
        squared_norm += float(delta @ delta)
    norm = float(np.sqrt(squared_norm))
    original_rechecked = load_checkpoint(base).checkpoint_sha256() == parent_sha
    expected_tensors = {name: details["candidate_sha256"] for name, details in saved["tensor_receipts"].items()}
    if (not original_rechecked or candidate_sha != candidate_record["checkpoint_sha256"]
            or candidate_sha != replay["candidate_sha256"] or candidate_sha != saved["candidate_sha256"]
            or tensor_hashes != expected_tensors or set(tensor_hashes.values()) != {candidate_record["tensor_sha256"]}
            or changed != candidate_record["changed_entries"] or changed != saved["changed_entries"]
            or changed != replay["changed_entries"]
            or not np.isclose(norm, candidate_record["realized_update_norm"], rtol=0, atol=1e-10)
            or file_hash(trial / "fit-report.json") != source_report_sha
            or file_hash(__file__) != reconstruction_source_sha):
        raise ValueError("Exact stored-precision candidate reconstruction failed")
    # Revalidate every fitting input after replay, including the source manifest.
    validate_fit_report(trial, material)
    receipt = {
        "schema": "weightbridge.phase_iiio.reconstruction.v1", "disposition": "PASS", "arm": arm,
        "reconstruction_source_sha256": reconstruction_source_sha,
        "source_fit_report_sha256": source_report_sha, "patch_sha256": hashes["patch"],
        "policy_sha256": hashes["policy"], "source_patch_receipt_sha256": hashes["receipt"],
        "manifest_sha256": report["manifest_sha256"],
        "base_sha256": parent_sha, "candidate_sha256": candidate_sha, "tensor_sha256": tensor_hashes,
        "changed_entries": changed, "realized_update_norm": norm, "exact_candidate_reconstruction": True,
        "original_checkpoint_sha256_rechecked": original_rechecked, "generation_evaluations": 0,
        "resources": _resources(wall, cpu),
    }
    json_dump_atomic(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "trial", "material", "scratch", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", default="C2")
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.base, args.trial, args.material, args.scratch, args.receipt, arm=args.arm), indent=2))
