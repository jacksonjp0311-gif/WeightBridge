"""Reconstruct a saved Phase III-K candidate from the public patch without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.phase_iiih.run_decimal_endpoint import _resources
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import apply_safetensors_patch


ROOT = Path(__file__).resolve().parents[2]


def reconstruct(base, archive, scratch, receipt_path):
    start, cpu = time.perf_counter(), time.process_time()
    if scratch.resolve().is_relative_to(ROOT) or scratch.exists():
        raise ValueError("Reconstruction requires a fresh checkpoint path outside the repository")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    report = json.loads((archive / "report.json").read_bytes())
    evidence = {}
    for suffix in ("patch", "policy", "receipt"):
        relative = f"patches/D-{suffix}.json"
        path = archive / relative
        if file_hash(path) != report["artifacts"][relative]:
            raise ValueError("Reconstruction inputs do not match the immutable trial")
        evidence[suffix] = json.loads(path.read_bytes())
    source = load_checkpoint(base)
    parent_sha = source.checkpoint_sha256()
    saved = evidence["receipt"]
    if parent_sha != saved["base_sha256"] or parent_sha != report["base_sha256"]:
        raise ValueError("Reconstruction parent differs from the verified original")
    replay = apply_safetensors_patch(base, evidence["patch"], evidence["policy"], scratch)
    candidate = load_checkpoint(scratch)
    candidate_sha = candidate.checkpoint_sha256()
    candidate_tensors = {name: candidate.tensor_sha256(name) for name in replay["tensor_receipts"]}
    expected = {name: record["candidate_sha256"] for name, record in saved["tensor_receipts"].items()}
    parent_restored = load_checkpoint(base).checkpoint_sha256() == parent_sha
    if (not parent_restored or candidate_sha != replay["candidate_sha256"]
            or candidate_sha != saved["candidate_sha256"] or candidate_tensors != expected
            or replay["changed_entries"] != saved["changed_entries"]):
        raise ValueError("Exact patch reconstruction check failed")
    receipt = {"schema": "weightbridge.phase_iiik.reconstruction.v1", "disposition": "PASS",
               "source_report_sha256": file_hash(archive / "report.json"),
               "patch_sha256": report["artifacts"]["patches/D-patch.json"],
               "policy_sha256": report["artifacts"]["patches/D-policy.json"],
               "base_sha256": parent_sha, "candidate_sha256": candidate_sha,
               "tensor_sha256": candidate_tensors, "changed_entries": replay["changed_entries"],
               "original_checkpoint_sha256_rechecked": parent_restored,
               "exact_candidate_reconstruction": True, "generation_evaluations": 0,
               "resources": _resources(start, cpu)}
    json_dump_atomic(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "archive", "scratch", "receipt"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.base, args.archive, args.scratch, args.receipt), indent=2))
