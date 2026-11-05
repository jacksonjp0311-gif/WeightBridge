"""Explicit-parent BF16 merge and streaming cumulative delta evidence for Phase S.

The unchanged LoRA engine performs each separately authorized merge. Sequential
reconstruction must first produce the exact rounded R checkpoint, then merge S.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiiq_baseline.integrity import read, verify_sources
from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _to_float64
from weightbridge.provenance import peak_rss_bytes
from .integrity import ORIGINAL, DECIMAL_PARENT, checked_sha256
from .train import runtime


def merge_authorization(protocol_path, adapter_sha256, targets):
    protocol = read(protocol_path)
    targets = tuple(targets)
    total = sum(math.prod(t.shape) for t in targets)
    if not targets or len({t.tensor_name for t in targets}) != len(targets):
        raise ValueError("Merge must have a nonempty distinct inspected tensor allowlist")
    return {"schema": "weightbridge.supervised_lora_merge.v1",
        "base_checkpoint_sha256": checked_sha256(protocol["checkpoint_sha256"]),
        "allow_tensors": [t.tensor_name for t in targets], "max_rank": 8, "alpha": 16.,
        "max_updated_entries": total, "max_tensor_bytes": max(2*math.prod(t.shape) for t in targets),
        "max_total_tensor_bytes": 2*total, "stored_dtype": "BF16", "max_update_norm": None,
        "merge_math": "float64_factor_product_then_round_once_into_BF16",
        "authority": "Phase III-S isolated conventional merge, exact actual training parent",
        "production_authority": False, "lineage_authority": False,
        "phase_s_binding": {"schema": "weightbridge.phase_iiis.merge_authorization.v1",
            "protocol_sha256": file_hash(protocol_path), "task_manifest_sha256": protocol["task_manifest_sha256"],
            "source_sha256": protocol["source_sha256"], "adapter_sha256": checked_sha256(adapter_sha256)}}


def validate_merge_authorization(protocol_path, policy, expected_adapter, expected_parent):
    protocol = read(protocol_path)
    if checked_sha256(expected_parent) != protocol["checkpoint_sha256"]:
        raise ValueError("Merge actual parent differs from the explicit training parent")
    expected = {"schema": "weightbridge.phase_iiis.merge_authorization.v1",
        "protocol_sha256": file_hash(protocol_path), "task_manifest_sha256": protocol["task_manifest_sha256"],
        "source_sha256": protocol["source_sha256"], "adapter_sha256": checked_sha256(expected_adapter)}
    if (policy.get("phase_s_binding") != expected or policy.get("base_checkpoint_sha256") != expected_parent
            or policy.get("schema") != "weightbridge.supervised_lora_merge.v1"
            or policy.get("max_rank") != 8 or policy.get("alpha") != 16
            or policy.get("stored_dtype") != "BF16"
            or policy.get("merge_math") != "float64_factor_product_then_round_once_into_BF16"
            or policy.get("production_authority") is not False or policy.get("lineage_authority") is not False):
        raise ValueError("Merge policy is not bound to this parent, adapter and Phase S protocol")
    verify_sources(protocol["source_sha256"])
    return protocol


def checkpoint_difference(original, candidate, *, expected_original, expected_candidate, chunk_elements=1048576):
    """Stream all stored tensors; measure actual cumulative delta, not added adapter norms."""
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= 1048576:
        raise ValueError("Cumulative delta requires a bounded streaming chunk")
    first, second = load_checkpoint(original), load_checkpoint(candidate)
    if (first.checkpoint_sha256() != checked_sha256(expected_original)
            or second.checkpoint_sha256() != checked_sha256(expected_candidate)
            or first.tensors.keys() != second.tensors.keys()):
        raise ValueError("Cumulative comparison checkpoint identity or tensor map differs")
    changed = total = 0
    parent_square = delta_square = 0.
    details = []
    for name in sorted(first.tensors):
        a, b = first.tensors[name], second.tensors[name]
        if (a.dtype, a.shape, a.nbytes) != (b.dtype, b.shape, b.nbytes):
            raise ValueError("Cumulative comparison tensor precision or shape differs")
        width = a.nbytes // a.elements if a.elements else 1
        square = source_square = 0.
        count = 0
        hashes = [hashlib.sha256(), hashlib.sha256()]
        with a.path.open("rb") as af, b.path.open("rb") as bf:
            af.seek(a.data_start + a.start)
            bf.seek(b.data_start + b.start)
            remaining = a.nbytes
            while remaining:
                size = min(remaining, chunk_elements*width)
                raw_a, raw_b = af.read(size), bf.read(size)
                if len(raw_a) != size or len(raw_b) != size:
                    raise ValueError("Truncated checkpoint during cumulative measurement")
                for digest, raw in zip(hashes, (raw_a, raw_b), strict=True):
                    digest.update(raw)
                if a.dtype not in ("BF16", "F16", "F32", "F64"):
                    if raw_a != raw_b:
                        raise ValueError("Non-floating checkpoint state changed")
                else:
                    x, y = _to_float64(raw_a, a.dtype), _to_float64(raw_b, b.dtype)
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise ValueError("Nonfinite stored tensor in cumulative comparison")
                    square += float(np.sum((y-x)**2))
                    source_square += float(np.sum(x*x))
                    count += int(np.count_nonzero(np.frombuffer(raw_a, dtype=f"V{width}") !=
                                                 np.frombuffer(raw_b, dtype=f"V{width}")))
                remaining -= size
        if count:
            details.append({"tensor": name, "dtype": a.dtype, "changed_entries": count,
                "stored_update_norm": math.sqrt(square), "source_sha256": hashes[0].hexdigest(),
                "candidate_sha256": hashes[1].hexdigest()})
        changed += count
        total += a.elements
        delta_square += square
        parent_square += source_square
    if first.checkpoint_sha256() != expected_original or second.checkpoint_sha256() != expected_candidate:
        raise ValueError("Checkpoint changed during cumulative delta measurement")
    return {"schema": "weightbridge.phase_iiis.cumulative_stored_delta.v1", "original_sha256": expected_original,
        "candidate_sha256": expected_candidate, "stored_update_norm": math.sqrt(delta_square),
        "original_parameter_norm": math.sqrt(parent_square),
        "relative_stored_update_norm": math.sqrt(delta_square/parent_square) if parent_square else None,
        "changed_entries": changed, "total_stored_entries": total, "changed_tensors": details,
        "scope": "All stored floating tensors; direct difference between original and final serialized weights"}


def verify_two_stage_provenance(original_sha, parent_sha, candidate_sha, prior_merge, current_merge):
    """Reject a collapsed sum-of-deltas reconstruction masquerading as two BF16 stages."""
    for digest in (original_sha, parent_sha, candidate_sha):
        checked_sha256(digest)
    checks = {"first_merge_parent": prior_merge.get("base_sha256") == original_sha,
        "rounded_intermediate": prior_merge.get("candidate_sha256") == parent_sha,
        "second_merge_parent": current_merge.get("base_sha256") == parent_sha,
        "second_merge_candidate": current_merge.get("candidate_sha256") == candidate_sha,
        "source_restoration": prior_merge.get("base_unchanged") is True and current_merge.get("base_unchanged") is True,
        "no_promotion": all(r.get("lineage_accepted") is False and r.get("production_deployment") is False
                            for r in (prior_merge, current_merge))}
    return {"passed": all(checks.values()), "checks": checks, "original_sha256": original_sha,
        "rounded_parent_sha256": parent_sha, "candidate_sha256": candidate_sha,
        "geometry": "Two separately rounded adapter merges, not a single cumulative rank-eight adapter"}


def run(base, protocol_path, adapter_path, expected_adapter, authorization, out, receipt, *,
        expected_parent, original, expected_original=ORIGINAL, prior_merge_receipt=None,
        expected_reconstruction=None):
    policy = read(authorization)
    protocol = validate_merge_authorization(protocol_path, policy, expected_adapter, expected_parent)
    out, receipt = Path(out), Path(receipt)
    verification = receipt.with_suffix(".verification.json")
    if out.exists() or receipt.exists() or verification.exists():
        raise FileExistsError("S merge/reconstruction artifacts are write-once")
    protected = (Path(base).resolve(), Path(original).resolve())
    if any(path.resolve().is_relative_to(parent) for path in (out, receipt, verification) for parent in protected):
        raise ValueError("S candidate and evidence must remain outside both immutable original and actual parent")
    if (expected_original != ORIGINAL or file_hash(adapter_path) != expected_adapter
            or expected_parent not in (ORIGINAL, DECIMAL_PARENT)
            or load_checkpoint(original).checkpoint_sha256() != expected_original):
        raise ValueError("S merge requires the pinned original and exact authorized adapter/parent")
    prior = None
    if expected_parent == DECIMAL_PARENT:
        if prior_merge_receipt is None:
            raise ValueError("Sequential merging requires recorded original-to-R rounded provenance")
        prior = read(prior_merge_receipt)
        if prior.get("base_sha256") != ORIGINAL or prior.get("candidate_sha256") != DECIMAL_PARENT:
            raise ValueError("Sequential parent provenance differs from the verified rounded R checkpoint")
    wall, cpu = time.perf_counter(), time.process_time()
    with runtime(base, protocol, expected_parent=expected_parent) as (source, runner):
        targets, _ = discover_lora_targets(source, runner)
        session = SupervisedLoRA.load(runner.model, targets, adapter_path,
            checkpoint_sha256=expected_parent, expected_sha256=expected_adapter)
        result = session.merge_candidate(base, out, authorization, receipt)
        cumulative = checkpoint_difference(original, out, expected_original=expected_original,
                                            expected_candidate=result["candidate_sha256"])
        sequential = verify_two_stage_provenance(ORIGINAL, DECIMAL_PARENT, result["candidate_sha256"], prior, result) if prior else None
        reconstructed = result["candidate_sha256"] == expected_reconstruction if expected_reconstruction is not None else None
        extended = {"schema": "weightbridge.phase_iiis.merge_verification.v1",
            "passed": (sequential is None or sequential["passed"]) and reconstructed is not False,
            "protocol_sha256": file_hash(protocol_path), "task_manifest_sha256": protocol["task_manifest_sha256"],
            "merge_receipt_sha256": file_hash(receipt), "adapter_sha256": expected_adapter,
            "parent_checkpoint_sha256": expected_parent, "original_sha256": expected_original,
            "candidate_sha256": result["candidate_sha256"], "exact_reconstruction": reconstructed,
            "expected_reconstruction_sha256": expected_reconstruction, "cumulative_original_to_candidate": cumulative,
            "two_stage_provenance": sequential,
            "prior_merge_receipt_sha256": file_hash(prior_merge_receipt) if prior_merge_receipt else None,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "peak_rss_bytes": peak_rss_bytes()}, "source_sha256": protocol["source_sha256"],
            "lineage_authority": False, "production_authority": False}
        verify_sources(protocol["source_sha256"])
        json_dump_atomic(verification, extended)
        if not extended["passed"]:
            raise ValueError("Merged candidate failed sequential provenance or exact reconstruction")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "protocol", "adapter", "authorization", "out", "receipt", "original"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("expected-adapter", "expected-parent"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--prior-merge-receipt", type=Path)
    parser.add_argument("--expected-reconstruction")
    args = parser.parse_args()
    run(args.base, args.protocol, args.adapter, args.expected_adapter, args.authorization, args.out, args.receipt,
        expected_parent=args.expected_parent, original=args.original, prior_merge_receipt=args.prior_merge_receipt,
        expected_reconstruction=args.expected_reconstruction)
