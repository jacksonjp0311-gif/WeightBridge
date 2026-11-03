"""Rebuild the authors' sparse p50c BF16 edit through WeightBridge's host gate.

Usage:
  python experiments/published_reproduction/rebuild_published_edit.py \
    --base <pinned-model.safetensors> --out <new-candidate.safetensors>

The script expects the reference patch artifacts in this experiment folder and
refuses a different pinned model revision or base file hash by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weightbridge.core import EditRejected, file_hash
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch

MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PINNED_SINGLE_FILE_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
UPSTREAM_COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
MANIFEST_PATH = ROOT / "experiments" / "published_reproduction" / "reference" / "p50c_manifest.json"
EDIT_PATH = ROOT / "experiments" / "published_reproduction" / "reference" / "p50c_edit.npz"


def bf16_to_float64(bits: np.ndarray) -> np.ndarray:
    word = np.asarray(bits, dtype="<u2").astype(np.uint32) << 16
    return word.view("<f4").astype(np.float64)


def rebuild(base: Path, output: Path, *, model_revision: str = MODEL_REVISION) -> dict:
    if model_revision != MODEL_REVISION:
        raise EditRejected("Only the published pinned model revision is accepted")
    if output.exists() or output.is_symlink():
        raise EditRejected(f"Candidate path already exists: {output}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with np.load(EDIT_PATH, allow_pickle=False) as archive:
        checkpoint = load_checkpoint(base)
        if checkpoint.root.is_file():
            observed_base_hash = checkpoint.checkpoint_sha256()
            if observed_base_hash != PINNED_SINGLE_FILE_SHA256:
                raise EditRejected("Base checkpoint file SHA-256 does not match the pinned revision")
        else:
            shard_hashes = {path.name: file_hash(path)
                            for path in checkpoint.shards}
            if PINNED_SINGLE_FILE_SHA256 not in shard_hashes.values():
                raise EditRejected("Model directory does not contain the exact pinned checkpoint shard")
            observed_base_hash = checkpoint.checkpoint_sha256()

        changes = []
        expected_tensors = {}
        observed_max_delta = 0.0
        sparse_payload_summary = []
        for index, item in enumerate(manifest["tensors"]):
            name = item["name"]
            if item.get("form") != "sparse":
                raise EditRejected(f"Unsupported published patch form: {item.get('form')}")
            ref = checkpoint.tensors.get(name)
            if ref is None or ref.dtype != "BF16" or list(ref.shape) != item["shape"]:
                raise EditRejected(f"Pinned base tensor metadata mismatch: {name}")
            if checkpoint.tensor_sha256(name) != item["base_sha256"]:
                raise EditRejected(f"Pinned base tensor hash mismatch: {name}")
            idx = np.asarray(archive[f"t{index}_idx"], dtype=np.int64)
            values = np.asarray(archive[f"t{index}_val"], dtype="<u2")
            if idx.ndim != 1 or values.ndim != 1 or idx.shape != values.shape or not len(idx):
                raise EditRejected(f"Malformed sparse data for {name}")
            if np.unique(idx).size != len(idx) or idx.min() < 0 or idx.max() >= ref.elements:
                raise EditRejected(f"Sparse coordinate bounds or uniqueness failed: {name}")
            base_values = bf16_to_float64(np.frombuffer(checkpoint.read_raw(name), dtype="<u2"))
            target_values = bf16_to_float64(values)
            observed_max_delta = max(observed_max_delta,
                                     float(np.max(np.abs(target_values - base_values[idx]))))
            flat_width = ref.shape[1]
            for flat_index, target in zip(idx.tolist(), target_values.tolist()):
                changes.append({"tensor": name,
                                "index": [int(flat_index // flat_width), int(flat_index % flat_width)],
                                "op": "set", "value": float(target)})
            expected_tensors[name] = item["edited_sha256"]
            changed_columns = np.unique(idx % flat_width)
            sparse_payload_summary.append({"tensor": name, "touched_entries": int(len(idx)),
                                           "changed_columns": int(len(changed_columns)),
                                           "published_changed_entries": int(item["changed_entries"]),
                                           "base_tensor_sha256": item["base_sha256"],
                                           "expected_edited_tensor_sha256": item["edited_sha256"]})

        total_parameters = sum(ref.elements for ref in checkpoint.tensors.values())
        patch = {
            "schema": "weightbridge.patch.v1", "base_sha256": observed_base_hash,
            "reason": "Deterministic reconstruction of published Experiment 8 p50c sparse BF16 patch",
            "evidence_ids": [f"lm_understand-logs@{UPSTREAM_COMMIT}:exp8_math/p50c"],
            "tensor_sha256": {item["name"]: item["base_sha256"] for item in manifest["tensors"]},
            "changes": changes, "projections": [],
        }
        touched = len(changes)
        policy = {
            "schema": "weightbridge.policy.v1",
            "allow_tensors": [item["name"] for item in manifest["tensors"]],
            "max_entries": touched,
            "max_fraction": touched / total_parameters,
            "max_abs_delta": max(1.0, observed_max_delta + 1e-5),
            "max_tensor_bytes": 16 * 1024 * 1024,
            "max_total_tensor_bytes": 32 * 1024 * 1024,
        }
        receipt = apply_safetensors_patch(base, patch, policy, output)
        candidate = load_checkpoint(output)
        tensor_verification = {}
        for name, expected in expected_tensors.items():
            actual = candidate.tensor_sha256(name)
            if actual != expected:
                raise EditRejected(f"Published edited tensor hash mismatch after host patch: {name}")
            tensor_verification[name] = {"expected_sha256": expected, "actual_sha256": actual, "match": True}
        source_hash_after = load_checkpoint(base).checkpoint_sha256()
        if source_hash_after != observed_base_hash:
            raise EditRejected("Base checkpoint changed during candidate creation")
        result = {
            "schema": "weightbridge.published_rebuild.v1", "status": "PASS",
            "source_repository": "https://github.com/RikiyaT/lm_understand-logs",
            "source_commit": UPSTREAM_COMMIT, "source_run": "exp8_math/runs/p50c",
            "model_id": "Qwen/Qwen3.5-0.8B", "model_revision": model_revision,
            "base_checkpoint_sha256": observed_base_hash,
            "pinned_single_file_sha256": PINNED_SINGLE_FILE_SHA256,
            "candidate_checkpoint_sha256": candidate.checkpoint_sha256(),
            "total_parameters": total_parameters, "touched_entries": touched,
            "published_changed_entries_after_bf16": int(manifest["changed_entries_total"]),
            "host_patch_changed_entries": int(receipt["changed_entries"]),
            "parameter_fraction_touched": touched / total_parameters,
            "observed_max_abs_delta": observed_max_delta,
            "tensor_verification": tensor_verification,
            "tensor_details": sparse_payload_summary,
            "host_receipt": receipt,
            "rollback_source_hash_unchanged": source_hash_after == observed_base_hash,
            "patch_artifact_sha256": hashlib.sha256(EDIT_PATH.read_bytes()).hexdigest(),
            "manifest_artifact_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "limitations": ["This verifies exact patch reconstruction, not benchmark reproduction."],
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Pinned single-file checkpoint or HF model directory")
    parser.add_argument("--out", required=True, help="New .safetensors candidate path")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    result = rebuild(Path(args.base), Path(args.out), model_revision=args.model_revision)
    report_path = Path(str(args.out) + ".rebuild.json")
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    summary = {key: result[key] for key in ("status", "base_checkpoint_sha256", "candidate_checkpoint_sha256",
                                             "touched_entries", "host_patch_changed_entries",
                                             "parameter_fraction_touched", "tensor_verification")}
    summary["report"] = str(report_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
