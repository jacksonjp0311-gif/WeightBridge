"""Create a random Qwen p50c-matched control candidate without evaluation labels.

The control has the same two tensors, selected 64 columns, changed-entry count
per layer, and empirical signed-magnitude delta multiset as the published edit.
If BF16 rounding would make an assigned edit a no-op, it is moved one BF16 ULP
in the assigned direction. The actual magnitude distribution is recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from weightbridge.core import EditRejected, file_hash
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch
from weightbridge.provenance import append_experiment, load_ledger, peak_rss_bytes

MANIFEST = ROOT / "experiments/published_reproduction/reference/p50c_manifest.json"
EDIT = ROOT / "experiments/published_reproduction/reference/p50c_edit.npz"
EXPECTED_BASE_FILE_HASH = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"


def _path_label(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _bf16_values(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype="<u2").astype(np.uint32) << 16).view("<f4").astype(np.float64)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).astype(np.uint64)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def _next_bf16(value_bits: int, direction: float) -> int:
    value = float(_bf16_values(np.asarray([value_bits], dtype="<u2"))[0])
    if value == 0:
        return 1 if direction > 0 else 0x8001
    positive = value > 0
    if direction > 0:
        return value_bits + 1 if positive else value_bits - 1
    return value_bits - 1 if positive else value_bits + 1


def _gzip_json(path: Path, value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    with path.open("xb") as file_obj:
        with gzip.GzipFile(fileobj=file_obj, mode="wb", compresslevel=9, mtime=0) as compressed:
            compressed.write(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base_path: Path, output_path: Path, *, seed: int = 20261103,
          patch_archive: Path, receipt_archive: Path, report_path: Path) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    base_path, output_path = base_path.resolve(), output_path.resolve()
    patch_archive, receipt_archive, report_path = (patch_archive.resolve(), receipt_archive.resolve(),
                                                     report_path.resolve())
    if patch_archive.exists() or receipt_archive.exists() or report_path.exists():
        raise FileExistsError("Random control evidence output already exists")
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(base_path)
    if checkpoint.root.is_file():
        base_sha = checkpoint.checkpoint_sha256()
        if base_sha != EXPECTED_BASE_FILE_HASH:
            raise EditRejected("Random control base file does not match pinned Qwen revision")
    else:
        shard_hashes = [file_hash(path) for path in checkpoint.shards]
        if EXPECTED_BASE_FILE_HASH not in shard_hashes:
            raise EditRejected("Random control model directory lacks the pinned base weight shard")
        base_sha = checkpoint.checkpoint_sha256()
    measurement_key = hashlib.sha256((base_sha + str(seed) + EDIT.name).encode()).hexdigest()
    existing = load_ledger(ROOT / "results/experiment_ledger.jsonl")
    if any(item.get("measurement_key") == measurement_key for item in existing):
        raise ValueError("This random-control seed and base are already recorded; do not count it twice")
    rng = np.random.default_rng(seed)
    with np.load(EDIT, allow_pickle=False) as archive:
        changes = []
        tensor_hashes = {}
        per_layer = []
        source_abs_deltas = []
        target_abs_deltas = []
        for i, item in enumerate(manifest["tensors"]):
            name = item["name"]
            ref = checkpoint.tensors.get(name)
            if ref is None or ref.dtype != "BF16" or list(ref.shape) != item["shape"]:
                raise EditRejected(f"Unexpected target tensor metadata: {name}")
            if checkpoint.tensor_sha256(name) != item["base_sha256"]:
                raise EditRejected(f"Unexpected base tensor hash: {name}")
            tensor_hashes[name] = item["base_sha256"]
            original_idx = np.asarray(archive[f"t{i}_idx"], dtype=np.int64)
            original_target_bits = np.asarray(archive[f"t{i}_val"], dtype="<u2")
            base_flat = np.frombuffer(checkpoint.read_raw(name), dtype="<u2").copy()
            base_values = _bf16_values(base_flat)
            original_deltas = _bf16_values(original_target_bits) - base_values[original_idx]
            columns = np.unique(original_idx % ref.shape[1])
            if len(columns) != 64:
                raise EditRejected(f"Published edit did not select exactly 64 columns in {name}")
            coordinate_pool = (np.arange(ref.shape[0], dtype=np.int64)[:, None] * ref.shape[1] + columns[None, :]).reshape(-1)
            selected = rng.choice(coordinate_pool, size=len(original_idx), replace=False)
            shuffled_deltas = rng.permutation(original_deltas)
            base_bits = base_flat[selected]
            target_bits = _bf16_bits(base_values[selected] + shuffled_deltas)
            no_op = target_bits == base_bits
            for j in np.flatnonzero(no_op):
                target_bits[j] = _next_bf16(int(base_bits[j]), float(shuffled_deltas[j]))
            actual_deltas = _bf16_values(target_bits) - base_values[selected]
            if np.any(actual_deltas == 0) or not np.isfinite(actual_deltas).all():
                raise EditRejected(f"Could not produce exact changed count for {name}")
            source_abs_deltas.extend(np.abs(original_deltas).tolist())
            target_abs_deltas.extend(np.abs(actual_deltas).tolist())
            width = ref.shape[1]
            for flat_index, target in zip(selected.tolist(), _bf16_values(target_bits).tolist()):
                changes.append({"tensor": name,
                                "index": [int(flat_index // width), int(flat_index % width)],
                                "op": "set", "value": float(target)})
            per_layer.append({"tensor": name, "layer": int(name.split("layers.")[1].split(".")[0]),
                              "changed_entries": len(selected), "selected_columns": columns.tolist(),
                              "unique_random_coordinates": int(len(np.unique(selected))),
                              "source_abs_delta_mean": float(np.mean(np.abs(original_deltas))),
                              "control_abs_delta_mean": float(np.mean(np.abs(actual_deltas))),
                              "source_abs_delta_p95": float(np.quantile(np.abs(original_deltas), .95)),
                              "control_abs_delta_p95": float(np.quantile(np.abs(actual_deltas), .95)),
                              "source_abs_delta_max": float(np.max(np.abs(original_deltas))),
                              "control_abs_delta_max": float(np.max(np.abs(actual_deltas)))})
    count = len(changes)
    total_parameters = sum(ref.elements for ref in checkpoint.tensors.values())
    max_delta = float(max(target_abs_deltas))
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": base_sha,
             "reason": "Randomized-coordinate control matched to published p50c target layers, selected columns, changed-entry counts, and empirical signed deltas.",
             "evidence_ids": ["published-p50c-manifest", f"random-control-seed-{seed}"],
             "tensor_sha256": tensor_hashes, "changes": changes, "projections": []}
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": list(tensor_hashes),
              "max_entries": count, "max_fraction": count / total_parameters,
              "max_abs_delta": max_delta + 1e-6, "max_tensor_bytes": 16 * 1024 * 1024,
              "max_total_tensor_bytes": 32 * 1024 * 1024}
    receipt = apply_safetensors_patch(base_path, patch, policy, output_path)
    candidate = load_checkpoint(output_path)
    source_after = load_checkpoint(base_path).checkpoint_sha256()
    if source_after != base_sha or receipt["changed_entries"] != count:
        raise EditRejected("Random control changed count mismatch or source checkpoint was modified")
    patch_gzip_sha = _gzip_json(patch_archive, patch)
    receipt_gzip_sha = _gzip_json(receipt_archive, receipt)
    build_wall = time.perf_counter() - wall_start
    build_cpu = time.process_time() - cpu_start
    result = {"schema": "weightbridge.random_control_rebuild.v1", "status": "PASS_REBUILD_ONLY",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "base_sha256": base_sha, "candidate_checkpoint_sha256": candidate.checkpoint_sha256(),
              "source_checkpoint_unchanged": True, "matched_changed_entries": count,
              "parameter_fraction_changed": count / total_parameters,
              "published_patch_sha256": hashlib.sha256(EDIT.read_bytes()).hexdigest(),
              "control_patch_sha256": hashlib.sha256(json.dumps(patch, sort_keys=True, separators=(",", ":"),
                                                               allow_nan=False).encode()).hexdigest(),
              "control_patch_archive": _path_label(patch_archive),
              "control_patch_archive_sha256": patch_gzip_sha,
              "host_receipt_archive": _path_label(receipt_archive),
              "host_receipt_archive_sha256": receipt_gzip_sha,
              "tensor_details": per_layer, "observed_max_abs_delta": max_delta,
              "resources": {"wall_seconds": build_wall, "cpu_seconds": build_cpu,
                            "peak_rss_bytes": peak_rss_bytes(), "peak_vram_bytes": None},
              "target_tensors": {name: candidate.tensor_sha256(name) for name in tensor_hashes},
              "candidate_path_external_only": _path_label(output_path),
              "limitations": ["Randomized patch was not behaviorally evaluated.",
                              "BF16 no-op replacements use the nearest one-ULP change in the sampled direction."]}
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    record = {"schema": "weightbridge.experiment.v1",
              "experiment_id": f"qwen-p50c-random-control-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
              "measurement_key": measurement_key, "timestamp_utc": result["timestamp_utc"],
              "hypothesis": "A random coordinate assignment with matched layer counts and edit magnitudes will serve as a negative control for the published p50c intervention.",
              "source_model": {"id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17", "sha256": base_sha},
              "proposed_modification": {"type": "randomized_coordinate_control", "seed": seed,
                                        "matched_to": "published p50c", "patch_sha256": result["control_patch_sha256"]},
              "affected_parameters": {"count": count, "fraction": count / total_parameters, "per_layer": per_layer},
              "evaluation": {"dataset": "Qwen Experiment 8 benchmark matrix (behavior not evaluated)",
                             "type": "matched random control checkpoint reconstruction",
                             "not_applicable": "No behavior dataset was run; practice/test separation and sampling do not apply to checkpoint construction.",
                             "seed": seed, "sampling": None, "sampling_applied": False,
                             "baseline_result": None, "candidate_result": None, "uncertainty": None,
                             "performed": False, "status": "HOLD",
                             "reason": "Full matched-seed Qwen benchmark is held because CPU generation cost is prohibitive."},
              "resources": {"wall_seconds": None, "cpu_seconds": None, "peak_rss_bytes": peak_rss_bytes(),
                            "peak_vram_bytes": None},
              "artifact_root": "results/published_reproduction",
              "artifacts": [_path_label(report_path), _path_label(patch_archive), _path_label(receipt_archive)],
              "disposition": "HOLD",
              "scientific_scope": "matched random control reconstructed only; no behavioral outcome"}
    record["resources"].update({"wall_seconds": build_wall, "cpu_seconds": build_cpu})
    append_experiment(ROOT / "results/experiment_ledger.jsonl", record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New HF model-directory candidate outside base")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--patch-archive", type=Path, required=True)
    parser.add_argument("--receipt-archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.base, args.out, seed=args.seed, patch_archive=args.patch_archive,
                   receipt_archive=args.receipt_archive, report_path=args.report)
    print(json.dumps({"status": result["status"], "candidate_checkpoint_sha256": result["candidate_checkpoint_sha256"],
                      "matched_changed_entries": result["matched_changed_entries"],
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
