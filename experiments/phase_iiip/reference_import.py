"""Isolated, hash-pinned external p50c reconstruction through the host patcher.

No upstream Python is imported or executed. This importer has no access to
Track A fitting data and cannot authorize lineage admission or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from experiments.phase_iiih.run_decimal_endpoint import _resources
from weightbridge.core import EditRejected, file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches import apply_safetensors_patch
from weightbridge.patches.huggingface import _to_float64

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPOSITORY = "https://github.com/RikiyaT/lm_understand-logs"
SOURCE_COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BASE_SHA256 = "7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8"
MANIFEST = "exp8_math/runs/p50c/edit/manifest.json"
EDIT = "exp8_math/runs/p50c/edit/edit.npz"
EXPECTED_CHANGED = 130188
TENSOR_NAMES = tuple(f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (15, 19))
SOURCE_HASHES = {
    "README.md": "b1c51eeb9b9e2935ff227fdc7796c2a67bd3a55014cf99291224f9cc7b93d342",
    "LICENSE": "395f7febedfda6b5498ca36cfe3344d3986659ce4385dd8b03d2d75604250a8e",
    "exp8_math/README.md": "473e386ade21ff52240103003b6b125e4362d9fa11010916c1b770ddb0edcbbd",
    "exp8_math/runs/p50c/final_answer.md": "497ea024210e9bbc10826756997ef41b454dd524390fff1b2d4cdec6782c30d1",
    "exp8_math/runs/p50c/work_notes/final_report.md": "d74088bc1ebe81aa00ebbef148dc40c446a99ec89c731b61b68a49061d452178",
    MANIFEST: "83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225",
    EDIT: "e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7",
    "exp8_math/code/apply_edit.py": "d2b627a6cf344f309cca4bcab4ff6ccc6ad1c8f29cabda78d11f99c19d1b1c5d",
    "exp8_math/code/st_numpy.py": "3c4e532fb8edf374e1a4636aa806a04d062f824943eaf4bca0d3af5b0e7dca40",
}
EXPORT = "exp8_math/base_model/Qwen3.5-0.8B-text"
EXPORT_HASHES = {
    "config.json": "3f56b6210db3c52e0bafb50da005eb52b6d340ab4ad980091879b397743675fc",
    "tokenizer_config.json": "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "generation_config.json": "62153eb6c69f2e1f426beaa8002b7186437e949c7588167085df14e10e9c0a73",
}


def _inputs(source):
    hashes = {**SOURCE_HASHES, **{f"{EXPORT}/{name}": digest for name, digest in EXPORT_HASHES.items()}}
    for name, digest in hashes.items():
        if file_hash(source / name) != digest:
            raise ValueError(f"Pinned external source hash mismatch: {name}")
    manifest = json.loads((source / MANIFEST).read_bytes())
    with zipfile.ZipFile(source / EDIT) as archive:
        if (len(archive.infolist()) != 4
                or sum(member.file_size for member in archive.infolist()) > 8 * 1024 * 1024):
            raise ValueError("External edit exceeds the bounded four-array payload")
    with np.load(source / EDIT, allow_pickle=False) as archive:
        if set(archive.files) != {f"t{i}_{kind}" for i in range(2) for kind in ("idx", "val")}:
            raise ValueError("External edit has missing or unrecognized arrays")
        arrays = {key: archive[key].copy() for key in archive.files}
    return manifest, arrays, hashes


def _assets(base, source):
    """Bind local input assets to the pinned HF revision, without downloading them."""
    api_path = source / "qwen-pinned-api.json"
    metadata = json.loads(api_path.read_bytes())
    if metadata.get("sha") != MODEL_REVISION:
        raise ValueError("Pinned model metadata has a different revision")
    siblings = {row["rfilename"]: row for row in metadata["siblings"]}
    rows = []
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        data = (base / name).read_bytes()
        reference = siblings[name]
        if "lfs" in reference:
            observed = hashlib.sha256(data).hexdigest()
            expected, algorithm = reference["lfs"]["sha256"], "sha256"
        else:
            observed = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
            expected, algorithm = reference["blobId"], "git_blob_sha1"
        if observed != expected or len(data) != reference["size"]:
            raise ValueError(f"Model input asset differs from the pinned revision: {name}")
        rows.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "expected_revision_digest": expected, "observed_revision_digest": observed,
            "revision_digest_algorithm": algorithm, "match": True})
    current = json.loads((base / "config.json").read_bytes())
    text = current.get("text_config", current)
    authored = json.loads((source / EXPORT / "config.json").read_bytes())
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "eos_token_id", "layer_types")
    structure = {key: {"local": text.get(key), "author_export": authored.get(key)} for key in keys}
    if any(value["local"] != value["author_export"] for value in structure.values()):
        raise ValueError("Local text architecture does not match the published export")
    template_match = file_hash(base / "chat_template.jinja") == EXPORT_HASHES["chat_template.jinja"]
    if not template_match:
        raise ValueError("Local chat template differs from the pinned published export")
    return {"revision": MODEL_REVISION, "hf_metadata_sha256": file_hash(api_path), "local_assets": rows,
        "text_structure": structure, "chat_template_matches_author_export": True,
        "local_architectures": current.get("architectures"), "author_export_architectures": authored.get("architectures"),
        "scope": "Pinned public-release asset identity and text structure. Authors used a text-only export and different inference/evaluation runtime; complete runtime equivalence is not established."}


def _payload(checkpoint, manifest, arrays):
    tensors = manifest.get("tensors", ())
    if (len(tensors) != 2 or tuple(row.get("name") for row in tensors) != TENSOR_NAMES
            or manifest.get("changed_tensors") != 2 or manifest.get("changed_entries_total") != EXPECTED_CHANGED):
        raise ValueError("External manifest exceeds the exact authorized two-tensor release")
    changes, details, tensor_hashes = [], [], {}
    square, source_square, maximum = 0., 0., 0.
    for i, item in enumerate(tensors):
        name = item["name"]
        ref = checkpoint.tensors.get(name)
        if (ref is None or ref.dtype != "BF16" or list(ref.shape) != item["shape"]
                or item.get("dtype") != "bfloat16" or item.get("form") != "sparse"
                or checkpoint.tensor_sha256(name) != item["base_sha256"]):
            raise ValueError(f"Published source tensor is incompatible: {name}")
        idx, val = arrays[f"t{i}_idx"], arrays[f"t{i}_val"]
        if (idx.ndim != 1 or idx.dtype.kind not in "iu" or val.dtype.kind != "u" or val.dtype.itemsize != 2
                or val.ndim != 1 or idx.shape != val.shape or not 0 < len(idx) <= 131072
                or len(np.unique(idx)) != len(idx) or idx.min() < 0 or idx.max() >= ref.elements):
            raise ValueError(f"Malformed or out-of-bounds external sparse payload: {name}")
        raw = checkpoint.read_raw(name, max_bytes=16 * 1024 * 1024)
        bits = np.frombuffer(raw, dtype="<u2")
        rebuilt = bits.copy()
        rebuilt[idx] = val
        edited_sha = hashlib.sha256(rebuilt.astype("<u2").tobytes()).hexdigest()
        actual = int(np.count_nonzero(rebuilt != bits))
        if edited_sha != item["edited_sha256"] or actual != item["changed_entries"]:
            raise ValueError(f"External edited tensor does not reproduce the published hash/count: {name}")
        source_values = _to_float64(raw, "BF16")
        new_values = _to_float64(val.astype("<u2").tobytes(), "BF16")
        delta = new_values - source_values[idx]
        if not np.isfinite(new_values).all() or not np.isfinite(source_values).all():
            raise ValueError("External source or edited tensor contains nonfinite values")
        norm, baseline = float(np.linalg.norm(delta)), float(np.linalg.norm(source_values))
        square += norm * norm
        source_square += baseline * baseline
        maximum = max(maximum, float(np.max(np.abs(delta))))
        for index, value in zip(idx.tolist(), new_values.tolist(), strict=True):
            changes.append({"tensor": name, "index": [int(index // ref.shape[1]), int(index % ref.shape[1])],
                            "op": "set", "value": value})
        tensor_hashes[name] = item["base_sha256"]
        details.append({"tensor": name, "shape": list(ref.shape), "dtype": ref.dtype,
            "source_sha256": item["base_sha256"], "edited_sha256": edited_sha,
            "sparse_payload_entries": len(idx), "changed_entries": actual,
            "changed_columns": int(len(np.unique(idx % ref.shape[1]))),
            "absolute_update_norm": norm, "source_full_matrix_norm": baseline,
            "relative_full_matrix_update_norm": norm / baseline if baseline else None})
    if sum(row["changed_entries"] for row in details) != EXPECTED_CHANGED or len(changes) > 131072:
        raise ValueError("External release support does not match its frozen count")
    patch = {"schema": "weightbridge.patch.v1", "base_sha256": checkpoint.checkpoint_sha256(),
        "reason": "Isolated external p50c reference reconstruction; not a Track A adaptation candidate",
        "evidence_ids": [f"{SOURCE_REPOSITORY}@{SOURCE_COMMIT}/exp8_math/runs/p50c"],
        "tensor_sha256": tensor_hashes, "changes": changes, "projections": []}
    norms = {"absolute_update_norm": float(np.sqrt(square)),
        "source_affected_full_matrices_norm": float(np.sqrt(source_square)),
        "relative_affected_full_matrices_update_norm": float(np.sqrt(square / source_square)) if source_square else None,
        "relative_norm_scope": "The two full modified matrices, not total-model parameter fraction or Track A norm authorization",
        "max_abs_delta": maximum, "changed_entries": EXPECTED_CHANGED, "sparse_payload_entries": len(changes)}
    return patch, details, norms


def _prepared(base, source):
    manifest, arrays, hashes = _inputs(source)
    checkpoint = load_checkpoint(base)
    if checkpoint.checkpoint_sha256() != BASE_SHA256:
        raise ValueError("Original checkpoint differs from the pinned Track B parent")
    assets = _assets(base, source)
    patch, details, norms = _payload(checkpoint, manifest, arrays)
    return checkpoint, patch, {"status": "REFERENCE_COMPATIBLE", "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION, "base_checkpoint_sha256": BASE_SHA256,
        "external_code_executed": False, "source_sha256": hashes, "compatibility": assets,
        "tensor_details": details, **norms,
        "licenses": {"upstream_python": "MIT", "patch_and_logs": "CC-BY-4.0", "model": "Apache-2.0; not redistributed"},
        "author_claims_not_independent_measurements": {"held_out_practice_items": 1152, "rescues": 579, "losses": 5},
        "runtime_equivalence_claimed": False}


def inspect_reference(base: Path, source: Path):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        _, _, result = _prepared(base, source)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        result = {"status": "REFERENCE_INCOMPATIBLE", "reason": str(error), "external_code_executed": False}
    return {"schema": "weightbridge.phase_iiip.reference_compatibility.v1", **result,
        "importer_source_sha256": file_hash(__file__), "resources": _resources(wall, cpu)}


def _authorize(path):
    if path is None or not path.is_file():
        raise EditRejected("Missing separate host authorization for isolated Track B reference artifact")
    record = json.loads(path.read_bytes())
    checks = {"schema": "weightbridge.phase_iiip.reference_authorization.v1", "track": "B",
        "base_checkpoint_sha256": BASE_SHA256, "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION,
        "manifest_sha256": SOURCE_HASHES[MANIFEST], "edit_sha256": SOURCE_HASHES[EDIT],
        "expected_changed_entries": EXPECTED_CHANGED}
    if (any(record.get(key) != value for key, value in checks.items())
            or any(record.get(key, False) is not False for key in
                   ("lineage_promotion", "lineage_authority", "production_authority"))
            or not record.get("authority")):
        raise EditRejected("Reference authorization does not bind the isolated pinned artifact")
    policy = record.get("patch_policy", {})
    if (set(policy.get("allow_tensors", ())) != set(TENSOR_NAMES)
            or policy.get("max_entries", 0) > 131072 or policy.get("max_entries", 0) < EXPECTED_CHANGED):
        raise EditRejected("Reference host policy exceeds the exact two-tensor support")
    return record, policy


def reconstruct_reference(base: Path, source: Path, authorization: Path, out: Path, receipt: Path):
    wall, cpu = time.perf_counter(), time.process_time()
    importer_sha = file_hash(__file__)
    if (out.exists() or out.is_symlink() or receipt.exists()
            or out.resolve().is_relative_to(ROOT) or out.resolve().is_relative_to(base.resolve())
            or out.resolve() == base.resolve() or receipt.resolve().is_relative_to(base.resolve())
            or receipt.resolve().is_relative_to(out.resolve())):
        raise EditRejected("Reference checkpoint requires a fresh isolated destination outside the original model and checkout")
    _, policy = _authorize(authorization)
    authorization_sha = file_hash(authorization)
    try:
        parent, patch, compatibility = _prepared(base, source)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        result = {"schema": "weightbridge.phase_iiip.reference_reconstruction.v1", "status": "REFERENCE_INCOMPATIBLE",
            "reason": str(error), "candidate_created": False, "resources": _resources(wall, cpu)}
        json_dump_atomic(receipt, result)
        return result
    patch_receipt = apply_safetensors_patch(base, patch, policy, out)
    candidate = load_checkpoint(out)
    if (set(patch_receipt["tensor_receipts"]) != set(TENSOR_NAMES)
            or patch_receipt["changed_entries"] != EXPECTED_CHANGED
            or any(candidate.tensor_sha256(row["tensor"]) != row["edited_sha256"] for row in compatibility["tensor_details"])
            or load_checkpoint(base).checkpoint_sha256() != BASE_SHA256
            or file_hash(authorization) != authorization_sha or file_hash(__file__) != importer_sha):
        raise EditRejected("External reconstruction failed edited tensor identity or original checkpoint restoration")
    result = {"schema": "weightbridge.phase_iiip.reference_reconstruction.v1", "status": "REFERENCE_RECONSTRUCTED",
        "candidate_created": True, "candidate_checkpoint_sha256": candidate.checkpoint_sha256(),
        "base_checkpoint_sha256_rechecked": BASE_SHA256, "source_checkpoint_preserved": True,
        "authorization_sha256": authorization_sha, "host_patch_receipt": patch_receipt,
        "compatibility": compatibility, "importer_source_sha256": importer_sha,
        "lineage_accepted": False, "production_deployment": False, "generation_evaluations": 0,
        "resources": _resources(wall, cpu)}
    json_dump_atomic(receipt, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "reconstruct"))
    for name in ("base", "source", "authorization", "out", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=name in {"base", "source"})
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_reference(args.base, args.source)
        if args.receipt:
            json_dump_atomic(args.receipt, result)
    else:
        if not all((args.authorization, args.out, args.receipt)):
            parser.error("reconstruct requires a separate authorization, new candidate destination and receipt")
        result = reconstruct_reference(args.base, args.source, args.authorization, args.out, args.receipt)
    print(json.dumps(result, indent=2), flush=True)
