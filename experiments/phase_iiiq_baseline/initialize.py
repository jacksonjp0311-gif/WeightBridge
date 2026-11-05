"""Host-only, write-once preparation of the conventional supervised study.

This command verifies checkpoint metadata and freezes data and authorizations.
It does not load a neural runtime, train, evaluate, merge, or promote a model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from weightbridge.adaptation.training_policy import TrainingAuthorization
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from . import data
from .integrity import PARENT, ROOT, read, source_manifest, verify_sources

PROTOCOL = Path(__file__).with_name("protocol.json")


def _private(path, base=None):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) or (base is not None and path.is_relative_to(Path(base).resolve())):
        raise ValueError("Host study artifacts must stay outside checkout and immutable model storage")
    return path


def _inventory(path, checkpoint=None):
    value = read(path)
    rows = value.get("allowed", ())
    if (value.get("checkpoint_sha256") != PARENT or value.get("allowed_count") != 186
            or len(rows) != 186 or len({row.get("name") for row in rows}) != 186):
        raise ValueError("Host authorization requires the exact inspected 186-module pinned inventory")
    for row in rows:
        name, shape = row["name"], row["weight_shape"]
        pieces = name.split(".")
        if (row.get("class") != "Linear" or row.get("weight_dtype") != "torch.bfloat16"
                or row.get("role") not in {"self_attn", "linear_attn", "mlp"}
                or len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)
                or any(part in pieces for part in ("vision", "visual", "lm_head", "embed_tokens", "embeddings"))
                or "layers" not in pieces or not 0 <= row.get("layer", -1) < 24):
            raise ValueError("Inventory contains an unsupported adaptation module")
        i = pieces.index("layers")
        if pieces[i+1:i+3] != [str(row["layer"]), row["role"]]:
            raise ValueError("Inventory native module path disagrees with layer/role")
        if checkpoint is not None:
            tensor = checkpoint.tensors.get(name + ".weight")
            if tensor is None or tensor.dtype != "BF16" or tuple(shape) != tensor.shape:
                raise ValueError("Inventory module is incompatible with pinned checkpoint storage")
    if {row["layer"] for row in rows} != set(range(24)):
        raise ValueError("Inventory must cover all twenty-four text layers")
    if sum(8*sum(row["weight_shape"]) for row in rows) != 5_411_328:
        raise ValueError("Inventory trainable parameter count differs from measured architecture")
    return tuple(sorted(rows, key=lambda row: row["name"]))


def _freeze_sources(path):
    value = read(path)
    if value.get("checkpoint_sha256") != PARENT:
        raise ValueError("Protocol is bound to another checkpoint")
    if value.get("source_sha256"):
        verify_sources(value["source_sha256"])
        return value
    value["source_sha256"] = source_manifest()
    # Only the explicitly empty prospective source map may be filled. Never
    # rewrite a populated frozen protocol or historical research artifact.
    original = file_hash(path)
    fd, temporary = tempfile.mkstemp(prefix=".freeze-source-", dir=Path(path).parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if file_hash(path) != original or read(path).get("source_sha256"):
            raise ValueError("Protocol changed during prospective source freeze")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    verify_sources(value["source_sha256"])
    return value


def _authorization(protocol, inventory, bundle, *, smoke, smoke_gate=None):
    settings = protocol["smoke_training" if smoke else "training"]
    expected_rates = [0.001] if smoke else [0.0001, 0.0003]
    if (settings["learning_rates"] != expected_rates or settings["optimizer_steps"] != (32 if smoke else 128)
            or protocol["adapter"]["rank"] != 8 or protocol["adapter"]["alpha"] != 16
            or protocol["adapter"]["dropout"] != 0):
        raise ValueError("Prospective conventional training settings differ from authorized study")
    return TrainingAuthorization(
        schema="weightbridge.supervised_training_authorization.v1", parent_sha256=PARENT,
        allowed_modules=tuple(row["name"] for row in inventory), max_trainable_parameters=5_411_328,
        rank=8, alpha=16., dropout=0., learning_rates=tuple(expected_rates), optimizer="AdamW",
        max_runs=1 if smoke else 2, max_steps_per_run=32 if smoke else 128,
        max_answer_tokens_per_run=10_000 if smoke else 40_000,
        max_input_tokens_per_run=50_000 if smoke else 400_000,
        max_memory_bytes=24*1024**3, max_wall_seconds_per_run=900. if smoke else 5400.,
        max_cpu_seconds_per_run=4500. if smoke else 30_000.,
        max_development_checkpoints_per_run=1 if smoke else 2, allow_merge=not smoke,
        training_bundle_sha256=file_hash(bundle),
        smoke_gate_sha256=file_hash(smoke_gate) if smoke_gate is not None else None)


def freeze(base, inventory, out, *, protocol=PROTOCOL):
    """Freeze all study populations without opening a model inference runtime."""
    base, inventory, protocol = Path(base), Path(inventory), Path(protocol)
    out = _private(out, base)
    hidden = _private(out.parent / (out.name + "-host") / "hidden.json", base)
    if out.exists() or hidden.exists():
        raise FileExistsError("Host initialization requires new write-once study and hidden paths")
    checkpoint = load_checkpoint(base)
    if checkpoint.checkpoint_sha256() != PARENT:
        raise ValueError("Cannot initialize study from another source checkpoint")
    rows = _inventory(inventory, checkpoint)
    config = _freeze_sources(protocol)
    out.mkdir(parents=True)
    with (out / "protocol.json").open("xb") as stream:
        stream.write(protocol.read_bytes())
    with (out / "architecture-inventory.json").open("xb") as stream:
        stream.write(inventory.read_bytes())
    frozen_protocol = out / "protocol.json"
    manifest = data.freeze(out / "material", hidden, protocol=frozen_protocol)
    manifest_sha = file_hash(out / "material/manifest.json")
    protocol_sha = file_hash(frozen_protocol)
    for name, splits in (("training", ("train", "replay")), ("smoke", ("smoke",))):
        material = [row for split in splits for row in data.load_split(
            out / "material", split, manifest, protocol=frozen_protocol)]
        json_dump_atomic(out / f"{name}-bundle.json", {
            "schema": "weightbridge.supervised_training_bundle.v1", "protocol_sha256": protocol_sha,
            "manifest_sha256": manifest_sha, "split_ids": list(splits), "rows": material})
    auth = _authorization(config, rows, out / "smoke-bundle.json", smoke=True)
    json_dump_atomic(out / "smoke-authorization.json", auth.to_dict())
    total = sum(math.prod(row["weight_shape"]) for row in rows)
    json_dump_atomic(out / "merge-authorization.json", {
        "schema": "weightbridge.supervised_lora_merge.v1", "base_checkpoint_sha256": PARENT,
        "allow_tensors": [row["name"] + ".weight" for row in rows], "max_rank": 8, "alpha": 16.,
        "max_updated_entries": total, "max_tensor_bytes": max(2*math.prod(row["weight_shape"]) for row in rows),
        "max_total_tensor_bytes": 2*total, "stored_dtype": "BF16", "max_update_norm": None,
        "merge_math": "float64_factor_product_then_round_once_into_BF16",
        "authority": "Phase III-Q separately authorized conventional supervised baseline, isolated research checkpoint only",
        "production_authority": False, "lineage_authority": False,
        "scope": "No inherited 0.025 sparse-edit cap; independently evaluated merged candidate required."})
    if checkpoint.checkpoint_sha256() != PARENT:
        raise RuntimeError("Source checkpoint changed during host initialization")
    verify_sources(config["source_sha256"])
    names = ("protocol.json", "architecture-inventory.json", "material/manifest.json", "training-bundle.json",
             "smoke-bundle.json", "smoke-authorization.json", "merge-authorization.json")
    result = {"schema": "weightbridge.supervised_baseline.freeze.v1", "status": "FROZEN",
        "checkpoint_sha256": PARENT, "protocol_source": str(protocol.resolve()),
        "artifacts": {name: file_hash(out / name) for name in names},
        "hidden_sha256": manifest["splits"]["hidden"]["sha256"], "hidden_path": str(hidden),
        "source_sha256": config["source_sha256"], "main_training_authorized": False,
        "inference_performed": False, "production_authority": False, "lineage_authority": False}
    json_dump_atomic(out / "freeze.json", result)
    return result


def authorize_main(out, smoke_gate):
    """Admit only the two prospectively frozen main runs after the smoke passes."""
    out = _private(out)
    receipt = read(out / "freeze.json")
    for name, digest in receipt["artifacts"].items():
        if file_hash(out / name) != digest:
            raise ValueError("Initialized host material changed before main authorization")
    protocol = read(out / "protocol.json")
    verify_sources(protocol["source_sha256"])
    gate = read(smoke_gate)
    if gate.get("passed") is not True or gate.get("protocol_sha256") != file_hash(out / "protocol.json"):
        raise ValueError("Main training requires the passed, exact-protocol pipeline smoke")
    inventory = _inventory(out / "architecture-inventory.json")
    auth = _authorization(protocol, inventory, out / "training-bundle.json", smoke=False, smoke_gate=smoke_gate)
    json_dump_atomic(out / "main-authorization.json", auth.to_dict())
    return auth.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for name in ("base", "inventory", "out"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, default=PROTOCOL)
    admit = commands.add_parser("authorize-main")
    admit.add_argument("--out", type=Path, required=True)
    admit.add_argument("--smoke-gate", type=Path, required=True)
    args = parser.parse_args()
    result = (freeze(args.base, args.inventory, args.out, protocol=args.protocol) if args.command == "freeze"
              else authorize_main(args.out, args.smoke_gate))
    print(json.dumps(result, indent=2))
