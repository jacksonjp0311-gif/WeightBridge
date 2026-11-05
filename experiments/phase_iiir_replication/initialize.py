"""Write-once host preparation of the replication and conditional rank comparison."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

from experiments.phase_iiiq_baseline import data as q_data
from experiments.phase_iiiq_baseline.initialize import _inventory, _private
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from . import data
from .integrity import PARENT, ROOT, read, source_manifest, verify_sources
from .training import PARAMETERS, authorization, data_exposure_receipt, validate_recipe

PROTOCOL = Path(__file__).with_name("protocol.json")


def freeze_sources(path):
    value = read(path)
    if value.get("schema") != "weightbridge.phase_iiir.study.v1" or value.get("checkpoint_sha256") != PARENT:
        raise ValueError("Replication protocol requires the original model and master schema")
    if value.get("source_sha256"):
        verify_sources(value["source_sha256"])
        return value
    before = file_hash(path)
    value["source_sha256"] = source_manifest()
    fd, temporary = tempfile.mkstemp(prefix=".r-freeze-", dir=Path(path).parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if file_hash(path) != before or read(path).get("source_sha256"):
            raise ValueError("Prospective protocol changed while freezing")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    verify_sources(value["source_sha256"])
    return value


def arm_protocol(master, arm, *, master_sha256, global_manifest_sha256):
    if arm not in ("rank8", "rank4"):
        raise ValueError("Undeclared rank arm")
    value = deepcopy(master)
    setting = value["arms"][arm]
    value.update(schema="weightbridge.phase_iiir.protocol.v1", replication_arm=arm,
        master_protocol_sha256=master_sha256, global_data_manifest_sha256=global_manifest_sha256)
    value["adapter"].update(rank=setting["rank"], alpha=setting["alpha"])
    value["data"]["populations"] = {
        canonical: master["data"]["populations"][actual]
        for canonical, actual in data.arm_mapping(arm).items()}
    value["data"]["global_partition_mapping"] = data.arm_mapping(arm)
    validate_recipe(value)
    return value


def merge_authorization(config, inventory):
    rank = validate_recipe(config)
    total = sum(math.prod(row["weight_shape"]) for row in inventory)
    return {"schema": "weightbridge.supervised_lora_merge.v1", "base_checkpoint_sha256": PARENT,
        "allow_tensors": [row["name"]+".weight" for row in inventory], "max_rank": rank, "alpha": 2.*rank,
        "max_updated_entries": total, "max_tensor_bytes": max(2*math.prod(row["weight_shape"]) for row in inventory),
        "max_total_tensor_bytes": 2*total, "stored_dtype": "BF16", "max_update_norm": None,
        "merge_math": "float64_factor_product_then_round_once_into_BF16",
        "authority": f"Phase III-R prospectively authorized {config['replication_arm']} isolated conventional merge",
        "production_authority": False, "lineage_authority": False,
        "scope": "No historical sparse-edit norm restriction; independent stored-model qualification required."}


def freeze(base, inventory, out, *, protocol=PROTOCOL, q_study, q_hidden, historical_roots=(), additional_rosters=()):
    base, inventory, protocol = map(Path, (base, inventory, protocol))
    out = _private(out, base)
    host_root = _private(out.parent/(out.name+"-host"), base)
    if out.exists() or host_root.exists():
        raise FileExistsError("Replication initialization requires fresh write-once paths")
    checkpoint = load_checkpoint(base)
    if checkpoint.checkpoint_sha256() != PARENT:
        raise ValueError("Replication cannot use an edited or foreign checkpoint")
    native = _inventory(inventory, checkpoint)
    for rank, expected in PARAMETERS.items():
        if rank*sum(sum(row["weight_shape"]) for row in native) != expected:
            raise ValueError("Rank-specific module parameter count differs")
    master = freeze_sources(protocol)
    out.mkdir(parents=True)
    for name, source in (("protocol.json", protocol), ("architecture-inventory.json", inventory)):
        with (out/name).open("xb") as stream:
            stream.write(source.read_bytes())
    master_path = out/"protocol.json"
    manifest = data.freeze(out/"material", host_root, protocol=master_path,
        q_study=q_study, q_hidden=q_hidden, historical_roots=historical_roots, additional_rosters=additional_rosters)
    master_sha, manifest_sha = file_hash(master_path), file_hash(out/"material/manifest.json")
    arm_records, exposures = {}, {}
    for arm in ("rank8", "rank4"):
        root = out/arm
        root.mkdir()
        config = arm_protocol(master, arm, master_sha256=master_sha, global_manifest_sha256=manifest_sha)
        arm_path = root/"protocol.json"
        json_dump_atomic(arm_path, config)
        view = data.create_q_view(out/"material", host_root, manifest, root/"material", arm=arm,
            arm_protocol=arm_path, master_protocol=master_path)
        view_sha = file_hash(root/"material/manifest.json")
        for name, splits in (("training", ("train", "replay")), ("smoke", ("smoke",))):
            rows = [row for split in splits for row in q_data.load_split(root/"material", split, view, protocol=arm_path)]
            json_dump_atomic(root/f"{name}-bundle.json", {"schema": "weightbridge.supervised_training_bundle.v1",
                "protocol_sha256": file_hash(arm_path), "manifest_sha256": view_sha, "split_ids": list(splits), "rows": rows})
        smoke = authorization(config, native, root/"smoke-bundle.json", smoke=True)
        main = authorization(config, native, root/"training-bundle.json", smoke=False)
        json_dump_atomic(root/"smoke-authorization.json", smoke.to_dict())
        # The immutable plan binds all settings before either arm is trained.
        # Later admission adds only the required exact smoke-gate digest.
        json_dump_atomic(root/"main-authorization-plan.json", main.to_dict())
        json_dump_atomic(root/"merge-authorization.json", merge_authorization(config, native))
        exposure = data_exposure_receipt(root/"training-bundle.json", config)
        json_dump_atomic(root/"prospective-data-order.json", exposure)
        exposures[arm] = exposure
        mapping = data.arm_mapping(arm)
        arm_records[arm] = {"rank": config["adapter"]["rank"], "alpha": config["adapter"]["alpha"],
            "trainable_parameters": PARAMETERS[config["adapter"]["rank"]], "material_manifest_sha256": view_sha,
            "protocol_sha256": file_hash(arm_path), "hidden_path": str(host_root/f"{mapping['hidden']}.json"),
            "hidden_sha256": manifest["splits"][mapping["hidden"]]["sha256"], "main_training_authorized": False}
    if (read(out/"rank8/training-bundle.json")["rows"] != read(out/"rank4/training-bundle.json")["rows"]
            or exposures["rank8"] != exposures["rank4"]):
        raise RuntimeError("Rank arms differ in ordered training evidence or exposure")
    for split in ("train", "replay", "development"):
        if file_hash(out/f"rank8/material/{split}.json") != file_hash(out/f"rank4/material/{split}.json"):
            raise RuntimeError("Shared fitting/development view changed across ranks")
    if checkpoint.checkpoint_sha256() != PARENT:
        raise RuntimeError("Original checkpoint changed during initialization")
    verify_sources(master["source_sha256"])
    artifacts = {path.relative_to(out).as_posix(): file_hash(path) for path in sorted(out.rglob("*.json"))}
    result = {"schema": "weightbridge.phase_iiir.freeze.v1", "status": "FROZEN",
        "checkpoint_sha256": PARENT, "protocol_sha256": master_sha, "global_manifest_sha256": manifest_sha,
        "arms": arm_records, "artifacts": artifacts, "host_root": str(host_root),
        "matching_training_rows": True, "matching_data_order": True,
        "source_sha256": master["source_sha256"], "inference_performed": False,
        "main_training_authorized": False, "lineage_authority": False, "production_authority": False}
    json_dump_atomic(out/"freeze.json", result)
    return result


def authorize_main(out, arm, smoke_gate, *, replication_gate=None):
    out = _private(out)
    receipt = read(out/"freeze.json")
    if arm not in receipt["arms"]:
        raise ValueError("Unknown frozen rank arm")
    for name, digest in receipt["artifacts"].items():
        if file_hash(out/name) != digest:
            raise ValueError("Frozen study artifact changed before admission")
    root = out/arm
    protocol = read(root/"protocol.json")
    verify_sources(protocol["source_sha256"])
    from .smoke import validate_gate
    gate = validate_gate(root/"protocol.json", root/"smoke-authorization.json", root/"smoke-bundle.json",
                         root/"material/smoke.json", smoke_gate)
    if (gate.get("passed") is not True or gate.get("protocol_sha256") != file_hash(root/"protocol.json")
            or gate.get("rank") != protocol["adapter"]["rank"]
            or gate.get("alpha") != protocol["adapter"]["alpha"]
            or gate.get("authorization_sha256") != file_hash(root/"smoke-authorization.json")
            or gate.get("training_bundle_sha256") != file_hash(root/"smoke-bundle.json")):
        raise ValueError("Main admission requires the exact rank-specific verified smoke")
    if arm == "rank4":
        from .study import validate_replication_gate
        validate_replication_gate(protocol, replication_gate)
    elif replication_gate is not None:
        raise ValueError("Rank eight must begin from its original fresh replication")
    inventory = _inventory(out/"architecture-inventory.json")
    auth = authorization(protocol, inventory, root/"training-bundle.json", smoke=False, smoke_gate=smoke_gate)
    planned = read(root/"main-authorization-plan.json")
    actual = json.loads(json.dumps(auth.to_dict(), allow_nan=False))
    if {**actual, "smoke_gate_sha256": None} != planned:
        raise ValueError("Admission changed the prospectively frozen training authorization")
    json_dump_atomic(root/"main-authorization.json", actual)
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for name in ("base", "inventory", "out", "q-study", "q-hidden"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, default=PROTOCOL)
    prepare.add_argument("--historical-root", type=Path, action="append", default=[])
    prepare.add_argument("--historical-roster", type=Path, action="append", default=[])
    admit = commands.add_parser("authorize-main")
    for name in ("out", "smoke-gate"):
        admit.add_argument(f"--{name}", type=Path, required=True)
    admit.add_argument("--arm", choices=("rank8", "rank4"), required=True)
    admit.add_argument("--replication-gate", type=Path)
    args = parser.parse_args()
    result = (freeze(args.base, args.inventory, args.out, protocol=args.protocol, q_study=args.q_study,
        q_hidden=args.q_hidden, historical_roots=args.historical_root, additional_rosters=args.historical_roster)
        if args.command == "freeze" else authorize_main(args.out, args.arm, args.smoke_gate,
                                                       replication_gate=args.replication_gate))
    # No raw roster, protected output, or data-generation secret is printed.
    print(json.dumps({key: value for key, value in result.items() if key not in ("source_sha256", "artifacts")}, indent=2))
