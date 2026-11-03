"""Freeze a larger, disjoint Qwen multiplication replication after the Phase III-G scaling study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("v4_protocol.json")
SCHEMA = "weightbridge.phase_iiig.roster.v4"
SPLITS = ("practice", "selection", "hidden")
DEMONSTRATIONS = {
    "multiplication": ((12, 2), (1204, 3), (10430, 12)),
    "copy": ((13981,), (28767,), (47915,)),
}
INSTRUCTIONS = {
    "multiplication": (
        "Multiply the integers. Write only decimal digits for the product.",
        "Find the exact product of each pair of factors. Use an integer without separators.",
    ),
    "copy": (
        "Repeat the integer. Give only the integer answer.",
        "Copy each number exactly and output only the number.",
    ),
}


def _identity(family, operands):
    return (family, *sorted(operands))


def _answer(family, operands):
    return str(operands[0] * operands[1] if family == "multiplication" else operands[0])


def _prompt(family, operands, wording):
    label = "Expression: {0} * {1}" if family == "multiplication" else "Number: {0}"
    examples = [f"{label.format(*example)}\nAnswer: {_answer(family, example)}"
                for example in DEMONSTRATIONS[family]]
    return "\n".join([INSTRUCTIONS[family][wording], *examples, label.format(*operands), "Answer:"])


def _draw(group, tier, rng):
    if group == "preserve":
        return (rng.randint(10000, 99999),)
    if group == "positive":
        return rng.randint(12, 27), 2
    ranges = (((1013, 9979), (3, 9)), ((1013, 9979), (103, 997)), ((10013, 99797), (12, 98)))
    left, right = ranges[tier]
    return rng.randint(*left), rng.randint(*right)


def _rows(config, split, seed, used):
    rng = random.Random(seed)
    counts = config["roster_counts"][split]
    rows = []
    for family, group in (("multiplication", "target"), ("multiplication", "positive"), ("copy", "preserve")):
        for index in range(counts[group]):
            tier = index % 3 if group == "target" else None
            for _ in range(4096):
                operands = _draw(group, tier, rng)
                identity = _identity(family, operands)
                if identity not in used:
                    used.add(identity)
                    break
            else:
                raise ValueError(f"Could not sample disjoint {family} {group} rows")
            item = f"replication/{split}/{family}/{group}/{index:02d}"
            truth = _answer(family, operands)
            for wording in range(2):
                rows.append({
                    "id": f"{item}/{wording}", "item": item, "family": family, "group": group,
                    "wording": wording, "operands": list(operands), "tier": tier,
                    "prompt": _prompt(family, operands, wording), "answer": truth,
                })
    return rows


def freeze(output, *, hidden_seed=None):
    output = Path(output)
    config = json.loads(PROTOCOL.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    seeds = {**config["seeds"], "hidden": hidden_seed or secrets.token_hex(32)}
    used = {_identity(family, operands) for family, examples in DEMONSTRATIONS.items() for operands in examples}
    manifest = {"schema": SCHEMA, "protocol_sha256": file_hash(PROTOCOL),
                "roster_source_sha256": file_hash(Path(__file__)), "splits": {}}
    for split in SPLITS:
        rows = _rows(config, split, seeds[split], used)
        path = output / f"{split}.json"
        json_dump_atomic(path, {"schema": SCHEMA, "split": split, "rows": rows})
        manifest["splits"][split] = {
            "sha256": file_hash(path), "rows": len(rows), "items": len(rows) // 2,
            "seed_commitment": hashlib.sha256(seeds[split].encode()).hexdigest(),
        }
    json_dump_atomic(output / "manifest.json", manifest)
    return manifest


def load_split(material, split, manifest):
    if split not in SPLITS:
        raise ValueError("Unknown roster split")
    if (manifest.get("schema") != SCHEMA or manifest.get("protocol_sha256") != file_hash(PROTOCOL)
            or manifest.get("roster_source_sha256") != file_hash(Path(__file__))):
        raise ValueError("Roster manifest does not bind the frozen generator and protocol")
    path = Path(material) / f"{split}.json"
    frozen = manifest["splits"][split]
    if file_hash(path) != frozen["sha256"]:
        raise ValueError(f"Frozen {split} roster hash differs")
    data = json.loads(path.read_bytes())
    rows = data["rows"]
    if (data.get("schema") != SCHEMA or data.get("split") != split
            or len(rows) != frozen["rows"] or len({row["id"] for row in rows}) != len(rows)
            or len({row["item"] for row in rows}) != frozen["items"]):
        raise ValueError(f"Frozen {split} roster is malformed")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out), indent=2))
