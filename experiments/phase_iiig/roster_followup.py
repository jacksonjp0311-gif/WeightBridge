"""Freeze the separate Phase III-G model-specific qualification follow-up."""
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


PROTOCOL = Path(__file__).with_name("v2_protocol.json")
SCHEMA = "weightbridge.phase_iiig.roster.v2"
SPLITS = ("practice", "selection", "hidden")
DEMONSTRATIONS = {
    "multiplication": ((4, 6), (13, 3), (26, 17)),
    "first-digit": ((47,), (219,), (73284,)),
    "last-digit": ((61,), (583,), (89546,)),
    "copy": ((173,), (286,), (479,)),
}
INSTRUCTIONS = {
    "multiplication": (
        "Multiply the integers. Write only decimal digits for the product.",
        "Find the exact product of each pair of factors. Use an integer without separators.",
    ),
    "first-digit": (
        "Extract the first digit of the given number. Give only one digit.",
        "Find the leftmost digit of each integer and answer with just that digit.",
    ),
    "last-digit": (
        "Extract the last digit of the given number. Give only one digit.",
        "Find the rightmost digit of each integer and answer with just that digit.",
    ),
    "copy": (
        "Repeat the integer. Give only the integer answer.",
        "Copy each number exactly and output only the number.",
    ),
}


def answer(family, operands):
    if family == "multiplication":
        return str(operands[0] * operands[1])
    if family == "first-digit":
        return str(operands[0])[0]
    if family == "last-digit":
        return str(operands[0])[-1]
    if family == "copy":
        return str(operands[0])
    raise ValueError(f"Unsupported family {family}")


def _identity(family, operands):
    return (family, *(sorted(operands) if family == "multiplication" else operands))


def _prompt(family, operands, wording):
    label = "Expression: {0} * {1}" if family == "multiplication" else "Number: {0}"
    examples = [f"{label.format(*example)}\nAnswer: {answer(family, example)}"
                for example in DEMONSTRATIONS[family]]
    return "\n".join([INSTRUCTIONS[family][wording], *examples, label.format(*operands), "Answer:"])


def _draw(family, group, tier, index, rng):
    if family == "copy":
        return (rng.randint(100, 999),)
    if family == "multiplication":
        if group == "positive":
            return rng.randint(2, 9), rng.randint(2, 9)
        if tier == 0:
            return rng.randint(12, 98), rng.randint(3, 9)
        if tier == 1:
            return rng.randint(12, 98), rng.randint(12, 98)
        return rng.randint(103, 997), rng.randint(12, 98)
    if group == "positive":
        return (rng.randint(1, 9),) if index < 3 else (rng.randint(12, 98),)
    digits = (2, rng.randint(3, 4), rng.randint(5, 6))[tier]
    return (rng.randint(10 ** (digits - 1), 10 ** digits - 1),)


def _rows(config, split, seed, used):
    rng = random.Random(seed)
    counts = config["roster_counts"][split]
    rows = []
    groups = [(family, group) for family in config["target_families"] for group in ("target", "positive")]
    groups.append(("copy", "preserve"))
    for family, group in groups:
        for index in range(counts[group]):
            tier = index % 3 if group == "target" else None
            for _ in range(4096):
                operands = _draw(family, group, tier, index, rng)
                identity = _identity(family, operands)
                if identity not in used:
                    used.add(identity)
                    break
            else:
                raise ValueError(f"Could not sample disjoint {family} {group} rows")
            item = f"followup/{split}/{family}/{group}/{index:02d}"
            truth = answer(family, operands)
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
