"""Freeze disjoint integer arithmetic practice, selection and hidden rosters."""
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


PROTOCOL = Path(__file__).with_name("protocol.json")
SCHEMA = "weightbridge.phase_iiig.roster.v1"
SPLITS = ("practice", "selection", "hidden")
FAMILIES = ("addition", "subtraction")
_DEMONSTRATIONS = {
    "addition": (((2, 3), (5, 3)), ((8, 1), (3, 3))),
    "subtraction": (((8, 2), (5, 1)), ((9, 2), (8, 3))),
    "copy": (((31,), (74,)), ((21,), (89,))),
}


def _answer(family, operands):
    if family == "addition":
        return str(sum(operands))
    if family == "subtraction":
        return str(operands[0] - operands[1])
    return str(operands[0])


def _identity(family, operands):
    return (family, *(sorted(operands) if family == "addition" else operands))


def _prompt(family, operands, wording):
    if family == "copy":
        instruction = ("Repeat the integer. Give only the integer answer.",
                       "Write the same number after Answer: without changing it.")[wording]
        template = ("Question: What integer is written here, {0}?",
                    "Number to copy = {0}")[wording]
    elif family == "addition":
        instruction = ("Compute the integer sum. Give only the integer answer.",
                       "Add the numbers and write the sum after Answer:.")[wording]
        template = ("Question: What is {0} plus {1}?", "{0} + {1}")[wording]
    else:
        instruction = ("Compute the integer difference. Give only the integer answer.",
                       "Subtract the second number from the first. Write the result after Answer:.")[wording]
        template = ("Question: What is {0} minus {1}?", "{0} - {1}")[wording]
    examples = [f"{template.format(*example)}\nAnswer: {_answer(family, example)}"
                for example in _DEMONSTRATIONS[family][wording]]
    return "\n".join([instruction, *examples, template.format(*operands), "Answer:"])


def _draw(family, group, tier, rng):
    if family == "copy":
        return (rng.randint(10, 99),)
    if group == "positive":
        if family == "addition":
            left = rng.randint(1, 8)
            return left, rng.randint(1, 9 - left)
        left = rng.randint(2, 9)
        return left, rng.randint(1, left - 1)
    if tier == 0:
        left, right = rng.randint(10, 99), rng.randint(1, 9)
    elif tier == 1:
        left, right = rng.randint(10, 99), rng.randint(10, 99)
    else:
        left, right = rng.randint(100, 999), rng.randint(100, 999)
    if family == "addition" and left % 10 + right % 10 >= 10:
        return left, right
    if family == "subtraction" and left > right and left % 10 < right % 10:
        return left, right
    return None


def _rows(config, split, seed, used):
    rng = random.Random(seed)
    counts = config["roster_counts"][split]
    rows = []
    groups = [(family, group) for family in FAMILIES for group in ("target", "positive")]
    groups.append(("copy", "preserve"))
    for family, group in groups:
        for index in range(counts[group]):
            tier = index % 3 if group == "target" else None
            for _ in range(4096):
                operands = _draw(family, group, tier, rng)
                if operands is not None and _identity(family, operands) not in used:
                    used.add(_identity(family, operands))
                    break
            else:
                raise ValueError(f"Could not sample disjoint {family} {group} rows")
            item = f"{split}/{family}/{group}/{index:02d}"
            answer = _answer(family, operands)
            for wording in range(2):
                rows.append({
                    "id": f"{item}/{wording}", "item": item, "family": family, "group": group,
                    "wording": wording, "operands": list(operands), "tier": tier,
                    "prompt": _prompt(family, operands, wording), "answer": answer,
                })
    return rows


def freeze(output, *, hidden_seed=None):
    """Write host material once. Keep the random hidden seed outside public output."""
    output = Path(output)
    config = json.loads(PROTOCOL.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    seeds = {**config["seeds"], "hidden": hidden_seed or secrets.token_hex(32)}
    used = {_identity(family, item) for family, forms in _DEMONSTRATIONS.items()
            for form in forms for item in form}
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
