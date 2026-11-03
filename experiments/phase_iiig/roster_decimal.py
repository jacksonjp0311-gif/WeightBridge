"""Freeze signed decimal comparison tasks and matched controls."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("v5_protocol.json")
SCHEMA = "weightbridge.phase_iiig.roster.v5"
SPLITS = ("practice", "selection", "hidden")
DEMONSTRATIONS = {
    "larger-decimal": (("2.45", "5.2"), ("-2.25", "-4.2")),
    "copy": (("1738.24",), ("4769.561",)),
}
INSTRUCTIONS = {
    "larger-decimal": (
        "Select the numerically larger decimal. Copy that number exactly. Give only the number.",
        "Which of the two decimals is greater on the number line? Write that option without explanation.",
    ),
    "copy": (
        "Repeat the decimal. Give only the number.",
        "Copy each decimal exactly and output only the number.",
    ),
}
STRATA = ("positive-close", "negative-close", "mixed-sign")


def _identity(family, operands):
    return (family, *sorted(Decimal(value) for value in operands))


def _answer(family, operands):
    return max(operands, key=Decimal) if family == "larger-decimal" else operands[0]


def _prompt(family, operands, wording):
    label = "Values: {0} or {1}" if family == "larger-decimal" else "Decimal: {0}"
    examples = [f"{label.format(*example)}\nAnswer: {_answer(family, example)}"
                for example in DEMONSTRATIONS[family]]
    return "\n".join([INSTRUCTIONS[family][wording], *examples, label.format(*operands), "Answer:"])


def _draw(group, tier, precision, truth_side, rng):
    if group == "preserve":
        return (f"{rng.randint(1000, 9999)}.{rng.randint(10, 99)}{rng.randint(1, 9)}",)
    if group == "positive":
        whole = rng.randint(1, 40)
        values = (f"{whole}.{rng.randint(1, 9)}", f"{whole + 12}.{rng.randint(10, 99)}{rng.randint(1, 9)}")
        if tier == "negative-far":
            values = tuple("-" + value for value in values)
    elif tier == "mixed-sign":
        short = f"0.{rng.randint(1, 3)}"
        long = f"0.{rng.randint(1, 3)}{rng.randint(1, 9)}"
        values = ("-" + short, long) if precision == "longer" else (short, "-" + long)
    else:
        whole, short_digit = rng.randint(1, 98), rng.randint(1, 8)
        digits = rng.choice((2, 3))
        magnitude_up = (precision == "longer") != (tier == "negative-close")
        offset = rng.randint(1, 9) * (1 if magnitude_up else -1)
        fractional = short_digit * 10 ** (digits - 1) + offset
        values = (f"{whole}.{short_digit}", f"{whole}.{fractional:0{digits}d}")
        if tier == "negative-close":
            values = tuple("-" + value for value in values)
    larger_side = max(range(2), key=lambda index: Decimal(values[index]))
    return values if larger_side == truth_side else values[::-1]


def _rows(config, split, seed, used):
    rng = random.Random(seed)
    counts = config["roster_counts"][split]
    rows = []
    for family, group in (("larger-decimal", "target"), ("larger-decimal", "positive"), ("copy", "preserve")):
        for index in range(counts[group]):
            if group == "target":
                tier = STRATA[index % 3]
                precision = ("shorter", "longer")[(index // 3) % 2]
                truth_side = (index // 6) % 2
            elif group == "positive":
                tier = ("positive-far", "negative-far")[index % 2]
                precision = None
                truth_side = index // 2
            else:
                tier = precision = truth_side = None
            for _ in range(4096):
                operands = _draw(group, tier, precision, truth_side, rng)
                identity = _identity(family, operands)
                if identity not in used:
                    used.add(identity)
                    break
            else:
                raise ValueError(f"Could not sample disjoint {family} {group} rows")
            item = f"decimal/{split}/{family}/{group}/{index:02d}"
            truth = _answer(family, operands)
            for wording in range(2):
                rows.append({
                    "id": f"{item}/{wording}", "item": item, "family": family, "group": group,
                    "wording": wording, "operands": list(operands), "tier": tier,
                    "truth_precision": precision, "truth_side": truth_side,
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
