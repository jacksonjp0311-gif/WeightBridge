"""Freeze a targeted Qwen study of negative-decimal comparison errors."""
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


PROTOCOL = Path(__file__).with_name("v6_protocol.json")
SCHEMA = "weightbridge.phase_iiig.roster.v6"
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
STRATA = ("negative-close", "negative-far", "positive-close", "mixed-sign")
TARGET_CELLS = tuple((tier, precision, side)
                     for tier in STRATA
                     for precision in ("shorter", "longer")
                     for side in (0, 1)
                     for _ in range(1 if tier.startswith("negative") else 2))


def _identity(family, operands):
    return (family, *sorted(Decimal(value) for value in operands))


def _answer(family, operands):
    return max(operands, key=Decimal) if family == "larger-decimal" else operands[0]


def _prompt(family, operands, wording):
    label = "Values: {0} or {1}" if family == "larger-decimal" else "Decimal: {0}"
    examples = [f"{label.format(*example)}\nAnswer: {_answer(family, example)}"
                for example in DEMONSTRATIONS[family]]
    return "\n".join([INSTRUCTIONS[family][wording], *examples, label.format(*operands), "Answer:"])


def _digits(rng, precision):
    if precision == "longer":
        return str(rng.randint(100, 999)) + str(rng.randint(1, 9))
    return str(rng.randint(1, 9))


def _draw(group, tier, precision, truth_side, rng):
    if group == "preserve":
        return (f"{rng.randint(1000, 9999)}.{_digits(rng, 'longer')}",)
    other_precision = "shorter" if precision == "longer" else "longer"
    if tier == "positive-far":
        smaller_whole = rng.randint(20, 59)
        values = (f"{smaller_whole}.{_digits(rng, other_precision)}",
                  f"{smaller_whole + rng.randint(21, 28)}.{_digits(rng, precision)}")
    elif tier == "mixed-sign":
        values = (f"-{rng.randint(1, 19)}.{_digits(rng, other_precision)}",
                  f"{rng.randint(1, 19)}.{_digits(rng, precision)}")
    elif tier == "negative-far":
        near_whole = rng.randint(10, 79)
        values = (f"-{near_whole + rng.randint(3, 8)}.{_digits(rng, other_precision)}",
                  f"-{near_whole}.{_digits(rng, precision)}")
    else:
        whole, short_digit = rng.randint(10, 98), rng.randint(1, 8)
        magnitude_up = (precision == "longer") != (tier == "negative-close")
        magnitude = rng.randint(1, 89) * 10 + rng.randint(1, 9)
        offset = magnitude if magnitude_up else -magnitude
        fractional = short_digit * 1000 + offset
        values = (f"{whole}.{short_digit}", f"{whole}.{fractional:04d}")
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
                tier, precision, truth_side = TARGET_CELLS[index]
            elif group == "positive":
                tier = ("positive-far", "mixed-sign")[index % 2]
                precision = ("shorter", "longer")[index // 2]
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
            item = f"negative-decimal/{split}/{family}/{group}/{index:02d}"
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
