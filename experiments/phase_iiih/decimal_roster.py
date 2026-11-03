"""Freeze fresh endpoint-practice and selection rows, with hidden bytes held separately."""
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

from experiments.phase_iiig import roster_decimal_targeted as previous
from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("decimal_protocol.json")
SCHEMA = "weightbridge.phase_iiih.decimal_roster.v1"
SPLITS = ("practice", "selection", "hidden")
DISCLOSED = tuple(f"results/phase-iii-g/{study}/material/{split}.json"
                  for study in ("decimal-v5", "negative-decimal-v6")
                  for split in SPLITS[:2])
TARGET_CELLS = tuple(dict.fromkeys(previous.TARGET_CELLS))


def _json(path):
    return json.loads(path.read_bytes())


def _source_hashes():
    paths = (Path(__file__), Path(previous.__file__), PROTOCOL, *(ROOT / path for path in DISCLOSED))
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}


def _blocked_identities():
    used = {previous._identity(family, operands)
            for family, examples in previous.DEMONSTRATIONS.items() for operands in examples}
    for path in DISCLOSED:
        for row in _json(ROOT / path)["rows"]:
            used.add(previous._identity(row["family"], row["operands"]))
    return used


def _rows(config, split, seed, used):
    rng = random.Random(seed)
    counts = config["roster_counts"]
    if counts["target"] != len(TARGET_CELLS) or counts["positive"] != 4:
        raise ValueError("Declared counts do not fit the frozen balanced strata")
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
                operands = previous._draw(group, tier, precision, truth_side, rng)
                identity = previous._identity(family, operands)
                if identity not in used:
                    used.add(identity)
                    break
            else:
                raise ValueError("Could not draw a fresh disjoint decimal question")
            item = f"endpoint-decimal/{split}/{family}/{group}/{index:02d}"
            for wording in config["endpoint"]["wordings"]:
                rows.append({
                    "id": f"{item}/{wording}", "item": item, "family": family, "group": group,
                    "wording": wording, "operands": list(operands), "tier": tier, "truth_precision": precision,
                    "truth_side": truth_side, "prompt": previous._prompt(family, operands, wording),
                    "answer": previous._answer(family, operands),
                })
    return rows


def freeze(material: Path, hidden_path: Path, *, hidden_seed: str | None = None) -> dict:
    if hidden_path.resolve().is_relative_to(ROOT) or hidden_path.resolve().is_relative_to(material.resolve()):
        raise ValueError("Hidden roster must reside outside the checkout and public material")
    if hidden_path.exists():
        raise FileExistsError(hidden_path)
    config = _json(PROTOCOL)
    material.mkdir(parents=True, exist_ok=False)
    seeds = {**config["seeds"], "hidden": hidden_seed or secrets.token_hex(32)}
    used = _blocked_identities()
    manifest = {"schema": SCHEMA, "protocol_sha256": file_hash(PROTOCOL), "sources": _source_hashes(), "splits": {}}
    for split in SPLITS:
        rows = _rows(config, split, seeds[split], used)
        path = hidden_path if split == "hidden" else material / f"{split}.json"
        json_dump_atomic(path, {"schema": SCHEMA, "split": split, "rows": rows})
        manifest["splits"][split] = {"sha256": file_hash(path), "rows": len(rows),
                                     "items": len(rows) // len(config["endpoint"]["wordings"]),
                                     "seed_commitment": hashlib.sha256(seeds[split].encode()).hexdigest()}
    json_dump_atomic(material / "manifest.json", manifest)
    return manifest


def read_manifest(material: Path) -> dict:
    manifest = _json(material / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["sources"] != _source_hashes() or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("Roster does not bind the frozen protocol and source bytes")
    return manifest


def load_split(material: Path, split: str, manifest: dict, *, hidden_path: Path | None = None) -> list[dict]:
    if split not in SPLITS or (split == "hidden") != (hidden_path is not None):
        raise ValueError("Hidden material requires an explicit host path")
    path = hidden_path if hidden_path is not None else material / f"{split}.json"
    digest = manifest["splits"][split]
    if file_hash(path) != digest["sha256"]:
        raise ValueError(f"Frozen {split} roster hash differs")
    data = _json(path)
    rows = data["rows"]
    if (data.get("schema") != SCHEMA or data.get("split") != split or len(rows) != digest["rows"]
            or len({row["id"] for row in rows}) != len(rows) or len({row["item"] for row in rows}) != digest["items"]):
        raise ValueError(f"Malformed {split} roster")
    for row in rows:
        operands, family = row["operands"], row["family"]
        if (row["answer"] != previous._answer(family, operands)
                or row["prompt"] != previous._prompt(family, operands, row["wording"])):
            raise ValueError("Roster row does not match its frozen semantic renderer")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--hidden-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.material, args.hidden_out), indent=2))
