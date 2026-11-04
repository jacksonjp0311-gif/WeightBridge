"""Freeze fresh exact-Decimal families for the Phase III-M factorial study."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import secrets

from experiments.phase_iiil import semantic_roster as previous
from weightbridge.core import file_hash, json_dump_atomic


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("semantic_protocol.json")
SCHEMA = "weightbridge.phase_iiim.semantic_roster.v1"
SPLITS = ("adaptation", "preservation-fit", "calibration", "development-transfer",
          "preservation-validation", "selection", "hidden")
DISCLOSED = (*previous.DISCLOSED, *(f"results/phase-iii-l/semantic-v1/material/{split}.json"
                                  for split in previous.SPLITS if split != "hidden"))


def _read(path):
    return json.loads(Path(path).read_bytes())


def _sources():
    dependencies = (Path(__file__), Path(previous.__file__), Path(previous.renderer.__file__), PROTOCOL,
                    Path(previous.phase_k.__file__), *(ROOT / path for path in DISCLOSED))
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in dependencies}


def _blocked():
    """Exclude exact pairs and the fixed short operands of disclosed negative families."""
    used = previous._blocked_identities()
    anchors = set()
    for path in DISCLOSED:
        for row in _read(ROOT / path)["rows"]:
            operands = row["operands"]
            used.add(previous.renderer._identity(row["family"], operands))
            if row["family"] == "larger-decimal" and all(Decimal(value) < 0 for value in operands):
                lengths = [len(value.partition(".")[2]) for value in operands]
                if len(set(lengths)) > 1:
                    anchors.add(Decimal(operands[lengths.index(min(lengths))]))
    return used, anchors


def freeze(material: Path, hidden_path: Path, *, hidden_seed: str | None = None):
    """Write one immutable roster per partition and keep sealed examples with the host."""
    if (hidden_path.resolve().is_relative_to(ROOT)
            or hidden_path.resolve().is_relative_to(material.resolve())):
        raise ValueError("Hidden roster must remain outside public material and the checkout")
    if material.exists() or hidden_path.exists():
        raise FileExistsError("Fresh material and hidden paths are required")
    config = _read(PROTOCOL)
    seeds = {**config["seeds"], "hidden": hidden_seed or secrets.token_hex(32)}
    if set(seeds) != set(SPLITS) or len(set(seeds.values())) != len(SPLITS):
        raise ValueError("Every partition requires a distinct seed")
    used, anchors = _blocked()
    manifest = {"schema": SCHEMA, "protocol_sha256": file_hash(PROTOCOL),
                "sources": _sources(), "splits": {}}
    material.mkdir(parents=True)
    for split in SPLITS:
        rows = previous._generate(config, split, seeds[split], used, anchors)
        path = hidden_path if split == "hidden" else material / f"{split}.json"
        json_dump_atomic(path, {"schema": SCHEMA, "split": split, "rows": rows})
        manifest["splits"][split] = {"sha256": file_hash(path), "rows": len(rows),
            "items": len({row["item"] for row in rows}),
            "semantic_comparisons": len({row["comparison_id"] for row in rows if row["group"] == "target"}),
            "pairs": config["roster_counts"][split]["target_pairs"],
            "seed_commitment": hashlib.sha256(seeds[split].encode()).hexdigest()}
    json_dump_atomic(material / "manifest.json", manifest)
    return manifest


def read_manifest(material):
    manifest = _read(material / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["sources"] != _sources() or set(manifest["splits"]) != set(SPLITS)):
        raise ValueError("Material differs from the frozen source or protocol")
    return manifest


def load_split(material, split, manifest, *, hidden_path=None):
    if split not in SPLITS or (split == "hidden") != (hidden_path is not None):
        raise ValueError("Hidden material requires an explicit host path")
    if hidden_path is not None and (hidden_path.resolve().is_relative_to(ROOT)
                                    or hidden_path.resolve().is_relative_to(material.resolve())):
        raise ValueError("Sealed examples require a host path outside the public checkout")
    path = hidden_path if hidden_path is not None else material / f"{split}.json"
    expected, data = manifest["splits"][split], _read(path)
    rows = data["rows"]
    if (file_hash(path) != expected["sha256"] or data.get("schema") != SCHEMA or data.get("split") != split
            or len(rows) != expected["rows"] or len({row["item"] for row in rows}) != expected["items"]):
        raise ValueError("Frozen semantic partition differs")
    previous._validate(rows, split, _read(PROTOCOL)["roster_counts"][split])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--hidden-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.material, args.hidden_out), indent=2))
