"""Freeze and verify fresh, balanced arithmetic task material."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic


PROTOCOL = Path(__file__).with_name("matrix_protocol.json")
SPLITS = ("practice", "evaluation")
ANSWER_TEXT = (" true", " false")
PROMPTS = {
    "parity": (
        "Is the integer {value} even? Reply with only true or false.",
        "Does dividing {value} by two give an integer? Give only true or false.",
    ),
    "sign": (
        "Is the integer {value} positive? Reply with only true or false.",
        "Is {value} greater than zero? Give only true or false.",
    ),
}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def _label(family, value):
    return value % 2 == 0 if family == "parity" else value > 0


def _build_items(split, seed, counts, used):
    rng = random.Random(seed)
    items = []
    for family, count in counts.items():
        if type(count) is not int or count < 2 or count % 2:
            raise ValueError("A balanced roster needs an even number of at least two items")
        labels = [False, True] * (count // 2)
        rng.shuffle(labels)
        for label in labels:
            while True:
                magnitude = rng.randrange(1000, 10000)
                value = -magnitude if family == "sign" and not label else magnitude
                if abs(value) not in used and _label(family, value) == label:
                    used.add(abs(value))
                    break
            items.append({"id": f"{split}/{family}/{value}", "family": family, "value": value})
    rng.shuffle(items)
    return items


def freeze(out):
    config = read_json(PROTOCOL)
    out.mkdir(parents=True, exist_ok=False)
    used = set()
    manifest = {
        "schema": "weightbridge.phase_iiif.roster.v1",
        "protocol_sha256": file_hash(PROTOCOL),
        "generator_sha256": file_hash(__file__),
        "artifacts": {},
    }
    for split in SPLITS:
        seed = config["practice_seed"] if split == "practice" else secrets.token_hex(32)
        items = _build_items(split, seed, config["roster_counts_per_split"], used)
        path = out / f"{split}.json"
        json_dump_atomic(path, {"split": split, "items": items})
        manifest["artifacts"][split] = {
            "sha256": file_hash(path), "count": len(items), "seed_commitment": sha256(seed.encode()).hexdigest(),
        }
    json_dump_atomic(out / "manifest.json", manifest)
    return manifest


def load_split(material, split, manifest, counts):
    path = material / f"{split}.json"
    if file_hash(path) != manifest["artifacts"][split]["sha256"]:
        raise ValueError(f"Frozen {split} roster hash changed")
    payload = read_json(path)
    items = payload["items"]
    if payload["split"] != split or len(items) != manifest["artifacts"][split]["count"]:
        raise ValueError("Roster identity differs from manifest")
    ids, magnitudes = set(), set()
    tallies = {(family, label): 0 for family in counts for label in (False, True)}
    for item in items:
        family, value, ident = item["family"], item["value"], item["id"]
        if (set(item) != {"family", "value", "id"} or family not in PROMPTS
                or type(value) is not int or not 1000 <= abs(value) < 10000
                or (family == "parity" and value < 0) or ident != f"{split}/{family}/{value}"
                or ident in ids or abs(value) in magnitudes):
            raise ValueError("Invalid or duplicate arithmetic roster item")
        ids.add(ident)
        magnitudes.add(abs(value))
        tallies[(family, _label(family, value))] += 1
    if any(count != counts[family] // 2 for (family, _), count in tallies.items()):
        raise ValueError("Roster family/label balance changed")
    return items


def render_rows(items):
    rows = []
    for item in items:
        family, value = item["family"], item["value"]
        correct, incorrect = ANSWER_TEXT if _label(family, value) else ANSWER_TEXT[::-1]
        for wording, template in enumerate(PROMPTS[family]):
            rows.append({"id": f"{item['id']}/w{wording}", "item": item["id"], "wording": wording,
                         "prompt": template.format(value=value), "answers": (correct, incorrect),
                         "preserve": family == "sign", "family": family})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().out), indent=2))
