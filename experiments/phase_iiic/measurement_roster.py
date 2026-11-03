"""Freeze disjoint underlying questions before running Phase III-C measurement."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


PROTOCOL = Path(__file__).with_name("measurement_protocol.json")
SCHEMA = "weightbridge.phase_iiic.behavior_roster.v1"
FAMILIES = ("word-order", "date-order", "fraction-order")
PARAPHRASES = {
    "word-order": ("Which string comes earlier in alphabetical order?",
                   "Compare these strings from the left. Which one is lexicographically smaller?"),
    "date-order": ("Which event occurs on the earlier calendar date?",
                   "Put these dates in chronological order. Which date would be first?"),
    "fraction-order": ("Which fraction represents the larger number?",
                       "Compare the quotients. Which of these two fractions has the greater value?"),
}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def _date_text(value, style):
    if style == 0:
        return f"{value.day} {MONTHS[value.month - 1]} {value.year}"
    return f"{MONTHS[value.month - 1]} {value.day}, {value.year}"


def _sample(rng, family, index):
    if family == "word-order":
        prefix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        first, second = rng.sample("bdfhjlnprtvxz", 2)
        lengths = (3, 3) if index % 2 == 0 else ((2, 5) if index % 4 == 1 else (5, 2))
        words = [prefix + letter + "".join(rng.choices("aeioukmn", k=length))
                 for letter, length in zip((first, second), lengths, strict=True)]
        ordered = sorted(words)
        return ordered, [[word] for word in ordered], ordered
    if family == "date-order":
        origin = date(rng.randint(2210, 2270), 1, 1)
        # Crossing months prevents a single comparison of day numbers from solving the bank.
        early = origin + timedelta(days=rng.randint(20, 160))
        late = origin + timedelta(days=rng.randint(170, 348))
        ordered = (early, late)
        answers = [_date_text(value, index % 2) for value in ordered]
        aliases = [[answer, _date_text(value, 1 - index % 2), value.isoformat()]
                   for answer, value in zip(answers, ordered, strict=True)]
        return answers, aliases, [value.isoformat() for value in ordered]
    if family == "fraction-order":
        denominator = rng.randint(151, 413)
        first = Fraction(3 * denominator + rng.randint(1, denominator - 1), denominator)
        denominator += rng.choice((-23, -11, 13, 27))
        numerator = (first.numerator * denominator) // first.denominator + rng.choice((-2, -1, 1, 2))
        second = Fraction(numerator, denominator)
        if first == second:
            return _sample(rng, family, index)
        ordered = sorted((first, second), reverse=True)
        canonical = [str(value) for value in ordered]
        # Render an equivalent unreduced answer on alternating items.
        answers = [f"{value.numerator * factor}/{value.denominator * factor}"
                   for value, factor in zip(ordered, (2, 1) if index % 2 else (1, 2), strict=True)]
        aliases = [list(dict.fromkeys((answer, key))) for answer, key in zip(answers, canonical, strict=True)]
        return answers, aliases, canonical
    raise ValueError(f"Unknown task family {family}")


def _question(seed, split, family, index, used):
    rng = random.Random(f"{seed}|{split}|{family}|{index}")
    while True:
        answers, aliases, canonical = _sample(rng, family, index)
        key = hashlib.sha256(json.dumps([family, sorted(canonical)], separators=(",", ":")).encode()).hexdigest()
        if key not in used:
            used.add(key)
            break
    item_id = f"{split}/{family}/{key[:16]}"
    labels = ("A", "B") if index % 2 == 0 else ("X", "Y")
    if (index // 2) % 2:
        labels = labels[::-1]
    presentations = []
    for paraphrase, question in enumerate(PARAPHRASES[family]):
        for order in ([0, 1], [1, 0]):
            first, second = [answers[choice] for choice in order]
            common = f"{question}\n{labels[0]}. {first}\n{labels[1]}. {second}\n"
            presentations.append({
                "id": f"{item_id}/p{paraphrase}-o{order[0]}", "order": order, "labels": list(labels),
                "paraphrase": paraphrase,
                "letter_prompt": common + f"Reply with only the label {labels[0]} or {labels[1]}.",
                "content_prompt": common + "Reply with only the exact answer text of the correct option, without its label.",
            })
    return {"id": item_id, "family": family, "semantic_key": key, "answers": answers, "aliases": aliases,
            "presentations": presentations}


def freeze(base, out):
    config = json.loads(PROTOCOL.read_bytes())
    checkpoint = load_checkpoint(base)
    if checkpoint.checkpoint_sha256() != config["model_sha256"]:
        raise ValueError("Phase III-C measurement requires the pinned Qwen checkpoint")
    out = Path(out)
    out.mkdir(parents=True)
    used, artifacts = set(), {}
    for split, count in config["underlying_items_per_family"].items():
        rows = [_question(config["seed"], split, family, index, used)
                for family in FAMILIES for index in range(count)]
        path = out / f"{split}.json"
        json_dump_atomic(path, {"schema": SCHEMA, "split": split, "items": rows})
        artifacts[split] = {"sha256": file_hash(path), "underlying_items": len(rows)}
    manifest = {"schema": "weightbridge.phase_iiic.measurement_manifest.v1", "model_sha256": config["model_sha256"],
                "protocol_sha256": file_hash(PROTOCOL), "generator_sha256": file_hash(Path(__file__)),
                "artifacts": artifacts, "underlying_items": len(used)}
    json_dump_atomic(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.base, args.out), indent=2))
