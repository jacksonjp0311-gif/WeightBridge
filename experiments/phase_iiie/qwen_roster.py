"""Freeze new ordering populations, with independent private seeds for held-back splits."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiid.roster import _date_text, _format_prompt
from weightbridge.core import file_hash, json_dump_atomic

PROTOCOL = Path(__file__).with_name("qwen_protocol.json")
FAMILIES = ("word-order", "date-order", "fraction-order")
SPLITS = ("practice", "selection", "qualification")
RESERVE = "sealed-adaptation"
LABELS = (("A", "B"), ("X", "Y"))
ORDERS = ((0, 1), (1, 0))
CONTEXTS = ("task", "reversed", "paraphrase", "paraphrase_reversed", "neutral", "irrelevant")
QUESTIONS = {
    "word-order": (
        "Which string sorts earlier alphabetically?", "Which string sorts later alphabetically?",
        "In a dictionary ordering, which string would precede the other?",
        "In a dictionary ordering, which string would follow the other?",
    ),
    "date-order": (
        "Which date comes earlier on the calendar?", "Which date comes later on the calendar?",
        "Which of these dates would be placed first in a timeline?",
        "Which of these dates would be placed second in a timeline?",
    ),
    "fraction-order": (
        "Which fraction represents the greater number?", "Which fraction represents the smaller number?",
        "Which of the fractions is higher on the number line?",
        "Which of the fractions is lower on the number line?",
    ),
}


def _choices(family, rng, index):
    if family == "word-order":
        prefix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        low, high = sorted(rng.sample("bcdfghjklmnpqrstvwxyz", 2))
        lengths = ((2, 2), (2, 3), (3, 2))[index % 3]
        answers = [prefix + letter + "".join(rng.choices("aeioukmn", k=length))
                   for letter, length in zip((low, high), lengths, strict=True)]
        return answers, [[answer] for answer in answers], answers
    if family == "date-order":
        origin = date(rng.randint(2460, 2559), 1, 1)
        dates = [origin + timedelta(days=rng.randrange(1, 165)),
                 origin + timedelta(days=rng.randrange(180, 358) + 365 * (index % 2))]
        canonical = [value.isoformat() for value in dates]
        answers = [_date_text(value, index % 2) for value in dates]
        return answers, [[answer, _date_text(value, 1 - index % 2), value.isoformat()]
                         for answer, value in zip(answers, dates, strict=True)], canonical
    if family == "fraction-order":
        while True:
            left, right = rng.sample(range(171, 311), 2)
            first = Fraction(2 * left + rng.randrange(2, left - 1), left)
            second = Fraction(int(first * right) + rng.choice((-2, -1, 1, 2)), right)
            if 2 < second < 3 and first != second:
                break
        fractions = sorted((first, second), reverse=True)
        factors = ((1, 1), (2, 1), (1, 2))[index % 3]
        answers = [f"{number.numerator * scale}/{number.denominator * scale}"
                   for number, scale in zip(fractions, factors, strict=True)]
        canonical = list(map(str, fractions))
        return answers, [[answer, value] for answer, value in zip(answers, canonical, strict=True)], canonical
    raise ValueError(f"Unknown family {family}")


def _question(seed, split, family, index, used):
    rng = random.Random(f"{seed}|{split}|{family}|{index}")
    while True:
        answers, aliases, canonical = _choices(family, rng, index)
        key = hashlib.sha256(json.dumps([family, sorted(canonical)], separators=(",", ":")).encode()).hexdigest()
        if key not in used:
            used.add(key)
            break
    item_id = f"{split}/{family}/{key[:16]}"
    questions = dict(zip(CONTEXTS, (*QUESTIONS[family], "Consider the two options below.",
                                   "Consider the two options below. Quartz is a mineral."), strict=True))
    rows = []
    for order in ORDERS:
        values = [answers[i] for i in order]
        forms = [aliases[i] for i in order]
        bullets = "\n".join(f"- {value}" for value in values)
        for labels in LABELS:
            options = "\n".join(f"{label}. {value}" for label, value in zip(labels, values, strict=True))
            rows.append({
                "order": list(order), "labels": list(labels), "answers": values, "aliases": forms,
                "prompts": {key: _format_prompt(question, options) for key, question in questions.items()},
                "letter_prompts": {
                    key: _format_prompt(questions[key], options, f"Reply with only the label {labels[0]} or {labels[1]}.")
                    for key in ("task", "reversed")
                },
                "generation_prompts": {
                    key: _format_prompt(questions[key], bullets) for key in ("task", "reversed")
                },
            })
    return {"id": item_id, "family": family, "semantic_key": key, "rows": rows}


def freeze(output):
    config = json.loads(PROTOCOL.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    used = set()
    artifacts = {}
    for split in (*SPLITS, RESERVE):
        seed = config["practice_seed"] if split == "practice" else secrets.token_hex(32)
        items = [_question(seed, split, family, index, used) for family in FAMILIES
                 for index in range(config["items_per_family"][split])]
        path = output / f"{split}.json"
        json_dump_atomic(path, {"split": split, "items": items})
        artifacts[split] = {"sha256": file_hash(path), "underlying_items": len(items),
                            "seed_commitment": hashlib.sha256(seed.encode()).hexdigest()}
    manifest = {"schema": "weightbridge.phase_iiie.qwen_roster.v1", "protocol_sha256": file_hash(PROTOCOL),
                "roster_source_sha256": file_hash(Path(__file__)), "artifacts": artifacts}
    json_dump_atomic(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().out), indent=2))
