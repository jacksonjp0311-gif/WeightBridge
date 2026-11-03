"""Prospectively freeze the Phase III-D question populations and prompt variants."""
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


PROTOCOL = Path(__file__).with_name("protocol.json")
SCHEMA = "weightbridge.phase_iiid.grounding_roster.v1"
MANIFEST_SCHEMA = "weightbridge.phase_iiid.grounding_manifest.v1"
FAMILIES = ("word-order", "date-order", "fraction-order")
SPLITS = ("practice", "selection", "qualification", "sealed-adaptation")
PARAPHRASES = {
    "word-order": (
        ("Which string comes first in alphabetical order?", "Which string comes last in alphabetical order?"),
        ("Sort the two strings lexicographically. Which is smaller?", "Sort the two strings lexicographically. Which is larger?"),
    ),
    "date-order": (
        ("Which calendar date is earlier?", "Which calendar date is later?"),
        ("In chronological order, which date would be first?", "In chronological order, which date would be last?"),
    ),
    "fraction-order": (
        ("Which fraction has the larger value?", "Which fraction has the smaller value?"),
        ("Compare the two quotients. Which is greater?", "Compare the two quotients. Which is less?"),
    ),
}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
CONTEXTS = {
    "neutral_absent": "",
    "neutral_generic": "Consider the two options below.",
    "irrelevant": "The word marble has six letters.",
}
INSTRUCTION = "Reply with only the exact text of one option."


def _date_text(value: date, style: int) -> str:
    if style == 0:
        return f"{value.day} {MONTHS[value.month - 1]} {value.year}"
    return f"{MONTHS[value.month - 1]} {value.day}, {value.year}"


def _sample(rng: random.Random, family: str, index: int):
    if family == "word-order":
        # Fresh seven-character synthetic strings. For unequal lengths, alternate
        # which canonical option is longer without changing the first differing letter.
        prefix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        low, high = sorted(rng.sample("bdfhjlnprtvxz", 2))
        lengths = ((2, 2), (2, 1), (1, 2))[index % 3]
        answers = [prefix + letter + "".join(rng.choices("aeioukmn", k=length))
                   for letter, length in zip((low, high), lengths, strict=True)]
        return answers, [[answer] for answer in answers], answers
    if family == "date-order":
        # Fresh year range and cross-month options. Some examples cross years.
        origin = date(rng.randint(2310, 2369), 1, 1)
        ordered = (origin + timedelta(days=rng.randint(21, 170)),
                   origin + timedelta(days=rng.randint(200, 350) + 365 * (index % 2)))
        answers = [_date_text(value, index % 2) for value in ordered]
        aliases = [[answer, _date_text(value, 1 - index % 2), value.isoformat()]
                   for answer, value in zip(answers, ordered, strict=True)]
        return answers, aliases, [value.isoformat() for value in ordered]
    if family == "fraction-order":
        # Fractions in (1, 2), unlike III-C's (3, 4). Cross multiplication
        # supplies exact ground truth. Alternating reduction changes token lengths.
        denominator = rng.randint(161, 307)
        first = Fraction(denominator + rng.randint(2, denominator - 2), denominator)
        denominator += rng.choice((-19, -7, 11, 31))
        quotient = first * denominator
        second = Fraction(quotient.numerator // quotient.denominator + rng.choice((-2, -1, 1, 2)), denominator)
        if not 1 < second < 2 or first == second:
            return _sample(rng, family, index)
        ordered = sorted((first, second), reverse=True)
        canonical = [str(value) for value in ordered]
        factors = ((1, 1), (2, 1), (1, 2))[index % 3]
        answers = [f"{value.numerator * factor}/{value.denominator * factor}"
                   for value, factor in zip(ordered, factors, strict=True)]
        aliases = [list(dict.fromkeys((answer, key))) for answer, key in zip(answers, canonical, strict=True)]
        return answers, aliases, canonical
    raise ValueError(f"Unknown family: {family}")


def _format_prompt(question: str, options: str, instruction: str = INSTRUCTION) -> str:
    return f"{question}\n{options}\n{instruction}" if question else f"{options}\n{instruction}"


def _question(seed: str, split: str, family: str, index: int, used: set[str]):
    split_index = SPLITS.index(split)
    rng = random.Random(f"{seed}|{split}|{family}|{index}")
    while True:
        answers, aliases, canonical = _sample(rng, family, index + split_index)
        key = hashlib.sha256(json.dumps([family, sorted(canonical)], separators=(",", ":")).encode()).hexdigest()
        if key not in used:
            used.add(key)
            break
    item_id = f"{split}/{family}/{key[:16]}"
    labels = ("A", "B") if index % 2 == 0 else ("X", "Y")
    if (index + split_index + FAMILIES.index(family)) % 2:
        labels = labels[::-1]
    presentations = []
    for paraphrase, (task, counterfactual) in enumerate(PARAPHRASES[family]):
        for order in ([0, 1], [1, 0]):
            first, second = [answers[choice] for choice in order]
            labeled = f"{labels[0]}. {first}\n{labels[1]}. {second}"
            bullets = f"- {first}\n- {second}"
            displays = {
                display: {kind: _format_prompt(question, options) for kind, question in
                          {"task": task, "counterfactual": counterfactual, **CONTEXTS}.items()}
                for display, options in {"labeled": labeled, "bullets": bullets}.items()
            }
            displays["labeled"]["letter_prompt"] = _format_prompt(
                task, labeled, f"Reply with only the label {labels[0]} or {labels[1]}.")
            presentations.append({
                "id": f"{item_id}/p{paraphrase}-o{order[0]}", "order": order, "labels": list(labels),
                "paraphrase": paraphrase, "displays": displays,
            })
    return {"id": item_id, "family": family, "semantic_key": key, "answers": answers, "aliases": aliases,
            "presentations": presentations}


def freeze(base: Path, out: Path):
    config = json.loads(PROTOCOL.read_bytes())
    if load_checkpoint(base).checkpoint_sha256() != config["model_sha256"]:
        raise ValueError("Phase III-D requires the pinned Qwen checkpoint")
    out = Path(out)
    # Never rewrite a freeze, including a partial freeze. A new attempt needs a new directory.
    out.mkdir(parents=True, exist_ok=False)
    used: set[str] = set()
    artifacts = {}
    for split in SPLITS:
        count = config["underlying_items_per_family"][split]
        rows = [_question(config["seed"], split, family, index, used)
                for family in FAMILIES for index in range(count)]
        path = out / f"{split}.json"
        json_dump_atomic(path, {"schema": SCHEMA, "split": split, "items": rows})
        artifacts[split] = {"sha256": file_hash(path), "underlying_items": len(rows)}
    manifest = {"schema": MANIFEST_SCHEMA, "model_sha256": config["model_sha256"],
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
