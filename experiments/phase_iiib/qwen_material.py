"""Freeze fresh paired suites for the Phase III-B Qwen comparison."""
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


DESIGN = ROOT / "benchmarks/configs/phase_iiib_qwen_design_v1.json"
TASK_FAMILIES = ("word-order", "date-order", "fraction-order")
GENERIC_FAMILIES = ("integer-control", "length-control")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ordered(rng, family):
    if family == "word-order":
        # Archived examples are four letters long. Keep the common-prefix
        # comparison, with newly generated six-letter strings.
        prefix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=4))
        a, b = sorted(prefix + "".join(tail) for tail in
                      rng.sample([(x, y) for x in "abcdefg" for y in "pqrstuv"], 2))
        return a, b
    if family == "date-order":
        # The previous calendar roster ends in 2025.
        origin = date(rng.randint(2035, 2095), 1, 1)
        a, b = sorted(origin + timedelta(days=offset) for offset in rng.sample(range(360), 2))
        return a.isoformat(), b.isoformat()
    if family == "fraction-order":
        # Both values are improper, whereas the archived roster uses proper
        # fractions. Canonicalize equivalent fractions before partitioning.
        while True:
            d1, d2 = rng.sample(range(12, 37), 2)
            values = [Fraction(d1 + rng.randint(1, 17), d1), Fraction(d2 + rng.randint(1, 17), d2)]
            if values[0] != values[1]:
                return tuple(f"{f.numerator}/{f.denominator}" for f in sorted(values, reverse=True))
    if family == "integer-control":
        return tuple(map(str, sorted(rng.sample(range(100, 1000), 2), reverse=True)))
    if family == "length-control":
        short, long = rng.sample(range(7, 16), 2)
        return tuple("".join(rng.choices("abcdefghijklmnpqrstuvwxyz", k=n)) for n in sorted((short, long), reverse=True))
    raise ValueError(f"Unknown family {family}")


_QUESTION = {
    "word-order": (
        "Which string comes earlier in alphabetical order?",
        "Sort the two strings alphabetically and identify the one listed first.",
        "Which entry would a dictionary put before the other?",
        "Starting from the first letter, pick the lexicographically smaller string.",
    ),
    "date-order": (
        "Which date is earlier on the calendar?",
        "Pick the date that occurs first in chronological order.",
        "If these are event dates, which event happens first?",
        "Choose the calendar date with fewer days elapsed since the start of its year.",
    ),
    "fraction-order": (
        "Which fraction has the larger value?",
        "Compare these two ratios and select the greater one.",
        "If each option is evaluated as a quotient, which result is bigger?",
        "Identify the larger number represented by these fractions.",
    ),
    "integer-control": (
        "Which integer is greater?",
        "Select the larger number.",
        "Which entry is larger on the number line?",
        "Pick the integer with higher magnitude.",
    ),
    "length-control": (
        "Which string has more letters?",
        "Select the longer sequence of letters.",
        "Count the characters and pick the larger count.",
        "Which entry contains the most letters?",
    ),
}


def _sample(roster, split, family, count, used, partition):
    rng = random.Random(f"{roster}|{split}|{family}")
    rows = []
    while len(rows) < count:
        options = _ordered(rng, family)
        key = _digest([family, sorted(options)])
        if key not in used and int(key[:8], 16) % 3 == partition:
            used.add(key)
            rows.append((family, key, options))
    return rows


def _render(items, choices, split, *, independent):
    rows = []
    for index, (family, key, ordered) in enumerate(items):
        # Counterbalance semantic positions and alternate the order in which
        # label names are presented. The four variants share an item key.
        for position in range(2):
            question = _QUESTION[family][2 * independent + (index + position) % 2]
            first, second = ordered if position == 0 else ordered[::-1]
            # Reverse label presentation for half the underlying items, with
            # one correct option in each printed position for every item.
            labels = choices if index % 2 == 0 else choices[::-1]
            rows.append({
                "id": f"{split}/{key[:16]}/{position}",
                "text": f"{question}\n{labels[0]}. {first}\n{labels[1]}. {second}\n"
                        f"Reply with {choices[0]} or {choices[1]} only.",
                "label": labels[position],
                "family": family if family in TASK_FAMILIES else "generic",
                "item_key": key,
                "answer_position": position,
                "presentation": labels,
            })
    return rows


def _suite(targets, sentinels, choices, split, *, independent):
    return {
        "schema": "weightbridge.hf_edit_suite.v1",
        "choices": choices,
        "target": _render(targets, choices, f"{split}/target", independent=independent),
        "sentinel": _render(sentinels, choices, f"{split}/sentinel", independent=independent),
    }


def materialize(base_path, output, *, pilot=False, roster_index=0):
    design = json.loads(DESIGN.read_bytes())
    checkpoint = load_checkpoint(base_path)
    if (checkpoint.checkpoint_sha256() != design["model_sha256"] or len(checkpoint.shards) != 1
            or file_hash(checkpoint.shards[0]) != design["shard_sha256"]):
        raise ValueError("Phase III-B requires the exact frozen Qwen checkpoint")
    if type(roster_index) is not int or not 0 <= roster_index < len(design["full_rosters"]):
        raise ValueError("Invalid qualification roster index")
    roster = design["pilot_roster"] if pilot else design["full_rosters"][roster_index]
    partition = 0 if pilot else 1 + roster_index
    schedule = design["pilot_cycles"] if pilot else design["cycles"]
    output = Path(output)
    output.mkdir(parents=True)
    used, artifacts, cycles = set(), {}, []

    def save(relative, doc):
        path = output / relative
        json_dump_atomic(path, doc)
        artifacts[relative] = file_hash(path)
        return relative

    def sample(split, family, count):
        return _sample(roster, split, family, count, used, partition)

    def generic(split, count):
        return [sample(f"{split}/{i}", GENERIC_FAMILIES[i % 2], 1)[0] for i in range(count)]

    formats = design["choice_formats"]
    tensors = [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in design["tensor_layers"]]
    for index, family in enumerate(schedule, 1):
        cycle = f"cycle-{index:02d}"
        prefix = f"practice/{cycle}"
        train = sample(f"{cycle}/target", family, design["practice_target_items"])
        generic_train = generic(f"{cycle}/generic", design["practice_sentinel_items"])
        # Freeze a same-size substitution of non-target family practice.
        # These families are available before the cycle, including the first
        # cycle. Their labels are independent of any accepted holdout.
        preservation = generic_train[:2] + [
            sample(f"{cycle}/preservation", previous, 1)[0] for previous in TASK_FAMILIES if previous != family
        ]
        practice = {
            "generic": save(f"{prefix}/generic.json", _suite(
                train, generic_train, formats["ab"], prefix, independent=False,
            )),
            "aware": save(f"{prefix}/task-aware.json", _suite(
                train, preservation, formats["ab"], prefix, independent=False,
            )),
        }
        # Common practice-only finalist screen includes all preservation
        # rows plus the generic controls hidden from task-aware fitting.
        # The same number of screening rows is charged to every arm.
        shared_sentinels = generic_train + preservation[2:]
        practice["screen"] = save(f"{prefix}/screen.json", _suite(
            train, shared_sentinels, formats["ab"], prefix, independent=False,
        ))
        intents = {}
        for arm, method in design["methods"].items():
            preserve = sorted({row[0] if row[0] in TASK_FAMILIES else "generic" for row in
                               (generic_train if arm == "A" else preservation)})
            intents[arm] = save(f"{prefix}/{arm}-intent.json", {
                "schema": "weightbridge.awl.v2", "objective": "binary_choice_margin",
                "target": tensors[:1] if arm in ("A", "B") else tensors,
                "preserve": preserve, "representation": "rank_one_slices",
                "optimize": {"method": method, "radius": design["ridge_strength"]
                             if arm in ("A", "B") else design["frob_cap"]},
            })
        unseen = sample(f"{cycle}/holdout", family, design["holdout_target_items"])
        controls = generic(f"{cycle}/holdout-generic", design["holdout_sentinel_items"])
        holdouts = {name: save(f"sealed/{cycle}-{name}.json", _suite(
            unseen, controls, choices, f"sealed/{cycle}/{name}", independent=True,
        )) for name, choices in formats.items()}
        cycles.append({"id": cycle, "family": family, "practice": practice, "intents": intents, "holdouts": holdouts})

    final_targets = [item for family in TASK_FAMILIES for item in
                     sample("final", family, design["final_target_items_per_family"])]
    final_sentinels = generic("final", design["final_sentinel_items"])
    finals = {name: save(f"sealed/final-{name}.json", _suite(
        final_targets, final_sentinels, choices, f"sealed/final/{name}", independent=True,
    )) for name, choices in formats.items()}
    policy = {"schema": "weightbridge.policy.v1", "allow_tensors": tensors}
    policy.update({key: design[key] for key in (
        "max_entries", "max_fraction", "max_abs_delta", "max_tensor_bytes", "max_total_tensor_bytes",
    )})
    protocol = {
        "schema": "weightbridge.phase_iiib.qwen_protocol.v1", "roster": roster,
        "scope": "bounded-pilot" if pilot else "five-cycle-qualification",
        "design_sha256": file_hash(DESIGN), "design": design,
        "model_sha256": design["model_sha256"], "policy": policy,
        "cycles": cycles, "final": finals, "artifacts": artifacts,
        "generator_sha256": file_hash(Path(__file__)),
        "item_count": len(used),
    }
    json_dump_atomic(output / "protocol.json", protocol)
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--roster-index", type=int, default=0)
    args = parser.parse_args()
    result = materialize(args.base, args.out, pilot=args.pilot, roster_index=args.roster_index)
    print(json.dumps({"scope": result["scope"], "roster": result["roster"], "items": result["item_count"],
                      "protocol_sha256": file_hash(args.out / "protocol.json")}, indent=2))
