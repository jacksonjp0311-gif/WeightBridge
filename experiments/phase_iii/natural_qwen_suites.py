"""Author and freeze disjoint next-token suites for naturally occurring Qwen errors."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.evaluation.hf_edit import _score_checkpoint
from weightbridge.inspection.safetensors import load_checkpoint


PINNED_SHARD = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
SEED = "weightbridge-phase-iii-natural-qwen-v1"
PROBE_EXAMPLES = 12
SOURCE_PATHS = (
    "experiments/phase_iii/natural_qwen_suites.py", "experiments/phase_iii/run_natural_qwen_study.py",
    "weightbridge/adaptation/awl.py", "weightbridge/adaptation/controller.py",
    "weightbridge/adaptation/lineage.py", "weightbridge/adaptation/synthesis.py",
    "weightbridge/agents/hf_direction.py", "weightbridge/agents/hf_session.py",
    "weightbridge/evaluation/hf_edit.py", "weightbridge/inspection/safetensors.py",
    "weightbridge/patches/huggingface.py",
)
FAMILIES = (
    "signed-arithmetic", "overlap-count", "weekday-offset",
    "bit-parity", "modular-arithmetic", "conjunction-count",
    "decimal-order", "signed-order", "word-order", "date-order",
    "fraction-order", "duration-order", "string-length",
)
ORDER_FAMILIES = FAMILIES[6:]
TENSORS = [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in (23, 21, 19)]
INFERENCE = {"device": "cpu", "dtype": "bfloat16", "cpu_threads": 6,
             "batch_size": 8, "max_input_tokens": 128}
POLICY = {
    "schema": "weightbridge.policy.v1", "allow_tensors": TENSORS,
    "max_abs_delta": 8.0, "max_entries": 65536, "max_fraction": 0.00008,
    "max_tensor_bytes": 8388608, "max_total_tensor_bytes": 8388608,
}


def question(family, rng):
    if family == "signed-arithmetic":
        a, b, c = rng.randint(100, 199), rng.randint(13, 43), rng.randint(3, 9)
        return f"Using standard precedence, compute {a} - (-{b}) * {c}.", a + b * c, (a + b) * c
    if family == "overlap-count":
        a, b, extra = rng.sample("ABCDEFGHKMNPQRSTVWXYZ", 3)
        unit = a + b
        needle = unit * rng.choice((2, 3, 4))
        haystack = unit * rng.randint(6, 11) + extra + unit * rng.randint(3, 7)
        answer = sum(haystack.startswith(needle, i) for i in range(len(haystack)))
        return (
            f"In {haystack}, how many times does {needle} occur as a contiguous substring? "
            "Count overlapping occurrences.",
            answer,
            haystack.count(needle),
        )
    if family == "weekday-offset":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        i, back, forward = rng.randrange(7), rng.randint(10, 48), rng.randint(12, 54)
        return (
            f"Start on {days[i]}, go back {back} days, then go forward {forward} days. "
            "What weekday do you land on?",
            days[(i - back + forward) % 7],
            days[(i - back + forward + rng.choice((-1, 1))) % 7],
        )
    if family == "bit-parity":
        bits = [rng.randrange(2) for _ in range(rng.randint(13, 19))]
        answer = sum(bits) % 2
        return f"What is the XOR of these bits? {', '.join(map(str, bits))}.", answer, 1 - answer
    if family == "modular-arithmetic":
        a, b, c = rng.randint(11, 61), rng.randint(12, 58), rng.randint(1, 49)
        p = rng.choice((7, 11, 13, 17, 19))
        r = (a * b + c) % p
        return f"What is the least nonnegative remainder when ({a} * {b} + {c}) is divided by {p}?", r, (r + 1) % p
    if family == "conjunction-count":
        cutoff = rng.randrange(11, 69, 2)
        values = [cutoff, *rng.sample([value for value in range(11, 69) if value != cutoff], 11)]
        rng.shuffle(values)
        answer = sum(value > cutoff and value % 2 == 1 for value in values)
        return (
            f"From {', '.join(map(str, values))}, count values that are odd and strictly greater than {cutoff}.",
            answer,
            answer + 1,
        )
    if family == "decimal-order":
        whole = rng.randint(1, 99)
        short = f"{whole}.{rng.randint(1, 9)}"
        hundredths = rng.randint(11, 99)
        long = f"{whole}.{hundredths:02}"
        if Decimal(short) == Decimal(long):
            long = f"{whole}.{hundredths + 1:02}"
        larger, smaller = sorted((short, long), key=Decimal, reverse=True)
        return "Which decimal number is larger?", larger, smaller
    if family == "signed-order":
        a, b = rng.sample(range(-99, 0), 2)
        return "Which integer is larger?", max(a, b), min(a, b)
    if family == "word-order":
        start = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=3))
        a, b = (start + ending for ending in rng.sample("abcdefghjkmnpqrstuvwxy", 2))
        return "Which of these strings comes first in alphabetical order?", min(a, b), max(a, b)
    if family == "date-order":
        origin = date(rng.randint(1982, 2025), 1, 1)
        a, b = sorted(origin + timedelta(days=i) for i in rng.sample(range(360), 2))
        months = ("January", "February", "March", "April", "May", "June", "July", "August",
                  "September", "October", "November", "December")
        formatted = [f"{item.day:02d} {months[item.month - 1]} {item.year}" for item in (a, b)]
        return "Which calendar date is earlier?", *formatted
    if family == "fraction-order":
        a, b = rng.sample(range(2, 12), 2)
        d = rng.randint(max(a, b) + 1, 17)
        other = rng.randint(b + 1, 18)
        choices = ((a, d), (b, other))
        ranked = sorted(choices, key=lambda item: Fraction(*item), reverse=True)
        if Fraction(*ranked[0]) == Fraction(*ranked[1]):
            return question(family, rng)
        return "Which fraction is larger?", *(f"{x}/{y}" for x, y in ranked)
    if family == "duration-order":
        hour, minute = rng.randint(1, 5), rng.randint(5, 55)
        duration = 60 * hour + minute
        other = duration + rng.choice((-1, 1)) * rng.randint(3, 24)
        choices = ((duration, f"{hour} hour(s) and {minute} minutes"), (other, f"{other} minutes"))
        return "Which duration is longer?", *(item[1] for item in sorted(choices, reverse=True))
    if family == "string-length":
        chars = "abcdefghijklmnopqrstuvwxyz"
        length = rng.randint(4, 12)
        left = "".join(rng.choices(chars, k=length))
        right = "".join(rng.choices(chars, k=length + rng.choice((1, 2))))
        return "Which string contains more letters?", right, left
    if family == "integer-control":
        a, b = rng.sample(range(10, 99), 2)
        return f"Which number is greater, {a} or {b}?", max(a, b), min(a, b)
    if family == "alphabet-control":
        a, b, c = rng.sample("ABCDEFGHIJKLMNOPQRSTUVW", 3)
        return f"In the list {a}, {b}, {c}, which letter comes first in alphabetical order?", min(a, b, c), max(a, b, c)
    raise ValueError(f"Unknown failure family: {family}")


def examples(family, count, split, used):
    rng = random.Random(f"{SEED}|{family}|{split}")
    partition = {"prefix-pilot": 0, "five-cycle": 1}.get(split.split("/")[0])
    order = ["A", "B"] * (count // 2)
    rng.shuffle(order)
    rows = []
    for index, label in enumerate(order, 1):
        while True:
            prompt, answer, distractor = question(family, rng)
            key = (prompt, *sorted((str(answer), str(distractor))))
            bucket = hashlib.sha256(json.dumps(key).encode()).digest()[0] % 2
            if key not in used and answer != distractor and (partition is None or partition == bucket):
                break
        used.add(key)
        a, b = (answer, distractor) if label == "A" else (distractor, answer)
        ending = "Answer with the letter A or B only." if family in ORDER_FAMILIES else "Reply with only A or B."
        rows.append({"id": f"{split}/{family}/{index:02d}",
                     "text": f"{prompt}\nA. {a}\nB. {b}\n{ending}",
                     "label": label})
    return rows


def suite(targets, sentinels):
    return {"schema": "weightbridge.hf_edit_suite.v1", "choices": ["A", "B"],
            "target": targets, "sentinel": sentinels}


def control_rows(split, used):
    return examples("integer-control", 6, split, used) + examples("alphabet-control", 6, split, used)


def calibration_rows(used):
    rows = {family: examples(family, PROBE_EXAMPLES, "screen", used) for family in FAMILIES}
    return {**rows, "controls": control_rows("screen", used)}


def pinned_base(path):
    base = load_checkpoint(path)
    if len(base.shards) != 1 or file_hash(base.shards[0]) != PINNED_SHARD:
        raise ValueError("Phase III requires the pinned, unmodified Qwen3.5-0.8B shard")
    return base


def screen(base_path, out, families):
    base = pinned_base(base_path)
    used = set()
    by_family = {family: examples(family, PROBE_EXAMPLES, "screen", used) for family in families}
    by_family["controls"] = control_rows("screen", used)
    rows = [row for items in by_family.values() for row in items]
    predictions = _score_checkpoint(base.root, ["A", "B"], rows, **INFERENCE)
    observed = {row["id"]: prediction for row, prediction in zip(rows, predictions, strict=True)}
    results = {}
    for family, items in by_family.items():
        scored = [{**row, "prediction": observed[row["id"]]} for row in items]
        results[family] = {
            "correct": sum(row["label"] == row["prediction"] for row in scored),
            "total": len(scored),
            "prediction_counts": {choice: sum(row["prediction"] == choice for row in scored) for choice in "AB"},
            "rows": scored,
        }
    report = {
        "schema": "weightbridge.phase_iii.calibration.v1", "base_sha256": base.checkpoint_sha256(),
        "inference": INFERENCE, "seed": SEED, "families": results,
    }
    json_dump_atomic(out, report)
    return {family: [result["correct"], result["total"]] for family, result in results.items()}


def materialize(base_path, out, families, *, pilot=False):
    base = pinned_base(base_path)
    expected = 2 if pilot else 5
    if len(families) != expected or len(set(families)) < min(expected, 3):
        raise ValueError(f"Expected {expected} cycles covering distinct natural failure families")
    out.mkdir(parents=True)
    public, sealed = out / "practice", out / "sealed"
    public.mkdir()
    sealed.mkdir()
    used = set()
    calibration_rows(used)
    variant = "prefix-pilot" if pilot else "five-cycle"
    cycles = []
    for i, family in enumerate(families, 1):
        cycle = f"cycle-{i:02d}"
        namespace = f"{variant}/{cycle}"
        practice_path = public / f"{cycle}.json"
        holdout_path = sealed / f"{cycle}.json"
        intent_path = public / f"{cycle}-intent.json"
        for split, path in (("practice", practice_path), ("holdout", holdout_path)):
            split_id = f"{namespace}/{split}"
            json_dump_atomic(path, suite(examples(family, 12, split_id, used), control_rows(split_id, used)))
        strengths = [0.03, 0.07] if pilot or i % 2 else [0.07, 0.03]
        json_dump_atomic(intent_path, {
            "schema": "weightbridge.awl.v1",
            "observe": f"{family} errors in the local model on this practice suite.",
            "diagnose": "Test whether a bounded rank-one edit can correct the signed choice response.",
            "target": TENSORS,
            "optimize": {"method": "activation_ridge", "strengths": strengths},
        })
        cycles.append({
            "id": cycle, "family": family, "practice_sha256": file_hash(practice_path),
            "holdout_sha256": file_hash(holdout_path), "intent_sha256": file_hash(intent_path),
        })
    final_path = sealed / "final.json"
    final_namespace = f"{variant}/final"
    final_targets = [row for family in dict.fromkeys(families) for row in examples(family, 8, final_namespace, used)]
    json_dump_atomic(final_path, suite(final_targets, control_rows(final_namespace, used)))
    protocol = {
        "schema": "weightbridge.phase_iii.protocol.v1", "base_sha256": base.checkpoint_sha256(),
        "policy": POLICY, "inference": INFERENCE, "max_candidates": 2,
        "min_target_gain": 1, "max_sentinel_flips": 0, "max_target_regressions": 0,
        "preservation_penalty": 1.0, "edit_cost_penalty": 1.0,
        "cycles": cycles, "final_sha256": file_hash(final_path),
    }
    protocol_path = out / "protocol.json"
    json_dump_atomic(protocol_path, protocol)
    json_dump_atomic(out / "preregistration.json", {
        "schema": "weightbridge.phase_iii.preregistration.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": file_hash(protocol_path), "namespace": variant,
        "source_sha256": {path: file_hash(ROOT / path) for path in SOURCE_PATHS},
        "checkpoint_shard_sha256": PINNED_SHARD, "prompt_seed": SEED,
        "random_control_seed": 20261103,
    })
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "freeze", "freeze-pilot"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    args = parser.parse_args()
    if args.stage == "screen":
        if args.out.exists():
            parser.error("Screen report already exists")
        result = screen(args.base, args.out, args.families or FAMILIES)
    else:
        if not args.families:
            parser.error("--families is required for a frozen study")
        result = materialize(args.base, args.out, args.families, pilot=args.stage == "freeze-pilot")
    print(json.dumps(result, indent=2, sort_keys=True))
