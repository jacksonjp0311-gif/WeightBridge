"""Q baseline endpoint: historical decimal grammar plus declared arithmetic controls."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from decimal import Decimal
import re

from weightbridge.adaptation.behavior_endpoint import (
    Question, canonical_decimal, decimal_endpoint, decimal_question, measure_endpoint,
)


def question(row):
    if row.get("family") in ("copy", "larger-decimal"):
        return decimal_question(row)
    if row.get("family") not in ("integer-add", "integer-successor") or row.get("group") != "preserve":
        raise ValueError("Unsupported supervised-baseline control")
    operands = row.get("operands")
    prompt = row.get("prompt", "")
    if not isinstance(operands, (list, tuple)) or not all(re.fullmatch(r"-?\d+", str(x)) for x in operands):
        raise ValueError("Arithmetic controls require exact integers")
    values = tuple(Decimal(x) for x in operands)
    tail = "\n".join(prompt.splitlines()[-2:])
    if row["family"] == "integer-add":
        if len(values) != 2 or tail != f"Integers: {operands[0]} and {operands[1]}\nAnswer:":
            raise ValueError("Addition control does not match displayed operands")
        answer = values[0] + values[1]
    else:
        if len(values) != 1 or tail != f"Integer: {operands[0]}\nAnswer:":
            raise ValueError("Successor control does not match displayed operand")
        answer = values[0] + 1
    expected = canonical_decimal(str(answer))
    if canonical_decimal(row.get("answer")) != expected:
        raise ValueError("Incorrect arithmetic control ground truth")
    return Question(expected)


def endpoint(settings):
    return replace(decimal_endpoint(settings), endpoint_id="weightbridge.supervised_decimal_first_line.v1",
                   question=question)


def measure(runner, rows, settings):
    runner.model.eval()
    return measure_endpoint(runner, rows, endpoint(settings))


def summarize(measurement):
    rows = measurement["rows"]
    comparisons, families = defaultdict(list), defaultdict(set)
    for row in rows:
        if row["group"] == "target":
            comparisons[row["comparison_id"]].append(row)
            families[row["pair_id"]].add(row["comparison_id"])
    if any(len(forms) != 4 or {(r["wording"], r["order"]) for r in forms}
           != {(0, 0), (0, 1), (1, 0), (1, 1)} for forms in comparisons.values()):
        raise ValueError("Strict comparisons require all four distinct presentations")
    if any(len(ids) != 2 for ids in families.values()):
        raise ValueError("Counterfactual families require two opposing comparisons")
    strict = {key for key, values in comparisons.items() if all(row["correct"] for row in values)}
    complete = {key for key, ids in families.items() if ids <= strict}
    groups = {}
    def group_key(row):
        return row.get("control_kind") or row.get("tier") or row["family"]
    for name in sorted({group_key(row) for row in rows}):
        subset = [row for row in rows if group_key(row) == name]
        groups[name] = {"presentations": len(subset), "correct": sum(r["correct"] for r in subset),
                        "valid": sum(r["valid"] for r in subset)}
    return {"presentations": len(rows), "correct": sum(r["correct"] for r in rows),
            "valid_completed_first_lines": sum(r["valid"] for r in rows),
            "completed_first_lines": sum(r["completed"] for r in rows),
            "decode_termination": dict(Counter(r["stop_reason"] for r in rows)),
            "strict_comparisons": len(strict), "comparisons": len(comparisons),
            "complete_families": len(complete), "families": len(families),
            "strict_comparison_ids": sorted(strict), "complete_family_ids": sorted(complete),
            "groups": groups}


def compare(parent, candidate, *, min_strict_gain=1, min_family_gain=1, min_repeatable=0):
    prior, post = parent["rows"], candidate["rows"]
    identity = ("id", "item", "prompt", "answer", "wording", "group", "family")
    if parent["endpoint"] != candidate["endpoint"] or [tuple(r[k] for k in identity) for r in prior] != [
            tuple(r[k] for k in identity) for r in post] or [
            tuple(str(r.get(k)) for k in ("pair_id", "comparison_id", "order", "operands")) for r in prior] != [
            tuple(str(r.get(k)) for k in ("pair_id", "comparison_id", "order", "operands")) for r in post]:
        raise ValueError("Cannot compare different frozen endpoints or rosters")
    before, after = summarize(parent), summarize(candidate)
    regressions = [a["id"] for a, b in zip(prior, post, strict=True) if a["correct"] and not b["correct"]]
    lost_valid = [a["id"] for a, b in zip(prior, post, strict=True) if a["valid"] and not b["valid"]]
    target_before, target_after = defaultdict(list), defaultdict(list)
    for a, b in zip(prior, post, strict=True):
        if a["group"] == "target":
            target_before[a["comparison_id"]].append(a)
            target_after[b["comparison_id"]].append(b)
    repeated = [key for key, rows in target_before.items()
                if all(r["status"] == "incorrect" for r in rows)
                and all(r["correct"] for r in target_after[key])]
    strict_gain = after["strict_comparisons"] - before["strict_comparisons"]
    family_gain = after["complete_families"] - before["complete_families"]
    return {"passed": not regressions and not lost_valid and strict_gain >= min_strict_gain
                      and family_gain >= min_family_gain and len(repeated) >= min_repeatable,
            "strict_comparison_gain": strict_gain, "complete_family_gain": family_gain,
            "correct_response_gain": after["correct"] - before["correct"],
            "newly_complete_families": sorted(set(after["complete_family_ids"]) - set(before["complete_family_ids"])),
            "repeatable_comparisons_corrected": repeated, "prohibited_regressions": regressions,
            "lost_valid_presentations": lost_valid, "parent": before, "candidate": after}
