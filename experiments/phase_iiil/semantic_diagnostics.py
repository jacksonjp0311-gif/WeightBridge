"""Analyze complete decimal responses over order, wording, and truth reversals."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import random

from weightbridge.adaptation.behavior_endpoint import canonical_decimal

from .semantic_roster import GROUPS


def _distribution():
    return {"presentations": 0, "completed": 0, "valid": 0, "correct": 0, "first_operand": 0,
            "second_operand": 0, "shorter_operand": 0, "longer_operand": 0,
            "competing_operand": 0, "teacher_preferences": 0, "teacher_correct": 0,
            "teacher_greedy_disagreements": 0}


def _measure(cell, row, teacher):
    cell["presentations"] += 1
    for key in ("completed", "valid", "correct"):
        cell[key] += row[key]
    if row["valid"]:
        options = [canonical_decimal(value) for value in row["operands"]]
        chosen = options.index(row["exact"])
        cell["first_operand" if chosen == 0 else "second_operand"] += 1
        lengths = [len(value.split(".")[1]) for value in row["operands"]]
        cell["shorter_operand" if lengths[chosen] == min(lengths) else "longer_operand"] += 1
        cell["competing_operand"] += not row["correct"]
        if teacher is not None:
            cell["teacher_greedy_disagreements"] += teacher != row["exact"]
    if teacher is not None:
        cell["teacher_preferences"] += 1
        cell["teacher_correct"] += teacher == canonical_decimal(row["answer"])


def _stable(rows):
    return all(row["valid"] for row in rows) and len({row["exact"] for row in rows}) == 1


def _common_prefix_length(first, second):
    for index, (left, right) in enumerate(zip(first, second)):
        if left != right:
            return index
    return min(len(first), len(second))


def _kind(token, eos_id, text):
    if token is None:
        return "missing"
    if token == eos_id:
        return "eos"
    if "\n" in text or "\r" in text:
        return "newline"
    if text.strip().isascii() and text.strip().isdigit():
        return "digit"
    return "other"


def describe_fork_tokens(fork, tokenizer, row):
    """Describe the native fork and its relation to the displayed answer strings."""
    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=False)

    accepted = [(token, decode([token])) for token in fork.accepted_token_ids]
    actual = "" if fork.actual_token_id is None else decode([fork.actual_token_id])
    shared = decode(fork.shared_token_ids)
    eos_id = fork.decoding["eos_token_id"]
    expected = row["answer"]
    rival = next(value for value in row["operands"] if value != expected)
    distinction = _common_prefix_length(expected, rival)
    expected_char = expected[distinction:distinction + 1]
    rival_char = rival[distinction:distinction + 1]
    cleaned = shared.strip()
    prefix_preserves_answer_surface = expected.startswith(cleaned) and rival.startswith(cleaned)
    first_digit = bool(expected_char.isdigit() and rival_char.isdigit())
    actual_continuation = (shared + actual).strip()
    distinguished = (first_digit and prefix_preserves_answer_surface and len(cleaned) <= distinction
                     and actual_continuation.startswith(rival[:distinction + 1])
                     and any((shared + text).strip().startswith(expected[:distinction + 1]) for _, text in accepted))
    return {"fork_status": fork.status, "replay_status": fork.replay_status, "shared_text": shared,
            "shared_tokens": len(fork.shared_token_ids), "actual_token": fork.actual_token_id,
            "actual_text": actual, "actual_kind": _kind(fork.actual_token_id, eos_id, actual),
            "accepted_tokens": [{"id": token, "text": text, "kind": _kind(token, eos_id, text)}
                                for token, text in accepted],
            "expected_text_distinction_index": distinction, "expected_distinguishing_char": expected_char,
            "competing_distinguishing_char": rival_char,
            "at_first_distinguishing_digit": bool(distinguished),
            "starts_at_text_distinction": bool(prefix_preserves_answer_surface and len(cleaned) == distinction),
            "at_answer_length_boundary": bool(prefix_preserves_answer_surface and len(cleaned) == distinction
                                              and (not expected_char or not rival_char)),
            "expected_surface_token_ids": tokenizer.encode(expected, add_special_tokens=False),
            "competing_surface_token_ids": tokenizer.encode(rival, add_special_tokens=False)}


def summarize(measurement, *, teacher_preferences=None, fork_tokens=None):
    """Summarize joint semantic correctness without counting paired forms as independent."""
    teachers = teacher_preferences or {}
    forks = fork_tokens or {}
    rows = measurement["rows"]
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows) or not (teachers.keys() | forks.keys()) <= ids:
        raise ValueError("Duplicate generation or diagnostic outside the measured roster")
    subgroups = {group: _distribution() for group in GROUPS}
    wording = {str(value): _distribution() for value in (0, 1)}
    order = {str(value): _distribution() for value in (0, 1)}
    statuses = Counter(row["status"] for row in rows)
    controls = {group: {"presentations": 0, "valid": 0, "correct": 0, "items": 0, "solved": 0}
                for group in ("positive-far", "mixed-sign", "copy")}
    comparisons, control_items = defaultdict(list), defaultdict(list)
    token_count = Counter()
    observations = []
    for row in rows:
        if row["group"] != "target":
            control_items[row["comparison_id"] or row["item"]].append(row)
            continue
        teacher = teachers.get(row["id"])
        if teacher is not None and teacher not in {canonical_decimal(value) for value in row["operands"]}:
            raise ValueError("Teacher-forced preference does not identify a displayed option")
        group = f"{row['tier']}/{row['truth_precision']}"
        for cell in (subgroups[group], wording[str(row["wording"])], order[str(row["order"])]):
            _measure(cell, row, teacher)
        comparisons[row["comparison_id"]].append(row)
        fork = forks.get(row["id"])
        if fork is not None:
            token_count[fork["actual_kind"]] += 1
        observations.append({"id": row["id"], "pair_id": row["pair_id"], "group": group,
                             "status": row["status"], "generated_choice": row["exact"],
                             "teacher_choice": teacher, "fork": fork})
    for forms in control_items.values():
        group = forms[0]["tier"] or "copy"
        control = controls[group]
        control["items"] += 1
        control["solved"] += all(row["correct"] for row in forms)
        control["presentations"] += len(forms)
        for key in ("valid", "correct"):
            control[key] += sum(row[key] for row in forms)
    comparison_rows, families = {}, defaultdict(dict)
    eligible_fit_items = {}
    all_four_repeatable = []
    for comparison_id, forms in comparisons.items():
        indexed = {(row["order"], row["wording"]): row for row in forms}
        if len(forms) != 4 or len(indexed) != 4:
            raise ValueError("Semantic diagnostics require both operand orders and both wordings")
        exemplar = indexed[(0, 0)]
        by_order = [[indexed[(order_id, wording_id)] for wording_id in (0, 1)] for order_id in (0, 1)]
        by_wording = [[indexed[(order_id, wording_id)] for order_id in (0, 1)] for wording_id in (0, 1)]
        eligible = [pair[0]["item"] for pair in by_order if _stable(pair) and not any(row["correct"] for row in pair)]
        if eligible:
            eligible_fit_items[comparison_id] = eligible
        if _stable(forms) and not any(row["correct"] for row in forms):
            all_four_repeatable.append(comparison_id)
        comparison_rows[comparison_id] = {
            "pair_id": exemplar["pair_id"], "counterfactual": exemplar["counterfactual"],
            "group": f"{exemplar['tier']}/{exemplar['truth_precision']}",
            "all_correct": all(row["correct"] for row in forms),
            "correct_presentations": sum(row["correct"] for row in forms),
            "all_valid": all(row["valid"] for row in forms),
            "order_consistent": all(_stable(pair) for pair in by_wording),
            "wording_consistent": all(_stable(pair) for pair in by_order),
            "response_ids": [row["id"] for row in forms],
        }
        families[exemplar["pair_id"]][exemplar["counterfactual"]] = comparison_id
    matrix = dict.fromkeys(("both_correct", "only_original_correct", "only_counterfactual_correct", "neither_correct"), 0)
    family_rows = {}
    for pair_id, variants in families.items():
        if set(variants) != {0, 1}:
            raise ValueError("Incomplete truth-flipping family in measured roster")
        original, counterfactual = (comparison_rows[variants[index]]["all_correct"] for index in (0, 1))
        status = {(True, True): "both_correct", (True, False): "only_original_correct",
                  (False, True): "only_counterfactual_correct", (False, False): "neither_correct"}[original, counterfactual]
        matrix[status] += 1
        family_rows[pair_id] = {"original": variants[0], "counterfactual": variants[1], "status": status}
    return {"presentations": len(rows), "status_counts": dict(sorted(statuses.items())),
            "valid_presentations": sum(row["valid"] for row in rows),
            "completed_presentations": sum(row["completed"] for row in rows),
            "target_comparisons": len(comparisons),
            "target_correct": sum(row["all_correct"] for row in comparison_rows.values()),
            "wording_consistent": sum(row["wording_consistent"] for row in comparison_rows.values()),
            "order_consistent": sum(row["order_consistent"] for row in comparison_rows.values()),
            "pair_families": len(families), "truth_flip_matrix": matrix, "families": family_rows,
            "comparisons": comparison_rows, "subgroups": subgroups, "by_wording": wording, "by_order": order,
            "controls": controls, "repeatable_errors": sorted(eligible_fit_items),
            "all_four_repeatable_errors": sorted(all_four_repeatable), "eligible_fit_items": eligible_fit_items,
            "fork_token_kinds": dict(sorted(token_count.items())), "target_observations": observations}


def _clustered_interval(changes):
    if len(changes) < 2:
        return None
    rng = random.Random(1996)
    size = len(changes)
    samples = sorted(sum(rng.choices(changes, k=size)) / (2 * size) for _ in range(4000))
    return {"low": samples[99], "high": samples[3899], "clusters": size,
            "resamples": 4000, "estimand": "paired target accuracy change"}


def compare_semantic(parent: Mapping, candidate: Mapping, *, min_comparison_gain=1, min_repeatable_corrections=0):
    """Require gain across both orders and wordings and preserve every correct response."""
    if (type(min_comparison_gain) is not int or min_comparison_gain < 1
            or type(min_repeatable_corrections) is not int or min_repeatable_corrections < 0):
        raise ValueError("Semantic gate requires positive gain and nonnegative repeatable correction threshold")
    keys = ("id", "item", "family", "group", "tier", "truth_precision", "truth_side", "operands", "prompt", "answer",
            "comparison_id", "pair_id", "counterfactual", "order", "wording")
    if (parent["endpoint"] != candidate["endpoint"] or [tuple(row[key] for key in keys) for row in parent["rows"]]
            != [tuple(row[key] for key in keys) for row in candidate["rows"]]):
        raise ValueError("Semantic comparison requires identical frozen presentation rows and endpoint")
    before, after = summarize(parent), summarize(candidate)
    regressions, lost_valid = [], []
    for initial, edited in zip(parent["rows"], candidate["rows"], strict=True):
        if initial["correct"] and not edited["correct"]:
            regressions.append(initial["id"])
        if initial["valid"] and not edited["valid"]:
            lost_valid.append(initial["id"])
    changes = {item: int(after["comparisons"][item]["all_correct"]) - int(record["all_correct"])
               for item, record in before["comparisons"].items()}
    corrections = sorted(item for item, change in changes.items() if change == 1)
    repeatable = sorted(set(corrections) & set(before["repeatable_errors"]))
    gains = sum(changes.values())
    family_changes = [changes[record["original"]] + changes[record["counterfactual"]]
                      for record in before["families"].values()]
    return {"passed": gains >= min_comparison_gain and len(repeatable) >= min_repeatable_corrections
                      and not regressions and not lost_valid,
            "target_gain": gains, "parent_target_correct": before["target_correct"],
            "candidate_target_correct": after["target_correct"], "target_comparisons": before["target_comparisons"],
            "corrected_target_comparisons": corrections, "corrected_repeatable_target_comparisons": repeatable,
            "lost_target_comparisons": sorted(item for item, change in changes.items() if change == -1),
            "regressions": regressions, "lost_valid": lost_valid,
            "parent_truth_flip_matrix": before["truth_flip_matrix"], "candidate_truth_flip_matrix": after["truth_flip_matrix"],
            "family_gain_interval": _clustered_interval(family_changes)}
