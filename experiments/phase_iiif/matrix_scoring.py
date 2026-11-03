"""Fixed conditional and generated-answer measurement for both model families."""
from __future__ import annotations

from collections import defaultdict
import time

from weightbridge.adaptation.content_scoring import (
    generate_responses, prepare_continuations, score_continuations,
)
from weightbridge.provenance import peak_rss_bytes


def _normalize(answer):
    return answer.strip().casefold()


def evaluate(runner, rows, inference):
    wall, cpu = time.perf_counter(), time.process_time()
    prompts = [row["prompt"] for row in rows]
    # Reverse score-list presentation independently of the actual correct answer.
    # The scoring arms receive identical prompts and rank ties are abstentions.
    options = [row["answers"] if index % 2 == 0 else row["answers"][::-1]
               for index, row in enumerate(rows)]
    continuations = prepare_continuations(
        runner.tokenizer, prompts, options, max_input_tokens=inference["max_input_tokens"],
    )
    scored, score_cost = score_continuations(runner, continuations, batch_size=inference["batch_size"])
    generated, generation_cost = generate_responses(
        runner, prompts, max_input_tokens=inference["max_input_tokens"],
        max_new_tokens=inference["max_new_tokens"], batch_size=inference["batch_size"],
    )
    predictions = []
    for index, (row, labels, response) in enumerate(zip(rows, options, generated, strict=True)):
        choices = tuple(map(_normalize, labels))
        values = scored[2 * index:2 * index + 2]
        left, right = (value["sum_log_probability"] for value in values)
        ranker = choices[0] if left > right else choices[1] if left < right else None
        generated_label = _normalize(response["text"])
        exact = generated_label if response["stop_reason"] == "eos" and generated_label in choices else None
        predictions.append({
            "id": row["id"], "item": row["item"], "wording": row["wording"], "family": row["family"],
            "preserve": row["preserve"], "correct": _normalize(row["answers"][0]), "ranker": ranker,
            "exact": exact, "choices": list(choices), "score": values,
            "answer_token_ids": [list(value.answer_ids) for value in continuations[2 * index:2 * index + 2]],
            "generation": response,
        })
    return {"rows": predictions, "summary": summarize(predictions),
            "resources": {"scoring": score_cost, "generation": generation_cost,
                          "process_cpu_seconds": time.process_time() - cpu,
                          "wall_seconds": time.perf_counter() - wall, "peak_rss_bytes": peak_rss_bytes()}}


def _by_item(rows):
    items = defaultdict(list)
    for row in rows:
        items[row["item"]].append(row)
    if any(sorted(row["wording"] for row in item) != [0, 1] for item in items.values()):
        raise ValueError("Each independent item requires exactly two distinct wordings")
    return items


def summarize(rows):
    by_family = {}
    for family in ("parity", "sign"):
        members = [row for row in rows if row["family"] == family]
        items = _by_item(members)
        by_family[family] = {
            "items": len(items), "presentations": len(members),
            "ranker_correct_presentations": sum(row["ranker"] == row["correct"] for row in members),
            "exact_correct_presentations": sum(row["exact"] == row["correct"] for row in members),
            "ranker_correct_items": sum(all(row["ranker"] == row["correct"] for row in group)
                                        for group in items.values()),
            "exact_correct_items": sum(all(row["exact"] == row["correct"] for row in group)
                                       for group in items.values()),
            "exact_parsed": sum(row["exact"] is not None for row in members),
        }
    return by_family


def qualify(rows, thresholds):
    targets = [row for row in rows if not row["preserve"]]
    target_items = _by_item(targets)
    stable = sum(group[0]["ranker"] is not None and group[0]["ranker"] == group[1]["ranker"]
                 for group in target_items.values())
    parsed = [row for row in rows if row["exact"] is not None]
    agreed = sum(row["ranker"] == row["exact"] for row in parsed)
    controls = _by_item([row for row in rows if row["preserve"]])
    solved = [item for item, group in controls.items()
              if all(row["ranker"] == row["exact"] == row["correct"] for row in group)]
    errors = [item for item, group in target_items.items()
              if all(row["ranker"] == row["exact"] and row["exact"] not in (None, row["correct"])
                     for row in group)]
    gates = {
        "ranker_paraphrase_stability": stable / len(target_items) >= thresholds["min_ranker_paraphrase_stability"],
        "exact_parse_rate": len(parsed) / len(rows) >= thresholds["min_exact_parse_rate"],
        "ranker_generation_agreement": bool(parsed) and agreed / len(parsed) >= thresholds["min_ranker_generation_agreement"],
        "solved_preservation": len(solved) >= thresholds["min_solved_preservation_items"],
        "persistent_target_errors": len(errors) >= thresholds["min_persistent_target_errors"],
    }
    return {
        "disposition": "PASS" if all(gates.values()) else "HOLD",
        "gates": gates, "target_items_stable": [stable, len(target_items)],
        "exact_parsed": [len(parsed), len(rows)], "ranker_generation_agreed": [agreed, len(parsed)],
        "solved_preservation_items": solved, "persistent_target_errors": errors,
    }


def compare(original, candidate, min_gain, max_regressions):
    if len(original) != len(candidate):
        raise ValueError("Independent evaluation rows differ")
    preservation_changes, ranker_regressions, exact_regressions = set(), set(), set()
    for before, after in zip(original, candidate, strict=True):
        identity = ("id", "item", "wording", "family", "preserve", "correct")
        if any(before[key] != after[key] for key in identity):
            raise ValueError("Independent evaluation row identity changed")
        ranker_change, exact_change = (before[key] != after[key] for key in ("ranker", "exact"))
        if before["preserve"] and (ranker_change or exact_change):
            preservation_changes.add(before["item"])
        elif not before["preserve"]:
            if ranker_change and before["ranker"] == before["correct"]:
                ranker_regressions.add(before["item"])
            if exact_change and before["exact"] == before["correct"]:
                exact_regressions.add(before["item"])
    initial, edited = (summarize(rows)["parity"] for rows in (original, candidate))
    exact_gain = edited["exact_correct_items"] - initial["exact_correct_items"]
    ranker_gain = edited["ranker_correct_items"] - initial["ranker_correct_items"]
    passed = (exact_gain >= min_gain and ranker_gain >= 0
              and len(preservation_changes | ranker_regressions | exact_regressions) <= max_regressions)
    return {"exact_item_gain": exact_gain, "ranker_item_gain": ranker_gain,
            "preservation_changes": sorted(preservation_changes),
            "ranker_regressions": sorted(ranker_regressions), "exact_regressions": sorted(exact_regressions),
            "disposition": "PASS" if passed else "FAIL"}
