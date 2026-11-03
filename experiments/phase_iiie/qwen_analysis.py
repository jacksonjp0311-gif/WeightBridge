"""Evaluate opposite questions without conflating paired ranking and generation."""
from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np

from experiments.phase_iiid.analysis import _bootstrap
from weightbridge.adaptation.counterfactual import align_margin, paired_components
from weightbridge.adaptation.grounding import prior_subtracted_scores

P = "sum_log_probability"
DIRECTIONS = ("task", "reversed")
EXPECTED = (0, 1)


def _margin(scores, order):
    return align_margin(*(entry[P] for entry in scores), order)


def _choice(margin):
    return 0 if margin > 0 else 1 if margin < 0 else None


def project_row(row, alpha):
    order, scores = row["order"], row["content"]
    raw = {name: _margin(entries, order) for name, entries in scores.items()}
    prior = {name: _margin(prior_subtracted_scores(entries, scores["neutral"], alpha=alpha), order)
             for name, entries in scores.items() if name in (*DIRECTIONS, "paraphrase", "paraphrase_reversed")}
    primary = (raw["task"], raw["reversed"])
    secondary = (raw["paraphrase"], raw["paraphrase_reversed"])
    evidence, preference = paired_components(*primary)
    other_evidence, other_preference = paired_components(*secondary)
    outcomes = {
        "raw": (primary, secondary),
        "prior": ((prior["task"], prior["reversed"]), (prior["paraphrase"], prior["paraphrase_reversed"])),
        "paired": ((evidence, -evidence), (other_evidence, -other_evidence)),
        "letter": (tuple(_margin(row["letter"][name], order) for name in DIRECTIONS), None),
    }
    lengths = [0, 0]
    for index, answer in zip(order, scores["task"], strict=True):
        lengths[index] = answer["answer_tokens"]
    return {
        "outcomes": {name: [None if pair is None else tuple(_choice(value) for value in pair) for pair in pairs]
                     for name, pairs in outcomes.items()},
        "evidence": evidence,
        "preference": preference,
        "paraphrase_evidence": other_evidence,
        "paraphrase_preference": other_preference,
        "irrelevant_shift": raw["irrelevant"] - raw["neutral"],
        "paraphrase_shift": (abs(secondary[0] - primary[0]) + abs(secondary[1] - primary[1])) / 2,
        "directional_advantage": evidence - abs(raw["irrelevant"] - raw["neutral"]),
        "answer_lengths": lengths,
    }


def _same(left, right):
    return all(a is not None and a == b for a, b in zip(left, right, strict=True))


def _groups(rows):
    groups = defaultdict(dict)
    for index, row in enumerate(rows):
        groups[row["item_id"]][(row["order"][0], row["labels"][0])] = index
    return groups


def _ranker(rows, projections, method, config):
    values = [p["outcomes"][method] for p in projections]
    primary = [pair[0] for pair in values]
    original = [p[0] == 0 for p in primary]
    reversed_ok = [p[1] == 1 for p in primary]
    joint = [p == EXPECTED for p in primary]
    by_item = _groups(rows)
    order_stable = [_same(primary[index[0, label]], primary[index[1, label]])
                    for index in by_item.values() for label in ("A", "X")]
    label_stable = [_same(primary[index[order, "A"]], primary[index[order, "X"]])
                    for index in by_item.values() for order in (0, 1)]
    length_strata = {
        name: {"rows": len(indices), "joint_correct": sum(joint[i] for i in indices),
               "mean_abs_preference": float(np.mean([abs(projections[i]["preference"]) for i in indices]))
               if indices else None}
        for name, indices in (
            ("equal", [i for i, p in enumerate(projections) if p["answer_lengths"][0] == p["answer_lengths"][1]]),
            ("unequal", [i for i, p in enumerate(projections) if p["answer_lengths"][0] != p["answer_lengths"][1]]),
        )
    }
    return {
        "original_accuracy": float(np.mean(original)),
        "reversed_accuracy": float(np.mean(reversed_ok)),
        "joint_accuracy": float(np.mean(joint)),
        "joint_accuracy_ci": _bootstrap(rows, joint, config),
        "order_stability": float(np.mean(order_stable)),
        "label_stability": float(np.mean(label_stable)),
        "paraphrase_stability": None if method == "letter" else float(np.mean([_same(*pair) for pair in values])),
        "abstained_directions": sum(value is None for pair in primary for value in pair),
        "length_strata": length_strata,
    }


def _generation(rows, config):
    rows = [row for row in rows if "generation" in row]
    answers = []
    for row in rows:
        answers.append(tuple(row["order"][position] if position is not None else None for position in
                             (row["generation"][name]["strict_position"] for name in DIRECTIONS)))
    by_item = defaultdict(list)
    for row, pair in zip(rows, answers, strict=True):
        by_item[row["item_id"]].append(pair)
    joint = [pair == EXPECTED for pair in answers]
    metrics = {
        "strict_parse_rate": sum(value is not None for pair in answers for value in pair) / (2 * len(rows)),
        "original_accuracy": float(np.mean([pair[0] == 0 for pair in answers])),
        "reversed_accuracy": float(np.mean([pair[1] == 1 for pair in answers])),
        "joint_accuracy": float(np.mean(joint)),
        "order_stability": float(np.mean([_same(*pairs) for pairs in by_item.values()])),
        "joint_accuracy_ci": _bootstrap(rows, joint, config),
        "presentations": len(rows),
        "eos_responses": sum(response["stop_reason"] == "eos" for row in rows
                             for response in row["generation"].values()),
    }
    metrics["gate"] = {key: metrics[key] >= threshold for key, threshold in config["generation_gate"].items()}
    metrics["pass"] = all(metrics["gate"].values())
    return metrics


def analyze(rows, config):
    projections = [project_row(row, config["scoring"]["prior_alpha"]) for row in rows]
    methods = (*config["scoring"]["methods_in_tie_order"], "letter")
    metrics = {method: _ranker(rows, projections, method, config) for method in methods}
    for method, comparator in product(methods, ("raw", "prior", "letter")):
        if method == comparator:
            continue
        difference = [int(p["outcomes"][method][0] == EXPECTED) -
                      int(p["outcomes"][comparator][0] == EXPECTED) for p in projections]
        metrics[method][f"joint_accuracy_difference_vs_{comparator}"] = {
            "mean": float(np.mean(difference)), "ci": _bootstrap(rows, difference, config)
        }
    advantages = [p["directional_advantage"] for p in projections]
    advantage_ci = _bootstrap(rows, advantages, config)
    thresholds = config["ranker_gate"]
    for method in config["scoring"]["methods_in_tie_order"]:
        m = metrics[method]
        gate = {key: m[key] >= value for key, value in thresholds.items() if key in m}
        gate["raw_standalone_joint_accuracy"] = (
            metrics["raw"]["joint_accuracy"] >= thresholds["raw_standalone_joint_accuracy"])
        gate["question_over_irrelevant"] = (
            advantage_ci[0] > thresholds["question_over_irrelevant_ci_lower_strictly_greater_than"])
        m.update(gate=gate, passed=all(gate.values()))
    diagnostics = {
        "question_over_irrelevant": float(np.mean(advantages)),
        "question_over_irrelevant_ci": advantage_ci,
        "mean_abs_paraphrase_shift": float(np.mean([p["paraphrase_shift"] for p in projections])),
        "mean_abs_irrelevant_shift": float(np.mean([abs(p["irrelevant_shift"]) for p in projections])),
        "mean_paired_evidence": float(np.mean([p["evidence"] for p in projections])),
        "mean_abs_paired_preference": float(np.mean([abs(p["preference"]) for p in projections])),
        "per_family": {
            family: {method: _ranker(subrows, subprojections, method, config)
                     for method in methods}
            for family in sorted({r["family"] for r in rows})
            for subset in [[(r, p) for r, p in zip(rows, projections, strict=True) if r["family"] == family]]
            for subrows, subprojections in [list(zip(*subset, strict=True))]
        },
    }
    return {"rankers": metrics, "diagnostics": diagnostics, "generation": _generation(rows, config)}, projections


def lock(analysis, config):
    def key(method):
        metrics = analysis["rankers"][method]
        return (sum(metrics["gate"].values()), metrics["joint_accuracy"],
                min(metrics["original_accuracy"], metrics["reversed_accuracy"]), metrics["order_stability"])

    return max(config["scoring"]["methods_in_tie_order"], key=key)


def classify_headroom(rows, projections, method):
    by_item = defaultdict(list)
    for row, projection in zip(rows, projections, strict=True):
        by_item[(row["family"], row["item_id"])].append(projection["outcomes"])
    results = defaultdict(dict)
    for (family, item_id), outcomes in by_item.items():
        pairs = [outcome[name][0] for outcome in outcomes for name in (method, "raw")]
        state = ("solved" if all(pair == EXPECTED for pair in pairs)
                 else "persistently_wrong" if all(pair[0] == 1 for pair in pairs)
                 else "invalid" if any(None in pair for pair in pairs)
                 else "unstable")
        results[family][item_id] = state
    return dict(results)
