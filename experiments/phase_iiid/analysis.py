"""Practice locking and question-level measurements for Phase III-D."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from weightbridge.adaptation.grounding import prior_subtracted_scores


DISPLAYS = ("labeled", "bullets")
NEUTRALS = ("neutral_absent", "neutral_generic")
FIELDS = {"sum": "sum_log_probability", "mean": "mean_log_probability",
          "eos_sum": "terminated_sum_log_probability"}


def _groups(rows, pair=False):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = (row["item_id"], row["paraphrase"]) if pair else row["item_id"]
        groups[key].append(index)
    return list(groups.values())


def _positions(row, display, context, rule):
    scores = row["displays"][display]["contexts"]
    if rule["kind"] == "prior_sum":
        return [entry[FIELDS["sum"]] for entry in
                prior_subtracted_scores(scores[context], scores[rule["neutral"]], alpha=rule["alpha"])]
    return [entry[FIELDS[rule["kind"]]] for entry in scores[context]]


def _margin(row, display, context, rule):
    left, right = _positions(row, display, context, rule)
    return left - right


def _letter(row):
    left, right = [value[FIELDS["sum"]] for value in row["displays"]["labeled"]["letter"]]
    return int(right > left)


def _bootstrap(rows, effects, config):
    seed = config["bootstrap"]["seed"]
    replicates = config["bootstrap"]["replicates"]
    rng = np.random.default_rng(seed)
    by_family = defaultdict(list)
    for group in _groups(rows):
        by_family[rows[group[0]]["family"]].append(float(np.mean([effects[index] for index in group])))
    samples = [rng.choice(values, size=(replicates, len(values)), replace=True) for values in by_family.values()]
    return np.quantile(np.concatenate(samples, axis=1).mean(axis=1), [0.025, 0.975]).tolist()


def ranker_predictions(rows, rule):
    display = rule["display"]
    neutral = rule.get("neutral", "neutral_generic")
    predictions = []
    for row in rows:
        margins = {name: _margin(row, display, name, rule)
                   for name in ("task", "counterfactual", "irrelevant", neutral)}
        predictions.append({"task": int(margins["task"] < 0),
                            "counterfactual": int(margins["counterfactual"] < 0),
                            "irrelevant": int(margins["irrelevant"] < 0), "neutral": int(margins[neutral] < 0),
                            "signed_task_margin": (1 if row["correct_position"] == 0 else -1) * margins["task"],
                            "directional_margin_advantage": (1 if row["correct_position"] == 0 else -1)
                            * (margins["task"] - margins["counterfactual"])
                            - abs(margins["irrelevant"] - margins[neutral])})
    return predictions


def ranker_metrics(rows, rule, config):
    predicted = ranker_predictions(rows, rule)
    task = [item["task"] == row["correct_position"] for row, item in zip(rows, predicted, strict=True)]
    changed = [item["counterfactual"] == row["counterfactual_position"]
               for row, item in zip(rows, predicted, strict=True)]
    stability = [len({rows[i]["order"][predicted[i]["task"]] for i in group}) == 1
                 for group in _groups(rows, pair=True)]
    letter = [_letter(row) == row["correct_position"] for row in rows]
    delta = [int(a) - int(b) for a, b in zip(task, letter, strict=True)]
    return {"rows": len(rows), "underlying_items": len(_groups(rows)), "task_accuracy": float(np.mean(task)),
            "counterfactual_accuracy": float(np.mean(changed)),
            "both_correct_fraction": float(np.mean(np.logical_and(task, changed))),
            "all_four_task_correct": float(np.mean([all(task[i] for i in group) for group in _groups(rows)])),
            "all_four_both_correct": float(np.mean([all(task[i] and changed[i] for i in group)
                                                     for group in _groups(rows)])),
            "order_stability": float(np.mean(stability)),
            "counterfactual_choice_changes": float(np.mean([p["task"] != p["counterfactual"] for p in predicted])),
            "irrelevant_choice_changes": float(np.mean([p["irrelevant"] != p["neutral"] for p in predicted])),
            "mean_directional_margin_advantage": float(np.mean([p["directional_margin_advantage"] for p in predicted])),
            "mean_signed_task_margin": float(np.mean([p["signed_task_margin"] for p in predicted])),
            "letter_accuracy": float(np.mean(letter)), "accuracy_gain_ci_vs_letter": _bootstrap(rows, delta, config)}


def _ranker_gate(metrics, config):
    thresholds = config["ranker_gate"]
    checks = {field: metrics[field] >= thresholds[f"min_{field}"] for field in
              ("task_accuracy", "counterfactual_accuracy", "both_correct_fraction", "all_four_task_correct",
               "order_stability")}
    checks["directional_margin_advantage"] = (metrics["mean_directional_margin_advantage"]
                                                > thresholds["min_mean_directional_margin_advantage"])
    checks["paired_accuracy_gain_ci_lower"] = (metrics["accuracy_gain_ci_vs_letter"][0]
                                                >= thresholds["min_paired_accuracy_gain_ci_lower"])
    return {"checks": checks, "passed": all(checks.values())}


def _generation_metrics(rows, display, rule, parser="strict"):
    reference = ranker_predictions(rows, rule)
    responses = [row["displays"][display]["generation"] for row in rows]
    task = [r["task"][f"{parser}_position"] for r in responses]
    changed = [r["counterfactual"][f"{parser}_position"] for r in responses]
    good = [position == row["correct_position"] for row, position in zip(rows, task, strict=True)]
    agreement = [p == r["task"] for p, r in zip(task, reference, strict=True) if p is not None]
    return {"rows": len(rows), "completed": sum(r["task"]["stop_reason"] == "eos" for r in responses),
            "completed_parse_rate": sum(p is not None for p in task) / len(rows),
            "generated_accuracy": float(np.mean(good)),
            "counterfactual_accuracy": float(np.mean([p == r["counterfactual_position"]
                                                        for p, r in zip(changed, rows, strict=True)])),
            "all_four_correct": float(np.mean([all(good[i] for i in group) for group in _groups(rows)])),
            "order_stability": float(np.mean([
                all(task[i] is not None for i in group) and len({rows[i]["order"][task[i]] for i in group}) == 1
                for group in _groups(rows, pair=True)])),
            "agreement_among_completed_parsed": float(np.mean(agreement)) if agreement else None,
            "semantic_kind_counts": dict(Counter(r["task"]["semantic_kind"] for r in responses)),
            "parser_disagreements": sum(r["task"]["strict_position"] != r["task"]["semantic_position"] for r in responses)}


def _generation_gate(metrics, config):
    checks = {field: metrics[field] is not None and metrics[field] >= threshold
              for key, threshold in config["generation_gate"].items() for field in (key.removeprefix("min_"),)}
    return {"checks": checks, "passed": all(checks.values())}


def _rules(config):
    rules = []
    for display in config["scoring"]["displays_in_tie_order"]:
        rules.extend({"display": display, "kind": kind} for kind in FIELDS)
        rules.extend({"display": display, "kind": "prior_sum", "alpha": alpha, "neutral": neutral}
                     for neutral in config["scoring"]["neutral_contexts_in_tie_order"]
                     for alpha in config["scoring"]["alphas_in_tie_order"])
    return rules


def select_practice(rows, config):
    """Select and freeze using the practice rows alone."""
    candidates = []
    for rule in _rules(config):
        metrics = ranker_metrics(rows, rule, config)
        candidates.append({"rule": rule, "metrics": metrics, "gate": _ranker_gate(metrics, config)})

    def rank(candidate):
        metrics, gate = candidate["metrics"], candidate["gate"]
        return (gate["passed"], sum(gate["checks"].values()), metrics["both_correct_fraction"],
                metrics["all_four_task_correct"], metrics["task_accuracy"], metrics["order_stability"])

    selected = max(candidates, key=rank)
    calibrated = max((candidate for candidate in candidates if candidate["rule"]["kind"] == "prior_sum"), key=rank)
    display_metrics = []
    for display in DISPLAYS:
        rule = {**selected["rule"], "display": display}
        metrics = _generation_metrics(rows, display, rule)
        display_metrics.append({"display": display, "metrics": metrics, "gate": _generation_gate(metrics, config)})
    generated = max(display_metrics, key=lambda item: (item["gate"]["passed"], sum(item["gate"]["checks"].values()),
                                                       item["metrics"]["generated_accuracy"]))
    return {"selected_ranker": selected, "selected_generation": generated,
            "factorial_prior": {key: calibrated["rule"][key] for key in ("alpha", "neutral")},
            "candidate_comparisons": candidates, "generation_comparisons": display_metrics}


def analyze_split(rows, config, decision):
    rule = decision["selected_ranker"]["rule"]
    selected = ranker_metrics(rows, rule, config)
    generation = {}
    for display in DISPLAYS:
        generation[display] = {parser: _generation_metrics(rows, display, {**rule, "display": display}, parser)
                               for parser in ("strict", "semantic")}
    factorial, effects, lengths = {}, {}, {}
    for display in DISPLAYS:
        factorial[display] = {kind: ranker_metrics(rows, {"display": display, "kind": kind,
                                                          **decision["factorial_prior"]}, config)
                              for kind in (*FIELDS, "prior_sum")}
        raw = ranker_predictions(rows, {"display": display, "kind": "sum"})
        calibrated = ranker_predictions(rows, {"display": display, "kind": "prior_sum", **decision["factorial_prior"]})
        effects[display] = [int(b["task"] == r["correct_position"]) - int(a["task"] == r["correct_position"])
                            for r, a, b in zip(rows, raw, calibrated, strict=True)]
        lengths[display] = {}
        for kind in (*FIELDS, "prior_sum"):
            predictions = ranker_predictions(rows, {"display": display, "kind": kind, **decision["factorial_prior"]})
            strata = {"equal": [], "unequal": []}
            for row, prediction in zip(rows, predictions, strict=True):
                left, right = row["displays"][display]["contexts"]["task"]
                group = "equal" if left["answer_tokens"] == right["answer_tokens"] else "unequal"
                strata[group].append(prediction["task"] == row["correct_position"])
            lengths[display][kind] = {group: {"rows": len(values), "correct": sum(values)}
                                      for group, values in strata.items()}
    interaction = [b - a for a, b in zip(effects["labeled"], effects["bullets"], strict=True)]
    selected_display = decision["selected_generation"]["display"]
    return {"selected_ranker": selected, "ranker_gate": _ranker_gate(selected, config),
            "selected_ranker_by_family": {family: ranker_metrics([r for r in rows if r["family"] == family], rule, config)
                                          for family in sorted({r["family"] for r in rows})},
            "factorial": factorial, "prior_minus_raw": {display: {"accuracy_difference": float(np.mean(delta)),
                                                                   "question_interval": _bootstrap(rows, delta, config)}
                                                         for display, delta in effects.items()},
            "factorial_interaction": {"accuracy_difference": float(np.mean(interaction)),
                                      "question_interval": _bootstrap(rows, interaction, config)},
            "answer_lengths": lengths, "generation_by_display": generation,
            "generation_gate": _generation_gate(generation[selected_display]["strict"], config)}
