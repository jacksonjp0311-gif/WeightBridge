"""Run the frozen instruction/scoring factorial and headroom audit on the original model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic.measurement_roster import FAMILIES, PROTOCOL, SCHEMA
from weightbridge.adaptation import content_scoring
from weightbridge.adaptation.bic import _runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


INSTRUCTIONS = ("letter", "content")
CONTENT_RULES = {"sum": "sum_log_probability", "mean": "mean_log_probability",
                 "eos_sum": "terminated_sum_log_probability"}


def _normalized(value):
    return " ".join(value.casefold().split())


def parse_response(text, presented_answers, labels, aliases, instruction):
    """Parse a complete answer by exact normalized match, returning the printed position."""
    if instruction not in INSTRUCTIONS or len(labels) != 2 or len(presented_answers) != 2 or len(aliases) != 2:
        raise ValueError("Expected two presented answers and a frozen response instruction")
    if instruction == "letter":
        options = [[label] for label in labels]
    else:
        options = [[answer, *forms] for answer, forms in zip(presented_answers, aliases, strict=True)]
    matches = [index for index, option in enumerate(options) if _normalized(text) in map(_normalized, option)]
    return matches[0] if len(matches) == 1 else None


def _load_split(root, manifest, split):
    path = root / f"{split}.json"
    expected = manifest["artifacts"][split]
    if file_hash(path) != expected["sha256"]:
        raise ValueError(f"{split} roster hash does not match the freeze")
    doc = json.loads(path.read_bytes())
    if (doc["schema"] != SCHEMA or doc["split"] != split or len(doc["items"]) != expected["underlying_items"]
            or len({item["semantic_key"] for item in doc["items"]}) != len(doc["items"])):
        raise ValueError(f"{split} roster structure does not match the freeze")
    return doc["items"]


def _rows(items):
    rows = []
    for item in items:
        for presentation in item["presentations"]:
            order = presentation["order"]
            rows.append({**presentation, "item_id": item["id"], "family": item["family"],
                         "answers": [item["answers"][choice] for choice in order],
                         "aliases": [item["aliases"][choice] for choice in order],
                         "correct_position": order.index(0)})
    return rows


def _paired_scores(scores, positions, fields):
    result = []
    for index, correct in enumerate(positions):
        options = scores[index * 2:index * 2 + 2]
        scores_by_rule = {}
        for rule, field in fields.items():
            values = [option[field] for option in options]
            scores_by_rule[rule] = {"position": int(values[1] > values[0]),
                                    "correct_margin": values[correct] - values[1 - correct], "scores": values}
        result.append((scores_by_rule, options))
    return result


def _evaluate_split(runner, items, config):
    rows = _rows(items)
    positions = [row["correct_position"] for row in rows]
    inference, generation = config["inference"], config["generation"]
    resources = {}
    for row in rows:
        row["conditions"] = {}
    for instruction in INSTRUCTIONS:
        prompts = [row[f"{instruction}_prompt"] for row in rows]
        options = content_scoring.prepare_continuations(runner.tokenizer, prompts, [row["labels"] for row in rows],
                                                        max_input_tokens=inference["max_input_tokens"])
        if any(len(option.answer_ids) != 1 for option in options):
            raise ValueError("Frozen letter choices require one token per rendered continuation")
        label_scores, label_resources = content_scoring.score_continuations(runner, options,
                                                                              batch_size=inference["batch_size"])
        options = content_scoring.prepare_continuations(runner.tokenizer, prompts, [row["answers"] for row in rows],
                                                        max_input_tokens=inference["max_input_tokens"],
                                                        include_eos=config["scoring"]["include_eos"])
        content_scores, content_resources = content_scoring.score_continuations(runner, options,
                                                                                  batch_size=inference["batch_size"])
        generated, generate_resources = content_scoring.generate_responses(
            runner, prompts, max_input_tokens=inference["max_input_tokens"],
            max_new_tokens=generation["max_new_tokens"], batch_size=generation["batch_size"],
        )
        label_pairs = _paired_scores(label_scores, positions, {"label": "sum_log_probability"})
        content_pairs = _paired_scores(content_scores, positions, CONTENT_RULES)
        for row, (label, label_tokens), (content, answer_tokens), response in zip(
                rows, label_pairs, content_pairs, generated, strict=True):
            parsed = parse_response(response["text"], row["answers"], row["labels"], row["aliases"], instruction)
            completed = response["stop_reason"] == "eos"
            row["conditions"][instruction] = {
                "likelihood": {**label, **content}, "label_continuations": label_tokens,
                "content_continuations": answer_tokens,
                "generation": {**response, "parsed_position": parsed,
                               "eligible_position": parsed if completed or not generation["require_eos_for_correctness"] else None},
            }
        resources[instruction] = {"letter_scores": label_resources, "content_scores": content_resources,
                                  "generation": generate_resources}
    return rows, resources


def _predict(row, instruction, rule):
    condition = row["conditions"][instruction]
    if rule == "generation":
        return condition["generation"]["eligible_position"]
    return condition["likelihood"][rule]["position"]


def _groups(rows, *, across_paraphrases):
    groups = defaultdict(list)
    for row in rows:
        key = row["item_id"] if across_paraphrases else (row["item_id"], row["paraphrase"])
        groups[key].append(row)
    return list(groups.values())


def _bootstrap(rows, prediction, comparison, config):
    settings = config["bootstrap"]
    rng = np.random.default_rng(settings["seed"])
    effects = []
    for family in FAMILIES:
        groups = _groups([row for row in rows if row["family"] == family], across_paraphrases=True)
        differences = np.array([np.mean([int(prediction(row) == row["correct_position"]) -
                                         int(comparison(row) == row["correct_position"]) for row in group])
                                for group in groups])
        effects.append(rng.choice(differences, size=(settings["replicates"], len(groups)), replace=True))
    samples = np.concatenate(effects, axis=1).mean(axis=1)
    tail = (1 - settings["interval"]) / 2
    return [float(value) for value in np.quantile(samples, [tail, 1 - tail])]


def _metrics(rows, instruction, rule):
    predict = lambda row: _predict(row, instruction, rule)
    correct = lambda row: predict(row) == row["correct_position"]
    across_orders = _groups(rows, across_paraphrases=False)
    by_item = _groups(rows, across_paraphrases=True)
    positions = Counter(predict(row) for row in rows)
    paired_stable = 0
    for group in across_orders:
        contents = {row["order"][predict(row)] if predict(row) is not None else None for row in group}
        paired_stable += len(contents) == 1 and None not in contents
    metrics = {
        "correct": sum(map(correct, rows)), "rows": len(rows), "accuracy": sum(map(correct, rows)) / len(rows),
        "both_order_correct": sum(all(map(correct, group)) for group in across_orders),
        "order_pairs": len(across_orders), "stable_order_pairs": paired_stable,
        "all_four_correct": sum(all(map(correct, group)) for group in by_item), "underlying_items": len(by_item),
        "first_position_predictions": positions[0], "second_position_predictions": positions[1],
        "unparsed_or_incomplete_predictions": positions[None],
        "printed_label_counts": dict(Counter(row["labels"][predict(row)] for row in rows if predict(row) is not None)),
    }
    if rule == "generation":
        responses = [row["conditions"][instruction]["generation"] for row in rows]
        metrics.update(completed=sum(row["stop_reason"] == "eos" for row in responses),
                       parsed=sum(row["parsed_position"] is not None for row in responses),
                       completed_and_parsed=sum(row["eligible_position"] is not None for row in responses))
    else:
        eligible = [row for row in rows if _predict(row, instruction, "generation") is not None]
        agreements = sum(predict(row) == _predict(row, instruction, "generation") for row in eligible)
        margins = [row["conditions"][instruction]["likelihood"][rule]["correct_margin"] for row in rows]
        token_lengths = [tuple(option["answer_tokens"] for option in row["conditions"][instruction]["content_continuations"])
                         for row in rows]
        unequal = [(row, sizes) for row, sizes in zip(rows, token_lengths, strict=True) if sizes[0] != sizes[1]]
        metrics.update(generation_agreements=agreements, eligible_generations=len(eligible),
                       unconditional_generation_agreement=agreements / len(rows),
                       mean_correct_margin=float(np.mean(margins)), tied_margins=sum(margin == 0 for margin in margins),
                       unequal_length_rows=len(unequal),
                       longer_content_preferred=sum(sizes[predict(row)] > sizes[1 - predict(row)] for row, sizes in unequal),
                       longer_content_correct=sum(sizes[row["correct_position"]] > sizes[1 - row["correct_position"]]
                                                  for row, sizes in unequal))
    return metrics


def _summary(rows, config):
    output = {}
    comparison = lambda row: _predict(row, "letter", "label")
    for instruction in INSTRUCTIONS:
        for rule in ("label", *CONTENT_RULES, "generation"):
            metrics = _metrics(rows, instruction, rule)
            predict = lambda row: _predict(row, instruction, rule)
            metrics["paired_accuracy_gain_ci_vs_letter_label"] = _bootstrap(rows, predict, comparison, config)
            output[f"{instruction}_{rule}"] = {**metrics, "by_family": {
                family: _metrics([row for row in rows if row["family"] == family], instruction, rule)
                for family in FAMILIES}}
    return output


def _gate(summary, rule, thresholds):
    score, generation = summary[f"content_{rule}"], summary["content_generation"]
    checks = {
        "content_accuracy": score["accuracy"] >= thresholds["min_content_accuracy"],
        "all_four_correct": score["all_four_correct"] / score["underlying_items"] >= thresholds["min_all_four_correct"],
        "completed_parse_rate": generation["completed_and_parsed"] / generation["rows"] >= thresholds["min_completed_parse_rate"],
        "generated_accuracy": generation["accuracy"] >= thresholds["min_generated_accuracy"],
        "generation_agreement": bool(score["eligible_generations"]) and score["generation_agreements"] /
        score["eligible_generations"] >= thresholds["min_agreement_among_completed_parsed"],
        "paired_accuracy_gain_ci_lower": score["paired_accuracy_gain_ci_vs_letter_label"][0] >=
        thresholds["min_paired_accuracy_gain_ci_lower"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _select_rule(summary, config):
    gates = {rule: _gate(summary, rule, config["measurement_gate"])
             for rule in config["scoring"]["eligible_rules_in_tie_order"]}

    def rank(rule):
        score = summary[f"content_{rule}"]
        return (gates[rule]["passed"], sum(gates[rule]["checks"].values()), score["all_four_correct"],
                score["accuracy"], score["unconditional_generation_agreement"])

    return max(gates, key=rank), gates


def _headroom(splits, rule, qualified, thresholds):
    profiles = {}
    for split, rows in splits.items():
        persistent, solved = defaultdict(list), defaultdict(list)
        for group in _groups(rows, across_paraphrases=True):
            if len(group) != 4:
                raise ValueError("Headroom requires four presentations per underlying item")
            initial = group[0]
            matches = [(_predict(row, "content", rule) == row["correct_position"],
                        _predict(row, "content", "generation")) for row in group]
            if all(not likelihood and generated is not None and generated != row["correct_position"]
                   for row, (likelihood, generated) in zip(group, matches, strict=True)):
                persistent[initial["family"]].append(initial["item_id"])
            if all(likelihood and generated == row["correct_position"]
                   for row, (likelihood, generated) in zip(group, matches, strict=True)):
                solved[initial["family"]].append(initial["item_id"])
        profiles[split] = (persistent, solved)
    by_family, eligible = {}, []
    for family in FAMILIES:
        by_family[family] = {split: {"persistent_error_ids": persistent[family], "solved_ids": solved[family],
                                     "cross_family_control_ids": [item for other in FAMILIES if other != family
                                                                  for item in solved[other]]}
                             for split, (persistent, solved) in profiles.items()}
        if all(len(profile["persistent_error_ids"]) >= thresholds["minimum_persistent_errors_per_split"]
               and len(profile["cross_family_control_ids"]) >= thresholds["minimum_cross_family_controls_per_split"]
               for profile in by_family[family].values()):
            eligible.append(family)
    return {"passed": qualified and bool(eligible), "measurement_qualified": qualified,
            "families_meeting_counts": eligible, "eligible_families": eligible if qualified else [],
            "by_family": by_family}


def run(base, material, output):
    wall, cpu = time.perf_counter(), time.process_time()
    manifest_path = material / "manifest.json"
    manifest, config = json.loads(manifest_path.read_bytes()), json.loads(PROTOCOL.read_bytes())
    checkpoint = load_checkpoint(base)
    base_hash = checkpoint.checkpoint_sha256()
    if (base_hash != config["model_sha256"] or manifest["model_sha256"] != base_hash
            or manifest["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("Measurement requires the frozen protocol and original model")
    output.mkdir(parents=True)
    records, split_rows = {}, {}
    with _runner(checkpoint.root, config["inference"]) as (runner, _):
        for split in ("practice", "selection", "qualification"):
            split_wall, split_cpu = time.perf_counter(), time.process_time()
            rows, resources = _evaluate_split(runner, _load_split(material, manifest, split), config)
            summary = _summary(rows, config)
            split_rows[split] = rows
            records[split] = {"summary": summary, "resources": {**resources, "wall_seconds": time.perf_counter() - split_wall,
                                                              "cpu_seconds": time.process_time() - split_cpu}}
            json_dump_atomic(output / f"{split}-predictions.json", {"rows": rows, **records[split]})
            if split == "practice":
                rule, gates = _select_rule(summary, config)
                decision = {"selected_rule": rule, "practice_gate": gates[rule], "practice_candidates": gates,
                            "practice_sha256": manifest["artifacts"]["practice"]["sha256"],
                            "practice_predictions_sha256": file_hash(output / "practice-predictions.json")}
                json_dump_atomic(output / "practice-decision.json", decision)
            print(f"Measured {split} with {len(rows)} presentations", file=sys.stderr, flush=True)
    if load_checkpoint(base).checkpoint_sha256() != base_hash:
        raise ValueError("Measurement source checkpoint changed")
    independent = _gate(records["qualification"]["summary"], rule, config["measurement_gate"])
    qualified = decision["practice_gate"]["passed"] and independent["passed"]
    headroom = _headroom({split: split_rows[split] for split in ("practice", "selection")}, rule, qualified,
                         config["headroom_gate"])
    report = {"schema": "weightbridge.phase_iiic.measurement_report.v1", "base_sha256": base_hash,
              "protocol_sha256": file_hash(PROTOCOL), "material_manifest_sha256": file_hash(manifest_path),
              "scoring_source_sha256": file_hash(content_scoring.__file__), "runner_source_sha256": file_hash(__file__),
              "practice_decision_sha256": file_hash(output / "practice-decision.json"), **decision,
              "qualification_gate": {"passed": qualified, "independent_gate": independent},
              "headroom_gate": headroom, "splits": records,
              "resources": {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                            "peak_rss_bytes": peak_rss_bytes(), "model_loads": 1}}
    json_dump_atomic(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("selected_rule", "practice_gate", "qualification_gate", "headroom_gate",
                                                   "resources")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.material, args.out)
