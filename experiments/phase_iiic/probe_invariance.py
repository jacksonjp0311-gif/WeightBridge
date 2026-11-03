"""Measure presentation and representation sensitivity on frozen practice only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiic.measurement_roster import FAMILIES, PROTOCOL
from experiments.phase_iiic.run_measurement import _load_split
from weightbridge.adaptation import content_scoring
from weightbridge.adaptation.bic import _runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


RULE = {"selection": "first two frozen practice items from each family, first paraphrase and both orders",
        "label_pairs": [["A", "B"], ["X", "Y"]], "content_score": "mean_log_probability",
        "neutral_prompt": "Reply with one answer string.", "neutral_score": "mean_log_probability",
        "tie_break": "first_printed_option", "used_for_measurement_rule_selection": False}


def _cases(items):
    rows = []
    sampled = []
    for family in FAMILIES:
        sampled.extend([row for row in items if row["family"] == family][:2])
    for item in sampled:
        for presentation in item["presentations"]:
            if presentation["paraphrase"] != 0:
                continue
            question = presentation["content_prompt"].splitlines()[0]
            order = presentation["order"]
            answers = [item["answers"][choice] for choice in order]
            alternatives = [next((alias for alias in item["aliases"][choice] if alias != item["answers"][choice]),
                                 item["answers"][choice]) for choice in order]
            rows.append({"id": presentation["id"], "item_id": item["id"], "family": item["family"], "order": order,
                         "question": question, "answers": answers, "alternate_answers": alternatives})
    return rows


def _prompt(question, answers, labels, *, content):
    first, second = labels
    request = ("Reply with only the exact answer text of the correct option, without its label." if content
               else f"Reply with only the label {first} or {second}.")
    return f"{question}\n{first}. {answers[0]}\n{second}. {answers[1]}\n{request}"


def _score(runner, config, prompts, options, field):
    completions = content_scoring.prepare_continuations(runner.tokenizer, prompts, options,
                                                        max_input_tokens=config["max_input_tokens"])
    scores, resources = content_scoring.score_continuations(runner, completions, batch_size=config["batch_size"])
    pairs = [(left, right) for left, right in zip(scores[::2], scores[1::2], strict=True)]
    return [[option[field] for option in pair] for pair in pairs], resources


def _decide(values):
    return int(values[1] > values[0])


def run(base, material, output):
    wall, cpu = time.perf_counter(), time.process_time()
    manifest = json.loads((material / "manifest.json").read_bytes())
    config = json.loads(PROTOCOL.read_bytes())
    checkpoint = load_checkpoint(base)
    model_hash = checkpoint.checkpoint_sha256()
    if (model_hash != config["model_sha256"] or manifest["model_sha256"] != model_hash
            or manifest["protocol_sha256"] != file_hash(PROTOCOL)):
        raise ValueError("Expected the frozen original model and practice roster")
    if output.exists():
        raise ValueError("Probe output already exists")
    rows = _cases(_load_split(material, manifest, "practice"))
    resources = {}
    with _runner(checkpoint.root, config["inference"]) as (runner, _):
        for labels in RULE["label_pairs"]:
            prompts = [_prompt(row["question"], row["answers"], labels, content=False) for row in rows]
            scores, resources["".join(labels)] = _score(runner, config["inference"], prompts, [labels] * len(rows),
                                                       "sum_log_probability")
            for row, values in zip(rows, scores, strict=True):
                row["".join(labels)] = values
        for key in ("answers", "alternate_answers"):
            prompts = [_prompt(row["question"], row[key], ["A", "B"], content=True) for row in rows]
            scores, resources[key] = _score(runner, config["inference"], prompts, [row[key] for row in rows],
                                           RULE["content_score"])
            for row, values in zip(rows, scores, strict=True):
                row[f"{key}_scores"] = values
        neutral, resources["neutral"] = _score(runner, config["inference"], [RULE["neutral_prompt"]] * len(rows),
                                                [row["answers"] for row in rows], RULE["neutral_score"])
        for row, values in zip(rows, neutral, strict=True):
            row["neutral_scores"] = values
    if load_checkpoint(base).checkpoint_sha256() != model_hash:
        raise ValueError("Source model changed during invariance probe")
    aliases = [row for row in rows if row["answers"] != row["alternate_answers"]]
    report = {
        "schema": "weightbridge.phase_iiic.practice_invariance.v1", "base_sha256": model_hash,
        "protocol_sha256": file_hash(PROTOCOL), "practice_sha256": manifest["artifacts"]["practice"]["sha256"],
        "source_sha256": file_hash(__file__), "scoring_source_sha256": file_hash(content_scoring.__file__),
        "rule": RULE, "rows": rows, "metrics": {"underlying_items": len({row["item_id"] for row in rows}),
            "rows": len(rows), "ab_correct": sum(row["order"][_decide(row["AB"])] == 0 for row in rows),
            "xy_correct": sum(row["order"][_decide(row["XY"])] == 0 for row in rows),
            "label_pair_choice_changes": sum(_decide(row["AB"]) != _decide(row["XY"]) for row in rows),
            "alternate_text_rows": len(aliases), "alternate_text_choice_changes": sum(
                _decide(row["answers_scores"]) != _decide(row["alternate_answers_scores"]) for row in aliases),
            "content_correct": sum(row["order"][_decide(row["answers_scores"])] == 0 for row in rows),
            "neutral_answer_prior_agreement": sum(_decide(row["answers_scores"]) == _decide(row["neutral_scores"])
                                                  for row in rows)},
        "resources": {"scoring": resources, "wall_seconds": time.perf_counter() - wall,
                      "cpu_seconds": time.process_time() - cpu, "peak_rss_bytes": peak_rss_bytes()},
    }
    json_dump_atomic(output, report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.material, args.out)
