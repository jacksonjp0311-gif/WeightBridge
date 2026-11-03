"""Audit the frozen Qwen AI edit pilot from its exported JSON evidence."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

import run_qwen_ai_edit_pilot as pilot
from run_blind_qwen_repair import BATCH, correct, digest, flips, read
from run_qwen_repair_cohort import ARMS, FAULT_STRENGTHS, outcome, practice_metrics, select_attempt
from verify_blind_qwen_evidence import require
from verify_qwen_repair_cohort import check_patch, check_rows


def check_budget(resources, split, passes):
    examples = sum(map(len, split.values()))
    require(resources["prompt_examples"] == examples * passes
            and resources["forward_batches"] == math.ceil(examples / BATCH) * passes,
            "inference budget differs from the frozen split")


def check_calibration(log, construction):
    baseline, attempts = log["baseline"], log["attempts"]
    check_rows(baseline, construction)
    strengths = [row["strength"] for row in attempts]
    require(strengths == list(FAULT_STRENGTHS[:len(attempts)]), "calibration schedule differs")
    initial = correct(baseline["target"])
    accepted = []
    for i, attempt in enumerate(attempts):
        rows = attempt["rows"]
        check_rows(rows, construction)
        loss = initial - correct(rows["target"])
        changed = flips(baseline["sentinel"], rows["sentinel"])
        require((loss, changed) == (attempt["loss"], attempt["unrelated_flips"]),
                "calibration metrics differ from raw rows")
        if loss >= 2 and changed <= 1:
            accepted.append(i)
    require((initial >= 6 and bool(attempts)) or (initial < 6 and not attempts),
            "calibration did not honor the pristine floor")
    exhausted = initial < 6 or len(attempts) == len(FAULT_STRENGTHS)
    require(accepted == [len(attempts) - 1] if accepted else exhausted,
            "calibration did not stop at the first accepted fault or exhaust the schedule")
    check_budget(log["resources"], construction, 1 + len(attempts))
    return bool(accepted)


def check_search(log, practice, manifest, slot):
    baseline, attempts, history = log["baseline"], log["attempts"], log["controller"]
    budget = manifest["candidate_evaluations_per_arm"]
    require(set(attempts) == set(ARMS) and len(history) == budget
            and all(len(rows) == budget for rows in attempts.values()), "candidate budgets differ")
    check_rows(baseline, practice)
    check_budget(log["resources"], practice, 1 + len(ARMS) * budget)
    options = [tuple(option) for option in manifest["options"]]
    rng = np.random.default_rng(manifest["seed"] + 100 + slot)
    remaining = [options[i] for i in rng.permutation(len(options))]
    feedback = []
    for i, decision in enumerate(history):
        letters = "ABCDEFGHI"[:len(remaining)]
        menu = "\n".join(f"{letter}. layer {layer}, strength {strength}"
                         for letter, (layer, strength) in zip(letters, remaining))
        previous = "\n".join(feedback) or "No edits have been tried."
        prompt = (
            "Choose the next bounded rank-one repair for a damaged language model. Improve target accuracy "
            "while keeping unrelated predictions stable. Each layer and strength pair is untested. The host fits "
            "the direction using practice activations, and returns practice results. "
            f"Damaged practice target accuracy is {correct(baseline['target'])}/{len(baseline['target'])}.\n"
            f"Prior guided results:\n{previous}\n\nAvailable edits:\n{menu}\n\nReply with the single option letter."
        )
        require(decision["prompt"] == prompt, "controller prompt differs from practice feedback and eligible options")
        logits = decision["logits"]
        require(set(logits) == set(letters) and all(math.isfinite(value) for value in logits.values()),
                "invalid controller option logits")
        choice = max(letters, key=logits.__getitem__)
        require(decision["choice"] == choice and type(decision["input_tokens"]) is int
                and decision["input_tokens"] > 0, "controller did not select the highest-logit eligible option")
        layer, strength = remaining.pop(letters.index(choice))
        require((decision["layer"], decision["strength"]) == (layer, strength), "controller option differs from decision")
        guided, random = (attempts[arm][i] for arm in ARMS)
        for attempt in (guided, random):
            require((attempt["layer"], attempt["strength"]) == (layer, strength), "candidate differs from controller option")
            check_rows(attempt["rows"], practice)
            require(math.isfinite(attempt["norm"]) and attempt["norm"] > 0
                    and math.isfinite(attempt["requested_norm"]) and attempt["requested_norm"] > 0, "invalid edit norm")
            metrics = practice_metrics(baseline, attempt["rows"], attempt["norm"])
            require(all(attempt[key] == value for key, value in metrics.items()), "practice metrics differ from raw rows")
        require(guided["indices"] == random["indices"]
                and abs(guided["norm"] - random["norm"]) <= 0.05 * guided["norm"]
                and math.isclose(guided["requested_norm"], random["requested_norm"], rel_tol=1e-12),
                "control does not match the guided candidate's support and norm")
        feedback.append(f"Layer {layer}, strength {strength}: target gain {guided['target_gain']}, "
                        f"sentinel flips {guided['unrelated_flips']}, sentinel margin drift {guided['drift']:.3f}.")
    selected = {arm: select_attempt(attempts[arm]) for arm in ARMS}
    require(log["selected"] == selected, "finalists differ from the frozen practice rule")
    return selected


def verify_case(case, spec, manifest):
    slot, layer = spec["slot"], spec["layer"]
    public, evaluator, search, evaluation = (case / name for name in ("public", "evaluator", "search", "evaluation"))
    sets = pilot.splits(slot)
    frozen = read(evaluator / "input.json")
    for name, folder in (("construction", evaluator), ("holdout", evaluator), ("practice", public)):
        path = folder / f"{name}.json"
        require(read(path) == sets[name], "archived examples differ from the frozen split")
        if name != "practice":
            require(digest(path) == frozen[f"{name}_sha256"], "evaluator input hash differs")

    accepted = check_calibration(read(evaluator / "calibration.json"), sets["construction"])
    read(evaluator / "resources.json")
    hold = evaluator / "hold.json"
    if hold.is_file():
        require(not accepted and not (public / "input.json").exists()
                and not search.exists() and not evaluation.exists(), "invalid calibration HOLD")
        return {**spec, "disposition": "HOLD_CALIBRATION", "reason": read(hold)["reason"]}

    inputs = read(public / "input.json")
    require(accepted, "fault receipt exists without accepted calibration")
    fault = check_patch(evaluator, "fault", manifest, frozen["pristine_sha256"], layer)
    require(inputs["damaged_sha256"] == fault["candidate_sha256"]
            and inputs["practice_sha256"] == digest(public / "practice.json"), "damaged input receipt differs")

    log = read(search / "search.json")
    read(search / "resources.json")
    require(log["practice_sha256"] == inputs["practice_sha256"], "search used a different practice split")
    selected = check_search(log, sets["practice"], manifest, slot)
    candidates = {}
    for arm, index in selected.items():
        if index is not None:
            finalist = log["attempts"][arm][index]
            receipt = check_patch(search, arm, manifest, fault["candidate_sha256"], finalist["layer"],
                                  {**finalist, "columns": finalist["indices"]})
            candidates[arm] = receipt["candidate_sha256"]

    scores = read(evaluation / "scores.json")
    summary = read(evaluation / "summary.json")
    read(evaluation / "resources.json")
    require(set(scores) == {"pristine", "damaged", *ARMS}, "held-out arms differ from the protocol")
    require(summary["scores_sha256"] == digest(evaluation / "scores.json"), "held-out score hash differs")
    require(set(summary["resources"]) == {"pristine", "damaged", *candidates},
            "evaluation resources differ from the scored checkpoint arms")
    for arm, groups in scores.items():
        check_rows(groups, sets["holdout"])
        if arm in ARMS and arm not in candidates:
            require(groups == scores["damaged"], "abstaining arm differs from damaged baseline")
        else:
            examples = sum(map(len, sets["holdout"].values()))
            batches = math.ceil(examples / BATCH)
            if arm in candidates:
                practice_examples = sum(map(len, sets["practice"].values()))
                examples += practice_examples
                batches += math.ceil(practice_examples / BATCH)
            resources = summary["resources"][arm]
            require(resources["prompt_examples"] == examples and resources["forward_batches"] == batches,
                    "evaluation budget differs from holdout and practice replay")
    derived = outcome(scores, sets["holdout"], selected, manifest["gate"])
    require(derived == {key: value for key, value in summary.items() if key not in {"resources", "scores_sha256"}},
            "recorded outcome differs from raw holdout scores")
    return {**spec, **derived, "candidate_sha256": candidates}


def verify(root):
    manifest = read(root / "manifest.json")
    require(manifest == pilot.protocol(), "manifest differs from frozen source or protocol")
    cases = []
    for spec in manifest["cases"]:
        case = root / f"case-{spec['slot']:02d}"
        try:
            cases.append(verify_case(case, spec, manifest))
        except FileNotFoundError as error:
            cases.append({**spec, "disposition": "INCOMPLETE",
                          "missing_evidence": Path(error.filename).resolve().relative_to(root.resolve()).as_posix()})
        except ValueError as error:
            raise ValueError(f"{case.name}: {error}") from error
    counts = dict(Counter(row["disposition"] for row in cases))
    return {"complete": "INCOMPLETE" not in counts, "counts": counts, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    print(json.dumps(verify(parser.parse_args().evidence), indent=2, sort_keys=True))
