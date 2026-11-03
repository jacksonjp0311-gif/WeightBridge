"""Audit archived prompts, budgets, patch receipts, and held-out outcomes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_blind_qwen_repair import BATCH, correct, flips
from run_qwen_wording_ablation import (
    ARMS, LAYERS, MODES, SOURCE, STRENGTHS, build_splits, digest, load_protocol, practice_metrics, read, report,
)
from verify_blind_qwen_evidence import require
from verify_qwen_repair_cohort import check_patch, check_rows


def verify(root):
    manifest = load_protocol(root / "manifest.json")
    derived = report(root, manifest)
    require(derived == read(root / "summary.json"), "archived summary differs from evidence")
    schedule = [(layer, strength) for layer in LAYERS for strength in STRENGTHS]
    for row in derived["cases"]:
        case = root / f"case-{row['slot']:02d}"
        if not case.exists():
            continue
        public, oracle, search, evaluation = (case / name for name in ("public", "evaluator", "search", "evaluation"))
        if not (oracle / "calibration.json").is_file():
            require(row["status"] in ("PARTIAL", "HOLD_EXECUTION"), "missing construction evidence")
            continue
        sets = build_splits(row["slot"])
        for name in sets:
            folder = public if name in MODES else oracle
            require(read(folder / f"{name}.json") == sets[name], "archived prompts differ from the frozen split")
        info = read(oracle / "case.json")
        require(info["spec"] == manifest["cases"][row["slot"] - 1], "sealed case differs from roster")
        require(digest(oracle / "holdout.json") == info["holdout_sha256"], "holdout hash differs")
        calibration = read(oracle / "calibration.json")
        baseline = calibration["baseline"]
        check_rows(baseline, sets["construction"])
        for attempt in calibration["attempts"]:
            rows = attempt["rows"]
            check_rows(rows, sets["construction"])
            require(correct(baseline["target"]) - correct(rows["target"]) == attempt["loss"]
                    and flips(baseline["sentinel"], rows["sentinel"]) == attempt["unrelated_flips"],
                    "calibration metrics differ from raw construction rows")
        if not (public / "input.json").is_file():
            continue
        inputs = read(public / "input.json")
        fault = check_patch(oracle, "fault", manifest, info["pristine_sha256"], row["layer"])
        require(inputs["damaged_sha256"] == fault["candidate_sha256"]
                and inputs["source_sha256"] == digest(SOURCE)
                and inputs["practice_sha256"] == {mode: digest(public / f"{mode}.json") for mode in MODES},
                "damaged input receipt differs")
        total_batches, selected = 0, {}
        for mode, arms in ARMS.items():
            log_path = search / f"{mode}.json"
            if not log_path.is_file():
                require(row["status"] in ("PARTIAL", "HOLD_EXECUTION"), "missing practice evidence")
                continue
            log = read(log_path)
            check_rows(log["baseline"], sets[mode])
            require(log["input_sha256"] == digest(public / "input.json"), "search input hash differs")
            batches = math.ceil(sum(map(len, sets[mode].values())) / BATCH)
            require(log["baseline_forward_batches"] == batches
                    and log["forward_batches"] == batches * (1 + len(arms) * len(schedule)),
                    "practice forward totals differ from the frozen budget")
            total_batches += log["forward_batches"]
            selected.update(log["selected"])
            for arm in arms:
                attempts = log["attempts"][arm]
                require([(r["layer"], r["strength"]) for r in attempts] == schedule, "practice schedule differs")
                for attempt in attempts:
                    check_rows(attempt["rows"], sets[mode])
                    require(attempt["forward_batches"] == batches, "candidate forward budget differs")
                    metrics = practice_metrics(log["baseline"], attempt["rows"], attempt["norm"])
                    require(all(attempt[k] == value for k, value in metrics.items()), "practice metrics differ")
                if selected[arm] is not None:
                    finalist = attempts[selected[arm]]
                    check_patch(search, arm, manifest, fault["candidate_sha256"], finalist["layer"], finalist)
            for guided, random in zip(*(log["attempts"][arm] for arm in arms)):
                require(guided["columns"] == random["columns"]
                        and abs(guided["norm"] - random["norm"]) <= 0.05 * guided["norm"]
                        and math.isclose(guided["requested_norm"], random["requested_norm"], rel_tol=1e-12),
                        "guided and random proposals differ in support or norm")
        if (search / "search_resources.json").is_file():
            require(read(search / "search_resources.json")["forward_batches"] == total_batches,
                    "total search forwards differ from practice logs")
        if (evaluation / "scores.json").is_file():
            scores = read(evaluation / "scores.json")
            require(set(scores) == {"pristine", "damaged", *selected}, "held-out arms differ from finalists")
            for name, groups in scores.items():
                check_rows(groups, sets["holdout"])
                if name in selected and selected[name] is None:
                    require(groups == scores["damaged"], "abstaining arm did not use the damaged baseline")
    return derived["dispositions"], derived["counts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    statuses, counts = verify(parser.parse_args().evidence)
    print(json.dumps({"dispositions": statuses, "counts": counts}, sort_keys=True))
