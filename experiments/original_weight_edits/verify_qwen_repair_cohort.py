"""Replay the frozen cohort report and audit its archived raw evidence."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from run_blind_qwen_repair import BATCH, correct, flips, tensor_name
from run_qwen_repair_cohort import (
    ARMS, LAYERS, STRENGTHS, build_splits, digest, load_protocol, practice_metrics, read, report,
)
from verify_blind_qwen_evidence import check_receipt, require


def check_rows(groups, expected):
    require(groups.keys() == expected.keys(), "score groups differ from the frozen split")
    for key, rows in groups.items():
        require([(r["id"], r["label"]) for r in rows] == [(r["id"], r["label"]) for r in expected[key]],
                "score row identities or labels differ from the frozen split")
        for row in rows:
            margin = row["margin"]
            require(math.isfinite(margin), "nonfinite answer margin")
            prediction = row["label"]
            if margin < 0:
                prediction = "B" if row["label"] == "A" else "A"
            elif margin == 0:
                prediction = "A"
            require(row["prediction"] == prediction and row["correct"] == (prediction == row["label"]),
                    "prediction or correctness differs from the signed margin")


def check_patch(folder, name, config, base, layer, selected=None):
    with gzip.open(folder / f"{name}_patch.json.gz", "rt", encoding="utf-8") as stream:
        patch = json.load(stream)
    receipt = read(folder / f"{name}_receipt.json")
    check_receipt(patch, receipt, config)
    require(receipt["base_sha256"] == base, "patch receipt uses a different base checkpoint")
    change, = patch["slices"]
    require(change["tensor"] == tensor_name(layer) and change["axis"] == "columns" and change["op"] == "add",
            "patch differs from the selected layer or intervention")
    if selected is not None:
        require(change["indices"] == selected["columns"] and
                math.isclose(float(np.linalg.norm(change["values"])), selected["requested_norm"], rel_tol=1e-12),
                "finalist patch differs from practice support or norm")
    return receipt


def verify(root):
    manifest = load_protocol(root / "manifest.json")
    derived = report(root, manifest)
    require(derived == read(root / "summary.json"), "archived summary differs from evidence")
    schedule = [(layer, strength) for layer in LAYERS for strength in STRENGTHS]
    for row in derived["cases"]:
        case = root / f"case-{row['slot']:02d}"
        if not case.exists():
            continue
        sets = build_splits(row["slot"])
        public, oracle, search, evaluation = (case / name for name in ("public", "evaluator", "search", "evaluation"))
        if not (oracle / "calibration.json").is_file():
            require(row["status"] in ("PARTIAL", "HOLD_EXECUTION"), "case has no construction evidence")
            continue
        for name, path in (("practice", public), ("construction", oracle), ("holdout", oracle)):
            require(read(path / f"{name}.json") == sets[name], "archived prompts differ from the frozen split")
        info = read(oracle / "case.json")
        require(info["spec"] == manifest["cases"][row["slot"] - 1], "sealed case differs from roster")
        require(digest(oracle / "holdout.json") == info["holdout_sha256"], "holdout hash differs")
        calibration = read(oracle / "calibration.json")
        check_rows(calibration["baseline"], sets["construction"])
        for attempt in calibration["attempts"]:
            check_rows(attempt["rows"], sets["construction"])
            baseline = calibration["baseline"]
            require(correct(baseline["target"]) - correct(attempt["rows"]["target"]) == attempt["loss"]
                    and flips(baseline["sentinel"], attempt["rows"]["sentinel"]) == attempt["unrelated_flips"],
                    "calibration result differs from raw construction rows")
        if not (public / "input.json").is_file():
            continue
        inputs = read(public / "input.json")
        fault = check_patch(oracle, "fault", manifest, info["pristine_sha256"], row["layer"])
        require(inputs["damaged_sha256"] == fault["candidate_sha256"]
                and inputs["practice_sha256"] == digest(public / "practice.json"), "damaged input receipt differs")
        if not (search / "search.json").is_file():
            continue
        log = read(search / "search.json")
        check_rows(log["baseline"], sets["practice"])
        require(log["input_sha256"] == digest(public / "input.json"), "search input hash differs")
        batches = math.ceil(sum(map(len, sets["practice"].values())) / BATCH)
        require(log["shared_baseline_forward_batches"] == batches
                and log["resources"]["forward_batches"] == batches * (1 + len(ARMS) * len(schedule)),
                "practice forward totals differ from the frozen budget")
        for arm in ARMS:
            attempts = log["attempts"][arm]
            require([(r["layer"], r["strength"]) for r in attempts] == schedule, "practice schedule differs")
            for attempt in attempts:
                check_rows(attempt["rows"], sets["practice"])
                require(attempt["forward_batches"] == batches, "candidate practice forward budget differs")
                metrics = practice_metrics(log["baseline"], attempt["rows"], attempt["norm"])
                require(all(attempt[k] == value for k, value in metrics.items()), "practice metrics differ from raw rows")
            if log["selected"][arm] is not None:
                selected = attempts[log["selected"][arm]]
                check_patch(search, arm, manifest, fault["candidate_sha256"], selected["layer"], selected)
        for guided, random in zip(*(log["attempts"][arm] for arm in ARMS)):
            require(guided["columns"] == random["columns"] and abs(guided["norm"] - random["norm"]) <= 0.05 * guided["norm"]
                    and math.isclose(guided["requested_norm"], random["requested_norm"], rel_tol=1e-12),
                    "paired proposals are not support and norm matched")
        if (evaluation / "scores.json").is_file():
            scores = read(evaluation / "scores.json")
            for groups in scores.values():
                check_rows(groups, sets["holdout"])
    return derived["disposition"], derived["counts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    status, counts = verify(parser.parse_args().evidence)
    print(json.dumps({"disposition": status, "counts": counts}, sort_keys=True))
