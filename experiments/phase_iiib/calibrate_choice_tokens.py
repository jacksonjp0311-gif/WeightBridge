"""Calibrate answer tokens on the frozen common practice screen only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.bic import _BinaryProbe, _margin_rows, _runner
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.evaluation.hf_edit import _load_suite
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.provenance import peak_rss_bytes


CHOICES = (("A", "B"), ("X", "Y"), ("1", "2"), ("C", "D"), ("P", "Q"), ("L", "R"))
RULE = {
    "minimum_accuracy": 0.60,
    "minimum_order_consistency": 0.60,
    "minimum_predicted_class_share": 0.25,
    "rank": ["both_orders_correct", "correct", "minority_prediction_count", "declared_pair_order"],
}


def _render(row, choices):
    mapping = dict(zip(("A", "B"), choices, strict=True))
    question, left, right, instruction = row["text"].rsplit("\n", 3)
    presentation = row["presentation"]
    if (sorted(presentation) != ["A", "B"] or not left.startswith(presentation[0] + ". ")
            or not right.startswith(presentation[1] + ". ") or instruction != "Reply with A or B only."):
        raise ValueError("Frozen practice prompt uses an unexpected answer format")
    first, second = [mapping[label] for label in presentation]
    return {**row, "label": mapping[row["label"]], "presentation": [first, second],
            "text": f"{question}\n{first}. {left[3:]}\n{second}. {right[3:]}\n"
                    f"Reply with {choices[0]} or {choices[1]} only."}


def _metrics(rows, choices):
    predictions = {choice: sum(row["prediction"] == choice for row in rows) for choice in choices}
    labels = {choice: sum(row["label"] == choice for row in rows) for choice in choices}
    correct = sum(row["prediction"] == row["label"] for row in rows)
    items = {}
    for row in rows:
        items.setdefault(row["item_key"], []).append(row)
    both_correct, both_wrong = 0, 0
    for pair in items.values():
        if len(pair) != 2 or sorted(row["answer_position"] for row in pair) != [0, 1]:
            raise ValueError("Every calibration item needs both option orders")
        outcomes = sum(row["prediction"] == row["label"] for row in pair)
        both_correct += int(outcomes == 2)
        both_wrong += int(outcomes == 0)
    return {"correct": correct, "rows": len(rows), "accuracy": correct / len(rows),
            "predictions": predictions, "labels": labels, "minority_prediction_count": min(predictions.values()),
            "item_count": len(items), "both_orders_correct": both_correct, "both_orders_wrong": both_wrong,
            "order_consistency": (both_correct + both_wrong) / len(items),
            "collapsed": not all(predictions.values())}


def calibrate(base, material, output):
    wall, cpu = time.perf_counter(), time.process_time()
    if output.exists():
        raise ValueError("Calibration report path already exists")
    protocol_path = material / "protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    if protocol.get("schema") != "weightbridge.phase_iiib.qwen_protocol.v1" or protocol.get("scope") != "bounded-pilot":
        raise ValueError("Calibration requires the frozen bounded-pilot protocol")
    relative = protocol["cycles"][0]["practice"]["screen"]
    screen = (material / relative).resolve()
    if (not screen.is_relative_to(material / "practice")
            or file_hash(screen) != protocol["artifacts"][relative]):
        raise ValueError("Common practice screen differs from its frozen hash")
    suite = _load_suite(screen.read_bytes())
    if suite["choices"] != ["A", "B"]:
        raise ValueError("Common practice screen must use A/B choices")
    rows = suite["target"] + suite["sentinel"]
    checkpoint = load_checkpoint(base)
    if checkpoint.checkpoint_sha256() != protocol["model_sha256"]:
        raise ValueError("Calibration requires the exact frozen base checkpoint")
    results, eligible = {}, []
    inference = protocol["design"]["inference"]
    with _runner(checkpoint.root, inference) as (runner, torch):
        for order, choices in enumerate(CHOICES):
            name = "".join(choices)
            ids = [runner.tokenizer.encode(label, add_special_tokens=False) for label in choices]
            if any(len(value) != 1 for value in ids) or ids[0] == ids[1]:
                results[name] = {"status": "UNSUPPORTED_TOKEN_PAIR", "token_ids": ids}
                continue
            started, started_cpu = time.perf_counter(), time.process_time()
            rendered = [_render(row, choices) for row in rows]
            probe = _BinaryProbe(runner, torch, rendered, choices, inference)
            margins, _, _ = probe.measure()
            scored = [{**metadata, **measurement} for metadata, measurement in zip(
                ({key: row[key] for key in ("item_key", "answer_position", "family", "presentation")}
                 for row in rendered), _margin_rows(rendered, choices, margins), strict=True)]
            metrics = _metrics(scored, choices)
            checks = {
                "accuracy": metrics["accuracy"] >= RULE["minimum_accuracy"],
                "order_consistency": metrics["order_consistency"] >= RULE["minimum_order_consistency"],
                "prediction_balance": metrics["minority_prediction_count"] >=
                                      math.ceil(len(rows) * RULE["minimum_predicted_class_share"]),
            }
            if order and all(checks.values()):
                eligible.append((metrics["both_orders_correct"], metrics["correct"],
                                 metrics["minority_prediction_count"], -order, name))
            results[name] = {"status": "MEASURED", "token_ids": [value[0] for value in ids],
                             "metrics": metrics, "checks": checks, "paired_predictions": scored,
                             "by_family": {family: _metrics([row for row in scored if row["family"] == family], choices)
                                           for family in sorted({row["family"] for row in scored})},
                             "resources": probe.resources(started, started_cpu)}
    if load_checkpoint(checkpoint.root).checkpoint_sha256() != protocol["model_sha256"]:
        raise ValueError("Calibration source checkpoint changed")
    record = {
        "schema": "weightbridge.phase_iiib.choice_calibration.v1",
        "scope": "Unedited pinned Qwen and frozen practice only. Pair selection requires independent confirmation.",
        "protocol_sha256": file_hash(protocol_path), "practice_screen_sha256": file_hash(screen),
        "base_sha256": protocol["model_sha256"], "source_sha256": file_hash(Path(__file__)),
        "configuration": inference, "rule": RULE, "declared_choices": CHOICES,
        "recommended_alternate": max(eligible)[-1] if eligible else None,
        "conditions": results,
        "resources": {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                      "peak_rss_bytes": peak_rss_bytes(), "model_loads": 1,
                      "forward_batches": sum(row.get("resources", {}).get("forward_batches", 0)
                                             for row in results.values())},
    }
    json_dump_atomic(output, record)
    print(json.dumps({"recommended_alternate": record["recommended_alternate"],
                      "conditions": {name: value.get("metrics", value["status"]) for name, value in results.items()},
                      "resources": record["resources"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.base, args.material.resolve(), args.out)
