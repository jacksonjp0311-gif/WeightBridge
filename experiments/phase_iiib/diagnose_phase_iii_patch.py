"""Retrospective diagnostic of the disclosed Phase III accepted Qwen patch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from weightbridge.adaptation.bic import score_binary_margins
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint


ARCHIVE = ROOT / "results/phase-iii-natural-qwen-v1/five-cycle"
TENSOR = "model.language_model.layers.23.mlp.down_proj.weight"


def variant(row, choices, swapped):
    prompt, left, right, ending = row["text"].rsplit("\n", 3)
    if not left.startswith("A. ") or not right.startswith("B. ") or "A or B" not in ending:
        raise ValueError(f"Unexpected archived prompt format for {row['id']}")
    answers = [left[3:], right[3:]]
    position = (row["label"] == "B") ^ swapped
    if swapped:
        answers.reverse()
    a, b = choices
    return {"id": row["id"], "label": choices[position],
            "text": f"{prompt}\n{a}. {answers[0]}\n{b}. {answers[1]}\n{ending.replace('A or B', f'{a} or {b}')}"}


def grouped_effect(before, after):
    groups = {}
    for old, new in zip(before, after, strict=True):
        if old["id"] != new["id"] or old["label"] != new["label"]:
            raise ValueError("Retrospective predictions do not pair")
        groups.setdefault(old["id"].split("/")[-2], []).append((old, new))
    results = {}
    for family, rows in groups.items():
        raw = np.array([new["margin"] - old["margin"] for old, new in rows])
        signed = np.array([new["signed_margin"] - old["signed_margin"] for old, new in rows])
        results[family] = {
            "examples": len(rows), "before_correct": sum(old["prediction"] == old["label"] for old, _ in rows),
            "after_correct": sum(new["prediction"] == new["label"] for _, new in rows),
            "mean_raw_margin_change": float(raw.mean()), "mean_signed_margin_change": float(signed.mean()),
            "positive_raw_shifts": int((raw > 0).sum()), "negative_raw_shifts": int((raw < 0).sum()),
        }
    return results


def run(base, candidate, out):
    report_path = ARCHIVE / "run/state/evaluation/final.json"
    suite_path = ARCHIVE / "material/sealed/final.json"
    report, suite = [json.loads(path.read_bytes()) for path in (report_path, suite_path)]
    base, candidate = [load_checkpoint(path) for path in (base, candidate)]
    if (base.checkpoint_sha256() != report["root_sha256"]
            or candidate.checkpoint_sha256() != report["head_sha256"]
            or file_hash(suite_path) != report["suite_sha256"]):
        raise ValueError("Retrospective comparison requires the exact historical checkpoints and suite")
    original = suite["target"]
    conditions, scores = {}, {}
    for choices in (("A", "B"), ("X", "Y")):
        for swapped in (False, True):
            name = "".join(choices) + ("-swapped" if swapped else "-original")
            rows = [variant(row, choices, swapped) for row in original]
            before, after = [score_binary_margins(checkpoint.root, rows, report["configuration"], choices=choices)
                             for checkpoint in (base, candidate)]
            if name == "AB-original":
                actual = [(a["prediction"], b["prediction"]) for a, b in zip(before["rows"], after["rows"], strict=True)]
                expected = [(row["base_prediction"], row["head_prediction"]) for row in report["target"]]
                if actual != expected:
                    raise ValueError("Archived final target predictions differ from the retrospective reload")
            scores[name] = (before["rows"], after["rows"])
            conditions[name] = {"base": before, "candidate": after, "by_family": grouped_effect(*scores[name])}
    decomposition = {}
    for label in ("AB", "XY"):
        ordinary = scores[label + "-original"]
        swapped = scores[label + "-swapped"]
        paired = {}
        for a, b, c, d in zip(*ordinary, *swapped, strict=True):
            raw = (b["margin"] - a["margin"] + d["margin"] - c["margin"]) / 2
            signed = (b["signed_margin"] - a["signed_margin"] + d["signed_margin"] - c["signed_margin"]) / 2
            paired.setdefault(a["id"].split("/")[-2], []).append((raw, signed))
        decomposition[label] = {family: {"mean_position_component": float(np.mean(rows, axis=0)[0]),
                                         "mean_signed_component": float(np.mean(rows, axis=0)[1]),
                                         "rms_position_component": float(np.sqrt(np.mean(np.square(rows), axis=0)[0])),
                                         "rms_signed_component": float(np.sqrt(np.mean(np.square(rows), axis=0)[1]))}
                                for family, rows in paired.items()}
    # One A and one B from each task family. These gradients are measured on the parent.
    sampled = {}
    for row in original:
        sampled.setdefault((row["id"].split("/")[-2], row["label"]), row)
    jacobian = score_binary_margins(base.root, list(sampled.values()), report["configuration"], jacobian_tensor=TENSOR)
    record = {
        "schema": "weightbridge.phase_iiib.retrospective.v1", "purpose": "Disclosed history only. No selection or qualification.",
        "suite_sha256": file_hash(suite_path), "prior_report_sha256": file_hash(report_path),
        "base_sha256": report["root_sha256"], "candidate_sha256": report["head_sha256"],
        "configuration": report["configuration"], "tensor": TENSOR, "conditions": conditions,
        "counterbalanced_decomposition": decomposition, "parent_jacobian_sample": jacobian,
    }
    json_dump_atomic(out, record)
    print(json.dumps({"by_format": {key: value["by_family"] for key, value in conditions.items()},
                      "counterbalanced_decomposition": decomposition}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base", "candidate", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.candidate, args.out)
