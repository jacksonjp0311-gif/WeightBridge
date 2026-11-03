"""Read-only audit of the frozen Phase III-G negative-decimal evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from experiments.phase_iiig.decimal_endpoint import canonical_decimal
from weightbridge.core import file_hash


def _read(path):
    return json.loads(path.read_bytes())


def _kind(fit, question):
    if fit["prompt"] != question["prompt"]:
        raise ValueError("Fitting prompt differs from the frozen question")
    correct, rival = [canonical_decimal(value) for value in fit["answers"]]
    if correct != question["answer"] or rival is None or correct == rival:
        raise ValueError("Malformed v6 fit contrast")
    plus_one = Decimal(correct) + 1 == Decimal(rival)
    if question["family"] == "copy":
        return "copy_plus_one_auxiliary" if plus_one else "copy_other_auxiliary"
    options = {canonical_decimal(value) for value in question["operands"]}
    if len(options) != 2 or correct not in options or Decimal(correct) != max(map(Decimal, options)):
        raise ValueError("Invalid frozen comparison")
    prefix = "preservation" if fit["preserve"] else "target"
    if rival in options:
        return f"{prefix}_actual_option"
    return f"{prefix}_plus_one_outside_options" if plus_one else f"{prefix}_other_outside_options"


def _first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def audit(archive, root):
    artifact_manifest = _read(archive / "artifact_manifest.json")
    report = _read(archive / "qwen/report.json")
    for base, hashes in ((archive, artifact_manifest["artifacts"]), (root, report["source_sha256"])):
        for rel, expected in hashes.items():
            if file_hash(base / rel) != expected:
                raise ValueError(f"Frozen evidence changed: {rel}")
    questions = {r["id"]: r for r in _read(archive / "material/practice.json")["rows"]}
    rows = _read(archive / "qwen/fit-rows.json")["rows"]
    contrasts = [{"id": row["id"], "kind": _kind(row, questions[row["id"]]),
                  "answers": row["answers"], "operands": questions[row["id"]].get("operands")}
                 for row in rows]

    def verification(label):
        return _read(archive / "qwen/verification" / f"selection-{label}.json")["measurement"]["rows"]

    parent = verification("base")
    arms = {}
    for arm in ("base", "constrained", "gradient"):
        stats, regressions, divergences = Counter(), Counter(), Counter()
        target = {}
        for before, after in zip(parent, verification(arm), strict=True):
            if before["id"] != after["id"] or before["prompt"] != after["prompt"]:
                raise ValueError("Frozen selection rows differ")
            correct = after["exact"] == after["answer"]
            stats["total"] += 1
            stats["historical_parsed"] += after["exact"] is not None
            stats["correct"] += correct
            stats["completed_line"] += "\n" in after["text"] or after["stop_reason"] == "eos"
            if after["family"] == "larger-decimal" and after["exact"] is not None:
                stats["parsed_outside_options"] += after["exact"] not in {
                    canonical_decimal(value) for value in after["operands"]}
            if after["group"] == "target":
                target.setdefault(after["item"], []).append(correct)
            if not correct and before["exact"] == before["answer"]:
                regressions[after["group"]] += 1
            index = _first_difference(before["generated_token_ids"], after["generated_token_ids"])
            divergences["identical" if index is None else str(index)] += 1
        arms[arm] = {**stats, "target_items_solved": sum(all(x) for x in target.values()),
                     "target_items": len(target), "regressions": dict(regressions),
                     "first_different_generated_token_zero_based": dict(divergences)}
    return {"schema": "weightbridge.phase_iiih.v6_audit.v1",
            "v6_artifact_manifest_sha256": file_hash(archive / "artifact_manifest.json"),
            "v6_report_sha256": file_hash(archive / "qwen/report.json"),
            "verified_artifact_count": len(artifact_manifest["artifacts"]),
            "verified_source_count": len(report["source_sha256"]),
            "fit_contrast_counts": dict(Counter(row["kind"] for row in contrasts)),
            "fit_contrasts": contrasts, "selection": arms, "historical_disposition": report["single_cycle"],
            "hidden_opened": report["hidden_opened"], "original_gates_reinterpreted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, default=Path("results/phase-iii-g/negative-decimal-v6"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
