"""Run the Qwen-only decimal comparison qualification and gated trial."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig import run_followup as followup, run_trial as trial
from experiments.phase_iiig.decimal_endpoint import canonical_decimal, measure_decimal
from experiments.phase_iiig.endpoint import answer_pair
from experiments.phase_iiig.roster_decimal import PROTOCOL, load_split
from weightbridge.core import file_hash


_V2_SOURCES = followup._sources


def _sources():
    names = ("run_decimal.py", "roster_decimal.py", "v5_protocol.json", "decimal_endpoint.py")
    return {**_V2_SOURCES(), **{f"experiments/phase_iiig/{name}": file_hash(Path(__file__).with_name(name))
                               for name in names}}


def _material(material):
    manifest = trial._json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["roster_source_sha256"] != file_hash(Path(__file__).with_name("roster_decimal.py"))):
        raise ValueError("Decimal roster and protocol commitments differ")
    return manifest


def _fit_rows(runner, measured, family, qualification, settings):
    limits = trial._json(PROTOCOL)["fit_limits"]
    errors = qualification["repeatable_errors"]
    target_items = set(list(errors)[:limits["max_repeatable_error_items"]])
    solved = qualification["solved"]
    preserved = set(solved["target"][:limits["max_solved_target_items"]])
    preserved.update(solved["positive"], solved["preserve"])
    fit = []
    for row in trial._scope(measured, family):
        item = row["item"]
        if item not in target_items and item not in preserved:
            continue
        answer = row["answer"]
        rival = errors[item] if item in target_items else canonical_decimal(format(Decimal(answer) + 1, "f"))
        if rival is None:
            raise ValueError("Preservation rival exceeds the frozen decimal endpoint")
        fit.append({"id": row["id"], "prompt": row["prompt"], "preserve": item in preserved,
                    "answers": answer_pair(runner, row["prompt"], answer, rival,
                                           max_input_tokens=settings["max_input_tokens"])})
    if len(fit) > limits["max_rows"]:
        raise ValueError("Decimal fit scope exceeds the frozen row budget")
    return fit


@contextmanager
def decimal_context(model=None):
    """Bind the trial driver to v5 without changing any earlier evidence source."""
    config = trial._json(PROTOCOL)
    if set(config["checkpoints"]) != {"qwen"} or set(config["eligible_families"]) != {"qwen"}:
        raise ValueError("The decimal trial requires one declared Qwen participant")
    followup_bindings = {"PROTOCOL": PROTOCOL, "load_split": load_split, "_material": _material,
                         "_sources": _sources, "__file__": __file__}
    trial_bindings = {"_fit_rows": _fit_rows, "measure": measure_decimal}
    saved_followup = {name: getattr(followup, name) for name in followup_bindings}
    saved_trial = {name: getattr(trial, name) for name in trial_bindings}
    try:
        for name, value in followup_bindings.items():
            setattr(followup, name, value)
        for name, value in trial_bindings.items():
            setattr(trial, name, value)
        with followup.followup_context(model):
            yield
    finally:
        for name, value in saved_followup.items():
            setattr(followup, name, value)
        for name, value in saved_trial.items():
            setattr(trial, name, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "finish", "worker"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--split", choices=("selection", "hidden"))
    parser.add_argument("--family")
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--seal-sha256")
    args = parser.parse_args()
    with decimal_context("qwen" if args.command == "run" else None):
        if args.command == "worker":
            if not all((args.checkpoint_sha256, args.manifest_sha256, args.split, args.family)):
                parser.error("worker requires checkpoint and manifest hashes, split, and family")
            trial._worker(args.base, args.checkpoint_sha256, args.material, args.split, args.family,
                          args.out, args.manifest_sha256, args.seal, args.seal_sha256)
        else:
            if args.command == "run":
                result = trial.run("qwen", args.base, args.material, args.out)
            else:
                result = trial.finish(args.base, args.material, args.out, args.out)
            print(json.dumps({key: result[key] for key in ("behavioral_qualification", "single_cycle")}), flush=True)
