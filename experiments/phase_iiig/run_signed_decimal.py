"""Run the preregistered Qwen negative-ordering trial through the frozen driver."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig import run_decimal as decimal, run_trial as trial
from experiments.phase_iiig.roster_decimal_targeted import PROTOCOL, load_split
from weightbridge.core import file_hash


_V5_SOURCES = decimal._sources
_QUALIFY = trial.qualify


def _sources():
    names = ("run_signed_decimal.py", "roster_decimal_targeted.py", "v6_protocol.json")
    return {**_V5_SOURCES(), **{f"experiments/phase_iiig/{name}": file_hash(Path(__file__).with_name(name))
                               for name in names}}


def _material(material):
    manifest = trial._json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["roster_source_sha256"] != file_hash(Path(__file__).with_name("roster_decimal_targeted.py"))):
        raise ValueError("Targeted decimal roster and protocol commitments differ")
    return manifest


def qualify_negative(rows, family, thresholds):
    """Require a repeated choice of the smaller operand on a negative pair."""
    result = _QUALIFY(rows, family, thresholds)
    items = trial._items(trial._scope(rows, family))
    excluded = {}
    for item, answer in result["repeatable_errors"].copy().items():
        operands = [Decimal(value) for value in items[item][0]["operands"]]
        if len(operands) != 2 or any(value >= 0 for value in operands) or Decimal(answer) != min(operands):
            excluded[item] = result["repeatable_errors"].pop(item)
    result["excluded_repeatable_outputs"] = excluded
    result["gates"]["repeatable_errors"] = len(result["repeatable_errors"]) >= thresholds["min_repeatable_errors"]
    result["disposition"] = "PASS" if all(result["gates"].values()) else "HOLD"
    return result


@contextmanager
def signed_context(model=None):
    """Bind the semantic gate and v6 task files to the immutable v5 driver."""
    bindings = {"PROTOCOL": PROTOCOL, "load_split": load_split, "_material": _material,
                "_sources": _sources, "__file__": __file__}
    saved = {name: getattr(decimal, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(decimal, name, value)
        trial.qualify = qualify_negative
        with decimal.decimal_context(model):
            yield
    finally:
        trial.qualify = _QUALIFY
        for name, value in saved.items():
            setattr(decimal, name, value)


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
    with signed_context("qwen" if args.command == "run" else None):
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
