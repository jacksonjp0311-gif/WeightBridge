"""Run the larger Qwen-only replication with frozen fitting limits."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig import run_followup as followup, run_trial as trial
from experiments.phase_iiig.roster_replication import PROTOCOL, load_split
from weightbridge.core import file_hash


_V2_SOURCES = followup._sources
_FIT_ROWS = trial._fit_rows


def _sources():
    names = ("run_replication.py", "roster_replication.py", "v4_protocol.json")
    return {**_V2_SOURCES(), **{f"experiments/phase_iiig/{name}": file_hash(Path(__file__).with_name(name))
                               for name in names}}


def _material(material):
    manifest = trial._json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["roster_source_sha256"] != file_hash(Path(__file__).with_name("roster_replication.py"))):
        raise ValueError("Replication roster and protocol commitments differ")
    return manifest


def _fit_rows(runner, measured, family, qualification, settings):
    limits = trial._json(PROTOCOL)["fit_limits"]
    solved = qualification["solved"]
    errors = list(qualification["repeatable_errors"].items())[:limits["max_repeatable_error_items"]]
    capped = {**qualification, "repeatable_errors": dict(errors),
              "solved": {**solved, "target": solved["target"][:limits["max_solved_target_items"]]}}
    fit = _FIT_ROWS(runner, measured, family, capped, settings)
    if len(fit) > limits["max_rows"]:
        raise ValueError("Replication fit scope exceeds the frozen row budget")
    return fit


@contextmanager
def replication_context(model=None):
    """Use the immutable trial driver with v4 material and a bounded fit roster."""
    bindings = {"PROTOCOL": PROTOCOL, "load_split": load_split, "_material": _material,
                "_sources": _sources, "__file__": __file__}
    saved = {name: getattr(followup, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(followup, name, value)
        trial._fit_rows = _fit_rows
        with followup.followup_context(model):
            yield
    finally:
        trial._fit_rows = _FIT_ROWS
        for name, value in saved.items():
            setattr(followup, name, value)


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
    with replication_context("qwen" if args.command == "run" else None):
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
