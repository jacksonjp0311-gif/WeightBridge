"""Replay the frozen trial procedure on the fresh Phase III-G v3 roster."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig import run_followup as followup, run_trial as trial
from experiments.phase_iiig.roster_scaling import PROTOCOL, load_split
from weightbridge.core import file_hash


_V2_SOURCES = followup._sources


def _sources():
    names = ("run_scaling.py", "roster_scaling.py", "v3_protocol.json")
    return {**_V2_SOURCES(), **{f"experiments/phase_iiig/{name}": file_hash(Path(__file__).with_name(name))
                               for name in names}}


def _material(material):
    manifest = trial._json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["roster_source_sha256"] != file_hash(Path(__file__).with_name("roster_scaling.py"))):
        raise ValueError("Scaling roster and protocol commitments differ")
    return manifest


@contextmanager
def scaling_context(model=None):
    """Reuse the archived v2 driver while binding the v3 worker entry point."""
    bindings = {"PROTOCOL": PROTOCOL, "load_split": load_split, "_material": _material,
                "_sources": _sources, "__file__": __file__}
    saved = {name: getattr(followup, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(followup, name, value)
        with followup.followup_context(model):
            yield
    finally:
        for name, value in saved.items():
            setattr(followup, name, value)


if __name__ == "__main__":
    models = tuple(trial._json(PROTOCOL)["checkpoints"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "finish", "worker"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--model", choices=models)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--split", choices=("selection", "hidden"))
    parser.add_argument("--family")
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--peer", type=Path)
    args = parser.parse_args()
    with scaling_context(args.model):
        if args.command == "worker":
            if not all((args.checkpoint_sha256, args.manifest_sha256, args.split, args.family)):
                parser.error("worker requires checkpoint and manifest hashes, split, and family")
            trial._worker(args.base, args.checkpoint_sha256, args.material, args.split, args.family,
                          args.out, args.manifest_sha256, args.seal, args.seal_sha256)
        elif args.command == "run":
            if not args.model:
                parser.error("run requires model")
            result = trial.run(args.model, args.base, args.material, args.out)
            print(json.dumps({key: result[key] for key in ("behavioral_qualification", "single_cycle")}), flush=True)
        else:
            if not args.peer and len(models) != 1:
                parser.error("finish requires the other model's locked run directory")
            result = trial.finish(args.base, args.material, args.out, args.peer or args.out)
            print(json.dumps({key: result[key] for key in ("behavioral_qualification", "single_cycle")}), flush=True)
