"""Apply the frozen Phase III-G trial procedure to model-specific v2 tasks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase_iiig import run_trial as trial
from experiments.phase_iiig.roster_followup import PROTOCOL, load_split
from weightbridge.core import file_hash


_V1_SOURCES = trial._sources


def _sources():
    return {**_V1_SOURCES(), **{f"experiments/phase_iiig/{name}": file_hash(Path(__file__).with_name(name))
            for name in ("run_followup.py", "roster_followup.py", "v2_protocol.json")}}


def _material(material):
    manifest = trial._json(material / "manifest.json")
    if (manifest["protocol_sha256"] != file_hash(PROTOCOL)
            or manifest["roster_source_sha256"] != file_hash(Path(__file__).with_name("roster_followup.py"))):
        raise ValueError("Follow-up roster and protocol commitments differ")
    return manifest


def _independent(base, expected, material, split, family, out, manifest_sha, timeout, seal=None):
    args = [sys.executable, str(Path(__file__).resolve()), "worker", "--base", str(base),
            "--checkpoint-sha256", expected, "--material", str(material), "--split", split,
            "--family", family, "--out", str(out), "--manifest-sha256", manifest_sha]
    if seal is not None:
        args.extend(["--seal", str(seal), "--seal-sha256", file_hash(seal)])
    subprocess.run(args, check=True, timeout=timeout)
    result = trial._json(out)
    if (result["checkpoint_sha256"] != expected or result["split"] != split or result["family"] != family
            or result["roster_sha256"] != _material(material)["splits"][split]["sha256"]):
        raise ValueError("Independent verification returned different identities")
    return result


@contextmanager
def followup_context(model=None):
    """Bind v2 material and subprocess entry point without changing v1 bytes."""
    config = trial._json(PROTOCOL)

    def scoped(material, split, manifest):
        rows = load_split(material, split, manifest)
        if model is None:
            return rows
        families = config["eligible_families"][model]
        return [row for row in rows if row["family"] in families or row["group"] == "preserve"]

    bindings = {"PROTOCOL": PROTOCOL, "load_split": scoped, "_material": _material,
                "_independent": _independent, "_sources": _sources}
    saved = {name: getattr(trial, name) for name in bindings}
    try:
        for name, value in bindings.items():
            setattr(trial, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(trial, name, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "finish", "worker"))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen", "smollm2"))
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--split", choices=("selection", "hidden"))
    parser.add_argument("--family")
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--seal-sha256")
    parser.add_argument("--peer", type=Path)
    args = parser.parse_args()
    with followup_context(args.model):
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
            if not args.peer:
                parser.error("finish requires the other model's locked run directory")
            result = trial.finish(args.base, args.material, args.out, args.peer)
            print(json.dumps({key: result[key] for key in ("behavioral_qualification", "single_cycle")}), flush=True)
