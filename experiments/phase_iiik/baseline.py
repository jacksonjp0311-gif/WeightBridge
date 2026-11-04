"""Measure immutable Qwen parent outcomes on Phase III-K's public partitions."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiik.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import measure_endpoint
from weightbridge.core import file_hash, json_dump_atomic


FITTING_SPLITS = ("adaptation", "preservation-fit", "calibration")
PUBLIC_SPLITS = (*FITTING_SPLITS, "preservation-validation", "selection")


def measure_parent(base, material, output, split):
    if split not in (*PUBLIC_SPLITS, "fit"):
        raise ValueError("Parent baseline measures only public partitions")
    if output.exists():
        raise FileExistsError(output)
    start, cpu = time.perf_counter(), time.process_time()
    config, manifest = read_json(PROTOCOL), read_manifest(material)
    manifest_sha, expected = file_hash(material / "manifest.json"), config["checkpoint_sha256"]
    splits = FITTING_SPLITS if split == "fit" else (split,)
    records = {}
    with _open(base, config, expected) as (_, runner, capabilities):
        for name in splits:
            rows = load_split(material, name, manifest)
            measured = measure_endpoint(runner, rows, endpoint(config))
            record = {
                "schema": "weightbridge.phase_iiih.verifier.v1", "checkpoint_sha256": expected,
                "split": name, "roster_sha256": manifest["splits"][name]["sha256"], "manifest_sha256": manifest_sha,
                "measurement": measured, "qualification": qualify(measured, config),
                "capabilities": capabilities.to_dict(), "resources": _resources(start, cpu),
            }
            path = output / f"{name}.json" if split == "fit" else output
            json_dump_atomic(path, record)
            records[name] = record
            start, cpu = time.perf_counter(), time.process_time()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("fit", *PUBLIC_SPLITS))
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    for name, record in measure_parent(args.base, args.material, args.out, args.split).items():
        q = record["qualification"]
        print(f"{name}: {q['disposition']}, {len(q['repeatable_errors'])} repeatable wrong-answer items", flush=True)
