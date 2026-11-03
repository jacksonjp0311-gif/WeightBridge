"""Measure one untouched parent split under the frozen Phase III-I endpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint, qualify, read_json
from experiments.phase_iiii.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import measure_endpoint
from weightbridge.core import file_hash, json_dump_atomic


def measure_parent(base, material, output, split):
    if split not in ("practice", "selection"):
        raise ValueError("Parent baseline measures only a public split")
    if output.exists():
        raise FileExistsError(output)
    started, cpu = time.perf_counter(), time.process_time()
    config = read_json(PROTOCOL)
    manifest = read_manifest(material)
    rows = load_split(material, split, manifest)
    expected = config["checkpoint_sha256"]
    with _open(base, config, expected) as (_, runner, capabilities):
        measured = measure_endpoint(runner, rows, endpoint(config))
    record = {
        "schema": "weightbridge.phase_iiih.verifier.v1",
        "checkpoint_sha256": expected,
        "split": split,
        "roster_sha256": manifest["splits"][split]["sha256"],
        "manifest_sha256": file_hash(material / "manifest.json"),
        "measurement": measured,
        "qualification": qualify(measured, config),
        "capabilities": capabilities.to_dict(),
        "resources": _resources(started, cpu),
    }
    json_dump_atomic(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("practice", "selection"))
    for name in ("base", "material", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = measure_parent(args.base, args.material, args.out, args.split)
    q = result["qualification"]
    print(f"{args.split}: {q['disposition']}, {len(q['repeatable_errors'])} repeatable wrong-answer items", flush=True)
