"""Measure group-aware trajectory coverage on the pinned SmolLM2 parent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint
from experiments.phase_iiii.cross_model import ORIGINAL_SMOLLM2
from experiments.phase_iiij.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import compare_endpoint, measure_endpoint
from weightbridge.adaptation.counterexample_update import measure_parent_decisions
from weightbridge.adaptation.preservation_geometry import sample_anchors
from weightbridge.core import file_hash, json_dump_atomic


def run(base, material, out):
    if out.exists():
        raise FileExistsError(out)
    wall, cpu = time.perf_counter(), time.process_time()
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    contract = endpoint(config)
    groups = config["probe_design"]["preservation_groups"]

    def group(row):
        if row["family"] == "copy":
            return "copy"
        if row["tier"] == "mixed-sign":
            return "mixed-sign"
        return row["group"]

    with _open(base, config, ORIGINAL_SMOLLM2) as (adapter, runner, capabilities):
        measurement = measure_endpoint(runner, load_split(material, "practice", manifest), contract)
        pool, decision_receipt = measure_parent_decisions(
            runner, measurement, contract, group_for=group, batch_size=contract.batch_size,
            max_memory_bytes=config["intent"]["limits"]["max_memory_bytes"],
        )
        sample = sample_anchors(pool.anchors, budget=len(groups) * len(contract.required_wordings),
                                required_groups=groups, required_wordings=contract.required_wordings)
        if adapter.checkpoint.checkpoint_sha256() != ORIGINAL_SMOLLM2:
            raise ValueError("SmolLM2 checkpoint identity changed")
    report = {"schema": "weightbridge.phase_iiij.portability.v1", "checkpoint_sha256": ORIGINAL_SMOLLM2,
              "protocol_sha256": file_hash(PROTOCOL), "practice_sha256": manifest["splits"]["practice"]["sha256"],
              "architecture": capabilities.architecture, "group_coverage": sample.record(),
              "parent_metrics": compare_endpoint(measurement, measurement, contract)["parent"],
              "generation_resources": measurement["resources"], "trajectory_resources": decision_receipt["resources"],
              "eligible_decisions": decision_receipt["eligible_decisions"], "resources": _resources(wall, cpu)}
    json_dump_atomic(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.base, args.material, args.out)
    print(json.dumps({key: record[key] for key in ("architecture", "parent_metrics", "resources")}))
