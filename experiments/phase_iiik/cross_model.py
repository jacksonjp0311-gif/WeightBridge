"""Exercise native BF16 sensitivity and exact restoration on SmolLM2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.phase_iiif.run_matrix import _open
from experiments.phase_iiih.run_decimal_endpoint import _resources, endpoint
from experiments.phase_iiii.cross_model import ORIGINAL_SMOLLM2
from experiments.phase_iiii.run_causal_fork import _plan
from experiments.phase_iiij.run_preservation import _preservation_group
from experiments.phase_iiik.decimal_roster import PROTOCOL, load_split, read_manifest
from weightbridge.adaptation.behavior_endpoint import compare_endpoint, measure_endpoint
from weightbridge.adaptation.counterexample_update import ForkBasis, measure_parent_decisions
from weightbridge.adaptation.jspace_update import measure_directional_secant
from weightbridge.adaptation.robust_jspace import select_preservation
from weightbridge.core import file_hash, json_dump_atomic


def run(base, material, out):
    if out.exists():
        raise FileExistsError(out)
    wall, cpu = time.perf_counter(), time.process_time()
    config, manifest = json.loads(PROTOCOL.read_bytes()), read_manifest(material)
    config["checkpoint_sha256"] = ORIGINAL_SMOLLM2
    contract = endpoint(config)
    design = config["probe_design"]
    with _open(base, config, ORIGINAL_SMOLLM2, gradient_probe=True) as (adapter, runner, capabilities):
        rows = load_split(material, "preservation-fit", manifest)
        baseline = measure_endpoint(runner, rows, contract)
        pool, pool_record = measure_parent_decisions(
            runner, baseline, contract, group_for=_preservation_group, batch_size=contract.batch_size,
            max_memory_bytes=config["intent"]["limits"]["max_memory_bytes"],
        )
        region = max((region for region in capabilities.editable_regions if region.role == "ffn_output"),
                     key=lambda region: region.layer_index)
        plan = _plan(config, capabilities, region.layer_index)
        height, width = adapter.checkpoint.tensors[region.tensor_name].shape
        columns = tuple(range(min(4, width, plan.max_columns)))
        left = (-1.0) ** np.arange(height) / np.sqrt(height)
        right = np.ones(len(columns)) / np.sqrt(len(columns))
        basis = ForkBasis(ORIGINAL_SMOLLM2, region.tensor_name, columns, left, right, (), (),
                          np.empty(0), np.empty((0, height)))
        slopes, sensitivity = measure_directional_secant(adapter, runner, plan, basis, pool,
            radius=design["directional_secant_radius"], batch_size=contract.batch_size)
        selection = select_preservation(pool.anchors, budget=design["anchors"],
            required_groups=design["preservation_groups"], required_wordings=contract.required_wordings,
            slopes=slopes, radius=design["directional_secant_radius"])
        if adapter.checkpoint.checkpoint_sha256() != ORIGINAL_SMOLLM2:
            raise ValueError("Original model bytes changed")
    report = {"schema": "weightbridge.phase_iiik.portability.v1", "checkpoint_sha256": ORIGINAL_SMOLLM2,
              "protocol_sha256": file_hash(PROTOCOL), "preservation_fit_sha256": manifest["splits"]["preservation-fit"]["sha256"],
              "architecture": capabilities.architecture, "layer": region.layer_index,
              "parent_metrics": compare_endpoint(baseline, baseline, contract)["parent"],
              "sensitivity": sensitivity, "sampling": selection.record(), "trajectory_resources": pool_record["resources"],
              "generation_resources": baseline["resources"], "resources": _resources(wall, cpu)}
    json_dump_atomic(out, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.material, args.out)
    print(json.dumps({key: result[key] for key in ("architecture", "sampling", "resources")}))
