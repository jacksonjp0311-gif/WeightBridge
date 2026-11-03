"""Compare soft scoring with hard preservation on two measured probe fixtures."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from weightbridge.adaptation import bic, bic_guard


POINTS = ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25),
          (0.25, 0.25), (0.25, -0.25), (-0.25, 0.25), (-0.25, -0.25))
RETENTION_WEIGHT = 2.0
UPDATE_PENALTY = 0.05


def fixture(preservation_response):
    signs = np.array([1, -1])
    parent_margins = np.array([-0.18, -0.03, 0.04, -0.05])
    target_response = signs[:, None] * np.array([[2.2, 1.7], [1.1, 0.8]])
    response = np.vstack((target_response, preservation_response))

    def measure(point):
        coefficients = np.array(point)
        margins = parent_margins + response @ coefficients
        soft_loss, target_loss, kl = bic._objective(margins, parent_margins, signs, RETENTION_WEIGHT)
        penalty = float(UPDATE_PENALTY * np.sum(coefficients**2))
        first_choice = margins >= 0
        target_correct = tuple((first_choice[:2] == (signs > 0)).tolist())
        probe = bic_guard.MeasuredProbe(point, soft_loss + penalty, target_correct, tuple(first_choice[2:].tolist()))
        return probe, {"raw_margins": margins.tolist(), "target_loss": target_loss,
                       "preservation_kl": kl, "update_penalty": penalty}

    parent, parent_details = measure((0.0, 0.0))
    measured = [measure(point) for point in POINTS]
    probes = [probe for probe, _ in measured]
    soft = min([parent, *probes], key=lambda probe: probe.objective)
    guard = bic_guard.select_measured_probe(parent, probes, require_target_gain=True)

    def metrics(probe):
        return {**asdict(probe), "objective_improvement": parent.objective - probe.objective,
                "target_gain": sum(probe.target_correct) - sum(parent.target_correct),
                "target_regressions": sum(before and not after for before, after in
                                          zip(parent.target_correct, probe.target_correct, strict=True)),
                "preservation_flips": sum(before != after for before, after in
                                          zip(parent.preservation_first_choice, probe.preservation_first_choice, strict=True))}

    return {"parent": {**metrics(parent), **parent_details},
            "measured_probes": [{**metrics(probe), **details} for probe, details in measured],
            "soft_selected": metrics(soft), "guard_selected": metrics(guard) if guard else None,
            "guard_decision": "select" if guard else "abstain", "probe_evaluations_per_arm": len(measured),
            "extra_guard_evaluations": 0}


def run():
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    results = {"safe_probe_exists": fixture(np.array([[-0.10, -0.10], [0.11, 0.11]])),
               "conflicting_probe_grid": fixture(np.array([[-0.35, 0.0], [0.0, 0.32]]))}
    return {"schema": "weightbridge.phase_iiib.measured_probe_guard.v1",
            "fixture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "guard_source_sha256": hashlib.sha256(Path(bic_guard.__file__).read_bytes()).hexdigest(),
            "objective_source_sha256": hashlib.sha256(Path(bic.__file__).read_bytes()).hexdigest(),
            "retention_weight": RETENTION_WEIGHT, "update_penalty": UPDATE_PENALTY,
            "require_target_gain": True, "results": results,
            "wall_seconds": time.perf_counter() - start_wall, "process_cpu_seconds": time.process_time() - start_cpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for name, case in result["results"].items():
        print(name, case["soft_selected"], case["guard_decision"], case["guard_selected"])


if __name__ == "__main__":
    main()
