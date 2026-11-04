"""Post-hoc two-comparison linear bounds, without fitting or model execution.

The bound concerns averaged replay-fork margins in the fixed, unrounded
left-factor coordinates. It excludes preservation, BF16, nonlinear execution
and independent behavior. It cannot authorize a candidate or establish global
neural unreachability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from weightbridge.core import file_hash, json_dump_atomic

SCOPE = (
    "Post-hoc diagnostic of two averaged comparison-margin linear inequalities in the audited fixed-right "
    "rank-one left-factor space. No preservation constraints, BF16 rounding or nonlinear generated behavior. "
    "No candidate selection, refitting, independent evaluation or promotion authority."
)


def minimum_pair_norm(gradients, parent_margins, *, tolerance=1e-9, max_condition=1e12):
    """Solve min ||u|| subject to h + A u >= 0 by enumerating two-row KKT sets.

    A nonnegative dual vector also gives a lower bound, sqrt(2*d(lambda)),
    independently of the primal residual. Near-dependent unsolved rows return
    INCONCLUSIVE rather than an infeasibility or global representational claim.
    """
    a, h = np.asarray(gradients, dtype=np.float64), np.asarray(parent_margins, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != 2 or not a.shape[1] or h.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(h).all()
            or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(max_condition) or max_condition <= 1):
        raise ValueError("Expected two finite comparison gradients, margins and bounded numerical tolerances")
    target, gram = -h, a @ a.T
    if not np.isfinite(gram).all():
        raise ValueError("Pair Gram matrix overflowed finite arithmetic")
    eigenvalues = np.linalg.eigvalsh(gram)
    condition = float(np.linalg.cond(gram))
    condition = condition if np.isfinite(condition) else None
    base = {"schema": "weightbridge.local_pair_bound.v1", "scope": SCOPE,
            "parent_average_margins": h.tolist(), "required_margin_changes": target.tolist(),
            "gram_matrix": gram.tolist(), "gram_eigenvalues": eigenvalues.tolist(),
            "gram_condition_number": condition, "tolerance": tolerance,
            "max_condition": max_condition, "model_forward_calls": 0, "model_backward_calls": 0}
    for index in range(2):
        if gram[index, index] == 0 and target[index] > tolerance:
            return {**base, "status": "LOCAL_LINEAR_INFEASIBLE_ZERO_ROW",
                    "zero_gradient_comparison_index": index, "minimum_norm": None,
                    "certificate": {"zero_gradient": True, "required_positive_change": float(target[index])},
                    "interpretation": "Exact zero derivative cannot meet its positive local linear requirement; nonlinear behavior is untested."}
    candidates = []

    def consider(dual, active):
        dual = np.asarray(dual, dtype=float)
        if np.any(dual < -tolerance):
            return
        dual = np.maximum(dual, 0.)
        update = a.T @ dual
        predicted = h + a @ update
        norm = float(np.linalg.norm(update))
        primal_violation = float(max(0., -np.min(predicted)))
        stationarity = float(np.linalg.norm(update - a.T @ dual))
        complementarity = float(np.max(np.abs(dual * predicted)))
        certified = (primal_violation <= tolerance and stationarity <= tolerance * (1 + norm)
                     and complementarity <= tolerance * (1 + norm * norm))
        dual_objective = float(dual @ target - .5 * norm * norm)
        lower_bound = float(np.sqrt(max(0., 2 * dual_objective)))
        if certified:
            candidates.append({"minimum_norm": norm, "predicted_average_margins": predicted.tolist(),
                "active_set": active, "dual_multipliers": dual.tolist(),
                "kkt_certificate": {"dual_nonnegative": bool(np.all(dual >= 0)),
                    "minimum_dual_multiplier": float(np.min(dual)),
                    "maximum_primal_violation": primal_violation, "stationarity_residual_norm": stationarity,
                    "maximum_complementarity_residual": complementarity,
                    "dual_objective": dual_objective, "primal_objective": .5 * norm * norm,
                    "duality_gap": float(.5 * norm * norm - dual_objective),
                    "certified_norm_lower_bound": lower_bound}})

    consider([0., 0.], [])
    for index in range(2):
        if gram[index, index] > 0 and target[index] >= 0:
            dual = np.zeros(2)
            dual[index] = target[index] / gram[index, index]
            consider(dual, [index])
    if condition is not None and condition <= max_condition and eigenvalues[0] > 0:
        consider(np.linalg.solve(gram, target), [0, 1])
    if not candidates:
        return {**base, "status": "INCONCLUSIVE_NO_CERTIFIED_WITNESS", "minimum_norm": None,
                "interpretation": "Bounded active-set arithmetic found no certified solution. Degeneracy or conflict is not an exact infeasibility certificate."}
    best = min(candidates, key=lambda row: row["minimum_norm"])
    return {**base, **best, "status": "KKT_CERTIFIED_LOCAL_LINEAR_BOUND",
            "interpretation": "Minimum for these two local averaged inequalities only. Nonlinear or global behavioral reachability remains untested."}


def analyze(trial, *, radii=(.0125, .0200, .025)):
    wall, cpu = time.perf_counter(), time.process_time()
    trial = Path(trial)
    paths = {"basis": trial / "basis.json", "selection": trial / "supervision-selection.json"}
    before = {key: file_hash(path) for key, path in paths.items()}
    basis, selection = (json.loads(paths[key].read_bytes()) for key in ("basis", "selection"))
    ids = basis["item_ids"]
    index = {name: position for position, name in enumerate(ids)}
    chosen = selection["arms"]["B"]
    if len(set(chosen["comparison_ids"])) != 8 or chosen["family_count"] != 4:
        raise ValueError("Post-hoc bound requires the prospectively selected four paired families")
    radii = tuple(float(radius) for radius in radii)
    if any(not np.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("Diagnostic radii must be finite and positive")
    families = {}
    for family in chosen["family_ids"]:
        members = [name for name in chosen["comparison_ids"] if any(
            row["comparison_id"] == name and row["metadata"]["pair_id"] == family for row in basis["forks"])]
        if len(members) != 2 or len(set(members)) != 2:
            raise ValueError("Selected paired family must bind exactly two distinct averaged gradients")
        selected = [index[name] for name in members]
        record = minimum_pair_norm(np.asarray(basis["projected_gradients"])[selected],
                                   np.asarray(basis["comparison_parent_margins"])[selected])
        lower = record.get("kkt_certificate", {}).get("certified_norm_lower_bound")
        record.update(comparison_ids=members, groups=[basis["item_groups"][i] for i in selected],
                      radius_diagnostics={str(radius): {
                          "zero_margin_bound_exceeds_radius": bool(lower > radius) if lower is not None else None,
                          "interpretation": "Necessary local average-margin condition only; passing this bound does not establish feasible preservation or correct generation."}
                          for radius in radii})
        families[family] = record
    if before != {key: file_hash(path) for key, path in paths.items()}:
        raise ValueError("Frozen pair-bound evidence changed during analysis")
    try:
        import psutil
        peak = psutil.Process().memory_info().rss
    except ImportError:
        peak = None
    return {"schema": "weightbridge.phase_iiin.posthoc_local_pair_bounds.v1", "scope": SCOPE,
            "posthoc": True, "used_for_selection_or_fitting": False, "source_sha256": file_hash(__file__),
            "input_sha256": before, "families": families,
            "resources": {"wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu,
                          "process_rss_bytes_at_completion": peak, "forward_calls": 0, "backward_calls": 0,
                          "generated_response_count": 0}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Post-hoc evidence is write-once")
    json_dump_atomic(args.out, analyze(args.trial))
