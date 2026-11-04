"""Small local block relaxations and bounded unit-right rank-one search.

All witnesses concern supplied parent-linearized decisions. A relaxed matrix
is diagnostic only. Neither this module nor a numerical witness authorizes a
stored checkpoint, behavior claim, hidden evaluation or model promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np

from .joint_feasibility import solve_minimum_norm


SCOPE = ("Finite local linear selected-block system, unrounded arithmetic. "
         "Numerical padding is not formally verified interval arithmetic. "
         "Relaxed matrices must never be installed as model updates.")
DEFAULT_SEARCH = {
    "starts": 4, "iterations": 8, "line_steps": 4, "initial_step": .25,
    "inner_max_sweeps": 4096, "tolerance": 1e-8,
}


@dataclass(frozen=True)
class BlockFeasibility:
    update: np.ndarray | None
    minimum_norm_update: np.ndarray | None
    evidence: dict

    def record(self):
        return self.evidence


@dataclass(frozen=True)
class RightFactorSearch:
    left: np.ndarray | None
    right: np.ndarray | None
    evidence: dict

    def record(self):
        return self.evidence


def _hash(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode("ascii") + b"\0" + array.tobytes()).hexdigest()


def _validated(blocks, rhs, radius, max_sweeps=4096, tolerance=1e-8, row_ids=None):
    g, b = np.asarray(blocks, dtype=np.float64), np.asarray(rhs, dtype=np.float64)
    if (g.ndim != 3 or not 1 <= g.shape[0] <= 64 or not 1 <= g.shape[1] <= 1024
            or not 1 <= g.shape[2] <= 16 or b.shape != (len(g),)
            or not np.isfinite(g).all() or not np.isfinite(b).all()
            or not np.isfinite(radius) or radius <= 0 or not np.isfinite(tolerance) or tolerance <= 0
            or type(max_sweeps) is not int or not 1 <= max_sweeps <= 65536):
        raise ValueError("Expected at most64 finite blocks of at most1024x16 and bounded numerical settings")
    ids = tuple(f"decision/{i}" for i in range(len(g))) if row_ids is None else tuple(row_ids)
    if len(ids) != len(g) or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Every block needs a distinct nonempty decision identity")
    return g, b, ids


def canonical_factors(left, right):
    """Normalize right and sign without changing the outer-product update."""
    u, v = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if (u.ndim != 1 or v.ndim != 1 or not len(u) or not len(v)
            or not np.isfinite(u).all() or not np.isfinite(v).all()):
        raise ValueError("Finite nonempty factor vectors are required")
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Right factor must have a finite nonzero norm")
    u, v = u * norm, v / norm
    if v[int(np.argmax(np.abs(v)))] < 0:
        u, v = -u, -v
    if not np.isfinite(u).all():
        raise ValueError("Factor rescaling overflowed")
    return u, v


def _right(value, width):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (width,):
        raise ValueError("Right factor width differs from selected columns")
    return canonical_factors(np.ones(1), v)[1]


def _dual_arithmetic(g, b, dual):
    weights = np.asarray(dual, dtype=np.float64)
    if weights.shape != b.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("A block bound requires original-row nonnegative finite multipliers")
    matrix = np.einsum("i,ijk->jk", weights, g, optimize=True)
    norm = float(np.linalg.norm(matrix))
    square = float(np.sum(matrix * matrix))
    numerator = float(weights @ b)
    eps = np.finfo(np.float64).eps
    scale = float(np.abs(b) @ weights)
    padding = float(64 * eps * (1 + scale + .5 * square))
    numerator_padding = float(64 * eps * (1 + scale))
    denominator_padding = float(64 * eps * (1 + norm))
    objective = numerator - .5 * square
    objective_bound = float(np.sqrt(max(0., 2 * (objective - padding))))
    ratio_bound = float(max(0., numerator - numerator_padding) / (norm + denominator_padding))
    singular = np.linalg.svd(matrix, compute_uv=False)
    spectral = float(singular[0])
    spectral_padding = float(64 * eps * max(matrix.shape) * (1 + norm))
    spectral_bound = float(max(0., numerator - numerator_padding) / (spectral + spectral_padding))
    return matrix, {
        "original_dual_multipliers": weights.tolist(), "dual_nonnegative": True,
        "dual_objective": objective, "dual_roundoff_padding": padding,
        "dual_numerator": numerator, "dual_numerator_padding": numerator_padding,
        "dual_matrix_frobenius_norm": norm, "dual_matrix_norm_padding": denominator_padding,
        "dual_objective_norm_lower_bound": objective_bound,
        "dual_cauchy_norm_lower_bound": ratio_bound,
        "certified_norm_lower_bound": max(objective_bound, ratio_bound),
        "dual_matrix_spectral_norm": spectral, "dual_spectral_padding": spectral_padding,
        "rank_one_norm_lower_bound": spectral_bound,
        "zero_spectral_denominator": spectral == 0,
    }


def verify_block_certificate(blocks, rhs, evidence):
    """Independently recompute a recorded bound from raw blocks and multipliers."""
    g, b, _ = _validated(blocks, rhs, evidence["radius"], row_ids=evidence["row_ids"])
    if evidence.get("input_sha256") != {"blocks": _hash(g), "rhs": _hash(b)}:
        raise ValueError("Block certificate input identity differs")
    certificate = evidence.get("certificate")
    if certificate is None:
        if evidence["status"] == "INFEASIBLE_ZERO_BLOCK":
            indices = np.flatnonzero((np.max(np.abs(g), axis=(1, 2)) == 0) & (b > 0))
            if not len(indices):
                raise ValueError("Claimed zero block contradiction is absent")
            return {"zero_block_verified": True}
        raise ValueError("Block numerical evidence lacks a dual certificate")
    _, replay = _dual_arithmetic(g, b, certificate["original_dual_multipliers"])
    for key, value in replay.items():
        if isinstance(value, list):
            continue
        if isinstance(value, bool):
            if certificate.get(key) is not value:
                raise ValueError("Block certificate boolean does not independently replay")
        elif not np.isclose(certificate.get(key, np.nan), value, rtol=1e-11, atol=1e-14):
            raise ValueError("Block numerical bound does not follow from original-row dual arithmetic")
    if not np.isclose(evidence["certified_norm_lower_bound"], replay["certified_norm_lower_bound"], rtol=1e-11, atol=1e-14):
        raise ValueError("Top-level block bound differs from the checked certificate")
    return replay


def solve_block_relaxation(blocks, rhs, *, radius, max_sweeps=4096, tolerance=1e-8, row_ids=None):
    """Minimum-Frobenius matrix relaxation via a bounded constraint Gram matrix.

    This interface deliberately does not flatten a1024x16 matrix into the
    existing <=1024-dimensional vector solver. The only quadratic matrix is
    a <=64x64 Gram matrix. Returned relaxed matrices are diagnostic only.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    g, b, ids = _validated(blocks, rhs, radius, max_sweeps, tolerance, row_ids)
    base = {"schema": "weightbridge.block_relaxation.v1", "scope": SCOPE,
        "radius": float(radius), "tolerance": tolerance, "maximum_sweeps": max_sweeps,
        "block_shape": list(g.shape[1:]), "row_count": len(g), "row_ids": list(ids),
        "input_sha256": {"blocks": _hash(g), "rhs": _hash(b)},
        "matrix_installation_authorized": False, "certificates_are_numerical_not_formal_intervals": True}
    def resources():
        return {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                "forward_calls": 0, "backward_calls": 0, "generated_response_count": 0}
    scales = np.max(np.abs(g), axis=(1, 2))
    scaled = np.divide(g, scales[:, None, None], out=np.zeros_like(g), where=scales[:, None, None] > 0)
    norms = scales * np.sqrt(np.sum(scaled * scaled, axis=(1, 2)))
    if not np.isfinite(norms).all():
        raise ValueError("Block row norm overflow")
    nonzero = norms > 0
    impossible = np.flatnonzero(~nonzero & (b > 0))
    if len(impossible):
        evidence = {**base, "status": "INFEASIBLE_ZERO_BLOCK", "within_radius": False,
            "minimum_norm": None, "certified_norm_lower_bound": None, "certificate": None,
            "contradiction_row_id": ids[int(impossible[0])], "converged": False, "sweeps": 0,
            "resources": resources()}
        return BlockFeasibility(None, None, evidence)
    axes = np.divide(g, norms[:, None, None], out=np.zeros_like(g), where=nonzero[:, None, None])
    levels = np.divide(b, norms, out=np.zeros_like(b), where=nonzero)
    if not np.isfinite(levels).all():
        raise ValueError("Normalized block inequality overflow")
    gram = np.einsum("ijk,ljk->il", axes, axes, optimize=True)
    dual, products = np.zeros(len(g)), np.zeros(len(g))
    best_matrix = best_primal = best_certificate = None
    converged = False
    failure = None
    for sweep in range(max_sweeps + 1):
        if sweep:
            for index in range(len(g)):
                if not nonzero[index]:
                    continue
                previous = dual[index]
                dual[index] = max(0., previous + (levels[index]-products[index])/gram[index, index])
                products += (dual[index]-previous)*gram[:, index]
            products = gram @ dual
        original = np.divide(dual, norms, out=np.zeros_like(dual), where=nonzero)
        if not np.isfinite(original).all():
            failure = "nonfinite_dual_iterate"
            break
        matrix, certificate = _dual_arithmetic(g, b, original)
        slack = np.einsum("ijk,jk->i", g, matrix, optimize=True)-b
        normalized_slack = np.einsum("ijk,jk->i", axes, matrix, optimize=True)-levels
        norm = float(np.linalg.norm(matrix))
        residual = np.where(dual > 0, np.abs(normalized_slack), np.maximum(0., -normalized_slack))
        primal_violation = max(0., float(-slack.min()))
        normalized_violation = max(0., float(-normalized_slack.min()))
        gap = .5*norm*norm-certificate["dual_objective"]
        observation = {**certificate, "primal_norm": norm, "constraint_slacks": slack.tolist(),
            "maximum_primal_violation": primal_violation,
            "maximum_normalized_primal_violation": normalized_violation,
            "projected_dual_residual": float(residual.max()), "duality_gap": gap,
            "active_constraint_ids": [item for item, value in zip(ids, slack, strict=True) if abs(value) <= tolerance],
            "normalized_dual_multipliers": dual.tolist()}
        if not all(np.isfinite(observation[k]) for k in ("primal_norm", "dual_objective", "certified_norm_lower_bound")):
            failure = "nonfinite_primal_or_dual_measurement"
            break
        if best_certificate is None or observation["certified_norm_lower_bound"] > best_certificate["certified_norm_lower_bound"]:
            best_certificate = observation
        feasible = primal_violation <= tolerance and normalized_violation <= tolerance
        if feasible and (best_primal is None or norm < best_primal["primal_norm"]):
            best_matrix, best_primal = matrix.copy(), observation
        if feasible and observation["projected_dual_residual"] <= tolerance and abs(gap) <= tolerance*(1+.5*norm*norm):
            converged, best_matrix, best_primal = True, matrix.copy(), observation
            break
    lower = best_certificate["certified_norm_lower_bound"] if best_certificate else None
    norm = best_primal["primal_norm"] if best_primal else None
    within = best_primal is not None and norm <= radius and not (lower is not None and lower > radius+tolerance)
    status = ("CERTIFIED_NORM_BOUND_EXCEEDED" if lower is not None and lower > radius+tolerance else
              "LOCAL_LINEAR_FEASIBLE" if within and converged else
              "FEASIBLE_BOUNDED_WITNESS" if within else "BOUNDED_NO_WITNESS")
    evidence = {**base, "status": status, "within_radius": bool(within), "converged": converged,
        "minimum_norm": norm if converged else None, "best_primal_norm": norm,
        "certified_norm_lower_bound": lower, "certificate": best_certificate,
        "primal_witness": best_primal, "sweeps": sweep, "numerical_failure": failure,
        "bounded_search_failure_proves_infeasibility": False, "resources": resources()}
    verify_block_certificate(g, b, evidence)
    return BlockFeasibility(best_matrix.copy() if within else None,
                            best_matrix.copy() if best_matrix is not None else None, evidence)


def outer_value_gradient(blocks, rhs, right, inner, *, tolerance=1e-8):
    """Envelope derivative, only where the convex inner KKT checks qualify it."""
    g, b, _ = _validated(blocks, rhs, inner.record()["radius"])
    v = np.asarray(right, dtype=np.float64)
    if v.shape != (g.shape[2],) or not np.isclose(np.linalg.norm(v), 1., atol=1e-10, rtol=1e-10):
        raise ValueError("Outer derivative requires the actual unit right factor")
    evidence = inner.record()
    witness = evidence.get("primal_witness")
    u = inner.minimum_norm_update
    if not evidence.get("converged") or witness is None or u is None:
        return None, None, {"qualified": False, "reason": "inner_minimum_or_kkt_unavailable"}
    dual = np.asarray(witness["original_dual_multipliers"])
    rows = np.einsum("ijk,k->ij", g, v)
    residual = float(np.linalg.norm(u-rows.T@dual))
    slack = rows@u-b
    complementarity = float(np.max(np.abs(dual*slack)))
    if (np.any(dual < 0) or np.min(slack) < -tolerance or residual > tolerance*(1+np.linalg.norm(u))
            or complementarity > tolerance*(1+np.linalg.norm(u)**2)):
        return None, None, {"qualified": False, "reason": "inner_kkt_check_failure",
            "stationarity_residual": residual, "complementarity_residual": complementarity}
    gradient = -np.einsum("i,ijk,j->k", dual, g, u, optimize=True)
    return .5*float(u@u), gradient, {"qualified": True, "reason": "original_row_envelope_derivative",
        "original_dual_multipliers": dual.tolist(), "stationarity_residual": residual,
        "complementarity_residual": complementarity,
        "tangent_gradient": (gradient-v*float(v@gradient)).tolist(),
        "scope": "Local envelope derivative away from active-set transitions; nonsmooth points remain possible."}


def search_right_factor(blocks, rhs, initial_right, *, radius, settings=None, row_ids=None, relaxation=None):
    """One preregistrable bounded multistart sphere search, not a global solver."""
    wall, cpu = time.perf_counter(), time.process_time()
    options = dict(DEFAULT_SEARCH)
    if settings:
        if set(settings)-set(options):
            raise ValueError("Unknown right-factor search setting")
        options.update(settings)
    if (any(type(options[k]) is not int for k in ("starts", "iterations", "line_steps", "inner_max_sweeps"))
            or not 1 <= options["starts"] <= 4 or not 0 <= options["iterations"] <= 8
            or not 1 <= options["line_steps"] <= 4 or not 0 < options["initial_step"] <= .5):
        raise ValueError("Right-factor outer budget exceeds the bounded research interface")
    g, b, ids = _validated(blocks, rhs, radius, options["inner_max_sweeps"], options["tolerance"], row_ids)
    initial = _right(initial_right, g.shape[2])
    relaxed = relaxation or solve_block_relaxation(g, b, radius=radius,
        max_sweeps=options["inner_max_sweeps"], tolerance=options["tolerance"], row_ids=ids)
    if relaxed.record()["radius"] != radius or relaxed.record()["row_ids"] != list(ids):
        raise ValueError("Supplied relaxation belongs to a different authorization or decision order")
    verify_block_certificate(g, b, relaxed.record())
    base = {"schema": "weightbridge.optimized_right_search.v1", "scope": SCOPE,
        "settings": options, "input_sha256": {"blocks": _hash(g), "rhs": _hash(b)},
        "radius": radius, "row_ids": list(ids), "initial_right": initial.tolist(),
        "global_rank_one_optimum_claimed": False, "production_authority": False,
        "maximum_inner_calls": options["starts"]*(1+options["iterations"]*options["line_steps"])}
    def finish(status, left=None, right=None, **extra):
        return RightFactorSearch(left, right, {**base, "status": status, **extra,
            "resources": {"wall_seconds": time.perf_counter()-wall, "cpu_seconds": time.process_time()-cpu,
                          "forward_calls": 0, "backward_calls": 0, "generated_response_count": 0}})
    if relaxed.record()["status"] in {"INFEASIBLE_ZERO_BLOCK", "CERTIFIED_NORM_BOUND_EXCEEDED"}:
        return finish("BLOCK_RELAXATION_REFUSAL", relaxation=relaxed.record(), inner_calls=0, starts=[])
    rank_bound = relaxed.record().get("certificate", {}).get("rank_one_norm_lower_bound", 0.)
    if rank_bound > radius+options["tolerance"]:
        return finish("RANK_ONE_NORM_BOUND_EXCEEDED", relaxation=relaxed.record(), inner_calls=0, starts=[])
    starts = [("inherited", initial)]
    matrix = relaxed.minimum_norm_update
    if matrix is not None and np.linalg.norm(matrix) > 0:
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        starts.append(("relaxed-leading", _right(vt[0], g.shape[2])))
    else:
        vt = None
    aggregate = np.einsum("i,ijk->jk", np.maximum(b, 0.), g, optimize=True)
    if np.linalg.norm(aggregate) > 0:
        starts.append(("evidence-leading", _right(np.linalg.svd(aggregate, full_matrices=False)[2][0], g.shape[2])))
    if vt is not None and len(vt) > 1:
        starts.append(("relaxed-balanced", _right(vt[0]+vt[1], g.shape[2])))
    chosen, excluded = [], []
    for name, v in starts[:options["starts"]]:
        if any(np.linalg.norm(v-prev) < 1e-10 for _, prev in chosen):
            excluded.append({"name": name, "reason": "duplicate_canonical_start", "right": v.tolist()})
        else:
            chosen.append((name, v))
    calls, traces, candidates = 0, [], []
    def compact(inner):
        detail = inner.record()
        result = {key: detail.get(key) for key in ("status", "sweeps", "converged", "primal_feasible",
            "within_radius", "minimum_norm", "best_primal_norm", "certified_norm_lower_bound",
            "constraint_slacks", "active_constraint_ids", "numerical_failure")}
        witness = detail.get("primal_witness") or detail.get("certificate") or {}
        result["residuals"] = {key: witness.get(key) for key in ("maximum_primal_violation",
            "maximum_normalized_primal_violation", "projected_dual_residual", "duality_gap",
            "maximum_complementarity_residual", "stationarity_residual_norm")}
        return result
    def evaluate(v):
        nonlocal calls
        calls += 1
        return solve_minimum_norm(np.einsum("ijk,k->ij", g, v), b, radius=radius,
            max_sweeps=options["inner_max_sweeps"], tolerance=options["tolerance"], row_ids=ids)
    for name, v in chosen:
        inner = evaluate(v)
        trace = {"name": name, "initial_right": v.tolist(), "steps": []}
        for iteration in range(options["iterations"]+1):
            value, gradient, deriv = outer_value_gradient(g, b, v, inner, tolerance=options["tolerance"])
            trace["steps"].append({"iteration": iteration, "right": v.tolist(), "inner": compact(inner), "derivative": deriv})
            u = inner.minimum_norm_update
            if u is not None:
                candidates.append((float(np.linalg.norm(u)), name, iteration, u.copy(), v.copy(), inner.record()))
            if iteration == options["iterations"] or gradient is None:
                break
            tangent = gradient-v*float(v@gradient)
            length = float(np.linalg.norm(tangent))
            if length <= options["tolerance"]:
                trace["stop_reason"] = "small_qualified_tangent_gradient"
                break
            accepted, trials = False, []
            for line in range(options["line_steps"]):
                step = options["initial_step"]*(.5**line)
                trial_v = _right(v-step*tangent/length, g.shape[2])
                trial = evaluate(trial_v)
                trial_value = (.5*float(trial.minimum_norm_update@trial.minimum_norm_update)
                               if trial.minimum_norm_update is not None else None)
                improved = bool(trial.record().get("converged") and trial_value is not None
                                and trial_value < value-options["tolerance"]*max(1., abs(value)))
                trials.append({"line": line, "step": step, "right": trial_v.tolist(),
                               "accepted": improved, "inner": compact(trial)})
                if improved:
                    v, inner, accepted = trial_v, trial, True
                    break
            trace["steps"][-1]["line_search"] = trials
            if not accepted:
                trace["stop_reason"] = "bounded_line_search_no_decrease"
                break
        traces.append(trace)
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    best = candidates[0] if candidates else None
    eligible = best is not None and best[0] <= radius and best[5].get("within_radius")
    left, right = canonical_factors(best[3], best[4]) if eligible else (None, None)
    return finish("LOCAL_RANK_ONE_WITNESS" if eligible else "BOUNDED_NO_RANK_ONE_WITNESS",
        left=left, right=right, relaxation=relaxed.record(), starts=traces, duplicate_starts=excluded,
        inner_calls=calls, best_norm=None if best is None else best[0],
        best_left=None if best is None else best[3].tolist(), best_right=None if best is None else best[4].tolist(),
        best_inner=None if best is None else best[5],
        selected_start=None if best is None else best[1], selected_iteration=None if best is None else best[2])
