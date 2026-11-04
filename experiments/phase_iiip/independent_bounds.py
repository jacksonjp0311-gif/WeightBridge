"""Model-free second implementation of selected-block numerical bound checks.

This reporter deliberately imports neither the relaxation nor its dual helper.
It reconstructs the weighted matrix row by row and obtains the spectral norm
from a small eigenproblem rather than the solver's SVD. These are numerical
cross-checks, not formally verified interval arithmetic or model evaluations.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _identity(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode("ascii") + b"\0" + array.tobytes()).hexdigest()


def verify_relaxation_bound(blocks, rhs, evidence):
    """Check a recorded Frobenius/spectral bound directly from its finite data."""
    gradients = np.asarray(blocks, dtype=np.float64)
    required = np.asarray(rhs, dtype=np.float64)
    if (gradients.ndim != 3 or not 1 <= gradients.shape[0] <= 64
            or not 1 <= gradients.shape[1] <= 1024 or not 1 <= gradients.shape[2] <= 16
            or required.shape != (len(gradients),) or not np.isfinite(gradients).all()
            or not np.isfinite(required).all()):
        raise ValueError("Independent block check requires bounded finite decision data")
    radius = evidence["radius"]
    tolerance = evidence["tolerance"]
    ids = evidence["row_ids"]
    if (not np.isfinite(radius) or radius <= 0 or not np.isfinite(tolerance) or tolerance <= 0
            or len(ids) != len(gradients) or len(set(ids)) != len(ids)
            or evidence["input_sha256"] != {"blocks": _identity(gradients), "rhs": _identity(required)}):
        raise ValueError("Independent block check input identity or authorization differs")
    certificate = evidence.get("certificate")
    if certificate is None:
        zero = np.all(gradients == 0, axis=(1, 2))
        bad = [item for item, is_zero, bound in zip(ids, zero, required, strict=True) if is_zero and bound > 0]
        if (evidence.get("status") != "INFEASIBLE_ZERO_BLOCK"
                or evidence.get("contradiction_row_id") not in bad):
            raise ValueError("Independent check found no claimed exact zero-block contradiction")
        return {"status": "EXACT_ZERO_BLOCK_CONTRADICTION_CHECKED", "row_id": evidence["contradiction_row_id"],
            "local_linear_scope_only": True, "model_forward_calls": 0, "model_backward_calls": 0}
    multipliers = np.asarray(certificate["original_dual_multipliers"], dtype=np.float64)
    if multipliers.shape != required.shape or not np.isfinite(multipliers).all() or np.any(multipliers < 0):
        raise ValueError("Independent bound requires original-row nonnegative finite dual weights")
    weighted = np.zeros(gradients.shape[1:], dtype=np.float64)
    numerator, absolute_numerator = 0., 0.
    for gradient, bound, weight in zip(gradients, required, multipliers, strict=True):
        weighted += float(weight)*gradient
        numerator += float(weight)*float(bound)
        absolute_numerator += float(weight)*abs(float(bound))
    squared = sum(float(row@row) for row in weighted)
    frobenius = float(np.sqrt(squared))
    # At most16x16, independent of the number of model parameters.
    eigenvalues = np.linalg.eigvalsh(weighted.T@weighted)
    spectral = float(np.sqrt(max(0., eigenvalues[-1])))
    epsilon = np.finfo(np.float64).eps
    objective = numerator-.5*squared
    objective_pad = float(64*epsilon*(1+absolute_numerator+.5*squared))
    numerator_pad = float(64*epsilon*(1+absolute_numerator))
    norm_pad = float(64*epsilon*(1+frobenius))
    spectral_pad = float(64*epsilon*max(weighted.shape)*(1+frobenius))
    objective_lower = float(np.sqrt(max(0., 2*(objective-objective_pad))))
    positive_numerator = max(0., numerator-numerator_pad)
    cauchy_lower = float(positive_numerator/(frobenius+norm_pad))
    rank_one_lower = float(positive_numerator/(spectral+spectral_pad))
    expected = {"dual_objective": objective, "dual_roundoff_padding": objective_pad,
        "dual_numerator": numerator, "dual_numerator_padding": numerator_pad,
        "dual_matrix_frobenius_norm": frobenius, "dual_matrix_norm_padding": norm_pad,
        "dual_objective_norm_lower_bound": objective_lower,
        "dual_cauchy_norm_lower_bound": cauchy_lower,
        "certified_norm_lower_bound": max(objective_lower, cauchy_lower),
        "dual_matrix_spectral_norm": spectral, "dual_spectral_padding": spectral_pad,
        "rank_one_norm_lower_bound": rank_one_lower}
    if not np.isfinite(list(expected.values())).all():
        raise ValueError("Independent bound arithmetic overflowed")
    for key, value in expected.items():
        absolute = 1e-18 if "padding" in key else 1e-13
        if not np.isclose(certificate.get(key, np.nan), value, rtol=1e-10, atol=absolute):
            raise ValueError(f"Independent arithmetic rejects block certificate component {key}")
    if (certificate.get("dual_nonnegative") is not True
            or certificate.get("zero_spectral_denominator") is not bool(np.all(weighted == 0))
            or not np.isclose(evidence["certified_norm_lower_bound"], expected["certified_norm_lower_bound"], rtol=1e-10, atol=1e-13)):
        raise ValueError("Independent arithmetic rejects block certificate summary")
    block_exceeded = expected["certified_norm_lower_bound"] > radius+tolerance
    if evidence["status"] == "CERTIFIED_NORM_BOUND_EXCEEDED" and not block_exceeded:
        raise ValueError("Claimed block radius exclusion lacks its independently checked bound")
    return {"status": "NUMERICAL_DUAL_ARITHMETIC_CHECKED", **expected,
        "checked_by": "explicit weighted sum plus Gram-eigenvalue spectral norm",
        "block_radius_excluded": bool(block_exceeded),
        "rank_one_radius_excluded": bool(rank_one_lower > radius+tolerance),
        "local_linear_scope_only": True, "formal_interval_proof": False,
        "model_forward_calls": 0, "model_backward_calls": 0}
