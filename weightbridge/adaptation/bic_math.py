"""Local objective models in a bounded two-coordinate update space."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LocalQuadratic:
    origin_loss: float
    gradient: np.ndarray
    hessian: np.ndarray
    axial_curvature: np.ndarray
    probe_step: float
    probes: tuple[tuple[tuple[float, float], float], ...]

    def predict(self, coefficients: np.ndarray) -> float:
        return float(self.origin_loss + self.gradient @ coefficients
                     + 0.5 * coefficients @ self.hessian @ coefficients)


def fit_local_quadratic(
    evaluate: Callable[[np.ndarray], float],
    origin_loss: float,
    *,
    radius: float,
    interaction: bool,
) -> LocalQuadratic:
    """Estimate an axial gradient and an optional mixed partial inside a disk.

    Both modes measure axial curvature. Neither uses it for selection, so the
    selection-model difference is the mixed term alone. The host must verify
    the selected point with the actual objective.
    """
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(origin_loss):
        raise ValueError("Local model requires a positive radius and finite origin loss")
    step = radius / math.sqrt(2)
    if step * step == 0:
        raise ValueError("Local model radius is too small for finite differences")
    probes = []

    def probe(x: float, y: float) -> float:
        measured = float(evaluate(np.array([x, y])))
        if not math.isfinite(measured):
            raise ValueError("Local objective probe returned a nonfinite loss")
        probes.append(((x, y), measured))
        return measured

    x_plus, x_minus = probe(step, 0.0), probe(-step, 0.0)
    y_plus, y_minus = probe(0.0, step), probe(0.0, -step)
    gradient = np.array([x_plus - x_minus, y_plus - y_minus]) / (2 * step)
    axial = np.array([x_plus + x_minus, y_plus + y_minus]) - 2 * origin_loss
    axial /= step**2
    hessian = np.zeros((2, 2))
    if interaction:
        pp, pm = probe(step, step), probe(step, -step)
        mp, mm = probe(-step, step), probe(-step, -step)
        hessian[0, 1] = hessian[1, 0] = ((pp - pm) - (mp - mm)) / (4 * step**2)
    if not all(np.isfinite(item).all() for item in (gradient, hessian, axial)):
        raise ValueError("Local finite differences overflowed")
    return LocalQuadratic(float(origin_loss), gradient, hessian, axial, step, tuple(probes))


def select_coefficients(local: LocalQuadratic, *, radius: float, update_penalty: float = 0.0) -> np.ndarray:
    """Minimize the surrogate on a fixed polar grid, including abstention."""
    if (not math.isfinite(radius) or radius <= 0
            or not math.isfinite(update_penalty) or update_penalty < 0):
        raise ValueError("Coefficient search requires a positive radius and nonnegative penalty")
    angles = np.arange(256) * (2 * np.pi / 256)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    radii = np.linspace(radius / 8, radius, 8)
    grid = np.vstack((np.zeros((1, 2)), (radii[:, None, None] * unit).reshape(-1, 2)))
    linear = grid @ local.gradient
    quadratic = 0.5 * np.sum((grid @ local.hessian) * grid, axis=1)
    penalty = update_penalty * np.sum(grid * grid, axis=1)
    return grid[np.argmin(linear + quadratic + penalty)]
