"""Small dependency-light statistical summaries for paired binary outcomes."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def paired_bootstrap_interval(baseline: Sequence[float], candidate: Sequence[float], *,
                              seed: int = 0, resamples: int = 10000,
                              confidence: float = 0.95) -> dict[str, float | int]:
    """Percentile interval for the paired mean score difference candidate - baseline."""
    before = np.asarray(baseline, dtype=np.float64)
    after = np.asarray(candidate, dtype=np.float64)
    if before.ndim != 1 or after.shape != before.shape or before.size == 0:
        raise ValueError("paired scores must be nonempty 1-D arrays of equal length")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("paired scores must be finite")
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 100:
        raise ValueError("seed must be nonnegative and resamples at least 100")
    if type(confidence) not in {int, float} or not 0 < confidence < 1 or not math.isfinite(confidence):
        raise ValueError("confidence must be in (0, 1)")
    differences = after - before
    rng = np.random.default_rng(seed)
    # Chunk bootstrap draws to keep memory bounded for large benchmark suites.
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = max(1, min(1000, 1_000_000 // before.size))
    offset = 0
    while offset < resamples:
        count = min(chunk_size, resamples - offset)
        indexes = rng.integers(0, before.size, size=(count, before.size))
        means[offset:offset + count] = differences[indexes].mean(axis=1)
        offset += count
    alpha = (1 - float(confidence)) / 2
    return {
        "n": int(before.size), "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, alpha)), "ci_high": float(np.quantile(means, 1 - alpha)),
        "confidence": float(confidence), "resamples": resamples, "seed": seed,
    }
