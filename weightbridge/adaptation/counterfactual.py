"""Paired score diagnostics in a fixed answer coordinate system."""
from __future__ import annotations

from numbers import Real

import numpy as np


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def align_margin(first_score: float, second_score: float, order: tuple[int, int]) -> float:
    """Return canonical answer 0 minus answer 1 regardless of the display."""
    printed = tuple(order)
    if len(printed) != 2 or any(type(index) is not int for index in printed) or set(printed) != {0, 1}:
        raise ValueError("Order must be a permutation of canonical answers 0 and 1")
    margin = _finite(first_score, "first_score") - _finite(second_score, "second_score")
    return _finite(margin if printed[0] == 0 else -margin, "aligned margin")


def paired_components(margin: float, reversed_margin: float) -> tuple[float, float]:
    """Return the odd (evidence) and even (preference) margin components."""
    original = 0.5 * _finite(margin, "margin")
    reversed_question = 0.5 * _finite(reversed_margin, "reversed_margin")
    return original - reversed_question, original + reversed_question


def paired_loss(margin: float, reversed_margin: float, *, preference_weight: float = 0.0
                ) -> tuple[float, tuple[float, float]]:
    """Return the proposed additive loss and its derivatives in margin space.

    The first margin must favor the first canonical answer and the second
    margin must favor the second. Validating that semantic pairing is the
    responsibility of the frozen measurement protocol.
    """
    _, preference = paired_components(margin, reversed_margin)
    weight = _finite(preference_weight, "preference_weight")
    if weight < 0:
        raise ValueError("Preference weight must be nonnegative")
    original_loss, reversed_loss = np.logaddexp(0, -margin), np.logaddexp(0, reversed_margin)
    loss = float(original_loss + reversed_loss) + (weight * preference * preference if weight else 0.0)
    if not np.isfinite(loss):
        raise ValueError("Paired loss must be finite")
    preference_gradient = weight * preference
    return loss, (-float(np.exp(-np.logaddexp(0, margin))) + preference_gradient,
                  float(np.exp(-np.logaddexp(0, -reversed_margin))) + preference_gradient)
