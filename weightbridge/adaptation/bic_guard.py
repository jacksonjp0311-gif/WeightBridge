"""Select from measured BIC probes under hard practice preservation gates."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MeasuredProbe:
    coefficients: tuple[float, float]
    objective: float
    target_correct: tuple[bool, ...]
    preservation_first_choice: tuple[bool, ...]


def _validate(probe: MeasuredProbe, target_count: int, preservation_count: int) -> None:
    if (len(probe.coefficients) != 2 or not all(math.isfinite(value) for value in probe.coefficients)
            or not math.isfinite(probe.objective)):
        raise ValueError("Probe requires two finite coefficients and a finite measured objective")
    for mask, count in ((probe.target_correct, target_count),
                        (probe.preservation_first_choice, preservation_count)):
        if count == 0 or len(mask) != count or any(type(value) is not bool for value in mask):
            raise ValueError("Probe masks must contain one boolean per target or preservation row")


def select_measured_probe(
    parent: MeasuredProbe,
    probes: Iterable[MeasuredProbe],
    *,
    require_target_gain: bool = False,
) -> MeasuredProbe | None:
    """Return the lowest-loss eligible measured probe, or abstain.

    The caller measures all probes on the same authorized practice rows in
    the same order. The two preservation choices are encoded as booleans.
    This selection leaves policy validation and independent verification
    with the host.
    """
    target_count, preservation_count = len(parent.target_correct), len(parent.preservation_first_choice)
    _validate(parent, target_count, preservation_count)
    if parent.coefficients != (0.0, 0.0) or type(require_target_gain) is not bool:
        raise ValueError("Guard requires an origin parent and a boolean target gain option")

    winner = None
    for probe in probes:
        _validate(probe, target_count, preservation_count)
        if (probe.coefficients == (0.0, 0.0) or probe.objective >= parent.objective
                or probe.preservation_first_choice != parent.preservation_first_choice
                or any(before and not after for before, after in zip(parent.target_correct, probe.target_correct, strict=True))
                or (require_target_gain and probe.target_correct == parent.target_correct)):
            continue
        if winner is None or probe.objective < winner.objective:
            winner = probe
    return winner
