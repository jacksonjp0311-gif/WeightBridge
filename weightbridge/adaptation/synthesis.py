"""Plan bounded AWL proposals and compile them on a host practice suite."""
from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from ..agents.hf_direction import propose_direction
from .awl import AdaptationIntent


def plan_candidates(
    intent: AdaptationIntent,
    allowed_tensors: Iterable[str],
    max_candidates: int,
    preferred_tensors: Iterable[str] = (),
) -> tuple[tuple[str, float], ...]:
    """Prioritize requested tensors within each strength and cap the plan."""
    if type(max_candidates) is not int or max_candidates < 0:
        raise ValueError("Host candidate budget must be a nonnegative integer")
    allowed = set(allowed_tensors)
    requested = set(intent.target)
    if requested - allowed:
        raise ValueError("AWL target is outside the host direction allowlist")
    ordered = dict.fromkeys(tensor for tensor in preferred_tensors if tensor in requested)
    ordered.update(dict.fromkeys(intent.target))
    candidates = ((tensor, strength) for strength in intent.optimize.strengths for tensor in ordered)
    return tuple(islice(candidates, min(max_candidates, len(ordered) * len(intent.optimize.strengths))))


def synthesize_patch(
    base: Path,
    practice: dict[str, Any],
    tensor: str,
    strength: float,
    inference: dict[str, Any],
) -> dict[str, Any]:
    """Compile a rank-one proposal for the host to validate and score."""
    return propose_direction(base, practice, tensor, strength, inference)
