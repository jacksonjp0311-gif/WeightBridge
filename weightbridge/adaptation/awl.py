"""Strict JSON intent for host-controlled weight adaptation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SCHEMA = "weightbridge.awl.v1"
_FIELDS = {"schema", "observe", "diagnose", "target", "optimize"}
_LIST_LIMIT = 32


@dataclass(frozen=True)
class OptimizationSpec:
    method: Literal["activation_ridge"]
    strengths: tuple[float, ...]


@dataclass(frozen=True)
class AdaptationIntent:
    observe: str
    diagnose: str
    target: tuple[str, ...]
    optimize: OptimizationSpec


def _short_text(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"AWL {name} must be nonempty text of at most {limit} characters")
    return value.strip()


def parse_awl(document: object) -> AdaptationIntent:
    """Parse a decoded AWL JSON document without granting host authority."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"AWL document must use {SCHEMA}")
    if set(document) != _FIELDS:
        raise ValueError("AWL fields must be exactly schema, observe, diagnose, target and optimize")
    observe = _short_text(document["observe"], "observe", 256)
    diagnose = _short_text(document["diagnose"], "diagnose", 512)
    target = document["target"]
    if (not isinstance(target, list) or not 1 <= len(target) <= _LIST_LIMIT
            or not all(isinstance(name, str) and name and name == name.strip() and len(name) <= 256
                       for name in target) or len(set(target)) != len(target)):
        raise ValueError("AWL target must list 1 to 32 distinct tensor names")
    optimize = document["optimize"]
    if (not isinstance(optimize, dict) or set(optimize) != {"method", "strengths"}
            or optimize["method"] != "activation_ridge"):
        raise ValueError("AWL optimize must use activation_ridge and specify strengths")
    strengths = optimize["strengths"]
    if (not isinstance(strengths, list) or not 1 <= len(strengths) <= _LIST_LIMIT
            or not all(type(value) in {int, float} and 0 < value <= 1 for value in strengths)
            or len(set(strengths)) != len(strengths)):
        raise ValueError("AWL strengths must list 1 to 32 distinct finite numbers in (0, 1]")
    return AdaptationIntent(observe, diagnose, tuple(target),
                            OptimizationSpec("activation_ridge", tuple(map(float, strengths))))
