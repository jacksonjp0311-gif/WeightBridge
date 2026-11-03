"""Host-authorized AWL intent for the Phase III-B synthesis backends."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Literal

from ..agents.hf_direction import _DOWN_PROJECTION, propose_direction
from ..evaluation.hf_edit import _load_suite
from ..patches.huggingface import _policy


Method = Literal["activation_ridge", "bic_first_order", "bic_interaction"]
_METHODS = ("activation_ridge", "bic_first_order", "bic_interaction")
_SCHEMA = "weightbridge.awl.v2"


@dataclass(frozen=True)
class BehavioralIntent:
    """A binary behavior objective and a bounded numerical request."""

    objective: Literal["binary_choice_margin"]
    target: tuple[str, ...]
    preserve: tuple[str, ...]
    representation: Literal["rank_one_slices"]
    method: Method
    radius: float


def _names(value, *, limit):
    return (isinstance(value, list) and len(value) <= limit
            and all(isinstance(item, str) and item.strip() == item and 1 <= len(item) <= 256 for item in value)
            and len(set(value)) == len(value))


def parse_awl_v2(document: object) -> BehavioralIntent:
    """Parse typed intent. The request grants no policy or evaluation authority."""
    if (not isinstance(document, dict)
            or document.keys() != {"schema", "objective", "target", "preserve", "optimize", "representation"}
            or document["schema"] != _SCHEMA):
        raise ValueError("Expected the exact AWL v2 fields")
    if document["objective"] != "binary_choice_margin" or document["representation"] != "rank_one_slices":
        raise ValueError("AWL v2 requires binary choice margins and rank-one slices")
    optimize, target, preserve = document["optimize"], document["target"], document["preserve"]
    if (not isinstance(optimize, dict) or optimize.keys() != {"method", "radius"}
            or optimize["method"] not in _METHODS):
        raise ValueError("AWL v2 requires a supported solver and radius")
    method, radius = optimize["method"], optimize["radius"]
    count = 1 if method == "activation_ridge" else 2
    if (not _names(target, limit=count) or len(target) != count
            or not all(_DOWN_PROJECTION.fullmatch(tensor) for tensor in target)):
        raise ValueError(f"Solver requires {count} distinct Qwen MLP projection names")
    if not _names(preserve, limit=16):
        raise ValueError("Preserve must name distinct task families")
    if type(radius) not in (int, float) or not 0 < radius <= 8 or (method == "activation_ridge" and radius > 1):
        raise ValueError("Invalid AWL v2 radius")
    return BehavioralIntent("binary_choice_margin", tuple(target), tuple(preserve), "rank_one_slices", method, float(radius))


def compile_awl_v2(
    base, practice, document, *, policy, inference, authorized_preservation_families,
    max_radius, max_probe_batches,
):
    """Compile a practice-only candidate within host limits, without writing weights."""
    intent = parse_awl_v2(document)
    allowed, max_entries, _, _, _, _, _, _ = _policy(policy)
    if not set(intent.target).issubset(allowed):
        raise ValueError("AWL v2 target is outside the host tensor policy")
    if (type(max_radius) not in (int, float) or not 0 < max_radius <= 8 or intent.radius > max_radius
            or type(max_probe_batches) is not int or not 1 <= max_probe_batches <= 512):
        raise ValueError("AWL v2 request exceeds host radius or probe limits")
    requested_families = set(intent.preserve) - {"generic"}
    if not requested_families.issubset(authorized_preservation_families):
        raise ValueError("AWL v2 preservation family is outside the host practice scope")
    if not isinstance(practice, dict) or practice.keys() != {"schema", "choices", "target", "sentinel"}:
        raise ValueError("Compiler accepts only a host-owned practice suite")
    suite = _load_suite(json.dumps(practice, allow_nan=False).encode())
    rows = suite["target"] + suite["sentinel"]
    if len(suite["choices"]) != 2 or len(rows) > 64:
        raise ValueError("AWL v2 requires binary choices and at most 64 practice rows")
    family_rows = [row.get("family", "generic") for row in suite["sentinel"]]
    if (not all(isinstance(family, str) for family in family_rows)
            or set(family_rows) - {"generic"} != requested_families
            or ("generic" in intent.preserve and "generic" not in family_rows)):
        raise ValueError("Authorized preservation families must match the practice sentinel rows")
    batch_size = inference.get("batch_size")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("Invalid host practice batch size")
    batches = (len(rows) + batch_size - 1) // batch_size
    passes = {"activation_ridge": 1, "bic_first_order": 6, "bic_interaction": 10}[intent.method]
    if batches * passes > max_probe_batches:
        raise ValueError("Requested solver exceeds the host forward batch budget")

    started, cpu = time.perf_counter(), time.process_time()
    if intent.method == "activation_ridge":
        patch = propose_direction(Path(base), suite, intent.target[0], intent.radius, inference)
        diagnostics = {"forward_batches": batches, "backward_batches": 0, "radius_semantics": "legacy_strength"}
    else:
        from .bic import propose_bic

        patch, diagnostics = propose_bic(Path(base), suite, intent.target, inference,
                                         interaction=intent.method == "bic_interaction", radius=intent.radius,
                                         max_entries=min(max_entries, 65536))
    return patch, {**diagnostics, "method": intent.method, "preservation_families": list(intent.preserve),
                   "forward_batch_cap": max_probe_batches, "elapsed_seconds": time.perf_counter() - started,
                   "process_cpu_seconds": time.process_time() - cpu}
