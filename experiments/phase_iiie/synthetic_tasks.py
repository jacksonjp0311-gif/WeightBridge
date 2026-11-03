"""Frozen, two-parameter behavioral repair fixtures for the ANMC study."""

from __future__ import annotations

from dataclasses import astuple, dataclass, replace
from hashlib import sha256
from math import isfinite

import numpy as np


FAMILIAR = (
    "first_order_repair",
    "preservation_constrained",
    "dormant_interaction",
    "measurement_artifact",
    "unreachable",
    "fragile_margin",
)
HELD = ("sensitivity_reversal", "expensive_diagnostic")
METHODS = ("first_order", "constrained", "interaction")
FEATURE_LEVELS = ("A", "B", "C", "D")
SPLITS = ("train", "calibration", "test", "shift")
_PRECISION = np.dtype("<f2")
_PRACTICE_FLOOR = 0.025
_SCALES = (1.0, 0.75, 0.5, 0.25)
_ANGLES = np.arange(64) * (2 * np.pi / 64)
_DIRECTIONS = np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))
_GRID = np.concatenate([_DIRECTIONS * scale for scale in _SCALES])


@dataclass(frozen=True)
class Cost:
    diagnostics: int = 0
    forwards: int = 0
    backwards: int = 0
    curvature: int = 0
    linear_trials: int = 0
    candidates: int = 0
    reloads: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(*(a + b for a, b in zip(astuple(self), astuple(other), strict=True)))

    def __sub__(self, other: Cost) -> Cost:
        return Cost(*(a - b for a, b in zip(astuple(self), astuple(other), strict=True)))

    @property
    def units(self) -> float:
        """Declared work proxy. Timing is measured separately by the runner."""
        return (self.diagnostics + self.forwards + 2 * self.backwards + 4 * self.curvature
                + self.linear_trials / 64 + 4 * self.candidates + self.reloads)


@dataclass(frozen=True)
class Margins:
    target: np.ndarray
    preservation: np.ndarray
    target_jacobian: np.ndarray
    preservation_jacobian: np.ndarray
    mixed: np.ndarray

    def __post_init__(self):
        for array in (self.target, self.preservation, self.target_jacobian,
                      self.preservation_jacobian, self.mixed):
            if not np.isfinite(array).all():
                raise ValueError("synthetic margin arrays must be finite")
            array.setflags(write=False)


@dataclass(frozen=True)
class Task:
    id: str
    mechanism: str
    radius: float
    parent_checkpoint: bytes
    practice: Margins
    hidden: Margins
    measurement_valid: bool
    observation_agreement: float
    diagnosed_practice: Margins | None = None
    twin_id: str | None = None


@dataclass(frozen=True)
class Outcome:
    improved: bool
    harmful: bool
    accepted: bool
    practice_pass: bool
    target_gain: int
    preservation_regressions: int
    cost: Cost
    checkpoint_sha256: str | None
    update_norm: float
    reason: str
    target_regressions: int = 0


def _base_sample(rng, mechanism):
    """Construct signed target and preservation margin models at the parent."""
    risk = rng.uniform(0.18, 0.56)
    margins = rng.uniform(-0.60, -0.25, size=4)
    margins[-1] = rng.uniform(0.16, 0.32)
    target_direction = np.array([rng.uniform(0.65, 1.25), rng.uniform(-0.16, 0.36)])
    preserve_direction = rng.normal(0, 0.22, size=2)
    mixed = rng.normal(0, 0.08, size=4)
    variation = rng.uniform(0.03, 0.28)
    if mechanism == "preservation_constrained":
        target_direction = np.array([rng.uniform(0.85, 1.35), rng.uniform(0.45, 0.9)])
        preserve_direction = np.array([-rng.uniform(0.75, 1.3), rng.uniform(-0.15, 0.15)])
        risk = rng.uniform(0.10, 0.30)
    elif mechanism == "dormant_interaction":
        target_direction = rng.normal(0, 0.05, size=2)
        mixed = np.repeat(rng.choice((-1, 1)) * rng.uniform(1.45, 2.7), 4) + mixed
        variation = rng.uniform(0.005, 0.04)
    elif mechanism == "unreachable":
        target_direction = rng.normal(0.065, 0.018, size=2)
        variation = rng.uniform(0.005, 0.02)
    elif mechanism == "fragile_margin":
        risk = rng.uniform(0.07, 0.29)
        preserve_direction = np.array([-rng.uniform(0.24, 0.59), rng.uniform(-0.16, 0.06)])
        variation = rng.uniform(0.04, 0.4)
    target_jacobian = target_direction + rng.normal(0, variation, size=(4, 2))
    preservation_jacobian = preserve_direction + rng.normal(0, 0.08, size=(3, 2))
    # Vary geometry across tasks while keeping the designated mixed term intact.
    orientation = rng.choice((-1, 1), size=2)
    return Margins(margins, risk * rng.uniform(0.85, 1.15, size=3),
                   target_jacobian * orientation, preservation_jacobian * orientation,
                   mixed * np.prod(orientation))


def _hidden_sample(rng, practice, mechanism):
    target = practice.target + rng.normal(0, 0.075, size=4)
    target[-1] = max(0.03, target[-1])
    preservation = np.maximum(0.035, practice.preservation + rng.normal(0, 0.055, size=3))
    target_jacobian = practice.target_jacobian + rng.normal(0, 0.12, size=(4, 2))
    preservation_jacobian = practice.preservation_jacobian + rng.normal(0, 0.11, size=(3, 2))
    mixed = practice.mixed + rng.normal(0, 0.12, size=4)
    if mechanism == "fragile_margin":
        preservation *= rng.uniform(0.32, 1.1)
        preservation_jacobian *= rng.uniform(1.04, 2.3)
        target_jacobian[-1] -= rng.uniform(0.25, 1.25) * practice.target_jacobian[:3].mean(axis=0)
    return Margins(target, preservation, target_jacobian, preservation_jacobian, mixed)


def _sample(rng, identity, mechanism):
    observed = _base_sample(rng, mechanism)
    valid = mechanism not in ("measurement_artifact", "expensive_diagnostic")
    agreement = float(rng.uniform(0.61, 1.0) if valid else rng.uniform(0.20, 0.79))
    diagnosed = None
    if not valid:
        true_errors = mechanism == "expensive_diagnostic" or rng.random() < 0.18 + 0.48 * agreement
        diagnosed = observed if true_errors else replace(observed, target=np.abs(observed.target))
    hidden = _hidden_sample(rng, observed if valid else diagnosed, mechanism)
    checkpoint = rng.uniform(-0.5, 0.5, size=2).astype(_PRECISION).tobytes()
    return Task(identity, mechanism, float(rng.uniform(0.78, 1.0)), checkpoint, observed,
                hidden, valid, agreement, diagnosed)


def _twins(rng, identity):
    direction = _DIRECTIONS[int(rng.integers(len(_DIRECTIONS)))]
    target = rng.uniform(-0.37, -0.25, size=4)
    target[-1] = 0.06
    jacobian = np.repeat(direction[None, :], 4, axis=0) * rng.uniform(1.12, 1.45)
    practice = Margins(target, rng.uniform(0.37, 0.52, size=3), jacobian,
                       rng.normal(0, 0.05, size=(3, 2)), np.zeros(4))
    reversed_hidden = replace(practice, target_jacobian=-practice.target_jacobian)
    checkpoint = rng.uniform(-0.5, 0.5, size=2).astype(_PRECISION).tobytes()
    common = dict(mechanism="sensitivity_reversal", radius=float(rng.uniform(0.80, 0.98)),
                  parent_checkpoint=checkpoint, practice=practice, measurement_valid=True,
                  observation_agreement=float(rng.uniform(0.74, 1.0)), twin_id=identity)
    # IDs contain no evidence about which member has the reversed hidden slope.
    hidden = (practice, reversed_hidden)
    if rng.integers(2):
        hidden = hidden[::-1]
    return tuple(Task(id=f"{identity}/{index}", hidden=margins, **common)
                 for index, margins in enumerate(hidden))


def make_tasks(split: str, *, seed: int, per_mechanism: int) -> tuple[Task, ...]:
    if split not in SPLITS or type(seed) is not int or type(per_mechanism) is not int or per_mechanism < 2:
        raise ValueError("invalid synthetic population request")
    if split == "shift" and per_mechanism % 2:
        raise ValueError("shift requires complete pairs")
    tasks = []
    for mechanism in HELD if split == "shift" else FAMILIAR:
        # Family streams are stable even when another population grows.
        digest = sha256(f"phase-iiie-v1/{seed}/{split}/{mechanism}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        count = per_mechanism // 2 if mechanism == "sensitivity_reversal" else per_mechanism
        for index in range(count):
            identity = f"{split}/{mechanism}/{index:04d}"
            if mechanism == "sensitivity_reversal":
                tasks.extend(_twins(rng, identity))
            else:
                tasks.append(_sample(rng, identity, mechanism))
    return tuple(tasks)


def _project(margins: Margins, deltas, *, quadratic: bool):
    target = margins.target + deltas @ margins.target_jacobian.T
    if quadratic:
        target += (deltas[..., 0] * deltas[..., 1])[..., None] * margins.mixed
    preservation = margins.preservation + deltas @ margins.preservation_jacobian.T
    return target, preservation


def _safe_grid(margins, radius, *, quadratic):
    deltas = _GRID * radius
    target, preservation = _project(margins, deltas, quadratic=quadratic)
    correct = margins.target > 0
    safe = ((target[:, correct] > 0).all(axis=1)
            & (preservation >= _PRACTICE_FLOOR).all(axis=1))
    corrections = (target[:, ~correct] > 0).sum(axis=1)
    return deltas[safe], target[safe], corrections[safe]


def feature_cost(task: Task, *, level: str) -> Cost:
    if level not in FEATURE_LEVELS:
        raise ValueError("unknown feature level")
    cost = Cost(diagnostics=1, forwards=1)
    if level in ("B", "C", "D"):
        cost += Cost(diagnostics=1, forwards=1)
    if level in ("C", "D"):
        cost += Cost(diagnostics=1, backwards=2, linear_trials=len(_GRID))
    if level == "D":
        cost += Cost(diagnostics=1, curvature=1)
    return cost


def features(task: Task, level: str = "A") -> tuple[dict[str, float], Cost]:
    cost = feature_cost(task, level=level)
    margins = task.practice
    target, _ = _project(margins, np.zeros(2), quadratic=True)
    features = {
        "measurement_valid": float(task.measurement_valid),
        "persistent_deficiency": float(task.measurement_valid and np.count_nonzero(target <= 0) >= 2),
        "target_error_rate": float(np.mean(target <= 0)),
        "target_margin": float(np.mean(target)),
        "observation_agreement": task.observation_agreement,
    }
    if level in ("B", "C", "D"):
        _, preservation = _project(margins, np.zeros(2), quadratic=True)
        features["preservation_margin"] = float(preservation.min())
        features["preservation_risk"] = float(1.0 - min(1.0, preservation.min()))
    if level in ("C", "D"):
        _, targets, corrections = _safe_grid(margins, task.radius, quadratic=False)
        features["locally_reachable"] = float(bool(np.any(corrections)))
        deficient = margins.target <= 0
        residual = np.maximum(0, -targets[:, deficient]).sum(axis=1)
        features["reachability_residual"] = float(min(residual, default=np.maximum(0, -margins.target).sum()))
        jacobian = margins.target_jacobian[deficient]
        features["sensitivity_dispersion"] = (float(np.linalg.norm(jacobian - jacobian.mean(axis=0), axis=1).mean())
                                               if len(jacobian) else 0.0)
    if level == "D":
        features["interaction_signal"] = float(np.max(np.abs(margins.mixed)) * task.radius**2 / 2)
    return features, cost


def diagnosis_cost(task: Task) -> Cost:
    if task.measurement_valid:
        return Cost()
    if task.mechanism == "expensive_diagnostic":
        return Cost(diagnostics=1, forwards=96)
    return Cost(diagnostics=1, forwards=4)


def diagnose(task: Task) -> tuple[Task, Cost]:
    """Execute the declared secondary practice measurement."""
    if task.measurement_valid:
        return task, Cost()
    cost = diagnosis_cost(task)
    reference = task.diagnosed_practice
    # Independent practice views at small deterministic offsets check whether
    # the alternate measurement identifies the same wrong target rows.
    corrections = []
    for index in range(cost.forwards):
        offset = _DIRECTIONS[index % len(_DIRECTIONS)] * (0.001 * (1 + index // len(_DIRECTIONS)))
        target, _ = _project(reference, offset, quadratic=True)
        corrections.append(np.array_equal(target > 0, reference.target > 0))
    return replace(task, practice=task.diagnosed_practice, measurement_valid=True,
                   observation_agreement=float(np.mean(corrections)), diagnosed_practice=None), cost


def method_cost_bound(task: Task, method: str) -> Cost:
    if method not in METHODS:
        raise ValueError("unknown synthetic solver")
    linear = len(_GRID) if method != "first_order" else 0
    curvature = int(method == "interaction")
    return Cost(forwards=len(_SCALES) + 2, backwards=1, curvature=curvature,
                linear_trials=linear, candidates=len(_SCALES), reloads=len(_SCALES) + 2)


def _direction(margins, radius, method):
    if method == "first_order":
        gradient = margins.target_jacobian[margins.target <= 0].mean(axis=0)
        norm = np.linalg.norm(gradient)
        return gradient * (radius / norm) if norm > 1e-9 else None
    deltas, target, corrections = _safe_grid(margins, radius, quadratic=method == "interaction")
    if not np.any(corrections):
        return None
    raw_score = target[:, margins.target <= 0].mean(axis=1)
    return deltas[np.lexsort((raw_score, corrections))[-1]]


def _weight_array(payload):
    return np.frombuffer(payload, dtype=_PRECISION).astype(np.float64)


def _behavior(before, after, preserve_before, preserve_after, *, floor=0.0):
    gain = int(np.count_nonzero((before <= 0) & (after > 0)))
    target_regressions = int(np.count_nonzero((before > 0) & (after <= 0)))
    preserve_regressions = int(np.count_nonzero((preserve_before > 0) & (preserve_after <= floor)))
    return gain, target_regressions, preserve_regressions


def run_method(task: Task, method: str, *, radius_scale: float = 1.0) -> Outcome:
    """Select on rounded practice candidates, then reload one for independent scoring."""
    bound = method_cost_bound(task, method)
    if not isfinite(radius_scale) or not 0 < radius_scale <= 1:
        raise ValueError("radius scale must be in (0, 1]")
    if not task.measurement_valid or np.count_nonzero(task.practice.target <= 0) < 2:
        return Outcome(False, False, False, False, 0, 0, Cost(), None, 0.0,
                       "MEASUREMENT_REQUIRED" if not task.measurement_valid else "NO_HEADROOM")
    radius = radius_scale * task.radius
    base = _weight_array(task.parent_checkpoint)
    margins = task.practice
    # Reserve space for the worst rounding displacement of these two FP16
    # coordinates. The stored edit is still checked against the full radius.
    proposal = _direction(margins, max(0.0, radius - 0.001), method)
    cost = Cost(backwards=bound.backwards, curvature=bound.curvature, linear_trials=bound.linear_trials)
    if proposal is None:
        return Outcome(False, False, False, False, 0, 0, cost, None, 0.0, "NO_DIRECTION")
    best = None
    seen = {task.parent_checkpoint}
    for scale in _SCALES:
        checkpoint = np.asarray(base + proposal * scale, dtype=_PRECISION).tobytes()
        cost += Cost(candidates=1)
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        delta = _weight_array(checkpoint) - base
        norm = float(np.linalg.norm(delta))
        cost += Cost(reloads=1)
        if norm > radius:
            continue
        target, preservation = _project(margins, delta, quadratic=True)
        cost += Cost(forwards=1)
        gain, target_loss, preservation_loss = _behavior(
            margins.target, target, margins.preservation, preservation, floor=_PRACTICE_FLOOR)
        if gain and not (target_loss or preservation_loss):
            score = (gain, -norm, float(target.mean()))
            if best is None or score > best[0]:
                best = (score, checkpoint, sha256(checkpoint).hexdigest())
    if best is None:
        return Outcome(False, False, False, False, 0, 0, cost, None, 0.0, "PRACTICE_REJECTED")
    _, stored, expected_hash = best
    if sha256(stored).hexdigest() != expected_hash:
        raise ValueError("stored candidate digest changed before reload")
    # The hidden evaluator sees only the stored parent/candidate and the hidden margin model.
    delta = _weight_array(stored) - _weight_array(task.parent_checkpoint)
    before, preservation_before = _project(task.hidden, np.zeros(2), quadratic=True)
    after, preservation_after = _project(task.hidden, delta, quadratic=True)
    cost += Cost(reloads=2, forwards=2)
    gain, target_loss, preserve_loss = _behavior(before, after, preservation_before, preservation_after)
    harm = bool(target_loss or preserve_loss)
    improved = gain > 0 and not harm
    reason = "VERIFIED_IMPROVEMENT" if improved else "HIDDEN_REGRESSION" if harm else "HIDDEN_NO_GAIN"
    return Outcome(improved, harm, improved, True, gain - target_loss, preserve_loss, cost,
                   expected_hash, float(np.linalg.norm(delta)), reason, target_loss)
