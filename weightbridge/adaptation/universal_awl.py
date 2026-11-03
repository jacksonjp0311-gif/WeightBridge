"""Model-independent AWL v3 intent and host-bound execution planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Literal

from .meta_controller import Action, AdmissionDecision, EvidenceState, MethodSelector
from .model_adapter import Capability, EditableRegion, ModelCapabilityContract


SCHEMA = "weightbridge.awl.v3"
Method = Literal["first_order", "constrained", "gradient", "direct_low_rank"]
_METHODS = ("first_order", "constrained", "gradient", "direct_low_rank")
_METRICS = ("exact_answer", "structured_output", "conditional_score", "forced_choice", "programmatic")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_WEIGHT_CAPABILITIES = frozenset({
    Capability.READ_WEIGHTS, Capability.INSPECT_TENSORS, Capability.MODIFY_WEIGHTS,
    Capability.EVALUATE_BEHAVIOR, Capability.SAVE_CHECKPOINT, Capability.RELOAD_CHECKPOINT,
    Capability.RESTORE_ORIGINAL,
})


def _fields(document: object, required: set[str], name: str) -> dict:
    if not isinstance(document, dict) or document.keys() != required:
        raise ValueError(f"AWL {name} requires exactly {', '.join(sorted(required))}")
    return document


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"AWL {name} must be a symbolic identifier")
    return value


def _integer(value: object, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid AWL {name}")
    return value


def _decimal(value: object, minimum: float, maximum: float, name: str) -> float:
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid AWL {name}")
    return float(value)


def _unique(value: object, name: str, choices=None, *, allow_empty=False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not (0 if allow_empty else 1) <= len(value) <= 32:
        raise ValueError(f"Invalid AWL {name} list")
    items = tuple(_identifier(item, name) for item in value)
    if len(set(items)) != len(items) or choices is not None and not set(items).issubset(choices):
        raise ValueError(f"AWL {name} contains duplicate or unsupported identifiers")
    return items


@dataclass(frozen=True)
class ResourceLimits:
    max_forward_calls: int
    max_backward_calls: int
    max_memory_bytes: int
    max_update_entries: int
    max_update_norm: float
    max_rank: int
    max_candidates: int

    def __post_init__(self):
        _integer(self.max_forward_calls, 1, 1_000_000, "max_forward_calls")
        _integer(self.max_backward_calls, 0, 1_000_000, "max_backward_calls")
        _integer(self.max_memory_bytes, 1, 2**50, "max_memory_bytes")
        _integer(self.max_update_entries, 1, 10_000_000, "max_update_entries")
        _decimal(self.max_update_norm, 1e-12, 1e6, "max_update_norm")
        _integer(self.max_rank, 1, 64, "max_rank")
        _integer(self.max_candidates, 1, 32, "max_candidates")

    def within(self, maximum: ResourceLimits) -> bool:
        return all(value <= getattr(maximum, key) for key, value in asdict(self).items())


@dataclass(frozen=True)
class BehaviorObjective:
    family: str
    metric: str
    min_gain: int


@dataclass(frozen=True)
class RegressionRisk:
    max_regressions: int
    max_harm_probability: float

    def __post_init__(self):
        _integer(self.max_regressions, 0, 10_000, "max_regressions")
        _decimal(self.max_harm_probability, 0., 1., "max_harm_probability")


@dataclass(frozen=True)
class Verification:
    fresh_evaluation: bool
    checkpoint_reload: bool
    compare_baselines: bool
    persistent: bool


@dataclass(frozen=True)
class AdaptationIntent:
    objective: BehaviorObjective
    preserve: tuple[str, ...]
    risk: RegressionRisk
    limits: ResourceLimits
    practice_data_id: str
    evaluation_data_id: str
    methods: tuple[Method, ...]
    verify: Verification

    def __post_init__(self):
        _identifier(self.objective.family, "objective family")
        if self.objective.metric not in _METRICS:
            raise ValueError("Unsupported AWL objective metric")
        _integer(self.objective.min_gain, 1, 10_000, "minimum gain")
        object.__setattr__(self, "preserve", _unique(self.preserve, "preservation family", allow_empty=True))
        if self.objective.family in self.preserve:
            raise ValueError("AWL target and preservation families must be distinct")
        _identifier(self.practice_data_id, "practice data")
        _identifier(self.evaluation_data_id, "evaluation data")
        if self.practice_data_id == self.evaluation_data_id:
            raise ValueError("AWL practice and evaluation data must be distinct")
        object.__setattr__(self, "methods", _unique(self.methods, "method", _METHODS))
        if not all(value is True for value in asdict(self.verify).values()):
            raise ValueError("AWL requires fresh evaluation, baseline comparison, checkpoint reload and persistence")


def parse_awl_v3(document: object) -> AdaptationIntent:
    """Parse an inert JSON request. IDs refer to host-owned material."""
    doc = _fields(document, {"schema", "objective", "preserve", "risk", "limits", "data", "methods", "verify"},
                  "document")
    if doc["schema"] != SCHEMA:
        raise ValueError(f"AWL document must use {SCHEMA}")
    if not isinstance(doc["preserve"], list) or not isinstance(doc["methods"], list):
        raise ValueError("AWL preservation and methods must be JSON arrays")
    objective = _fields(doc["objective"], {"family", "metric", "min_gain"}, "objective")
    risk = _fields(doc["risk"], {"max_regressions", "max_harm_probability"}, "risk")
    limits = _fields(doc["limits"], set(ResourceLimits.__dataclass_fields__), "limits")
    data = _fields(doc["data"], {"practice", "evaluation"}, "data")
    verify = _fields(doc["verify"], set(Verification.__dataclass_fields__), "verification")
    return AdaptationIntent(
        BehaviorObjective(**objective), _unique(doc["preserve"], "preservation family", allow_empty=True),
        RegressionRisk(**risk), ResourceLimits(**limits), data["practice"], data["evaluation"],
        _unique(doc["methods"], "method", _METHODS), Verification(**verify),
    )


@dataclass(frozen=True)
class HostAuthorization:
    """Trusted limits for one source checkpoint and two authorized data roles."""

    checkpoint_sha256: str
    allowed_methods: tuple[Method, ...]
    allowed_tensors: tuple[str, ...]
    authorized_families: tuple[str, ...]
    practice_data_id: str
    evaluation_data_id: str
    limits: ResourceLimits
    max_regressions: int = 0
    max_harm_probability: float = .12
    authorized_metrics: tuple[str, ...] = ("exact_answer",)

    def __post_init__(self):
        if not isinstance(self.checkpoint_sha256, str) or not _SHA256.fullmatch(self.checkpoint_sha256):
            raise ValueError("Host must pin a canonical checkpoint SHA-256")
        object.__setattr__(self, "allowed_methods", _unique(self.allowed_methods, "authorized method", _METHODS))
        object.__setattr__(self, "allowed_tensors", _unique(self.allowed_tensors, "authorized tensor", allow_empty=True))
        object.__setattr__(self, "authorized_families", _unique(self.authorized_families, "authorized family"))
        object.__setattr__(self, "authorized_metrics", _unique(self.authorized_metrics, "authorized metric", _METRICS))
        _identifier(self.practice_data_id, "host practice data")
        _identifier(self.evaluation_data_id, "host evaluation data")
        if self.practice_data_id == self.evaluation_data_id:
            raise ValueError("Host must distinguish practice and evaluation data")
        RegressionRisk(self.max_regressions, self.max_harm_probability)


@dataclass(frozen=True)
class ExecutionPlan:
    intent: AdaptationIntent
    checkpoint_sha256: str
    family: str
    architecture: str
    method: Method
    eligible_methods: tuple[Method, ...]
    region: EditableRegion
    limits: ResourceLimits
    max_columns: int

    def record(self) -> dict:
        return {"schema": "weightbridge.execution_plan.v1", **asdict(self)}


def compile_awl_v3(intent: AdaptationIntent, contract: ModelCapabilityContract, host: HostAuthorization, *,
                   method: Method | None = None) -> ExecutionPlan:
    """Translate an authorized intent into an affine slice update plan without editing."""
    missing = _WEIGHT_CAPABILITIES - contract.capabilities
    if missing:
        raise ValueError(f"Model cannot execute a persistent weight plan: {', '.join(sorted(cap.value for cap in missing))}")
    if contract.checkpoint_sha256 != host.checkpoint_sha256:
        raise ValueError("AWL source checkpoint differs from the host authorization")
    if (intent.practice_data_id != host.practice_data_id or intent.evaluation_data_id != host.evaluation_data_id
            or not {intent.objective.family, *intent.preserve}.issubset(host.authorized_families)
            or intent.objective.metric not in host.authorized_metrics):
        raise ValueError("AWL objective, preservation family or data IDs are outside the host scope")
    if (not intent.limits.within(host.limits) or intent.risk.max_regressions > host.max_regressions
            or intent.risk.max_harm_probability > host.max_harm_probability):
        raise ValueError("AWL resource or regression limits exceed host authorization")
    if contract.checkpoint_bytes is not None and contract.checkpoint_bytes > intent.limits.max_memory_bytes:
        raise ValueError("Checkpoint payload exceeds the AWL memory preflight limit")
    methods = tuple(name for name in intent.methods if name in host.allowed_methods and
                    (name == "direct_low_rank" or Capability.COMPUTE_GRADIENTS in contract.capabilities
                     and intent.limits.max_backward_calls > 0))
    selected = method if method is not None else next(iter(methods), None)
    if selected not in methods:
        raise ValueError("Requested method is outside the authorized, supported model capabilities")
    regions = [
        region for region in contract.editable_regions
        if region.role == "ffn_output" and region.tensor_name in host.allowed_tensors
        and region.shape[0] <= intent.limits.max_update_entries
    ]
    if not regions:
        raise ValueError("No host-allowed feed-forward affine region fits the update budget")
    region = max(regions, key=lambda item: (item.layer_index, item.tensor_name))
    columns = min(region.shape[1], intent.limits.max_update_entries // region.shape[0])
    return ExecutionPlan(intent, contract.checkpoint_sha256, contract.family, contract.architecture,
                         selected, methods, region, intent.limits, columns)


@dataclass(frozen=True)
class ModelEvidence:
    family: str
    checkpoint_sha256: str
    state: EvidenceState


@dataclass(frozen=True)
class ModelPredictions:
    family: str
    checkpoint_sha256: str
    probabilities: tuple[tuple[float, float], ...]


def recommend_awl_v3(plan: ExecutionPlan, evidence: ModelEvidence,
                     predictions: ModelPredictions | None = None) -> AdmissionDecision:
    """Select among supported methods with evidence bound to this model."""
    identity = (plan.family, plan.checkpoint_sha256)
    if (evidence.family, evidence.checkpoint_sha256) != identity:
        raise ValueError("AWL evidence belongs to another model")
    if predictions is not None and (predictions.family, predictions.checkpoint_sha256) != identity:
        raise ValueError("AWL risk predictions belong to another model")
    methods = tuple(name for name in evidence.state.allowed_methods if name in plan.eligible_methods)
    if not methods:
        return AdmissionDecision(Action.HOLD, "no_supported_selector_method")
    state = replace(evidence.state, allowed_methods=methods)
    return MethodSelector(harm_limit=plan.intent.risk.max_harm_probability).choose(
        state, None if predictions is None else predictions.probabilities,
    )
