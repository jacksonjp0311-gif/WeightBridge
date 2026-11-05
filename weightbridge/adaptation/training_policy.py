"""Host authorization for a frozen-base supervised adapter experiment.

This does not alter the direct-checkpoint patch policy or grant deployment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class TrainingAuthorization:
    schema: str
    parent_sha256: str
    allowed_modules: tuple[str, ...]
    max_trainable_parameters: int
    rank: int
    alpha: float
    dropout: float
    learning_rates: tuple[float, ...]
    optimizer: str
    max_runs: int
    max_steps_per_run: int
    max_answer_tokens_per_run: int
    max_input_tokens_per_run: int
    max_memory_bytes: int
    max_wall_seconds_per_run: float
    max_cpu_seconds_per_run: float
    max_development_checkpoints_per_run: int
    allow_merge: bool
    training_bundle_sha256: str
    smoke_gate_sha256: str | None = None
    base_frozen: bool = True
    production_deployment: bool = False

    def __post_init__(self):
        if self.schema != "weightbridge.supervised_training_authorization.v1":
            raise ValueError("Unknown training authorization version")
        if (len(self.parent_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.parent_sha256)
                or len(self.training_bundle_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.training_bundle_sha256)
                or not self.allowed_modules or len(set(self.allowed_modules)) != len(self.allowed_modules)):
            raise ValueError("Exact parent and distinct explicit modules are required")
        for name in ("max_trainable_parameters", "rank", "max_runs", "max_steps_per_run",
                     "max_answer_tokens_per_run", "max_input_tokens_per_run", "max_memory_bytes",
                     "max_development_checkpoints_per_run"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid positive training bound: {name}")
        for value in (*self.learning_rates, self.alpha, self.max_wall_seconds_per_run,
                      self.max_cpu_seconds_per_run):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Training rates and resource bounds must be finite and positive")
        if (not self.learning_rates or self.optimizer != "AdamW" or self.dropout != 0
                or not self.base_frozen or self.production_deployment
                or type(self.allow_merge) is not bool):
            raise ValueError("Authorization requires frozen-base AdamW and no deployment authority")
        if self.smoke_gate_sha256 is not None and (len(self.smoke_gate_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.smoke_gate_sha256)):
            raise ValueError("Main training smoke admission must be an exact SHA-256")
        forbidden = ("vision", "visual", "lm_head", "embed_tokens")
        if any(any(part in name.lower() for part in forbidden) for name in self.allowed_modules):
            raise ValueError("Embeddings, vision and output heads are not adapter-authorized")

    @classmethod
    def from_dict(cls, value: Mapping):
        fields = dict(value)
        fields["allowed_modules"] = tuple(fields["allowed_modules"])
        fields["learning_rates"] = tuple(fields["learning_rates"])
        return cls(**fields)

    def to_dict(self):
        return asdict(self)

    def validate_plan(self, *, parent_sha256, module_names, trainable_parameters, rank, alpha,
                      dropout, learning_rate, steps, run_index):
        if (parent_sha256 != self.parent_sha256 or tuple(module_names) != self.allowed_modules
                or trainable_parameters > self.max_trainable_parameters or trainable_parameters <= 0
                or rank != self.rank or alpha != self.alpha or dropout != self.dropout
                or learning_rate not in self.learning_rates or not 1 <= steps <= self.max_steps_per_run
                or not 0 <= run_index < self.max_runs):
            raise ValueError("Supervised training plan exceeds host authorization")

    def check_usage(self, *, steps, answer_tokens, input_tokens, wall_seconds, cpu_seconds, rss_bytes):
        observed = (steps, answer_tokens, input_tokens, wall_seconds, cpu_seconds, rss_bytes)
        limits = (self.max_steps_per_run, self.max_answer_tokens_per_run,
                  self.max_input_tokens_per_run, self.max_wall_seconds_per_run,
                  self.max_cpu_seconds_per_run, self.max_memory_bytes)
        if any(not math.isfinite(value) or value < 0 or value > bound
               for value, bound in zip(observed, limits, strict=True)):
            raise RuntimeError("Supervised training resource authorization exhausted")
