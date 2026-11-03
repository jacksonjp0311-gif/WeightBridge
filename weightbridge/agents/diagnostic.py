"""A bounded, observable diagnostic search agent for the synthetic MLP.

This deterministic research agent is an algorithmic baseline, not an LLM. Its
tool surface exposes model summaries and practice-only evaluator feedback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from ..core import apply_patch, file_hash
from ..diagnostics.toy_mlp import ToyMLP


@dataclass(frozen=True)
class AgentResult:
    hypothesis: str
    weight_inspection: dict[str, Any]
    activation_summary: dict[str, Any]
    chosen_patch: dict[str, Any] | None
    candidate_trials: list[dict[str, Any]]
    practice_baseline_accuracy: float
    practice_candidate_accuracy: float
    evaluator_calls: int
    iteration_budget: int
    stop_reason: str
    holdout_accessed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiagnosticAgent:
    """Try a fixed search over one policy-authorized output-bias scalar.

    The agent is not supplied a defect value or a correct checkpoint. It gets
    an initial practice report, a small weight summary and an authorized score
    callback. Candidate patches are validated and materialized by the host.
    """

    def __init__(self, *, deltas: tuple[float, ...] = (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, -5.0),
                 iteration_budget: int = 12, memory_context: str = "",
                 memory_evidence_ids: tuple[str, ...] = ()):
        if not deltas or any(not np.isfinite(value) or abs(value) > 5 for value in deltas):
            raise ValueError("candidate deltas must be nonempty and bounded by 5")
        if type(iteration_budget) is not int or not 1 <= iteration_budget <= 32:
            raise ValueError("iteration_budget must be in [1, 32]")
        self.deltas = deltas[:iteration_budget]
        self.iteration_budget = iteration_budget
        if not isinstance(memory_context, str) or len(memory_context) > 2000:
            raise ValueError("memory_context must be bounded text")
        if any(not isinstance(value, str) or not value for value in memory_evidence_ids):
            raise ValueError("memory evidence IDs must be nonempty text")
        self.memory_context = memory_context
        self.memory_evidence_ids = memory_evidence_ids

    def run(self, *, base_path: str | Path, model: ToyMLP,
            practice_features: np.ndarray, practice_labels: np.ndarray,
            evaluate_practice: Callable[[Path], dict[str, Any]],
            candidate_dir: str | Path, allowed_tensors: list[str]) -> AgentResult:
        if allowed_tensors != ["output_bias"]:
            raise ValueError("Agent permission surface must be the fixed output_bias allowlist")
        baseline = evaluate_practice(Path(base_path))
        weight_inspection = model.inspect()
        logits, activations = model.forward(practice_features, return_activations=True)
        prediction = logits.argmax(axis=1)
        error_labels = practice_labels[prediction != practice_labels]
        class_errors = {str(label): int(np.count_nonzero(error_labels == label)) for label in (0, 1)}
        output_bias = model.tensors["output_bias"]
        hypothesis = (
            "Practice failures are one-sided and the output-layer intercept is a plausible shared cause; "
            "test bounded scalar shifts on the authorized output_bias tensor."
        )
        if self.memory_context:
            hypothesis += " Retrieved verified synthetic experience: " + self.memory_context
        candidate_dir = Path(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        trials: list[dict[str, Any]] = []
        best_patch = None
        best_accuracy = float(baseline["accuracy"])
        stop_reason = "iteration_budget_exhausted"
        for trial_index, delta in enumerate(self.deltas):
            patch = {
                "schema": "weightbridge.patch.v1",
                "base_sha256": file_hash(base_path),
                "reason": hypothesis,
                "evidence_ids": ["practice-only-baseline", "one-sided-class-errors", *self.memory_evidence_ids],
                "changes": [{"tensor": "output_bias", "index": [1], "op": "add", "value": float(delta)}],
                "projections": [],
            }
            proposed = candidate_dir / f"trial-{trial_index:02d}.npz"
            policy = {"schema": "weightbridge.policy.v1", "allow_tensors": ["output_bias"],
                      "max_entries": 1, "max_fraction": 1 / 12, "max_abs_delta": 5.0}
            receipt = apply_patch(base_path, patch, policy, proposed)
            score = evaluate_practice(proposed)
            trial = {"trial": trial_index, "proposed_delta": delta,
                     "practice_accuracy": score["accuracy"],
                     "class_errors": score["class_errors"],
                     "candidate_sha256": receipt["candidate_sha256"],
                     "patch_sha256": receipt["patch_sha256"],
                     "changed_entries": receipt["changed_entries"],
                     "max_abs_delta": receipt["max_abs_delta"]}
            trials.append(trial)
            # Improvement must be repeatable on the practice cases and exceed a
            # fixed 5-point criterion; the final holdout is never consulted.
            if score["accuracy"] >= best_accuracy + 0.05:
                best_accuracy = float(score["accuracy"])
                best_patch = patch
            elif score["accuracy"] == best_accuracy and best_patch is not None:
                prior = abs(float(best_patch["changes"][0]["value"]))
                if abs(float(delta)) < prior:
                    best_patch = patch
            if score["accuracy"] >= 1.0:
                stop_reason = "practice_target_reached"
                break
        activation_summary = {
            "hidden_shape": list(activations["hidden"].shape),
            "hidden_rms": float(np.sqrt(np.mean(activations["hidden"] ** 2))),
            "logit_mean": float(activations["logits"].mean()),
            "logit_rms": float(np.sqrt(np.mean(activations["logits"] ** 2))),
        }
        return AgentResult(
            hypothesis=hypothesis + f" Observed practice errors by true class: {class_errors}; "
                                    f"output-bias mean={weight_inspection['output_bias']['mean']:.6f}; "
                                    f"hidden activation RMS={activation_summary['hidden_rms']:.6f}.",
            weight_inspection=weight_inspection,
            activation_summary=activation_summary,
            chosen_patch=best_patch, candidate_trials=trials,
            practice_baseline_accuracy=float(baseline["accuracy"]),
            practice_candidate_accuracy=best_accuracy,
            evaluator_calls=len(trials) + 1, iteration_budget=self.iteration_budget,
            stop_reason=stop_reason,
            holdout_accessed=False,
        )
