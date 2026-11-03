"""Host-bounded MCP tools for practice-only synthetic weight diagnosis.

The host supplies one immutable base checkpoint, one practice-only dataset, an
output directory, and the patch policy. Tool callers never supply filesystem
paths, evaluator data, policy limits, or candidate output paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
import uuid

import numpy as np

from ..core import EditRejected, _check_policy, apply_patch, file_hash, json_dump_atomic
from ..diagnostics.toy_mlp import ToyMLP


class AgentSessionError(ValueError):
    """A host session or bounded agent-tool request is invalid."""


_CONFIG_KEYS = {
    "schema", "session_id", "base_checkpoint", "practice_dataset",
    "candidate_directory", "policy", "max_candidates", "max_failure_examples",
}
_PATCH_KEYS = {"schema", "base_sha256", "reason", "evidence_ids", "changes", "projections"}
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()


class ToyMCPToolSession:
    """One persisted MCP tool session with a host-owned practice evaluator."""

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
            raise AgentSessionError("Host config has missing or unknown fields")
        if config.get("schema") != "weightbridge.agent_session.v1":
            raise AgentSessionError("Unsupported host config schema")
        session_id = config.get("session_id")
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            raise AgentSessionError("Invalid session_id")
        self.session_id = session_id

        base_path = Path(config["base_checkpoint"])
        practice_path = Path(config["practice_dataset"])
        if not base_path.is_absolute() or not practice_path.is_absolute():
            raise AgentSessionError("Host paths must be absolute")
        self.base_path = base_path.resolve(strict=True)
        self.practice_path = practice_path.resolve(strict=True)
        if self.base_path.suffix != ".npz" or self.practice_path.suffix != ".npz":
            raise AgentSessionError("The MCP pilot currently accepts .npz toy tasks only")

        candidate_root = Path(config["candidate_directory"])
        if not candidate_root.is_absolute():
            raise AgentSessionError("candidate_directory must be absolute")
        candidate_root.mkdir(parents=True, exist_ok=True)
        self.candidate_root = candidate_root.resolve(strict=True)
        if self.base_path.is_relative_to(self.candidate_root) or self.practice_path.is_relative_to(self.candidate_root):
            raise AgentSessionError("Base and practice inputs must be outside the candidate directory")
        self.session_dir = self.candidate_root / "sessions" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.event_path = self.session_dir / "events.jsonl"
        self.base_sha256 = file_hash(self.base_path)
        self.practice_sha256 = file_hash(self.practice_path)

        self.policy = config["policy"]
        try:
            allow, max_entries, max_fraction, max_abs_delta = _check_policy(self.policy)
        except (TypeError, ValueError) as exc:
            raise AgentSessionError(f"Invalid host patch policy: {exc}") from exc
        if not allow <= set(ToyMLP.tensor_names):
            raise AgentSessionError("Host allowlist contains a tensor outside the toy model")
        if max_entries > 128 or max_fraction > 0.10 or max_abs_delta > 10:
            raise AgentSessionError("Host policy exceeds the MCP pilot hard limits")
        # Round-trip through JSON so callers cannot mutate host policy by reference.
        self.policy = json.loads(json.dumps(self.policy, allow_nan=False))

        max_candidates = config["max_candidates"]
        max_failure_examples = config["max_failure_examples"]
        if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
            raise AgentSessionError("max_candidates must be in [1, 32]")
        if type(max_failure_examples) is not int or not 1 <= max_failure_examples <= 128:
            raise AgentSessionError("max_failure_examples must be in [1, 128]")
        self.max_candidates = max_candidates
        self.max_failure_examples = max_failure_examples

        self.model = ToyMLP.load(self.base_path)
        with np.load(self.practice_path, allow_pickle=False) as archive:
            if set(archive.files) != {"features", "labels"}:
                raise AgentSessionError("Practice archive must contain only features and labels")
            self.features = np.array(archive["features"], dtype=np.float64, copy=True)
            self.labels = np.array(archive["labels"], dtype=np.int64, copy=True)
        if (self.features.ndim != 2 or self.features.shape[1] != 2
                or self.labels.shape != (len(self.features),)
                or not 1 <= len(self.features) <= 10000
                or not np.isfinite(self.features).all()
                or not np.isin(self.labels, [0, 1]).all()):
            raise AgentSessionError("Practice data has invalid dimensions, labels, or values")
        self._assert_inputs_unchanged()

        self._lock = threading.RLock()
        self.base_score = self._score(self.base_path)
        self.events = self._read_events()
        self.candidates: dict[str, dict[str, Any]] = {}
        self.finalized: dict[str, Any] | None = None
        self.stopped: dict[str, Any] | None = None
        for event in self.events:
            if event["type"] == "candidate_submitted":
                item = event["candidate"]
                candidate_id = item.get("candidate_id")
                if (not isinstance(candidate_id, str) or not re.fullmatch(r"[0-9a-f]{32}", candidate_id)
                        or item.get("file_name") != f"candidate-{candidate_id}.npz"
                        or item.get("patch_file") != f"patch-{candidate_id}.json"):
                    raise AgentSessionError("Invalid candidate identity in session ledger")
                self.candidates[item["candidate_id"]] = item
            elif event["type"] == "practice_evaluated":
                self.candidates[event["candidate_id"]]["practice_result"] = event["result"]
            elif event["type"] == "candidate_finalized":
                self.finalized = event
            elif event["type"] == "session_stopped":
                self.stopped = event

        if not self.events:
            self._append_event("session_started", {
                "base_sha256": self.base_sha256,
                "practice_sha256": self.practice_sha256,
                "policy_sha256": _canonical_hash(self.policy),
                "max_candidates": self.max_candidates,
            })
        else:
            start = self.events[0]
            expected = {
                "base_sha256": self.base_sha256,
                "practice_sha256": self.practice_sha256,
                "policy_sha256": _canonical_hash(self.policy),
                "max_candidates": self.max_candidates,
            }
            if start.get("type") != "session_started" or any(
                    start.get(key) != value for key, value in expected.items()):
                raise AgentSessionError("Resumed session does not match its original base, practice set, or host policy")

    def _assert_inputs_unchanged(self) -> None:
        if file_hash(self.base_path) != self.base_sha256:
            raise AgentSessionError("Host base checkpoint changed during the MCP session")
        if file_hash(self.practice_path) != self.practice_sha256:
            raise AgentSessionError("Host practice dataset changed during the MCP session")

    @classmethod
    def from_file(cls, path: str | Path) -> "ToyMCPToolSession":
        config_path = Path(path).resolve(strict=True)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentSessionError(f"Cannot read host session config: {exc}") from exc
        return cls(config)

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.event_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.event_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise AgentSessionError(f"Blank audit event at line {line_number}")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AgentSessionError(f"Invalid audit event at line {line_number}") from exc
                if (not isinstance(event, dict) or event.get("schema") != "weightbridge.agent_event.v1"
                        or event.get("session_id") != self.session_id
                        or event.get("sequence") != line_number):
                    raise AgentSessionError(f"Invalid audit event identity at line {line_number}")
                events.append(event)
        return events

    def _append_event(self, event_type: str, fields: dict[str, Any]) -> dict[str, Any]:
        event = {"schema": "weightbridge.agent_event.v1", "session_id": self.session_id,
                 "sequence": len(self.events) + 1, "timestamp_utc": _utc_now(),
                 "type": event_type, **fields}
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with self.event_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    def _score(self, checkpoint: Path) -> dict[str, Any]:
        candidate = ToyMLP.load(checkpoint)
        logits = candidate.forward(self.features)
        predictions = logits.argmax(axis=1)
        correct = predictions == self.labels
        base_logits = self.model.forward(self.features)
        base_correct = base_logits.argmax(axis=1) == self.labels
        regressions = base_correct & ~correct
        rescues = ~base_correct & correct
        baseline_correct_count = int(base_correct.sum())
        return {
            "n": int(len(self.labels)), "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "class_errors": {str(label): int(np.count_nonzero((self.labels == label) & ~correct))
                             for label in (0, 1)},
            "rescues": int(rescues.sum()), "regressions": int(regressions.sum()),
            "baseline_correct_n": baseline_correct_count,
            "regression_rate": (float(regressions.sum() / baseline_correct_count)
                                if baseline_correct_count else 0.0),
        }

    def _require_open(self) -> None:
        if self.finalized is not None or self.stopped is not None:
            raise AgentSessionError("This agent session is already finalized")

    def get_diagnostic_report(self) -> dict[str, Any]:
        with self._lock:
            if self.finalized is not None:
                raise AgentSessionError("Session is finalized; no further diagnostic reads")
            self._assert_inputs_unchanged()
            logits, activations = self.model.forward(self.features, return_activations=True)
            predictions = logits.argmax(axis=1)
            incorrect = np.flatnonzero(predictions != self.labels)
            examples = []
            for index in incorrect[:self.max_failure_examples]:
                examples.append({
                    "practice_index": int(index),
                    "input": self.features[index].tolist(),
                    "expected_class": int(self.labels[index]),
                    "predicted_class": int(predictions[index]),
                    "logits": logits[index].tolist(),
                    "hidden_activation": activations["hidden"][index].tolist(),
                })
            self._append_event("diagnostic_report_read", {"failure_examples_returned": len(examples)})
            return {
                "schema": "weightbridge.diagnostic_report.v1",
                "session_id": self.session_id,
                "base_sha256": self.base_sha256,
                "total_parameters": sum(value.size for value in self.model.tensors.values()),
                "allowed_tensors": list(self.policy["allow_tensors"]),
                "host_policy": {key: self.policy[key] for key in
                                ("max_entries", "max_fraction", "max_abs_delta")},
                "tensor_summary": self.model.inspect(),
                "practice_baseline": self.base_score,
                "practice_failures": {"total": int(len(incorrect)),
                                       "class_errors": self.base_score["class_errors"],
                                       "examples": examples},
                "activation_summary": {
                    "hidden_rms": float(np.sqrt(np.mean(activations["hidden"] ** 2))),
                    "logit_rms": float(np.sqrt(np.mean(logits ** 2))),
                },
                "hidden_evaluation_accessible": False,
                "candidate_budget_remaining": self.max_candidates - len(self.candidates),
            }

    def submit_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            self._assert_inputs_unchanged()
            if not isinstance(patch, dict) or set(patch) - _PATCH_KEYS:
                raise AgentSessionError("Patch must be a bounded weightbridge.patch.v1 object")
            if patch.get("schema") != "weightbridge.patch.v1":
                raise AgentSessionError("Invalid patch schema")
            encoded = json.dumps(patch, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if len(encoded.encode("utf-8")) > 32768:
                raise AgentSessionError("Patch JSON exceeds the 32 KiB tool limit")
            reason = patch.get("reason", "")
            evidence = patch.get("evidence_ids", [])
            if not isinstance(reason, str) or len(reason) > 2048:
                raise AgentSessionError("Patch reason must be at most 2048 characters")
            if (not isinstance(evidence, list) or len(evidence) > 32
                    or any(not isinstance(item, str) or len(item) > 128 for item in evidence)):
                raise AgentSessionError("Patch evidence_ids must contain at most 32 short strings")
            if len(self.candidates) >= self.max_candidates:
                raise AgentSessionError("Host candidate budget is exhausted")

            candidate_id = uuid.uuid4().hex
            candidate_path = self.session_dir / f"candidate-{candidate_id}.npz"
            receipt = apply_patch(self.base_path, patch, self.policy, candidate_path)
            patch_path = self.session_dir / f"patch-{candidate_id}.json"
            try:
                json_dump_atomic(patch_path, patch)
            except Exception:
                # The immutable candidate is retained as an orphan for audit.
                raise
            candidate = {
                "candidate_id": candidate_id,
                "file_name": candidate_path.name,
                "patch_file": patch_path.name,
                "patch": patch,
                "receipt": {key: value for key, value in receipt.items() if key != "candidate_path"},
                "practice_result": None,
            }
            self._append_event("candidate_submitted", {"candidate": candidate})
            self.candidates[candidate_id] = candidate
            return {"candidate_id": candidate_id,
                    "base_sha256": receipt["base_sha256"],
                    "candidate_sha256": receipt["candidate_sha256"],
                    "patch_sha256": receipt["patch_sha256"],
                    "changed_entries": receipt["changed_entries"],
                    "changed_fraction": receipt["changed_fraction"],
                    "max_abs_delta": receipt["max_abs_delta"],
                    "status": receipt["status"],
                    "evaluation_status": "not_yet_evaluated"}

    def evaluate_practice(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            self._assert_inputs_unchanged()
            if candidate_id == "base":
                self._append_event("base_practice_evaluated", {"result": self.base_score})
                return {"candidate_id": "base", "candidate_sha256": file_hash(self.base_path),
                        "practice_only": True, "result": self.base_score}
            candidate = self.candidates.get(candidate_id)
            if candidate is None:
                raise AgentSessionError("Unknown candidate_id")
            candidate_path = self.session_dir / candidate["file_name"]
            if file_hash(candidate_path) != candidate["receipt"]["candidate_sha256"]:
                raise AgentSessionError("Candidate checkpoint hash changed after submission")
            if candidate["practice_result"] is not None:
                return {"candidate_id": candidate_id, "candidate_sha256": candidate["receipt"]["candidate_sha256"],
                        "practice_only": True, "cached": True,
                        "result": candidate["practice_result"]}
            evaluated_count = sum(item["practice_result"] is not None for item in self.candidates.values())
            if evaluated_count >= self.max_candidates:
                raise AgentSessionError("Host practice-evaluation budget is exhausted")
            result = self._score(candidate_path)
            self._append_event("practice_evaluated", {"candidate_id": candidate_id, "result": result})
            candidate["practice_result"] = result
            return {"candidate_id": candidate_id,
                    "candidate_sha256": candidate["receipt"]["candidate_sha256"],
                    "practice_only": True, "cached": False, "result": result}

    def finalize_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            self._assert_inputs_unchanged()
            if self.finalized is not None:
                if self.finalized["candidate_id"] == candidate_id:
                    return {"session_id": self.session_id, "finalized": True,
                            "candidate_id": candidate_id,
                            "candidate_sha256": self.finalized["candidate_sha256"],
                            "hidden_evaluation_performed": False, "idempotent": True}
                raise AgentSessionError("A different candidate is already finalized")
            if self.stopped is not None:
                raise AgentSessionError("This session was stopped without a candidate")
            candidate = self.candidates.get(candidate_id)
            if candidate is None or candidate["practice_result"] is None:
                raise AgentSessionError("Candidate must exist and pass practice evaluation before finalization")
            candidate_path = self.session_dir / candidate["file_name"]
            if file_hash(candidate_path) != candidate["receipt"]["candidate_sha256"]:
                raise AgentSessionError("Candidate checkpoint hash changed after practice evaluation")
            event = self._append_event("candidate_finalized", {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate["receipt"]["candidate_sha256"],
                "practice_result": candidate["practice_result"],
                "hidden_evaluation_performed": False,
            })
            self.finalized = event
            return {"session_id": self.session_id, "finalized": True,
                    "candidate_id": candidate_id,
                    "candidate_sha256": event["candidate_sha256"],
                    "practice_result": candidate["practice_result"],
                    "hidden_evaluation_performed": False}

    def stop_without_candidate(self, reason: str) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 2048:
                raise AgentSessionError("A short stop reason is required")
            event = self._append_event("session_stopped", {"reason": reason.strip()})
            self.stopped = event
            return {"session_id": self.session_id, "stopped": True,
                    "reason": event["reason"], "hidden_evaluation_performed": False}

    def get_session_status(self) -> dict[str, Any]:
        with self._lock:
            self._assert_inputs_unchanged()
            return {"session_id": self.session_id,
                    "candidate_count": len(self.candidates),
                    "candidate_budget": self.max_candidates,
                    "evaluated_count": sum(item["practice_result"] is not None
                                            for item in self.candidates.values()),
                    "finalized": self.finalized is not None,
                    "stopped": self.stopped is not None,
                    "base_sha256": self.base_sha256,
                    "hidden_evaluation_accessible": False}


def create_mcp_server(session: ToyMCPToolSession):
    """Create an MCP stdio server. Import the optional SDK only when requested."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install WeightBridge with `pip install -e '.[mcp]'` to use MCP tools") from exc

    server = FastMCP("WeightBridge")

    @server.tool()
    def get_diagnostic_report() -> dict[str, Any]:
        """Inspect the host-selected toy checkpoint and practice-only failures."""
        return session.get_diagnostic_report()

    @server.tool()
    def submit_patch(patch: dict[str, Any]) -> dict[str, Any]:
        """Submit a proposed sparse patch under the host-owned policy."""
        return session.submit_patch(patch)

    @server.tool()
    def evaluate_practice(candidate_id: str) -> dict[str, Any]:
        """Score a candidate on the authorized practice set only."""
        return session.evaluate_practice(candidate_id)

    @server.tool()
    def finalize_candidate(candidate_id: str) -> dict[str, Any]:
        """Freeze one practice-evaluated candidate; this does not run hidden evaluation."""
        return session.finalize_candidate(candidate_id)

    @server.tool()
    def stop_without_candidate(reason: str) -> dict[str, Any]:
        """End the session when no reliable candidate is found."""
        return session.stop_without_candidate(reason)

    @server.tool()
    def get_session_status() -> dict[str, Any]:
        """Read candidate and practice-evaluation budget usage."""
        return session.get_session_status()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Host-bounded WeightBridge MCP tool server")
    parser.add_argument("--config", required=True, help="Host-owned JSON session configuration")
    args = parser.parse_args()
    session = ToyMCPToolSession.from_file(args.config)
    create_mcp_server(session).run(transport="stdio")


if __name__ == "__main__":
    main()
