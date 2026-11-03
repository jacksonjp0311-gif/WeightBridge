"""Host-owned local practice sessions for editing HF checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re

from ..core import EditRejected, file_hash, json_dump_atomic
from ..evaluation.hf_edit import _load_suite, _score_checkpoint, evaluate_hf_edit
from ..inference import InferenceConfig
from ..inspection.safetensors import load_checkpoint
from ..patches import apply_safetensors_patch
from ..patches.huggingface import _policy
from .hf_direction import _DOWN_PROJECTION, propose_direction


_SCHEMA = "weightbridge.hf_edit_session.v1"
_CANDIDATE = re.compile(r"candidate-[0-9]{4}")
_EXAMPLE_LIMIT = 6
_PRACTICE_BYTES = 8 * 1024 * 1024
_SCORE_FIELDS = (
    "target_count", "base_target_correct", "candidate_target_correct", "target_gain",
    "target_corrections", "target_regressions", "sentinel_count", "sentinel_prediction_flips",
)


def _read(path):
    return json.loads(path.read_bytes())


@contextmanager
def _locked(session):
    root = Path(session).resolve()
    if not (root / "session.json").is_file():
        raise ValueError("HF edit session is missing its host configuration")
    # An OS lock releases if the process exits. Attempt directories keep the
    # consumed budget when a process stops before writing its result.
    with (root / ".lock").open("a+b", buffering=0) as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("HF edit session is busy") from exc
        yield root


def _state(root):
    state = _read(root / "session.json")
    if state.get("schema") != _SCHEMA:
        raise ValueError("Unsupported HF edit session configuration")
    inputs = {}
    for name in ("policy", "practice", "baseline"):
        raw = (root / f"{name}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != state[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} changed after session initialization")
        inputs[name] = json.loads(raw)
    base = load_checkpoint(state["base_path"])
    if base.checkpoint_sha256() != state["base_sha256"]:
        raise ValueError("Base checkpoint changed after session initialization")
    return state, inputs


def _decision(root):
    path = root / "decision.json"
    return _read(path) if path.exists() else None


def _require_open(root):
    if _decision(root) is not None:
        raise ValueError("HF edit session already has a decision")


def _attempts(root):
    return sorted(path for path in (root / "attempts").iterdir()
                  if path.is_dir() and _CANDIDATE.fullmatch(path.name))


def _practice_failures(practice, predictions):
    return [
        {"id": row["id"], "text": row["text"][:1024], "label": row["label"], "prediction": prediction}
        for row, prediction in zip(practice["target"], predictions, strict=True)
        if prediction != row["label"]
    ][:_EXAMPLE_LIMIT]


def _describe(root, state, inputs):
    attempts = []
    for attempt in _attempts(root):
        path = attempt / "result.json"
        result = _read(path) if path.exists() else {"candidate_id": attempt.name, "disposition": "INCOMPLETE"}
        attempts.append({key: value for key, value in result.items() if key != "target_failures"})
    decision = _decision(root)
    return {
        "schema": _SCHEMA,
        "state": decision["disposition"] if decision else "OPEN",
        "base_sha256": state["base_sha256"],
        "allowed_tensors": state["allowed_tensors"],
        "direction_tensors": state["direction_tensors"],
        "max_direction_strength": 1.0,
        "policy": inputs["policy"],
        "practice_sha256": state["practice_sha256"],
        "practice_choices": inputs["practice"]["choices"],
        "practice_gate": state["practice_gate"],
        "baseline_practice": {key: value for key, value in inputs["baseline"].items() if key != "predictions"},
        "max_candidates": state["max_candidates"],
        "used_candidates": len(attempts),
        "remaining_candidates": state["max_candidates"] - len(attempts),
        "attempts": attempts,
        "decision": decision,
    }


def init_hf_session(
    session, base, policy, practice, *,
    max_candidates=8, min_target_gain=1, max_sentinel_flips=0,
    device="cpu", dtype="auto", cpu_threads=6, batch_size=8, max_input_tokens=512,
):
    """Freeze a trusted base, policy and practice suite in a fresh directory."""
    root = Path(session).resolve()
    if root.exists() or Path(session).is_symlink():
        raise ValueError("HF edit session directory already exists")
    checkpoint = load_checkpoint(base)
    if not checkpoint.root.is_dir():
        raise ValueError("An HF edit session requires a local model directory")
    if root.is_relative_to(checkpoint.root):
        raise ValueError("HF edit session must be outside the base checkpoint")
    policy_doc = _read(Path(policy))
    allowed, _, _, _, _, _, _, _ = _policy(policy_doc)
    if len(allowed) > 32 or not allowed.issubset(checkpoint.tensors):
        raise ValueError("Session policy must list at most 32 tensors present in the base checkpoint")
    with Path(practice).open("rb") as stream:
        raw_practice = stream.read(_PRACTICE_BYTES + 1)
    if len(raw_practice) > _PRACTICE_BYTES:
        raise ValueError("Practice suite exceeds the 8 MiB input limit")
    suite = _load_suite(raw_practice)
    if set(suite) != {"schema", "choices", "target", "sentinel"}:
        raise ValueError("Session practice suite may contain only schema, choices, target and sentinel")
    for group in ("target", "sentinel"):
        suite[group] = [{key: row[key] for key in ("id", "text", "label")} for row in suite[group]]
    if len(json.dumps(suite, indent=2, sort_keys=True).encode("utf-8")) + 1 > _PRACTICE_BYTES:
        raise ValueError("Frozen practice suite exceeds the 8 MiB input limit")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 32:
        raise ValueError("max_candidates must be in [1, 32]")
    if type(min_target_gain) is not int or not 1 <= min_target_gain <= len(suite["target"]):
        raise ValueError("min_target_gain must be positive and fit within the practice target count")
    if type(max_sentinel_flips) is not int or not 0 <= max_sentinel_flips <= len(suite["sentinel"]):
        raise ValueError("max_sentinel_flips must fit within the practice sentinel count")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be in [1, 16]")
    if not isinstance(device, str) or not device.strip() or dtype not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Invalid inference device or dtype")
    InferenceConfig(cpu_threads=cpu_threads, max_input_tokens=max_input_tokens).validate(max_input_budget=8192)
    inference = {"device": device, "dtype": dtype, "cpu_threads": cpu_threads,
                 "batch_size": batch_size, "max_input_tokens": max_input_tokens}
    base_hash = checkpoint.checkpoint_sha256()
    allowed_tensors = [
        {"name": name, "tensor_sha256": checkpoint.tensor_sha256(name),
         "shape": list(checkpoint.tensors[name].shape), "dtype": checkpoint.tensors[name].dtype}
        for name in sorted(allowed)
    ]
    rows = suite["target"] + suite["sentinel"]
    predictions = _score_checkpoint(checkpoint.root, suite["choices"], rows, **inference)
    paired = list(zip(rows, predictions, strict=True))
    n_target = len(suite["target"])
    baseline = {
        "target_count": n_target,
        "target_correct": sum(row["label"] == prediction for row, prediction in paired[:n_target]),
        "sentinel_count": len(suite["sentinel"]),
        "sentinel_correct": sum(row["label"] == prediction for row, prediction in paired[n_target:]),
        "target_failures": _practice_failures(suite, predictions[:n_target]),
        "predictions": predictions,
    }
    if load_checkpoint(checkpoint.root).checkpoint_sha256() != base_hash:
        raise ValueError("Base checkpoint changed during session initialization")
    root.mkdir(parents=True)
    (root / "attempts").mkdir()
    inputs = {"policy": policy_doc, "practice": suite, "baseline": baseline}
    for name, document in inputs.items():
        json_dump_atomic(root / f"{name}.json", document)
    direction_tensors = []
    if checkpoint.config.get("model_type") == "qwen3_5" and len(suite["choices"]) == 2:
        direction_tensors = [entry["name"] for entry in allowed_tensors
                             if (_DOWN_PROJECTION.fullmatch(entry["name"])
                                 and len(entry["shape"]) == 2 and all(entry["shape"]))]
    state = {
        "schema": _SCHEMA, "base_path": str(checkpoint.root), "base_sha256": base_hash,
        "allowed_tensors": allowed_tensors,
        "direction_tensors": direction_tensors,
        "inference": inference, "max_candidates": max_candidates,
        "practice_gate": {"min_target_gain": min_target_gain, "max_sentinel_flips": max_sentinel_flips},
        **{f"{name}_sha256": file_hash(root / f"{name}.json") for name in inputs},
    }
    json_dump_atomic(root / "session.json", state)
    return _describe(root, state, inputs)


def describe_hf_session(session):
    """Return the allowed tensor identities and bounded practice feedback."""
    with _locked(session) as root:
        state, inputs = _state(root)
        return _describe(root, state, inputs)


def _try_hf_edit(session, proposal):
    with _locked(session) as root:
        _require_open(root)
        state, inputs = _state(root)
        used = len(_attempts(root))
        if used >= state["max_candidates"]:
            raise ValueError("HF edit session candidate budget is exhausted")
        attempt = root / "attempts" / f"candidate-{used + 1:04d}"
        attempt.mkdir()
        patch_path, candidate = attempt / "patch.json", attempt / "model"
        report_path = attempt / "practice_report.json"
        result = {"candidate_id": attempt.name, "remaining_candidates": state["max_candidates"] - used - 1}
        try:
            patch, details = proposal(state, inputs)
            result.update(details)
            if not isinstance(patch, dict):
                raise ValueError("Patch proposal must be a JSON object")
            if patch.get("base_sha256") != state["base_sha256"]:
                raise EditRejected("Patch must bind the session's frozen base checkpoint")
            json_dump_atomic(patch_path, patch)
            receipt = apply_safetensors_patch(state["base_path"], patch, inputs["policy"], candidate)
        except (TypeError, ValueError) as exc:
            result.update(disposition="REJECTED", reason=str(exc)[:512])
        except (OSError, RuntimeError) as exc:
            result.update(disposition="ERROR", reason=str(exc)[:512])
        else:
            result.update(candidate_path=str(candidate), candidate_sha256=receipt["candidate_sha256"],
                          patch_path=str(patch_path), patch_file_sha256=file_hash(patch_path),
                          patch_sha256=receipt["patch_sha256"], changed_entries=receipt["changed_entries"],
                          parameter_fraction_changed=receipt["parameter_fraction_changed"])
            try:
                if receipt["candidate_sha256"] == state["base_sha256"]:
                    raise EditRejected("Patch left checkpoint weights unchanged at stored precision")
                report = evaluate_hf_edit(
                    state["base_path"], candidate, root / "practice.json", report_path,
                    **state["practice_gate"], **state["inference"],
                )
                rows = report["target"] + report["sentinel"]
                if [row["base_prediction"] for row in rows] != inputs["baseline"]["predictions"]:
                    raise ValueError("Practice evaluation did not reproduce frozen base predictions")
            except EditRejected as exc:
                result.update(disposition="REJECTED", reason=str(exc)[:512])
            except (OSError, RuntimeError, ValueError) as exc:
                result.update(disposition="ERROR", reason=str(exc)[:512])
            else:
                result.update(
                    disposition=report["disposition"], practice_report_path=str(report_path),
                    practice_report_sha256=file_hash(report_path),
                    target_failures=_practice_failures(inputs["practice"],
                                                       [row["candidate_prediction"] for row in report["target"]]),
                    **{key: report[key] for key in _SCORE_FIELDS},
                )
        json_dump_atomic(attempt / "result.json", result)
        return result


def try_hf_patch(session, patch):
    """Apply and score an agent-supplied patch under the frozen host policy."""
    return _try_hf_edit(session, lambda state, inputs: (patch, {"proposal_kind": "raw_patch"}))


def try_hf_direction(session, tensor, strength):
    """Fit a host-bounded rank-one direction requested by the agent."""
    def proposal(state, inputs):
        if not isinstance(tensor, str) or tensor not in state["direction_tensors"]:
            raise EditRejected("Tensor is outside the session's direction allowlist")
        if type(strength) not in (int, float) or not 0 < strength <= 1 or not math.isfinite(strength):
            raise EditRejected("Direction strength must be finite and in (0, 1]")
        patch = propose_direction(Path(state["base_path"]), inputs["practice"], tensor, strength, state["inference"])
        return patch, {"proposal_kind": "direction", "tensor": tensor, "strength": strength}

    return _try_hf_edit(session, proposal)


def finalize_hf_candidate(session, candidate_id):
    """Select a practice PASS and close the session for further attempts."""
    if not isinstance(candidate_id, str) or not _CANDIDATE.fullmatch(candidate_id):
        raise ValueError("Invalid HF edit candidate ID")
    with _locked(session) as root:
        _require_open(root)
        state, _ = _state(root)
        attempt = root / "attempts" / candidate_id
        result_path = attempt / "result.json"
        if not result_path.is_file():
            raise ValueError("Candidate has no completed practice evaluation")
        result = _read(result_path)
        if result["disposition"] != "PASS":
            raise ValueError("Candidate must pass the practice gate before finalization")
        candidate, patch = attempt / "model", attempt / "patch.json"
        report_path = attempt / "practice_report.json"
        if (load_checkpoint(candidate).checkpoint_sha256() != result["candidate_sha256"]
                or file_hash(patch) != result["patch_file_sha256"]
                or file_hash(report_path) != result["practice_report_sha256"]):
            raise ValueError("Candidate or its practice evidence changed before finalization")
        report = _read(report_path)
        if (report.get("schema") != "weightbridge.hf_edit_evaluation.v1" or report.get("disposition") != "PASS"
                or report.get("base_sha256") != state["base_sha256"]
                or report.get("suite_sha256") != state["practice_sha256"]
                or report.get("candidate_sha256") != result["candidate_sha256"]
                or file_hash(candidate / ".weightbridge_receipt.json") != report.get("receipt_sha256")):
            raise ValueError("Practice report does not bind the frozen base, suite and candidate")
        decision = {**result, "schema": "weightbridge.hf_edit_session_decision.v1",
                    "disposition": "FINALIZED", "practice_disposition": "PASS"}
        json_dump_atomic(root / "decision.json", decision)
        return decision


def stop_hf_session(session, reason):
    """Close a session without finalizing a candidate."""
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
        raise ValueError("Stop reason must be 1 to 512 characters")
    with _locked(session) as root:
        _require_open(root)
        _state(root)
        decision = {"schema": "weightbridge.hf_edit_session_decision.v1", "disposition": "STOPPED",
                    "reason": reason}
        json_dump_atomic(root / "decision.json", decision)
        return decision
