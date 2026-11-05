"""Small explicit-parent identity and replay checks for Phase III-S.

The host owns staged access. These helpers refuse unbound workers and replay
measured outcomes; they cannot authorize training, lineage or deployment.
"""
from __future__ import annotations

from pathlib import Path
import re

from experiments.phase_iiiq_baseline.integrity import read, verify_sources
from experiments.phase_iiiq_baseline import measurement as decimal
from experiments.phase_iiil.analyze import _checked_measurement
from weightbridge.core import file_hash, json_dump_atomic
from .measurement import checked_tools, summarize_tools

SCHEMA = "weightbridge.phase_iiis.evaluation.v1"
_SHA = re.compile(r"[0-9a-f]{64}")
PROTECTED = ("preservation-validation", "qualification", "confirmation", "selection", "hidden", "historical-retention")
STAGES = frozenset((*PROTECTED, "calibration", "smoke", "smoke-before", "smoke-after", "smoke-reload",
    "smoke-disabled", "development", "practice", "merged-development", "merged-practice",
    "reconstruction-development", "reconstruction-practice", "reconstruction-disposable", "demo"))


def _requirements(stage):
    return {"preservation-validation": {"merged-development", "exact-reconstruction"},
        "historical-retention": {"merged-development", "exact-reconstruction", "both-arms-locked"},
        "qualification": {"merged-development", "exact-reconstruction", "both-arms-locked", "historical-retention"},
        "selection": {"merged-development", "exact-reconstruction", "both-arms-locked", "historical-retention"},
        "confirmation": {"merged-development", "exact-reconstruction", "both-arms-locked", "historical-retention", "qualification"},
        "hidden": {"merged-development", "exact-reconstruction", "both-arms-locked", "historical-retention", "qualification"}}[stage]


def sha(value):
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError("Expected an exact SHA-256 identity")
    return value


def endpoint_settings(protocol, task):
    if task not in ("tool", "decimal"):
        raise ValueError("Unsupported evaluation task endpoint")
    settings = protocol["endpoints"][task]
    limits = {"cpu_threads": 6, "batch_size": 8, "max_new_tokens": 96 if task == "tool" else 16}
    limits["max_input_tokens"] = 512 if task == "tool" else 128
    if any(type(settings.get(key)) is not int or settings[key] != value for key, value in limits.items()):
        raise ValueError("Phase S endpoint changed its frozen CPU, batch or decode configuration")
    if type(settings.get("max_input_tokens")) is not int or settings["max_input_tokens"] < 1:
        raise ValueError("Endpoint input budget must be explicit")
    return settings


def validate_authorization(protocol_path, authorization, *, stage, task, expected_rows,
                           expected_checkpoint, expected_adapter=None):
    """Validate the host token before reading a roster or loading any model."""
    protocol = read(protocol_path)
    if stage not in STAGES:
        raise ValueError("Unknown Phase S evaluation stage; aliases cannot bypass protected access")
    endpoint_settings(protocol, task)
    for digest in (expected_rows, expected_checkpoint, protocol["checkpoint_sha256"], protocol["task_manifest_sha256"]):
        sha(digest)
    if expected_adapter is not None:
        sha(expected_adapter)
    if authorization is None:
        raise ValueError("Every S evaluator needs a hash-bound host roster authorization")
    auth = read(authorization)
    identity = {"schema": "weightbridge.phase_iiis.evaluation_authorization.v1", "stage": stage,
        "task": task, "protocol_sha256": file_hash(protocol_path), "roster_sha256": expected_rows,
        "task_manifest_sha256": protocol["task_manifest_sha256"], "source_sha256": protocol["source_sha256"],
        "parent_sha256": protocol["checkpoint_sha256"], "representation_sha256": expected_checkpoint,
        "adapter_sha256": expected_adapter, "production_authority": False, "lineage_authority": False}
    if not isinstance(stage, str) or not stage or any(auth.get(k) != value for k, value in identity.items()):
        raise ValueError("Host evaluation authorization belongs to another source, task, stage or model")
    # Protected stages are only entered with a stored finalist and explicit host
    # commitments to the earlier gate receipts. Actual gates are replayed by host.
    if stage in PROTECTED:
        if expected_adapter is not None:
            raise ValueError("Independent S verification evaluates exact stored checkpoints")
        sha(auth.get("finalist_lock_sha256"))
        required = _requirements(stage)
        gates = auth.get("prerequisite_receipts")
        if not isinstance(gates, dict) or set(gates) != required:
            raise ValueError("Host stage authorization lacks its completed prerequisite commitments")
        for digest in gates.values():
            sha(digest)
    verify_sources(protocol["source_sha256"])
    return protocol, auth


def authorize_evaluation(out, protocol_path, *, stage, task, roster_sha256, representation_sha256,
                         adapter_sha256=None, finalist_lock_path=None, prerequisite_paths=None):
    """Host-only versioned authorization; no rosters are opened by this operation."""
    if Path(out).exists():
        raise FileExistsError("Host evaluation authorization is write-once")
    protocol = read(protocol_path)
    if stage not in STAGES:
        raise ValueError("Unknown Phase S evaluation stage; aliases cannot bypass protected access")
    endpoint_settings(protocol, task)
    value = {"schema": "weightbridge.phase_iiis.evaluation_authorization.v1", "stage": stage, "task": task,
        "protocol_sha256": file_hash(protocol_path), "roster_sha256": sha(roster_sha256),
        "task_manifest_sha256": sha(protocol["task_manifest_sha256"]), "source_sha256": protocol["source_sha256"],
        "parent_sha256": sha(protocol["checkpoint_sha256"]), "representation_sha256": sha(representation_sha256),
        "adapter_sha256": sha(adapter_sha256) if adapter_sha256 else None,
        "production_authority": False, "lineage_authority": False}
    if stage in PROTECTED:
        if adapter_sha256 is not None or finalist_lock_path is None or set(prerequisite_paths or {}) != _requirements(stage):
            raise ValueError("Protected evaluation requires its locked stored model and completed host stages")
        lock = read(finalist_lock_path)
        if (lock.get("protocol_sha256") != value["protocol_sha256"] or lock.get("passed") is not True
                or lock.get("task_manifest_sha256") != value["task_manifest_sha256"]):
            raise ValueError("Finalist lock belongs to another protocol or failed its gate")
        value["finalist_lock_sha256"] = file_hash(finalist_lock_path)
        value["prerequisite_receipts"] = {}
        for name, path in prerequisite_paths.items():
            gate = read(path)
            if gate.get("passed") is not True or gate.get("task_manifest_sha256") != value["task_manifest_sha256"]:
                raise ValueError(f"Protected evaluation refused after incomplete or failed {name}")
            # Global both-arm lock has its own study identity; all model-specific
            # gates additionally bind the exact finalist and arm protocol.
            if name != "both-arms-locked" and (gate.get("protocol_sha256") != value["protocol_sha256"]
                    or gate.get("representation_sha256") != lock.get("representation_sha256")):
                raise ValueError("Prerequisite gate belongs to another stored finalist")
            value["prerequisite_receipts"][name] = file_hash(path)
    verify_sources(protocol["source_sha256"])
    json_dump_atomic(out, value)
    return value


def validate_worker(path, rows_path, protocol_path, checkpoint, adapter_sha, stage, task, *, authorization=None):
    protocol, result = read(protocol_path), read(path)
    settings = endpoint_settings(protocol, task)
    expected = {"schema": SCHEMA, "stage": stage, "task": task, "checkpoint_sha256": checkpoint,
        "parent_sha256": protocol["checkpoint_sha256"], "adapter_sha256": adapter_sha,
        "protocol_sha256": file_hash(protocol_path), "roster_sha256": file_hash(rows_path),
        "task_manifest_sha256": protocol["task_manifest_sha256"], "source_sha256": protocol["source_sha256"],
        "representation": "fp32_adapter_on_bf16_parent" if adapter_sha else "stored_checkpoint"}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("S worker identity differs from the frozen endpoint, roster or representation")
    if authorization is not None:
        validate_authorization(protocol_path, authorization, stage=stage, task=task,
            expected_rows=expected["roster_sha256"], expected_checkpoint=checkpoint, expected_adapter=adapter_sha)
        if result.get("authorization_sha256") != file_hash(authorization):
            raise ValueError("S worker authorization identity changed")
    verify_sources(protocol["source_sha256"])
    measured, rows = result["measurement"], read(rows_path)
    if task == "tool":
        checked_tools(measured, rows, settings, protocol["tool_completion"])
        summary = summarize_tools(measured)
    else:
        _checked_measurement(measured, rows, decimal.endpoint(settings))
        summary = decimal.summarize(measured)
    if summary != result.get("summary"):
        raise ValueError("S worker summary does not reproduce from complete responses")
    return result


def exact_native_replay(reference, replay):
    """Compare each representation with its own archived output, never with its parent."""
    if reference.get("endpoint") != replay.get("endpoint") or reference.get("completion") != replay.get("completion"):
        return False
    if len(reference.get("rows", [])) != len(replay.get("rows", [])) or not reference.get("rows"):
        return False
    fields = ("id", "text", "generated_token_ids", "stop_reason", "prefix_token_ids", "prefix_tokens",
              "native_end_token_id", "unexpected_special_token_ids")
    return all(all(a.get(k) == b.get(k) for k in fields)
               for a, b in zip(reference["rows"], replay["rows"], strict=True))


def replay_pair(reference_path, replay_path, rows_path, protocol_path, *, checkpoint, adapter_sha, stage, task):
    original = validate_worker(reference_path, rows_path, protocol_path, checkpoint, adapter_sha, stage, task)
    replay = validate_worker(replay_path, rows_path, protocol_path, checkpoint, adapter_sha, stage, task)
    exact = exact_native_replay(original["measurement"], replay["measurement"])
    return {"passed": exact, "exact_native_generation": exact,
        "reference_sha256": file_hash(reference_path), "replay_sha256": file_hash(replay_path),
        "checkpoint_sha256": checkpoint, "adapter_sha256": adapter_sha, "task": task,
        "scope": "Exact full native token/text/termination replay against this representation's own prior output"}


def final_replay_gate(replays, *, reconstruction, original_restore, sequential=False,
                      sequential_parent_restore=None, required_keys=()):
    """Missing replay or restoration is an incomplete result, never a synthetic PASS."""
    if not required_keys or set(replays) != set(required_keys):
        return {"passed": False, "reason": "missing_or_unexpected_replay_stages"}
    checks = {"exact_reconstruction": (reconstruction or {}).get("passed") is True,
        "original_restoration": (original_restore or {}).get("passed") is True,
        "exact_replays": all((r or {}).get("passed") is True and (r or {}).get("exact_native_generation") is True
                             for r in replays.values())}
    if sequential:
        checks["sequential_parent_restoration"] = (sequential_parent_restore or {}).get("passed") is True
    return {"passed": all(checks.values()), "checks": checks, "lineage_accepted": False,
            "production_authority": False}
