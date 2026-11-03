"""Append-only selection of verified local research checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re

from ..core import file_hash, json_dump_atomic
from ..evaluation.hf_edit import _load_suite
from ..inspection.safetensors import load_checkpoint


_SCHEMA = "weightbridge.checkpoint_lineage.v1"
_EVENT_SCHEMA = "weightbridge.checkpoint_lineage_event.v1"
_CYCLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@contextmanager
def workspace_lock(root):
    """Serialize host operations on an existing workspace directory."""
    with (Path(root) / ".lock").open("a+b", buffering=0) as lock:
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
            raise ValueError("Research workspace is busy") from exc
        yield


def _artifact(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, raw


def _checkpoint(path, workspace):
    checkpoint = load_checkpoint(path)
    if not checkpoint.root.is_dir():
        raise ValueError("Lineage requires local HF checkpoint directories")
    if workspace.is_relative_to(checkpoint.root) or checkpoint.root.is_relative_to(workspace):
        raise ValueError("Checkpoint directory and lineage metadata must be separate")
    return {"path": str(checkpoint.root), "sha256": checkpoint.checkpoint_sha256()}


def _check_id(cycle_id):
    if not isinstance(cycle_id, str) or not _CYCLE_ID.fullmatch(cycle_id):
        raise ValueError("Invalid research cycle ID")


def _check_reason(reason):
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
        raise ValueError("Lineage reason must be 1 to 1024 characters")


def _event_checksum(event):
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _describe(root, *, audit=False, verify_head=True):
    metadata, raw = _artifact(root / "lineage.json")
    manifest = json.loads(raw)
    if manifest.get("schema") != _SCHEMA:
        raise ValueError("Unsupported checkpoint lineage")
    original = manifest["root"]
    active = [{**original, "cycle_id": None, "holdout_path": None, "holdout_sha256": None}]
    accepted = {original["sha256"]}
    events, cycles = [], set()
    previous = metadata["sha256"]
    event_dir = root / "events"
    if not event_dir.is_dir() or event_dir.is_symlink():
        raise ValueError("Lineage event directory is missing or invalid")
    for number, path in enumerate(sorted(event_dir.glob("*.json")), 1):
        evidence, raw = _artifact(path)
        event = json.loads(raw)
        if (path.name != f"{number:08d}.json" or path.is_symlink()
                or event.get("schema") != _EVENT_SCHEMA or event.get("sequence") != number
                or event.get("previous_sha256") != previous
                or event.get("event_sha256") != _event_checksum(event)
                or event.get("parent_sha256") != active[-1]["sha256"]):
            raise ValueError("Lineage event sequence or hash chain differs")
        _check_reason(event.get("reason"))
        kind = event.get("kind")
        if kind in ("ACCEPT", "REJECT"):
            cycle_id = event.get("cycle_id")
            _check_id(cycle_id)
            if cycle_id in cycles:
                raise ValueError("Lineage contains a duplicate cycle decision")
            cycles.add(cycle_id)
            verification = event.get("verification")
            if audit and verification is not None:
                for name in ("report", "receipt", "suite", "source_holdout"):
                    artifact = verification[name]
                    if file_hash(artifact["path"]) != artifact["sha256"]:
                        raise ValueError(f"Recorded lineage {name} changed")
            if kind == "ACCEPT":
                candidate = event["candidate"]
                if (verification is None or verification["disposition"] != "PASS"
                        or candidate["sha256"] in accepted
                        or not Path(candidate["path"]).is_relative_to(root.parent)):
                    raise ValueError("Accepted lineage event is inconsistent")
                accepted.add(candidate["sha256"])
                if audit and _checkpoint(candidate["path"], root) != candidate:
                    raise ValueError("Recorded accepted checkpoint changed")
                source = verification["source_holdout"]
                active.append({**candidate, "cycle_id": cycle_id,
                               "holdout_path": source["path"], "holdout_sha256": source["sha256"]})
        elif kind == "ROLLBACK":
            for index, ancestor in enumerate(active[:-1]):
                if ancestor["sha256"] == event.get("ancestor_sha256"):
                    active = active[:index + 1]
                    break
            else:
                raise ValueError("Rollback target is outside the accepted ancestry")
        else:
            raise ValueError("Unknown checkpoint lineage event")
        previous = evidence["sha256"]
        events.append(event)
    head = active[-1]
    if verify_head and _checkpoint(head["path"], root) != {key: head[key] for key in ("path", "sha256")}:
        raise ValueError("Active checkpoint changed")
    if audit and head["sha256"] != original["sha256"] and _checkpoint(original["path"], root) != original:
        raise ValueError("Pinned root checkpoint changed")
    return {
        "schema": _SCHEMA,
        "root_path": original["path"], "root_sha256": original["sha256"],
        "head_path": head["path"], "head_sha256": head["sha256"],
        "ancestry": active, "events": events, "last_event_sha256": previous,
    }


def init_lineage(workspace, base):
    """Pin the original checkpoint in a fresh metadata directory."""
    root = Path(workspace).resolve()
    if root.exists() or Path(workspace).is_symlink():
        raise ValueError("Checkpoint lineage directory already exists")
    checkpoint = _checkpoint(base, root)
    root.mkdir(parents=True)
    (root / "events").mkdir()
    json_dump_atomic(root / "lineage.json", {"schema": _SCHEMA, "root": checkpoint})
    return lineage_status(root)


def lineage_status(workspace):
    """Verify the event chain and report the current accepted ancestry."""
    root = Path(workspace).resolve()
    with workspace_lock(root):
        return _describe(root)


def audit_lineage(workspace):
    """Check all recorded verification artifacts and accepted checkpoints."""
    root = Path(workspace).resolve()
    with workspace_lock(root):
        return _describe(root, audit=True)


def _check_cycle(state, cycle_id, reason, expected_parent_sha256):
    _check_id(cycle_id)
    _check_reason(reason)
    if expected_parent_sha256 is not None and state["head_sha256"] != expected_parent_sha256:
        raise ValueError("Lineage parent changed before recording the cycle")
    if any(event.get("cycle_id") == cycle_id for event in state["events"]):
        raise ValueError("Research cycle already has a lineage decision")


def _append(root, state, kind, reason, **fields):
    number = len(state["events"]) + 1
    event = {
        "schema": _EVENT_SCHEMA,
        "sequence": number,
        "previous_sha256": state["last_event_sha256"],
        "parent_sha256": state["head_sha256"],
        "kind": kind,
        "reason": reason,
        **fields,
    }
    event["event_sha256"] = _event_checksum(event)
    json_dump_atomic(root / "events" / f"{number:08d}.json", event)
    return _describe(root)


def _verification(candidate, report, suite, source_holdout, parent_sha256):
    receipt_ref, raw = _artifact(Path(candidate["path"]) / ".weightbridge_receipt.json")
    receipt = json.loads(raw)
    report_ref, raw = _artifact(report)
    outcome = json.loads(raw)
    suite_ref, raw = _artifact(suite)
    suite_doc = _load_suite(raw)
    if (receipt.get("schema") != "weightbridge.receipt.v2"
            or receipt.get("status") != "candidate_only_not_promoted"
            or receipt.get("base_sha256") != parent_sha256
            or receipt.get("candidate_sha256") != candidate["sha256"]):
        raise ValueError("Candidate receipt does not bind the active parent and candidate")
    if (outcome.get("schema") != "weightbridge.hf_edit_evaluation.v1"
            or outcome.get("base_sha256") != parent_sha256
            or outcome.get("candidate_sha256") != candidate["sha256"]
            or outcome.get("receipt_sha256") != receipt_ref["sha256"]
            or outcome.get("suite_sha256") != suite_ref["sha256"]):
        raise ValueError("Verification report does not bind the parent, candidate, receipt and suite")
    source_ref = suite_ref
    if source_holdout is not None:
        source_ref, raw = _artifact(source_holdout)
        source = _load_suite(raw)
        if (source["choices"] != suite_doc["choices"] or source["target"] != suite_doc["target"]
                or source["sentinel"] != suite_doc["sentinel"][:len(source["sentinel"])]):
            raise ValueError("Evaluation suite does not preserve the source holdout")

    choices = suite_doc["choices"]
    for group in ("target", "sentinel"):
        results = outcome.get(group)
        prompts = suite_doc[group]
        if not isinstance(results, list) or len(results) != len(prompts):
            raise ValueError(f"Verification {group} rows do not match the suite")
        for result, prompt in zip(results, prompts, strict=True):
            if (not isinstance(result, dict) or result.get("id") != prompt["id"]
                    or result.get("label") != prompt["label"]
                    or result.get("base_prediction") not in choices
                    or result.get("candidate_prediction") not in choices):
                raise ValueError(f"Verification {group} row does not match the suite")
    before = sum(row["base_prediction"] == row["label"] for row in outcome["target"])
    after = sum(row["candidate_prediction"] == row["label"] for row in outcome["target"])
    gain = after - before
    flips = sum(row["base_prediction"] != row["candidate_prediction"] for row in outcome["sentinel"])
    minimum = outcome.get("min_target_gain")
    maximum = outcome.get("max_sentinel_flips")
    if (type(minimum) is not int or not 1 <= minimum <= len(suite_doc["target"])
            or type(maximum) is not int or not 0 <= maximum <= len(suite_doc["sentinel"])):
        raise ValueError("Invalid verification thresholds")
    disposition = "PASS" if gain >= minimum and flips <= maximum else "FAIL"
    if (outcome.get("disposition") != disposition or outcome.get("target_gain") != gain
            or outcome.get("sentinel_prediction_flips") != flips
            or outcome.get("base_target_correct") != before or outcome.get("candidate_target_correct") != after):
        raise ValueError("Verification gate differs from paired prediction rows")
    return {
        "report": report_ref, "receipt": receipt_ref, "suite": suite_ref, "source_holdout": source_ref,
        "disposition": disposition, "target_gain": gain, "sentinel_prediction_flips": flips,
    }


def record_lineage_decision(
    workspace, candidate, report, *, suite, cycle_id, accept, reason, expected_parent_sha256=None,
    source_holdout=None,
):
    """Record a trusted independent evaluation and accept only a verified PASS."""
    if type(accept) is not bool:
        raise ValueError("Lineage acceptance must be a boolean")
    root = Path(workspace).resolve()
    with workspace_lock(root):
        state = _describe(root)
        _check_cycle(state, cycle_id, reason, expected_parent_sha256)
        selected = _checkpoint(candidate, root)
        if not Path(selected["path"]).is_relative_to(root.parent):
            raise ValueError("Candidate must be within the host research workspace")
        if selected["sha256"] == state["head_sha256"]:
            raise ValueError("Candidate is identical to its active parent")
        verification = _verification(selected, report, suite, source_holdout, state["head_sha256"])
        if accept:
            if verification["disposition"] != "PASS":
                raise ValueError("A failed verification cannot advance the lineage")
            if selected["sha256"] == state["root_sha256"] or any(
                event["kind"] == "ACCEPT" and event["candidate"]["sha256"] == selected["sha256"]
                for event in state["events"]
            ):
                raise ValueError("This checkpoint was already accepted in the lineage")
        return _append(root, state, "ACCEPT" if accept else "REJECT", reason,
                       cycle_id=cycle_id, candidate=selected, verification=verification)


def record_lineage_rejection(workspace, *, cycle_id, reason, expected_parent_sha256):
    """Record a stopped cycle with no independently evaluated finalist."""
    if not isinstance(expected_parent_sha256, str):
        raise ValueError("Rejection requires the expected parent hash")
    root = Path(workspace).resolve()
    with workspace_lock(root):
        state = _describe(root)
        _check_cycle(state, cycle_id, reason, expected_parent_sha256)
        return _append(root, state, "REJECT", reason, cycle_id=cycle_id)


def rollback_lineage(workspace, ancestor_sha256, *, reason):
    """Select an accepted ancestor after verifying its checkpoint bytes."""
    _check_reason(reason)
    root = Path(workspace).resolve()
    with workspace_lock(root):
        state = _describe(root, verify_head=False)
        ancestor = next((item for item in state["ancestry"][:-1] if item["sha256"] == ancestor_sha256), None)
        if ancestor is None:
            raise ValueError("Rollback target must be an earlier accepted ancestor")
        if _checkpoint(ancestor["path"], root)["sha256"] != ancestor_sha256:
            raise ValueError("Rollback ancestor checkpoint changed")
        return _append(root, state, "ROLLBACK", reason, ancestor_sha256=ancestor_sha256)
