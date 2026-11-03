"""Append-only experiment ledger with unique IDs and versioned corrections."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


class LedgerError(ValueError):
    """Invalid, duplicate or unreadable evidence record."""


def validate_experiment_record(record: dict[str, Any]) -> None:
    """Require auditable identity and an explicit evaluation applicability record."""
    if record.get("schema") != "weightbridge.experiment.v1":
        return
    required = {
        "experiment_id", "measurement_key", "timestamp_utc", "hypothesis", "source_model",
        "proposed_modification", "affected_parameters", "evaluation", "resources",
        "artifact_root", "artifacts", "disposition",
    }
    missing = sorted(required - set(record))
    if missing:
        raise LedgerError("Experiment record missing required fields: " + ", ".join(missing))
    if not isinstance(record.get("measurement_key"), str) or not record["measurement_key"]:
        raise LedgerError("Experiment records require a deterministic measurement_key")
    for key in ("source_model", "affected_parameters", "evaluation", "resources"):
        if not isinstance(record[key], dict):
            raise LedgerError(f"Experiment {key} must be a mapping")
    evaluation = record["evaluation"]
    if not any(key in evaluation for key in ("dataset", "suite", "type")):
        raise LedgerError("Experiment evaluation must identify a dataset, suite or measurement type")
    if not any(key in evaluation for key in ("practice_test_separated", "split", "not_applicable")):
        raise LedgerError("Experiment evaluation must state practice/test separation or not_applicable")
    if not any(key in evaluation for key in ("seed", "seeds", "sampling", "not_applicable")):
        raise LedgerError("Experiment evaluation must identify seeds/sampling or not_applicable")


def load_ledger(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise LedgerError(f"Blank line at ledger line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"Invalid JSON at ledger line {line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("experiment_id"), str):
                raise LedgerError(f"Invalid record at ledger line {line_number}")
            if record["experiment_id"] in seen:
                raise LedgerError(f"Duplicate experiment_id: {record['experiment_id']}")
            seen.add(record["experiment_id"])
            records.append(record)
    return records


def append_experiment(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one canonical JSON record, refusing duplicate run identities.

    A correction is a new record with ``supersedes`` set to the prior ID. No
    historical line is edited or removed.
    """
    if not isinstance(record, dict) or not isinstance(record.get("experiment_id"), str) or not record["experiment_id"]:
        raise LedgerError("experiment_id is required")
    validate_experiment_record(record)
    if not isinstance(record.get("disposition"), str):
        raise LedgerError("disposition is required")
    path = Path(path)
    existing = load_ledger(path)
    if any(item["experiment_id"] == record["experiment_id"] for item in existing):
        raise LedgerError(f"Duplicate experiment_id: {record['experiment_id']}")
    measurement_key = record.get("measurement_key")
    if measurement_key is not None:
        if not isinstance(measurement_key, str) or not measurement_key:
            raise LedgerError("measurement_key must be nonempty text when provided")
        if any(item.get("measurement_key") == measurement_key for item in existing):
            raise LedgerError("Duplicate measurement_key: same run identity cannot be counted twice")
    supersedes = record.get("supersedes")
    if supersedes is not None and not any(item["experiment_id"] == supersedes for item in existing):
        raise LedgerError("supersedes must reference an existing experiment_id")
    value = dict(record)
    value.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return value
