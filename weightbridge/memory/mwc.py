"""Synthetic-only Memory-to-Weights Consolidation records and workflow.

No external/private memory connector is enabled. Records are immutable JSON
objects created for explicitly synthetic experiments.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


class ExperienceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> str:
        if record.get("data_class") != "synthetic_authorized":
            raise ValueError("MWC pilot accepts only explicitly authorized synthetic records")
        observation_id = record.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("Each experience must refer to one unique observation_id")
        normalized = dict(record)
        normalized.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        normalized.setdefault("schema", "weightbridge.mwc.experience.v1")
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        record_id = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()
        target = self.root / f"{record_id}.json"
        with target.open("xb") as stream:
            stream.write(encoded + b"\n")
            stream.flush()
        return record_id

    def retrieve(self, failure_fingerprint: str, *, limit: int = 8) -> list[dict[str, Any]]:
        if type(limit) is not int or not 0 <= limit <= 32:
            raise ValueError("memory retrieval limit must be in [0, 32]")
        records = []
        for path in sorted(self.root.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("data_class") == "synthetic_authorized" and record.get("failure_fingerprint") == failure_fingerprint:
                records.append(record)
        return records[:limit]


class MemoryToWeightsConsolidation:
    """Record the provenance chain from failure through independent evaluation."""

    def __init__(self, store: ExperienceStore):
        self.store = store

    def consolidate(self, *, failure_fingerprint: str, observation_id: str,
                    diagnosis: str, patch_sha256: str, evaluation: dict[str, Any],
                    intervention: dict[str, Any] | None = None,
                    source_model_sha256: str | None = None,
                    data_class: str = "synthetic_authorized") -> str:
        if not all(isinstance(value, str) and value for value in
                   (failure_fingerprint, observation_id, diagnosis, patch_sha256)):
            raise ValueError("MWC evidence identifiers and diagnosis must be nonempty strings")
        return self.store.add({
            "data_class": data_class,
            "failure_fingerprint": failure_fingerprint,
            "observation_id": observation_id,
            "diagnosis": diagnosis,
            "patch_sha256": patch_sha256,
            "intervention": intervention,
            "source_model_sha256": source_model_sha256,
            "evaluation": evaluation,
            "workflow": ["execution_failure", "immutable_evidence", "pattern_analysis",
                         "neural_diagnosis", "candidate_edit", "independent_evaluation",
                         "retention_or_rollback"],
        })
