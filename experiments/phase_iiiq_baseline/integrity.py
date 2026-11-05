"""Write-once study identities and host-owned stage boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from weightbridge.core import file_hash, json_dump_atomic

ROOT = Path(__file__).resolve().parents[2]
PARENT = "7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8"


def read(path):
    return json.loads(Path(path).read_bytes())


def source_manifest():
    files = set((ROOT / "weightbridge").rglob("*.py"))
    files.update((ROOT / "experiments/phase_iiiq_baseline").glob("*.py"))
    for folder in ("phase_iiig", "phase_iiih", "phase_iiil"):
        files.update((ROOT / "experiments" / folder).glob("*.py"))
    files.add(ROOT / "pyproject.toml")
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(files)}


def verify_sources(manifest):
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Frozen study source changed: {name}")


def directory_hash(path):
    path = Path(path)
    files = {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("Empty artifact directory")
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), files


def lock_finalist(out, *, adapter_path, protocol_sha256, manifest_sha256, practice_gate, development_gate):
    if not practice_gate.get("passed") or not development_gate.get("passed"):
        raise ValueError("Cannot lock a behaviorally unqualified finalist")
    identity, files = directory_hash(adapter_path)
    value = {"schema": "weightbridge.supervised_baseline.finalist.v1", "adapter_sha256": identity,
        "adapter_files": files, "protocol_sha256": protocol_sha256, "manifest_sha256": manifest_sha256,
        "practice_gate": practice_gate, "development_gate": development_gate,
        "production_authority": False, "lineage_authority": False}
    json_dump_atomic(Path(out) / "finalist-lock.json", value)
    return value


def authorize_stage(out, stage, *, representation_sha256, roster_sha256):
    """Only host code calls this, after independently recomputing each prior gate."""
    root = Path(out)
    lock = read(root / "finalist-lock.json")
    if not lock["practice_gate"]["passed"] or not lock["development_gate"]["passed"]:
        raise ValueError("Finalist lock failed its frozen gates")
    requirements = {"preservation-validation": ("merged-development",),
                    "selection": ("merged-development", "preservation-validation"),
                    "hidden": ("merged-development", "preservation-validation", "selection")}
    if stage not in requirements:
        raise ValueError("Unknown independent host stage")
    receipts = {}
    for name in requirements[stage]:
        receipt = read(root / f"{name}-gate.json")
        if (not receipt.get("passed") or receipt.get("representation_sha256") != representation_sha256
                or receipt.get("protocol_sha256") != lock["protocol_sha256"]
                or receipt.get("manifest_sha256") != lock["manifest_sha256"]):
            raise ValueError(f"Independent stage refused after incomplete or failed {name}")
        receipts[name] = file_hash(root / f"{name}-gate.json")
    return {"schema": "weightbridge.supervised_baseline.stage_authorization.v1", "stage": stage,
        "parent_sha256": PARENT,
        "roster_sha256": roster_sha256,
        "representation_sha256": representation_sha256, "protocol_sha256": lock["protocol_sha256"],
        "manifest_sha256": lock["manifest_sha256"], "prerequisite_receipts": receipts,
        "finalist_lock_sha256": file_hash(root / "finalist-lock.json"), "production_authority": False}
