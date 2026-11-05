"""Prospective Phase III-R sources; historical protocols are never rewritten."""
from pathlib import Path

from experiments.phase_iiiq_baseline.integrity import (
    PARENT, ROOT, read, verify_sources, directory_hash, source_manifest as q_sources,
)
from weightbridge.core import file_hash


def source_manifest():
    result = q_sources()
    for path in sorted((ROOT / "experiments/phase_iiir_replication").glob("*.py")):
        result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return dict(sorted(result.items()))
