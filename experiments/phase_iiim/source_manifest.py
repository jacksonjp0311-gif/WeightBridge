"""Bind fitting and verifier implementations to separate immutable evidence stages."""
from pathlib import Path

from experiments.phase_iiil.source_manifest import source_hashes as previous_hashes
from weightbridge.core import file_hash


ROOT = Path(__file__).resolve().parents[2]
_FIT = (
    *(f"experiments/phase_iiim/{name}" for name in (
        "semantic_protocol.json", "semantic_roster.py", "semantic_metrics.py", "run_factorial.py", "source_manifest.py",
    )),
    *(f"weightbridge/adaptation/{name}.py" for name in ("factorial_geometry", "factorial_update", "factorial_probe")),
)


def source_hashes(*, verifier=False):
    paths = (*_FIT, "experiments/phase_iiim/verification.py", "experiments/phase_iiil/verification.py") if verifier else _FIT
    return {**previous_hashes(), **{name: file_hash(ROOT / name) for name in sorted(paths)}}
