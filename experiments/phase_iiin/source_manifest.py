"""Bind the paired trial to inherited verified machinery and its own sources."""
from pathlib import Path

from experiments.phase_iiim.source_manifest import source_hashes as inherited_hashes
from weightbridge.core import file_hash

ROOT = Path(__file__).resolve().parents[2]
_FIT = (
    *(f"experiments/phase_iiin/{name}" for name in (
        "semantic_protocol.json", "semantic_roster.py", "semantic_metrics.py", "source_manifest.py", "run_paired.py")),
    "weightbridge/adaptation/paired_supervision.py", "weightbridge/adaptation/paired_update.py",
)


def source_hashes(*, verifier=False):
    extra = ("experiments/phase_iiin/verification.py", "experiments/phase_iiim/verification.py") if verifier else ()
    return {**inherited_hashes(verifier=verifier),
            **{name: file_hash(ROOT / name) for name in sorted((*_FIT, *extra))}}
