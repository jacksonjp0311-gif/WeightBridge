"""Hash-bind Track A without allowing separately built Track B to change its identity."""
from pathlib import Path

from experiments.phase_iiio.source_manifest import source_hashes as inherited_hashes
from weightbridge.core import file_hash

ROOT = Path(__file__).resolve().parents[2]
_FIT = (
    *(f"experiments/phase_iiip/{name}" for name in (
        "semantic_protocol.json", "semantic_roster.py", "semantic_metrics.py", "source_manifest.py",
        "run_right.py", "execute.py", "independent_bounds.py")),
    "weightbridge/adaptation/right_factor_geometry.py", "weightbridge/adaptation/selected_block.py",
)


def source_hashes(*, verifier=False):
    extra = ("experiments/phase_iiip/verification.py",) if verifier else ()
    return {**inherited_hashes(verifier=verifier),
            **{name: file_hash(ROOT / name) for name in sorted((*_FIT, *extra))}}
