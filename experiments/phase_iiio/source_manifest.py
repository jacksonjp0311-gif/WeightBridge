"""Hash-bind joint feasibility to inherited adaptation and verification code."""
from pathlib import Path

from experiments.phase_iiin.source_manifest import source_hashes as inherited_hashes
from weightbridge.core import file_hash

ROOT = Path(__file__).resolve().parents[2]
_FIT = (
    *(f"experiments/phase_iiio/{name}" for name in (
        "semantic_protocol.json", "semantic_roster.py", "semantic_metrics.py", "source_manifest.py",
        "run_joint.py", "execute.py")),
    "weightbridge/adaptation/joint_feasibility.py", "weightbridge/adaptation/joint_update.py",
)


def source_hashes(*, verifier=False):
    extra = ("experiments/phase_iiio/verification.py", "experiments/phase_iiin/verification.py") if verifier else ()
    return {**inherited_hashes(verifier=verifier),
            **{name: file_hash(ROOT / name) for name in sorted((*_FIT, *extra))}}
