"""Source boundaries for the fitting and separately launched verifier stages."""

from pathlib import Path

from experiments.phase_iiik.verification import source_hashes as prior_source_hashes
from weightbridge.core import file_hash


ROOT = Path(__file__).resolve().parents[2]
_FIT = (
    *(f"experiments/phase_iiil/{name}" for name in (
        "semantic_protocol.json", "semantic_roster.py", "semantic_diagnostics.py", "run_semantic.py", "source_manifest.py",
        "replay_repair.json",
    )),
    *(f"weightbridge/adaptation/{name}.py" for name in ("semantic_consensus", "semantic_update")),
)


def source_hashes(*, verifier=False):
    paths = (*_FIT, "experiments/phase_iiil/verification.py") if verifier else _FIT
    return {**prior_source_hashes(), **{name: file_hash(ROOT / name) for name in sorted(paths)}}
