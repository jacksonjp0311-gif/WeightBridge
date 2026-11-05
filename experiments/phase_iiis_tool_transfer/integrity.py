"""Phase S identities, independent of the immutable Q/R source manifests."""
from __future__ import annotations

import re
from pathlib import Path

from experiments.phase_iiir_replication.integrity import source_manifest as r_sources
from experiments.phase_iiiq_baseline.integrity import ROOT, read, verify_sources, directory_hash
from weightbridge.core import file_hash, json_dump_atomic

ORIGINAL = '7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8'
DECIMAL_PARENT = '2d9102982dc0f0fdf3397308cd397bc2d5239eeb99ef6297e5f09f6f6deba158'
PARENTS = {'sequential': DECIMAL_PARENT, 'original-joint': ORIGINAL}


def checked_sha256(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected identity must contain exactly 64 lowercase hexadecimal characters')
    return value


def source_manifest():
    result = r_sources()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return dict(sorted(result.items()))


def write_once(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing evidence is immutable: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    json_dump_atomic(path, value)
    return file_hash(path)
