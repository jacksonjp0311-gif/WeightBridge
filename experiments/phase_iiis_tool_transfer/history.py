"""Host-only freshness audit with an explicit unopened-R-bank exclusion boundary."""
from __future__ import annotations

from pathlib import Path

from experiments.phase_iiir_replication.data import historical_exclusions, _block_rows
from weightbridge.core import file_hash
from .integrity import ROOT, read

OLDER_ROOTS = (
    'wb-iiii-v1-trial', 'wb-iiij-v1-trial', 'wb-iiik-v1-trial',
    'wb-iiil-semantic-v1', 'wb-iiim-factorial-v1', 'wb-iiin-paired-v1',
    'wb-iiio-joint-v1', 'wb-iiip-optimized-v1', 'wb-iiip-reference-v1',
    'weightbridge-iiih-private-e84230cf-0ba1-4e79-98b2-f83cd597127c',
)
R_ALLOWED = ('train', 'replay', 'development', 'smoke-r8', 'smoke-r4')


def audit(temp_root, calibration_decimal):
    """Return exclusions, not training labels. Never recurse into either R bank."""
    temp_root = Path(temp_root)
    q = temp_root/'wb-iiiq-baseline/baseline-v1'
    q_hidden = temp_root/'wb-iiiq-baseline/baseline-v1-host/hidden.json'
    r = temp_root/'wb-iiir-replication/replication-v1'
    public_r = ROOT/'results/phase-iii-r-replication/outcome-v1'
    public_manifest = public_r/'material/manifest.json'
    exported = read(public_r/'export-manifest.json')['artifacts']['material/manifest.json']
    if (file_hash(public_manifest) != exported['public_sha256']
            or file_hash(r/'material/manifest.json') != exported['source_sha256']
            or exported['mode'] != 'exact_public_scope'
            or exported['public_sha256'] != exported['source_sha256']):
        raise ValueError('R historical commitment differs from its published bytes')
    manifest = read(public_manifest)
    additional = []
    for split in R_ALLOWED:
        path = r/'material'/f'{split}.json'
        if file_hash(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Allowed R fitting/development roster changed')
        additional.append(path)
    roots = [temp_root/name for name in OLDER_ROOTS]
    available, missing = [p for p in roots if p.is_dir()], [p.name for p in roots if not p.is_dir()]
    used, anchors, receipt = historical_exclusions(q, q_hidden, additional_rosters=additional,
                                                  historical_roots=available, published_root=ROOT)
    before = len(used)
    _block_rows(read(calibration_decimal), used, anchors)
    receipt = {**receipt, 'schema': 'weightbridge.phase_iiis.historical_exclusion.v1',
        'r_manifest_sha256': file_hash(public_manifest), 'r_allowed_splits': list(R_ALLOWED),
        's_calibration_decimal_sha256': file_hash(calibration_decimal),
        's_calibration_new_task_identities': len(used)-before,
        'excluded_task_identities': len(used), 'excluded_negative_anchors': len(anchors),
        'missing_declared_older_roots': missing,
        'r_bank_a_raw_access': False, 'r_bank_b_raw_access': False,
        'coverage_limitation': 'The locked R 832-case battery and unopened R Bank B are not read for freshness. No disjointness claim against their inaccessible numeric identities; their published hash commitments remain intact.'}
    return used, anchors, receipt
