"""Public-only, clean-checkout audit; never executes inference or a host driver.

Run in a fresh checkout at the exact published commit. Receipt must be outside
the checkout. --self-test checks only synthetic public-artifact refusal cases.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = Path(__file__).resolve().parent
BASELINE = 'baa336e7c517139876a1a5d6309af7199ea1e661'
DEMO = 'results/phase-iii-s-tool-transfer/public-demo-v1/rows.json'
DEMO_MANIFEST = 'results/phase-iii-s-tool-transfer/public-demo-v1/manifest.json'
EXPORTERS = {
    'weightbridge.phase_iiis.public_preflight.v1': 'export_preflight.py',
    'weightbridge.phase_iiis.public_preregistration.v1': 'export_preregistration.py',
    'weightbridge.phase_iiis.public_outcome.v1': 'export_outcome.py',
}
RAW_KEYS = {'rows', 'records', 'outputs', 'prompt', 'text', 'answer', 'completion',
            'messages', 'fixture', 'request', 'tokenization', 'generated_token_ids', 'prefix_token_ids', 'target_ids',
            'input_ids', 'gold_action', 'parsed_action', 'oracle_answer',
            'operands', 'optimizer_state', 'state_dict', 'tensor_values'}
WEIGHT_SUFFIXES = {'.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.npz', '.npy', '.pkl', '.pickle'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounded(root, name):
    posix = PurePosixPath(str(name).replace('\\', '/'))
    if not posix.parts or posix.is_absolute() or '..' in posix.parts or ':' in str(posix):
        raise ValueError('Artifact path must be a bounded relative name')
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes public directory')
    return path


def private_path(value):
    return bool(re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[/\\\\]|\\\\\\\\[A-Za-z0-9]', value))


def raw_check(value, *, declared_seed_metadata=False):
    if isinstance(value, dict):
        if any(key in RAW_KEYS and not (key == 'rows' and type(item) is int and item >= 0)
               for key, item in value.items()):
            raise ValueError('Raw private-payload field in public JSON')
        for key, item in value.items():
            if private_path(key): raise ValueError('Private path in a JSON key')
            if 'seed' in key.lower() and not declared_seed_metadata:
                commitment = key == 'seed_commitment' and isinstance(item, str) and re.fullmatch('[0-9a-f]{64}', item)
                notice = key in {'publication_includes_private_seeds_rosters_or_outputs',
                    'private_requests_or_rosters_or_seeds_or_outputs_published', 'private_seed_values_recorded'} and item is False
                if not commitment and not notice: raise ValueError('Undeclared seed field in public JSON')
            raw_check(item, declared_seed_metadata=declared_seed_metadata)
    elif isinstance(value, list):
        for item in value: raw_check(item, declared_seed_metadata=declared_seed_metadata)
    elif isinstance(value, str) and private_path(value):
        raise ValueError('Private local path in public JSON')


def artifact_check(name, payload=None, *, declared_seed_metadata=False):
    path = PurePosixPath(name)
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        raise ValueError('New numerical model artifact is not public evidence')
    if path.suffix.lower() == '.json' and name not in (DEMO, DEMO_MANIFEST):
        lowered = path.name.lower()
        if any(word in lowered for word in ('training-bundle', 'smoke-bundle', 'private-roster', 'host-request', 'optimizer-state')):
            raise ValueError('Private execution artifact cannot be published')
        if payload is not None: raw_check(json.loads(payload), declared_seed_metadata=declared_seed_metadata)


def load(name):
    spec = importlib.util.spec_from_file_location('public_audit_'+name.removesuffix('.py'), PUBLIC/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.STDOUT).decode('utf-8').strip()


def audit(commit, output, *, required_exports=(), observe_remote=False):
    wall, cpu = time.perf_counter(), time.process_time()
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An exact full published commit is required')
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError('Use a new receipt path outside the clean checkout')
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Checkout must be clean at the requested commit')
    tracked = set(git('ls-files', '-z').split('\0'))
    tracked.discard('')
    modules = {name: load(name) for name in set(EXPORTERS.values())}
    validators, exports, hashed, declared_seed_files = {}, {}, {}, set()
    def checked(path, expected):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in tracked:
            raise ValueError('Untracked public input: '+relative)
        if relative not in hashed: hashed[relative] = sha(path)
        actual = hashed[relative]
        if actual != expected:
            raise ValueError('Public byte mismatch: '+relative)
        return actual
    for name, module in modules.items():
        validators[name] = {'sha256': sha(PUBLIC/name), 'self_test': module.self_test()}
    validators['final_publication.py'] = {'sha256': sha(PUBLIC/'final_publication.py'),
                                         'self_test': load('final_publication.py').self_test()}
    for manifest_path in sorted(PUBLIC.rglob('export-manifest.json')):
        manifest = json.loads(manifest_path.read_bytes())
        name = EXPORTERS.get(manifest.get('schema'))
        if name is None:
            raise ValueError('Unknown public export schema: '+str(manifest_path.relative_to(PUBLIC)))
        result = modules[name].verify(manifest_path.parent)
        if result.get('passed') is not True:
            raise ValueError('Public validator did not pass')
        exports[manifest_path.parent.relative_to(PUBLIC).as_posix()] = {
            **result, 'export_manifest_sha256': sha(manifest_path), 'validator': name}
        if name == 'export_preregistration.py':
            declared_seed_files.add(manifest_path.relative_to(ROOT).as_posix())
            declared_seed_files.update(bounded(manifest_path.parent, filename).relative_to(ROOT).as_posix()
                                       for filename in manifest['public_bytes_sha256'])
    if not set(required_exports).issubset(exports):
        raise ValueError('Requested final export is absent')
    source_path = PUBLIC/'preregistration-v1/source-manifest.json'
    scientific = json.loads(source_path.read_bytes())
    if len(scientific) != 162:
        raise ValueError('Unexpected scientific manifest size')
    endings = {'lf_only': 0, 'crlf_only': 0, 'mixed': 0, 'no_newline': 0}
    for name, expected in scientific.items():
        path = bounded(ROOT, name); checked(path, expected)
        content = path.read_bytes(); crlf = b'\r\n' in content; lf = b'\n' in content.replace(b'\r\n', b'')
        endings['mixed' if crlf and lf else 'crlf_only' if crlf else 'lf_only' if lf else 'no_newline'] += 1
    for path in PUBLIC.rglob('source-manifest.json'):
        if json.loads(path.read_bytes()) != scientific:
            raise ValueError('Public snapshots disagree about scientific sources')
    calibration = {}
    for arm in ('sequential', 'original-joint'):
        path = PUBLIC/'preflight-v1'/f'{arm}-calibration-protocol.json'
        mapping = json.loads(path.read_bytes())['source_sha256']
        if len(mapping) != 151: raise ValueError('Unexpected calibration manifest size')
        for name, expected in mapping.items(): checked(bounded(ROOT, name), expected)
        calibration[arm] = {'source_files': len(mapping), 'protocol_sha256': sha(path)}
    packages = {}
    for relative in ('host_tools', 'host_tools/protected-v1', 'host_tools/protected-v2', 'host_tools/protected-v3'):
        folder = PUBLIC/relative; manifest = json.loads((folder/'manifest.json').read_bytes())
        for name, metadata in manifest['files'].items():
            path = bounded(folder, name); checked(path, metadata['sha256'])
            if path.stat().st_size != metadata['bytes']: raise ValueError('Supplemental byte count mismatch')
        packages[relative] = {'files': len(manifest['files']), 'manifest_sha256': sha(folder/'manifest.json')}
    recorded = json.loads((PUBLIC/'host_tools/protected-v3/selftest-receipt.json').read_bytes())
    if (recorded.get('passed') is not True or len(recorded['tests']) != 26
            or not all(recorded['tests'].values()) or recorded['model_calls'] != 0 or recorded['protected_roster_reads'] != 0
            or recorded['source_sha256'] != sha(PUBLIC/'host_tools/protected-v3/protected_host_v3.py')
            or recorded['selftest_sha256'] != sha(PUBLIC/'host_tools/protected-v3/protected_host_selftest_v3.py')):
        raise ValueError('Supplemental synthetic receipt/source binding mismatch')
    packages['host_tools/protected-v3']['recorded_checks_not_rerun'] = 26
    demo = load('demo.py').verify(PUBLIC/'public-demo-v1')
    figures = {}
    for path in PUBLIC.rglob('manifest.json'):
        manifest = json.loads(path.read_bytes())
        if manifest.get('schema') != 'weightbridge.phase_iiis.public_figures.v1': continue
        for name, expected in manifest['artifacts'].items(): checked(bounded(path.parent, name), expected)
        snapshot = bounded(PUBLIC, manifest['source_snapshot'])
        if snapshot.relative_to(PUBLIC).as_posix() not in exports: raise ValueError('Figure source is not a verified public export')
        for name, expected in manifest['input_sha256'].items(): checked(bounded(snapshot, name), expected)
        if manifest['script_sha256'] != sha(PUBLIC/'figures.py'):
            raise ValueError('Figure renderer differs from its recorded source; an explicit historical-source audit is required')
        figures[path.parent.relative_to(PUBLIC).as_posix()] = {'manifest_sha256': sha(path),
            'artifacts': len(manifest['artifacts']), 'input_count': len(manifest['input_sha256']),
            'recorded_renderer_sha256': manifest['script_sha256'], 'rendered_again': False}
    # All pre-existing code and research evidence stay byte-identical to R.
    historical = [line for line in git('diff', '--name-only', '--diff-filter=DMRT', BASELINE, commit).splitlines()
                  if line.startswith(('weightbridge/', 'experiments/', 'tests/', 'results/'))
                  or line in ('docs/PHASE_III_Q_BASELINE.md', 'docs/PHASE_III_R_REPLICATION.md')]
    if historical: raise ValueError('Historical source/evidence changed: '+', '.join(historical))
    changed = git('diff', '--name-only', BASELINE, commit).splitlines()
    for name in changed:
        path = bounded(ROOT, name)
        artifact_check(name, path.read_bytes() if path.suffix.lower() == '.json' else None,
                       declared_seed_metadata=name in declared_seed_files)
    remote = None
    if observe_remote:
        lines = git('ls-remote', '--heads', 'origin', 'dev/phase-iii-s-tool-transfer').splitlines()
        if len(lines) != 1: raise ValueError('Remote branch observation is ambiguous')
        remote = lines[0].split()[0]
    if git('status', '--porcelain', '--untracked-files=all') or git('rev-parse', 'HEAD') != commit:
        raise ValueError('Checkout changed during public validation')
    result = {'schema': 'weightbridge.phase_iiis.clean_public_checkout_review.v3', 'passed': True,
        'published_commit': commit, 'baseline_commit': BASELINE, 'git_clean_before_and_after': True,
        'remote_sha_observed': remote, 'remote_matches_reviewed_commit': remote == commit if remote else None,
        'validators': validators, 'public_exports': exports, 'public_export_count': len(exports),
        'public_artifacts_verified': sum(record['artifacts'] for record in exports.values()),
        'source_manifest_sha256': sha(source_path), 'scientific_source_files_verified': len(scientific),
        'calibration_manifests': calibration, 'source_line_endings': endings, 'supplemental_packages': packages,
        'public_demo': demo, 'public_figures': figures, 'historical_sources_and_evidence_unchanged': True,
        'changed_tracked_files_screened': len(changed), 'private_or_model_artifacts_found': False,
        'model_execution': False, 'protected_roster_access': False, 'scientific_outcomes_reverified': False,
        'audit_script_sha256': sha(__file__),
        'resources': {'audit_wall_seconds': time.perf_counter()-wall, 'audit_cpu_seconds': time.process_time()-cpu,
                      'audit_peak_memory_bytes': None, 'git_child_cpu_seconds': None, 'clone_cpu_seconds': None,
                      'clone_wall_seconds': None, 'scope': 'Public host Python audit only; Git/network child CPU and prior clone costs UNMEASURED; no model work.'},
        'scope': 'Public bytes, redaction checks and deterministic fixture validation. Does not reproduce private behavior or confer checkpoint admission.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    return {'passed': True, 'receipt_sha256': sha(output), 'public_exports': len(exports), 'model_execution': False}


def self_test():
    refused = 0
    for action in (lambda: bounded(Path('.'), '../private.json'), lambda: bounded(Path('.'), 'C:/private.json'),
                   lambda: artifact_check('new/adapter.safetensors'),
                   lambda: artifact_check('new/training-bundle.json', b'{}'),
                   lambda: artifact_check('new/result.json', b'{"prompt":"private"}'),
                   lambda: artifact_check('new/result.json', b'{"nested":[{"generated_token_ids":[1]}]}'),
                   lambda: artifact_check('new/result.json', b'{"location":"C:/private/bank.json"}'),
                   lambda: artifact_check('new/result.json', b'{"location":"found at C:/private/bank.json"}'),
                   lambda: artifact_check('new/result.json', b'{"C:/private/bank.json":"hash"}'),
                   lambda: artifact_check('new/result.json', b'{"dataset_seed":123}')):
        try: action()
        except ValueError: refused += 1
        else: raise AssertionError('Unsafe synthetic fixture admitted')
    artifact_check(DEMO, b'[{"prompt":"explicit illustrative demo"}]')
    artifact_check('new/summary.json', b'{"unopened":null,"status":"UNMEASURED"}')
    artifact_check('new/summary.json', b'{"rows":40}')
    artifact_check('new/summary.json', b'{"training":{"seed":606118}}', declared_seed_metadata=True)
    with tempfile.TemporaryDirectory() as directory:
        assert bounded(Path(directory), 'public.json').parent == Path(directory).resolve()
    return {'passed': True, 'refusal_checks': refused, 'public_demo_exception': True,
            'null_preserved': True, 'model_execution': False, 'protected_roster_access': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit'); parser.add_argument('--out', type=Path)
    parser.add_argument('--require-export', action='append', default=[])
    parser.add_argument('--observe-remote', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    options = parser.parse_args()
    if options.self_test: result = self_test()
    elif options.commit and options.out:
        result = audit(options.commit, options.out, required_exports=options.require_export, observe_remote=options.observe_remote)
    else: parser.error('Use --self-test or --commit and --out')
    print(json.dumps(result, indent=2, sort_keys=True))
