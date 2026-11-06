"""Private post-freeze host receipt glue. No training or model execution.

Uses frozen S gates and unchanged Q/R validators. The pinned source hash in every
request is checked before evidence access. It does not change any scientific gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

S = Path('C:/Users/jacks/AppData/Local/Temp/wb-phase-iii-s-tool-transfer')
R = Path('C:/Users/jacks/AppData/Local/Temp/wb-phase-iii-r-replication')
SELF = Path(__file__).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def commit(paths):
    return {key: {'path': str(Path(path).resolve()), 'sha256': digest(path)} for key, path in paths.items()}


def check_hashes(values):
    if any(digest(v['path']) != v['sha256'] for v in values.values()):
        raise ValueError('Host evidence commitment changed')


def s_imports():
    sys.path.insert(0, str(S))
    from experiments.phase_iiis_tool_transfer import study, verification
    return study, verification


def protected_precheck(req, stage):
    study, _ = s_imports()
    lock, _ = study.validate_prerequisites(req['protocol'], stage, req['lock'], req['prerequisites'])
    return lock


def checked_spec(spec, protocol, checkpoint):
    study, verification = s_imports()
    if set(spec) != {'path', 'rows', 'authorization', 'protocol', 'stage', 'execution'}:
        raise ValueError('Incomplete measured worker specification')
    p, wp = read(protocol), read(spec['protocol'])
    if any(p[k] != wp[k] for k in ('source_sha256', 'task_manifest_sha256', 'endpoints', 'tool_completion')):
        raise ValueError('Replay differs from the frozen source, population or endpoint')
    task = read(spec['path'])['task']
    result = verification.validate_worker(spec['path'], spec['rows'], spec['protocol'], checkpoint, None,
        spec['stage'], task, authorization=spec['authorization'])
    study.validate_execution(spec['execution'], module='experiments.phase_iiis_tool_transfer.evaluate',
        out=spec['path'], protocol_path=spec['protocol'])
    return result['measurement']


def reconstruction(req):
    study, verification = s_imports()
    from weightbridge.inspection.safetensors import load_checkpoint
    p = study._protocol(req['protocol'])
    gate = study.replay_stage(req['merged_development'])
    if not gate['finalist_eligible'] or gate['stage'] != 'merged-development':
        raise ValueError('Only the exact development-qualified stored step128 may advance')
    candidate = gate['representation_sha256']
    raw, rebuilt = read(req['merge_receipt']), read(req['reconstruction_receipt'])
    mv, rv = read(req['merge_verification']), read(req['reconstruction_verification'])
    paths = {k: req[k] for k in ('protocol', 'adapter', 'merged_development', 'merge_receipt', 'merge_verification',
        'reconstruction_receipt', 'reconstruction_verification', 'merge_execution', 'reconstruction_execution')}
    checks = {'numerical_receipts_equal': raw == rebuilt,
        'raw_parent_candidate': raw.get('base_sha256') == p['checkpoint_sha256'] and raw.get('candidate_sha256') == candidate,
        'raw_parent_unchanged': raw.get('base_unchanged') is True,
        'raw_no_promotion': raw.get('lineage_accepted') is False and raw.get('production_deployment') is False,
        'exact_reconstruction': rv.get('exact_reconstruction') is True,
        'stored_candidate': load_checkpoint(req['candidate_directory']).checkpoint_sha256() == candidate,
        'stored_rebuild': load_checkpoint(req['reconstructed_directory']).checkpoint_sha256() == candidate,
        'original_restore': load_checkpoint(req['original_directory']).checkpoint_sha256() == study.ORIGINAL,
        'actual_parent_restore': load_checkpoint(req['actual_parent_directory']).checkpoint_sha256() == p['checkpoint_sha256']}
    for label, value, raw_path, worker in (('merge', mv, req['merge_receipt'], req['merge_execution']),
            ('reconstruction', rv, req['reconstruction_receipt'], req['reconstruction_execution'])):
        for key, expected in {'schema': 'weightbridge.phase_iiis.merge_verification.v1', 'passed': True,
                'protocol_sha256': digest(req['protocol']), 'task_manifest_sha256': p['task_manifest_sha256'],
                'source_sha256': p['source_sha256'], 'candidate_sha256': candidate,
                'parent_checkpoint_sha256': p['checkpoint_sha256'], 'adapter_sha256': digest(req['adapter']),
                'merge_receipt_sha256': digest(raw_path)}.items():
            checks[f'{label}/{key}'] = value.get(key) == expected
        out = req['candidate_directory'] if label == 'merge' else req['reconstructed_directory']
        process = study.validate_execution(worker, module='experiments.phase_iiis_tool_transfer.merge', out=out, protocol_path=req['protocol'])
        command = process['command']
        for flag, expected in (('--adapter', req['adapter']), ('--receipt', raw_path), ('--original', req['original_directory'])):
            checks[f'{label}/{flag}'] = Path(study._command_value(command, flag)).resolve() == Path(expected).resolve()
        checks[f'{label}/expected_adapter'] = study._command_value(command, '--expected-adapter') == digest(req['adapter'])
        checks[f'{label}/expected_parent'] = study._command_value(command, '--expected-parent') == p['checkpoint_sha256']
        checks[f'{label}/actual_source_bytes'] = load_checkpoint(study._command_value(command, '--base')).checkpoint_sha256() == p['checkpoint_sha256']
        authorization = study._command_value(command, '--authorization')
        from experiments.phase_iiis_tool_transfer.merge import validate_merge_authorization
        validate_merge_authorization(req['protocol'], read(authorization), digest(req['adapter']), p['checkpoint_sha256'])
        checks[f'{label}/raw_authorization'] = read(raw_path).get('authorization_sha256') == digest(authorization)
        paths[f'{label}/authorization'] = authorization
        if label == 'reconstruction':
            checks['reconstruction/expected_sha'] = study._command_value(command, '--expected-reconstruction') == candidate
    if p['arm'] == 'sequential':
        # The R checkpoint must itself be a checked rounded original->R rebuild.
        # Its historical source is immutable; use recorded R numerical receipts,
        # never a sum of R and S deltas followed by one rounding operation.
        prior = read(req['r_reconstruction_receipt'])
        checks['sequential/rounded_r_rebuild'] = (prior.get('base_sha256') == study.ORIGINAL
            and prior.get('candidate_sha256') == study.DECIMAL_PARENT and prior.get('base_unchanged') is True
            and load_checkpoint(req['r_reconstruction_directory']).checkpoint_sha256() == study.DECIMAL_PARENT)
        rebuilt_command = read(req['reconstruction_execution'])['command']
        checks['sequential/s_rebuild_uses_rounded_r_rebuild'] = Path(study._command_value(rebuilt_command, '--base')).resolve() == Path(req['r_reconstruction_directory']).resolve()
        paths['r_reconstruction_receipt'] = req['r_reconstruction_receipt']
    expected = {f'{role}/{task}' for role in ('candidate', 'parent') for task in ('tool', 'decimal')}
    if set(req['replays']) != expected:
        raise ValueError('Both complete development endpoints require reconstructed and parent rollback replays')
    exact = {}
    own = 'r8' if p['arm'] == 'sequential' else 'original'
    for key, pair in req['replays'].items():
        role, task = key.split('/')
        if set(pair) != {'recorded', 'replayed'} or pair['recorded'] != gate['workers']['candidate' if role == 'candidate' else own][task]:
            raise ValueError('Replay reference is not the actually measured merged-development record')
        a, b = pair['recorded'], pair['replayed']
        if (Path(a['path']).resolve() == Path(b['path']).resolve() or Path(a['execution']).resolve() == Path(b['execution']).resolve()
                or digest(a['rows']) != digest(b['rows']) or b['stage'] != 'reconstruction-development'):
            raise ValueError('Development replay is not a fresh process on its own exact roster')
        sha = candidate if role == 'candidate' else p['checkpoint_sha256']
        exact[key] = verification.exact_native_replay(checked_spec(a, req['protocol'], sha), checked_spec(b, req['protocol'], sha))
        for kind, spec in pair.items():
            paths.update({f'{key}/{kind}/{k}': spec[k] for k in ('path', 'rows', 'authorization', 'protocol', 'execution')})
    checks['exact_development_and_parent_replays'] = all(exact.values())
    return {'schema': 'weightbridge.phase_iiis.host_exact_reconstruction.v1', 'passed': all(checks.values()),
        'protocol_sha256': digest(req['protocol']), 'task_manifest_sha256': p['task_manifest_sha256'],
        'representation_sha256': candidate, 'candidate_sha256': candidate, 'parent_checkpoint_sha256': p['checkpoint_sha256'],
        'exact_reconstruction': checks['exact_reconstruction'] and checks['stored_rebuild'] and checks['numerical_receipts_equal'],
        'checks': checks, 'exact_native_replays': exact, 'input_sha256': commit(paths),
        'production_authority': False, 'lineage_authority': False}


def historical_release(req, request_path):
    # This process exclusively uses the pinned R checkout. A separate S process
    # must replay both locks before this process opens any historical roster.
    subprocess.run([sys.executable, str(SELF), '--request', str(request_path), '--check-only'], cwd=S, check=True)
    sys.path.insert(0, str(R))
    from experiments.phase_iiir_replication.study import validate_replication_gate
    from experiments.phase_iiiq_baseline import verify_final as qv
    from experiments.phase_iiiq_baseline.integrity import verify_sources
    root = Path(req['historical_root']).resolve()
    expected = Path('C:/Users/jacks/AppData/Local/Temp/wb-iiir-replication/replication-v1').resolve()
    if root != expected:
        raise ValueError('Historical release is restricted to the recorded R rank8 Bank A')
    local = root/'rank8'
    pp, assessment = local/'protocol.json', local/'assessment'
    protocol = read(pp)
    verified = validate_replication_gate(protocol, root/'replication-verification.json')
    verify_sources(protocol['source_sha256'])
    independent = read(assessment/'independent.json')
    qv._qualified_independent(independent)
    identity = {'protocol_sha256': digest(pp), 'manifest_sha256': digest(local/'material/manifest.json'),
                'representation_sha256': verified['representation_sha256']}
    artifacts = {'historical_protocol': pp, 'historical_manifest': local/'material/manifest.json',
        'historical_replication': root/'replication-verification.json', 'historical_independent': assessment/'independent.json'}
    _, _, measured = qv._prior_evidence(assessment, local/'material', pp, protocol, independent, identity, artifacts)
    stages = {}
    count = 0
    for stage, (roster, auth, original, r8) in measured.items():
        if stage not in ('preservation-validation', 'selection', 'hidden'):
            raise ValueError('Unexpected historical bank')
        count += len(original['measurement']['rows'])
        stages[stage] = {'rows': str(roster), 'authorization': str(auth), 'protocol': str(pp),
            'original': str(assessment/'evaluation'/f'{stage}-parent.json'),
            'r8': str(assessment/'evaluation'/f'{stage}-merged.json'), 'roster_sha256': digest(roster),
            'presentations': len(original['measurement']['rows'])}
    if count != 832 or set(stages) != set(qv.STAGES):
        raise ValueError('Historical battery differs from the complete original R 832 cases')
    artifacts.update({f'historical_source/{name}': R/name for name in protocol['source_sha256']})
    artifacts.update({f's-prerequisite/{name}': path for name, path in req['prerequisites'].items()})
    artifacts.update(s_lock=req['lock'], s_protocol=req['protocol'])
    return {'schema': 'weightbridge.phase_iiis.historical_release.v1', 'passed': True, 'presentations': count,
        'stages': stages, 'input_sha256': commit(artifacts), 'source_checkout': str(R),
        'r_bank_b_access': False, 'model_execution': False,
        'scope': 'Replayed historical rank8 Bank A authorizations and scored records; locked regression evidence, not fresh transfer.'}


def historical_gate(req):
    lock = protected_precheck(req, 'historical-retention')
    study, verification = s_imports()
    from experiments.phase_iiiq_baseline.study import _validated_worker
    from experiments.phase_iiis_tool_transfer.measurement import legacy_retention
    release = read(req['release'])
    if (release.get('schema') != 'weightbridge.phase_iiis.historical_release.v1' or release.get('passed') is not True
            or release.get('presentations') != 832 or release.get('r_bank_b_access') is not False):
        raise ValueError('Historical rosters require a checked, complete rank8 release')
    check_hashes(release['input_sha256'])
    protocol = read(req['protocol'])
    if set(req['workers']) != set(release['stages']):
        raise ValueError('All three original R stages must be retained')
    paths = {'release': req['release'], 'protocol': req['protocol'], 'lock': req['lock'], **req['prerequisites']}
    gates = {}
    for stage, old in release['stages'].items():
        values = {}
        for role, checkpoint in (('original', study.ORIGINAL), ('r8', study.DECIMAL_PARENT)):
            values[role] = _validated_worker(old[role], old['rows'], old['protocol'], checkpoint, None, stage)['measurement']
        bundle = req['workers'][stage]
        if set(bundle) != {'candidate', 'reconstructed', 'parent'}:
            raise ValueError('Historical candidate and exact own replay/parent rollback are required')
        for spec in bundle.values():
            if spec['stage'] != 'historical-retention' or digest(spec['rows']) != old['roster_sha256']:
                raise ValueError('Historical worker used another stage or changed ordering/batching roster')
        candidate = checked_spec(bundle['candidate'], req['protocol'], lock['representation_sha256'])
        rebuilt = checked_spec(bundle['reconstructed'], req['protocol'], lock['representation_sha256'])
        parent = checked_spec(bundle['parent'], req['protocol'], protocol['checkpoint_sha256'])
        for role in ('reconstructed', 'parent'):
            if (Path(bundle['candidate']['path']).resolve() == Path(bundle[role]['path']).resolve()
                    or Path(bundle['candidate']['execution']).resolve() == Path(bundle[role]['execution']).resolve()):
                raise ValueError('Historical replays require distinct process receipts')
        own = values['r8' if protocol['arm'] == 'sequential' else 'original']
        refs = [own]
        if protocol['arm'] == 'sequential':
            general = {**values['original'], 'rows': [r for r in values['original']['rows'] if r['group'] != 'target']}
            if general['rows']:
                current = {**candidate, 'rows': [r for r in candidate['rows'] if r['group'] != 'target']}
                extra = legacy_retention([general], current)
            else:
                extra = None
        else:
            extra = None
        retention = legacy_retention(refs, candidate)
        native = {'candidate': verification.exact_native_replay(candidate, rebuilt),
                  'parent': verification.exact_native_replay(own, parent)}
        passed = retention['passed'] and (extra is None or extra['passed']) and all(native.values())
        gates[stage] = {'passed': passed, 'retention': retention, 'general_union': extra, 'exact_native_replays': native}
        for role, spec in bundle.items():
            paths.update({f'{stage}/{role}/{k}': spec[k] for k in ('path', 'rows', 'authorization', 'protocol', 'execution')})
    return {'schema': 'weightbridge.phase_iiis.host_historical_retention.v1', 'passed': all(g['passed'] for g in gates.values()),
        'protocol_sha256': digest(req['protocol']), 'task_manifest_sha256': protocol['task_manifest_sha256'],
        'representation_sha256': lock['representation_sha256'], 'presentations': 832,
        'gates': gates, 'input_sha256': commit(paths), 'r_bank_b_access': False,
        'scope': 'Locked R battery retention and own-output replay, not fresh generalization.',
        'production_authority': False, 'lineage_authority': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    req = read(args.request)
    if req.get('schema') != 'weightbridge.phase_iiis.host_bridge_request.v1' or req.get('bridge_sha256') != digest(SELF):
        raise ValueError('Host bridge request must pin these reviewed bytes before any stage access')
    action = req['action']
    if args.check_only:
        if action != 'release-historical':
            raise ValueError('Check-only is restricted to the historical pre-access gate')
        protected_precheck(req, 'historical-retention')
        print(json.dumps({'passed': True, 'protected_rows_read': False}))
        return
    if Path(req['out']).exists():
        raise FileExistsError('Host bridge receipts are write-once')
    wall, cpu = time.perf_counter(), time.process_time()
    result = {'exact-reconstruction': reconstruction, 'release-historical': lambda r: historical_release(r, args.request),
              'historical-retention': historical_gate}[action](req)
    result.update(bridge_sha256=digest(SELF), bridge_request_sha256=digest(args.request),
        bridge_request_path=str(args.request.resolve()), model_execution_by_bridge=False,
        resources={'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu,
                   'peak_memory_bytes': None, 'scope': 'Host evidence checking only; child source-check CPU for release is not measured here.'})
    # Request and source are included in every receipt's tamper-evident inputs.
    result['input_sha256'].update(commit({'host_bridge_source': SELF, 'host_bridge_request': args.request}))
    Path(req['out']).parent.mkdir(parents=True, exist_ok=True)
    with Path(req['out']).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'passed': result['passed'], 'schema': result['schema'], 'receipt_sha256': digest(req['out'])}))


if __name__ == '__main__':
    main()
