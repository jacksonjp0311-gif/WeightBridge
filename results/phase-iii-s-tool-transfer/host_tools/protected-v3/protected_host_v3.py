"""Prospective private Phase S protected-stage orchestration v3; explicit actions only.

Every action replays both arm decisions and its own prior gates before any bank
is read. All model work uses frozen S evaluation and the shared bounded ledger.
No fitting, alternate finalist selection, redraw, checkpoint editing or promotion.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

S = Path('C:/Users/jacks/AppData/Local/Temp/wb-phase-iii-s-tool-transfer')
PRIVATE = Path('C:/Users/jacks/AppData/Local/Temp/wb-iiis-tool-transfer')
BRIDGE = PRIVATE/'host_stage_bridge_v1.py'
BRIDGE_SHA = '258ee4e5b20992895588f072f6e0d42659e73cb847b06398f07e6104d5ddf67c'
SELF = Path(__file__).resolve()
sys.path.insert(0, str(S))

from experiments.phase_iiis_tool_transfer import execute, host, study
from experiments.phase_iiis_tool_transfer.integrity import ORIGINAL, DECIMAL_PARENT, PARENTS, read, write_once
from experiments.phase_iiis_tool_transfer.verification import authorize_evaluation, validate_worker, exact_native_replay
from weightbridge.core import file_hash


def commitment(paths):
    return {key: {'path': str(Path(path).resolve()), 'sha256': file_hash(path)} for key, path in paths.items()}


def check_base(spec, expected):
    actual = study._command_value(read(spec['execution'])['command'], '--base')
    if Path(actual).resolve() != Path(expected).resolve():
        raise ValueError('Worker did not load the exact declared physical checkpoint directory')
    return {'passed': True, 'execution_sha256': file_hash(spec['execution']), 'base': str(Path(expected).resolve())}


def context(request, *, required_stage=None):
    root, _, manifest = host._initialized(request['study'])
    arm = request['arm']
    if root != (PRIVATE/'study-v1').resolve() or arm not in PARENTS:
        raise ValueError('Only the existing two-arm frozen Phase S study is supported')
    ledger = Path(request['ledger']).resolve()
    if ledger != (PRIVATE/'execution').resolve() or file_hash(BRIDGE) != BRIDGE_SHA:
        raise ValueError('Shared ledger or reviewed host bridge identity differs')
    # Both decisions are checked before loading even a candidate-specific stage.
    both = root/'locks/both-arms.json'
    study.validate_arm_locks(both)
    local, pp = root/arm, root/arm/'protocol.json'
    if any((local/f'protected-v{v}').exists() for v in (1,2)):
        raise ValueError('An earlier protected attempt exists; v3 cannot replace or retry it')
    protocol = study._protocol(pp)
    lockpath = root/'locks'/f'{arm}.json'
    lock = study.validate_arm_lock(lockpath)
    if lock['status'] != 'FINALIST_LOCKED' or not lock['passed']:
        raise ValueError('A terminal arm cannot consume protected evaluation')
    post = local/'post-development-v1'
    rebuild_request = read(post/'reconstruction-host-request.json')
    rebuild_gate = read(post/'exact-reconstruction-gate.json')
    study._assert_hashes(rebuild_gate['input_sha256'])
    if (rebuild_gate['candidate_sha256'] != lock['representation_sha256']
            or rebuild_gate['passed'] is not True or rebuild_gate['exact_reconstruction'] is not True):
        raise ValueError('Locked candidate lacks its completed pre-bank reconstruction and replay')
    prerequisites = {'merged-development': str(post/'merged-development-gate.json'),
        'exact-reconstruction': str(post/'exact-reconstruction-gate.json'), 'both-arms-locked': str(both)}
    if required_stage in ('qualification', 'confirmation'):
        prerequisites['historical-retention'] = str(local/'protected-v3/historical-retention/gate.json')
    if required_stage == 'confirmation':
        prerequisites['qualification'] = str(local/'protected-v3/qualification/gate.json')
    if required_stage is not None:
        study.validate_prerequisites(pp, required_stage, lockpath, prerequisites)
    return {'root': root, 'local': local, 'protocol': protocol, 'protocol_path': pp, 'manifest': manifest,
        'ledger': ledger, 'lock': lock, 'lock_path': lockpath, 'prerequisites': prerequisites,
        'original': Path(request['original']).resolve(), 'r8': Path(request['decimal_parent']).resolve(),
        'candidate': Path(rebuild_request['candidate_directory']).resolve(),
        'reconstructed': Path(rebuild_request['reconstructed_directory']).resolve(),
        'actual_parent': Path(rebuild_request['actual_parent_directory']).resolve(),
        'reconstruction_gate': post/'exact-reconstruction-gate.json'}


def check_worker(ctx, spec, expected_sha, expected_base, task, stage):
    if spec['stage'] != stage:
        raise ValueError('Cached worker belongs to a different protected bank')
    wp = read(spec['protocol'])
    if any(wp.get(k) != ctx['protocol'].get(k) for k in ('source_sha256','task_manifest_sha256','endpoints','tool_completion')):
        raise ValueError('A cached parent changed the population, endpoint or scientific source')
    result = validate_worker(spec['path'], spec['rows'], spec['protocol'], expected_sha, None, stage, task,
        authorization=spec['authorization'])
    study.validate_execution(spec['execution'], module='experiments.phase_iiis_tool_transfer.evaluate',
        out=spec['path'], protocol_path=spec['protocol'])
    base = check_base(spec, expected_base)
    return result, base


def worker(ctx, *, model, sha, rows, row_sha, task, stage, name, directory, specifications,
           attempts, base_checks, worker_protocol=None):
    pp = Path(worker_protocol or ctx['protocol_path'])
    # Authorization for a cached-parent replay remains bound to its recorded
    # protocol and corresponding arm lock, even when another arm uses the cache.
    auth_arm = read(pp)['arm']
    auth_local = ctx['root']/auth_arm
    auth_lock = ctx['root']/'locks'/f'{auth_arm}.json'
    auth_prerequisites = {'merged-development': str(auth_local/'post-development-v1/merged-development-gate.json'),
        'exact-reconstruction': str(auth_local/'post-development-v1/exact-reconstruction-gate.json'),
        'both-arms-locked': str(ctx['root']/'locks/both-arms.json')}
    if stage in ('qualification','confirmation'):
        auth_prerequisites['historical-retention'] = str(auth_local/'protected-v3/historical-retention/gate.json')
    if stage == 'confirmation':
        auth_prerequisites['qualification'] = str(auth_local/'protected-v3/qualification/gate.json')
    if auth_arm != ctx['protocol']['arm'] and stage in ('qualification','confirmation'):
        owner=context({'study':str(ctx['root']),'arm':auth_arm,'ledger':str(ctx['ledger']),
            'original':str(ctx['original']),'decimal_parent':str(ctx['r8'])},required_stage=stage)
        history=historical_closure(owner)
        if not history['completed_valid'] or not history['passed']:
            raise ValueError('Cached-parent replay owner has no completed historical retention')
        if stage=='confirmation':
            qualified=formal_closure(owner,'qualification')
            if not qualified['completed_valid'] or not qualified['passed']:
                raise ValueError('Cached-parent replay owner has no completed qualification')
    study.validate_prerequisites(pp, stage, auth_lock, auth_prerequisites)
    if file_hash(rows) != row_sha:
        raise ValueError('Protected roster changed after its prospective commitment')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out, auth = directory/f'{task}.json', directory/f'{task}-authorization.json'
    if out.exists() or auth.exists():
        raise FileExistsError('No implicit protected-worker retry or record replacement')
    authorize_evaluation(auth, pp, stage=stage, task=task, roster_sha256=row_sha,
        representation_sha256=sha, finalist_lock_path=auth_lock, prerequisite_paths=auth_prerequisites)
    args = ['--base', str(model), '--protocol', str(pp), '--rows', str(rows), '--out', str(out),
        '--expected-checkpoint', sha, '--expected-rows', row_sha, '--stage', stage, '--task', task,
        '--authorization', str(auth)]
    limits = ctx['protocol']['resources']['evaluation_limits']
    process_path = ctx['ledger']/f'{name}.receipt.json'
    result = execute.run(ctx['ledger'], name, 'experiments.phase_iiis_tool_transfer.evaluate', args,
        wall_limit=limits['max_wall_seconds'], cpu_limit=limits['max_cpu_seconds'], memory_limit=limits['max_memory_bytes'])
    attempts.append({'path': str(process_path), 'sha256': file_hash(process_path), 'passed': result['passed']})
    if not result['passed']:
        raise RuntimeError('Protected worker or enclosing resource scope failed')
    spec = {'path': str(out), 'rows': str(rows), 'authorization': str(auth), 'protocol': str(pp),
        'stage': stage, 'execution': str(process_path)}
    _, base = check_worker(ctx, spec, sha, model, task, stage)
    base_checks.append(base)
    specifications.append(spec)
    return spec


def historical(ctx, out, attempts, bases, specs):
    release = ctx['root']/'historical-release-v3.json'
    if not release.exists():
        request_path = ctx['root']/'historical-release-request-v3.json'
        write_once(request_path, {'schema':'weightbridge.phase_iiis.host_bridge_request.v1',
            'bridge_sha256':BRIDGE_SHA,'action':'release-historical','out':str(release),
            'protocol':str(ctx['protocol_path']),'lock':str(ctx['lock_path']),
            'prerequisites':ctx['prerequisites'],
            'historical_root':'C:/Users/jacks/AppData/Local/Temp/wb-iiir-replication/replication-v1'})
        subprocess.run([sys.executable,str(BRIDGE),'--request',str(request_path)],cwd=S,check=True)
    released = read(release)
    if (released.get('passed') is not True or released.get('presentations') != 832
            or released.get('r_bank_b_access') is not False or released.get('bridge_sha256') != BRIDGE_SHA):
        raise ValueError('Historical source release did not validate the fixed R battery')
    study._assert_hashes(released['input_sha256'])
    workers = {}
    for old_stage in ('preservation-validation','selection','hidden'):
        entry = released['stages'][old_stage]
        workers[old_stage] = {}
        for role, model, sha in (('candidate',ctx['candidate'],ctx['lock']['representation_sha256']),
                ('reconstructed',ctx['reconstructed'],ctx['lock']['representation_sha256']),
                ('parent',ctx['actual_parent'],ctx['protocol']['checkpoint_sha256'])):
            workers[old_stage][role] = worker(ctx,model=model,sha=sha,rows=entry['rows'],row_sha=entry['roster_sha256'],
                task='decimal',stage='historical-retention',name=f"{ctx['protocol']['arm']}-historical-{old_stage}-{role}",
                directory=out/old_stage/role,specifications=specs,attempts=attempts,base_checks=bases)
    request_path = out/'host-request.json'
    write_once(request_path, {'schema':'weightbridge.phase_iiis.host_bridge_request.v1','bridge_sha256':BRIDGE_SHA,
        'action':'historical-retention','out':str(out/'gate.json'),'protocol':str(ctx['protocol_path']),
        'lock':str(ctx['lock_path']),'prerequisites':ctx['prerequisites'],'release':str(release),'workers':workers})
    subprocess.run([sys.executable,str(BRIDGE),'--request',str(request_path)],cwd=S,check=True)
    gate = read(out/'gate.json')
    study._assert_hashes(gate['input_sha256'])
    return gate


def formal(ctx, stage, out, attempts, bases, specs):
    workers = {'original':{},'r8':{},'candidate':{}}
    # Only this committed bank; no redraw or borrowing from the other bank.
    for task in ('tool','decimal'):
        entry = ctx['manifest']['splits'][stage]['files'][task]
        rows = Path(entry['path']).resolve()
        expected_path = (PRIVATE/'study-v1-host'/stage/f'{task}.json').resolve()
        if rows != expected_path or file_hash(rows) != entry['sha256']:
            raise ValueError('Bank path/hash differs from the prospectively frozen manifest')
        for role, sha in (('original',ORIGINAL),('r8',DECIMAL_PARENT)):
            cache = ctx['root']/'formal-parent-cache-v3'/stage/role
            cached = cache/f'{task}-cache.json'
            if cached.exists():
                record=read(cached)
                spec=record['worker_spec']
                owner=record['owner_arm']
                claim=cache/'attempts'/owner/f'{task}-attempt.json'
                if (record.get('schema')!='weightbridge.phase_iiis.parent_cache.v3' or owner not in PARENTS
                        or read(spec['protocol'])['arm']!=owner or record.get('roster_sha256')!=entry['sha256']
                        or record.get('stage')!=stage or record.get('task')!=task or record.get('role')!=role
                        or record.get('owner_attempt_sha256')!=file_hash(claim)
                        or record.get('worker_sha256')!=file_hash(spec['path'])
                        or record.get('execution_sha256')!=file_hash(spec['execution'])):
                    raise ValueError('Published parent cache ownership or actual worker identity changed')
                if file_hash(spec['rows']) != entry['sha256'] or Path(spec['rows']).resolve() != rows:
                    raise ValueError('Cached parent used another exact bank roster')
                _, checked = check_worker(ctx,spec,sha,ctx[role],task,stage)
                bases.append({**checked,'cached_parent':True,'cache_sha256':file_hash(cached),
                    'cost_owner':owner,'additional_model_calls':0})
            else:
                directory,claim,name=claim_parent_attempt(ctx,stage,role,task,entry['sha256'],sha)
                spec = worker(ctx,model=ctx[role],sha=sha,rows=rows,row_sha=entry['sha256'],task=task,stage=stage,
                    name=name,directory=directory,
                    specifications=specs,attempts=attempts,base_checks=bases)
                write_once(cached,{'schema':'weightbridge.phase_iiis.parent_cache.v3','owner_arm':ctx['protocol']['arm'],
                    'owner_attempt_sha256':file_hash(claim),'stage':stage,'task':task,'role':role,
                    'roster_sha256':entry['sha256'],'worker_spec':spec,'worker_sha256':file_hash(spec['path']),
                    'execution_sha256':file_hash(spec['execution']),
                    'cost_scope':'One owned reference attempt charged to the shared ledger; later consumers do not add this cost again.'})
            workers[role][task] = spec
        workers['candidate'][task] = worker(ctx,model=ctx['candidate'],sha=ctx['lock']['representation_sha256'],
            rows=rows,row_sha=entry['sha256'],task=task,stage=stage,name=f"{ctx['protocol']['arm']}-{stage}-candidate-{task}",
            directory=out/'candidate',specifications=specs,attempts=attempts,base_checks=bases)
    gate = study.record_stage(out/'gate.json',ctx['protocol_path'],stage=stage,workers=workers,
        candidate_checkpoint=ctx['lock']['representation_sha256'],step=128,lock_path=ctx['lock_path'],
        prerequisite_paths=ctx['prerequisites'])
    # Replay each already opened stage even when its behavioral gate fails.
    # A failed gate still prevents opening the later bank.
    replays, native = {}, {}
    for role in ('candidate','parent'):
        reference_role = 'candidate' if role=='candidate' else 'r8' if ctx['protocol']['arm']=='sequential' else 'original'
        model = ctx['reconstructed'] if role=='candidate' else ctx['actual_parent']
        sha = ctx['lock']['representation_sha256'] if role=='candidate' else ctx['protocol']['checkpoint_sha256']
        for task in ('tool','decimal'):
            recorded = workers[reference_role][task]
            replayed = worker(ctx,model=model,sha=sha,rows=recorded['rows'],row_sha=file_hash(recorded['rows']),
                task=task,stage=stage,name=f"{ctx['protocol']['arm']}-{stage}-replay-{role}-{task}",
                directory=out/f'replay-{role}',specifications=specs,attempts=attempts,base_checks=bases,
                worker_protocol=recorded['protocol'])
            a,_ = check_worker(ctx,recorded,sha,ctx['candidate'] if role=='candidate' else ctx['actual_parent'],task,stage)
            b,_ = check_worker(ctx,replayed,sha,model,task,stage)
            key=f'{stage}/{role}/{task}'
            native[key]=exact_native_replay(a['measurement'],b['measurement'])
            replays[key]={'recorded':recorded,'replayed':replayed}
    write_once(out/'replays.json',{'schema':'weightbridge.phase_iiis.private_stage_replay.v1',
        'passed':all(native.values()),'exact_native':native,'replays':replays,
        'source_sha256':ctx['protocol']['source_sha256'],'gate_sha256':file_hash(out/'gate.json')})
    return {**gate,'enclosing_exact_replay_passed':all(native.values())}


def claim_parent_attempt(ctx,stage,role,task,row_sha,checkpoint):
    """One reference attempt per arm; preserve incomplete attempts owned by others."""
    arm=ctx['protocol']['arm']
    cache=ctx['root']/'formal-parent-cache-v3'/stage/role
    directory=cache/'attempts'/arm
    claim=directory/f'{task}-attempt.json'
    name=f'{arm}-{stage}-reference-{role}-{task}'
    if claim.exists():
        raise FileExistsError('The same arm cannot retry its reference attempt')
    prior=[]
    for previous in sorted(cache.glob(f'attempts/*/{task}-attempt.json')):
        value=read(previous)
        owner=previous.parent.name
        if (value.get('schema')!='weightbridge.phase_iiis.parent_reference_attempt.v3'
                or owner not in PARENTS or value.get('owner_arm')!=owner
                or value.get('stage')!=stage or value.get('role')!=role or value.get('task')!=task
                or value.get('checkpoint_sha256')!=checkpoint or value.get('roster_sha256')!=row_sha
                or value.get('task_manifest_sha256')!=ctx['protocol']['task_manifest_sha256']
                or Path(value['planned_execution']).resolve()!=(ctx['ledger']/f'{owner}-{stage}-reference-{role}-{task}.receipt.json').resolve()):
            raise ValueError('An unpublished reference attempt has incompatible ownership or scope')
        if owner==arm:
            raise ValueError('A same-arm attempt cannot be relabelled as a fallback')
        paths={'claim':previous,'authorization':previous.with_name(f'{task}-authorization.json'),
            'worker':previous.with_name(f'{task}.json'),'execution':Path(value['planned_execution'])}
        prior.append({'owner_arm':value['owner_arm'],'artifacts':commitment({k:p for k,p in paths.items() if p.exists()}),
            'scope':'Preserved unpublished attempt; not reused as a validated parent measurement. Its recorded model cost remains in the shared ledger.'})
    write_once(claim,{'schema':'weightbridge.phase_iiis.parent_reference_attempt.v3','owner_arm':arm,
        'stage':stage,'task':task,'role':role,'checkpoint_sha256':checkpoint,'roster_sha256':row_sha,
        'protocol_sha256':file_hash(ctx['protocol_path']),'task_manifest_sha256':ctx['protocol']['task_manifest_sha256'],
        'planned_execution':str(ctx['ledger']/f'{name}.receipt.json'),'prior_unpublished_other_arm_attempts':prior,
        'maximum_attempts_for_this_arm_role_task':1,'same_arm_retry_authorized':False,
        'cost_scope':'This arm owns any new reference call; all attempts count against the unchanged shared resource limits.'})
    return directory,claim,name


def validate_formal_replays(ctx, stage):
    """Recompute the closed stage, including physical replay sources, before access."""
    directory=ctx['local']/'protected-v3'/stage
    gate=study.replay_stage(directory/'gate.json')
    record=read(directory/'replays.json')
    expected={f'{stage}/{role}/{task}' for role in ('candidate','parent') for task in ('tool','decimal')}
    if (set(record.get('replays',{}))!=expected or record.get('source_sha256')!=ctx['protocol']['source_sha256']
            or record.get('gate_sha256')!=file_hash(directory/'gate.json')):
        raise ValueError('Closed formal replay population or source binding changed')
    native={}
    for key,pair in record['replays'].items():
        _,role,task=key.split('/')
        reference_role='candidate' if role=='candidate' else 'r8' if ctx['protocol']['arm']=='sequential' else 'original'
        if set(pair)!={'recorded','replayed'} or pair['recorded']!=gate['workers'][reference_role][task]:
            raise ValueError('Replay reference differs from the exact independently scored worker')
        a,b=pair['recorded'],pair['replayed']
        if (Path(a['path']).resolve()==Path(b['path']).resolve() or Path(a['execution']).resolve()==Path(b['execution']).resolve()
                or file_hash(a['rows'])!=file_hash(b['rows']) or file_hash(a['protocol'])!=file_hash(b['protocol'])):
            raise ValueError('Replay was not a separate worker on its own frozen protocol and cases')
        sha=ctx['lock']['representation_sha256'] if role=='candidate' else ctx['protocol']['checkpoint_sha256']
        before,_=check_worker(ctx,a,sha,ctx['candidate'] if role=='candidate' else ctx['actual_parent'],task,stage)
        after,_=check_worker(ctx,b,sha,ctx['reconstructed'] if role=='candidate' else ctx['actual_parent'],task,stage)
        native[key]=exact_native_replay(before['measurement'],after['measurement'])
    if record.get('exact_native')!=native or record.get('passed')!=all(native.values()):
        raise ValueError('Recorded formal replay pass does not reproduce from actual full outputs')
    return gate,record


def stage_closure(ctx, stage):
    """A scored gate is not a completed host stage; missing work stays unmeasured."""
    if stage not in ('qualification','confirmation','historical-retention'):
        raise ValueError('Unknown completed-stage boundary')
    formal=stage!='historical-retention'
    directory=ctx['local']/'protected-v3'/stage
    completed_path,failed_path=directory/'completed.json',directory/'failed.json'
    result={'stage':stage,'completed_valid':False,'passed':False,'gate_path':None,'replays':{},
        'completed_sha256':None,'failed_sha256':None,'disposition':'UNOPENED',
        'reason':'No host stage directory exists; no protected evidence was accessed.'}
    if not directory.exists():
        return result
    if failed_path.exists() or not completed_path.exists():
        result.update(disposition='INTERRUPTED_INCOMPLETE',
            reason='Stage has a failed record.' if failed_path.exists() else 'Stage started but has no completed host marker.',
            failed_sha256=file_hash(failed_path) if failed_path.exists() else None,
            completed_sha256=file_hash(completed_path) if completed_path.exists() else None)
        # Do not use a partial gate/replays pair to manufacture a completed pass.
        return result
    completed=read(completed_path)
    expected={'schema':'weightbridge.phase_iiis.private_protected_execution.v3','action':stage,
        'arm':ctx['protocol']['arm'],'driver_sha256':file_hash(SELF),'bridge_sha256':BRIDGE_SHA,
        'protocol_sha256':file_hash(ctx['protocol_path']),'task_manifest_sha256':ctx['protocol']['task_manifest_sha256'],
        'representation_sha256':ctx['lock']['representation_sha256'],'lock_sha256':file_hash(ctx['lock_path']),
        'all_arms_lock_sha256':file_hash(ctx['root']/'locks/both-arms.json'),'completed':True}
    if any(completed.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed formal marker differs from its exact driver, protocol, lock or model')
    study._assert_hashes(completed.get('input_sha256') or {})
    required={'protocol','lock','all_arms_lock','gate','host_source','host_request','started'}
    if formal:required.add('replays')
    if set(completed.get('input_sha256',{}))!=required:
        raise ValueError('Completed formal marker is missing its full evidence commitments')
    if (completed.get('gate_sha256')!=file_hash(directory/'gate.json')
            or formal and completed.get('replay_receipt_sha256')!=file_hash(directory/'replays.json')
            or completed['input_sha256']['host_source']['sha256']!=file_hash(SELF)
            or completed['input_sha256']['host_request']['sha256']!=completed.get('request_sha256')):
        raise ValueError('Completed formal gate, replay or request identity changed')
    # Require commitments to these actual local artifacts, not unrelated hashes.
    paths={'protocol':ctx['protocol_path'],'lock':ctx['lock_path'],'all_arms_lock':ctx['root']/'locks/both-arms.json',
        'gate':directory/'gate.json','host_source':SELF,'started':directory/'started.json'}
    if formal:paths['replays']=directory/'replays.json'
    if any(Path(completed['input_sha256'][k]['path']).resolve()!=Path(v).resolve() for k,v in paths.items()):
        raise ValueError('Completed formal marker points at another stage artifact')
    started=read(directory/'started.json')
    if any(started.get(k)!=v for k,v in expected.items() if k!='completed') or started.get('request_sha256')!=completed.get('request_sha256'):
        raise ValueError('Completed stage does not close its exact original started record')
    request=read(completed['input_sha256']['host_request']['path'])
    if (request.get('schema')!='weightbridge.phase_iiis.protected_host_request.v3'
            or request.get('driver_sha256')!=file_hash(SELF) or request.get('action')!=stage
            or request.get('arm')!=ctx['protocol']['arm']):
        raise ValueError('Completed formal marker is not bound to this explicit stage request')
    gate,replay=validate_formal_replays(ctx,stage) if formal else validate_historical_replay(ctx,directory)
    actual_pass=gate['passed'] and replay['passed']
    if completed.get('passed')!=actual_pass or completed.get('exact_replays_passed')!=replay['passed']:
        raise ValueError('Completed formal success flags differ from the actual gate and native replay')
    disposition=gate['disposition'] if replay['passed'] else 'NATIVE_REPLAY_FAILURE'
    if completed.get('disposition')!=disposition:
        raise ValueError('Completed formal disposition differs from measured outcomes')
    result.update(completed_valid=True,passed=actual_pass,gate_path=directory/'gate.json',replays=replay['replays'],
        completed_sha256=file_hash(completed_path),gate_sha256=file_hash(directory/'gate.json'),
        replay_receipt_sha256=file_hash(directory/'replays.json') if formal else None,disposition=disposition,
        reason='Completed marker, exact identities, measured behavior and full native replay independently rechecked.')
    return result


def formal_closure(ctx,stage):
    if stage not in ('qualification','confirmation'):raise ValueError('Expected a formal bank')
    return stage_closure(ctx,stage)


def historical_closure(ctx):
    return stage_closure(ctx,'historical-retention')


def validate_historical_replay(ctx,directory):
    """Recompute the bridge's measured retention and native replay, without inference."""
    gate=read(directory/'gate.json')
    request_path=directory/'host-request.json'
    if (gate.get('bridge_sha256')!=BRIDGE_SHA or gate.get('bridge_request_sha256')!=file_hash(request_path)
            or Path(gate.get('bridge_request_path','')).resolve()!=request_path.resolve()):
        raise ValueError('Historical retention changed its exact host bridge request')
    study._assert_hashes(gate['input_sha256'])
    request=read(request_path)
    if (request.get('schema')!='weightbridge.phase_iiis.host_bridge_request.v1'
            or request.get('bridge_sha256')!=BRIDGE_SHA or request.get('action')!='historical-retention'
            or Path(request.get('protocol','')).resolve()!=ctx['protocol_path'].resolve()
            or Path(request.get('lock','')).resolve()!=ctx['lock_path'].resolve()
            or Path(request.get('out','')).resolve()!=(directory/'gate.json').resolve()):
        raise ValueError('Historical bridge is not scoped to this exact arm and completed record')
    spec=importlib.util.spec_from_file_location('phase_s_recorded_host_bridge',BRIDGE)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    measured=bridge.historical_gate(request)
    expected_inputs={**measured['input_sha256'],**commitment({'host_bridge_source':BRIDGE,'host_bridge_request':request_path})}
    if gate['input_sha256']!=expected_inputs or any(gate.get(k)!=v for k,v in measured.items() if k!='input_sha256'):
        raise ValueError('Historical retention does not reproduce from all measured workers and native replays')
    for bundle in request['workers'].values():
        for role,base in (('candidate',ctx['candidate']),('reconstructed',ctx['reconstructed']),('parent',ctx['actual_parent'])):
            check_base(bundle[role],base)
    exact=len(measured['gates'])==3 and all(all(g['exact_native_replays'].values()) for g in measured['gates'].values())
    measured['disposition']='HISTORICAL_RETENTION_PASS' if measured['passed'] else 'HISTORICAL_RETENTION_FAILURE'
    return measured,{'passed':exact,'replays':{}}


def run(request, request_path):
    action=request['action']
    if action not in ('historical-retention','qualification','confirmation','finalize'):
        raise ValueError('Unknown fixed host action')
    ctx=context(request,required_stage=action if action!='finalize' else None)
    out=ctx['local']/'protected-v3'/action
    if out.exists():raise FileExistsError('Protected stage attempts are write-once and cannot be silently resumed')
    if action in ('qualification','confirmation'):
        previous=historical_closure(ctx)
        if not previous['completed_valid'] or not previous['passed']:
            raise ValueError('Formal bank refused after incomplete or failed historical retention')
    # Confirmation additionally requires a successful replay of qualification,
    # not merely a behavioral gate written before an interrupted replay worker.
    if action=='confirmation':
        previous=formal_closure(ctx,'qualification')
        if not previous['completed_valid'] or not previous['passed']:
            raise ValueError('Confirmation refused after incomplete or failed qualification/replay')
    out.mkdir(parents=True)
    started={'schema':'weightbridge.phase_iiis.private_protected_execution.v3','action':action,'arm':ctx['protocol']['arm'],
        'driver_sha256':file_hash(SELF),'bridge_sha256':BRIDGE_SHA,'request_sha256':file_hash(request_path),
        'protocol_sha256':file_hash(ctx['protocol_path']),'task_manifest_sha256':ctx['protocol']['task_manifest_sha256'],
        'representation_sha256':ctx['lock']['representation_sha256'],'lock_sha256':file_hash(ctx['lock_path']),
        'all_arms_lock_sha256':file_hash(ctx['root']/'locks/both-arms.json'),'no_retry':True,'no_refit':True,
        'r_bank_b_access':False,'production_authority':False,'lineage_authority':False}
    write_once(out/'started.json',started)
    wall,cpu=time.perf_counter(),time.process_time();attempts=[];bases=[];specs=[]
    try:
        if action=='historical-retention':gate=historical(ctx,out,attempts,bases,specs)
        elif action in ('qualification','confirmation'):gate=formal(ctx,action,out,attempts,bases,specs)
        else:
            stages={};replays={};closures={}
            history=historical_closure(ctx)
            closures['historical-retention']={k:str(v) if isinstance(v,Path) else v for k,v in history.items() if k!='replays'}
            for stage in ('qualification','confirmation'):
                closure=formal_closure(ctx,stage)
                stages[stage]=closure['gate_path'] if closure['completed_valid'] else None
                replays.update(closure['replays'])
                closures[stage]={k:str(v) if isinstance(v,Path) else v for k,v in closure.items() if k!='replays'}
            ready=all(c['completed_valid'] and c['passed'] for c in closures.values())
            if not history['completed_valid'] or not history['passed']:
                # The historical lifecycle is required for independent access.
                # Do not credit any later artifacts after that boundary failed.
                stages={stage:None for stage in stages};replays={}
            # Missing/failed host transactions are omitted from the frozen
            # verifier inputs, forcing missing stages/replays to remain false.
            gate=study.finalize(out/'final-verification.json',ctx['protocol_path'],ctx['lock_path'],
                qualification_path=stages['qualification'],confirmation_path=stages['confirmation'],
                reconstruction_path=ctx['reconstruction_gate'],replays=replays,
                original=ctx['original'],actual_parent=ctx['actual_parent'])
            if gate['passed'] and not ready:
                raise AssertionError('An incomplete formal lifecycle cannot yield final PASS')
            if any(c['disposition']=='INTERRUPTED_INCOMPLETE' for c in closures.values()):
                final_disposition='INTERRUPTED_INCOMPLETE'
            elif any(c['disposition']=='INSUFFICIENT_IMPROVEMENT_HEADROOM' for c in closures.values()):
                final_disposition='INSUFFICIENT_IMPROVEMENT_HEADROOM'
            else:
                final_disposition=gate['disposition']
            write_once(out/'formal-stage-closure.json',{'schema':'weightbridge.phase_iiis.formal_stage_closure.v3',
                'passed':ready and gate['passed'],'ready_for_final_pass':ready,'stages':closures,
                'frozen_verifier_sha256':file_hash(out/'final-verification.json'),'disposition':final_disposition,
                'no_unopened_roster_access':True})
        if action=='historical-retention':
            exact=len(gate.get('gates',{}))==3 and all(all(g['exact_native_replays'].values()) for g in gate['gates'].values())
        elif action=='finalize':
            exact=gate['checks']['all_native_replays']
        else:
            exact=gate['enclosing_exact_replay_passed']
        disposition=(final_disposition if action=='finalize' else 'NATIVE_REPLAY_FAILURE' if not exact
            else gate.get('disposition','HISTORICAL_RETENTION_PASS' if gate['passed'] else 'HISTORICAL_RETENTION_FAILURE'))
        gate_path=out/('final-verification.json' if action=='finalize' else 'gate.json')
        inputs={'protocol':ctx['protocol_path'],'lock':ctx['lock_path'],'all_arms_lock':ctx['root']/'locks/both-arms.json',
            'gate':gate_path,'host_source':SELF,'host_request':request_path,'started':out/'started.json'}
        if action in ('qualification','confirmation'):inputs['replays']=out/'replays.json'
        if action=='finalize':inputs['formal_stage_closure']=out/'formal-stage-closure.json'
        result={**started,'completed':True,'passed':gate['passed'] and exact,'exact_replays_passed':exact,
            'disposition':disposition,
            'gate_sha256':file_hash(out/('final-verification.json' if action=='finalize' else 'gate.json')),
            'replay_receipt_sha256':file_hash(out/'replays.json') if action in ('qualification','confirmation') else None,
            'input_sha256':commitment(inputs),'model_workers':attempts,'base_command_checks':bases,
            'resources':{'host_wall_seconds':time.perf_counter()-wall,'host_cpu_seconds':time.process_time()-cpu,
                'scope':'Host elapsed includes waiting; disjoint model costs remain exclusively in the shared execution ledger.'}}
        write_once(out/'completed.json',result)
        return result
    except BaseException as error:
        write_once(out/'failed.json',{**started,'completed':False,'passed':False,'error':{'type':type(error).__name__,'message':str(error)},
            'model_workers':attempts,'base_command_checks':bases,'unmeasured_stages_are_not_passes':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=read(args.request)
    if (request.get('schema')!='weightbridge.phase_iiis.protected_host_request.v3'
            or request.get('driver_sha256')!=file_hash(SELF)):
        raise ValueError('Protected host execution requires its prospectively pinned reviewed source')
    result=run(request,args.request)
    print({k:result.get(k) for k in ('action','arm','completed','passed','disposition')})


if __name__=='__main__':main()
