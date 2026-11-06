"""Private, source-pinned Phase S host orchestration. Never opens protected banks.

Only already completed fixed-step development may trigger the existing frozen
merge/evaluation workers. Every model call uses the common owned-worker ledger.
No training, replacement experiment, source edit, or automatic promotion occurs.
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
PRIVATE = Path('C:/Users/jacks/AppData/Local/Temp/wb-iiis-tool-transfer')
BRIDGE = PRIVATE/'host_stage_bridge_v1.py'
BRIDGE_SHA = '258ee4e5b20992895588f072f6e0d42659e73cb847b06398f07e6104d5ddf67c'
SELF = Path(__file__).resolve()
sys.path.insert(0, str(S))

from experiments.phase_iiis_tool_transfer import execute, host, merge, study
from experiments.phase_iiis_tool_transfer.integrity import ORIGINAL, DECIMAL_PARENT, PARENTS, read, write_once
from experiments.phase_iiis_tool_transfer.verification import authorize_evaluation, validate_worker
from weightbridge.adaptation.supervised_lora import LoRATarget
from weightbridge.core import file_hash
from weightbridge.inspection.safetensors import load_checkpoint


def paths_commit(paths):
    return {key: {'path': str(Path(path).resolve()), 'sha256': file_hash(path)} for key, path in paths.items()}


def assert_base_command(receipt, expected):
    observed = study._command_value(read(receipt)['command'], '--base')
    if Path(observed).resolve() != Path(expected).resolve():
        raise ValueError('Fresh worker did not load the explicitly requested checkpoint directory')
    return {'passed': True, 'execution_sha256': file_hash(receipt), 'base': str(Path(expected).resolve())}


def finish_arm(request, request_path):
    wall, cpu = time.perf_counter(), time.process_time()
    root, _, manifest = host._initialized(request['study'])
    if root != (PRIVATE/'study-v1').resolve() or request['arm'] not in PARENTS:
        raise ValueError('Private driver is restricted to the existing two-arm Phase S study')
    arm = request['arm']
    local, protocol_path = root/arm, root/arm/'protocol.json'
    protocol = study._protocol(protocol_path)
    ledger = Path(request['ledger']).resolve()
    if ledger != (PRIVATE/'execution').resolve() or file_hash(BRIDGE) != BRIDGE_SHA:
        raise ValueError('Shared ledger or prospectively reviewed host bridge changed')
    executed = read(local/'run-arm.json')
    if executed.get('completed') is not True or executed.get('status') != 'DEVELOPMENT_MEASURED':
        raise ValueError('The fixed main and both development checkpoints must finish before host continuation')
    development_path = local/'assessment/step-0128.json'
    development = study.replay_stage(development_path)
    if development['stage'] != 'development' or development['step'] != 128:
        raise ValueError('Only the exact fixed-step128 adapter receives post-development handling')
    adapter = local/'training/checkpoints/step-0128/adapter.safetensors'
    training = study.validate_main_checkpoint(protocol_path, local/'training/completed.json',
        ledger/f'{arm}-main-train.receipt.json', adapter)
    if training != read(local/'main-training-validation.json'):
        raise ValueError('Original completed main validation does not replay')
    out, lock_path = local/'post-development-v1', root/'locks'/f'{arm}.json'
    if out.exists() or lock_path.exists():
        raise FileExistsError('No implicit post-development retries or changed arm locks')
    base, original = Path(request['actual_parent']).resolve(), Path(request['original']).resolve()
    if (load_checkpoint(base).checkpoint_sha256() != PARENTS[arm]
            or load_checkpoint(original).checkpoint_sha256() != ORIGINAL):
        raise ValueError('Explicit original or actual training parent has changed')
    out.mkdir()
    started = {'schema': 'weightbridge.phase_iiis.private_postdevelopment.v1', 'arm': arm,
        'driver_sha256': file_hash(SELF), 'request_sha256': file_hash(request_path),
        'bridge_sha256': BRIDGE_SHA, 'protocol_sha256': file_hash(protocol_path),
        'task_manifest_sha256': protocol['task_manifest_sha256'], 'protected_banks_opened': False,
        'model_workers': [], 'base_command_checks': [], 'production_authority': False, 'lineage_authority': False}
    write_once(out/'started.json', started)
    attempts, base_checks = [], []
    last_execution = None

    def worker(name, module, args, limits, expected_base, expected_out):
        nonlocal last_execution
        name = f'{arm}-post-{name}'
        last_execution = ledger/f'{name}.receipt.json'
        result = execute.run(ledger, name, f'experiments.phase_iiis_tool_transfer.{module}', args,
            wall_limit=limits['max_wall_seconds'], cpu_limit=limits['max_cpu_seconds'],
            memory_limit=limits['max_memory_bytes'])
        attempts.append({'path': str(last_execution), 'sha256': file_hash(last_execution), 'passed': result['passed']})
        if not result['passed']:
            raise RuntimeError('Owned worker failed; later stages are not authorized')
        study.validate_execution(last_execution, module=f'experiments.phase_iiis_tool_transfer.{module}',
            out=expected_out, protocol_path=protocol_path if module == 'merge' else study._command_value(result['command'], '--protocol'))
        base_checks.append(assert_base_command(last_execution, expected_base))
        return last_execution

    def evaluation(name, model, sha, stage, task, *, worker_protocol=protocol_path):
        rows = root/'material/development'/f'{task}.json'
        entry = manifest['splits']['development']['files'][task]
        if Path(entry['path']).resolve() != rows.resolve() or file_hash(rows) != entry['sha256']:
            raise ValueError('Only the original frozen DEVELOPMENT roster is available to this driver')
        record = out/'evaluation'/f'{name}-{task}.json'
        record.parent.mkdir(exist_ok=True)
        authorization = record.with_name(f'{name}-{task}-authorization.json')
        authorize_evaluation(authorization, worker_protocol, stage=stage, task=task,
            roster_sha256=entry['sha256'], representation_sha256=sha)
        args = ['--base', str(model), '--protocol', str(worker_protocol), '--rows', str(rows),
            '--out', str(record), '--expected-checkpoint', sha, '--expected-rows', entry['sha256'],
            '--stage', stage, '--task', task, '--authorization', str(authorization)]
        outer = worker(f'{name}-{task}', 'evaluate', args, protocol['resources']['evaluation_limits'], model, record)
        validate_worker(record, rows, worker_protocol, sha, None, stage, task, authorization=authorization)
        return {'path': str(record), 'rows': str(rows), 'authorization': str(authorization),
            'protocol': str(worker_protocol), 'stage': stage, 'execution': str(outer)}

    def completed(lock, reason):
        result = {**started, 'completed': True, 'passed': lock['passed'], 'status': lock['status'],
            'disposition': lock['disposition'], 'reason': reason, 'lock_sha256': file_hash(lock_path),
            'model_workers': attempts, 'base_command_checks': base_checks,
            'resources': {'host_wall_seconds': time.perf_counter()-wall, 'host_cpu_seconds': time.process_time()-cpu,
                'scope': 'Host time includes waiting; model costs are exclusively in the shared owned-worker ledger, not added twice.'}}
        write_once(out/'completed.json', result)
        return result

    try:
        if not development['finalist_eligible']:
            lock = study.lock_arm(lock_path, protocol_path, development_path, adapter_path=adapter, training_validation=training)
            return completed(lock, 'fixed_adapter_development_failed_no_merge')
        inventory = read(local/'training/inventory.json')
        targets = tuple(LoRATarget(**{**row, 'shape': tuple(row['shape'])}) for row in inventory['modules'])
        if len(targets) != 186 or set(t.module_name for t in targets) != set(read(local/'main-authorization.json')['allowed_modules']):
            raise ValueError('Merge allowlist differs from the actual frozen training inventory')
        policy_path = out/'merge-authorization.json'
        write_once(policy_path, merge.merge_authorization(protocol_path, file_hash(adapter), targets))
        merged_dir, raw_merge = out/'merged', out/'merge-receipt.json'
        prior = Path(request['r_merge_receipt']).resolve() if arm == 'sequential' else None
        def merge_args(parent, destination, receipt, expected=None):
            args = ['--base', str(parent), '--protocol', str(protocol_path), '--adapter', str(adapter),
                '--expected-adapter', file_hash(adapter), '--expected-parent', PARENTS[arm],
                '--authorization', str(policy_path), '--out', str(destination), '--receipt', str(receipt),
                '--original', str(original)]
            if prior is not None:
                args += ['--prior-merge-receipt', str(prior)]
            if expected is not None:
                args += ['--expected-reconstruction', expected]
            return args
        outer_merge = worker('merge', 'merge', merge_args(base, merged_dir, raw_merge),
            protocol['resources']['merge_limits'], base, merged_dir)
        candidate_sha = load_checkpoint(merged_dir).checkpoint_sha256()
        if read(raw_merge)['candidate_sha256'] != candidate_sha:
            raise ValueError('Merge receipt differs from actual stored checkpoint')
        candidate_specs = {task: evaluation('merged', merged_dir, candidate_sha, 'merged-development', task)
            for task in ('tool', 'decimal')}
        parents = read(root/'baselines/completed.json')['workers']
        merged_gate_path = out/'merged-development-gate.json'
        merged_gate = study.record_stage(merged_gate_path, protocol_path, stage='merged-development',
            workers={**parents, 'candidate': candidate_specs}, candidate_checkpoint=candidate_sha, step=128)
        if not merged_gate['finalist_eligible']:
            # Honest terminal bookkeeping, never a completed reconstruction or PASS.
            not_run = out/'reconstruction-not-run-terminal.json'
            write_once(not_run, {'schema': 'weightbridge.phase_iiis.terminal_stage_not_run.v1',
                'status': 'NOT_RUN', 'passed': False, 'exact_reconstruction': None,
                'candidate_sha256': candidate_sha, 'parent_checkpoint_sha256': PARENTS[arm],
                'protocol_sha256': file_hash(protocol_path), 'task_manifest_sha256': protocol['task_manifest_sha256'],
                'reason': 'The actual merged-development behavioral gate failed; no reconstruction was executed.',
                'input_sha256': paths_commit({'merged_development': merged_gate_path}),
                'scope': 'Terminal arm bookkeeping only. Not reconstruction verification or authorization.'})
            lock = study.lock_arm(lock_path, protocol_path, development_path, adapter_path=adapter,
                training_validation=training, merged_development_path=merged_gate_path,
                merge_verification_path=raw_merge.with_suffix('.verification.json'), reconstruction_path=not_run)
            return completed(lock, 'merged_development_failed_reconstruction_not_run')
        rebuilt_dir, raw_rebuild = out/'reconstructed', out/'reconstruction-merge.json'
        rebuilt_parent = Path(request['r_reconstruction_directory']).resolve() if arm == 'sequential' else original
        if load_checkpoint(rebuilt_parent).checkpoint_sha256() != PARENTS[arm]:
            raise ValueError('The recorded rounded R rebuild or original source identity differs')
        outer_rebuild = worker('reconstruct', 'merge', merge_args(rebuilt_parent, rebuilt_dir, raw_rebuild, candidate_sha),
            protocol['resources']['merge_limits'], rebuilt_parent, rebuilt_dir)
        replay_specs = {}
        own = 'r8' if arm == 'sequential' else 'original'
        for role in ('candidate', 'parent'):
            for task in ('tool', 'decimal'):
                recorded = candidate_specs[task] if role == 'candidate' else parents[own][task]
                model = rebuilt_dir if role == 'candidate' else base
                sha = candidate_sha if role == 'candidate' else PARENTS[arm]
                replayed = evaluation(f'replay-{role}', model, sha, 'reconstruction-development', task,
                    worker_protocol=recorded['protocol'])
                replay_specs[f'{role}/{task}'] = {'recorded': recorded, 'replayed': replayed}
        request_path = out/'reconstruction-host-request.json'
        reconstruction_path = out/'exact-reconstruction-gate.json'
        bridge_request = {'schema': 'weightbridge.phase_iiis.host_bridge_request.v1', 'bridge_sha256': BRIDGE_SHA,
            'action': 'exact-reconstruction', 'out': str(reconstruction_path), 'protocol': str(protocol_path),
            'adapter': str(adapter), 'merged_development': str(merged_gate_path),
            'merge_receipt': str(raw_merge), 'merge_verification': str(raw_merge.with_suffix('.verification.json')),
            'merge_execution': str(outer_merge), 'reconstruction_receipt': str(raw_rebuild),
            'reconstruction_verification': str(raw_rebuild.with_suffix('.verification.json')),
            'reconstruction_execution': str(outer_rebuild), 'candidate_directory': str(merged_dir),
            'reconstructed_directory': str(rebuilt_dir), 'original_directory': str(original),
            'actual_parent_directory': str(base), 'replays': replay_specs}
        if arm == 'sequential':
            bridge_request.update(r_reconstruction_directory=str(rebuilt_parent),
                r_reconstruction_receipt=request['r_reconstruction_receipt'])
        write_once(request_path, bridge_request)
        subprocess.run([sys.executable, str(BRIDGE), '--request', str(request_path)], cwd=S, check=True)
        lock = study.lock_arm(lock_path, protocol_path, development_path, adapter_path=adapter,
            training_validation=training, merged_development_path=merged_gate_path,
            merge_verification_path=raw_merge.with_suffix('.verification.json'), reconstruction_path=reconstruction_path)
        study.validate_arm_lock(lock_path)
        return completed(lock, 'development_storage_and_replay_measured')
    except BaseException as error:
        # Never infer a behavioral failure from missing or interrupted execution.
        if last_execution is not None and last_execution.exists() and not lock_path.exists():
            failed = read(last_execution)
            if failed.get('passed') is False and failed.get('error') is not None:
                study.lock_operational_failure(lock_path, protocol_path, last_execution,
                    disposition='RESOURCE_INFEASIBLE' if (failed.get('execution') or {}).get('budget_failure') else 'INTERRUPTED_INCOMPLETE')
        write_once(out/'failed.json', {**started, 'completed': False, 'passed': False,
            'error': {'type': type(error).__name__, 'message': str(error)}, 'model_workers': attempts,
            'base_command_checks': base_checks, 'retry_authorized': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = read(args.request)
    if (request.get('schema') != 'weightbridge.phase_iiis.postdevelopment_request.v1'
            or request.get('driver_sha256') != file_hash(SELF)):
        raise ValueError('Host request must prospectively pin this reviewed private driver')
    if request['action'] == 'finish-arm':
        result = finish_arm(request, args.request)
    elif request['action'] == 'lock-both':
        root, _, _ = host._initialized(request['study'])
        if root != (PRIVATE/'study-v1').resolve():
            raise ValueError('Only the existing frozen study may be locked')
        result = study.lock_arms(root/'locks/both-arms.json', {arm: root/'locks'/f'{arm}.json' for arm in PARENTS})
    else:
        raise ValueError('This private driver has no historical, formal-bank or training action')
    print(json.dumps({k: result.get(k) for k in ('schema', 'status', 'passed', 'disposition')}))


if __name__ == '__main__':
    main()
