"""Host execution of one fixed S arm, with one non-nested scope per worker.

Stops this arm on pipeline/resource failure. Development correctness is left to
the host's frozen gate; step 64 is diagnostic and only step 128 is eligible.
No protected-bank authorization, merging, retraining or deployment is issued.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from weightbridge.core import file_hash
from . import execute, initialize, smoke
from .integrity import PARENTS, read, verify_sources, write_once
from .verification import authorize_evaluation, validate_worker


def run(study_root, base, arm, *, ledger):
    if arm not in PARENTS:
        raise ValueError('Unknown Phase S arm')
    study, ledger = Path(study_root).resolve(), Path(ledger).resolve()
    root = study/arm
    protocol_path = root/'protocol.json'
    protocol = read(protocol_path)
    if protocol.get('arm') != arm or protocol.get('checkpoint_sha256') != PARENTS[arm]:
        raise ValueError('Arm protocol does not identify the intended immutable parent')
    verify_sources(protocol['source_sha256'])
    if (root/'run-arm.started.json').exists() or (root/'run-arm.json').exists():
        raise ValueError('Arm execution cannot be retried or resumed implicitly')
    # Explicit shared ledger is required; per-arm ledgers would hide cumulative
    # model cost and are not created by this entry point.
    if ledger == root or ledger.is_relative_to(root):
        raise ValueError('Arm model work needs the host-supplied shared cumulative ledger')
    started = {'schema': 'weightbridge.phase_iiis.arm_execution.v1', 'arm': arm,
        'parent_sha256': PARENTS[arm], 'protocol_sha256': file_hash(protocol_path),
        'ledger': str(ledger), 'retry_authorized': False, 'fixed_finalist_step': 128,
        'production_authority': False, 'lineage_authority': False}
    write_once(root/'run-arm.started.json', started)
    records, development, stage = [], {}, 'not_started'
    def worker(name, module, args, limits):
        nonlocal stage
        stage = name
        result = execute.run(ledger, f'{arm}-{name}', f'experiments.phase_iiis_tool_transfer.{module}', args,
            wall_limit=limits['max_wall_seconds'], cpu_limit=limits['max_cpu_seconds'],
            memory_limit=min(limits.get('max_memory_bytes', protocol['resources']['max_memory_bytes']),
                protocol['resources']['max_memory_bytes']))
        path = ledger/f'{arm}-{name}.receipt.json'
        records.append({'stage': name, 'path': str(path), 'sha256': file_hash(path), 'passed': result['passed']})
        if not result['passed']:
            raise RuntimeError(f'Owned worker or frozen resource gate failed at {name}')
        return path
    def common(auth, bundle, out):
        return ['--base', str(Path(base).resolve()), '--protocol', str(protocol_path),
            '--authorization', str(auth), '--bundle', str(bundle), '--out', str(out)]
    try:
        smoke_limit = protocol['resources']['mode_limits']['smoke']
        smoke_root = root/'smoke'
        auth, bundle = root/'smoke-authorization.json', root/'smoke-bundle.json'
        zero_out, train_out, reload_out = smoke_root/'zero', smoke_root/'training', smoke_root/'reload'
        outer_zero = worker('smoke-zero', 'smoke', [*common(auth, bundle, zero_out), '--stage', 'zero'], smoke_limit)
        initialize.claim_smoke(root)
        outer_train = worker('smoke-train', 'train', [*common(auth, bundle, train_out), '--mode', 'smoke'], smoke_limit)
        completed = read(train_out/'completed.json')
        checkpoint = Path(completed['checkpoints'][-1])
        saved = read(checkpoint/'receipt.json')
        outer_reload = worker('smoke-reload', 'smoke', [*common(auth, bundle, reload_out), '--stage', 'reload',
            '--adapter', str(checkpoint/'adapter.safetensors'), '--adapter-sha256', saved['sha256']], smoke_limit)
        stage = 'smoke-admission'
        gate = smoke.validate_gate(protocol_path, zero_out/'smoke-check.json', train_out/'completed.json', reload_out/'smoke-check.json')
        gate_path = root/'smoke-gate.json'
        write_once(gate_path, gate)
        if not gate['passed']:
            raise RuntimeError('Disposable smoke did not qualify the training pipeline')
        initialize.authorize_main(root, gate_path, execution_receipts=[outer_zero, outer_train, outer_reload])
        initialize.claim_main(root)
        main_out = root/'training'
        outer_main = worker('main-train', 'train', [*common(root/'main-authorization.json', root/'training-bundle.json', main_out),
            '--mode', 'main', '--smoke-gate', str(gate_path)], protocol['resources']['mode_limits']['main'])
        from .study import validate_main_checkpoint
        stage = 'main-completion-verification'
        training_check = validate_main_checkpoint(protocol_path, main_out/'completed.json', outer_main,
            main_out/'checkpoints/step-0128/adapter.safetensors')
        write_once(root/'main-training-validation.json', training_check)
        limits = protocol['resources']['evaluation_limits']
        for step in (64, 128):
            checkpoint = main_out/'checkpoints'/f'step-{step:04d}'
            saved = read(checkpoint/'receipt.json')
            adapter_path, adapter_sha = checkpoint/'adapter.safetensors', saved['sha256']
            if saved.get('step') != step or file_hash(adapter_path) != adapter_sha:
                raise ValueError('Declared development checkpoint identity changed')
            specs = {}
            for task in ('tool', 'decimal'):
                rows = study/'material/development'/f'{task}.json'
                out = root/'development'/f'step-{step:04d}'/f'{task}.json'
                out.parent.mkdir(parents=True, exist_ok=True)
                authorization = out.with_name(f'{task}-authorization.json')
                authorize_evaluation(authorization, protocol_path, stage='development', task=task,
                    roster_sha256=file_hash(rows), representation_sha256=PARENTS[arm], adapter_sha256=adapter_sha)
                args = ['--base', str(Path(base).resolve()), '--protocol', str(protocol_path), '--rows', str(rows),
                    '--out', str(out), '--authorization', str(authorization), '--stage', 'development', '--task', task,
                    '--expected-checkpoint', PARENTS[arm], '--expected-rows', file_hash(rows),
                    '--adapter', str(adapter_path), '--expected-adapter', adapter_sha]
                outer = worker(f'development-{step}-{task}', 'evaluate', args, limits)
                validate_worker(out, rows, protocol_path, PARENTS[arm], adapter_sha, 'development', task,
                    authorization=authorization)
                specs[task] = {'path': str(out), 'rows': str(rows), 'authorization': str(authorization),
                    'protocol': str(protocol_path), 'stage': 'development', 'execution': str(outer)}
            development[str(step)] = {'diagnostic_only': step == 64, 'adapter_sha256': adapter_sha, 'workers': specs}
        outcome = {**started, 'status': 'DEVELOPMENT_MEASURED', 'completed': True, 'worker_receipts': records,
            'development': development, 'protected_evaluation_opened': False, 'behavioral_qualification': None,
            'scope': 'Execution and raw development evidence only; host must apply the frozen fixed-step gate.'}
        write_once(root/'run-arm.json', outcome)
        return outcome
    except BaseException as exc:
        write_once(root/'run-arm.json', {**started, 'status': 'INCOMPLETE_ARM_EXECUTION', 'completed': False,
            'failed_stage': stage, 'error': {'type': type(exc).__name__, 'message': str(exc)},
            'worker_receipts': records, 'development': development, 'protected_evaluation_opened': False,
            'behavioral_qualification': None, 'other_arm_recipe_changed': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'base', 'ledger'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(PARENTS), required=True)
    args = parser.parse_args()
    result = run(args.study, args.base, args.arm, ledger=args.ledger)
    print({'arm': args.arm, 'status': result['status']}, flush=True)
