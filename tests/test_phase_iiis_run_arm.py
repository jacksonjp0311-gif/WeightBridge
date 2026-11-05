"""Mocked host orchestration only: no models, private banks or optimizers run."""
import json
from pathlib import Path

import pytest

from experiments.phase_iiis_tool_transfer import run_arm
from experiments.phase_iiis_tool_transfer.integrity import PARENTS, ROOT, read
from weightbridge.core import file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def prepared(tmp_path):
    study = tmp_path/'study'
    root = study/'sequential'
    protocol = {'schema': 'weightbridge.phase_iiis.protocol.v1', 'arm': 'sequential',
        'checkpoint_sha256': PARENTS['sequential'], 'task_manifest_sha256': 'd'*64,
        'source_sha256': {'weightbridge/core.py': file_hash(ROOT/'weightbridge/core.py')},
        'endpoints': {'tool': {'cpu_threads': 6, 'batch_size': 8, 'max_input_tokens': 512, 'max_new_tokens': 96},
            'decimal': {'cpu_threads': 6, 'batch_size': 8, 'max_input_tokens': 128, 'max_new_tokens': 16}},
        'resources': {'max_memory_bytes': 32*1024**3, 'mode_limits': {
            'smoke': {'max_wall_seconds': 1000, 'max_cpu_seconds': 5000},
            'main': {'max_wall_seconds': 7200, 'max_cpu_seconds': 40000}},
            'evaluation_limits': {'max_wall_seconds': 1800, 'max_cpu_seconds': 10000}}}
    write(root/'protocol.json', protocol)
    for task in ('tool', 'decimal'):
        write(study/'material/development'/f'{task}.json', [])
    return study, root, tmp_path/'ledger'


@pytest.mark.parametrize('failed', ['smoke-zero', 'smoke-train'])
def test_failure_stops_only_this_arm_and_never_retries(tmp_path, monkeypatch, failed):
    study, root, ledger = prepared(tmp_path)
    calls, claims = [], []
    monkeypatch.setattr(run_arm.initialize, 'claim_smoke', lambda path: claims.append('smoke'))
    def fake(ledger, name, module, args, **limits):
        calls.append(name)
        result = {'passed': not name.endswith(failed)}
        write(Path(ledger)/f'{name}.receipt.json', result)
        return result
    monkeypatch.setattr(run_arm.execute, 'run', fake)
    with pytest.raises(RuntimeError, match='worker or frozen resource'):
        run_arm.run(study, tmp_path/'base', 'sequential', ledger=ledger)
    outcome = read(root/'run-arm.json')
    assert not outcome['completed'] and not outcome['protected_evaluation_opened']
    assert not outcome['other_arm_recipe_changed'] and calls[-1] == f'sequential-{failed}'
    assert claims == ([] if failed == 'smoke-zero' else ['smoke'])
    assert not (study/'original-joint').exists()
    with pytest.raises(ValueError, match='retried'):
        run_arm.run(study, tmp_path/'base', 'sequential', ledger=ledger)


def test_fixed_schedule_has_only_development_and_no_step64_selection(tmp_path, monkeypatch):
    study, root, ledger = prepared(tmp_path)
    calls, claims = [], []
    monkeypatch.setattr(run_arm.initialize, 'claim_smoke', lambda path: claims.append('smoke'))
    monkeypatch.setattr(run_arm.initialize, 'claim_main', lambda path: claims.append('main'))
    monkeypatch.setattr(run_arm.initialize, 'authorize_main', lambda *args, **kwargs: {'passed': True})
    monkeypatch.setattr(run_arm.smoke, 'validate_gate', lambda *args: {'passed': True})
    monkeypatch.setattr(run_arm, 'validate_worker', lambda *args, **kwargs: {})
    import experiments.phase_iiis_tool_transfer.study as host_study
    monkeypatch.setattr(host_study, 'validate_main_checkpoint', lambda *args: {'passed': True})
    def fake(ledger, name, module, args, **limits):
        calls.append((name, module, list(args)))
        out = Path(args[args.index('--out')+1])
        if module.endswith('.train'):
            smoke = args[args.index('--mode')+1] == 'smoke'
            steps = [16] if smoke else [64, 128]
            checkpoints = []
            for step in steps:
                checkpoint = out/'checkpoints'/f'step-{step:04d}'
                checkpoint.mkdir(parents=True)
                (checkpoint/'adapter.safetensors').write_bytes(f'fixture-step{step}'.encode())
                write(checkpoint/'receipt.json', {'step': step, 'sha256': file_hash(checkpoint/'adapter.safetensors')})
                checkpoints.append(str(checkpoint))
            write(out/'completed.json', {'checkpoints': checkpoints})
        elif module.endswith('.evaluate'):
            write(out, {'mocked': True})
        result = {'passed': True}
        write(Path(ledger)/f'{name}.receipt.json', result)
        return result
    monkeypatch.setattr(run_arm.execute, 'run', fake)
    outcome = run_arm.run(study, tmp_path/'base', 'sequential', ledger=ledger)
    assert outcome['status'] == 'DEVELOPMENT_MEASURED' and outcome['behavioral_qualification'] is None
    assert not outcome['protected_evaluation_opened'] and claims == ['smoke', 'main']
    assert len(calls) == 8
    assert [name for name, _, _ in calls][-4:] == ['sequential-development-64-tool',
        'sequential-development-64-decimal', 'sequential-development-128-tool', 'sequential-development-128-decimal']
    assert outcome['development']['64']['diagnostic_only'] is True
    assert outcome['development']['128']['diagnostic_only'] is False
    assert all('--stage' not in args or args[args.index('--stage')+1] in {'zero', 'reload', 'development'} for _, _, args in calls)
