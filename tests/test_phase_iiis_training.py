"""S parent/task policy, unchanged geometry and training-only timing bounds."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase_iiis_tool_transfer import train
from experiments.phase_iiis_tool_transfer.preflight import timing_indices
from experiments.phase_iiiq_baseline.integrity import ROOT, read
from weightbridge.core import file_hash
from weightbridge.adaptation.completion_training import SupervisedCompletion


def config(parent='7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8'):
    q = read(ROOT/'experiments/phase_iiiq_baseline/protocol.json')
    settings = {**q['training'], 'learning_rates': [.0003], 'seed': 606118}
    return {'schema': 'weightbridge.phase_iiis.protocol.v1', 'checkpoint_sha256': parent,
        'source_sha256': {'weightbridge/core.py': file_hash(ROOT/'weightbridge/core.py')},
        'training': settings, 'smoke_training': {**settings, 'optimizer_steps': 16, 'checkpoint_steps': [16], 'epochs': 8},
        'resources': {'max_memory_bytes': 32*1024**3,
            'mode_limits': {mode: {'max_wall_seconds': 1000., 'max_cpu_seconds': 5000.,
                'max_answer_tokens': 100000, 'max_input_tokens': 1000000} for mode in ('main', 'smoke', 'timing')}},
        'endpoints': {'tool': {'cpu_threads': 6}, 'decimal': {'cpu_threads': 6}}}


def inventory():
    return sorted(read(ROOT/'results/phase-iii-q-baseline/preflight/inventory-v1/architecture-inventory.json')['allowed'],
        key=lambda row: row['name'])


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def inputs(tmp_path, mode='main', parent=None):
    cfg = config() if parent is None else config(parent)
    protocol = write(tmp_path/'protocol.json', cfg)
    rows = [{'id': f'{i:04d}', 'task': 'tool' if i < 1024 else 'decimal',
        'category': 'tool' if i < 1024 else 'decimal-target' if i < 1536 else 'general'} for i in range(2048)]
    bundle = write(tmp_path/'bundle.json', {'schema': 'weightbridge.phase_iiis.training_bundle.v1',
        'protocol_sha256': file_hash(protocol), 'rows': rows,
        'manifest_sha256': 'f'*64, 'split_ids': ['train', 'replay'] if mode == 'main' else [mode]})
    policy = train.authorization(cfg, inventory(), bundle, mode=mode)
    auth = write(tmp_path/'authorization.json', policy.to_dict())
    return protocol, auth, bundle


@pytest.mark.parametrize('parent', ['7d4532858cf839d05ec34da088114df47238322f86f5e6e5c6693594a21a03b8',
    '2d9102982dc0f0fdf3397308cd397bc2d5239eeb99ef6297e5f09f6f6deba158'])
@pytest.mark.parametrize('mode', ['main', 'smoke', 'timing'])
def test_explicit_parents_share_exact_adapter_policy_without_global_substitution(tmp_path, parent, mode):
    protocol, auth, bundle = inputs(tmp_path, mode, parent)
    cfg, policy, _, _ = train.checked_inputs(protocol, auth, bundle, mode=mode)
    assert cfg['checkpoint_sha256'] == policy.parent_sha256 == parent
    assert policy.max_trainable_parameters == 5411328 and len(policy.allowed_modules) == 186
    assert policy.rank == 8 and policy.alpha == 16
    assert policy.max_steps_per_run == {'main': 128, 'smoke': 16, 'timing': 1}[mode]
    assert policy.allow_merge is (mode == 'main')


def test_sealed_material_and_policy_extensions_are_refused(tmp_path):
    protocol, auth, bundle = inputs(tmp_path)
    value = read(bundle)
    value['split_ids'].append('hidden')
    write(bundle, value)
    changed = read(auth)
    changed['training_bundle_sha256'] = file_hash(bundle)
    write(auth, changed)
    with pytest.raises(ValueError, match='fitting bundle'):
        train.checked_inputs(protocol, auth, bundle, mode='main')
    value['split_ids'] = ['train', 'replay']
    write(bundle, value)
    changed['training_bundle_sha256'] = file_hash(bundle)
    changed['max_cpu_seconds_per_run'] += 1
    write(auth, changed)
    with pytest.raises(ValueError, match='stage limits'):
        train.checked_inputs(protocol, auth, bundle, mode='main')


def test_different_parent_never_passes_explicit_runtime_identity(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import weightbridge.adaptation.model_adapter as adapters
    fake = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_sha256=lambda: 'a'*64))
    monkeypatch.setattr(adapters, 'select_hf_adapter', lambda base: fake)
    cfg = config()
    with pytest.raises(ValueError, match='explicitly authorized'):
        with train.runtime(tmp_path, cfg, expected_parent=cfg['checkpoint_sha256']):
            pytest.fail('Wrong checkpoint was loaded')
    with pytest.raises(ValueError, match='Explicit parent'):
        with train.runtime(tmp_path, cfg, expected_parent='a'*64):
            pytest.fail('Protocol parent was silently replaced')


def test_data_order_and_token_exposure_do_not_depend_on_parent():
    rows = [{'id': f'{i:04d}', 'task': 'tool' if i%2 else 'decimal'} for i in range(2048)]
    values = [SupervisedCompletion((1, 2), (3,)*(3 if i%2 else 1), '', '', 'a'*64) for i in range(2048)]
    a, b = config(), config('b'*64)
    first = train.exposure_receipt(rows, values, a['training'])
    second = train.exposure_receipt(rows, values, b['training'])
    assert first == second
    assert first[1]['examples_seen'] == 2048 and first[1]['answer_tokens'] == 4096
    assert first[1]['task_examples'] == {'tool': 1024, 'decimal': 1024}
    with pytest.raises(ValueError, match='Insufficient'):
        train.exposure_receipt(rows[:10], values[:10], a['training'])


def test_timing_prospectively_includes_longest_and_both_tasks_without_outcomes():
    rows = [{'id': str(i), 'task': 'decimal' if i%2 else 'tool'} for i in range(12)]
    values = [SupervisedCompletion((1,)*(i+1), (2,), '', '', 'a'*64) for i in range(12)]
    groups = timing_indices(rows, values)
    assert len(groups) == 4 and all(len(g) == 4 for g in groups)
    assert groups[0] == groups[1]
    assert {rows[i]['task'] for i in groups[0]} == {'tool', 'decimal'}
    assert 11 in groups[2]
    with pytest.raises(ValueError):
        timing_indices(rows[:4], values[:4])


def test_authorization_refuses_module_omission_or_rank_geometry_change(tmp_path):
    bundle = write(tmp_path/'bundle.json', {})
    with pytest.raises(ValueError, match='186-module'):
        train.authorization(config(), inventory()[:-1], bundle)
    altered = deepcopy(inventory())
    altered[0]['weight_shape'][0] += 1
    with pytest.raises(ValueError, match='186-module'):
        train.authorization(config(), altered, bundle)


def test_same_fresh_factor_hash_does_not_depend_on_parent_weights():
    torch = pytest.importorskip('torch')
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, LoRATarget, _hash_parameter
    sessions = []
    for seed, parent in ((11, 'a'*64), (97, 'b'*64)):
        torch.manual_seed(seed)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.layers[0].mlp = torch.nn.Module()
        model.layers[0].mlp.proj = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        name = 'layers.0.mlp.proj'
        target = LoRATarget(name, name+'.weight', 0, 'mlp', (3, 4), 'BF16', _hash_parameter(model.layers[0].mlp.proj.weight))
        session = SupervisedLoRA(model, (target,), checkpoint_sha256=parent, rank=8, alpha=16, seed=606118)
        sessions.append((model, session))
    assert sessions[0][1].factor_sha256() == sessions[1][1].factor_sha256()
    assert sessions[0][1].configuration()['checkpoint_sha256'] != sessions[1][1].configuration()['checkpoint_sha256']
