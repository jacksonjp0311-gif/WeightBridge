"""Host initialization tests use synthetic fixtures; no real study is written."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.phase_iiis_tool_transfer import initialize, populations, data
from experiments.phase_iiis_tool_transfer.integrity import PARENTS
from experiments.phase_iiis_tool_transfer.train import checked_inputs
from experiments.phase_iiiq_baseline.integrity import ROOT, read
from weightbridge.core import file_hash


@pytest.fixture(scope='module')
def material():
    calibration = data.generate_tool_families('calibration', 10, 'initializer-disposable-tests')
    rows, manifest = populations.build_populations({split: f'initializer-fixture/{split}' for split in populations.SPLITS},
        used_decimal=set(), decimal_anchors=set(), calibration_rows=calibration)
    inventory = read(ROOT/'results/phase-iii-q-baseline/preflight/inventory-v1/architecture-inventory.json')['allowed']
    return rows, manifest, calibration, inventory


@pytest.fixture
def template(monkeypatch):
    monkeypatch.setattr(initialize, 'source_manifest', lambda: {'weightbridge/core.py': file_hash(ROOT/'weightbridge/core.py')})
    return {'initial_factor_sha256': 'a'*64,
        'endpoints': {'tool': {'cpu_threads': 6, 'batch_size': 8, 'max_input_tokens': 512, 'max_new_tokens': 96},
            'decimal': {'cpu_threads': 6, 'batch_size': 8, 'max_input_tokens': 128, 'max_new_tokens': 16}},
        'tool_completion': {'schema': 'weightbridge.native_tool_completion.v1',
            'end_token_id': 248046, 'end_token_surface': '<|im_end|>'},
        'resources': {'max_memory_bytes': 32*1024**3, 'mode_limits': {
            'main': {'max_wall_seconds': 7200., 'max_cpu_seconds': 40000., 'max_answer_tokens': 100000, 'max_input_tokens': 1000000},
            'smoke': {'max_wall_seconds': 900., 'max_cpu_seconds': 5000., 'max_answer_tokens': 20000, 'max_input_tokens': 100000}}}}


def test_host_freeze_separates_banks_and_binds_identical_training_rows(tmp_path, material, template):
    rows, manifest, calibration, inventory = material
    out, host = tmp_path/'study', tmp_path/'host'
    result = initialize.freeze(out, host, template, rows, manifest, inventory, calibration_rows=calibration)
    assert not result['formal_evaluation_authorized'] and not result['lineage_authority']
    stored = read(out/'manifest.json')
    for split in populations.SPLITS:
        for task, entry in stored['splits'][split]['files'].items():
            path = Path(entry['path'])
            assert file_hash(path) == entry['sha256']
            if split in populations.HOST_SPLITS:
                assert path.is_relative_to(host)
                assert not path.is_relative_to(out)
    first, second = [read(out/arm/'training-bundle.json') for arm in PARENTS]
    assert first['protocol_sha256'] != second['protocol_sha256']
    assert {k: v for k, v in first.items() if k != 'protocol_sha256'} == {k: v for k, v in second.items() if k != 'protocol_sha256'}
    assert len(first['rows']) == 2048 and first['split_ids'] == ['train', 'replay']
    for arm, parent in PARENTS.items():
        root = out/arm
        protocol = read(root/'protocol.json')
        assert protocol['checkpoint_sha256'] == parent and protocol['initial_factor_sha256'] == 'a'*64
        assert protocol['fixed_finalist_step'] == 128 and protocol['diagnostic_only_steps'] == [64]
        assert not (root/'main-authorization.json').exists()
        assert not read(root/'main-authorization-plan.json')['main_launch_authorized']
        assert len(read(root/'smoke-bundle.json')['rows']) == 96
        checked_inputs(root/'protocol.json', root/'smoke-authorization.json', root/'smoke-bundle.json', mode='smoke')
    assert read(out/'sequential/prospective-data-order.json') == read(out/'original-joint/prospective-data-order.json')
    with pytest.raises(ValueError, match='new private path'):
        initialize.freeze(out, host, template, rows, manifest, inventory)


def test_disposable_timing_has_no_main_or_formal_authority(tmp_path, material, template):
    rows, _, calibration, inventory = material
    timing_rows = calibration + [row for row in rows['smoke'] if row['task'] == 'decimal']
    out = tmp_path/'timing'
    result = initialize.freeze_timing(out, template, timing_rows, inventory)
    protocol, auth, bundle, settings = checked_inputs(out/'protocol.json', out/'authorization.json', out/'training-bundle.json', mode='timing')
    assert not protocol['training_authorized'] and not protocol['formal_banks_created']
    assert bundle['split_ids'] == ['timing'] and auth.max_steps_per_run == 1
    assert not auth.allow_merge and auth.max_wall_seconds_per_run == 1200
    assert auth.max_cpu_seconds_per_run == 8000 and auth.max_memory_bytes == 32*1024**3
    assert read(out/'manifest.json')['rows_sha256'] == file_hash(out/'rows.json')
    assert result['manifest_sha256'] == file_hash(out/'manifest.json')
    assert not (out/'main-authorization.json').exists()


def test_no_nested_host_banks_or_mutated_manifest(tmp_path, material, template):
    rows, manifest, calibration, inventory = material
    with pytest.raises(ValueError, match='disjoint'):
        initialize.freeze(tmp_path/'study', tmp_path/'study/hidden', template, rows, manifest, inventory)
    altered = deepcopy(manifest)
    altered['splits']['train']['rows_sha256'] = '0'*64
    with pytest.raises(ValueError, match='logical commitments'):
        initialize.freeze(tmp_path/'study', tmp_path/'host', template, rows, altered, inventory)
    assert not (tmp_path/'study').exists()


def test_missing_initialization_commitment_or_resource_bound_refuses_before_write(tmp_path, material, template):
    rows, manifest, _, inventory = material
    altered = deepcopy(template)
    altered.pop('initial_factor_sha256')
    with pytest.raises(ValueError, match='64 lowercase'):
        initialize.freeze(tmp_path/'study', tmp_path/'host', altered, rows, manifest, inventory)
    altered = deepcopy(template)
    altered['resources']['mode_limits'].pop('smoke')
    with pytest.raises(ValueError, match='resource limits'):
        initialize.freeze(tmp_path/'study', tmp_path/'host', altered, rows, manifest, inventory)
    assert not (tmp_path/'study').exists()


def test_parent_inventory_geometry_must_match(tmp_path, material, template):
    rows, manifest, _, inventory = material
    inv = {arm: {'checkpoint_sha256': parent, 'allowed': deepcopy(inventory)} for arm, parent in PARENTS.items()}
    inv['original-joint']['allowed'][0]['weight_shape'][0] += 1
    with pytest.raises(ValueError, match='186-module'):
        initialize.freeze(tmp_path/'study', tmp_path/'host', template, rows, manifest, inv)
    assert not (tmp_path/'study').exists()


def _smoke_receipts(root):
    from experiments.phase_iiis_tool_transfer.smoke import validate_gate
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path
    protocol_path = root/'protocol.json'
    protocol = read(protocol_path)
    identity = {'protocol_sha256': file_hash(protocol_path), 'parent_sha256': protocol['checkpoint_sha256'],
        'authorization_sha256': file_hash(root/'smoke-authorization.json'),
        'bundle_sha256': file_hash(root/'smoke-bundle.json'), 'base_unchanged': True,
        'resources': {'wall_seconds': 1., 'cpu_seconds': 1., 'peak_rss_bytes': 100}}
    measurement = {task: {'endpoint': {'v': task}, 'rows': [
        {'id': task, 'prompt': 'Q', 'generated_token_ids': [1], 'text': 'x', 'stop_reason': 'eos'}]}
        for task in ('tool', 'decimal')}
    loss = {'tokenization': [{'id': 'a', 'prefix_ids': [1], 'target_ids': [2]}], 'row_ids': ['a'], 'mean_token_nll': 1.}
    zero = write(root/'smoke/zero/smoke-check.json', {**identity, 'stage': 'zero', 'passed': True,
        'matched_loss': loss, 'parent_measurement': measurement, 'zero_measurement': measurement,
        'initial_factor_sha256': protocol['initial_factor_sha256']})
    checkpoint = root/'smoke/training/checkpoints/step-0016'
    checkpoint.mkdir(parents=True)
    (checkpoint/'adapter.safetensors').write_bytes(b'fixture-only-adapter')
    sha = file_hash(checkpoint/'adapter.safetensors')
    write(checkpoint/'receipt.json', {'sha256': sha, 'factor_sha256': 'f'*64,
        'configuration': {'checkpoint_sha256': protocol['checkpoint_sha256']}})
    completed = write(root/'smoke/training/completed.json', {**identity, 'mode': 'smoke', 'status': 'COMPLETED',
        'checkpoints': [str(checkpoint)], 'counts': {'optimizer_steps': 16, 'examples_seen': 256, 'forward_calls': 64, 'backward_calls': 64},
        'all_factors_changed': True, 'all_factors_received_nonzero_gradient': True,
        'initial_factor_sha256': protocol['initial_factor_sha256'], 'parameter_changes': [
            {'changed_entries': 1, 'norm': .01, 'nonzero_gradient_steps': 1} for _ in range(372)]})
    reload = write(root/'smoke/reload/smoke-check.json', {**identity, 'stage': 'reload', 'passed': True,
        'matched_loss': {**loss, 'mean_token_nll': .1}, 'disabled_measurement': measurement,
        'adapter_sha256': sha, 'factor_sha256': 'f'*64})
    gate = write(root/'smoke-gate.json', validate_gate(protocol_path, zero, completed, reload))
    ledger = root.parent/'fixture-ledger'
    ledger.mkdir(exist_ok=True)
    executions = []
    for kind, inner in (('zero', zero), ('train', completed), ('reload', reload)):
        module = 'train' if kind == 'train' else 'smoke'
        command = ['python', '-m', f'experiments.phase_iiis_tool_transfer.{module}',
            '--mode' if kind == 'train' else '--stage', 'smoke' if kind == 'train' else kind,
            '--protocol', str(protocol_path), '--authorization', str(root/'smoke-authorization.json'),
            '--bundle', str(root/'smoke-bundle.json'), '--out', str(inner.parent)]
        log = ledger/f'{kind}.log'
        log.write_text('fixture worker exited', encoding='utf-8')
        outer = {'schema': 'weightbridge.phase_iiis.execution.v1', 'name': kind, 'command': command,
            'passed': True, 'error': None, 'log_sha256': file_hash(log),
            'temporary_sleep_request': {'requested': False},
            'limits': {'wall_limit': 900, 'cpu_limit': 5000, 'memory_limit': 32*1024**3},
            'execution': {'exit_code': 0, 'budget_failure': None,
                'processes': [{'cpu_seconds': 1.5, 'peak_rss_bytes': 100}],
                'resources': {'wall_seconds': 4., 'cpu_seconds': 2., 'host_cpu_seconds': .5,
                    'observed_tree_cpu_seconds': 1.5, 'peak_rss_bytes': 100,
                    'cpu_accounting_complete_for_observed_processes': True}}}
        executions.append(write(ledger/f'{kind}.receipt.json', outer))
    return gate, executions


def test_main_admission_replays_smoke_and_consumes_one_launch(tmp_path, material, template):
    rows, manifest, calibration, inventory = material
    out = tmp_path/'study'
    initialize.freeze(out, tmp_path/'host', template, rows, manifest, inventory, calibration_rows=calibration)
    root = out/'sequential'
    gate, executions = _smoke_receipts(root)
    auth = initialize.authorize_main(root, gate, execution_receipts=executions)
    assert auth['smoke_gate_sha256'] == file_hash(gate)
    assert read(root/'main-admission.json')['smoke_resources']['cpu_seconds'] == 6
    claim = initialize.claim_main(root)
    assert not claim['retries_authorized'] and claim['interruption_disposition'] == 'INTERRUPTED_INCOMPLETE'
    before = {str(path): file_hash(path) for path in root.rglob('*') if path.is_file()}
    replay = initialize.validate_main_admission(root)
    assert replay['passed'] and replay['parent_sha256'] == PARENTS['sequential']
    assert replay['input_sha256'][str(root/'main-launch-claim.json')] == file_hash(root/'main-launch-claim.json')
    assert before == {str(path): file_hash(path) for path in root.rglob('*') if path.is_file()}
    # Even a launch interrupted before output creation has consumed the slot.
    assert not (root/'training').exists()
    with pytest.raises(ValueError, match='one-shot'):
        initialize.claim_main(root)
    with pytest.raises(ValueError, match='fresh one-shot'):
        initialize.authorize_main(root, gate, execution_receipts=executions)


def test_inner_smoke_pass_cannot_override_failed_outer_or_resource_totals(tmp_path, material, template):
    rows, manifest, calibration, inventory = material
    out = tmp_path/'study'
    initialize.freeze(out, tmp_path/'host', template, rows, manifest, inventory, calibration_rows=calibration)
    root = out/'sequential'
    gate, executions = _smoke_receipts(root)
    altered = read(executions[1])
    altered['passed'] = False
    executions[1].write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError, match='outer smoke execution'):
        initialize.authorize_main(root, gate, execution_receipts=executions)
    assert not (root/'main-authorization.json').exists()
    altered['passed'] = True
    altered['execution']['resources']['cpu_seconds'] = 0
    executions[1].write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError, match='totals do not replay'):
        initialize.authorize_main(root, gate, execution_receipts=executions)


def test_smoke_optimizer_launch_is_one_shot_before_child_exists(tmp_path, material, template):
    rows, manifest, calibration, inventory = material
    out = tmp_path/'study'
    initialize.freeze(out, tmp_path/'host', template, rows, manifest, inventory, calibration_rows=calibration)
    root = out/'original-joint'
    value = initialize.claim_smoke(root)
    assert value['out'] == str(root/'smoke/training')
    with pytest.raises(ValueError, match='one-shot'):
        initialize.claim_smoke(root)
    assert not (root/'smoke/training').exists()


def test_main_admission_replay_refuses_edited_claim(tmp_path, material, template):
    rows, manifest, calibration, inventory = material
    out = tmp_path/'study'
    initialize.freeze(out, tmp_path/'host', template, rows, manifest, inventory, calibration_rows=calibration)
    root = out/'sequential'
    gate, executions = _smoke_receipts(root)
    initialize.authorize_main(root, gate, execution_receipts=executions)
    initialize.claim_main(root)
    path = root/'main-launch-claim.json'
    changed = read(path)
    changed['out'] = str(root/'replacement')
    path.write_text(json.dumps(changed), encoding='utf-8')
    with pytest.raises(ValueError, match='launch claim'):
        initialize.validate_main_admission(root)
