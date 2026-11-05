"""Deterministic forecast arithmetic and failed-preflight refusal; no inference."""
import json
from pathlib import Path

import pytest

from experiments.phase_iiis_tool_transfer.freeze_study import resource_forecast
from weightbridge.core import file_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


@pytest.fixture
def preflight(tmp_path):
    root = tmp_path/'preflight'
    protocol = write(root/'timing-v1/protocol.json', {'fixture': True})
    batches = [{'mode': 'all' if i == 0 else 'tail', 'row_ids': ['a', 'b', 'c', 'd'],
        'category_accounting': i >= 2, 'forward': {'wall_seconds': 1.}, 'backward': {'wall_seconds': 1.},
        'input_shape': [4, 415 if i < 3 else 50]} for i in range(4)]
    timing = {'passed': True, 'counts': {'optimizer_steps': 0, 'forward_calls': 4, 'backward_calls': 4},
        'equivalence': {'passed': True}, 'factors_unchanged': True, 'base_unchanged': True,
        'batches': batches, 'protocol_sha256': file_hash(protocol)}
    write(root/'timing-v1/measurement/timing.json', timing)
    def outer(name, module, out, protocol_path, wall):
        log = root/'execution'/f'{name}.log'
        log.parent.mkdir(exist_ok=True)
        log.write_text('fixture worker completed', encoding='utf-8')
        value = {'schema': 'weightbridge.phase_iiis.execution.v1', 'name': name,
            'command': ['python', '-m', module, '--protocol', str(protocol_path), '--out', str(out)],
            'passed': True, 'error': None, 'log_sha256': file_hash(log), 'temporary_sleep_request': {'requested': False},
            'limits': {'wall_limit': 100000, 'cpu_limit': 100000, 'memory_limit': 100000},
            'execution': {'exit_code': 0, 'budget_failure': None, 'processes': [{'cpu_seconds': 2., 'peak_rss_bytes': 100}],
                'resources': {'wall_seconds': wall, 'cpu_seconds': 3., 'host_cpu_seconds': 1.,
                    'observed_tree_cpu_seconds': 2., 'peak_rss_bytes': 100,
                    'cpu_accounting_complete_for_observed_processes': True}}}
        write(root/'execution'/f'{name}.receipt.json', value)
    outer('timing-sequential', 'experiments.phase_iiis_tool_transfer.preflight',
        root/'timing-v1/measurement', protocol, 10.)
    for arm in ('sequential', 'original'):
        actual = arm if arm == 'sequential' else 'original-joint'
        proto = write(root/'calibration-v1'/f'{actual}-protocol.json', {'fixture': True})
        for task, count in (('tool', 80), ('decimal', 72)):
            worker = write(root/'calibration-v1'/f'{actual}-{task}.json', {'summary': {'presentations': count}})
            outer(f'calibration-{arm}-{task}', 'experiments.phase_iiis_tool_transfer.evaluate', worker, proto, float(count))
    return root


def test_forecast_counts_both_complete_paths_and_actual_prior_cost(preflight):
    value = resource_forecast(preflight, intended_max_sequence_length=415)
    stages = value['stage_estimates_seconds']
    assert stages['two_main_training'] == 2*(512*2+212)
    assert stages['two_formal_banks_all_models_replay_rollback'] == 4096+9216
    assert stages['already_executed'] == 10+2*80+2*72
    assert value['estimated_model_wall_seconds'] == sum(stages.values())
    assert value['estimate_with_margin_seconds'] == pytest.approx(sum(stages.values())*1.10)
    assert len(value['preflight_receipt_sha256']) == 5 and value['matched_two_arm_feasible']


@pytest.mark.parametrize('change', ['outer_fail', 'outer_exit', 'gradient_fail', 'extra_backward', 'wrong_population'])
def test_failed_or_unmatched_preflight_cannot_authorize_forecast(preflight, change):
    path = preflight/'timing-v1/measurement/timing.json'
    if change in {'outer_fail', 'outer_exit'}:
        path = preflight/'execution/calibration-original-tool.receipt.json'
    elif change == 'wrong_population':
        path = preflight/'calibration-v1/original-joint-tool.json'
    value = json.loads(path.read_bytes())
    if change == 'outer_fail': value['passed'] = False
    elif change == 'outer_exit': value['execution']['exit_code'] = 1
    elif change == 'gradient_fail': value['equivalence']['passed'] = False
    elif change == 'extra_backward': value['counts']['backward_calls'] = 5
    else: value['summary']['presentations'] = 79
    write(path, value)
    with pytest.raises(ValueError):
        resource_forecast(preflight)


def test_longer_prospective_rows_are_costed_not_silently_truncated(preflight):
    base = resource_forecast(preflight, intended_max_sequence_length=415)
    long = resource_forecast(preflight, intended_max_sequence_length=830)
    assert long['training_length_scale'] == 4.
    assert long['measured_worst_tail_microbatch_seconds'] == base['measured_worst_tail_microbatch_seconds']
    assert long['forecast_tail_microbatch_seconds'] == 8.
    assert long['estimated_model_wall_seconds'] > base['estimated_model_wall_seconds']
    impossible = resource_forecast(preflight, intended_max_sequence_length=4000)
    assert not impossible['matched_two_arm_feasible']
