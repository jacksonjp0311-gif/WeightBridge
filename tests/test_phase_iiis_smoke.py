"""Smoke gate replay binds the actual saved adapter and immutable parent."""
from copy import deepcopy
import json

import pytest

from experiments.phase_iiis_tool_transfer.smoke import exact_predictions, validate_gate
from weightbridge.core import file_hash


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def evidence(tmp_path):
    protocol = write(tmp_path/'protocol.json', {'checkpoint_sha256': 'a'*64})
    identity = {'protocol_sha256': file_hash(protocol), 'parent_sha256': 'a'*64,
        'authorization_sha256': 'b'*64, 'bundle_sha256': 'c'*64, 'base_unchanged': True}
    measurement = {task: {'endpoint': {'version': task}, 'rows': [
        {'id': task, 'prompt': 'Q', 'generated_token_ids': [1, 2], 'text': 'x', 'stop_reason': 'eos'}]}
        for task in ('tool', 'decimal')}
    loss = {'tokenization': [{'id': 'a', 'prefix_ids': [1], 'target_ids': [2]}],
        'row_ids': ['a'], 'mean_token_nll': 1.}
    zero = write(tmp_path/'zero.json', {**identity, 'passed': True, 'matched_loss': loss,
        'parent_measurement': measurement, 'zero_measurement': measurement, 'initial_factor_sha256': 'd'*64})
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'adapter.safetensors').write_bytes(b'exact fixture adapter')
    sha = file_hash(checkpoint/'adapter.safetensors')
    write(checkpoint/'receipt.json', {'sha256': sha, 'factor_sha256': 'f'*64,
        'configuration': {'checkpoint_sha256': 'a'*64}})
    trained = write(tmp_path/'completed.json', {**identity, 'status': 'COMPLETED',
        'checkpoints': [str(checkpoint)], 'counts': {'optimizer_steps': 16},
        'all_factors_changed': True, 'all_factors_received_nonzero_gradient': True,
        'initial_factor_sha256': 'd'*64, 'parameter_changes': [
            {'changed_entries': 1, 'norm': .01, 'nonzero_gradient_steps': 1} for _ in range(372)]})
    reload = write(tmp_path/'reloaded.json', {**identity, 'passed': True, 'matched_loss': {**loss, 'mean_token_nll': .2},
        'disabled_measurement': measurement, 'adapter_sha256': sha, 'factor_sha256': 'f'*64})
    return protocol, zero, trained, reload


def test_smoke_replays_source_factors_and_disabled_parent(tmp_path):
    paths = evidence(tmp_path)
    gate = validate_gate(*paths)
    assert gate['passed'] and gate['checks']['measured_factor_changes']
    assert len(gate['source_receipts']) == 3


def test_smoke_rejects_forged_reloaded_parent_even_when_passed_true(tmp_path):
    paths = evidence(tmp_path)
    changed = json.loads(paths[-1].read_text())
    changed['parent_sha256'] = 'e'*64
    write(paths[-1], changed)
    with pytest.raises(ValueError, match='another protocol or parent'):
        validate_gate(*paths)


def test_smoke_cannot_hide_changed_adapter_bytes_or_failed_loss(tmp_path):
    paths = evidence(tmp_path)
    checkpoint = tmp_path/'checkpoint/adapter.safetensors'
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='exact saved adapter'):
        validate_gate(*paths)


def test_exact_smoke_replay_compares_native_tokens_and_task_identities():
    first = {task: {'endpoint': {}, 'rows': [{'id': task, 'prompt': 'Q', 'generated_token_ids': [1], 'text': 'x',
        'stop_reason': 'eos'}]} for task in ('tool', 'decimal')}
    other = deepcopy(first)
    other['tool']['rows'][0]['generated_token_ids'] = [2]
    assert not exact_predictions(first, other)
    other = deepcopy(first)
    other['decimal']['rows'][0]['id'] = 'other'
    assert not exact_predictions(first, other)
