"""Identity and non-nested resource accounting without model execution."""
import pytest

from experiments.phase_iiis_tool_transfer.integrity import checked_sha256, write_once
from experiments.phase_iiis_tool_transfer.execute import accounting, run


def test_exact_parent_identity_format():
    assert checked_sha256('a'*64) == 'a'*64
    for value in ('a'*62, 'G'*64, None, ' a'+'b'*62):
        with pytest.raises(ValueError):
            checked_sha256(value)


def test_write_once_and_non_nested_accounting(tmp_path):
    body = {'schema': 'weightbridge.phase_iiis.execution.v1', 'name': 'a', 'passed': True,
            'execution': {'resources': {'wall_seconds': 10., 'cpu_seconds': 50., 'peak_rss_bytes': 100}}}
    path = tmp_path/'a.receipt.json'
    write_once(path, body)
    with pytest.raises(FileExistsError):
        write_once(path, body)
    write_once(tmp_path/'b.receipt.json', {**body, 'name': 'b'})
    result = accounting(tmp_path)
    assert result['wall_seconds'] == 20 and result['cpu_seconds'] == 100
    assert result['peak_rss_bytes'] == 100


def test_unfinished_stage_blocks_new_owned_worker(tmp_path):
    write_once(tmp_path/'a.started.json', {})
    assert accounting(tmp_path)['unfinished'] == ['a.started.json']
    with pytest.raises(RuntimeError, match='interrupted or active'):
        run(tmp_path, 'b', 'experiments.phase_iiis_tool_transfer.evaluate', [], wall_limit=1, cpu_limit=1)


def test_unknown_resource_use_blocks_budget_reset(tmp_path):
    write_once(tmp_path/'a.receipt.json', {'schema': 'weightbridge.phase_iiis.execution.v1',
                                        'name': 'a', 'passed': False, 'execution': None})
    with pytest.raises(ValueError, match='unknown resource use'):
        accounting(tmp_path)
