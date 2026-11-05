"""Replay already scored public B evidence; these tests run no model or fitting."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from experiments.phase_iiip import reference_analysis as analysis
from experiments.phase_iiip import reference_import as importer
from weightbridge.core import file_hash

ARCHIVE = Path(__file__).resolve().parents[1] / 'results/phase-iii-p/reference-v1'


def read(root, name):
    return json.loads((root / name).read_bytes())


def forge(root, name, value):
    # Deliberately bypass immutable creation only in an isolated damaged-copy
    # fixture. Never replace a real experiment or its historical expectations.
    (root / name).write_bytes((json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def rebind_comparison(root, comparison):
    forge(root, 'evaluation/comparison.json', comparison)
    execution = read(root, 'execution.json')
    execution['comparison_sha256'] = file_hash(root / 'evaluation/comparison.json')
    forge(root, 'execution.json', execution)


def rebind_worker(root, worker, *, name='reference'):
    forge(root, f'evaluation/{name}.json', worker)
    comparison = read(root, 'evaluation/comparison.json')
    comparison['worker_sha256'][name] = file_hash(root / f'evaluation/{name}.json')
    rebind_comparison(root, comparison)


@pytest.fixture
def damaged(tmp_path):
    destination = tmp_path / 'damaged-copy'
    shutil.copytree(ARCHIVE, destination)
    return destination


def test_archived_reference_replays_without_conflating_presentation_and_rule_success():
    result = analysis.summarize(ARCHIVE)
    target = result['target_presentations']
    assert (target['parent_correct'], target['reference_correct'], target['presentations']) == (3, 8, 48)
    assert len(target['corrected_presentations']) == 6 and len(target['regressed_presentations']) == 1
    assert result['all_presentations']['parent_valid'] == result['all_presentations']['reference_valid'] == 72
    gate = result['strict_and_family_gate']
    assert gate['parent_target_correct'] == gate['candidate_target_correct'] == 0
    assert gate['target_comparisons'] == 12 and gate['complete_family_gain'] == 0 and not gate['passed']
    assert len(result['families']) == 6 and all(len(f['rows']) == 8 for f in result['families'].values())
    assert result['presentation_gain_interval']['clusters'] == 6
    assert result['presentation_gain_interval']['point_estimate'] == pytest.approx(5 / 48)
    assert result['presentation_gain_interval']['low'] == pytest.approx(1 / 48)
    assert result['presentation_gain_interval']['high'] == pytest.approx(11 / 48)
    assert result['strict_comparison_gain_interval']['low'] == result['strict_comparison_gain_interval']['high'] == 0
    assert not result['lineage_accepted'] and not result['new_weightbridge_learning_claimed']
    assert not result['long_benchmark_reproduction']


def test_resources_count_outer_wrappers_once_and_report_unknown_costs():
    result = analysis.summarize(ARCHIVE)
    compute = result['compute']
    assert compute['recorded_cpu_seconds_lower_bound'] == 503.234375
    assert compute['model_forward_calls'] == 216 and compute['model_backward_calls'] == 0
    assert compute['generated_responses'] == 144 and compute['generated_native_tokens'] == 1728
    assert compute['recorded_serial_wall_seconds'] == pytest.approx(150.1556993999402)
    assert compute['preliminary_source_inspection_cpu_seconds'] is None
    assert not compute['all_phase_costs_fully_accounted']
    assert 'reconstruction_function' not in compute['components']
    assert 'assessment_host_nested' not in compute['components']
    assert result['patch_receipt_replay']['every_exported_value_matches_pinned_npz']
    assert result['patch_receipt_replay']['host_before_after_delta_norm_replayed'] == pytest.approx(2.2394522487131594)


def test_semantic_flags_cannot_be_forged_even_with_updated_enclosing_hashes(damaged):
    worker = read(damaged, 'evaluation/reference.json')
    worker['measurement']['rows'][0]['correct'] = True
    rebind_worker(damaged, worker)
    with pytest.raises(ValueError, match='inconsistent parsing or correctness'):
        analysis.summarize(damaged)


def test_hidden_regression_in_summary_is_rejected_after_hash_rebinding(damaged):
    comparison = read(damaged, 'evaluation/comparison.json')
    comparison['paired']['regressions'] = []
    rebind_comparison(damaged, comparison)
    with pytest.raises(ValueError, match='hides a correction, regression'):
        analysis.summarize(damaged)


def test_checkpoint_identity_swap_is_rejected_even_with_updated_worker_hash(damaged):
    worker = read(damaged, 'evaluation/reference.json')
    worker['checkpoint_sha256'] = 'f' * 64
    rebind_worker(damaged, worker)
    with pytest.raises(ValueError, match='worker does not bind its model'):
        analysis.summarize(damaged)


def test_generation_work_tampering_does_not_become_recorded_compute(damaged):
    worker = read(damaged, 'evaluation/reference.json')
    worker['measurement']['resources']['generated_examples'] += 1
    rebind_worker(damaged, worker)
    with pytest.raises(ValueError, match='generation work does not follow'):
        analysis.summarize(damaged)


def test_pinned_external_source_mutation_is_not_accepted_as_fresh_provenance(damaged):
    (damaged / 'source' / importer.EDIT).write_bytes(b'forged sparse edit')
    with pytest.raises(ValueError, match='Pinned external source hash mismatch'):
        analysis.summarize(damaged)


def test_pinned_model_revision_metadata_cannot_change_after_compatibility(damaged):
    metadata = read(damaged, 'source/qwen-pinned-api.json')
    metadata['sha'] = 'f' * 40
    forge(damaged, 'source/qwen-pinned-api.json', metadata)
    with pytest.raises(ValueError, match='Pinned model revision metadata'):
        analysis.summarize(damaged)


def test_norm_is_recomputed_from_native_host_values_not_just_two_matching_summaries():
    reconstruction = read(ARCHIVE, 'reconstruction.json')
    host = deepcopy(reconstruction['host_patch_receipt'])
    host['changes'][0]['before'] = 0.  # Native BF16 but not the observed source value.
    manifest, arrays, _ = importer._inputs(ARCHIVE / 'source')
    authorization = json.loads(analysis.PROTOCOL.with_name('reference_authorization.json').read_bytes())
    with pytest.raises(ValueError, match='stored update norm does not replay'):
        analysis._verify_patch_receipt(host, reconstruction['compatibility'], manifest, arrays, authorization)


def test_family_interval_never_treats_partial_renderings_as_independent_samples():
    parent = {'rows': [{'id': f'{f}-{i}', 'group': 'target', 'pair_id': f, 'correct': False}
                       for f in ('a', 'b') for i in range(8)]}
    candidate = deepcopy(parent)
    for row in candidate['rows'][:8]: row['correct'] = True
    result = analysis.clustered_presentation_interval(parent, candidate)
    assert result['clusters'] == 2 and result['point_estimate'] == .5
    candidate['rows'][0]['id'] = 'different'
    with pytest.raises(ValueError, match='identical presentation order'):
        analysis.clustered_presentation_interval(parent, candidate)
    parent['rows'].pop(); candidate = deepcopy(parent)
    with pytest.raises(ValueError, match='complete eight-response'):
        analysis.clustered_presentation_interval(parent, candidate)
