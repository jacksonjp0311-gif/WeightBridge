"""Tiny synthetic artifacts test import boundaries, not external model efficacy."""
from copy import deepcopy
import hashlib
import json
import struct

import numpy as np
import pytest

from experiments.phase_iiip import reference_import as ref
from weightbridge.core import EditRejected, file_hash, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _from_float64


@pytest.fixture
def release(tmp_path, monkeypatch):
    # Explicitly replace public pins only inside this deterministic fixture.
    # Production remains bound to the published hashes and actual Qwen parent.
    base, source = tmp_path / 'base', tmp_path / 'source'
    base.mkdir(); source.mkdir()
    weights = {name: np.array([[1., 2.], [3., 4.]]) + 4 * i for i, name in enumerate(ref.TENSOR_NAMES)}
    weights['untouched.weight'] = np.array([[.5, -.5]])
    blobs, header, offset = [], {}, 0
    for name, values in weights.items():
        blob = _from_float64(values, 'BF16')
        header[name] = {'dtype': 'BF16', 'shape': list(values.shape), 'data_offsets': [offset, offset + len(blob)]}
        offset += len(blob); blobs.append(blob)
    encoded = json.dumps(header, separators=(',', ':')).encode(); encoded += b' ' * (-len(encoded) % 8)
    (base / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(blobs))
    (base / 'config.json').write_text('{"model_type":"fixture"}')
    checkpoint = load_checkpoint(base)
    arrays, manifest = {}, {'changed_tensors': 2, 'changed_entries_total': 2, 'tensors': []}
    for i, name in enumerate(ref.TENSOR_NAMES):
        raw = checkpoint.read_raw(name); bits = np.frombuffer(raw, dtype='<u2').copy()
        index = 0 if i == 0 else 3
        value = np.frombuffer(_from_float64(np.array([weights[name].reshape(-1)[index] + .125]), 'BF16'), dtype='<u2')
        arrays[f't{i}_idx'] = np.array([index], dtype=np.int64)
        arrays[f't{i}_val'] = value
        bits[index] = value[0]
        manifest['tensors'].append({'name': name, 'shape': [2, 2], 'dtype': 'bfloat16', 'form': 'sparse',
            'changed_entries': 1, 'base_sha256': checkpoint.tensor_sha256(name),
            'edited_sha256': hashlib.sha256(bits.tobytes()).hexdigest()})
    mpath, epath = source / ref.MANIFEST, source / ref.EDIT
    mpath.parent.mkdir(parents=True)
    json_dump_atomic(mpath, manifest)
    np.savez(epath, **arrays)
    monkeypatch.setattr(ref, 'SOURCE_HASHES', {ref.MANIFEST: file_hash(mpath), ref.EDIT: file_hash(epath)})
    monkeypatch.setattr(ref, 'EXPORT_HASHES', {})
    monkeypatch.setattr(ref, 'EXPECTED_CHANGED', 2)
    monkeypatch.setattr(ref, 'BASE_SHA256', checkpoint.checkpoint_sha256())
    monkeypatch.setattr(ref, '_assets', lambda *args: {'scope': 'Synthetic fixture asset compatibility only'})
    authorization = {'schema': 'weightbridge.phase_iiip.reference_authorization.v1', 'track': 'B',
        'base_checkpoint_sha256': ref.BASE_SHA256, 'source_commit': ref.SOURCE_COMMIT, 'model_revision': ref.MODEL_REVISION,
        'manifest_sha256': file_hash(mpath), 'edit_sha256': file_hash(epath), 'expected_changed_entries': 2,
        'authority': 'Synthetic unit fixture authorization; no real model', 'lineage_promotion': False,
        'patch_policy': {'schema': 'weightbridge.policy.v1', 'allow_tensors': list(ref.TENSOR_NAMES),
            'max_entries': 2, 'max_fraction': 1., 'max_abs_delta': 1., 'max_tensor_bytes': 1024,
            'max_total_tensor_bytes': 2048}}
    authpath = tmp_path / 'authorization.json'; json_dump_atomic(authpath, authorization)
    return {'base': base, 'source': source, 'checkpoint': checkpoint, 'manifest': manifest,
        'arrays': arrays, 'authorization': authorization, 'authpath': authpath,
        'out': tmp_path / 'candidate', 'receipt': tmp_path / 'reconstruction.json'}


def test_pinned_compatible_reference_reconstructs_through_host_and_preserves_original(release):
    f = release
    original = f['checkpoint'].checkpoint_sha256()
    result = ref.reconstruct_reference(f['base'], f['source'], f['authpath'], f['out'], f['receipt'])
    candidate = load_checkpoint(f['out'])
    assert result['status'] == 'REFERENCE_RECONSTRUCTED' and result['candidate_created']
    assert result['candidate_checkpoint_sha256'] == candidate.checkpoint_sha256() != original
    assert load_checkpoint(f['base']).checkpoint_sha256() == original
    assert result['base_checkpoint_sha256_rechecked'] == original and result['source_checkpoint_preserved']
    assert result['host_patch_receipt']['changed_entries'] == 2
    assert result['compatibility']['absolute_update_norm'] == pytest.approx(np.sqrt(2 * .125 ** 2))
    assert candidate.tensor_sha256('untouched.weight') == f['checkpoint'].tensor_sha256('untouched.weight')
    assert result['generation_evaluations'] == 0 and not result['lineage_accepted'] and not result['production_deployment']
    for item in f['manifest']['tensors']:
        assert candidate.tensor_sha256(item['name']) == item['edited_sha256']


def test_source_tensor_mismatch_is_incompatible_and_never_creates_checkpoint(release, monkeypatch):
    f = release
    original_inputs = ref._inputs

    def wrong_manifest(source):
        manifest, arrays, hashes = original_inputs(source)
        manifest['tensors'][0]['base_sha256'] = 'a' * 64
        return manifest, arrays, hashes

    monkeypatch.setattr(ref, '_inputs', wrong_manifest)
    monkeypatch.setattr(ref, 'apply_safetensors_patch', lambda *a: pytest.fail('Mismatch must not reach patcher'))
    inspected = ref.inspect_reference(f['base'], f['source'])
    assert inspected['status'] == 'REFERENCE_INCOMPATIBLE'
    assert 'source tensor' in inspected['reason']
    result = ref.reconstruct_reference(f['base'], f['source'], f['authpath'], f['out'], f['receipt'])
    assert result['status'] == 'REFERENCE_INCOMPATIBLE' and not result['candidate_created']
    assert not f['out'].exists()


def test_pinned_external_artifact_bytes_are_checked_before_model_access(release, monkeypatch):
    f = release
    (f['source'] / ref.EDIT).write_bytes(b'forged external artifact')
    monkeypatch.setattr(ref, 'load_checkpoint', lambda *a: pytest.fail('Unpinned input must refuse first'))
    result = ref.inspect_reference(f['base'], f['source'])
    assert result['status'] == 'REFERENCE_INCOMPATIBLE' and 'source hash mismatch' in result['reason']


@pytest.mark.parametrize('field,value', [('track', 'A'), ('source_commit', 'f' * 40),
    ('model_revision', 'e' * 40), ('base_checkpoint_sha256', 'd' * 64),
    ('manifest_sha256', 'c' * 64), ('edit_sha256', 'b' * 64), ('expected_changed_entries', 3),
    ('lineage_promotion', True), ('lineage_authority', True), ('production_authority', True), ('authority', '')])
def test_host_authorization_cannot_be_borrowed_or_expand_authority(release, field, value):
    f = release
    data = deepcopy(f['authorization']); data[field] = value
    f['authpath'].write_text(json.dumps(data))  # Deliberate forged host document fixture.
    with pytest.raises(EditRejected, match='authorization'):
        ref.reconstruct_reference(f['base'], f['source'], f['authpath'], f['out'], f['receipt'])
    assert not f['out'].exists()


def test_missing_or_broadened_reference_policy_refuses_before_reconstruction(release):
    f = release
    with pytest.raises(EditRejected, match='Missing separate host authorization'):
        ref.reconstruct_reference(f['base'], f['source'], None, f['out'], f['receipt'])
    data = deepcopy(f['authorization']); data['patch_policy']['allow_tensors'].append('unrelated.weight')
    f['authpath'].write_text(json.dumps(data))
    with pytest.raises(EditRejected, match='two-tensor support'):
        ref.reconstruct_reference(f['base'], f['source'], f['authpath'], f['out'], f['receipt'])


@pytest.mark.parametrize('kind', ['duplicate', 'negative', 'too_large', 'float_index', 'float_value'])
def test_sparse_payload_indices_and_native_bf16_values_are_strict(release, kind):
    f = release; arrays = deepcopy(f['arrays'])
    if kind == 'duplicate':
        arrays['t0_idx'] = np.array([0, 0]); arrays['t0_val'] = np.repeat(arrays['t0_val'], 2)
    elif kind == 'negative': arrays['t0_idx'] = np.array([-1])
    elif kind == 'too_large': arrays['t0_idx'] = np.array([4])
    elif kind == 'float_index': arrays['t0_idx'] = arrays['t0_idx'].astype(float)
    elif kind == 'float_value': arrays['t0_val'] = arrays['t0_val'].astype(float)
    with pytest.raises(ValueError, match='sparse payload'):
        ref._payload(f['checkpoint'], f['manifest'], arrays)


def test_external_edited_tensor_hash_is_verified_before_any_export(release):
    f = release; manifest = deepcopy(f['manifest'])
    manifest['tensors'][0]['edited_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='published hash/count'):
        ref._payload(f['checkpoint'], manifest, f['arrays'])


def test_receipt_cannot_modify_base_or_change_candidate_identity(release):
    f = release
    for receipt in (f['base'] / 'forbidden.json', f['out'] / 'forbidden.json'):
        with pytest.raises(EditRejected, match='fresh isolated destination'):
            ref.reconstruct_reference(f['base'], f['source'], f['authpath'], f['out'], receipt)
    assert not f['out'].exists()


def test_existing_candidate_and_write_once_evidence_are_not_overwritten(release):
    f = release
    f['receipt'].write_text('historical evidence')
    with pytest.raises(EditRejected, match='fresh isolated destination'):
        ref.reconstruct_reference(f['base'], f['source'], f['authpath'], f['out'], f['receipt'])
    assert f['receipt'].read_text() == 'historical evidence'


def test_external_reference_never_imports_or_executes_upstream_python(release, monkeypatch):
    f = release
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {'st_numpy', 'apply_edit'}:
            pytest.fail('External Python may not execute')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    assert ref.inspect_reference(f['base'], f['source'])['status'] == 'REFERENCE_COMPATIBLE'
