"""Conventional adapter mechanics on tiny fixtures, not research efficacy."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
from weightbridge.core import EditRejected, json_dump_atomic
from weightbridge.inspection.safetensors import load_checkpoint
from weightbridge.patches.huggingface import _from_float64


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.mlp.down_proj(torch.tanh(self.self_attn.q_proj(x)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.visual = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.fixture
def base(tmp_path):
    import struct
    with torch.random.fork_rng():
        torch.manual_seed(781)
        model = Model().eval()
    folder = tmp_path / 'base'; folder.mkdir()
    header, blobs, offset = {}, [], 0
    for name, parameter in model.named_parameters():
        raw = parameter.detach().contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {'dtype': 'BF16', 'shape': list(parameter.shape), 'data_offsets': [offset, offset + len(raw)]}
        blobs.append(raw); offset += len(raw)
    encoded = json.dumps(header, separators=(',', ':')).encode(); encoded += b' ' * (-len(encoded) % 8)
    (folder / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + b''.join(blobs))
    (folder / 'config.json').write_text('{"model_type":"fixture"}')
    checkpoint = load_checkpoint(folder)
    adapter = SimpleNamespace(checkpoint=checkpoint, _validate_runner=lambda runner: None,
        _decoder=lambda runtime: runtime.model)
    runner = SimpleNamespace(model=model)
    targets, inventory = discover_lora_targets(adapter, runner, expected_layers=2)
    return SimpleNamespace(model=model, folder=folder, checkpoint=checkpoint, adapter=adapter,
        runner=runner, targets=targets, inventory=inventory, sha=checkpoint.checkpoint_sha256())


def session(base, **kwargs):
    return SupervisedLoRA(base.model, base.targets, checkpoint_sha256=base.sha, rank=2, alpha=4., **kwargs)


def test_discovery_is_explicit_text_only_bf16_and_excludes_head_and_vision(base):
    assert len(base.targets) == 4 and base.inventory['roles'] == {'self_attn': 2, 'mlp': 2}
    assert all('visual' not in t.module_name and 'lm_head' not in t.module_name for t in base.targets)
    assert {t.layer_index for t in base.targets} == {0, 1}
    with pytest.raises(ValueError, match='layer count'):
        discover_lora_targets(base.adapter, base.runner, expected_layers=24)


def test_tied_or_changed_source_module_cannot_enter_inventory(base):
    base.model.lm_head.weight = base.model.model.layers[0].self_attn.q_proj.weight
    with pytest.raises(ValueError, match='tied'):
        discover_lora_targets(base.adapter, base.runner, expected_layers=2)


def test_zero_init_exactly_parent_and_disable_detach_restore_flags(base):
    x = torch.tensor([[1., 2., -1., .5]], dtype=torch.bfloat16)
    before = base.model(x).detach().clone()
    original_flags = {name: parameter.requires_grad for name, parameter in base.model.named_parameters()}
    adapter = session(base).attach()
    assert torch.equal(base.model(x), before)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in adapter.parameters())
    assert not any(p.requires_grad for p in base.model.parameters())
    with torch.no_grad():
        for layer in adapter.adapters.values(): layer.B.fill_(.1)
    edited = base.model(x).detach()
    assert not torch.equal(edited, before)
    with adapter.disabled():
        assert torch.equal(base.model(x), before)
    assert torch.equal(base.model(x), edited)
    adapter.detach()
    assert torch.equal(base.model(x), before)
    assert original_flags == {name: p.requires_grad for name, p in base.model.named_parameters()}


def test_both_fp32_factors_receive_actual_training_gradients_after_two_steps(base):
    x = torch.tensor([[1., .5, -2., 1.], [-1., 2., 1., .5]], dtype=torch.bfloat16)
    adapter = session(base).attach()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.05, weight_decay=0.)
    first_a = [layer.A.detach().clone() for layer in adapter.adapters.values()]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (base.model(x).float() - .5).square().mean()
        loss.backward()
        assert any(float(layer.B.grad.norm()) > 0 for layer in adapter.adapters.values())
        if step == 0:
            assert all(float(layer.A.grad.norm()) == 0 for layer in adapter.adapters.values())
        else:
            assert any(float(layer.A.grad.norm()) > 0 for layer in adapter.adapters.values())
        optimizer.step()
        assert adapter.assert_base_unchanged()
        assert all(p.grad is None for p in base.model.parameters())
    assert any(not torch.equal(layer.A, before) for layer, before in zip(adapter.adapters.values(), first_a))
    adapter.detach()


def test_save_reload_has_exact_fp32_factors_predictions_and_metadata(base, tmp_path):
    x = torch.tensor([[1., .5, -2., 1.]], dtype=torch.bfloat16)
    adapter = session(base).attach()
    with torch.no_grad():
        for layer in adapter.adapters.values(): layer.B.fill_(.03125)
    expected = base.model(x).detach().clone()
    path = tmp_path / 'adapter.safetensors'
    receipt = adapter.save(path, metadata={'training_id': 'synthetic-test'})
    adapter.detach()
    loaded = SupervisedLoRA.load(base.model, base.targets, path, checkpoint_sha256=base.sha,
        expected_metadata={'training_id': 'synthetic-test'}).attach()
    assert torch.equal(base.model(x), expected)
    assert receipt['configuration']['rank'] == 2 and receipt['trainable_parameters'] == 64
    with pytest.raises(FileExistsError): loaded.save(path)
    loaded.detach()
    with pytest.raises(ValueError, match='host metadata'):
        SupervisedLoRA.load(base.model, base.targets, path, checkpoint_sha256=base.sha,
            expected_metadata={'training_id': 'wrong'})


def test_base_version_and_byte_guards_catch_bypassed_parameter_mutation(base):
    adapter = session(base).attach()
    parameter = next(base.model.parameters())
    with torch.no_grad(): parameter.add_(1)
    with pytest.raises(RuntimeError, match='version'):
        adapter.assert_base_unchanged()
    with pytest.raises(RuntimeError): adapter.detach()
    assert not adapter.attached
    assert load_checkpoint(base.folder).checkpoint_sha256() == base.sha


def authorization(base):
    return {'schema': 'weightbridge.supervised_lora_merge.v1', 'base_checkpoint_sha256': base.sha,
        'allow_tensors': [t.tensor_name for t in base.targets], 'max_rank': 2, 'alpha': 4.,
        'max_updated_entries': 64, 'max_tensor_bytes': 32, 'max_total_tensor_bytes': 128,
        'stored_dtype': 'BF16', 'merge_math': 'float64_factor_product_then_round_once_into_BF16',
        'authority': 'Synthetic merge fixture only', 'production_authority': False, 'lineage_authority': False,
        'max_update_norm': None}


def test_authorized_native_merge_exceeds_sparse_cap_without_changing_parent_or_other_tensors(base, tmp_path):
    adapter = session(base).attach()
    with torch.no_grad():
        for layer in adapter.adapters.values(): layer.A.fill_(.25); layer.B.fill_(.25)
    statistics = adapter.effective_statistics(chunk_rows=1)
    assert statistics['stored_update_norm'] > .025
    policy = tmp_path / 'authorization.json'; json_dump_atomic(policy, authorization(base))
    out, receipt = tmp_path / 'candidate', tmp_path / 'merge.json'
    merged = adapter.merge_candidate(base.folder, out, policy, receipt)
    checkpoint = load_checkpoint(out)
    assert merged['stored_update_norm'] == pytest.approx(statistics['stored_update_norm'])
    assert merged['parent_selected_weight_norm'] == pytest.approx(statistics['parent_selected_weight_norm'])
    assert merged['relative_stored_update_norm'] == pytest.approx(statistics['relative_stored_update_norm'])
    assert merged['rounding_residual_norm'] == pytest.approx(statistics['rounding_residual_norm'])
    assert merged['changed_entries'] == 64 and merged['base_unchanged']
    assert checkpoint.checkpoint_sha256() == merged['candidate_sha256'] != base.sha
    assert checkpoint.tensor_sha256('visual.weight') == base.checkpoint.tensor_sha256('visual.weight')
    assert checkpoint.tensor_sha256('lm_head.weight') == base.checkpoint.tensor_sha256('lm_head.weight')
    assert load_checkpoint(base.folder).checkpoint_sha256() == base.sha
    for row in merged['tensor_receipts']:
        assert checkpoint.tensor_sha256(row['tensor']) == row['candidate_sha256']
    adapter.detach()


def test_merge_requires_separate_allowlist_and_refuses_receipt_inside_parent(base, tmp_path):
    adapter = session(base).attach()
    policy = tmp_path / 'authorization.json'
    bad = authorization(base); bad['allow_tensors'].append('visual.weight')
    json_dump_atomic(policy, bad)
    with pytest.raises(EditRejected, match='authorization'):
        adapter.merge_candidate(base.folder, tmp_path / 'candidate', policy, tmp_path / 'receipt.json')
    with pytest.raises(EditRejected, match='immutable parent'):
        adapter.merge_candidate(base.folder, tmp_path / 'candidate', policy, base.folder / 'receipt.json')
    adapter.detach()


def test_constructor_is_seeded_without_mutating_global_rng_or_base(base):
    before = torch.get_rng_state().clone()
    first = session(base, seed=78)
    second = session(base, seed=78)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert first.assert_base_unchanged(hash_check=True)
    with pytest.raises(ValueError, match='zero dropout'):
        session(base, dropout=.1)


def test_constructor_and_typed_inventory_cannot_redirect_into_head_or_tied_weights(base):
    with pytest.raises(ValueError, match='text-layer'):
        replace(base.targets[0], module_name='lm_head', tensor_name='lm_head.weight')
    base.model.lm_head.weight = base.model.get_submodule(base.targets[0].module_name).weight
    with pytest.raises(ValueError, match='inspected native base'):
        session(base)


def test_final_hash_guard_detects_data_mutation_even_when_version_counter_does_not(base):
    adapter = session(base).attach()
    parameter = next(base.model.parameters()); version = parameter._version
    parameter.data.add_(1)
    assert parameter._version == version
    with pytest.raises(RuntimeError, match='bytes changed'):
        adapter.assert_base_unchanged(hash_check=True)
    with pytest.raises(RuntimeError): adapter.detach()
    assert not adapter.attached and load_checkpoint(base.folder).checkpoint_sha256() == base.sha


def test_adapter_load_requires_expected_hash_when_supplied(base, tmp_path):
    adapter = session(base)
    path = tmp_path / 'adapter.safetensors'
    saved = adapter.save(path)
    assert SupervisedLoRA.load(base.model, base.targets, path, checkpoint_sha256=base.sha,
        expected_sha256=saved['sha256']).factor_sha256() == adapter.factor_sha256()
    with pytest.raises(ValueError, match='artifact hash'):
        SupervisedLoRA.load(base.model, base.targets, path, checkpoint_sha256=base.sha,
            expected_sha256='0' * 64)


def test_failed_merge_leaves_no_candidate_and_preserves_original(base, tmp_path, monkeypatch):
    from weightbridge.adaptation import supervised_lora
    adapter = session(base).attach()
    policy = tmp_path / 'authorization.json'; json_dump_atomic(policy, authorization(base))
    def fail(*args): raise RuntimeError('fixture interrupted native merge')
    monkeypatch.setattr(supervised_lora, '_from_float64', fail)
    out = tmp_path / 'candidate'
    with pytest.raises(RuntimeError, match='interrupted native merge'):
        adapter.merge_candidate(base.folder, out, policy, tmp_path / 'receipt.json')
    assert not out.exists() and not list(tmp_path.glob('.weightbridge-lora-*'))
    assert load_checkpoint(base.folder).checkpoint_sha256() == base.sha
    adapter.detach()
