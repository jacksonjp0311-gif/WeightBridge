"""Conventional supervised FP32 LoRA over an explicitly inspected text backbone.

The base tensors remain immutable. Adapter training and separately authorized
BF16 merging do not inherit the sparse-edit experiment's numerical envelope.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import weakref

import numpy as np
import torch

from ..core import EditRejected, file_hash, json_dump_atomic
from ..inspection.safetensors import load_checkpoint
from ..patches.huggingface import _copy_checkpoint, _from_float64, _to_float64


def _bytes(parameter):
    return parameter.detach().cpu().contiguous().view(torch.uint8).numpy()


def _hash_parameter(parameter):
    return hashlib.sha256(_bytes(parameter)).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class LoRATarget:
    module_name: str
    tensor_name: str
    layer_index: int
    role: str
    shape: tuple[int, int]
    dtype: str
    source_sha256: str

    def __post_init__(self):
        parts = self.module_name.split('.')
        layer_positions = [i for i, part in enumerate(parts) if part == 'layers']
        if (len(layer_positions) != 1 or type(self.layer_index) is not int or self.layer_index < 0
                or self.role not in {'self_attn', 'linear_attn', 'mlp'} or self.dtype != 'BF16'
                or self.tensor_name != self.module_name + '.weight' or len(self.shape) != 2
                or any(type(size) is not int or size <= 0 for size in self.shape)
                or len(self.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_sha256)
                or set(parts) & {'vision', 'visual', 'lm_head', 'embed_tokens', 'embeddings'}):
            raise ValueError('LoRA target requires an explicit untied text-layer attention/MLP weight')
        position = layer_positions[0]
        if (position + 2 >= len(parts) or parts[position + 1] != str(self.layer_index)
                or parts[position + 2] != self.role):
            raise ValueError('LoRA target layer and role do not match its native module path')
        object.__setattr__(self, 'shape', tuple(self.shape))


def discover_lora_targets(adapter, runner, *, expected_layers=24):
    """Resolve only untied dense attention/MLP linears in supported text layers."""
    adapter._validate_runner(runner)
    if runner.model.training or type(expected_layers) is not int or not 1 <= expected_layers <= 128:
        raise ValueError("LoRA discovery requires an evaluated model and explicit layer count")
    decoder = adapter._decoder(runner.model)
    layers = tuple(decoder.layers)
    if len(layers) != expected_layers:
        raise ValueError("Text decoder layer count differs from the authorized architecture")
    names = {id(module): name for name, module in runner.model.named_modules()}
    pointers = Counter(parameter.data_ptr() for _, parameter in runner.model.named_parameters(remove_duplicate=False))
    checkpoint, targets, rejected = adapter.checkpoint, [], []
    initial = checkpoint.checkpoint_sha256()
    for index, layer in enumerate(layers):
        selected = []
        for local_name, module in layer.named_modules():
            if type(module) is not torch.nn.Linear:
                continue
            role = local_name.partition('.')[0]
            name = names[id(module)]
            tensor_name = name + '.weight'
            if role not in {'self_attn', 'linear_attn', 'mlp'}:
                rejected.append({'module': name, 'reason': 'outside_explicit_attention_and_mlp_roles'})
                continue
            ref = checkpoint.tensors.get(tensor_name)
            if (ref is None or ref.dtype != 'BF16' or tuple(module.weight.shape) != ref.shape
                    or module.weight.dtype != torch.bfloat16 or module.weight.device.type == 'meta'
                    or pointers[module.weight.data_ptr()] != 1):
                raise ValueError(f"Eligible LoRA module is unsupported, tied or not native BF16: {name}")
            digest = _hash_parameter(module.weight)
            if digest != checkpoint.tensor_sha256(tensor_name):
                raise ValueError(f"Live LoRA base tensor differs from immutable storage: {tensor_name}")
            selected.append(LoRATarget(name, tensor_name, index, role, ref.shape, ref.dtype, digest))
        if not selected or not any(t.role == 'mlp' for t in selected) or not any(t.role != 'mlp' for t in selected):
            raise ValueError(f"Text layer {index} lacks the declared dense attention and MLP pathway")
        targets.extend(selected)
    targets = tuple(sorted(targets, key=lambda target: target.module_name))
    if len({target.tensor_name for target in targets}) != len(targets):
        raise ValueError("LoRA inventory contains repeated or aliased target tensors")
    if checkpoint.checkpoint_sha256() != initial:
        raise ValueError("Source checkpoint changed during LoRA discovery")
    inventory = {'schema': 'weightbridge.supervised_lora_inventory.v1', 'checkpoint_sha256': initial,
        'expected_layers': expected_layers, 'module_count': len(targets),
        'roles': dict(sorted(Counter(t.role for t in targets).items())),
        'target_parameters': sum(math.prod(t.shape) for t in targets),
        'target_tensor_bytes': sum(math.prod(t.shape) * 2 for t in targets),
        'modules': [asdict(target) for target in targets], 'rejected': rejected,
        'scope': 'Explicit dense text-layer attention and MLP linears only; no vision, embeddings, output head, normalization or tied modules.'}
    return targets, inventory


class TorchLoRA(torch.nn.Module):
    """Two genuinely trainable FP32 factors; randomized A, zero B at attachment."""

    def __init__(self, in_features, out_features, *, rank=8, alpha=16., generator=None, device='cpu'):
        super().__init__()
        self.rank, self.alpha = rank, float(alpha)
        self.scale = self.alpha / rank
        a = torch.empty((rank, in_features), dtype=torch.float32)
        a.uniform_(-1 / math.sqrt(in_features), 1 / math.sqrt(in_features), generator=generator)
        self.A = torch.nn.Parameter(a.to(device))
        self.B = torch.nn.Parameter(torch.zeros((out_features, rank), dtype=torch.float32, device=device))

    def forward(self, inputs):
        return torch.nn.functional.linear(torch.nn.functional.linear(inputs.float(), self.A), self.B) * self.scale


class SupervisedLoRA:
    """Observable hook adapters with independent parameters and exact base guards."""

    def __init__(self, model, targets, *, checkpoint_sha256, rank=8, alpha=16., dropout=0., seed=0):
        if (type(rank) is not int or not 1 <= rank <= 8 or not math.isfinite(alpha) or not 0 < alpha <= 32
                or dropout != 0. or type(seed) is not int or not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
            raise ValueError("LoRA requires bounded rank, finite scale, zero dropout and a pinned source SHA-256")
        self.targets = tuple(targets)
        if (not self.targets or len(self.targets) > 1024
                or len({t.module_name for t in self.targets}) != len(self.targets)
                or len({t.tensor_name for t in self.targets}) != len(self.targets)):
            raise ValueError("LoRA requires a distinct explicit module inventory")
        self._model = weakref.ref(model)
        self.checkpoint_sha256, self.rank, self.alpha, self.dropout, self.seed = checkpoint_sha256, rank, float(alpha), 0., seed
        self._base = tuple((name, parameter) for name, parameter in model.named_parameters())
        self._versions = {name: parameter._version for name, parameter in self._base}
        self._flags = {name: parameter.requires_grad for name, parameter in self._base}
        self._base_hashes = {name: _hash_parameter(parameter) for name, parameter in self._base}
        self._handles, self.enabled = [], True
        self.adapters = torch.nn.ModuleDict()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        pointers = Counter(p.data_ptr() for _, p in model.named_parameters(remove_duplicate=False))
        for index, target in enumerate(self.targets):
            module = model.get_submodule(target.module_name)
            if (type(module) is not torch.nn.Linear or target.tensor_name != target.module_name + '.weight'
                    or tuple(module.weight.shape) != tuple(target.shape) or target.dtype != 'BF16'
                    or module.weight.dtype != torch.bfloat16 or pointers[module.weight.data_ptr()] != 1
                    or _hash_parameter(module.weight) != target.source_sha256):
                raise ValueError("LoRA target does not match the inspected native base tensor")
            self.adapters[f'm{index:04d}'] = TorchLoRA(module.in_features, module.out_features, rank=rank,
                alpha=alpha, generator=generator, device=module.weight.device)

    @property
    def attached(self):
        return bool(self._handles)

    def _model_or_fail(self):
        model = self._model()
        if model is None:
            raise ValueError("LoRA source model no longer exists")
        return model

    def parameters(self):
        return self.adapters.parameters()

    def attach(self):
        if self.attached:
            raise ValueError("LoRA adapter is already attached")
        self.assert_base_unchanged(hash_check=True)
        model = self._model_or_fail()
        for _, parameter in self._base:
            parameter.requires_grad_(False)
        try:
            for target, adapter in zip(self.targets, self.adapters.values(), strict=True):
                def additive(module, args, output, *, lora=adapter):
                    return output + lora(args[0]).to(output.dtype) if self.enabled else output
                self._handles.append(model.get_submodule(target.module_name).register_forward_hook(additive))
        except BaseException:
            self.detach()
            raise
        return self

    def assert_base_unchanged(self, *, hash_check=False):
        model = self._model_or_fail()
        observed = tuple(model.named_parameters())
        if len(observed) != len(self._base) or any(name != old_name or value is not old
                for (name, value), (old_name, old) in zip(observed, self._base, strict=True)):
            raise RuntimeError("LoRA base parameter identity changed")
        for name, parameter in self._base:
            if parameter._version != self._versions[name] or (self.attached and parameter.requires_grad):
                raise RuntimeError(f"LoRA base parameter version/freeze changed: {name}")
            if hash_check and _hash_parameter(parameter) != self._base_hashes[name]:
                raise RuntimeError(f"LoRA base parameter bytes changed: {name}")
        return True

    @contextmanager
    def disabled(self):
        previous, self.enabled = self.enabled, False
        try:
            yield self
        finally:
            self.enabled = previous

    def detach(self):
        try:
            self.assert_base_unchanged(hash_check=True)
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            for name, parameter in self._base:
                parameter.requires_grad_(self._flags[name])

    def configuration(self):
        return {'schema': 'weightbridge.supervised_lora.v1', 'checkpoint_sha256': self.checkpoint_sha256,
            'rank': self.rank, 'alpha': self.alpha, 'dropout': self.dropout, 'seed': self.seed,
            'factor_dtype': 'F32', 'base_dtype': 'BF16', 'modules': [asdict(target) for target in self.targets]}

    def factor_sha256(self):
        digest = hashlib.sha256()
        for target, adapter in zip(self.targets, self.adapters.values(), strict=True):
            for name in ('A', 'B'):
                value = getattr(adapter, name)
                digest.update(_canonical([target.module_name, name, list(value.shape), str(value.dtype)]).encode())
                digest.update(_bytes(value))
        return digest.hexdigest()

    def save(self, path, *, metadata=None):
        from safetensors.torch import save
        path = Path(path)
        if path.exists() or path.is_symlink():
            raise FileExistsError("Adapter artifacts are write-once")
        self.assert_base_unchanged(hash_check=True)
        factors_sha = self.factor_sha256()
        tensors = {}
        for target, adapter in zip(self.targets, self.adapters.values(), strict=True):
            for factor in ('A', 'B'):
                value = getattr(adapter, factor).detach().cpu().contiguous()
                if value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("Adapter contains nonfinite or non-FP32 factors")
                tensors[f'{target.module_name}.lora_{factor}'] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = save(tensors, metadata={'configuration': _canonical(self.configuration()),
            'host_metadata': _canonical(metadata or {})})
        with path.open('xb') as output:
            output.write(payload)
        if self.factor_sha256() != factors_sha:
            raise RuntimeError('LoRA adapter factors changed while serializing')
        return {'schema': 'weightbridge.supervised_lora_artifact.v1', 'sha256': file_hash(path),
            'configuration': self.configuration(), 'host_metadata': metadata or {},
            'factor_sha256': factors_sha, 'trainable_parameters': sum(p.numel() for p in self.parameters()), 'base_unchanged': True}

    @classmethod
    def load(cls, model, targets, path, *, checkpoint_sha256, expected_metadata=None, expected_sha256=None):
        from safetensors import safe_open
        from safetensors.torch import load_file
        path = Path(path)
        if expected_sha256 is not None and file_hash(path) != expected_sha256:
            raise ValueError('Adapter artifact hash differs from the independently frozen receipt')
        targets = tuple(targets)
        maximum = 4 * 8 * sum(sum(target.shape) for target in targets) + 2 * 1024 * 1024
        if path.stat().st_size > maximum:
            raise ValueError('Adapter safetensors exceeds the inspected factor memory bound')
        with safe_open(str(path), framework='pt', device='cpu') as archive:
            metadata = archive.metadata()
        if set(metadata or {}) != {'configuration', 'host_metadata'}:
            raise ValueError("Adapter safetensors metadata is missing or unrecognized")
        configuration, host = json.loads(metadata['configuration']), json.loads(metadata['host_metadata'])
        if expected_metadata is not None and host != expected_metadata:
            raise ValueError("Adapter host metadata differs from the authorized training identity")
        session = cls(model, targets, checkpoint_sha256=checkpoint_sha256,
            **{key: configuration[key] for key in ('rank', 'alpha', 'dropout', 'seed')})
        if _canonical(configuration) != _canonical(session.configuration()):
            raise ValueError("Adapter configuration does not match the inspected source architecture")
        values = load_file(str(path), device='cpu')
        expected = {f'{target.module_name}.lora_{factor}' for target in session.targets for factor in ('A', 'B')}
        if set(values) != expected:
            raise ValueError("Adapter tensor allowlist differs from its inspected modules")
        with torch.no_grad():
            for target, adapter in zip(session.targets, session.adapters.values(), strict=True):
                for factor in ('A', 'B'):
                    value, destination = values[f'{target.module_name}.lora_{factor}'], getattr(adapter, factor)
                    if value.dtype != torch.float32 or value.shape != destination.shape or not torch.isfinite(value).all():
                        raise ValueError("Adapter factor dtype, shape or finite values differ")
                    destination.copy_(value.to(destination.device))
        return session

    def effective_statistics(self, *, chunk_rows=128):
        """Measure proposed and actual BF16 changes without installing them."""
        self.assert_base_unchanged()
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= 1024:
            raise ValueError("Effective-update measurement requires bounded row chunks")
        rows, proposed_square, stored_square, parent_square, residual_square, changed = [], 0., 0., 0., 0., 0
        model = self._model_or_fail()
        for target, adapter in zip(self.targets, self.adapters.values(), strict=True):
            weight = model.get_submodule(target.module_name).weight
            a, b = (p.detach().cpu().double().numpy() for p in (adapter.A, adapter.B))
            square, actual_square, source_square, rounding_square, support = 0., 0., 0., 0., 0
            digest = hashlib.sha256()
            for first in range(0, target.shape[0], chunk_rows):
                original = weight[first:first + chunk_rows].detach().cpu().float().double().numpy()
                delta = (b[first:first + chunk_rows] @ a) * adapter.scale
                merged = original + delta
                merged[delta == 0] = original[delta == 0]
                payload = _from_float64(merged, 'BF16')
                rounded = _to_float64(payload, 'BF16').reshape(original.shape)
                difference = rounded - original
                square += float(np.sum(delta * delta))
                actual_square += float(np.sum(difference * difference))
                source_square += float(np.sum(original * original))
                rounding_square += float(np.sum((difference - delta) ** 2))
                support += int(np.count_nonzero(np.frombuffer(payload, dtype='<u2') !=
                    np.frombuffer(_from_float64(original, 'BF16'), dtype='<u2')))
                digest.update(payload)
            proposed_square += square; stored_square += actual_square; changed += support
            parent_square += source_square; residual_square += rounding_square
            rows.append({'tensor': target.tensor_name, 'proposed_update_norm': math.sqrt(square),
                'stored_update_norm': math.sqrt(actual_square), 'changed_entries': support,
                'parent_selected_weight_norm': math.sqrt(source_square), 'rounding_residual_norm': math.sqrt(rounding_square),
                'relative_stored_update_norm': math.sqrt(actual_square / source_square) if source_square else None,
                'candidate_tensor_sha256': digest.hexdigest()})
        return {'schema': 'weightbridge.supervised_lora_update.v1', 'proposed_update_norm': math.sqrt(proposed_square),
            'stored_update_norm': math.sqrt(stored_square), 'changed_entries': changed, 'modules': rows,
            'parent_selected_weight_norm': math.sqrt(parent_square), 'rounding_residual_norm': math.sqrt(residual_square),
            'relative_stored_update_norm': math.sqrt(stored_square / parent_square) if parent_square else None,
            'relative_norm_scope': 'Full adapted attention/MLP weight matrices only; excludes head, embeddings, vision and other frozen parameters',
            'trainable_parameters': sum(p.numel() for p in self.parameters()),
            'merge_math': 'float64_factor_product_then_round_once_into_BF16',
            'scope': 'Conventional adapter baseline; no inherited sparse-edit 0.025 cap. Native merged weights may differ from unmerged adapter execution.'}

    def merge_candidate(self, base, out, authorization, receipt):
        """Serialize only authorized dense LoRA merges into a separate checkpoint."""
        base, out, authorization, receipt = map(Path, (base, out, authorization, receipt))
        if (out.exists() or out.is_symlink() or receipt.exists() or out.resolve().is_relative_to(base.resolve())
                or receipt.resolve().is_relative_to(base.resolve()) or receipt.resolve().is_relative_to(out.resolve())):
            raise EditRejected("LoRA merge requires new candidate/evidence paths outside the immutable parent")
        policy = json.loads(authorization.read_bytes())
        names = {target.tensor_name for target in self.targets}
        if (policy.get('schema') != 'weightbridge.supervised_lora_merge.v1'
                or policy.get('base_checkpoint_sha256') != self.checkpoint_sha256
                or set(policy.get('allow_tensors', ())) != names or type(policy.get('max_rank')) is not int
                or not self.rank <= policy['max_rank'] <= 8
                or policy.get('alpha') != self.alpha or policy.get('stored_dtype') != 'BF16'
                or policy.get('merge_math') != 'float64_factor_product_then_round_once_into_BF16'
                or not policy.get('authority') or policy.get('production_authority', False) is not False
                or policy.get('lineage_authority', False) is not False):
            raise EditRejected("Separate conventional LoRA merge authorization differs from the inspected adapter")
        total = sum(math.prod(t.shape) for t in self.targets)
        if (total > policy.get('max_updated_entries', 0) or 2 * total > policy.get('max_total_tensor_bytes', 0)
                or any(2 * math.prod(t.shape) > policy.get('max_tensor_bytes', 0) for t in self.targets)):
            raise EditRejected("LoRA merge exceeds the conventional baseline tensor budget")
        self.assert_base_unchanged(hash_check=True)
        factors_sha = self.factor_sha256()
        checkpoint = load_checkpoint(base)
        if checkpoint.checkpoint_sha256() != self.checkpoint_sha256:
            raise EditRejected("LoRA source checkpoint identity differs")
        policy_sha = file_hash(authorization)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows, total_square, proposed_square, parent_square, residual_square, changed = [], 0., 0., 0., 0., 0
        with tempfile.TemporaryDirectory(prefix='.weightbridge-lora-', dir=out.parent) as directory:
            candidate_path = _copy_checkpoint(checkpoint, Path(directory))
            candidate = load_checkpoint(candidate_path)
            for target, adapter in zip(self.targets, self.adapters.values(), strict=True):
                original, destination = checkpoint.tensors[target.tensor_name], candidate.tensors[target.tensor_name]
                if (original.dtype != 'BF16' or tuple(original.shape) != tuple(target.shape)
                        or checkpoint.tensor_sha256(target.tensor_name) != target.source_sha256):
                    raise EditRejected("LoRA merge source tensor differs from its inspected version")
                a, b = (p.detach().cpu().double().numpy() for p in (adapter.A, adapter.B))
                square, proposed, source_square, rounding_square, count, digest = 0., 0., 0., 0., 0, hashlib.sha256()
                with original.path.open('rb') as source, destination.path.open('r+b') as output:
                    source.seek(original.data_start + original.start)
                    output.seek(destination.data_start + destination.start)
                    for first in range(0, target.shape[0], 128):
                        height = min(128, target.shape[0] - first)
                        raw = source.read(height * target.shape[1] * 2)
                        if len(raw) != height * target.shape[1] * 2:
                            raise EditRejected("Truncated immutable source during LoRA merge")
                        values = _to_float64(raw, 'BF16').reshape(height, target.shape[1])
                        delta = (b[first:first + height] @ a) * adapter.scale
                        merged = values + delta
                        merged[delta == 0] = values[delta == 0]
                        payload = _from_float64(merged, 'BF16')
                        actual = _to_float64(payload, 'BF16').reshape(values.shape) - values
                        output.write(payload); digest.update(payload)
                        square += float(np.sum(actual * actual)); proposed += float(np.sum(delta * delta))
                        source_square += float(np.sum(values * values))
                        rounding_square += float(np.sum((actual - delta) ** 2))
                        count += int(np.count_nonzero(np.frombuffer(payload, dtype='<u2') != np.frombuffer(raw, dtype='<u2')))
                rows.append({'tensor': target.tensor_name, 'source_sha256': target.source_sha256,
                    'candidate_sha256': digest.hexdigest(), 'stored_update_norm': math.sqrt(square),
                    'proposed_update_norm': math.sqrt(proposed), 'changed_entries': count,
                    'parent_selected_weight_norm': math.sqrt(source_square), 'rounding_residual_norm': math.sqrt(rounding_square),
                    'relative_stored_update_norm': math.sqrt(square / source_square) if source_square else None})
                total_square += square; proposed_square += proposed; changed += count
                parent_square += source_square; residual_square += rounding_square
            candidate = load_checkpoint(candidate_path)
            if any(candidate.tensor_sha256(row['tensor']) != row['candidate_sha256'] for row in rows):
                raise EditRejected("Merged LoRA candidate tensor identity differs")
            if (file_hash(authorization) != policy_sha or load_checkpoint(base).checkpoint_sha256() != self.checkpoint_sha256
                    or self.factor_sha256() != factors_sha):
                raise EditRejected("LoRA merge source or authorization changed")
            self.assert_base_unchanged(hash_check=True)
            bound = policy.get('max_update_norm')
            if bound is not None and (not isinstance(bound, (float, int)) or math.sqrt(total_square) > bound):
                raise EditRejected("LoRA merged update exceeds the separately authorized norm envelope")
            candidate_sha = candidate.checkpoint_sha256()
            os.replace(candidate_path, out)
        result = {'schema': 'weightbridge.supervised_lora_merge_receipt.v1', 'status': 'ISOLATED_CANDIDATE',
            'base_sha256': self.checkpoint_sha256, 'candidate_sha256': candidate_sha,
            'authorization_sha256': policy_sha, 'rank': self.rank, 'alpha': self.alpha,
            'factor_sha256': factors_sha,
            'stored_update_norm': math.sqrt(total_square), 'proposed_update_norm': math.sqrt(proposed_square),
            'parent_selected_weight_norm': math.sqrt(parent_square), 'rounding_residual_norm': math.sqrt(residual_square),
            'relative_stored_update_norm': math.sqrt(total_square / parent_square) if parent_square else None,
            'relative_norm_scope': 'Full adapted attention/MLP weight matrices only; not the total model parameter count',
            'changed_entries': changed, 'tensor_receipts': rows, 'base_unchanged': True,
            'lineage_accepted': False, 'production_deployment': False,
            'limitations': ['Merged BF16 execution may differ from additive FP32 adapter execution; independent behavioral verification remains required.']}
        json_dump_atomic(receipt, result)
        return result
