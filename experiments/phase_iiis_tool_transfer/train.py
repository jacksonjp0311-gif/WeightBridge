"""Training-only mixed-task SFT with explicit immutable parent identity.

Reuses the existing LoRA implementation, authorization and causal loss. No
development/selection/hidden data, execution of predicted tools or promotion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

from weightbridge.adaptation.training_policy import TrainingAuthorization
from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from experiments.phase_iiiq_baseline.integrity import read, verify_sources
from experiments.phase_iiiq_baseline.train import data_order, learning_rate_factor


def authorization(protocol, inventory, bundle_path, *, mode='main', smoke_gate=None):
    """Build a narrow per-parent policy without changing historical patch caps."""
    if mode not in {'main', 'smoke', 'timing'}:
        raise ValueError('Unknown S authorization mode')
    rows = tuple(inventory)
    names = tuple(row.get('module_name', row.get('name')) for row in rows)
    shapes = tuple(row.get('shape', row.get('weight_shape')) for row in rows)
    if (len(rows) != 186 or len(set(names)) != 186 or names != tuple(sorted(names))
            or any(shape is None or len(shape) != 2 for shape in shapes)
            or sum(sum(shape) for shape in shapes)*8 != 5_411_328):
        raise ValueError('S authorization requires the exact sorted 186-module rank-eight geometry')
    settings = protocol['smoke_training' if mode == 'smoke' else 'training']
    limits = protocol['resources']['mode_limits'][mode]
    return TrainingAuthorization(schema='weightbridge.supervised_training_authorization.v1',
        parent_sha256=protocol['checkpoint_sha256'], allowed_modules=names, max_trainable_parameters=5_411_328,
        rank=8, alpha=16., dropout=0., learning_rates=(.0003,), optimizer='AdamW', max_runs=1,
        max_steps_per_run=1 if mode == 'timing' else settings['optimizer_steps'],
        max_answer_tokens_per_run=limits['max_answer_tokens'], max_input_tokens_per_run=limits['max_input_tokens'],
        max_memory_bytes=protocol['resources']['max_memory_bytes'], max_wall_seconds_per_run=limits['max_wall_seconds'],
        max_cpu_seconds_per_run=limits['max_cpu_seconds'], max_development_checkpoints_per_run=2 if mode == 'main' else 1,
        allow_merge=mode == 'main', training_bundle_sha256=file_hash(bundle_path),
        smoke_gate_sha256=file_hash(smoke_gate) if smoke_gate is not None else None)


@contextmanager
def runtime(base, protocol, *, expected_parent):
    import torch
    from weightbridge.adaptation.model_adapter import select_hf_adapter
    if expected_parent != protocol['checkpoint_sha256']:
        raise ValueError('Explicit parent differs from the frozen protocol')
    adapter = select_hf_adapter(base)
    if adapter.checkpoint.checkpoint_sha256() != expected_parent:
        raise ValueError('Training source is not the explicitly authorized immutable parent')
    previous = torch.get_num_threads()
    torch.set_num_threads(protocol['endpoints']['tool']['cpu_threads'])
    try:
        runner = adapter.load_runner(device='cpu', dtype='auto', max_rss_bytes=protocol['resources']['max_memory_bytes'])
        runner.model.eval()
        yield adapter, runner
    finally:
        torch.set_num_threads(previous)
        if adapter.checkpoint.checkpoint_sha256() != expected_parent:
            raise RuntimeError('Immutable parent checkpoint bytes changed')


def checked_inputs(protocol_path, authorization_path, bundle_path, *, mode):
    if mode not in {'main', 'smoke', 'timing'}:
        raise ValueError('Unknown S training stage')
    protocol, bundle = read(protocol_path), read(bundle_path)
    auth = TrainingAuthorization.from_dict(read(authorization_path))
    if (not protocol.get('source_sha256') or protocol.get('schema') != 'weightbridge.phase_iiis.protocol.v1'
            or auth.parent_sha256 != protocol['checkpoint_sha256']):
        raise ValueError('Training requires prospective S sources and an explicit parent')
    verify_sources(protocol['source_sha256'])
    splits = {'train', 'replay'} if mode == 'main' else {mode}
    if (set(bundle) != {'schema', 'protocol_sha256', 'rows', 'manifest_sha256', 'split_ids'}
            or set(bundle['split_ids']) != splits or file_hash(bundle_path) != auth.training_bundle_sha256
            or bundle['protocol_sha256'] != file_hash(protocol_path)
            or not bundle['rows'] or len({row['id'] for row in bundle['rows']}) != len(bundle['rows'])
            or {row.get('task') for row in bundle['rows']} != {'tool', 'decimal'}
            or any(row.get('training_weight', 1.) != 1. for row in bundle['rows'])):
        raise ValueError('Worker may receive only the exact authorized mixed fitting bundle')
    settings = protocol['smoke_training' if mode == 'smoke' else 'training']
    if (settings['seed'] != 606118 or settings['effective_batch_size'] != 16
            or settings['micro_batch_size'] != 4 or settings['learning_rates'] != [.0003]
            or settings['optimizer'] != 'AdamW' or auth.rank != 8 or auth.alpha != 16
            or auth.dropout != 0 or auth.max_runs != 1 or len(auth.allowed_modules) != 186
            or auth.max_trainable_parameters != 5_411_328):
        raise ValueError('S requires the same rank-eight optimizer recipe for both authorized parents')
    required = {'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': .01,
        'gradient_clip': 1., 'warmup_fraction': .05, 'logits_mode': 'tail'}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError('S optimizer settings differ from the declared baseline')
    limits = protocol['resources']['mode_limits'][mode]
    expected = {'max_steps_per_run': 1 if mode == 'timing' else settings['optimizer_steps'],
        'max_answer_tokens_per_run': limits['max_answer_tokens'], 'max_input_tokens_per_run': limits['max_input_tokens'],
        'max_memory_bytes': protocol['resources']['max_memory_bytes'], 'max_wall_seconds_per_run': limits['max_wall_seconds'],
        'max_cpu_seconds_per_run': limits['max_cpu_seconds'], 'allow_merge': mode == 'main',
        'max_development_checkpoints_per_run': 2 if mode == 'main' else 1, 'learning_rates': (.0003,)}
    if any(getattr(auth, key) != value for key, value in expected.items()):
        raise ValueError('Training authorization differs from the frozen S stage limits')
    if (mode == 'main' and (settings['optimizer_steps'] != 128 or settings['checkpoint_steps'] != [64, 128])
            or mode == 'smoke' and (not 1 <= settings['optimizer_steps'] <= 16
                or settings['checkpoint_steps'] != [settings['optimizer_steps']])):
        raise ValueError('S training or smoke exceeds the prospectively bounded schedule')
    if mode == 'main' and (settings['epochs'] != 1 or len(bundle['rows']) != 2048
            or sum(row['task'] == 'tool' for row in bundle['rows']) != 1024
            or sum(row.get('category') == 'decimal-target' for row in bundle['rows']) != 512):
        raise ValueError('Main training requires the frozen one-epoch 1024/512/512 task allocation')
    return protocol, auth, bundle, settings


def prepare_rows(runner, rows, protocol):
    from weightbridge.adaptation.behavior_endpoint import endpoint_runner
    from weightbridge.adaptation.task_completion import prepare_task_completion
    from experiments.phase_iiiq_baseline.measurement import endpoint
    decimal = endpoint_runner(runner, endpoint(protocol['endpoints']['decimal'])).tokenizer
    return [prepare_task_completion(runner.tokenizer, decimal, row, protocol['tool_completion'],
        tool_limits=protocol['endpoints']['tool'], decimal_limits=protocol['endpoints']['decimal']) for row in rows]


def exposure_receipt(rows, prepared, settings):
    order = data_order(len(rows), settings['seed'], settings['epochs'])
    expected = settings['optimizer_steps'] * settings['effective_batch_size']
    if len(order) < expected:
        raise ValueError('Insufficient prospectively declared example exposure')
    used = order[:expected]
    token_rows = [{'id': row['id'], 'task': row['task'], 'prefix_ids': list(value.prefix_ids),
        'target_ids': list(value.target_ids), 'weight': value.training_weight}
        for row, value in zip(rows, prepared, strict=True)]
    canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return order, {'data_order_sha256': hashlib.sha256(bytes(str(order), 'ascii')).hexdigest(),
        'ordered_row_ids_sha256': canonical([rows[i]['id'] for i in used]),
        'tokenization_sha256': canonical(token_rows), 'examples_seen': len(used),
        'answer_tokens': sum(len(prepared[i].target_ids) for i in used),
        'input_tokens': sum(len(prepared[i].input_ids) for i in used),
        'task_examples': {task: sum(rows[i]['task'] == task for i in used) for task in ('tool', 'decimal')},
        'order_rng': 'independent random.Random per seed/epoch; independent of adapter initialization'}


def run(base, protocol_path, authorization_path, bundle_path, out, *, mode='main', smoke_gate=None):
    import torch
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
    from weightbridge.adaptation.task_completion import completion_accumulation
    protocol, auth, bundle, settings = checked_inputs(protocol_path, authorization_path, bundle_path, mode=mode)
    torch.manual_seed(settings['seed'])
    if mode == 'timing':
        raise ValueError('Timing uses the non-stepping preflight entry point')
    if mode == 'main':
        if not smoke_gate or auth.smoke_gate_sha256 != file_hash(smoke_gate):
            raise ValueError('Main training requires the exact host-admitted smoke receipt')
        gate = read(smoke_gate)
        if (gate.get('passed') is not True or gate.get('protocol_sha256') != file_hash(protocol_path)
                or gate.get('parent_sha256') != auth.parent_sha256):
            raise ValueError('Main training smoke is failed or belongs to a different parent')
    out = Path(out)
    if out.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError('Training output must stay outside the immutable parent')
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    counts = dict(optimizer_steps=0, forward_calls=0, backward_calls=0, answer_tokens=0,
        input_tokens=0, padded_input_token_positions=0, examples_seen=0)
    identity = {'protocol_sha256': file_hash(protocol_path), 'authorization_sha256': file_hash(authorization_path),
        'bundle_sha256': file_hash(bundle_path), 'manifest_sha256': bundle['manifest_sha256'],
        'parent_sha256': auth.parent_sha256, 'run_index': 0, 'smoke': mode == 'smoke', 'mode': mode}
    json_dump_atomic(out/'started.json', {'schema': 'weightbridge.phase_iiis.training_started.v1', **identity})
    session = None
    def usage():
        return {'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu, 'peak_rss_bytes': peak_rss_bytes()}
    def budget():
        resource = usage()
        auth.check_usage(steps=counts['optimizer_steps'], answer_tokens=counts['answer_tokens'],
            input_tokens=counts['input_tokens'], wall_seconds=resource['wall_seconds'],
            cpu_seconds=resource['cpu_seconds'], rss_bytes=resource['peak_rss_bytes'])
        return resource
    try:
        with runtime(base, protocol, expected_parent=auth.parent_sha256) as (adapter, runner):
            targets, inventory = discover_lora_targets(adapter, runner)
            session = SupervisedLoRA(runner.model, targets, checkpoint_sha256=auth.parent_sha256,
                rank=8, alpha=16, dropout=0, seed=settings['seed']).attach()
            initial_factor_hash = session.factor_sha256()
            expected_initial = protocol.get('initial_factor_sha256')
            if expected_initial is not None and initial_factor_hash != expected_initial:
                raise ValueError('Initial adapter factors differ from the matched-parent commitment')
            parameters = tuple(session.parameters())
            auth.validate_plan(parent_sha256=auth.parent_sha256, module_names=tuple(t.module_name for t in targets),
                trainable_parameters=sum(p.numel() for p in parameters), rank=8, alpha=16, dropout=0,
                learning_rate=settings['learning_rates'][0], steps=settings['optimizer_steps'], run_index=0)
            runner.model.train()
            rows = prepare_rows(runner, bundle['rows'], protocol)
            order, exposure = exposure_receipt(bundle['rows'], rows, settings)
            if exposure['answer_tokens'] > auth.max_answer_tokens_per_run or exposure['input_tokens'] > auth.max_input_tokens_per_run:
                raise ValueError('Prospective complete token exposure exceeds authorization')
            json_dump_atomic(out/'inventory.json', {**inventory, 'trainable_parameters': sum(p.numel() for p in parameters),
                'initial_factor_sha256': initial_factor_hash, 'training_examples': len(rows), **exposure,
                'trainable_tensors': [{'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype)}
                    for name, p in session.adapters.named_parameters()]})
            initial = [p.detach().clone() for p in parameters]
            gradient_steps = {name: 0 for name, _ in session.adapters.named_parameters()}
            optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rates'][0], betas=tuple(settings['betas']),
                eps=settings['epsilon'], weight_decay=settings['weight_decay'])
            warmup = math.ceil(settings['optimizer_steps'] * settings['warmup_fraction'])
            checkpoints, losses = [], []
            pad = runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None else runner.tokenizer.eos_token_id
            for step in range(settings['optimizer_steps']):
                budget()
                indices = order[step*16:(step+1)*16]
                selected = [rows[index] for index in indices]
                optimizer.zero_grad(set_to_none=True)
                rate = settings['learning_rates'][0] * learning_rate_factor(step, settings['optimizer_steps'], warmup)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                value, work = completion_accumulation(runner.model, selected, micro_batch_size=4,
                    pad_token_id=pad, device=runner.device, logits_mode=settings['logits_mode'],
                    categories=[bundle['rows'][i].get('category', bundle['rows'][i]['task']) for i in indices])
                category_losses = {}
                for measured in work:
                    counts['forward_calls'] += 1
                    counts['backward_calls'] += 1
                    counts['answer_tokens'] += measured['supervised_tokens']
                    counts['input_tokens'] += measured['unmasked_input_tokens']
                    counts['padded_input_token_positions'] += measured['padded_input_token_positions']
                    counts['examples_seen'] += measured['examples_seen']
                    for category, totals in measured['by_category'].items():
                        observed = category_losses.setdefault(category, {key: 0 for key in totals})
                        for key, total in totals.items():
                            observed[key] += total
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError('Missing or nonfinite adapter gradient')
                for name, p in session.adapters.named_parameters():
                    gradient_steps[name] += int(bool(torch.count_nonzero(p.grad)))
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip']))
                if not math.isfinite(norm):
                    raise RuntimeError('Nonfinite gradient norm')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise RuntimeError('Nonfinite adapter factor')
                session.assert_base_unchanged()
                counts['optimizer_steps'] += 1
                record = {'step': step+1, 'loss': value, 'learning_rate': rate, 'gradient_norm_before_clip': norm,
                    'counts': dict(counts), 'resources': budget(), 'by_category': category_losses,
                    'task_examples': {task: sum(bundle['rows'][i]['task'] == task for i in indices) for task in ('tool', 'decimal')}}
                if step == 0 or (step+1) % settings['norm_measurement_interval'] == 0:
                    with torch.no_grad():
                        square = sum(float(torch.sum((part.B.double().T @ part.B.double()) *
                            (part.A.double() @ part.A.double().T))) * part.scale**2 for part in session.adapters.values())
                    record['unrounded_effective_update_norm'] = math.sqrt(max(0., square))
                json_dump_atomic(out/'steps'/f'{step+1:04d}.json', record)
                losses.append(record)
                print(f"step={step+1}/{settings['optimizer_steps']} loss={value:.6f} wall={record['resources']['wall_seconds']:.1f}s", flush=True)
                if step+1 in settings['checkpoint_steps']:
                    checkpoint = out/'checkpoints'/f'step-{step+1:04d}'
                    saved = session.save(checkpoint/'adapter.safetensors', metadata={**identity, 'step': step+1})
                    json_dump_atomic(checkpoint/'receipt.json', {**saved, 'step': step+1,
                        'effective_update': session.effective_statistics(), 'counts': dict(counts), 'training_loss': value,
                        'finalist_eligible_step': mode == 'main' and step+1 == 128})
                    checkpoints.append(str(checkpoint))
            changed = [{'name': name, 'changed_entries': int(torch.count_nonzero(p.detach()-old)),
                'norm': float(torch.linalg.vector_norm((p.detach()-old).double())), 'nonzero_gradient_steps': gradient_steps[name]}
                for (name, p), old in zip(session.adapters.named_parameters(), initial, strict=True)]
            session.assert_base_unchanged(hash_check=True)
            session.detach()
            session = None
        verify_sources(protocol['source_sha256'])
        outcome = {'schema': 'weightbridge.phase_iiis.training_completed.v1', **identity, 'status': 'COMPLETED',
            'counts': counts, 'checkpoints': checkpoints, 'parameter_changes': changed,
            'all_factors_changed': all(row['changed_entries'] > 0 for row in changed),
            'all_factors_received_nonzero_gradient': all(row['nonzero_gradient_steps'] > 0 for row in changed),
            'initial_factor_sha256': initial_factor_hash, 'first_step_loss': losses[0]['loss'],
            'last_step_loss': losses[-1]['loss'], 'base_unchanged': True, 'resources': budget()}
        json_dump_atomic(out/'completed.json', outcome)
        return outcome
    except BaseException as exc:
        json_dump_atomic(out/'failed.json', {**identity, 'status': 'FAILED', 'error': type(exc).__name__,
            'message': str(exc), 'counts': counts, 'resources': usage(), 'retry_authorized': False})
        raise
    finally:
        if session is not None:
            session.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'protocol', 'authorization', 'bundle', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--mode', choices=('main', 'smoke'), default='main')
    parser.add_argument('--smoke-gate', type=Path)
    args = parser.parse_args()
    run(args.base, args.protocol, args.authorization, args.bundle, args.out, mode=args.mode, smoke_gate=args.smoke_gate)
