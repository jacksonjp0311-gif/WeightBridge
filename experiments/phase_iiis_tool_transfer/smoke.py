"""Disposable mixed-task pipeline checks; no development eligibility claims."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from experiments.phase_iiiq_baseline.integrity import read
from .train import checked_inputs, runtime, prepare_rows


def measure(runner, rows, protocol):
    from .measurement import measure_tools
    from experiments.phase_iiiq_baseline.measurement import measure as measure_decimal
    tools = [row for row in rows if row['task'] == 'tool']
    decimals = [row for row in rows if row['task'] == 'decimal']
    return {'tool': measure_tools(runner, tools, protocol['endpoints']['tool'], protocol['tool_completion']),
        'decimal': measure_decimal(runner, decimals, protocol['endpoints']['decimal'])}


def exact_predictions(left, right):
    for task in ('tool', 'decimal'):
        first, second = left[task], right[task]
        if first['endpoint'] != second['endpoint'] or len(first['rows']) != len(second['rows']):
            return False
        for a, b in zip(first['rows'], second['rows'], strict=True):
            for key in ('id', 'prompt', 'generated_token_ids', 'text', 'stop_reason'):
                if a.get(key) != b.get(key):
                    return False
    return True


def matched_loss(runner, rows, protocol):
    import torch
    from weightbridge.adaptation.completion_training import collate_completions
    from weightbridge.adaptation.task_completion import _category_loss
    values = prepare_rows(runner, rows, protocol)
    pad = runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None else runner.tokenizer.eos_token_id
    total, count, forwards, categories = 0., 0., 0, {}
    runner.model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 4):
            subset = values[start:start+4]
            batch = collate_completions(subset, pad, device=runner.device)
            loss, work = _category_loss(runner.model, batch,
                [row.get('category', row['task']) for row in rows[start:start+4]], 'tail')
            total += float(loss) * work['weighted_supervised_tokens']
            count += work['weighted_supervised_tokens']
            forwards += 1
            for category, stats in work['by_category'].items():
                target = categories.setdefault(category, {key: 0 for key in stats})
                for key, value in stats.items():
                    target[key] += value
    return {'mean_token_nll': total/count, 'weighted_answer_tokens': count, 'forward_calls': forwards,
        'backward_calls': 0, 'by_category': categories, 'row_ids': [row['id'] for row in rows],
        'tokenization': [{'id': row['id'], 'prefix_ids': list(value.prefix_ids), 'target_ids': list(value.target_ids)}
            for row, value in zip(rows, values, strict=True)]}


def run(base, protocol_path, authorization_path, bundle_path, out, *, stage, adapter_path=None, adapter_sha256=None):
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
    if stage not in {'zero', 'reload'} or (stage == 'reload') != (adapter_path is not None and adapter_sha256 is not None):
        raise ValueError('Smoke stage requires zero initialization or an exact saved adapter')
    protocol, auth, bundle, _ = checked_inputs(protocol_path, authorization_path, bundle_path, mode='smoke')
    out = Path(out)
    if out.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError('Smoke output cannot enter immutable model storage')
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    identity = {'protocol_sha256': file_hash(protocol_path), 'authorization_sha256': file_hash(authorization_path),
        'bundle_sha256': file_hash(bundle_path), 'parent_sha256': auth.parent_sha256, 'stage': stage}
    session = None
    def resources():
        return {'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu, 'peak_rss_bytes': peak_rss_bytes()}
    try:
        with runtime(base, protocol, expected_parent=auth.parent_sha256) as (adapter, runner):
            targets, _ = discover_lora_targets(adapter, runner)
            if stage == 'zero':
                parent = measure(runner, bundle['rows'], protocol)
                before = matched_loss(runner, bundle['rows'], protocol)
                session = SupervisedLoRA(runner.model, targets, checkpoint_sha256=auth.parent_sha256,
                    rank=8, alpha=16, dropout=0, seed=606118).attach()
                observed = measure(runner, bundle['rows'], protocol)
                result = {'parent_measurement': parent, 'zero_measurement': observed,
                    'matched_loss': before, 'zero_init_parent_identical': exact_predictions(parent, observed),
                    'initial_factor_sha256': session.factor_sha256()}
                passed = result['zero_init_parent_identical']
            else:
                session = SupervisedLoRA.load(runner.model, targets, adapter_path,
                    checkpoint_sha256=auth.parent_sha256, expected_sha256=adapter_sha256).attach()
                observed = measure(runner, bundle['rows'], protocol)
                after = matched_loss(runner, bundle['rows'], protocol)
                with session.disabled():
                    disabled = measure(runner, bundle['rows'], protocol)
                result = {'trained_measurement': observed, 'disabled_measurement': disabled,
                    'matched_loss': after, 'adapter_sha256': adapter_sha256, 'factor_sha256': session.factor_sha256()}
                passed = True
            session.assert_base_unchanged(hash_check=True)
            session.detach()
            session = None
        outcome = {'schema': 'weightbridge.phase_iiis.smoke_check.v1', **identity, **result,
            'passed': passed, 'base_unchanged': True, 'resources': resources()}
        json_dump_atomic(out/'smoke-check.json', outcome)
        return outcome
    except BaseException as exc:
        json_dump_atomic(out/'failed.json', {**identity, 'error': type(exc).__name__, 'message': str(exc), 'resources': resources()})
        raise
    finally:
        if session is not None:
            session.detach()


def validate_gate(protocol_path, zero_path, completed_path, reload_path):
    """Recompute smoke evidence; caller separately enforces owned-process costs."""
    protocol = read(protocol_path)
    zero, trained, reloaded = map(read, (zero_path, completed_path, reload_path))
    for value in (zero, trained, reloaded):
        if (value.get('protocol_sha256') != file_hash(protocol_path)
                or value.get('parent_sha256') != protocol['checkpoint_sha256']
                or value.get('authorization_sha256') != zero.get('authorization_sha256')
                or value.get('bundle_sha256') != zero.get('bundle_sha256')):
            raise ValueError('Smoke evidence belongs to another protocol or parent')
    if len(trained.get('checkpoints', ())) != 1:
        raise ValueError('Smoke must have exactly one bounded final checkpoint')
    checkpoint = Path(trained['checkpoints'][0])
    saved = read(checkpoint/'receipt.json')
    if (saved.get('sha256') != file_hash(checkpoint/'adapter.safetensors')
            or saved['sha256'] != reloaded.get('adapter_sha256')
            or saved.get('factor_sha256') != reloaded.get('factor_sha256')
            or saved.get('configuration', {}).get('checkpoint_sha256') != protocol['checkpoint_sha256']):
        raise ValueError('Smoke reload is not the exact saved adapter from its parent')
    before, after = zero['matched_loss'], reloaded['matched_loss']
    if before['tokenization'] != after['tokenization'] or before['row_ids'] != after['row_ids']:
        raise ValueError('Smoke loss reduction changed its supervised examples')
    import math
    finite = all(math.isfinite(value) and value >= 0 for value in (before['mean_token_nll'], after['mean_token_nll']))
    checks = {'zero_initialization_exact': exact_predictions(zero['parent_measurement'], zero['zero_measurement']),
        'disabled_parent_exact': exact_predictions(zero['parent_measurement'], reloaded['disabled_measurement']),
        'training_completed': trained.get('status') == 'COMPLETED',
        'smoke_steps_bounded': 1 <= trained['counts']['optimizer_steps'] <= 16,
        'loss_decreased': finite and after['mean_token_nll'] < before['mean_token_nll'],
        'all_factors_changed': trained.get('all_factors_changed') is True,
        'all_factor_gradients': trained.get('all_factors_received_nonzero_gradient') is True,
        'base_unchanged': all(value.get('base_unchanged') is True for value in (zero, trained, reloaded)),
        'worker_checks_passed': zero.get('passed') is True and reloaded.get('passed') is True,
        'same_initial_factors': zero['initial_factor_sha256'] == trained['initial_factor_sha256']}
    changes = trained.get('parameter_changes', ())
    checks['measured_factor_changes'] = len(changes) == 372 and all(row['changed_entries'] > 0
        and row['nonzero_gradient_steps'] > 0 and math.isfinite(row['norm']) and row['norm'] > 0 for row in changes)
    return {'schema': 'weightbridge.phase_iiis.smoke_gate.v1', 'protocol_sha256': file_hash(protocol_path),
        'parent_sha256': protocol['checkpoint_sha256'], 'passed': all(checks.values()), 'checks': checks,
        'source_receipts': {str(Path(path).resolve()): file_hash(path) for path in (zero_path, completed_path, reload_path)},
        'loss_before': before['mean_token_nll'], 'loss_after': after['mean_token_nll'],
        'scope': 'Disposable pipeline fit; no independent improvement or behavioral preservation claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'protocol', 'authorization', 'bundle', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--stage', choices=('zero', 'reload'), required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--adapter-sha256')
    args = parser.parse_args()
    result = run(args.base, args.protocol, args.authorization, args.bundle, args.out, stage=args.stage,
        adapter_path=args.adapter, adapter_sha256=args.adapter_sha256)
    raise SystemExit(0 if result['passed'] else 1)
