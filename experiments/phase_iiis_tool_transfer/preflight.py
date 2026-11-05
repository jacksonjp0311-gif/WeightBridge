"""At most four disposable training-only F/B batches; never an optimizer step."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from weightbridge.core import file_hash, json_dump_atomic
from weightbridge.provenance import peak_rss_bytes
from .train import checked_inputs, runtime, prepare_rows


def timing_indices(rows, prepared):
    """Prospective length-only selection: mixed representative and longest rows."""
    if len(rows) < 8 or len(rows) != len(prepared) or {row['task'] for row in rows} != {'tool', 'decimal'}:
        raise ValueError('Timing needs at least eight training-only mixed examples')
    ranked = sorted(range(len(rows)), key=lambda i: (-len(prepared[i].input_ids), rows[i]['id']))
    by_task = {task: [i for i in ranked if rows[i]['task'] == task] for task in ('tool', 'decimal')}
    if min(map(len, by_task.values())) < 2:
        raise ValueError('Timing needs two examples per task')
    mixed = [by_task['tool'][0], by_task['decimal'][0], by_task['tool'][1], by_task['decimal'][1]]
    return [mixed, mixed, ranked[:4], ranked[-4:]]


def run(base, protocol_path, authorization_path, bundle_path, out):
    import torch
    from weightbridge.adaptation.supervised_lora import SupervisedLoRA, discover_lora_targets
    from weightbridge.adaptation.completion_training import collate_completions, completion_loss
    from weightbridge.adaptation.task_completion import _category_loss
    protocol, auth, bundle, settings = checked_inputs(protocol_path, authorization_path, bundle_path, mode='timing')
    out = Path(out)
    if out.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError('Timing output cannot enter immutable checkpoint storage')
    out.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    counts = dict(forward_calls=0, backward_calls=0, optimizer_steps=0, answer_tokens=0, input_tokens=0)
    identity = {'protocol_sha256': file_hash(protocol_path), 'authorization_sha256': file_hash(authorization_path),
        'bundle_sha256': file_hash(bundle_path), 'parent_sha256': auth.parent_sha256}
    session, outcome = None, None
    def resources():
        return {'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu, 'peak_rss_bytes': peak_rss_bytes()}
    def budget():
        used = resources()
        auth.check_usage(steps=0, answer_tokens=counts['answer_tokens'], input_tokens=counts['input_tokens'],
            wall_seconds=used['wall_seconds'], cpu_seconds=used['cpu_seconds'], rss_bytes=used['peak_rss_bytes'])
    try:
        with runtime(base, protocol, expected_parent=auth.parent_sha256) as (adapter, runner):
            targets, inventory = discover_lora_targets(adapter, runner)
            session = SupervisedLoRA(runner.model, targets, checkpoint_sha256=auth.parent_sha256,
                rank=8, alpha=16, dropout=0, seed=606118).attach()
            parameters = tuple(session.parameters())
            auth.validate_plan(parent_sha256=auth.parent_sha256, module_names=tuple(t.module_name for t in targets),
                trainable_parameters=sum(p.numel() for p in parameters), rank=8, alpha=16, dropout=0,
                learning_rate=.0003, steps=1, run_index=0)
            initial = session.factor_sha256()
            prepared = prepare_rows(runner, bundle['rows'], protocol)
            batches = timing_indices(bundle['rows'], prepared)
            json_dump_atomic(out/'inventory.json', {**inventory, 'initial_factor_sha256': initial,
                'trainable_parameters': sum(p.numel() for p in parameters),
                'max_input_length': max(len(row.prefix_ids) for row in prepared),
                'max_answer_length': max(len(row.target_ids) for row in prepared),
                'max_total_length': max(len(row.input_ids) for row in prepared)})
            runner.model.train()
            pad = runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None else runner.tokenizer.eos_token_id
            measurements, reference, equivalence = [], None, None
            for index, indices in enumerate(batches):
                budget()
                for p in parameters:
                    p.grad = None
                batch = collate_completions([prepared[i] for i in indices], pad, device=runner.device)
                started, started_cpu = time.perf_counter(), time.process_time()
                mode = 'all' if index == 0 else 'tail'
                if index < 2:
                    loss, work = completion_loss(runner.model, batch, logits_mode=mode)
                else:
                    loss, work = _category_loss(runner.model, batch,
                        [bundle['rows'][i].get('category', bundle['rows'][i]['task']) for i in indices], mode)
                counts['forward_calls'] += 1
                forward = {'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-started_cpu}
                started, started_cpu = time.perf_counter(), time.process_time()
                loss.backward()
                counts['backward_calls'] += 1
                backward = {'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-started_cpu}
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise ValueError('Timing has missing or nonfinite adapter gradients')
                if index == 0:
                    reference = (float(loss.detach()), [p.grad.detach().clone() for p in parameters])
                elif index == 1:
                    absolute = max(float((p.grad-old).abs().max()) for p, old in zip(parameters, reference[1], strict=True))
                    equivalence = {'all_loss': reference[0], 'tail_loss': float(loss.detach()),
                        'loss_absolute_error': abs(float(loss.detach())-reference[0]),
                        'max_gradient_absolute_error': absolute, 'rtol': .0002, 'atol': .000001,
                        'passed': abs(float(loss.detach())-reference[0]) <= 1e-5 + .0002*abs(reference[0])
                            and all(torch.allclose(p.grad, old, rtol=.0002, atol=.000001)
                                    for p, old in zip(parameters, reference[1], strict=True))}
                    reference = None
                counts['answer_tokens'] += work['supervised_tokens']
                counts['input_tokens'] += work['unmasked_input_tokens']
                measurements.append({'batch': index, 'mode': mode, 'category_accounting': index >= 2,
                    'row_ids': [bundle['rows'][i]['id'] for i in indices],
                    'loss': float(loss.detach()), 'forward': forward, 'backward': backward, 'counts': work,
                    'input_shape': list(batch['input_ids'].shape)})
                del loss, batch
                budget()
            factors_unchanged = session.factor_sha256() == initial
            session.assert_base_unchanged(hash_check=True)
            session.detach()
            session = None
        budget()
        outcome = {'schema': 'weightbridge.phase_iiis.training_timing.v1', **identity,
            'passed': factors_unchanged and equivalence['passed'], 'optimizer_steps': 0,
            'counts': counts, 'initial_factor_sha256': initial, 'factors_unchanged': factors_unchanged,
            'base_unchanged': True, 'equivalence': equivalence, 'batches': measurements, 'resources': resources(),
            'scope': 'Training runtime feasibility only; no model accuracy or learning claim.'}
        json_dump_atomic(out/'timing.json', outcome)
        return outcome
    except BaseException as exc:
        json_dump_atomic(out/'failed.json', {**identity, 'error': type(exc).__name__, 'message': str(exc),
            'counts': counts, 'resources': resources(), 'retry_authorized': False})
        raise
    finally:
        if session is not None:
            session.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'protocol', 'authorization', 'bundle', 'out'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.protocol, args.authorization, args.bundle, args.out)
    raise SystemExit(0 if result['passed'] else 1)
