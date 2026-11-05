"""One owned model worker per receipt, with a cumulative non-nested ledger.

Uses the established Windows process-tree monitor and temporary sleep request.
An interrupted stage is never implicitly resumed or replaced. This launcher
does not grant access to a roster; the task-specific worker checks its binding.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from experiments.phase_iiiq_baseline.run_training import launch_owned_tree, _temporary_sleep_request
from weightbridge.core import file_hash
from .integrity import ROOT, read, write_once

GLOBAL_WALL_LIMIT = 18 * 3600
MEMORY_LIMIT = 32 * 1024**3


def accounting(root):
    rows = []
    for path in sorted(Path(root).glob('*.receipt.json')):
        record = read(path)
        if record.get('schema') != 'weightbridge.phase_iiis.execution.v1':
            raise ValueError('Unexpected model-work ledger record')
        execution = record.get('execution') or {}
        resources = execution.get('resources')
        if resources is None:
            raise ValueError('Incomplete execution has unknown resource use; manual audit required')
        rows.append({'name': record['name'], 'receipt_sha256': file_hash(path),
                     'passed': record['passed'], **resources})
    unfinished = [p.name for p in Path(root).glob('*.started.json')
                  if not p.with_name(p.name.replace('.started.json', '.receipt.json')).exists()]
    return {'scopes': rows, 'unfinished': unfinished,
            'wall_seconds': sum(r['wall_seconds'] for r in rows),
            'cpu_seconds': sum(r['cpu_seconds'] for r in rows),
            'peak_rss_bytes': max((r['peak_rss_bytes'] for r in rows), default=0),
            'scope': 'Sum of disjoint owned-worker scopes including monitor CPU; inner worker measurements are not added again.'}


def run(root, name, module, arguments, *, wall_limit, cpu_limit, memory_limit=MEMORY_LIMIT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name)
            or not module.startswith('experiments.phase_iiis_tool_transfer.')
            or not 0 < wall_limit <= GLOBAL_WALL_LIMIT or not 0 < memory_limit <= MEMORY_LIMIT
            or not math.isfinite(cpu_limit) or cpu_limit <= 0):
        raise ValueError('Invalid bounded Phase S worker')
    prior = accounting(root)
    if prior['unfinished']:
        raise RuntimeError('An interrupted or active model stage requires resolution before further work')
    if prior['wall_seconds'] + wall_limit > GLOBAL_WALL_LIMIT:
        raise RuntimeError('Requested stage exceeds the remaining 18-hour cumulative model-work budget')
    started, receipt, log = (root / f'{name}{suffix}' for suffix in ('.started.json', '.receipt.json', '.log'))
    if any(path.exists() for path in (started, receipt, log)):
        raise FileExistsError('No implicit retry, overwrite or exact-resume claim')
    command = [sys.executable, '-m', module, *map(str, arguments)]
    limits = {'wall_limit': wall_limit, 'cpu_limit': cpu_limit, 'memory_limit': memory_limit}
    record = {'schema': 'weightbridge.phase_iiis.execution.v1', 'name': name, 'command': command,
              'limits': limits, 'prior_wall_seconds': prior['wall_seconds'], 'execution': None,
              'passed': False, 'error': None, 'production_authority': False}
    write_once(started, record)
    try:
        with _temporary_sleep_request() as sleep:
            record['temporary_sleep_request'] = sleep
            if sleep['requested'] and not sleep['succeeded']:
                raise RuntimeError('Windows sleep-prevention request failed before worker launch')
            record['execution'] = launch_owned_tree(command, log, cwd=ROOT, **limits)
        execution, resources = record['execution'], record['execution']['resources']
        record['passed'] = (execution['exit_code'] == 0 and execution['budget_failure'] is None
            and resources['cpu_accounting_complete_for_observed_processes']
            and (not sleep['requested'] or sleep['released'] is True)
            and all(math.isfinite(resources[key]) and 0 <= resources[key] <= limits[bound]
                    for key, bound in (('wall_seconds', 'wall_limit'), ('cpu_seconds', 'cpu_limit'),
                                       ('peak_rss_bytes', 'memory_limit'))))
        if not record['passed']:
            record['error'] = {'type': 'ExecutionGateFailure', 'message': 'Owned worker or resource gate failed'}
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        record['log_sha256'] = file_hash(log) if log.exists() else None
        write_once(receipt, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--wall', type=float, required=True)
    parser.add_argument('--cpu', type=float, required=True)
    parser.add_argument('--memory', type=int, default=MEMORY_LIMIT)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    tail = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    result = run(args.ledger, args.name, args.module, tail,
                 wall_limit=args.wall, cpu_limit=args.cpu, memory_limit=args.memory)
    print({'name': args.name, 'passed': result['passed'], 'execution': result['execution']})
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
