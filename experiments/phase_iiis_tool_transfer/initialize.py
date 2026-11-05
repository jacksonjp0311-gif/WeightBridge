"""Host-only write-once initialization from prebuilt Phase S populations.

This module never generates private seeds, evaluates a model or grants a worker
access to confirmation data. Pure builders supply rows; host orchestration owns
calibration admission and later evaluation authorization.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from experiments.phase_iiiq_baseline.train import data_order
from weightbridge.core import file_hash
from .integrity import PARENTS, ROOT, checked_sha256, read, source_manifest, verify_sources, write_once
from .train import authorization


TRAINING = {'learning_rates': [.0003], 'optimizer_steps': 128, 'checkpoint_steps': [64, 128],
    'seed': 606118, 'effective_batch_size': 16, 'micro_batch_size': 4, 'epochs': 1,
    'optimizer': 'AdamW', 'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': .01,
    'gradient_clip': 1., 'warmup_fraction': .05, 'logits_mode': 'tail',
    'scheduler': 'linear warmup then linear decay', 'norm_measurement_interval': 8}
SMOKE = {**TRAINING, 'optimizer_steps': 16, 'checkpoint_steps': [16], 'epochs': 3,
    'norm_measurement_interval': 4, 'disposable': True}


def _fresh_private(path):
    path = Path(path).resolve()
    if path.exists() or path.is_relative_to(ROOT.resolve()):
        raise ValueError('Initialization requires a new private path outside the public repository')
    return path


def _inventory_rows(inventory):
    if isinstance(inventory, dict):
        rows = inventory.get('modules', inventory.get('allowed'))
    else:
        rows = inventory
    if not isinstance(rows, (tuple, list)):
        raise ValueError('Explicit inspected module inventory is required')
    rows = tuple(sorted(deepcopy(rows), key=lambda row: row.get('module_name', row.get('name', ''))))
    names = tuple(row.get('module_name', row.get('name')) for row in rows)
    shapes = tuple(row.get('shape', row.get('weight_shape')) for row in rows)
    if (len(rows) != 186 or len(set(names)) != 186 or any(shape is None or len(shape) != 2 for shape in shapes)
            or 8*sum(sum(shape) for shape in shapes) != 5_411_328):
        raise ValueError('S requires the inspected 186-module rank-eight geometry')
    return rows


def _recipe(template, *, arm, manifest_sha256, sources):
    if arm not in PARENTS:
        raise ValueError('Unknown Phase S parent arm')
    protocol = deepcopy(template)
    if protocol.get('checkpoint_sha256') not in (None, PARENTS[arm]):
        raise ValueError('Template checkpoint differs from the requested explicit parent')
    protocol.update(schema='weightbridge.phase_iiis.protocol.v1', arm=arm,
        checkpoint_sha256=PARENTS[arm], source_sha256=sources, task_manifest_sha256=manifest_sha256,
        training=deepcopy(TRAINING), smoke_training=deepcopy(SMOKE),
        production_authority=False, lineage_authority=False)
    return protocol


def _bundle(path, protocol, rows, manifest_sha256, split_ids):
    value = {'schema': 'weightbridge.phase_iiis.training_bundle.v1', 'protocol_sha256': file_hash(protocol),
        'rows': deepcopy(rows), 'manifest_sha256': manifest_sha256, 'split_ids': list(split_ids)}
    write_once(path, value)
    return value


def freeze_timing(out, protocol_template, rows, inventory, *, arm='sequential'):
    """Commit disposable timing only: four F/B, no optimizer step or formal bank.

    The caller supplies calibration-only tool and decimal rows. At this stage
    an inspected historical inventory can authorize shapes; the worker still
    re-audits all native modules against its actual explicit parent.
    """
    out = _fresh_private(out)
    rows, inv = deepcopy(rows), _inventory_rows(inventory)
    if (len(rows) < 8 or len({r['id'] for r in rows}) != len(rows)
            or {r.get('task') for r in rows} != {'tool', 'decimal'}
            or any(r.get('training_weight', 1.) != 1. for r in rows)
            or any(r.get('wording_group') != 'calibration' for r in rows if r['task'] == 'tool')):
        raise ValueError('Timing accepts only distinct disposable mixed calibration rows')
    out.mkdir(parents=True)
    write_once(out/'rows.json', rows)
    write_once(out/'inventory.json', inv)
    manifest = {'schema': 'weightbridge.phase_iiis.timing_manifest.v1', 'rows_sha256': file_hash(out/'rows.json'),
        'inventory_sha256': file_hash(out/'inventory.json'), 'rows': len(rows),
        'four_forward_four_backward_maximum': True, 'optimizer_steps': 0,
        'scope': 'Disposable training-only timing; not a formal bank or accuracy experiment.'}
    write_once(out/'manifest.json', manifest)
    # Calibration/inference is already frozen separately. Unrelated new host
    # orchestration may continue implementation while this narrow preflight runs.
    needed = {'__init__.py', 'train.py', 'preflight.py', 'integrity.py'}
    sources = {k: v for k, v in source_manifest().items()
        if not k.startswith('experiments/phase_iiis_tool_transfer/') or Path(k).name in needed}
    protocol = _recipe(protocol_template, arm=arm, manifest_sha256=file_hash(out/'manifest.json'), sources=sources)
    protocol.update(stage='disposable-training-timing', training_authorized=False, formal_banks_created=False)
    protocol['resources'] = {**protocol.get('resources', {}), 'max_memory_bytes': 32*1024**3,
        'mode_limits': {'timing': {'max_wall_seconds': 1200., 'max_cpu_seconds': 8000.,
            'max_answer_tokens': 4*4*96, 'max_input_tokens': 4*4*(512+96)}}}
    write_once(out/'protocol.json', protocol)
    _bundle(out/'training-bundle.json', out/'protocol.json', rows, file_hash(out/'manifest.json'), ['timing'])
    policy = authorization(protocol, inv, out/'training-bundle.json', mode='timing')
    write_once(out/'authorization.json', policy.to_dict())
    return {'protocol': str(out/'protocol.json'), 'authorization': str(out/'authorization.json'),
        'bundle': str(out/'training-bundle.json'), 'manifest_sha256': file_hash(out/'manifest.json'),
        'resources': protocol['resources']}


def _validate_formal(populations, manifest, calibration_rows):
    from .populations import SPLITS, TOOL_FAMILIES, DECIMAL_SPECS, CANONICAL, validate_populations
    from .data import digest, validate_tool_rows, assert_partition_isolation
    from experiments.phase_iiiq_baseline import data as decimal_data
    if (set(populations) != set(SPLITS) or manifest.get('schema') != 'weightbridge.phase_iiis.populations.v1'
            or set(manifest.get('splits', {})) != set(SPLITS) or 'seeds' in manifest or 'seed' in manifest):
        raise ValueError('Initializer requires every fresh population and seed commitments only')
    seen_ids, seen_families, seen_tasks, tool_populations = set(), set(), set(), {}
    for split in SPLITS:
        rows, entry = populations[split], manifest['splits'][split]
        checked_sha256(entry['seed_commitment'])
        if (entry.get('rows_sha256') != digest(rows) or entry.get('rows') != len(rows)
                or len(rows) != 8*TOOL_FAMILIES[split] + DECIMAL_SPECS[split].row_count
                or any(row.get('training_weight') != 1. for row in rows)):
            raise ValueError('Population rows differ from their frozen logical commitments')
        ids = {row['id'] for row in rows}
        if len(ids) != len(rows) or ids & seen_ids:
            raise ValueError('Presentation identities leak between partitions')
        seen_ids.update(ids)
        tools = [row for row in rows if row.get('task') == 'tool']
        decimals = [row for row in rows if row.get('task') == 'decimal']
        if len(tools)+len(decimals) != len(rows):
            raise ValueError('Unknown task in a prospective population')
        if tools:
            validate_tool_rows(tools, expected_families=TOOL_FAMILIES[split])
            tool_populations[split] = tools
        decimal_data.validate(decimals, CANONICAL[split], DECIMAL_SPECS[split])
        families = {row['family_id'] for row in decimals}
        tasks = {decimal_data.semantic_key(row['family'], row['operands']) for row in decimals}
        if families & seen_families or tasks & seen_tasks:
            raise ValueError('Decimal family or equivalent comparison leaked across partitions')
        seen_families.update(families)
        seen_tasks.update(tasks)
        if split in {'qualification', 'confirmation'} and sum(row['wording_familiarity'] == 'unseen' for row in tools) < len(tools)/2:
            raise ValueError('Formal transfer needs at least half independently worded families')
    assert_partition_isolation(tool_populations)
    if calibration_rows is not None:
        validate_populations(populations, calibration_rows=calibration_rows)


def prospective_order(rows, settings):
    order = data_order(len(rows), settings['seed'], settings['epochs'])
    if len(order) != 2048 or len(rows) != 2048:
        raise ValueError('Main exposure is exactly one shuffled 2048-row epoch')
    ordered = [rows[index]['id'] for index in order]
    return {'schema': 'weightbridge.phase_iiis.prospective_order.v1', 'seed': settings['seed'],
        'data_order_sha256': hashlib.sha256(bytes(str(order), 'ascii')).hexdigest(),
        'ordered_row_ids_sha256': hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'examples_seen': 2048, 'optimizer_steps': 128, 'microbatches': 512,
        'scope': 'Identical additional training exposure for both parents; inherited R training is separate.'}


def freeze(out, host_root, protocol_template, populations, manifest, inventory, *, calibration_rows=None):
    """Host commits both arms and all banks before any formal optimizer launch.

    ``inventory`` may be one independently inspected shared geometry or a map
    of both actual-parent inventories. Each model worker always re-discovers
    names, shapes and source bytes against its explicit parent before fitting.
    No evaluation authorization is created here; main authorization remains a
    plan until the host admits that arm's one disposable smoke.
    """
    from .populations import SPLITS, HOST_SPLITS
    from .data import digest
    out, host = _fresh_private(out), _fresh_private(host_root)
    if out == host or out.is_relative_to(host) or host.is_relative_to(out):
        raise ValueError('Protected host banks must be disjoint from fitting-worker directories')
    _validate_formal(populations, manifest, calibration_rows)
    initial = checked_sha256(protocol_template.get('initial_factor_sha256'))
    resources = protocol_template.get('resources', {})
    if (not 0 < resources.get('max_memory_bytes', 0) <= 32*1024**3
            or not {'main', 'smoke'} <= set(resources.get('mode_limits', {}))
            or not 0 < resources['mode_limits']['main'].get('max_wall_seconds', 0) <= 4*3600):
        raise ValueError('Prospectively measured training/smoke resource limits are required')
    if isinstance(inventory, dict) and set(inventory) == set(PARENTS):
        inventories = {arm: _inventory_rows(inventory[arm]) for arm in PARENTS}
        for arm in PARENTS:
            if isinstance(inventory[arm], dict) and inventory[arm].get('checkpoint_sha256') != PARENTS[arm]:
                raise ValueError('Actual-parent inventory identity differs')
        geometry = lambda rows: [(row.get('module_name', row.get('name')), row.get('shape', row.get('weight_shape'))) for row in rows]
        if geometry(inventories['sequential']) != geometry(inventories['original-joint']):
            raise ValueError('Matched arms require identical ordered module names and shapes')
        inventory_scope = 'Both supplied parent inventories agree in ordered module names and shapes.'
    else:
        common = _inventory_rows(inventory)
        inventories = {arm: common for arm in PARENTS}
        inventory_scope = 'Shared inspected geometry authorization; each parent must independently re-audit all native modules during smoke and main.'
    sources = source_manifest()
    out.mkdir(parents=True)
    host.mkdir(parents=True)
    material = {}
    for split in SPLITS:
        rows = populations[split]
        target = (host if split in HOST_SPLITS else out/'material')/split
        target.mkdir(parents=True)
        paths = {}
        for task in ('tool', 'decimal'):
            path = target/f'{task}.json'
            write_once(path, [row for row in rows if row['task'] == task])
            paths[task] = {'path': str(path), 'sha256': file_hash(path),
                'rows': sum(row['task'] == task for row in rows)}
        if split not in HOST_SPLITS:
            path = target/'rows.json'
            write_once(path, rows)
            paths['rows'] = {'path': str(path), 'sha256': file_hash(path), 'rows': len(rows)}
        material[split] = {**manifest['splits'][split], 'files': paths,
            'host_private': split in HOST_SPLITS, 'file_hashes_are_serialized_bytes': True}
    frozen = {**deepcopy(manifest), 'splits': material, 'schema': 'weightbridge.phase_iiis.frozen_manifest.v1',
        'population_manifest_sha256': digest(manifest), 'source_sha256': sources,
        'inventory_scope': inventory_scope, 'private_seed_values_recorded': False}
    write_once(out/'manifest.json', frozen)
    manifest_sha = file_hash(out/'manifest.json')
    common_template = deepcopy(protocol_template)
    common_template.pop('checkpoint_sha256', None)
    combined = populations['train'] + populations['replay']
    order = prospective_order(combined, TRAINING)
    write_once(out/'prospective-data-order.json', order)
    arms = {}
    for arm in PARENTS:
        root = out/arm
        root.mkdir()
        protocol = _recipe(common_template, arm=arm, manifest_sha256=manifest_sha, sources=sources)
        protocol.update(stage='prospective-main-study', initial_factor_sha256=initial,
            training_authorized=False, fixed_finalist_step=128, diagnostic_only_steps=[64],
            task_population_counts={'training': 2048, 'smoke': 96, 'development': 336,
                'qualification': 832, 'confirmation': 832})
        path = root/'protocol.json'
        write_once(path, protocol)
        write_once(root/'inventory-authorization.json', {'parent_sha256': PARENTS[arm],
            'scope': inventory_scope, 'modules': inventories[arm]})
        _bundle(root/'training-bundle.json', path, combined, manifest_sha, ['train', 'replay'])
        _bundle(root/'smoke-bundle.json', path, populations['smoke'], manifest_sha, ['smoke'])
        smoke = authorization(protocol, inventories[arm], root/'smoke-bundle.json', mode='smoke')
        plan = authorization(protocol, inventories[arm], root/'training-bundle.json', mode='main')
        write_once(root/'smoke-authorization.json', smoke.to_dict())
        write_once(root/'main-authorization-plan.json', {'schema': 'weightbridge.phase_iiis.authorization_plan.v1',
            'policy': plan.to_dict(), 'requires_smoke_gate': True, 'main_launch_authorized': False})
        write_once(root/'prospective-data-order.json', order)
        arms[arm] = {'protocol_sha256': file_hash(path), 'parent_sha256': PARENTS[arm],
            'initial_factor_sha256': initial, 'training_bundle_sha256': file_hash(root/'training-bundle.json'),
            'smoke_bundle_sha256': file_hash(root/'smoke-bundle.json')}
    receipt = {'schema': 'weightbridge.phase_iiis.initialization.v1', 'manifest_sha256': manifest_sha,
        'source_sha256': sources, 'arms': arms, 'prospective_data_order_sha256': file_hash(out/'prospective-data-order.json'),
        'same_ordered_training_rows': True, 'same_smoke_rows': True,
        'initial_factor_sha256': initial, 'formal_evaluation_authorized': False,
        'production_authority': False, 'lineage_authority': False}
    write_once(out/'initialized.json', receipt)
    return receipt


def _argument(command, flag):
    if not isinstance(command, list) or command.count(flag) != 1:
        raise ValueError('Smoke worker command has missing or repeated binding flags')
    position = command.index(flag)
    if position+1 >= len(command):
        raise ValueError('Smoke worker command is incomplete')
    return command[position+1]


def _validated_main_policy(arm_root, smoke_gate, *, execution_receipts, require_fresh):
    """Replay a complete owned smoke before issuing one executable main policy.

    Failed/interrupted process scopes cannot be replaced by a passing inner
    receipt. An existing authorization or main launch claim consumes admission;
    this API does not implement resume or issue replacement training attempts.
    """
    from .smoke import validate_gate
    root, gate_path = Path(arm_root).resolve(), Path(smoke_gate).resolve()
    protocol_path = root/'protocol.json'
    protocol, gate = read(protocol_path), read(gate_path)
    verify_sources(protocol['source_sha256'])
    if (not gate_path.is_relative_to(root) or require_fresh and any((root/name).exists() for name in
            ('main-authorization.json', 'main-admission.json', 'main-launch-claim.json', 'training/started.json'))):
        raise ValueError('Main admission requires a fresh one-shot arm and its own smoke gate')
    source_receipts = gate.get('source_receipts', {})
    if len(source_receipts) != 3:
        raise ValueError('Smoke admission needs the three exact raw worker receipts')
    raw = {}
    for name, expected in source_receipts.items():
        path = Path(name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError('Smoke raw receipt identity or arm boundary changed')
        record = read(path)
        if (record.get('authorization_sha256') != file_hash(root/'smoke-authorization.json')
                or record.get('bundle_sha256') != file_hash(root/'smoke-bundle.json')):
            raise ValueError('Smoke raw evidence used another authorization or fitting bundle')
        kind = 'train' if record.get('mode') == 'smoke' else record.get('stage')
        if kind not in {'zero', 'train', 'reload'} or kind in raw:
            raise ValueError('Smoke must contain exactly zero/train/reload evidence')
        raw[kind] = (path, record)
    recomputed = validate_gate(protocol_path, raw['zero'][0], raw['train'][0], raw['reload'][0])
    if gate != recomputed or not recomputed['passed']:
        raise ValueError('Smoke gate is failed or does not independently replay')
    trained = raw['train'][1]
    steps = protocol['smoke_training']['optimizer_steps']
    expected_counts = {'optimizer_steps': steps, 'forward_calls': steps*4,
        'backward_calls': steps*4, 'examples_seen': steps*16}
    if (any(trained['counts'].get(key) != value for key, value in expected_counts.items())
            or trained.get('initial_factor_sha256') != protocol['initial_factor_sha256']):
        raise ValueError('Smoke executed another optimizer exposure or initialization')
    executions = tuple(Path(path).resolve() for path in execution_receipts)
    if len(executions) != 3 or len(set(executions)) != 3:
        raise ValueError('All three distinct owned-process smoke receipts are required')
    stages, resource_rows = {}, []
    limits = protocol['resources']['mode_limits']['smoke']
    for path in executions:
        record = read(path)
        execution, command = record.get('execution') or {}, record.get('command')
        if (record.get('schema') != 'weightbridge.phase_iiis.execution.v1'
                or record.get('passed') is not True or record.get('error') is not None
                or execution.get('exit_code') != 0 or execution.get('budget_failure') is not None
                or not isinstance(command, list) or len(command) < 4 or command[1] != '-m'):
            raise ValueError('A failed, incomplete or foreign outer smoke execution cannot admit main training')
        if command[2] == 'experiments.phase_iiis_tool_transfer.train':
            if _argument(command, '--mode') != 'smoke':
                raise ValueError('Outer training scope is not the authorized disposable smoke')
            kind = 'train'
        elif command[2] == 'experiments.phase_iiis_tool_transfer.smoke':
            kind = _argument(command, '--stage')
        else:
            raise ValueError('Unexpected executable in the smoke admission chain')
        if kind not in raw or kind in stages:
            raise ValueError('Outer smoke scopes are duplicated or missing')
        for flag, expected in (('--protocol', protocol_path), ('--authorization', root/'smoke-authorization.json'),
                ('--bundle', root/'smoke-bundle.json'), ('--out', raw[kind][0].parent)):
            if Path(_argument(command, flag)).resolve() != expected.resolve():
                raise ValueError('Outer process command does not bind the exact smoke inputs and output')
        log = path.parent/f"{record['name']}.log"
        if not log.is_file() or record.get('log_sha256') != file_hash(log):
            raise ValueError('Outer smoke execution log changed')
        resources = execution.get('resources') or {}
        processes = execution.get('processes') or []
        sleep = record.get('temporary_sleep_request') or {}
        if (resources.get('cpu_accounting_complete_for_observed_processes') is not True
                or not processes or sleep.get('requested') and (sleep.get('succeeded') is not True or sleep.get('released') is not True)):
            raise ValueError('Smoke process-tree accounting or sleep-prevention release is incomplete')
        if (not math.isclose(sum(p['cpu_seconds'] for p in processes), resources['observed_tree_cpu_seconds'], abs_tol=1e-7)
                or not math.isclose(resources['host_cpu_seconds']+resources['observed_tree_cpu_seconds'], resources['cpu_seconds'], abs_tol=1e-7)
                or sum(p['peak_rss_bytes'] for p in processes) != resources['peak_rss_bytes']):
            raise ValueError('Smoke process-tree resource totals do not replay')
        for observed, declared in (('wall_seconds', 'wall_limit'), ('cpu_seconds', 'cpu_limit'), ('peak_rss_bytes', 'memory_limit')):
            value, bound = resources[observed], record['limits'][declared]
            if not math.isfinite(value) or value < 0 or value > bound:
                raise ValueError('Outer smoke resource limit was violated')
        inner = raw[kind][1].get('resources', {})
        if (resources['observed_tree_cpu_seconds']+.01 < inner.get('cpu_seconds', 0)
                or resources['peak_rss_bytes'] < inner.get('peak_rss_bytes', 0)):
            raise ValueError('Smoke outer monitor omitted worker resource measurements')
        stages[kind] = {'path': str(path), 'sha256': file_hash(path)}
        resource_rows.append(resources)
    total = {'wall_seconds': sum(row['wall_seconds'] for row in resource_rows),
        'cpu_seconds': sum(row['cpu_seconds'] for row in resource_rows),
        'peak_rss_bytes': max(row['peak_rss_bytes'] for row in resource_rows)}
    if (total['wall_seconds'] > limits['max_wall_seconds'] or total['cpu_seconds'] > limits['max_cpu_seconds']
            or total['peak_rss_bytes'] > protocol['resources']['max_memory_bytes']):
        raise ValueError('Complete disposable smoke exceeded its frozen aggregate budget')
    inv = read(root/'inventory-authorization.json')['modules']
    policy = authorization(protocol, inv, root/'training-bundle.json', mode='main', smoke_gate=gate_path)
    plan = read(root/'main-authorization-plan.json')
    planned = deepcopy(plan['policy'])
    planned['smoke_gate_sha256'] = file_hash(gate_path)
    if (plan.get('schema') != 'weightbridge.phase_iiis.authorization_plan.v1'
            or plan.get('main_launch_authorized') is not False
            or json.loads(json.dumps(policy.to_dict())) != planned):
        raise ValueError('Executable main authorization differs from the precommitted policy plan')
    admission = {'schema': 'weightbridge.phase_iiis.main_admission.v1',
        'protocol_sha256': file_hash(protocol_path), 'smoke_gate_sha256': file_hash(gate_path),
        'smoke_gate_path': str(gate_path),
        'outer_executions': stages, 'smoke_resources': total, 'parent_sha256': protocol['checkpoint_sha256'],
        'passed': True, 'planned_authorization': policy.to_dict(),
        'retries_authorized': False, 'production_authority': False}
    return policy, admission


def authorize_main(arm_root, smoke_gate, *, execution_receipts):
    policy, admission = _validated_main_policy(arm_root, smoke_gate,
        execution_receipts=execution_receipts, require_fresh=True)
    root = Path(arm_root).resolve()
    # A crash after admission but before policy output is incomplete admission,
    # never permission to issue a replacement optimizer attempt.
    write_once(root/'main-admission.json', admission)
    write_once(root/'main-authorization.json', policy.to_dict())
    return policy.to_dict()


def validate_main_admission(arm_root):
    """Read-only independent replay of the exact smoke, policy and launch claim."""
    root = Path(arm_root).resolve()
    admission = read(root/'main-admission.json')
    gate_path = Path(admission['smoke_gate_path'])
    executions = []
    for entry in admission['outer_executions'].values():
        path = Path(entry['path'])
        if file_hash(path) != entry['sha256']:
            raise ValueError('Admitted outer smoke process identity changed')
        executions.append(path)
    policy, recomputed = _validated_main_policy(root, gate_path, execution_receipts=executions, require_fresh=False)
    normalized = lambda value: json.loads(json.dumps(value))
    if (admission != normalized(recomputed) or read(root/'main-authorization.json') != normalized(policy.to_dict())):
        raise ValueError('Stored main admission or authorization does not independently replay')
    claim = read(root/'main-launch-claim.json')
    expected = {'schema': 'weightbridge.phase_iiis.optimizer_launch_claim.v1', 'mode': 'main',
        'protocol_sha256': file_hash(root/'protocol.json'),
        'authorization_sha256': file_hash(root/'main-authorization.json'),
        'bundle_sha256': file_hash(root/'training-bundle.json'), 'parent_sha256': policy.parent_sha256,
        'out': str(root/'training'), 'run_index': 0, 'retries_authorized': False, 'resume_implemented': False,
        'interruption_disposition': 'INTERRUPTED_INCOMPLETE', 'production_authority': False}
    if claim != expected:
        raise ValueError('Main launch claim differs from the admitted one-shot worker inputs')
    names = ['protocol.json', 'main-authorization.json', 'main-authorization-plan.json', 'main-admission.json',
        'main-launch-claim.json', 'smoke-authorization.json', 'smoke-bundle.json', 'training-bundle.json',
        'inventory-authorization.json']
    paths = [root/name for name in names] + [gate_path] + executions
    paths.extend(Path(name) for name in read(gate_path)['source_receipts'])
    for path in executions:
        paths.append(path.parent/f"{read(path)['name']}.log")
    return {'schema': 'weightbridge.phase_iiis.main_admission_replay.v1', 'passed': True,
        'parent_sha256': policy.parent_sha256, 'protocol_sha256': file_hash(root/'protocol.json'),
        'input_sha256': {str(path.resolve()): file_hash(path) for path in paths}}


def claim_training(arm_root, *, mode, out=None):
    """Consume the one optimizer attempt before a host creates its child process."""
    from .train import checked_inputs
    if mode not in {'main', 'smoke'}:
        raise ValueError('Only the two declared optimizer stages have launch claims')
    root = Path(arm_root).resolve()
    expected_out = root/('training' if mode == 'main' else 'smoke/training')
    out = expected_out if out is None else Path(out).resolve()
    claim = root/f'{mode}-launch-claim.json'
    if out != expected_out or out.exists() or claim.exists():
        raise ValueError('Optimizer launch is one-shot at its fixed new output path; no retries or resume')
    protocol = root/'protocol.json'
    auth = root/f'{mode}-authorization.json'
    bundle = root/('training-bundle.json' if mode == 'main' else 'smoke-bundle.json')
    config, policy, _, _ = checked_inputs(protocol, auth, bundle, mode=mode)
    if mode == 'main':
        admission = read(root/'main-admission.json')
        if (admission.get('passed') is not True or admission.get('protocol_sha256') != file_hash(protocol)
                or admission.get('parent_sha256') != config['checkpoint_sha256']
                or admission.get('planned_authorization') != read(auth)
                or admission.get('smoke_gate_sha256') != policy.smoke_gate_sha256):
            raise ValueError('Main launch has no independently admitted exact executable policy')
    value = {'schema': 'weightbridge.phase_iiis.optimizer_launch_claim.v1', 'mode': mode,
        'protocol_sha256': file_hash(protocol), 'authorization_sha256': file_hash(auth),
        'bundle_sha256': file_hash(bundle), 'parent_sha256': config['checkpoint_sha256'],
        'out': str(out), 'run_index': 0, 'retries_authorized': False, 'resume_implemented': False,
        'interruption_disposition': 'INTERRUPTED_INCOMPLETE', 'production_authority': False}
    write_once(claim, value)
    return value


def claim_main(arm_root, out=None):
    return claim_training(arm_root, mode='main', out=out)


def claim_smoke(arm_root, out=None):
    return claim_training(arm_root, mode='smoke', out=out)
