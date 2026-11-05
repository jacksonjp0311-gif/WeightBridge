"""Host-only disposable calibration commitments, before any formal bank exists."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import secrets

from experiments.phase_iiiq_baseline import data as decimals
from weightbridge.adaptation.task_completion import prepare_tool_completion
from weightbridge.core import file_hash
from .data import generate_tool_families, decimal_rows, validate_tool_rows
from .integrity import ORIGINAL, DECIMAL_PARENT, source_manifest, read, write_once
from .verification import authorize_evaluation

TOOL_COMPLETION = {'schema': 'weightbridge.native_tool_completion.v1',
                   'end_token_id': 248046, 'end_token_surface': '<|im_end|>'}
ENDPOINTS = {
    'tool': {'cpu_threads': 6, 'batch_size': 8, 'max_input_tokens': 512, 'max_new_tokens': 96},
    'decimal': {'cpu_threads': 6, 'batch_size': 8, 'max_input_tokens': 128, 'max_new_tokens': 16},
}
CALIBRATION_HEADROOM = {
    'minimum_schema_valid_fraction': .5,
    'minimum_potential_strict_gain': .10,
    'minimum_incomplete_family_fraction': 6/32,
    'minimum_schema_valid_semantic_error_fraction': 8/256,
    'minimum_semantic_error_family_fraction': 4/32,
    'minimum_real_action_types_with_semantic_error': 2,
    'interpretation': 'Prospective disposable plausibility screen, not a prediction or replacement of formal additive-count gates.',
}


def headroom(measurement):
    from .measurement import summarize_tools
    summary = summarize_tools(measurement)
    wrong = [r for r in measurement['rows'] if r['schema_valid'] and not r['semantic_correct']]
    kinds = {r['gold_action']['tool'] for r in wrong} & {'search_files', 'read_file', 'run_tests'}
    checks = {
        'competent_schema': summary['schema_valid']/summary['presentations'] >= .5,
        'ten_point_gain_possible': 1-summary['strict_accuracy'] >= .10-1e-12,
        'incomplete_family_opportunities': 1-summary['complete_families']/summary['families'] >= 6/32,
        'semantic_action_errors': len(wrong)/summary['presentations'] >= 8/256,
        'semantic_error_family_diversity': len({r['family_id'] for r in wrong})/summary['families'] >= 4/32,
        'semantic_error_action_diversity': len(kinds) >= 2,
    }
    return {'plausible': all(checks.values()), 'checks': checks, 'summary': summary,
            'schema_valid_semantic_errors': len(wrong),
            'semantic_error_families': len({r['family_id'] for r in wrong}),
            'semantic_error_real_action_types': sorted(kinds), 'frozen_screen': CALIBRATION_HEADROOM}


def freeze(out, base):
    """Disposable populations and immutable calibration settings; no formal rosters."""
    from transformers import AutoTokenizer
    out = Path(out)
    if out.exists():
        raise FileExistsError('Disposable calibration is write-once')
    out.mkdir(parents=True)
    seeds = {task: secrets.token_hex(32) for task in ('tool', 'decimal')}
    tools = generate_tool_families('calibration', 10, seeds['tool'])
    used, anchors = decimals._blocked()
    spec = decimals.PopulationSpec(target_families=4, comparison_controls=4,
                                   copy_controls=4, addition_controls=4, successor_controls=4)
    legacy = decimal_rows(decimals.generate('smoke', spec, seeds['decimal'], used, anchors))
    write_once(out/'private-seeds.json', seeds)
    for task, rows in (('tool', tools), ('decimal', legacy)):
        write_once(out/f'{task}.json', rows)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    encoded = [prepare_tool_completion(tokenizer, r['prompt'], r['gold_action'], TOOL_COMPLETION) for r in tools]
    manifest = {'schema': 'weightbridge.phase_iiis.disposable_manifest.v1',
        'role': 'Disposable calibration only; exclude every family and wording template from formal data',
        'splits': {task: {'sha256': file_hash(out/f'{task}.json'), 'rows': len(rows),
                         'seed_commitment': hashlib.sha256(seeds[task].encode()).hexdigest()}
                   for task, rows in (('tool', tools), ('decimal', legacy))},
        'tool_population': validate_tool_rows(tools), 'decimal_population': asdict(spec),
        'tool_native_tokens': {'minimum_prompt': min(len(r.prefix_ids) for r in encoded),
                              'maximum_prompt': max(len(r.prefix_ids) for r in encoded),
                              'maximum_answer_including_terminator': max(len(r.target_ids) for r in encoded)},
        'protected_formal_data_created': False}
    write_once(out/'manifest.json', manifest)
    # Training-only files can continue implementation while inference is frozen.
    names = {'__init__.py', 'calibration.py', 'data.py', 'evaluate.py', 'measurement.py',
             'verification.py', 'integrity.py', 'execute.py'}
    sources = {k: v for k, v in source_manifest().items()
               if not k.startswith('experiments/phase_iiis_tool_transfer/') or Path(k).name in names}
    for arm, parent in (('sequential', DECIMAL_PARENT), ('original-joint', ORIGINAL)):
        protocol = {'schema': 'weightbridge.phase_iiis.protocol.v1', 'stage': 'disposable-calibration',
            'arm': arm, 'checkpoint_sha256': parent, 'source_sha256': sources,
            'task_manifest_sha256': file_hash(out/'manifest.json'), 'endpoints': ENDPOINTS,
            'tool_completion': TOOL_COMPLETION, 'resources': {'max_memory_bytes': 32*1024**3},
            'limits': {'per_worker_wall_seconds': 1800, 'per_worker_cpu_seconds': 12000,
                       'cumulative_model_wall_seconds': 18*3600},
            'calibration_headroom': CALIBRATION_HEADROOM,
            'training_authorized': False, 'production_authority': False, 'lineage_authority': False}
        path = out/f'{arm}-protocol.json'
        write_once(path, protocol)
        for task in ('tool', 'decimal'):
            authorize_evaluation(out/f'{arm}-{task}-authorization.json', path, stage='calibration', task=task,
                                 roster_sha256=file_hash(out/f'{task}.json'), representation_sha256=parent)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    args = parser.parse_args()
    print(freeze(args.out, args.base))
