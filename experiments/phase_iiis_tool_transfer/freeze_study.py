"""Freeze the authorized two-parent study after disposable measured preflight.

This host entry point reads historical exclusions and creates private banks.
It never performs model inference, fits parameters, or opens a bank for fitting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import math
from pathlib import Path
import secrets

from weightbridge.core import file_hash
from .calibration import ENDPOINTS, TOOL_COMPLETION, headroom
from .execute import accounting
from .initialize import freeze
from .integrity import ORIGINAL, DECIMAL_PARENT, read, write_once
from .populations import SPLITS, build_populations
from .verification import validate_worker


def resource_forecast(preflight_root, *, intended_max_sequence_length=None):
    from .study import validate_execution
    root = Path(preflight_root)
    timing = read(root/'timing-v1/measurement/timing.json')
    if (timing.get('passed') is not True or timing.get('counts', {}).get('optimizer_steps') != 0
            or timing['counts'].get('forward_calls') != 4 or timing['counts'].get('backward_calls') != 4
            or timing.get('equivalence', {}).get('passed') is not True
            or timing.get('factors_unchanged') is not True or timing.get('base_unchanged') is not True):
        raise ValueError('Training timing and native mixed-length gradient equivalence must pass')
    batches = timing['batches']
    if (len(batches) != 4 or [b['mode'] for b in batches] != ['all', 'tail', 'tail', 'tail']
            or batches[0]['row_ids'] != batches[1]['row_ids']
            or any(b.get('category_accounting') is not True for b in batches[2:])
            or any(not math.isfinite(b[part]['wall_seconds']) or b[part]['wall_seconds'] < 0
                   for b in batches for part in ('forward', 'backward'))):
        raise ValueError('Timing batch schedule, mixed-loss comparison or accounting differs')
    timing_protocol = root/'timing-v1/protocol.json'
    if timing.get('protocol_sha256') != file_hash(timing_protocol):
        raise ValueError('Timing belongs to another prospective protocol')
    outer_paths = [root/'execution/timing-sequential.receipt.json']
    validate_execution(outer_paths[0], module='experiments.phase_iiis_tool_transfer.preflight',
        out=root/'timing-v1/measurement', protocol_path=timing_protocol)
    measured_length = max(b['input_shape'][1] for b in batches)
    if intended_max_sequence_length is not None and (type(intended_max_sequence_length) is not int or intended_max_sequence_length < 1):
        raise ValueError('Prospective maximum sequence length must be a positive integer')
    length_scale = max(1., (intended_max_sequence_length/measured_length)**2) if intended_max_sequence_length else 1.
    measured_micro = max(b['forward']['wall_seconds']+b['backward']['wall_seconds'] for b in batches if b['mode'] == 'tail')
    measured_forward = max(b['forward']['wall_seconds'] for b in batches)
    micro, forward = measured_micro*length_scale, measured_forward*length_scale
    rates = {}
    for task, count in (('tool', 80), ('decimal', 72)):
        observed = []
        for arm in ('sequential', 'original'):
            protocol_arm = 'original-joint' if arm == 'original' else arm
            path = root/'execution'/f'calibration-{arm}-{task}.receipt.json'
            worker = root/'calibration-v1'/f'{protocol_arm}-{task}.json'
            record = validate_execution(path, module='experiments.phase_iiis_tool_transfer.evaluate',
                out=worker, protocol_path=root/'calibration-v1'/f'{protocol_arm}-protocol.json')
            if read(worker).get('summary', {}).get('presentations') != count:
                raise ValueError('Generation throughput denominator differs from the executed population')
            outer_paths.append(path)
            observed.append(record['execution']['resources']['wall_seconds']/count)
        rates[task] = max(observed)
    tool, legacy = rates['tool'], rates['decimal']
    stages = {
        'two_main_training': 2*(512*micro+212),
        'two_smokes': 128*micro+512*tool+256*legacy+96*forward+240,
        'development_all_representations': 1536*tool+2496*legacy,
        'historical_candidate_replay_rollback': 4992*legacy,
        'two_formal_banks_all_models_replay_rollback': 4096*tool+9216*legacy,
        'four_merges_and_demo': 480+8*tool,
        'already_executed': accounting(root/'execution')['wall_seconds'],
    }
    estimate = sum(stages.values())
    feasible = estimate*1.10 <= 18*3600 and 512*micro+212 <= 13200
    return {'schema': 'weightbridge.phase_iiis.resource_forecast.v1',
        'matched_two_arm_feasible': feasible, 'estimated_model_wall_seconds': estimate,
        'margin_fraction': .10, 'estimate_with_margin_seconds': estimate*1.10,
        'hard_cumulative_wall_seconds': 18*3600, 'stage_estimates_seconds': stages,
        'measured_worst_tail_microbatch_seconds': measured_micro, 'measured_long_forward_seconds': measured_forward,
        'measured_max_sequence_length': measured_length, 'intended_max_sequence_length': intended_max_sequence_length,
        'training_length_scale': length_scale, 'forecast_tail_microbatch_seconds': micro,
        'measured_response_seconds': rates,
        'preflight_receipt_sha256': {str(path): file_hash(path) for path in outer_paths},
        'measured_tree_peak_bytes': read(root/'execution/timing-sequential.receipt.json')['execution']['resources']['peak_rss_bytes'],
        'assumptions': [
            'Every training microbatch costs the slowest measured mixed/longest batch, even the shorter examples.',
            'If prospective training sequences are longer, F/B and loss-forward costs are extrapolated by the squared maximum-length ratio; this is an explicit conservative estimate, not a measured longer batch.',
            'Generation uses the slower parent rate including worker loading and accounting overhead.',
            'Forecast covers both successful arms, both formal banks, distinct reconstruction replays and parent rollbacks.',
            'Four separately authorized merge/reconstruction workers reserve 120 seconds each; main setup/statistics reserve 212 seconds each.',
            'Stage caps can sum above 18 hours; the single cumulative ledger remains authoritative and refuses over-budget launches.',
            'A timing forecast is uncertain; a budget stop remains RESOURCE_LIMITED_OR_INCOMPLETE, not evidence of failed learning.',
        ]}


def run(preflight_root, base, historical_inventory):
    from transformers import AutoTokenizer
    from weightbridge.adaptation.behavior_endpoint import _BareTokenizer
    from weightbridge.adaptation.task_completion import prepare_task_completion
    from weightbridge.inspection.safetensors import load_checkpoint
    root = Path(preflight_root)
    study, host = root/'study-v1', root/'study-v1-host'
    if study.exists() or host.exists():
        raise FileExistsError('No replacement populations or study freeze')
    if load_checkpoint(base).checkpoint_sha256() != ORIGINAL:
        raise ValueError('Tokenization preflight requires the exact original checkpoint')
    cal = root/'calibration-v1'
    admissions = {}
    for arm, identity in (('sequential', DECIMAL_PARENT), ('original-joint', ORIGINAL)):
        worker = validate_worker(cal/f'{arm}-tool.json', cal/'tool.json', cal/f'{arm}-protocol.json',
            identity, None, 'calibration', 'tool', authorization=cal/f'{arm}-tool-authorization.json')
        admissions[arm] = headroom(worker['measurement'])
    if not any(a['plausible'] for a in admissions.values()):
        raise ValueError('NO_USEFUL_BASELINE_HEADROOM: neither unchanged parent passes disposable plausibility')
    forecast = resource_forecast(root)
    if not forecast['matched_two_arm_feasible']:
        raise ValueError('RESOURCE_INFEASIBLE for the proposed matched study; freeze a prospective reduction before any main attempt')
    historical = read(root/'historical-exclusion-v1.json')
    keys = read(root/'private-exclusion-identities-v1.json')
    used = {(row[0], *(Decimal(v) for v in row[1:])) for row in keys['tasks']}
    anchors = {Decimal(v) for v in keys['anchors']}
    seeds = {split: secrets.token_hex(32) for split in SPLITS}
    calibration_rows = read(cal/'tool.json')
    populations, manifest = build_populations(seeds, used_decimal=used, decimal_anchors=anchors,
        calibration_rows=calibration_rows, historical_exclusion_receipt=historical)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    bare = _BareTokenizer(tokenizer)
    token_receipt = {}
    for split, rows in populations.items():
        totals = defaultdict(lambda: {'presentations': 0, 'supervised_tokens': 0, 'input_tokens': 0,
                                     'max_prompt_tokens': 0, 'max_answer_tokens': 0, 'max_total_tokens': 0})
        for row in rows:
            value = prepare_task_completion(tokenizer, bare, row, TOOL_COMPLETION,
                tool_limits=ENDPOINTS['tool'], decimal_limits=ENDPOINTS['decimal'])
            item = totals[row['category']]
            item['presentations'] += 1
            item['supervised_tokens'] += len(value.target_ids)
            item['input_tokens'] += len(value.input_ids)
            item['max_prompt_tokens'] = max(item['max_prompt_tokens'], len(value.prefix_ids))
            item['max_answer_tokens'] = max(item['max_answer_tokens'], len(value.target_ids))
            item['max_total_tokens'] = max(item['max_total_tokens'], len(value.input_ids))
        token_receipt[split] = dict(totals)
    training = [v for s in ('train', 'replay') for v in token_receipt[s].values()]
    answer_tokens = sum(v['supervised_tokens'] for v in training)
    input_tokens = sum(v['input_tokens'] for v in training)
    forecast = resource_forecast(root, intended_max_sequence_length=max(v['max_total_tokens'] for v in training))
    if not forecast['matched_two_arm_feasible']:
        write_once(root/'resource-forecast-refused-v1.json', {'forecast': forecast, 'token_exposure': token_receipt,
            'scope': 'Prospective length-aware resource refusal before fitting; populations were not redrawn.'})
        raise ValueError('RESOURCE_INFEASIBLE after measured-length extrapolation; no populations may be silently reduced')
    template = {'starting_commit': 'baa336e7c517139876a1a5d6309af7199ea1e661',
        'authorized_arms': ['sequential', 'original-joint'], 'run_order': ['sequential', 'original-joint'],
        'task_contract': 'weightbridge.tool_task.v1', 'endpoints': ENDPOINTS, 'tool_completion': TOOL_COMPLETION,
        'initial_factor_sha256': read(root/'timing-v1/measurement/timing.json')['initial_factor_sha256'],
        'calibration_admission': {'at_least_one_parent_plausible': True, 'per_parent': admissions,
            'scope': 'The R8 calibration format and semantic-diversity concerns are retained; a formal headroom shortage cannot be relabeled failed learning.'},
        'resource_forecast': forecast, 'token_exposure': token_receipt,
        'resources': {'max_memory_bytes': 24*1024**3, 'cumulative_model_wall_seconds': 18*3600,
            'mode_limits': {
                'main': {'max_wall_seconds': 13200., 'max_cpu_seconds': 84000.,
                         'max_answer_tokens': answer_tokens, 'max_input_tokens': input_tokens},
                'smoke': {'max_wall_seconds': 3000., 'max_cpu_seconds': 20000.,
                          'max_answer_tokens': 256*96, 'max_input_tokens': 256*(512+96)}},
            'evaluation_limits': {'max_wall_seconds': 1800., 'max_cpu_seconds': 12000., 'max_memory_bytes': 8*1024**3},
            'merge_limits': {'max_wall_seconds': 900., 'max_cpu_seconds': 6000., 'max_memory_bytes': 24*1024**3}},
        'gates': {'tool_minimum_strict_accuracy': .90, 'tool_minimum_own_parent_gain': .10,
            'formal_new_complete_families': 6, 'formal_lost_complete_families': 0,
            'formal_corrected_schema_valid_semantic_errors': 8, 'formal_corrected_error_families': 4,
            'formal_corrected_real_action_types': 2, 'all_prohibited_regressions': 0,
            'development': 'Accuracy/gain/preservation only; formal 32-family and semantic-count requirements apply to both formal banks.',
            'tool_reference': 'Each arm own-parent strict successes, schema-valid semantic successes and schema validity.',
            'decimal_reference': 'Sequential R8; original-joint original; preserve correct responses, complete families and legacy validity.',
            'general_reference': 'Sequential union of original/R8 correctness and applicable legacy validity; control original.',
            'historical_reference': 'Locked old R 832 cases only after both decisions lock; regression evidence, never fresh transfer.',
            'finalist': 'Step128 only, merged BF16 representation; full reconstruction/replay/actual-parent rollback required.',
            'independent_banks': 'Qualification then sealed confirmation. No retraining or other-arm changes from either bank.'},
        'bootstrap': {'unit': 'semantic_family', 'draws': 2000, 'seed': 606118, 'confidence': .95},
        'inherited_r_main_cost': {'wall_seconds': 3123.8772233, 'cpu_seconds': 19433.96875,
                                 'source': 'published Phase R rank8 main execution receipt; separate from additional Phase S cost'},
        'no_resume_or_retry': True, 'no_production_promotion': True}
    receipt = freeze(study, host, template, populations, manifest, read(historical_inventory),
                     calibration_rows=calibration_rows)
    write_once(host/'private-seeds.json', seeds)
    write_once(root/'resource-forecast-v1.json', forecast)
    write_once(root/'calibration-admission-v1.json', template['calibration_admission'])
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight-root', 'base', 'historical-inventory'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.preflight_root, args.base, args.historical_inventory)
    print({'manifest_sha256': result['manifest_sha256'], 'arms': result['arms'],
           'formal_evaluation_authorized': result['formal_evaluation_authorized']})
