"""Summarize the preregistered matched100 screen before production admission."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.t2a.experiments.ar_instruction_t200_v3 import policy
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write


def metrics(row):
    result = {k: float(v) for k, v in row['native']['metrics'].items() if isinstance(v, (int, float))}
    result['parse_success'] = float(row['native']['status'] == 'ok')
    result.update({k: float(v) for k, v in row['requested']['metrics'].items() if v is not None})
    for role in ('edited', 'unchanged'):
        sources = [s for s in row['source_fields']['sources'] if s['role'] == role]
        result[f'{role}_semantic_token_f1'] = sum(s['semantic_token_f1_with_missing_zero'] or 0. for s in sources) / len(sources) if sources else None
    return result


def review(stage):
    refs = {str(Path(__file__)): sha(__file__)}
    def read(path):
        refs[str(path)] = sha(path)
        return policy.read(path)
    protocol = read(stage / 'PROTOCOL.json')
    plan = read(stage / 'evaluation/PLAN.json')
    training = read(stage / 'TRAINING_REVIEW.json')
    assert training['matched250_training_complete'] and training['three_rank_native_resume_exact']
    for path, expected in plan['source_sha256'].items(): assert sha(path) == expected, path
    assert plan['protocol_sha256'] == sha(stage / 'PROTOCOL.json')
    cases = {c['pair_ordinal']: c for c in protocol['population']['cases']}
    arms = {arm['name']: {} for arm in plan['arms']}
    teacher = {name: {} for name in arms}
    for rank in range(3):
        directory = stage / f'evaluation/rank{rank}'
        done = read(directory / 'COMPLETE.json')
        assert done['independent_test_used'] is False and set(done['arms']) == set(arms)
        for name in arms:
            assert done['arms'][name]['model_and_frozen_Qwen_unchanged']
            raw = read(directory / name / 'TEACHER.json')
            for i, ordinal in enumerate(raw['ordinals']):
                assert ordinal not in teacher[name]
                teacher[name][ordinal] = {condition: values[i] for condition, values in raw['losses'].items()}
            for path in sorted((directory / name / 'scored').glob('*.json')):
                row = read(path); ordinal = row['native']['pair_ordinal']
                assert ordinal not in arms[name] and row['requested']['pair_ordinal'] == ordinal
                arms[name][ordinal] = row
    assert all(set(rows) == set(cases) == set(teacher[name]) for name, rows in arms.items())
    values = {name: {ordinal: metrics(row) for ordinal, row in rows.items()} for name, rows in arms.items()}
    keys = ['parse_success', *protocol['short_admission']['max_absolute_drop_vs_control']]
    groups = {'all': sorted(cases)}
    for field in ('operation', 'source_count', 'latent_bucket_frames', 'source_domain'):
        for value in sorted({str(c[field]) for c in cases.values()}):
            groups[f'{field}={value}'] = [o for o, c in cases.items() if str(c[field]) == value]
    reports = {}
    for group, ordinals in groups.items():
        summary = {'cases': len(ordinals), 'arms': {}, 'selected250_minus_control250': {}}
        for name in arms:
            summary['arms'][name] = {'metrics': {}, 'statuses': dict(Counter(arms[name][o]['native']['status'] for o in ordinals)),
                'extra_predicted_sources': sum(len(arms[name][o]['source_fields']['unassigned_predicted_sources']) for o in ordinals),
                'missing_or_failed_target_sources': sum(s['assignment'] != 'matched' for o in ordinals for s in arms[name][o]['source_fields']['sources']),
                'teacher_CE': {condition: float(np.mean([teacher[name][o][condition] for o in ordinals])) for condition in next(iter(teacher[name].values()))}}
            for key in keys:
                eligible = [o for o in ordinals if values[name][o].get(key, 0.) is not None]
                summary['arms'][name]['metrics'][key] = {'eligible_cases': len(eligible),
                    'mean': float(np.mean([values[name][o].get(key, 0.) for o in eligible])) if eligible else None}
        for key in keys:
            eligible = [o for o in ordinals if values['selected250'][o].get(key, 0.) is not None and values['control250'][o].get(key, 0.) is not None]
            if not eligible: continue
            delta = np.asarray([values['selected250'][o].get(key, 0.) - values['control250'][o].get(key, 0.) for o in eligible])
            item = {'point': float(delta.mean()), 'eligible_cases': len(eligible)}
            if group == 'all':
                rng = np.random.default_rng(20260910)
                boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
                item['paired_case_bootstrap99_descriptive'] = np.quantile(boot, [.005, .995]).tolist()
            summary['selected250_minus_control250'][key] = item
        reports[group] = summary
    all_rows = reports['all']; cfg = protocol['short_admission']
    end = all_rows['arms']['selected250']; start = all_rows['arms']['selected_start']; control = all_rows['arms']['control250']
    checks = {'parse_success': end['metrics']['parse_success']['mean'] >= cfg['parse_success_min'],
        'parse_regression': all_rows['selected250_minus_control250']['parse_success']['point'] >= -cfg['parse_success_max_drop_vs_control'],
        'CE_vs_selected_start': end['teacher_CE']['clean'] <= cfg['mean_clean_teacher_CE_max_ratio_to_selected_start'] * start['teacher_CE']['clean'],
        'CE_vs_control_end': end['teacher_CE']['clean'] <= cfg['mean_clean_teacher_CE_max_ratio_to_control_end'] * control['teacher_CE']['clean']}
    for key, limit in cfg['max_absolute_drop_vs_control'].items():
        checks[key] = all_rows['selected250_minus_control250'][key]['point'] >= -limit
    return {'at': datetime.now().astimezone().isoformat(), 'groups': reports, 'checks': checks,
        'short_screen_checks_passed': all(checks.values()), 'three_rank_native_resume_exact': True,
        'source_sha256': refs, 'full_AR_expansion_allowed': False, 'AR_quality_gate_passed': False,
        'independent_test_used': False, 'next': 'Review worst groups and preserved fields before issuing production admission.',
        'limitations': ['Reused100 validation cases, one training lineage; intervals are descriptive.',
            'Native content metrics normalize over target sources; extras are separately counted and penalized by source assignment/count checks.',
            'Lexical content/transcript scores and GT-assisted assignment do not establish final semantic or audio quality.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', type=Path, required=True)
    stage = parser.parse_args().stage
    report = review(stage); write(stage / 'EFFECT_REPORT.json', report)
    print(json.dumps({'short_screen_checks_passed': report['short_screen_checks_passed'], 'checks': report['checks']}))
