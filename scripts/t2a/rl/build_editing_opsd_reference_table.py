"""Compare one completed 1000-example OPSD result to supplied AMBIT numbers.

This is a CPU-only reporting command. It never loads or evaluates AMBIT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METRICS = ['Paired CLAP', 'FD-CLAP', 'FAD', 'FD-PANN', 'KL', 'LSD', 'GCC', 'CRW', 'FSAD']


def build(result, reference, *, model_id=None):
    if result.get('phase') != 'COMPLETE' or result.get('rows') != 1000:
        raise ValueError('Use a completed 1000-example result, not the small development panel.')
    candidates = result['results']
    if model_id is not None:
        candidates = [r for r in candidates if r['id'] == model_id]
    if len(candidates) != 1 or 'opsd' not in candidates[0]['id'].lower():
        raise ValueError('Select exactly one OPSD result; AMBIT is a supplied reference only.')
    candidate = candidates[0]
    if candidate['rows'] != 1000:
        raise ValueError('The selected OPSD model must cover 1000 examples.')
    values = candidate['metrics']
    baseline = reference['metrics']
    for row in (baseline, values):
        if set(row) != set(METRICS) or any(not isinstance(row[k], (int, float)) or not math.isfinite(row[k]) for k in METRICS):
            raise ValueError('All nine finite metrics are required.')
    delta = {k: values[k] - baseline[k] for k in METRICS}
    gain = {k: (1 if k == 'Paired CLAP' else -1) * delta[k] / baseline[k] * 100 for k in METRICS}
    note = ('AMBIT采用用户提供的原报告固定数值；OPSD为COMMON1000测试结果。'
            '基线不重新生成或评分，也不标作本轮同1000条重新测得的对照。')
    report = dict(reference=reference, opsd=candidate, raw_delta=delta, improvement_percent=gain,
                  improved_metric_count=sum(v > 0 for v in gain.values()),
                  comparison_note=note, opsd_cohort=result.get('cohort'),
                  baseline_inference_performed=False)
    headers = [k + (' ↑' if k == 'Paired CLAP' else ' ↓') for k in METRICS]
    lines = ['# Editing OPSD：九项指标', '', note, '',
             '| System | ' + ' | '.join(headers) + ' |', '|---|' + '---:|' * len(METRICS),
             '| AMBIT (0.8B Ours，原报告) | ' + ' | '.join(f'{baseline[k]:.5f}' for k in METRICS) + ' |',
             '| ' + candidate['model'].replace('|', '/') + '（1000条） | ' + ' | '.join(f'{values[k]:.5f}' for k in METRICS) + ' |',
             '| 相对改善 | ' + ' | '.join(f'{gain[k]:+.2f}%' for k in METRICS) + ' |', '',
             f"改善 {report['improved_metric_count']}/9 项；相对改善的正数表示更好。"]
    coverage = candidate.get('scalar_coverage')
    if coverage:
        lines += ['', '标量指标有效覆盖数：' + '；'.join(f'{k}={n}/1000' for k, n in coverage.items()) + '。']
    return report, '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id')
    args = parser.parse_args()
    report, markdown = build(json.loads(args.result.read_text()), json.loads(args.reference.read_text()), model_id=args.model_id)
    report['inputs'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in [('result', args.result), ('reference', args.reference)]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'COMPARISON.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'TABLE.md').write_text(markdown)
    with (args.output_dir / 'TABLE.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['System', *METRICS])
        writer.writeheader()
        writer.writerow({'System': 'AMBIT (0.8B Ours, reported)', **report['reference']['metrics']})
        writer.writerow({'System': report['opsd']['model'], **report['opsd']['metrics']})
    print(json.dumps({'table': str(args.output_dir / 'TABLE.md'), 'improved_metrics': report['improved_metric_count'],
                      'baseline_inference_performed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
