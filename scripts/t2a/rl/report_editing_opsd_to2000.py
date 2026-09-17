"""Compare fixed500x2 outputs and rank immutable full-state checkpoints."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.t2a.rl.launch_editing_opsd_comparison import read, write
from scripts.t2a.rl.launch_editing_opsd_selective import sha
from stable_audio_tools.training.transfusion_opsd.top_checkpoints import (
    METRIC_DIRECTIONS, relative_metric_score, retain_top_checkpoints)


def load_evaluation(run, name):
    directory = Path(run) / 'evaluations' / name
    receipt = read(directory / 'INTERVENTION.json')
    result = read(directory / f'EVALUATION_step{receipt["step"]:06d}.json')
    configuration = Path(run) / 'config.json'
    q = read(configuration)
    if (receipt['config']['sha256'] != sha(configuration) or result['requests'] != 500
            or result['outputs'] != 1000 or result['target_audio_in_inference']
            or result.get('audio_retained') is not False):
        raise ValueError('An evaluation must use the complete fixed500x2 panel without stored WAVs.')
    rows = []
    for rank in range(4):
        rows.extend(read(directory / f'eval_step{receipt["step"]:06d}_rank{rank}.json')['rows'])
    expected = {(i, s) for i in q['validation_ordinals'] for s in q['evaluation_seeds']}
    actual = [(r['ordinal'], r['seed']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Missing, duplicated or mismatched validation samples.')
    return receipt, result, {(r['ordinal'], r['seed']):r for r in rows}


def matched_diagnostics(baseline, candidate, panel, seeds):
    ordinals = [r['pair_ordinal'] for r in panel['rows']]
    names = list(next(iter(baseline.values()))['scalar'])
    def array(records):
        return np.asarray([[np.mean([records[i, s]['scalar'][m] for s in seeds])
                            for m in names] for i in ordinals], dtype=np.float64)
    b, c = array(baseline), array(candidate)
    directions = np.asarray([METRIC_DIRECTIONS[m] for m in names])
    # Each unit is a source scene with its two paired noises. Do not count
    # the two noises as two independent requests.
    rng = np.random.default_rng(829217991)
    indices = rng.integers(0, len(ordinals), size=(1000, len(ordinals)))
    gains = 100*directions*(c[indices].mean(1)-b[indices].mean(1))/b[indices].mean(1)
    intervals = np.percentile(gains, [2.5, 97.5], axis=0)
    overall = {m:dict(relative_gain_percent=float(100*directions[j]*(c[:,j].mean()-b[:,j].mean())/b[:,j].mean()),
                      paired_bootstrap95_percent=intervals[:,j].tolist()) for j,m in enumerate(names)}
    groups = {}
    for field in ('operation', 'speech'):
        labels = [r['operation'] if field == 'operation' else
                  ('speech' if r['target_domain'] in ('speech_only','speech_mixed') else 'non_speech')
                  for r in panel['rows']]
        for label in sorted(set(labels)):
            mask = np.asarray([value == label for value in labels])
            groups[label] = dict(requests=int(mask.sum()), scalar_metrics={m:dict(
                baseline=float(b[mask,j].mean()), candidate=float(c[mask,j].mean()),
                relative_gain_percent=float(100*directions[j]*(c[mask,j].mean()-b[mask,j].mean())/b[mask,j].mean()))
                for j,m in enumerate(names)})
    return dict(paired_scalar_metrics=overall, groups=groups,
        uncertainty_scope='1000 paired source-scene bootstrap replicates over500 requests; each combines2 noises. '
        'CIs cover only the five per-output metrics, not distribution metrics FD/FAD/FSAD.')


def promote(run, name):
    run = Path(run)
    _, baseline, baseline_rows = load_evaluation(run, 'original40k')
    receipt, result, rows = load_evaluation(run, name)
    comparison = relative_metric_score(result['metrics'], baseline['metrics'])
    panel = read(run / 'VALIDATION500_PANEL.json')
    q = read(run / 'config.json')
    diagnostics = matched_diagnostics(baseline_rows, rows, panel, q['evaluation_seeds'])
    output = run / 'results'; output.mkdir(exist_ok=True)
    write(output / f'{name}.json', dict(name=name, step=result['step'], metrics=result['metrics'],
        comparison=comparison, diagnostics=diagnostics, checkpoint=receipt['updated_checkpoint'],
        target_audio_in_inference=False, scope='Fixed validation500x2, not a blind test.'))
    if name != 'candidate100':
        selection = run / 'selection'; selection.mkdir(exist_ok=True)
        reference = dict(baseline, step=0, source_evaluation=str(run/'evaluations/original40k'),
                         model_scope='original40k planner and executor; continuation update0 comparator')
        write(selection / 'EVALUATION_step000000.json', reference)
        step = result['step']
        checkpoint = Path(receipt['updated_checkpoint']['path'])
        if sha(checkpoint) != receipt['updated_checkpoint']['sha256']:
            raise ValueError('Evaluated checkpoint changed.')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if saved['step'] != step or saved['config_sha256'] != sha(run/'config.json'):
            raise ValueError('Top3 requires a checkpoint of the new continuation contract.')
        # This is a separate selection directory. Never change training's
        # rolling recovery while asynchronous evaluations complete.
        temp = selection / 'resume_link.tmp.pt'; temp.unlink(missing_ok=True)
        os.link(checkpoint, temp); temp.replace(selection / 'resume_latest.pt')
        write(selection/'RESUME.json', dict(step=step, model_sha256=saved['model_sha256']))
        path = selection / f'EVALUATION_step{step:06d}.json'; write(path, result)
        retain_top_checkpoints(selection, path, run/'config.json', q['top_checkpoint_policy'])
    reports = [read(p) for p in sorted(output.glob('candidate*.json'))]
    reports.sort(key=lambda r:r['step'])
    lines = ['# OPSD500→2000：固定500请求×2噪声', '',
        '原始40k作同面板基线。正数表示改善；综合排名不能替代逐项判断。音频不落盘，保留完整指标特征。', '',
        '| 步数 | 改善项数 | 九项均值 | 最差单项 | '+' | '.join(METRIC_DIRECTIONS)+' |',
        '|---:|---:|---:|---:|'+'---:|'*9]
    for report in reports:
        c = report['comparison']
        lines.append(f'| {report["step"]} | {c["improved_metrics"]}/9 | {c["score"]:+.3f}% | '
            f'{c["worst_relative_improvement_percent"]:+.3f}% | '+' | '.join(
                f'{c["relative_improvement_percent"][m]:+.3f}%' for m in METRIC_DIRECTIONS)+' |')
    lines += ['', 'results/*.json 包含五项逐样本指标的配对置信区间，以及分操作、speech/非speech结果。',
        '完整20k AR/RF/structured诊断保存在各评测目录 native_validation；这些损失不代替音频九指标。',
        'top3为已评测候选中的综合排名；1000/1500/2000和既有100/500另外永久保留。']
    (run/'TABLE.md').write_text('\n'.join(lines)+'\n')
