"""Keep complete checkpoints selected by actual nine-metric development results.

Selection is for candidate retention, not evidence of test-set improvement.
The caller runs on rank zero after saving and evaluating the same update.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path


METRIC_DIRECTIONS = {
    'Paired CLAP': 1, 'FD-CLAP': -1, 'FAD': -1, 'FD-PANN': -1,
    'KL': -1, 'LSD': -1, 'GCC': -1, 'CRW': -1, 'FSAD': -1,
}
RANKING = 'mean_signed_relative_improvement_percent'
GUARDED_RANKING = 'guardrails_then_median_relative_gain_v1'
OPERATIONS = ('event_addition', 'event_removal', 'static_to_linear',
              'linear_to_static', 'stationary_spatial_relocation')
SCALAR_METRICS = ('Paired CLAP', 'KL', 'LSD', 'GCC', 'CRW')
PRIORITY_METRICS = ('FAD', 'FD-PANN', 'KL', 'LSD', 'GCC', 'CRW', 'FSAD')


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_metric_score(metrics, baseline):
    """Equal metric weights after unit-free, direction-correct normalization."""
    changes = {}
    for name, direction in METRIC_DIRECTIONS.items():
        value, reference = float(metrics[name]), float(baseline[name])
        if not math.isfinite(value) or not math.isfinite(reference) or reference <= 0:
            raise ValueError('Require finite metrics and a positive fixed baseline: ' + name)
        changes[name] = 100. * direction * (value - reference) / reference
    return dict(score=sum(changes.values()) / len(changes),
                relative_improvement_percent=changes,
                improved_metrics=sum(change > 0 for change in changes.values()),
                priority_seven_improved_metrics=sum(changes[name] > 0 for name in PRIORITY_METRICS),
                priority_seven_all_improved=all(changes[name] > 0 for name in PRIORITY_METRICS),
                all_nine_improved=all(change > 0 for change in changes.values()),
                worst_relative_improvement_percent=min(changes.values()))


def _managed_path(folder, step):
    return folder / f'step-{step:08d}.pt'


def guarded_selection(score, diagnostics, policy):
    """Predeclared non-regression limits; unavailable evidence is not a pass."""
    overall = policy['maximum_overall_regression_percent']
    operation = policy['maximum_operation_regression_percent']
    if any(not math.isfinite(v) or v < 0 for v in (overall, operation)):
        raise ValueError('Regression tolerances must be finite and nonnegative.')
    if diagnostics is None:
        raise ValueError('Guarded selection requires per-operation validation evidence.')
    failures = [dict(scope='overall', metric=k, relative_gain_percent=v, limit=overall)
                for k,v in score['relative_improvement_percent'].items() if v < -overall]
    evidence = {}
    for name in OPERATIONS:
        group = diagnostics['groups'][name]
        if group['requests'] <= 0:
            raise ValueError('An operation has no validation coverage: '+name)
        evidence[name] = dict(requests=group['requests'], relative_gains={})
        for metric in SCALAR_METRICS:
            value = group['scalar_metrics'][metric]['relative_gain_percent']
            if value is None or not math.isfinite(value):
                raise ValueError('Missing finite per-operation metric.')
            evidence[name]['relative_gains'][metric] = value
            if value < -operation:
                failures.append(dict(scope=name, metric=metric, relative_gain_percent=value, limit=operation))
    return dict(eligible=not failures, guardrail_failures=failures, guardrail_evidence=evidence,
                median_relative_improvement_percent=statistics.median(score['relative_improvement_percent'].values()))


def _prune_retired_links(folder, ledger):
    # Only unlink paths recorded by this manager. Milestone/recovery paths
    # are separate directory entries, even when they share the same inode.
    retained = set(ledger['ranked_steps'])
    for item in ledger['history'].values():
        if item['step'] in retained or item.get('link_identity') is None:
            continue
        path = _managed_path(folder, item['step'])
        if not path.exists() and not path.is_symlink():
            continue
        stat = path.lstat()
        if path.is_symlink() or [stat.st_dev, stat.st_ino] != item['link_identity']:
            raise RuntimeError('Refusing to remove a replaced Top-K path: ' + str(path))
        path.unlink()


def retain_top_checkpoints(output, evaluation_path, config_path, policy, *, diagnostics=None):
    """Persist ranked hardlinks, their complete metrics and an eviction history.

    A crash after writing the index but before unlinking a retired candidate is
    harmless: the next call prunes only the already-recorded retired links.
    """
    output, evaluation_path, config_path = map(Path, (output, evaluation_path, config_path))
    keep = policy['keep']
    if type(keep) is not int or keep < 1 or policy['ranking'] not in (RANKING, GUARDED_RANKING):
        raise ValueError('Unsupported Top-K selection policy.')
    baseline_path = output / 'EVALUATION_step000000.json'
    baseline, evaluation = _read(baseline_path), _read(evaluation_path)
    config = _read(config_path)
    expected_requests = len(config['validation_ordinals'])
    expected_outputs = expected_requests * len(config['evaluation_seeds'])
    for result in (baseline, evaluation):
        if (result['requests'] != expected_requests or result['outputs'] != expected_outputs
                or result.get('target_audio_in_inference') is not False):
            raise ValueError('Top-K requires the same complete development evaluation panel.')
    if baseline['step'] != 0 or type(evaluation['step']) is not int or evaluation['step'] < 0:
        raise ValueError('Invalid evaluation update or baseline.')
    score = relative_metric_score(evaluation['metrics'], baseline['metrics'])
    folder = output / 'top_checkpoints'
    folder.mkdir(exist_ok=True)
    index = folder / 'INDEX.json'
    identity = dict(schema='editing_nine_metric_top_checkpoints_v1',
        config_sha256=_sha(config_path), policy=policy,
        baseline=dict(path=str(baseline_path), sha256=_sha(baseline_path), metrics=baseline['metrics']))
    if index.exists():
        ledger = _read(index)
        if any(ledger.get(key) != value for key, value in identity.items()):
            raise ValueError('Top-K configuration, baseline or ranking changed.')
        # These are reporting annotations on existing measurements, not a
        # change to the predeclared ranking or the evaluation panel.
        for item in ledger['history'].values():
            item.update(relative_metric_score(item['metrics'], baseline['metrics']))
        _prune_retired_links(folder, ledger)
    else:
        ledger = dict(identity, ranked_steps=[], history={})
    step = evaluation['step']
    if step == 0:
        # Original40k remains the baseline; it does not occupy a trained slot.
        _write(index, ledger)
        return ledger

    import torch
    recovery = output / 'resume_latest.pt'
    metadata = _read(output / 'RESUME.json')
    checkpoint = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
    if (checkpoint['step'] != step or metadata['step'] != step
            or checkpoint['model_sha256'] != metadata['model_sha256']
            or checkpoint['config_sha256'] != identity['config_sha256']):
        raise ValueError('Evaluation and recoverable checkpoint must identify the same update.')
    item = dict(step=step, metrics=evaluation['metrics'], **score,
        evaluation_path=str(evaluation_path), evaluation_sha256=_sha(evaluation_path),
        checkpoint_path=str(_managed_path(folder, step)), model_sha256=checkpoint['model_sha256'],
        logical_bytes=recovery.stat().st_size, link_identity=None)
    if policy['ranking'] == GUARDED_RANKING:
        item.update(guarded_selection(score, diagnostics, policy))
    del checkpoint
    old = ledger['history'].get(str(step))
    if old is not None:
        if any(old[key] != item[key] for key in ('model_sha256', 'evaluation_sha256', 'score', 'metrics')):
            raise ValueError('An already-ranked update has changed its model or evaluation.')
        item = old
    else:
        ledger['history'][str(step)] = item
    if policy['ranking'] == GUARDED_RANKING:
        ranked = sorted((row for row in ledger['history'].values() if row['eligible']),
            key=lambda row: (-row['median_relative_improvement_percent'], -row['score'], row['step']))[:keep]
    else:
        ranked = sorted(ledger['history'].values(),
                        key=lambda row: (-row['score'], -row['improved_metrics'], row['step']))[:keep]
    for row in ranked:
        path = _managed_path(folder, row['step'])
        if not path.exists():
            if row['step'] != step:
                raise RuntimeError('A previously retained Top-K checkpoint is missing: ' + str(path))
            # The recovery writer uses atomic replacement, never in-place
            # updates, so future saves cannot modify this retained checkpoint.
            os.link(recovery, path)
            stat = path.stat()
            row['link_identity'] = [stat.st_dev, stat.st_ino]
        else:
            stat = path.lstat()
            if path.is_symlink() or [stat.st_dev, stat.st_ino] != row['link_identity']:
                raise RuntimeError('Unexpected checkpoint at Top-K destination: ' + str(path))
    ledger['ranked_steps'] = [row['step'] for row in ranked]
    ledger['last_evaluated_step'] = max(map(int, ledger['history']))
    ledger['metric_goal_steps'] = {
        'priority_seven_all_improved': sorted(row['step'] for row in ledger['history'].values()
                                              if row['priority_seven_all_improved']),
        'all_nine_improved': sorted(row['step'] for row in ledger['history'].values() if row['all_nine_improved']),
    }
    _write(index, ledger)
    _prune_retired_links(folder, ledger)
    method = (f"Only candidates meeting all nine overall limits ({policy['maximum_overall_regression_percent']}%) "
        f"and all five operations' scalar limits ({policy['maximum_operation_regression_percent']}%) qualify. "
        'Ranked by median relative improvement, then mean. Fewer than five may qualify. '
        if policy['ranking'] == GUARDED_RANKING else
        'Ranked by the equal mean of nine relative improvements against this run’s step0. ')
    lines = ['# Top development checkpoints', '', method +
        'Positive is better. Ranking among candidates does not guarantee improvement over the baseline. '
        'Loss values and test-set results do not select these checkpoints.', '',
        '| Rank | Update | Mean relative gain | Improved metrics | Priority seven | Checkpoint |',
        '|---:|---:|---:|---:|---:|---|']
    for rank, row in enumerate(ranked, 1):
        lines.append(f"| {rank} | {row['step']} | {row['score']:+.4f}% | {row['improved_metrics']}/9 | "
                     f"{row['priority_seven_improved_metrics']}/7 | "
                     f"[full state]({row['checkpoint_path']}) |")
    lines += ['', 'Priority seven: FAD, FD-PANN, KL, LSD, GCC, CRW, FSAD. '
              'A positive average does not imply all seven or all nine improve.', '',
              'Evaluated updates improving all seven: ' + str(ledger['metric_goal_steps']['priority_seven_all_improved']),
              '', 'Evaluated updates improving all nine: ' + str(ledger['metric_goal_steps']['all_nine_improved']), '',
              '| Update | ' + ' | '.join(METRIC_DIRECTIONS) + ' |',
              '|---:|' + '---:|' * len(METRIC_DIRECTIONS),
              '| Baseline | ' + ' | '.join(f'{baseline["metrics"][name]:.6f}' for name in METRIC_DIRECTIONS) + ' |']
    for row in ranked:
        lines.append(f'| {row["step"]} | ' + ' | '.join(f'{row["metrics"][name]:.6f}' for name in METRIC_DIRECTIONS) + ' |')
    lines += ['', 'All metric values, evaluation bindings and retired candidate records are in INDEX.json. '
              'Periodic milestones and rolling recovery states are managed separately.']
    (folder / 'README.md').write_text('\n'.join(lines) + '\n')
    return ledger
