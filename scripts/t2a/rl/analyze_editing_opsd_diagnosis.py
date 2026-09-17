"""Read completed diagnostic outputs; never generate, train, or relabel metrics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

ARMS = ('paired_keep', 'ar_only', 'dit_only', 'joint')
METRICS = ('Paired CLAP', 'FD-CLAP', 'FAD', 'FD-PANN', 'KL', 'LSD', 'GCC', 'CRW', 'FSAD')
SCALARS = ('Paired CLAP', 'KL', 'LSD', 'GCC', 'CRW')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def improvement(value, reference, name):
    return 100 * (value - reference) / abs(reference) * (1 if name == 'Paired CLAP' else -1)


def key(row):
    return row['ordinal'], row['seed']


def quantile(values, q):
    values = sorted(values)
    where = (len(values) - 1) * q
    lo = int(where)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (where - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--wait-seconds', type=float, default=0.,
                        help='Optional bounded background postprocessing wait; never starts training.')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.wait_seconds:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            status = read(run / 'STATUS.json')
            if status['phase'] == 'COMPLETE':
                break
            if status['phase'] == 'FAILED':
                raise RuntimeError('Training failed; do not report partial metrics as complete.')
            if time.monotonic() >= deadline:
                raise TimeoutError('Postprocessing wait expired; no training process was changed.')
            time.sleep(10)
    result = dict(step=args.step, scope='20 development requests x 2 noises; not COMMON1000 or 5,000 evaluated requests',
        caveats=['Public retention arm is not pure paired SFT.',
                 'All arms train the complete AR, DiT and shared Transformer.',
                 'Per-operation statistics use scalar metrics; global Frechet metrics are not additive.',
                 'Post-update trajectories may differ by design; original data ordinals must match.',
                 'Four short updates do not establish long-run improvement.'], arms={})
    all_rows, inputs = {}, {}
    reference = read(run / ARMS[0] / 'EVALUATION_step000000.json')['metrics']
    result['original40k'] = reference
    for arm in ARMS:
        directory = run / arm
        q = read(run / (arm + '.json'))
        evaluation = read(directory / f'EVALUATION_step{args.step:06d}.json')
        rows = []
        for rank in range(len(q['physical_gpus'])):
            rows.extend(read(directory / f'eval_step{args.step:06d}_rank{rank}.json')['rows'])
        expected_keys = {(ordinal, seed) for ordinal in q['validation_ordinals'] for seed in q['evaluation_seeds']}
        if len(rows) != len(expected_keys) or {key(row) for row in rows} != expected_keys:
            raise ValueError('Missing or repeated evaluation row: ' + arm)
        if read(directory / 'EVALUATION_step000000.json')['metrics'] != reference:
            raise ValueError('Different initial metrics: ' + arm)
        all_rows[arm] = {key(row): row for row in rows}
        for name in SCALARS:
            if abs(statistics.mean(row['scalar'][name] for row in rows) - evaluation['metrics'][name]) > 1e-8:
                raise ValueError('Scalar mean disagrees with original report: ' + name)
        by_operation = defaultdict(list)
        for row in rows:
            by_operation[row['operation']].append(row)
        operation_metrics = {operation: dict(outputs=len(group), requests=len({r['ordinal'] for r in group}),
            metrics={name: statistics.mean(row['scalar'][name] for row in group) for name in SCALARS})
            for operation, group in by_operation.items()}
        updates = defaultdict(list)
        for rank in range(len(q['physical_gpus'])):
            path = directory / f'UPDATES_rank{rank}.jsonl'
            for line in path.read_text().splitlines():
                update = json.loads(line)
                if update['step'] <= args.step:
                    updates[update['step']].append((rank, update))
        if set(updates) != set(range(1, args.step + 1)) or any(len(v) != 2 for v in updates.values()):
            raise ValueError('Incomplete two-rank update history: ' + arm)
        coverage = defaultdict(Counter)
        for rank_updates in updates.values():
            for _, update in rank_updates:
                for request in update['request_updates']:
                    counts = coverage[request['requested_operation']]
                    counts['requests'] += 1
                    counts['content_binding_available'] += int(request['binding']['available'])
                    counts['direct_request_fields_available'] += int(bool(request['request_constraint_fields']))
                    counts['qualified_teacher_requests'] += int(request['enabled'])
                    for metric in request['execution_metrics']:
                        counts['proposed_terminals'] += 1
                        counts['qualified_terminals'] += int(metric['qualified_terminal'])
                        counts['qualified_with_unobservable_target_windows'] += int(
                            metric['qualified_terminal'] and metric['spatial'].get('unobservable', 0) > 0)
                        counts['qualified_without_observable_unchanged_windows'] += int(
                            metric['qualified_terminal'] and not metric['unchanged_windows']['available'])
                        counts['qualified_original_plan_terminals'] += int(
                            metric['qualified_terminal'] and metric['plan_index'] == 0)
        order = {}
        for step, rank_updates in updates.items():
            rank_updates.sort()
            order[step] = dict(requests=[x['ordinal'] for _, u in rank_updates for x in u['request_updates']],
                              paired=[x for _, u in rank_updates for x in u['paired_ordinals']])
        inputs[arm] = order
        relative = {name: improvement(evaluation['metrics'][name], reference[name], name) for name in METRICS}
        tail = {name: dict(p95=quantile([r['scalar'][name] for r in rows], .95),
                          maximum=max(r['scalar'][name] for r in rows),
                          worst=[dict(ordinal=r['ordinal'], seed=r['seed'], operation=r['operation'], value=r['scalar'][name],
                                      audio=r['audio']) for r in sorted(rows, key=lambda r: r['scalar'][name], reverse=True)[:3]])
                for name in ('GCC', 'CRW')}
        final = [u for _, u in updates[args.step]]
        result['arms'][arm] = dict(metrics=evaluation['metrics'], relative_percent=relative,
            improved_metrics=sum(value > 0 for value in relative.values()), per_operation=operation_metrics,
            tails=tail, training_evidence_coverage=coverage,
            qualified_request_count=sum(int(r['enabled']) for us in updates.values() for _, u in us for r in u['request_updates']),
            update_count=args.step, request_examples=16 * args.step, paired_examples=512 * args.step,
            compute=dict(training_update_wall_seconds=sum(max(u['performance']['step_seconds'] for _, u in us) for us in updates.values()),
                         cumulative_rank_costs={name: sum(u['costs'].get(name, 0) for u in final)
                                               for name in set().union(*(u['costs'] for u in final))}))
    if any(order != inputs[ARMS[0]] for order in inputs.values()):
        raise ValueError('Arms did not use identical request/paired order at every step.')
    result['matched_training_ordinals'] = True
    result['data_order_sha256'] = hashlib.sha256(json.dumps(inputs[ARMS[0]], sort_keys=True).encode()).hexdigest()
    control = all_rows['paired_keep']
    for arm, rows in all_rows.items():
        diffs = {name: [rows[k]['scalar'][name] - control[k]['scalar'][name] for k in control] for name in SCALARS}
        result['arms'][arm]['vs_public_retention'] = dict(
            different_plans=sum(rows[k]['plan'] != control[k]['plan'] for k in control) // len(q['evaluation_seeds']),
            scalar_changes={name: dict(mean_delta=statistics.mean(values),
                worse_outputs=sum(value < 0 if name == 'Paired CLAP' else value > 0 for value in values),
                paired_delta_p95=quantile(values, .95)) for name, values in diffs.items()})
    write(run / f'DIAGNOSIS_step{args.step:06d}.json', result)
    lines = [f'# Editing OPSD目标拆解：第{args.step}步', '',
             '同一原始40k起点；每组完整AR＋DiT共享训练，16请求＋512配对。20条开发请求×2噪声，实际40个输出。', '',
             '| 指标 | 原始40k | 公共保持 | ＋AR教师 | ＋DiT自终态RF | 两项联合 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in METRICS:
        values = [f"{result['arms'][arm]['metrics'][name]:.6f} ({result['arms'][arm]['relative_percent'][name]:+.2f}%)" for arm in ARMS]
        lines.append(f"| {name} | {reference[name]:.6f} | " + ' | '.join(values) + ' |')
    lines += ['', '括号为相对同面板原始40k的改善率，正值表示更好。公共保持不是纯配对SFT；所有组均保留共同请求／参照／配对空间监督。',
              '四组每步请求和配对样本顺序已核对一致。第一步之后由各自当前模型刷新反馈，故采样轨迹不再要求相同。',
              '逐操作均值、GCC／CRW尾部案例及成本见同名JSON；不得将40输出与COMMON1000绝对Fréchet值混比。', '']
    (run / f'DIAGNOSIS_step{args.step:06d}.md').write_text('\n'.join(lines))
    protocol = read(run / 'PROTOCOL.json')
    if protocol.get('paired_diagnostic'):
        previous = read(Path(protocol['paired_diagnostic']) / f'DIAGNOSIS_step{args.step:06d}.json')
        if previous['data_order_sha256'] != result['data_order_sha256']:
            raise ValueError('Update-scale comparison has different training data order.')
        transition = dict(changed=protocol['update_scale_revision'], step=args.step,
                          matched_training_order=True, arms={})
        for arm in ARMS:
            transition['arms'][arm] = dict(
                original40k_relative_percent=result['arms'][arm]['relative_percent'],
                versus_previous_scale_relative_percent={name: improvement(result['arms'][arm]['metrics'][name],
                    previous['arms'][arm]['metrics'][name], name) for name in METRICS})
        write(run / f'UPDATE_SCALE_COMPARISON_step{args.step:06d}.json', transition)
    status_path = run / 'STATUS.json'
    if status_path.exists():
        status = read(status_path)
        if status['phase'] == 'COMPLETE':
            write(run / 'COST.json', dict(
                elapsed_seconds=status['elapsed_seconds'],
                total_allocated_GPU_hours=status['elapsed_seconds'] * 8 / 3600,
                includes='Four parallel two-GPU arms; model loading, teacher construction, training, evaluation and checkpoint I/O.',
                kernel_busy_time_measured=False))
    print(json.dumps({arm: result['arms'][arm]['relative_percent'] for arm in ARMS}, ensure_ascii=False))


if __name__ == '__main__':
    main()
