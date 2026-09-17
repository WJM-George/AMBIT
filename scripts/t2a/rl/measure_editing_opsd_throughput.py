"""Summarize recorded GPU samples during completed continuation updates."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics


def summary(values):
    if not values:
        return None
    return dict(samples=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    gpu_samples = {gpu: [] for gpu in range(4, 8)}
    with (run / 'GPU_UTILIZATION.csv').open() as handle:
        for row in csv.reader(handle):
            if len(row) != 7:
                continue
            try:
                stamp = datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
                gpu = int(row[1])
                utilization, bandwidth, used, total, watts = map(float, row[2:])
            except ValueError:
                continue
            gpu_samples[gpu].append(dict(time=stamp, utilization=utilization,
                memory_used_MiB=used, memory_total_MiB=total, power_W=watts))
    ranks = []
    for rank in range(4):
        path = Path(protocol['training_output']) / f'UPDATES_rank{rank}.jsonl'
        updates = []
        with path.open() as handle:
            handle.seek(protocol['update_log_offsets'][str(rank)])
            for line in handle:
                if not line.endswith('\n'):
                    break
                value = json.loads(line)
                if 'started_unix' in value.get('performance', {}):
                    updates.append(value)
        windows = []
        for update in updates:
            p = update['performance']
            paired_start = p['started_unix'] + p['collection_seconds'] + p['self_distillation_seconds']
            windows.append(dict(start=p['started_unix'], end=p['finished_unix'],
                paired_start=paired_start, paired_end=paired_start + p['paired_update_seconds']))
        measured = [s for s in gpu_samples[rank + 4]
                    if any(w['start'] <= s['time'] <= w['end'] for w in windows)]
        paired = [s for s in measured
                  if any(w['paired_start'] <= s['time'] <= w['paired_end'] for w in windows)]
        def gpu_summary(samples):
            return dict(utilization_percent=summary([s['utilization'] for s in samples]),
                memory_used_MiB=summary([s['memory_used_MiB'] for s in samples]),
                power_W=summary([s['power_W'] for s in samples]),
                fraction_samples_at_least90_percent=(sum(s['utilization'] >= 90 for s in samples) / len(samples)
                                                    if samples else None))
        ranks.append(dict(rank=rank, physical_gpu=rank + 4, completed_steps=[u['step'] for u in updates],
            step_seconds=summary([u['performance']['step_seconds'] for u in updates]),
            collection_seconds=summary([u['performance']['collection_seconds'] for u in updates]),
            training_gpu=gpu_summary(measured), paired_phase_gpu=gpu_summary(paired),
            training_peak_allocated_MiB=summary([u['performance']['peak_allocated_MiB'] for u in updates]),
            training_peak_reserved_MiB=summary([u['performance']['peak_reserved_MiB'] for u in updates])))
    result = dict(scope='Only complete updates after the protected100-step recovery; initialization, calibration, saves and evaluation excluded. Paired phase boundaries derived from recorded timings.',
                  observed_local_time=datetime.now().isoformat(), ranks=ranks,
                  previous_baseline=json.loads((run / 'BEFORE_PERFORMANCE.json').read_text()))
    target = run / 'PERFORMANCE_OBSERVATION.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
