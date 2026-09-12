#!/usr/bin/env python3
"""Quiet status monitor: immediate failures/completion, 30-minute summaries."""
import argparse
import json
from pathlib import Path
import time


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(root, interval):
    old_health = json.loads((root / 'HEALTH.json').read_text()) if (root / 'HEALTH.json').exists() else {}
    deadline = old_health.get('next_routine_check_unix', time.time() + interval)
    previous_stage = None; last_alert = None; validation_reported = False; audio_reported = False; extra_reported = set()
    while True:
        paths = [root / 'STATUS.json', root / 'training/STATUS.json', root / 'validation_after_epoch/STATUS.json',
            *sorted((root / 'validation_after_epoch').glob('raw_gpu*/STATUS.json')),
            root / 'validation_after_epoch/baseline_raw/STATUS.json',
            root / 'validation_after_epoch/semantic_full/scoring/STATUS.json',
            root / 'validation_after_epoch/semantic_baseline/scoring/STATUS.json',
            root / 'p10_validation16/STATUS.json', root / 'p10_validation16/output/STATUS.json',
            root / 'validation_recovery/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/features/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/training/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/raw_validation/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/raw_validation_v2/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/semantic/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/semantic/scoring/STATUS.json',
            root.parent / 'learned_copy_pointer_20260906_v1/p10_validation16/STATUS.json']
        extra_path = root / 'ADDITIONAL_JOBS.json'
        extra_jobs = json.loads(extra_path.read_text())['jobs'] if extra_path.exists() else []
        paths.extend(Path(job['status']) for job in extra_jobs)
        statuses = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): json.loads(p.read_text()) for p in paths if p.exists()}
        pipeline = statuses.get('STATUS.json', {}); stage = pipeline.get('stage', pipeline.get('status', 'WAITING'))
        train = statuses.get('training/STATUS.json', {})
        failures = {k: v for k, v in statuses.items() if v.get('status', '').startswith(('FAILED', 'BUDGET_STOP'))}
        for job in extra_jobs:
            result_path = Path(job['result'])
            if result_path.exists():
                if job['name'] not in extra_reported:
                    print(json.dumps({'event': 'EXPERIMENT_RESULT_READY', 'name': job['name'], 'result': str(result_path)}), flush=True)
                    extra_reported.add(job['name'])
                continue
            launch_path = Path(job['launch'])
            if launch_path.exists():
                child = json.loads(launch_path.read_text()); proc = Path('/proc') / str(child['pid']) / 'cmdline'
                if not proc.exists() or str(child['command'][1]).encode() not in proc.read_bytes():
                    failures[job['name']] = {'status': 'JOB_PROCESS_MISSING', 'pid': child['pid']}
        terminal = pipeline.get('status') in ('VALIDATION_STAGE_COMPLETE', 'TRAINING_COMPLETE_VALIDATION_PENDING')
        launch = json.loads((root / 'PIPELINE_LAUNCH.json').read_text())
        proc = Path('/proc') / str(launch['pid']) / 'cmdline'
        expected_entry = str(launch['command'][1]).encode()
        if not terminal and (not proc.exists() or expected_entry not in proc.read_bytes()):
            failures['supervisor'] = {'status': 'SUPERVISOR_MISSING', 'pid': launch['pid']}
        recent = {'checked_unix': time.time(), 'stage': stage, 'training_step': train.get('step'),
            'training_steps': train.get('steps', 8334), 'next_routine_check_unix': deadline,
            'failures': failures, 'routine_interval_s': interval}
        if train.get('status') == 'RUNNING' and time.time() - train.get('updated_unix', time.time()) > 600:
            recent['stale_training_status_seconds'] = time.time() - train['updated_unix']
            failures['training_stale'] = {'status': 'POSSIBLE_STALL', 'seconds': recent['stale_training_status_seconds']}
        atomic(root / 'HEALTH.json', recent)
        if failures:
            signature = json.dumps(failures, sort_keys=True)
            if signature != last_alert:
                atomic(root / 'ATTENTION_REQUIRED.json', recent); print(json.dumps({'event': 'ATTENTION_REQUIRED', **recent}), flush=True); last_alert = signature
        elif last_alert is not None:
            atomic(root / 'ATTENTION_REQUIRED.json', {'status': 'RESOLVED', 'updated_unix': time.time()})
            print(json.dumps({'event': 'RECOVERED', 'stage': stage}), flush=True); last_alert = None
        if stage != previous_stage:
            print(json.dumps({'event': 'STAGE_CHANGED', 'stage': stage, 'training_step': train.get('step')}), flush=True); previous_stage = stage
        if not validation_reported and (root / 'validation_after_epoch/RESULT.json').exists():
            result = json.loads((root / 'validation_after_epoch/RESULT.json').read_text())
            print(json.dumps({'event': 'VALIDATION_RESULT_READY', 'status': result['status'], 'result': str(root / 'validation_after_epoch/RESULT.json')}), flush=True)
            validation_reported = True
        if not audio_reported and (root / 'p10_validation16/TECHNICAL_GATE.json').exists():
            print(json.dumps({'event': 'P10_TECHNICAL_RESULT_READY', 'result': str(root / 'p10_validation16/TECHNICAL_GATE.json')}), flush=True)
            audio_reported = True
        copy_root = root.parent / 'learned_copy_pointer_20260906_v1'
        copy_ready = not (copy_root / 'PROTOCOL.json').exists() or (copy_root / 'PILOT_RESULT.json').exists()
        if (copy_root / 'RAW_RECOVERY_V2_PROTOCOL.json').exists():
            copy_ready = copy_ready and (copy_root / 'raw_validation_v2/SUMMARY.json').exists()
        if validation_reported and audio_reported and copy_ready and all(Path(job['result']).exists() for job in extra_jobs):
            atomic(root / 'MONITOR_DONE.json', {'status': 'RESULTS_READY_FOR_REVIEW', 'goal_complete': False, 'updated_unix': time.time()}); return
        if time.time() >= deadline:
            print(json.dumps({'event': 'ROUTINE_CHECK', **recent, 'training_eta_s': train.get('eta_s')}), flush=True)
            deadline = time.time() + interval
        time.sleep(15)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True); p.add_argument('--interval', type=int, default=1800)
    args = p.parse_args(); assert args.interval in (1800, 3600); main(args.root, args.interval)
