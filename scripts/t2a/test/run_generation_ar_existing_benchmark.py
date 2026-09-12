#!/usr/bin/env python3
"""Run only the new AR-to-P10 column for an already materialized benchmark."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def launch(command, folder, gpu, cap):
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    with (folder / 'launcher.log').open('ab') as log:
        p = subprocess.Popen(['timeout', '--kill-after=30s', str(cap), *command],
               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
               start_new_session=True)
    save(folder / 'LAUNCH.json', {'pid': p.pid, 'command': command, 'gpu': str(gpu),
         'started_unix': time.time(), 'wall_cap_s': cap})
    return p


def main(root):
    lock = (root / 'SUPERVISOR_LOCK').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(root / 'FREEZE.json')
    assert sha(__file__) == cfg['supervisor_sha256']
    for path, digest in cfg['files_sha256'].items():
        assert sha(path) == digest, path
    assert read(cfg['validation_result'])['status'] == 'PASS_FULL_VALIDATION_BEFORE_COMPLETION_AND_AUDIO'

    def status(stage, **extra):
        save(root / 'STATUS.json', {'status': stage, 'pid': os.getpid(), 'updated_unix': time.time(),
             'test_used': True, 'goal_complete': False, **extra})

    smoke = Path(cfg['bulk_validation_smoke'])
    status('WAITING_FOR_PRETEST_VALIDATION_TECHNICAL_GATES')
    while True:
        state = read(smoke / 'output/STATUS.json') if (smoke / 'output/STATUS.json').exists() else {}
        if str(state.get('status', '')).startswith('FAIL'):
            raise RuntimeError(f'Pretest bulk-entry gate failed: {state}')
        if state.get('status') == 'COMPLETE':
            break
        time.sleep(5)
    report = read(cfg['validation_boundary_report'])
    assert report['status'] == 'PASS' and report['passed'] == report['rows'] == 32000
    # Repeat invocation validates complete raw/FOA reuse with the new explicit
    # P10 seed map. This executes no new raw generation or audio synthesis.
    resumed = launch(read(smoke / 'LAUNCH.json')['command'], smoke / 'resume_check', 0, 300)
    if resumed.wait():
        raise RuntimeError('Pretest completed-entry resume failed')
    smoke_report = read(smoke / 'output/SUMMARY.json')
    assert smoke_report['status'] == 'COMPLETE' and smoke_report['reused_completed_rows'] == 4
    assert all(row['repeat_bit_exact'] for row in smoke_report['rows'])
    save(root / 'PRETEST_TECHNICAL_GATE.json', {'status': 'PASS', 'validation_rows': 4,
         'new_bulk_entry_native_FOA_repeat_exact': True, 'completed_AR_and_FOA_reused': 4,
         'full32000_P10_compiler_pass': True, 'test_inference_started': False})
    while True:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                '--format=csv,noheader,nounits'], text=True)
        scoped = {int(line.split(',')[0]): int(line.split(',')[1]) for line in used.splitlines()
                  if int(line.split(',')[0]) in (0, 1, 2)}
        if len(scoped) == 3 and max(scoped.values()) < 2048:
            break
        status('WAITING_FOR_ASSIGNED_GPU_RELEASE', memory_mib=scoped)
        time.sleep(15)
    processes = {}
    for gpu in (0, 1, 2):
        folder = root / f'gpu{gpu}'
        command = [cfg['python'], cfg['foa_entry'], '--snapshot', cfg['model_snapshot'],
                   '--checkpoint', cfg['checkpoint'], '--raw-entry', cfg['raw_entry'],
                   '--raw-decoder', cfg['raw_decoder'], '--raw-gate-requests', cfg['validation_gate_requests'],
                   '--normalizer', cfg['normalizer'], '--requests', str(root / 'inputs' / f'raw_gpu{gpu}.json'),
                   '--render-seeds', str(root / 'inputs' / f'p10_seeds_gpu{gpu}.json'),
                   '--output', str(folder / 'output'), '--ar-batch-size', '32',
                   '--ar-max-wall-seconds', str(cfg['budget']['raw_wall_s']),
                   '--max-render-wall-seconds', str(cfg['budget']['p10_wall_s']), '--compact-progress']
        processes[gpu] = launch(command, folder, gpu, cfg['budget']['process_wall_s'])
    save(root / 'TEST_INFERENCE_STARTED.json', {'started_unix': time.time(),
         'freeze_sha256': sha(root / 'FREEZE.json'), 'rows': 8000, 'gpu_scope': [0, 1, 2]})
    announced_raw = False
    while True:
        progress = {}
        raw_done = True
        for gpu, process in processes.items():
            folder = root / f'gpu{gpu}/output'
            state = read(folder / 'STATUS.json') if (folder / 'STATUS.json').exists() else {'status': 'STARTING'}
            progress[str(gpu)] = {k: state[k] for k in ('status', 'rows_done', 'rows') if k in state}
            raw_state = read(folder / 'ar/STATUS.json') if (folder / 'ar/STATUS.json').exists() else {}
            raw_done &= raw_state.get('status') == 'COMPLETE'
            if str(state.get('status', '')).startswith('FAIL') or process.poll() not in (None, 0):
                raise RuntimeError(f'Worker GPU{gpu} failed; preserve receipts and inspect {folder}: {state}')
        if raw_done and not announced_raw:
            save(root / 'RAW_ALL_COMPLETE.json', {'status': 'COMPLETE', 'rows': 8000, 'updated_unix': time.time(),
                 'shards': [str(root / f'gpu{g}/output/ar/predictions.sqlite') for g in (0, 1, 2)]})
            announced_raw = True
        status('RUNNING_NEW_AR_TO_P10_COLUMN', shards=progress)
        if all(p.poll() == 0 for p in processes.values()):
            break
        time.sleep(15)
    mapping = {row['id']: row for row in read(root / 'BENCHMARK_MAPPING.json')['rows']}
    outputs = {}
    for gpu in (0, 1, 2):
        folder = root / f'gpu{gpu}/output'
        summary = read(folder / 'SUMMARY.json')
        assert summary['status'] == 'COMPLETE'
        assert read(folder / 'ar/RAW_GPU_GATE.json')['status'] == 'PASS'
        for row in summary['rows']:
            sid = row['id']
            assert sid not in outputs and sid in mapping
            assert row['status'] == 'FOA_WRITTEN' and sha(row['foa']) == row['foa_sha256']
            assert row['seed'] == mapping[sid]['noise_seed']
            outputs[sid] = {**mapping[sid], 'new_ar_foa': row['foa'], 'new_ar_foa_sha256': row['foa_sha256'],
                 'new_ar_sceneplan': row['sceneplan'], 'model_num_samples': row['model_num_samples'],
                 'latent_frames': row['latent_frames'], 'render_elapsed_s': row['render_elapsed_s']}
    assert set(outputs) == set(mapping) and len(outputs) == 8000
    save(root / 'AUDIO_MANIFEST.json', {'schema': 'generation_ar_existing_benchmark_new_column_v1',
         'status': 'COMPLETE', 'rows': [outputs[sid] for sid in mapping], 'freeze_sha256': sha(root / 'FREEZE.json'),
         'existing_GT_P10_and_baselines_regenerated': False, 'test_used': True, 'goal_complete': False})
    status('NEW_AR_P10_AUDIO_COMPLETE_READY_FOR_EXISTING_BENCHMARK_METRICS', rows=8000)
    save(root / 'COMPLETE.json', {'status': 'COMPLETE', 'audio_manifest': str(root / 'AUDIO_MANIFEST.json'),
         'metrics_complete': False, 'goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root.resolve())
    except BaseException as exc:
        save(args.root / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
             'error': f'{type(exc).__name__}: {exc}', 'test_used': (args.root / 'TEST_INFERENCE_STARTED.json').exists()})
        raise
