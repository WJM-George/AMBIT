#!/usr/bin/env python3
"""CPU data-ready -> GPU0–2 gate -> full epoch, with compact job receipts."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from generation_ar_sampling import ShuffledGlobalBatchSampler
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (
    GenerationARSQLiteDataset, LengthBucketDistributedSampler)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n'); temp.replace(path)


def prepare(root, data):
    quality = json.loads((data / 'QUALITY_REPORT.json').read_text()); assert quality['status'] == 'PASS'
    columns_path = data / 'train_columns.npz'
    if not columns_path.exists():
        db = sqlite3.connect('file:' + str(data / 'train.sqlite') + '?mode=ro&immutable=1', uri=True)
        columns = np.asarray(db.execute('SELECT ordinal,target_token_count,source_count,raw_request_qwen_tokens FROM rows ORDER BY ordinal').fetchall(), dtype=np.int32)
        db.close(); assert columns.shape == (1600000, 4)
        assert np.array_equal(columns[:, 0], np.arange(1600000))
        np.savez(columns_path, lengths=columns[:, 1] - 1, source_counts=columns[:, 2], request_lengths=columns[:, 3])
    columns = np.load(columns_path)
    # Audit actual full-epoch membership and the synchronized uneven tail.
    seen = np.zeros(1600000, dtype=np.uint8); last = []; sizes = []
    for rank in range(3):
        base = LengthBucketDistributedSampler(columns['lengths'], num_replicas=3, rank=rank, batch_size=64)
        sampler = ShuffledGlobalBatchSampler(base); sampler.set_epoch(12)
        rows = np.fromiter(iter(sampler), dtype=np.int64)
        assert len(np.unique(rows)) == len(rows); assert not seen[rows].any()
        seen[rows] += 1; sizes.append(len(rows)); last.append(len(rows) % 64)
    assert seen.min() == seen.max() == 1 and last == [22, 21, 21]
    for split in ('train', 'validation'):
        dataset = GenerationARSQLiteDataset(data / (split + '.sqlite'), split=split)
        for i in [0, len(dataset) // 2, len(dataset) - 1]: assert dataset[i]['raw_user_request']
        dataset.close()
    atomic(root / 'TRAINING_DATA_GATE.json', {'status': 'PASS', 'sampler_epoch': 12, 'unique_epoch_rows': 1600000,
        'per_rank_rows': sizes, 'tail_per_rank': last, 'global_steps': 8334, 'row_repeats': 0,
        'train_columns_sha256': sha(columns_path), 'quality_report_sha256': sha(data / 'QUALITY_REPORT.json'),
        'loader_contract': 'New qualitative manifests accepted; old numeric manifests remain supported'})
    sources = json.loads((REPO / 'T100_TRAINING_SNAPSHOT.json').read_text())['files']
    config = {'schema': 'generation_ar_template100_training_v1', 'data': str(data), 'train_rows': 1600000,
        'quality_report_sha256': sha(data / 'QUALITY_REPORT.json'), 'source_sha256': sources,
        'initialize_checkpoint': os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/run/training/checkpoints/step_00002084.pt",
        'initialize_checkpoint_sha256': '9f7006171b64c061daa6cfda2bfef8621af4a19dacb6b624437d50f062b29bcb',
        'codec': os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4",
        'acceptance': str(REPO / 'docs/sceneplan_v2/generation_ar_template100_protocol_20260906.md'),
        'acceptance_sha256': sha(REPO / 'docs/sceneplan_v2/generation_ar_template100_protocol_20260906.md'),
        'hypothesis': 'Full GT coverage paired with consistent qualitative English teaches raw template execution without hidden numeric acceptance targets.',
        'steps': 8334, 'warmup_steps': 200, 'global_batch_size': 192, 'micro_batch_size': 32,
        'binding_strength': 0, 'sampler_epoch': 12, 'seed': 42, 'adapter_lr': 1e-5, 'lora_lr': 1e-4,
        'checkpoint_every': 1000, 'max_wall_seconds': 14400,
        'precision': 'FP32 AR math SDPA, BF16 frozen request encoder, TF32 off',
        'loss': 'Global nonpadding token cross entropy, unweighted',
        'success_rule': 'Request-derived validation metrics per source count; protocol thresholds, never exact hidden numeric completion'}
    path = root / 'TRAINING_CONFIG.json'
    if path.exists(): assert json.loads(path.read_text()) == config
    else: atomic(path, config)
    return path


def resource_status():
    raw = subprocess.check_output(['nvidia-smi', '--id=0,1,2', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [dict(zip(['index', 'memory_mb', 'utilization'], map(int, row.split(',')))) for row in raw.strip().splitlines()]


def run_stage(root, name, command, env, timeout):
    started = time.time()
    with (root / (name + '.log')).open('ab') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=REPO)
    atomic(root / (name + '_LAUNCH.json'), {'pid': child.pid, 'command': command, 'started_unix': started, 'wall_cap_s': timeout})
    atomic(root / 'STATUS.json', {'status': 'RUNNING', 'stage': name, 'child_pid': child.pid, 'started_unix': started, 'log': str(root / (name + '.log'))})
    try:
        next_resource = 0
        while child.poll() is None:
            if time.time() >= next_resource:
                resource = {'stage': name, 'updated_unix': time.time(), 'gpus': resource_status(), 'child_pid': child.pid}
                atomic(root / 'RESOURCE_STATUS.json', resource)
                with (root / 'resource_metrics.jsonl').open('a') as f: f.write(json.dumps(resource) + '\n')
                next_resource = time.time() + 60
            if time.time() - started > timeout: raise TimeoutError(name + ' exceeded its process wall cap')
            time.sleep(10)
        if child.returncode != 0: raise RuntimeError(f'{name} failed with exit {child.returncode}; see {root / (name + ".log")}')
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        raise
    atomic(root / (name + '_DONE.json'), {'status': 'COMPLETE', 'returncode': child.returncode, 'elapsed_s': time.time() - started})


def main(args):
    root = args.root.resolve(); data = root / 'data_v2'; root.mkdir(exist_ok=True)
    lock = (root / 'PIPELINE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time(); atomic(root / 'STATUS.json', {'status': 'WAITING_DATA_QUALITY', 'pid': os.getpid()})
    while not (data / 'QUALITY_REPORT.json').exists():
        if (data / 'STATUS.json').exists():
            s = json.loads((data / 'STATUS.json').read_text())
            if s['status'].startswith('FAILED'): raise RuntimeError('Data build failed: ' + str(s))
        if time.time() - started > 1800: raise TimeoutError('Data quality wait exceeded 30 minutes')
        time.sleep(5)
    config = prepare(root, data)
    for p in [data / 'train.sqlite', data / 'validation.sqlite', data / 'test.sqlite']: p.chmod(0o444)
    available = resource_status()
    assert all(g['memory_mb'] < 100 for g in available), 'GPU0–2 not free: ' + str(available)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0,1,2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NCCL_DEBUG='WARN')
    base = [args.python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=3',
        str(REPO / 'scripts/t2a/train/train_generation_ar_template100.py'), '--config', str(config)]
    proof = root / 'gpu_gate/GATE.json'
    if not proof.exists(): run_stage(root, 'gpu_gate', base + ['--run-dir', str(root / 'gpu_gate'), '--gate'], env, 1500)
    assert json.loads(proof.read_text())['status'] == 'PASS'
    command = base + ['--run-dir', str(root / 'training'), '--gate-proof', str(proof)]
    training_status = root / 'training/STATUS.json'
    already_complete = training_status.exists() and json.loads(training_status.read_text())['status'] == 'COMPLETE'
    if not already_complete:
        checkpoint = root / 'training/LATEST_CHECKPOINT.json'
        if checkpoint.exists():
            receipt = json.loads(checkpoint.read_text()); assert sha(receipt['path']) == receipt['sha256']
            command += ['--resume', receipt['path']]
        run_stage(root, 'training', command, env, 15300)
    atomic(root / 'READY_FOR_VALIDATION.json', {'status': 'TRAINING_COMPLETE',
        'checkpoint': str(root / 'training/checkpoints/step_00008334.pt'), 'test_used': False, 'goal_complete': False})
    # The evaluation recipe can be finalized while the immutable training job runs.
    recipe = root / 'VALIDATION_LAUNCH.json'
    if recipe.exists():
        evaluation = json.loads(recipe.read_text())
        assert evaluation['gpu_scope'] == [0, 1, 2] and evaluation['test_used'] is False
        assert sha(evaluation['entrypoint']) == evaluation['entrypoint_sha256']
        run_stage(root, 'validation', evaluation['command'], env, evaluation['wall_cap_s'])
    atomic(root / 'STATUS.json', {'status': 'TRAINING_COMPLETE_VALIDATION_PENDING' if not recipe.exists() else 'VALIDATION_STAGE_COMPLETE',
        'goal_complete': False, 'test_used': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', default=os.environ.get("AMBIT_PYTHON", "python3")); args = p.parse_args()
    try: main(args)
    except BaseException as exc:
        atomic(args.root / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
