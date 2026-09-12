#!/usr/bin/env python3
"""Start a frozen full-validation comparison after its prerequisite releases GPUs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(root):
    protocol = read(root / 'RAW_VALIDATION_PROTOCOL.json')
    scheduler = read(root / 'SCHEDULE_PROTOCOL.json')
    assert sha(Path(__file__)) == scheduler['script_sha256']
    assert sha(root / 'RAW_VALIDATION_PROTOCOL.json') == scheduler['raw_protocol_sha256']
    prerequisite = Path(scheduler['prerequisite_root'])
    atomic(root / 'STATUS.json', {'status': 'WAITING_FOR_CACHED_FULL_VALIDATION',
           'prerequisite': str(prerequisite), 'pid': os.getpid()})
    started = time.monotonic()
    while not (prerequisite / 'COLLECTOR_COMPLETE.json').exists():
        status = read(prerequisite / 'STATUS.json')
        if status.get('status') == 'FAILED_NEEDS_ATTENTION':
            raise RuntimeError(f'Prerequisite needs recovery: {status}')
        if time.monotonic() - started > scheduler['maximum_wait_s']:
            raise TimeoutError('Prerequisite exceeded deferred-launch waiting budget')
        time.sleep(15)
    for gpu in range(3):
        # Read-only check. An unexpected GPU user is never stopped here.
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True).splitlines()
        if int(lines[gpu].split(',')[1].strip()) > 2000:
            raise RuntimeError(f'GPU{gpu} is still occupied after the prerequisite; inspect before resuming')
    snapshot = Path(protocol['inference_snapshot'])
    assert sha(snapshot / 'MANIFEST.json') == protocol['inference_snapshot_manifest_sha256']
    for rel, expected in read(snapshot / 'MANIFEST.json')['files'].items():
        assert sha(snapshot / rel) == expected
    assert sha(protocol['checkpoint']) == protocol['checkpoint_sha256']
    atomic(root / 'STATUS.json', {'status': 'RAW_GENERATION_GPU0_1_2', 'pid': os.getpid()})

    def generate(gpu):
        folder = root / 'raw_evaluation' / f'raw_gpu{gpu}'
        folder.mkdir(parents=True, exist_ok=True)
        assert sha(protocol['raw_shards'][str(gpu)]) == protocol['raw_shards_sha256'][str(gpu)]
        command = [sys.executable, str(snapshot / 'scripts/t2a/inference/generate_sceneplan_with_learned_copy.py'),
                   '--snapshot', protocol['training_snapshot'], '--checkpoint', protocol['checkpoint'],
                   '--requests', protocol['raw_shards'][str(gpu)], '--output', str(folder),
                   '--gate-requests', scheduler['gate_requests'], '--batch-size', str(protocol['raw_batch_size']),
                   '--max-plan-tokens', str(protocol['max_plan_tokens']),
                   '--max-wall-seconds', str(protocol['budget']['raw_generation_wall_s_per_gpu']),
                   '--decoder', str(snapshot / 'stable_audio_tools/inference/sceneplan_generation_ar_decision_blocks.py')]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        cap = protocol['budget']['raw_process_wall_s_per_gpu']
        with (folder / 'driver.log').open('ab') as log:
            child = subprocess.Popen(['timeout', '--kill-after=20s', str(cap), *command],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=environment)
        atomic(folder / 'LAUNCH.json', {'pid': child.pid, 'command': command, 'gpu': [gpu],
               'started_unix': time.time(), 'wall_cap_s': cap})
        code = child.wait()
        if code:
            raise RuntimeError(f'GPU{gpu} raw worker exited {code}: {folder / "driver.log"}')

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(generate, gpu) for gpu in range(3)]
        for future in futures:
            future.result()
    collector = root / 'source_snapshot/collect.py'
    assert sha(collector) == read(root / 'COLLECTOR_PROTOCOL.json')['script_sha256']
    subprocess.run([sys.executable, str(collector), '--root', str(root)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root.resolve())
    except BaseException as exc:
        atomic(args.root / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
               'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
