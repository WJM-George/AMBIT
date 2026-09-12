#!/usr/bin/env python3
"""Run a versioned English-request Generation AR bundle, optionally through P10."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing to replace a different committed artifact: {path}')
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def load_bundle(path):
    bundle = json.loads(path.read_text())
    if bundle.get('schema') != 'generation_ar_local_inference_bundle_v1':
        raise ValueError('Unsupported bundle schema')
    for name, entry in bundle['components'].items():
        artifact = Path(entry['path'])
        if sha(artifact) != entry['sha256']:
            raise ValueError(f'Bundle component changed: {name}: {artifact}')
    for entry in bundle['source_snapshots']:
        root = Path(entry['root'])
        manifest_path = root / entry['manifest']
        if sha(manifest_path) != entry['manifest_sha256']:
            raise ValueError(f'Source manifest changed: {manifest_path}')
        manifest = json.loads(manifest_path.read_text())
        files = manifest[entry['files_key']] if entry['files_key'] else manifest
        for name, expected in files.items():
            if sha(root / name) != expected:
                raise ValueError(f'Frozen source changed: {root / name}')
    return bundle


def main(args):
    bundle = load_bundle(args.bundle.resolve())
    if args.verify_only:
        print(json.dumps({'status': 'VERIFIED', 'bundle_id': bundle['bundle_id'],
                          'acceptance_status': bundle['acceptance_status'],
                          'gpu_used': False, 'data_content_rehashed': False}))
        return
    if args.output is None:
        raise ValueError('--output is required for generation')
    if (args.request is None) == (args.requests is None):
        raise ValueError('Supply exactly one of --request or --requests')
    inputs = ({'requests': [{'id': 'raw_request', 'request': args.request}]}
              if args.request is not None else json.loads(args.requests.read_text()))
    if set(inputs) != {'requests'} or not isinstance(inputs['requests'], list) or not inputs['requests']:
        raise ValueError('Expected a nonempty requests list')
    seen = set()
    for row in inputs['requests']:
        if (set(row) != {'id', 'request'} or any(not isinstance(row[key], str) or not row[key].strip()
                                               for key in ('id', 'request')) or row['id'] in seen):
            raise ValueError('Each input must contain only a unique id and nonempty raw request')
        seen.add(row['id'])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    request_file = args.output / 'requests.json'
    save(request_file, inputs)
    components = {name: entry['path'] for name, entry in bundle['components'].items()}
    contract = {'schema': 'generation_ar_bundle_run_v1', 'bundle_sha256': sha(args.bundle),
                'requests_sha256': sha(request_file), 'audio': args.audio,
                'seed': bundle['inference']['seed'],
                'input_policy': 'The exact raw English string is sent to the learned model. No request rewrite, template parser or scoring annotation.',
                'acceptance_status': bundle['acceptance_status']}
    save(args.output / 'RUN_CONTRACT.json', contract)
    command = [components['python']]
    if args.audio:
        destination = args.output / 'p10'
        command.extend([components['foa_entry'], '--raw-entry', components['raw_entry'],
                        '--normalizer', components['normalizer'], '--seed', str(bundle['inference']['seed'])])
        if 'raw_decoder' in components:
            command.extend(['--raw-decoder', components['raw_decoder'],
                            '--ar-batch-size', str(bundle['inference']['batch_size']),
                            '--ar-max-wall-seconds', str(bundle['inference']['max_wall_seconds']),
                            '--compact-progress'])
        if 'raw_gate_requests' in components:
            command.extend(['--raw-gate-requests', components['raw_gate_requests']])
        if args.verify_repeat:
            command.append('--verify-repeat')
    else:
        destination = args.output / 'ar'
        command.extend([components['raw_entry'], '--batch-size', str(bundle['inference']['batch_size']),
                        '--max-plan-tokens', str(bundle['inference']['max_plan_tokens']),
                        '--max-wall-seconds', str(bundle['inference']['max_wall_seconds'])])
        if 'raw_decoder' in components:
            command.extend(['--decoder', components['raw_decoder']])
        if 'raw_gate_requests' in components:
            command.extend(['--gate-requests', components['raw_gate_requests']])
    command.extend(['--snapshot', bundle['model_snapshot'], '--checkpoint', components['checkpoint'],
                    '--requests', str(request_file), '--output', str(destination)])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    with (args.output / 'inference.log').open('ab') as log:
        subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.audio:
        summary = json.loads((destination / 'SUMMARY.json').read_text())
        assert summary['status'] == 'COMPLETE'
        outputs = [{'id': row['id'], 'sceneplan': row['sceneplan'], 'foa': row['foa'],
                    'foa_sha256': row['foa_sha256']} for row in summary['rows']]
    else:
        assert json.loads((destination / 'STATUS.json').read_text())['status'] == 'COMPLETE'
        database = sqlite3.connect('file:' + str(destination / 'predictions.sqlite') + '?mode=ro&immutable=1', uri=True)
        try:
            predictions = {key: json.loads(value) for key, value in database.execute('SELECT id,payload FROM results')}
        finally:
            database.close()
        assert set(predictions) == seen
        plans = args.output / 'plans'
        plans.mkdir(exist_ok=True)
        outputs = []
        for index, row in enumerate(inputs['requests']):
            prediction = predictions[row['id']]
            assert prediction['request'] == row['request']
            if prediction['prediction'] is None:
                raise RuntimeError(f'AR failed to produce a valid plan for {row["id"]}: {prediction["error"]}')
            path = plans / (f'{index:04d}_' + hashlib.sha256(row['id'].encode()).hexdigest()[:12] + '.sceneplan.json')
            save(path, prediction['prediction'])
            outputs.append({'id': row['id'], 'sceneplan': str(path), 'sceneplan_sha256': sha(path)})
    result = {'status': 'COMPLETE', 'bundle_id': bundle['bundle_id'], 'rows': outputs,
              'acceptance_status': bundle['acceptance_status'],
              'audio_format': '44100 Hz FLOAT WAV, WYZX / ACN / SN3D' if args.audio else None}
    save(args.output / 'RESULT.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--request')
    inputs.add_argument('--requests', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--audio', action='store_true')
    parser.add_argument('--verify-repeat', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    main(parser.parse_args())
