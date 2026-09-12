#!/usr/bin/env python3
"""Reuse frozen context caches and add only request-owned qualitative labels."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(args):
    import torch
    torch.set_num_threads(4)
    module_path = Path(__file__).resolve().parents[3] / 'stable_audio_tools/models/sceneplan_generation_ar_qualitative_head.py'
    spec = importlib.util.spec_from_file_location('qualitative_label_module', module_path)
    head = importlib.util.module_from_spec(spec); spec.loader.exec_module(head)
    sys.path.insert(0, str(args.snapshot))
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import validate_requirements
    copy = args.root.parent / 'learned_copy_pointer_20260906_v1'; features = copy / 'features'
    base = args.root.parent / 'template100_rebuild_20260905_v1'; data = base / 'data_v2'
    output = args.root / 'features'; output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.root / 'PROTOCOL.json').read_text())
    assert sha(features / 'CONTRACT.json') == protocol['feature_contract_sha256']
    assert sha(data / 'QUALITY_REPORT.json') == json.loads((features / 'CONTRACT.json').read_text())['full_data_quality_report_sha256']
    atomic(output / 'STATUS.json', {'status': 'PREPARING_REQUEST_OWNED_LABELS', 'pid': os.getpid()})
    started = time.monotonic(); results = {}
    for split in ('train', 'validation'):
        source = data / (split + '.sqlite'); before = source.stat()
        db = sqlite3.connect('file:' + str(source) + '?mode=ro&immutable=1', uri=True)
        manifest = json.loads((features / split / 'MANIFEST.json').read_text()); rows = []
        for shard in manifest['shards']:
            assert sha(shard['path']) == shard['sha256']
            cached = torch.load(shard['path'], map_location='cpu', weights_only=False)
            assert [row['id'] for row in cached] == shard['ids']
            for row in cached:
                sid, raw, compressed = db.execute('SELECT sample_id,raw_user_request,request_requirements_zlib FROM rows WHERE ordinal=?', (row['ordinal'],)).fetchone()
                assert sid == row['id'] and raw == row['request']
                requirements = json.loads(zlib.decompress(compressed)); validate_requirements(raw, requirements)
                target_labels = head.annotation_targets(requirements)
                core = [(index, target) for index, target in enumerate(row['targets']) if target['field'] != 'transcript']
                assert len(core) == len(target_labels) == row['source_count']
                entries = []
                for (query_index, target), ref, labels in zip(core, requirements['sources'], target_labels):
                    assert target['text'] == ' '.join(ref['core'].split())
                    assert target['field'] == ('speaker_description' if ref['kind'] == 'speech' else 'description')
                    entries.append({'query_index': query_index, 'labels': labels, 'source_key': ref['key'],
                        'source_kind': ref['kind'], 'start': target['start'], 'end': target['end'], 'field_type': target['field_type']})
                rows.append({'id': sid, 'source_count': row['source_count'], 'entries': entries,
                    'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                    'request_requirements_sha256': hashlib.sha256(zlib.decompress(compressed)).hexdigest()})
        db.close(); after = source.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        assert len(rows) == {'train': 16384, 'validation': 1024}[split]
        path = output / (split + '.labels.json'); atomic(path, {'rows': rows})
        results[split] = {'rows': len(rows), 'source_fields': sum(len(r['entries']) for r in rows), 'labels': str(path),
            'labels_sha256': sha(path), 'context_manifest': str(features / split / 'MANIFEST.json'),
            'context_manifest_sha256': sha(features / split / 'MANIFEST.json'),
            'source_stat': {'path': str(source), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns}}
    contract = {'schema': 'generation_ar_request_owned_qualitative_features_v1', 'splits': results,
        'copy_feature_contract_sha256': sha(features / 'CONTRACT.json'), 'protocol_sha256': sha(args.root / 'PROTOCOL.json'),
        'head_module_sha256': sha(module_path), 'script_sha256': sha(Path(__file__)),
        'constraint_validation_sha256': sha(args.snapshot / 'stable_audio_tools/data/sceneplan_generation_ar_natural_constraints.py'),
        'hidden_witness_numbers_used': False, 'test_used': False, 'gpu_used': False, 'elapsed_s': time.monotonic() - started}
    atomic(output / 'CONTRACT.json', contract)
    atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'rows': {split: info['rows'] for split, info in results.items()}})
    print(json.dumps({'status': 'COMPLETE', 'elapsed_s': contract['elapsed_s'], 'rows': {split: info['rows'] for split, info in results.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    try: main(args)
    except BaseException as exc:
        atomic(args.root / 'features/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
