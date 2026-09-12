#!/usr/bin/env python3
"""Add training-only evidence-region targets to the unchanged cached examples."""
import argparse
import copy
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
    source_root = Path(__file__).resolve().parents[3]
    path = source_root / 'stable_audio_tools/data/sceneplan_generation_ar_template_evidence.py'
    spec = importlib.util.spec_from_file_location('training_template_evidence', path); trace = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace)
    base = args.base_root or args.root.parent / 'template100_rebuild_20260905_v1'
    sys.path.insert(0, str(base / 'evaluation_source_snapshot'))
    from stable_audio_tools.data.sceneplan_generation_template100 import RECIPES
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import validate_requirements
    parent = args.parent_features or args.root.parent / 'learned_qualitative_execution_20260906_v1/features'
    contract = json.loads((parent / 'CONTRACT.json').read_text()); output = args.root / 'features'
    atomic(output / 'STATUS.json', {'status': 'TRACING_REQUEST_OWNED_FIELD_EVIDENCE', 'pid': os.getpid()})
    started = time.monotonic(); splits = {}
    for split, feature in contract['splits'].items():
        assert sha(feature['labels']) == feature['labels_sha256']
        labels = {row['id']: row for row in json.loads(Path(feature['labels']).read_text())['rows']}
        db = sqlite3.connect('file:' + str(base / 'data_v2' / (split + '.sqlite')) + '?mode=ro&immutable=1', uri=True)
        result = []
        for shard in json.loads(Path(feature['context_manifest']).read_text())['shards']:
            assert sha(shard['path']) == shard['sha256']
            records = torch.load(shard['path'], map_location='cpu', weights_only=False, mmap=True)
            assert [row['id'] for row in records] == shard['ids']
            for row in records:
                raw, recipe_id, compressed = db.execute('SELECT raw_user_request,template_id,request_requirements_zlib FROM rows WHERE sample_id=?', (row['id'],)).fetchone()
                assert raw == row['request'] and recipe_id == row['template']
                requirements = json.loads(zlib.decompress(compressed)); validate_requirements(raw, requirements)
                spans = trace.template_evidence_spans(raw, requirements, RECIPES[recipe_id]); owned = copy.deepcopy(labels[row['id']])
                assert len(spans) == len(owned['entries']) == owned['source_count']
                for entry, spans_, source in zip(owned['entries'], spans, requirements['sources']):
                    assert entry['source_key'] == spans_['source_key']
                    entry['evidence_chars'] = {}; entry['evidence_tokens'] = {}
                    focused = trace.focused_motion_evidence(raw, source, spans_) if args.focused_motion else None
                    if args.include_event_spans:
                        low, high = spans_['block']
                        covering = [index for index, (start, end) in enumerate(row['offsets'])
                                    if start < end and start <= high and end > low]
                        assert covering
                        entry['event_chars'] = [low, high]
                        entry['event_tokens'] = [min(covering), max(covering)]
                    for name in entry['labels']:
                        low, high = focused[name] if focused is not None else spans_['activity' if name in ('onset', 'offset') else 'motion']
                        covering = [index for index, (start, end) in enumerate(row['offsets']) if start < end and start <= high and end > low]
                        assert covering
                        entry['evidence_chars'][name] = [low, high]
                        entry['evidence_tokens'][name] = [min(covering), max(covering)]
                result.append(owned)
        db.close(); assert len(result) == feature['rows']
        label_path = output / (split + '.labels.json'); atomic(label_path, {'rows': result})
        splits[split] = {**feature, 'labels': str(label_path), 'labels_sha256': sha(label_path),
            'parent_label_sha256': feature['labels_sha256']}
    result = {**contract, 'splits': splits, 'parent_feature_contract_sha256': sha(parent / 'CONTRACT.json'),
        'evidence_tracer_sha256': sha(path), 'evidence_preparer_sha256': sha(Path(__file__)),
        'evidence_scope': 'Training-only template producer trace disambiguates source-owned activity/motion regions. These regions enter an auxiliary training loss only, never model.forward or raw inference.',
        'focused_motion_evidence': args.focused_motion,
        'event_spans_included': args.include_event_spans,
        'source_data_changed': False, 'elapsed_evidence_preparation_s': time.monotonic() - started}
    atomic(output / 'CONTRACT.json', result); atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'rows': {key: value['rows'] for key, value in splits.items()}})
    print(json.dumps({'status': 'COMPLETE', 'elapsed_s': result['elapsed_evidence_preparation_s']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base-root', type=Path); parser.add_argument('--parent-features', type=Path)
    parser.add_argument('--focused-motion', action='store_true')
    parser.add_argument('--include-event-spans', action='store_true'); args = parser.parse_args()
    try: main(args)
    except BaseException as exc:
        atomic(args.root / 'features/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
