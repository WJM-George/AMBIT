#!/usr/bin/env python3
"""Audit token coefficients in a frozen natural-data training schedule, on CPU.

This does not estimate gradient importance from token counts. It identifies
what the uniform CE objective supervises and provides a reproducible input for
a subsequent measured-loss diagnostic. No model, validation or test is used.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import validate_requirements
from stable_audio_tools.data.sceneplan_generation_ar_supervision import annotate_target_tokens


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    training = json.loads(args.training.read_text())
    records = {r['id']: r for r in json.loads(args.request_first.read_text())['records']}
    codec = ModelScenePlanCodecV4(args.codec)
    cache = {}
    all_categories = set()
    for index, parent in enumerate(training['parents']):
        for view, tokens in enumerate(parent['target_token_ids_by_view']):
            precise = annotate_target_tokens(codec, tokens, fully_specified=True)
            if parent['route'] == 'request_to_plan':
                record = records[parent['id']]
                assert record['targets'][view]['tokens'] == tokens
                assert record['views'][view]['request'] == parent['natural_requests'][view]
                validate_requirements(parent['natural_requests'][view], record['requirements'][view])
                natural = annotate_target_tokens(
                    codec, tokens, fully_specified=False,
                    provenance=record['completion_provenance_by_view'][view],
                    source_bindings=record['source_bindings_by_view'][view],
                )
            elif parent['route'] == 'plan_to_request':
                # The frozen five_view renderer explicitly mentions all codec
                # fields except canonical gain, which is not serialized.
                natural = precise
            else:
                raise ValueError('Unknown training route')
            cache[index, view] = {
                'precise': [x.category for x in precise[1:]],
                'natural': [x.category for x in natural[1:]],
            }
            for values in cache[index, view].values():
                assert len(values) == len(tokens) - 1
                all_categories.update(values)
    categories = sorted(all_categories)
    category_ids = {name: i for i, name in enumerate(categories)}
    totals = Counter()
    by_mode = defaultdict(Counter)
    by_count = defaultdict(Counter)
    by_mode_count = defaultdict(Counter)
    presentations = Counter()
    uniform_coefficients = Counter()
    for batch in training['schedule']:
        denominator = sum(len(training['parents'][x['index']]['target_token_ids_by_view'][x['view']]) - 1 for x in batch)
        for item in batch:
            parent = training['parents'][item['index']]
            mode = item['mode']
            values = cache[item['index'], item['view']]['precise' if mode == 'precise' else 'natural']
            counts = Counter(values)
            totals.update(counts); by_mode[mode].update(counts)
            by_count[parent['source_count']].update(counts)
            by_mode_count[f"{mode}/{parent['source_count']}"].update(counts)
            presentations[mode] += 1
            for category, count in counts.items():
                uniform_coefficients[category] += count / denominator / len(training['schedule'])
    assert sum(totals.values()) == sum(sum(c.values()) for c in by_mode.values())
    assert abs(sum(uniform_coefficients.values()) - 1.) < 1e-9
    packed = []
    for (index, view), values in cache.items():
        packed.append({'index': index, 'view': view, **{k: [category_ids[c] for c in v] for k, v in values.items()}})
    packed_path = args.output / 'TOKEN_CATEGORIES.json'
    packed_path.write_text(json.dumps({'categories': categories, 'rows': packed}, separators=(',', ':')) + '\n')
    source = Path(sys.modules[annotate_target_tokens.__module__].__file__)
    result = {
        'status': 'COMPLETE', 'training_sha256': sha(args.training),
        'request_first_sha256': sha(args.request_first), 'codec_fingerprint': codec.fingerprint,
        'annotation_module_sha256': sha(source), 'script_sha256': sha(Path(__file__)),
        'annotations_sha256': sha(packed_path), 'parent_views': len(cache),
        'presentations': dict(presentations), 'target_token_occurrences': dict(totals),
        'by_mode': {k: dict(v) for k, v in by_mode.items()},
        'by_source_count': {str(k): dict(v) for k, v in by_count.items()},
        'by_mode_source_count': {k: dict(v) for k, v in by_mode_count.items()},
        'mean_per_update_uniform_ce_coefficients': dict(uniform_coefficients),
        'elapsed_s': time.monotonic() - started, 'gpu_used': False, 'test_used': False,
        'limits': [
            'Coefficient mass is not measured loss mass or gradient norm; it does not establish a cause of errors.',
            'An explicit numeric label has an evaluation tolerance; relative and free witness values are not unique correct answers.',
            'Core text is a valid supervised example, not a claim that semantic acceptance requires verbatim reproduction.',
            'All values still receive uniform CE in the frozen N2 run. This audit does not change that run or its data.',
        ],
    }
    (args.output / 'REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'parent_views', 'presentations', 'target_token_occurrences', 'mean_per_update_uniform_ce_coefficients', 'elapsed_s')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'request-first', 'codec', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    audit(p.parse_args())
