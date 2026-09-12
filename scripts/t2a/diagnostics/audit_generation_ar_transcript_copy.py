#!/usr/bin/env python3
"""Optimistic transcript-only diagnostic of already generated raw plans.

Maximum exact matching ignores speaker and other fields, so this is an upper
bound on coupled request acceptance. Nearest-transcript WER is a lower bound,
not a substitute for one shared source assignment in the official evaluation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sqlite3
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def edits(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main(args):
    sys.path.insert(0, str(args.snapshot))
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import normalized_words
    pairs = json.loads(args.references.read_text())['pairs']
    db = sqlite3.connect('file:' + str(args.predictions) + '?mode=ro&immutable=1', uri=True)
    predictions = {k: json.loads(p) for k, p in db.execute('SELECT id,payload FROM results')}; db.close()
    assert set(predictions) == {r['id'] for r in pairs}
    by_count = {str(n): Counter() for n in range(1, 5)}; by_length = {}; examples = []
    for pair in pairs:
        refs = [(s['key'], c['value']) for s in pair['requirements']['sources'] for c in s['constraints'] if c['op'] == 'transcript']
        if not refs: continue
        generated = predictions[pair['id']]; assert generated['request'] == pair['request']
        pred = [s['transcript'] for s in (generated['prediction'] or {}).get('sources', []) if s['kind'] == 'speech']
        reference_words = [normalized_words(t) for _, t in refs]; candidate_words = [normalized_words(t) for t in pred]
        choices = list(range(len(pred))) + [None] * max(0, len(refs) - len(pred))
        exact = max(sum(j is not None and reference_words[i] == candidate_words[j] for i, j in enumerate(p))
            for p in set(itertools.permutations(choices, len(refs))))
        group = by_count[str(pair['source_count'])]
        group.update(speech_scenes=1, requested_speech=len(refs), predicted_speech=len(pred),
            exact_transcripts_upper_bound=exact, all_transcripts_exact_upper_bound=int(exact == len(refs)),
            speech_count_correct=int(len(refs) == len(pred)))
        for (key, text), words in zip(refs, reference_words):
            distance = [(edits(words, w), k) for k, w in enumerate(candidate_words)]
            error, nearest = min(distance, default=(len(words), None))
            bucket = '01-05' if len(words) <= 5 else '06-15' if len(words) <= 15 else '16-30' if len(words) <= 30 else '31+'
            by_length.setdefault(bucket, Counter()).update(transcripts=1, exact_any_candidate_upper_bound=int(error == 0),
                reference_words=len(words), nearest_word_edits_lower_bound=error)
            group.update(reference_words=len(words), nearest_word_edits_lower_bound=error)
            if error and len(examples) < 24:
                examples.append({'id': pair['id'], 'key': key, 'reference': text,
                    'nearest_generated': pred[nearest] if nearest is not None else None, 'word_edits': error, 'words': len(words)})
    for group in by_count.values():
        group['transcript_exact_upper_bound_rate'] = group['exact_transcripts_upper_bound'] / max(1, group['requested_speech'])
        group['nearest_wer_lower_bound'] = group['nearest_word_edits_lower_bound'] / max(1, group['reference_words'])
    for group in by_length.values():
        group['transcript_exact_upper_bound_rate'] = group['exact_any_candidate_upper_bound'] / group['transcripts']
        group['nearest_wer_lower_bound'] = group['nearest_word_edits_lower_bound'] / group['reference_words']
    result = {'status': 'COMPLETE_OPTIMISTIC_TRANSCRIPT_DIAGNOSTIC', 'by_source_count': by_count, 'by_reference_word_count': by_length,
        'examples': examples, 'prediction_sha256': sha(args.predictions), 'reference_sha256': sha(args.references), 'script_sha256': sha(Path(__file__)),
        'normalization_source_sha256': sha(args.snapshot / 'stable_audio_tools/data/sceneplan_generation_ar_natural_constraints.py'),
        'test_used': False, 'gpu_used': False, 'scope': __doc__}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'by_source_count': by_count, 'by_reference_word_count': by_length}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'references', 'predictions', 'output'): p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
