#!/usr/bin/env python3
"""Freeze semantic pairs from completed AR prediction artifacts, after inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, help='label=/absolute/predictions.sqlite')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    pairs = {}; bindings = []; scenes = []
    for candidate in args.candidate:
        label, path = candidate.split('=', 1); count = 0
        db = sqlite3.connect('file:' + path + '?mode=ro&immutable=1', uri=True)
        for (payload,) in db.execute('SELECT payload FROM results ORDER BY panel_index'):
            v = json.loads(payload); count += 1
            predictions = {s['source_id']: s for s in v['prediction']['sources']} if v['prediction'] else {}
            for ref in v['target']['sources']:
                hyp = predictions.get(ref['source_id'])
                b = {'candidate': label, 'ordinal': v['ordinal'], 'source_count': v['source_count'], 'source_id': ref['source_id']}
                if ref['kind'] == 'speech':
                    words = lambda text: re.findall(r"[a-z0-9]+(?:['’][a-z0-9]+)*", text.lower().replace('’', "'"))
                    b['transcript_word_sequence_equal_auxiliary'] = bool(hyp and hyp['kind'] == 'speech' and words(ref['transcript']) == words(hyp['transcript']))
                if hyp is None or hyp['kind'] != ref['kind']:
                    b['reason'] = 'missing_or_wrong_kind'
                else:
                    field = 'speaker_description' if ref['kind'] == 'speech' else 'description'
                    if ' '.join(ref[field].split()) == ' '.join(hyp[field].split()):
                        b['reason'] = 'exact_text_up_to_whitespace'
                    else:
                        pair = {'kind': ref['kind'], 'reference': ref[field], 'candidate': hyp[field]}
                        pid = digest(pair); pairs[pid] = {'id': pid, **pair}; b['pair_id'] = pid
                bindings.append(b)
        db.close(); scenes.append({'candidate': label, 'prediction_db': str(Path(path).resolve()), 'rows': count})
    for name, value in [('pairs.json', {'pairs': [pairs[k] for k in sorted(pairs)], 'test_used': False}),
                        ('bindings.json', {'scenes': scenes, 'sources': bindings, 'test_used': False,
                          'scope': 'Exact persistent-source pairs for existing ordered precise-request protocol; natural unordered requests use the v2 evaluator.'})]:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'; path = args.output / name
        if path.exists(): assert path.read_text() == text
        else: path.write_text(text)
    print(json.dumps({'unique_pairs': len(pairs), 'source_bindings': len(bindings), 'candidates': scenes}))


if __name__ == '__main__': main()
