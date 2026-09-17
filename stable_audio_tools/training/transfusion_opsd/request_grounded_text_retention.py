"""Reinforce already-correct quoted fields at visited native plan prefixes.

The caller must establish that the request describes desired sources (the
current experiment uses additions). Removal/negation instructions need their
own binding parser. This is instruction supervision, not an audio teacher or
a claim that every reference field is correct. No paired target is consulted.
"""
from collections import Counter
import re

from .native_prefix_supervision import _sites
from .native_token_alignment import validate_native_plan_tokens


_QUOTED = re.compile(
    r'(?P<cue>voice described as|speech saying|sound described as|music described as)'
    r'\s*["“](?P<text>[^"”]+)["”]', re.IGNORECASE)
_FIELD = {'voice described as': 'speaker_description', 'speech saying': 'transcript',
          'sound described as': 'description', 'music described as': 'description'}


def normalized_text(value):
    return ' '.join(value.split()).casefold()


def request_quoted_text_targets(codec, native_tokens, native_plan, request, allowed_fn):
    """Return legal CE targets only for fully matching, unambiguous text fields.

    Matching is a conservative eligibility rule. It does not require future
    decoded text to remain byte-identical, label incorrect transcripts, or
    authorize recovery of hidden numeric fields.
    """
    ids = list(map(int, native_tokens))
    validate_native_plan_tokens(codec, ids, native_plan)
    _, sites = _sites(codec, ids)
    quotes = [dict(field=_FIELD[m['cue'].casefold()], text=m['text'],
                   request_span=[m.start('text'), m.end('text')]) for m in _QUOTED.finditer(request)]
    counts = Counter((field, normalized_text(s[field])) for s in native_plan['sources']
                     for field in ('speaker_description', 'description', 'transcript') if field in s)
    targets, fields = [], []
    for source in native_plan['sources']:
        for field in ('speaker_description', 'description', 'transcript'):
            if field not in source:
                continue
            text = source[field]
            matches = [q for q in quotes if q['field'] == field
                       and normalized_text(q['text']) == normalized_text(text)]
            if len(matches) != 1 or counts[(field, normalized_text(text))] != 1:
                continue
            key = (source['source_id'], '<' + field + '>')
            if key not in sites:
                raise ValueError('Quoted field is absent from the source-bound token sequence.')
            start = len(targets)
            for pos in sites[key]:
                allowed = sorted(allowed_fn(ids[:pos]))
                if ids[pos] not in allowed or len(allowed) != len(set(allowed)):
                    raise ValueError('Visited text choice must be a unique legal token.')
                if len(allowed) > 1:
                    targets.append(dict(position=pos, token_id=ids[pos], observed_token_id=ids[pos],
                                        allowed_ids=allowed, field='/'.join(key),
                                        role='request_entailed_correct_text_retention'))
            fields.append(dict(source_id=source['source_id'], field=field, text=text,
                               request_span=matches[0]['request_span'],
                               supervised_positions=len(targets)-start))
    return dict(targets=targets, fields=fields,
                scope='Correct quoted desired-source fields only; no correction or equality acceptance gate.')


def quoted_field_behavior(fields, after_plan):
    """Report text changes and explicit male/female contradictions separately.

    The narrow contradiction check concerns a requested textual voice label;
    it does not identify a person from audio or certify paraphrase equivalence.
    """
    sources = {s['source_id']: s for s in after_plan['sources']}
    result = []
    for item in fields:
        after = sources.get(item['source_id'], {}).get(item['field'])
        expected = set(re.findall(r'\b(?:male|female)\b', item['text'].casefold()))
        actual = set(re.findall(r'\b(?:male|female)\b', (after or '').casefold()))
        conflict = (item['field'] == 'speaker_description' and len(expected) == 1
                    and bool(actual - expected))
        result.append(dict(**item, after=after, missing=after is None,
                           text_unchanged=after is not None and normalized_text(after) == normalized_text(item['text']),
                           explicit_voice_label_conflict=conflict,
                           semantic_equivalence_certified=False))
    return result
