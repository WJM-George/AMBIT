"""Versioned lexical scoring of recorded speech hypotheses.

Boundary quotations are punctuation. Internal apostrophes remain part of a
word, so negation, omitted words and contractions are not silently merged.
Original ASR outputs and previous scoring are retained alongside the new view.
"""
from __future__ import annotations

from collections import Counter
import copy

from ...data.sceneplan_generation_ar_natural_constraints import normalized_words
from .native_clap_level_content import compare_level_canonical_native_content

WORD_RULE = 'english_alnum_internal_apostrophe_lexical_v2'
CONTENT_CONTRACT = 'level_canonical_native_foa_clap_content_lexical_v2'
LEGACY_CONTRACT = 'level_canonical_native_foa_clap_content_v1'


def _distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def lexical_error_rates(hypothesis, reference):
    if not isinstance(hypothesis, str) or not isinstance(reference, str):
        raise ValueError('Speech observations and the requested reference must be strings.')
    target, observed = normalized_words(reference), normalized_words(hypothesis)
    if not target:
        raise ValueError('Requested speech must contain lexical words.')
    target_chars, observed_chars = list(''.join(target)), list(''.join(observed))
    word_edits, char_edits = _distance(observed, target), _distance(observed_chars, target_chars)
    return dict(wer=word_edits / len(target), cer=char_edits / len(target_chars),
        word_edits=word_edits, reference_words=len(target), hypothesis_words=len(observed),
        char_edits=char_edits, reference_chars=len(target_chars))


def lexical_transcript_bounds(texts, reference):
    if len(texts) != 3 or any(not isinstance(text, str) for text in texts):
        raise ValueError('Keep exactly the three fixed Whisper hypotheses.')
    errors = [lexical_error_rates(text, reference) for text in texts]
    low, high = min(x['wer'] for x in errors), max(x['wer'] for x in errors)
    return dict(reference_words=normalized_words(reference), raw_texts=list(texts), errors=errors,
        lower_wer=low, upper_wer=high, all_requested_words_supported=high == 0., word_rule=WORD_RULE,
        scope='Bounds of these three recorded lexical WERs, not statistical or acoustic confidence bounds.')


def lexical_observation(record, requirements):
    """Return a new scoring view; never mutate the old record or regenerate ASR."""
    if record['content']['contract'] != LEGACY_CONTRACT:
        raise ValueError('Upgrade exactly the original level-canonical native observation once.')
    result = copy.deepcopy(record)
    content = result['content']
    result['legacy_word_observations'] = dict(asr=copy.deepcopy(content['asr']),
        independent_ctc=copy.deepcopy(result['independent_ctc']), content_contract=LEGACY_CONTRACT)
    content.update(contract=CONTENT_CONTRACT, word_rule=WORD_RULE)
    references = [c['value'] for source in requirements['sources'] for c in source['constraints'] if c['op'] == 'transcript']
    if bool(references) != content['asr']['required'] or len(references) > 1:
        raise ValueError('Require the actual sole requested speech transcript, when speech is present.')
    if not references:
        if result['independent_ctc']:
            raise ValueError('No speech means no CTC transcript observations.')
        return result
    reference = references[0]
    if len(result['independent_ctc']) != 2:
        raise ValueError('Keep both fixed independent CTC views.')
    asr = content['asr']
    texts = [x['text'] for x in asr['decodes']]
    asr['observed_error_bounds'] = lexical_transcript_bounds(texts, reference)
    tokens = [tuple(normalized_words(text)) for text in texts]
    counts = Counter(tokens)
    best = max(counts, key=counts.get)
    representative = tokens.index(best) if counts[best] >= 2 else None
    asr['consensus'] = dict(status='agreed' if representative is not None else 'uncertain',
        normalized_text=' '.join(best) if representative is not None else None,
        representative_index=representative, support_count=counts[best], raw_texts=list(texts),
        votes=[dict(text=' '.join(value), count=count) for value, count in counts.items()], word_rule=WORD_RULE)
    asr['requested_transcript_error'] = lexical_error_rates(texts[representative], reference) if representative is not None else None
    for row in result['independent_ctc']:
        row['error'] = lexical_error_rates(row['text'], reference)
        row['word_rule'] = WORD_RULE
    return result


def compare_lexical_native_content(after, before, *, maximum_semantic_drop):
    if any(x.get('contract') != CONTENT_CONTRACT or x.get('word_rule') != WORD_RULE for x in (after, before)):
        raise ValueError('Compare observations from the same explicit lexical scoring version.')
    # Only adapt local dictionary views for the unchanged native semantic and
    # three-view interval comparison. Saved records retain their true version.
    result = compare_level_canonical_native_content(
        dict(after, contract=LEGACY_CONTRACT), dict(before, contract=LEGACY_CONTRACT),
        maximum_semantic_drop=maximum_semantic_drop)
    return dict(result, contract='lexical_native_content_comparison_v2', word_rule=WORD_RULE)
