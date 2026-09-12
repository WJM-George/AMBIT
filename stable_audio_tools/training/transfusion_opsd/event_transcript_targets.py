"""Requested-word correctness and eligibility for a speech retention target.

Relative nonregression can retain a consistently wrong utterance. This check
keeps absolute requested-word support separate. It uses the existing three
recognition views and does not certify voice, semantics, space, or perception.
"""
from __future__ import annotations

from .event_asr_bounds import transcript_error_bounds
from .event_composed_content import requested_transcript


def requested_transcript_status(content, requirements):
    """Recompute support from all observed strings, without best-view selection."""
    reference = requested_transcript(requirements)
    asr = content.get('asr', {})
    if reference is None:
        if asr.get('required'):
            raise ValueError('Recognition evidence and requested transcript scope differ.')
        return dict(status='not_requested', absolute_word_check_passed=True,
            eligible_as_word_retention_reference=True, bounds=None,
            scope='No transcript was requested; other content checks remain necessary.')
    if not asr.get('required'):
        raise ValueError('Requested speech is missing its recognition evidence.')
    texts = [row['text'] for row in asr.get('decodes', [])]
    bounds = transcript_error_bounds(texts, reference)
    stored = asr.get('observed_error_bounds')
    if stored is not None and any(stored.get(key) != bounds[key]
            for key in ('reference_words', 'raw_texts', 'lower_wer', 'upper_wer')):
        raise ValueError('Stored recognition bounds do not match the observed strings and request.')
    if bounds['upper_wer'] == 0:
        status = 'all_three_observed_correct'
    elif bounds['lower_wer'] > 0:
        status = 'all_three_observed_incorrect'
    else:
        status = 'mixed_word_support'
    passed = status == 'all_three_observed_correct'
    return dict(status=status, absolute_word_check_passed=passed,
        eligible_as_word_retention_reference=passed, bounds=bounds,
        scope='Observed support for the requested words only. Agreement is not an acoustic guarantee; an uncertain or observed-wrong reference cannot be labeled a correct-word retention target.')
