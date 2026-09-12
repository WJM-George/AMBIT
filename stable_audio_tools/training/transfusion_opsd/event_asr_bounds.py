"""Observed transcription error bounds, not statistical confidence intervals.

The three fixed recognition views remain unchanged. Their minimum and maximum
WER describe only the observed hypotheses. Overlap means uncertain; it never
becomes a successful content guard or a zero-error transcript.
"""
from .event_composed_content import ComposedEventContentObserver, compare_content_evidence, requested_transcript


def transcript_error_bounds(texts, reference):
    from scripts.t2a.eval.score_sceneplan_dit_p10_speech import _error_rates, _normalize_words
    if len(texts) != 3 or any(not isinstance(text, str) for text in texts):
        raise ValueError('Preserve all three fixed recognition outputs.')
    if not isinstance(reference, str) or not _normalize_words(reference):
        raise ValueError('Requested speech needs a nonempty word reference.')
    errors = [_error_rates(text, reference) for text in texts]
    minimum, maximum = min(x['wer'] for x in errors), max(x['wer'] for x in errors)
    return dict(reference_words=_normalize_words(reference), raw_texts=list(texts), errors=errors,
        lower_wer=minimum, upper_wer=maximum, all_requested_words_supported=maximum == 0.,
        scope='Range of observed WERs; not an acoustic guarantee or statistical confidence interval.')


def compare_transcript_bounds(after, before):
    if after['reference_words'] != before['reference_words']:
        raise ValueError('Cannot compare different requested transcripts.')
    # Every combination of recorded hypotheses must support the comparison.
    lower_delta = after['lower_wer'] - before['upper_wer']
    upper_delta = after['upper_wer'] - before['lower_wer']
    if upper_delta <= 1e-9:
        status = 'observed_nonregression'
    elif lower_delta > 1e-9:
        status = 'observed_regression'
    else:
        status = 'uncertain'
    return dict(status=status, lower_delta=lower_delta, upper_delta=upper_delta,
        after_all_requested_words_supported=after['all_requested_words_supported'],
        scope='Conservative comparison over these observed hypotheses only; equality is not transcript correctness.')


class BoundedASREventContentObserver(ComposedEventContentObserver):
    """Reuse presence, PCM16 CLAP and all three ASR views; add error bounds."""
    def __init__(self, scorer, **kwargs):
        super().__init__(scorer, **kwargs)
        original_receipt = self.receipt
        self.receipt = dict(contract='event_content_observed_asr_bounds_v1_experimental',
            observation=original_receipt,
            asr_comparison='All three fixed hypotheses determine WER bounds. Compare upper-after to lower-before for observed nonregression; lower-after above upper-before for regression; otherwise uncertain.',
            positive_teacher_speech='All observed transcripts must have WER zero for a requested-speech positive teacher.',
            scope='No new ASR model, crop, prompt, vote threshold or recognition call. Unknown remains unknown; not a validated complete content certificate.')

    def asr_evidence(self, canonical, requirements):
        evidence = super().asr_evidence(canonical, requirements)
        evidence['observed_error_bounds'] = (transcript_error_bounds(
            [row['text'] for row in evidence['decodes']], requested_transcript(requirements))
            if evidence['required'] else None)
        return evidence


def compare_bounded_content_evidence(after, before, *, clap_drop=.005):
    """Replace majority-WER comparison; all non-ASR protections stay intact."""
    result = compare_content_evidence(after, before, clap_drop=clap_drop)
    result['majority_wer_delta_diagnostic'] = result.pop('wer_delta')
    result['failures'] = [name for name in result['failures'] if name != 'asr_wer']
    result['uncertain'] = [name for name in result['uncertain'] if name != 'asr_consensus']
    result['asr_bounds_comparison'] = None
    if after['asr']['required']:
        evidence = compare_transcript_bounds(after['asr']['observed_error_bounds'], before['asr']['observed_error_bounds'])
        result['asr_bounds_comparison'] = evidence
        if evidence['status'] == 'observed_regression':
            result['failures'].append('asr_observed_bounds')
        elif evidence['status'] == 'uncertain':
            result['uncertain'].append('asr_observed_bounds')
    result['passed'] = not result['failures'] and not result['uncertain']
    return result
