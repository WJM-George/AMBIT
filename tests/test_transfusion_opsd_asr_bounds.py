import pytest

from stable_audio_tools.training.transfusion_opsd.event_asr_bounds import (
    transcript_error_bounds, compare_transcript_bounds, compare_bounded_content_evidence)

REFERENCE = 'The priest hesitated.'
EXTRA = 'The priest hesitated it.'


def bounds(texts):
    return transcript_error_bounds(texts, REFERENCE)


def test_a_majority_flip_is_uncertain_without_manufacturing_a_regression():
    before = bounds([REFERENCE, EXTRA, REFERENCE])
    after = bounds([EXTRA, EXTRA, REFERENCE])
    result = compare_transcript_bounds(after, before)
    assert result['status'] == 'uncertain'
    assert result['lower_delta'] == pytest.approx(-1/3)
    assert result['upper_delta'] == pytest.approx(1/3)
    assert not result['after_all_requested_words_supported']


def test_one_good_recognition_cannot_hide_a_bad_alternative():
    before = bounds([REFERENCE] * 3)
    after = bounds([REFERENCE, REFERENCE, EXTRA])
    assert after['lower_wer'] == 0
    assert compare_transcript_bounds(after, before)['status'] == 'uncertain'


def test_clear_repair_is_supported_while_unchanged_errors_are_not_correctness():
    correct, wrong = bounds([REFERENCE]*3), bounds([EXTRA]*3)
    assert compare_transcript_bounds(correct, wrong)['status'] == 'observed_nonregression'
    assert compare_transcript_bounds(wrong, correct)['status'] == 'observed_regression'
    unchanged = compare_transcript_bounds(wrong, wrong)
    assert unchanged['status'] == 'observed_nonregression'
    assert not unchanged['after_all_requested_words_supported']


def test_repeated_utterances_remain_real_errors():
    before = bounds([REFERENCE]*3)
    repeated = bounds([REFERENCE + ' ' + REFERENCE]*3)
    assert repeated['lower_wer'] == repeated['upper_wer'] == 1
    assert compare_transcript_bounds(repeated, before)['status'] == 'observed_regression'


@pytest.mark.parametrize('texts,reference', [([REFERENCE], REFERENCE), ([REFERENCE, None, EXTRA], REFERENCE), ([REFERENCE]*3, '')])
def test_missing_observations_or_reference_are_not_zero_error(texts, reference):
    with pytest.raises(ValueError):
        transcript_error_bounds(texts, reference)


def test_reference_cannot_change_between_comparisons():
    other = transcript_error_bounds(['hello']*3, 'hello')
    with pytest.raises(ValueError, match='different requested'):
        compare_transcript_bounds(other, bounds([REFERENCE]*3))


def test_uncertainty_does_not_pass_or_remove_semantic_and_presence_failures():
    def evidence(texts, point_wer, clap, presence):
        return dict(presence={'source_presence_failure':presence}, content_view={'admissible':True},
            clap_source_mean=clap, asr=dict(required=True, requested_transcript_error={'wer':point_wer},
                observed_error_bounds=bounds(texts)))
    before = evidence([REFERENCE, EXTRA, REFERENCE], 0., .2, 0.)
    after = evidence([EXTRA, EXTRA, REFERENCE], 1/3, .19, 1.)
    result = compare_bounded_content_evidence(after, before)
    assert set(result['failures']) == {'clap_similarity', 'source_presence'}
    assert result['uncertain'] == ['asr_observed_bounds']
    assert result['majority_wer_delta_diagnostic'] == pytest.approx(1/3)
    assert not result['passed']
