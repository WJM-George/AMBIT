import pytest

from stable_audio_tools.training.transfusion_opsd.event_asr_bounds import compare_transcript_bounds
from stable_audio_tools.training.transfusion_opsd.event_transcript_targets import requested_transcript_status


REQUIREMENTS = {'sources': [{'constraints': [{'op': 'transcript', 'value': 'The priest hesitated.'}]}]}


def content(texts):
    return {'asr': {'required': True, 'decodes': [{'text': text} for text in texts]}}


def test_consistently_wrong_retention_is_not_mistaken_for_correct_requested_words():
    result = requested_transcript_status(content(['The priesthood hesitated.']*3), REQUIREMENTS)
    assert compare_transcript_bounds(result['bounds'], result['bounds'])['status'] == 'observed_nonregression'
    assert result['status'] == 'all_three_observed_incorrect'
    assert not result['absolute_word_check_passed'] and not result['eligible_as_word_retention_reference']


def test_all_views_must_support_requested_words_without_requiring_written_punctuation():
    result = requested_transcript_status(content(['The priest hesitated.', 'the priest hesitated', 'The priest hesitated!']), REQUIREMENTS)
    assert result['absolute_word_check_passed'] and result['eligible_as_word_retention_reference']


def test_an_ambiguous_reference_fails_even_its_identity_comparison_and_cannot_be_a_correct_target():
    result = requested_transcript_status(content(['The priest hesitated.', 'The priesthood hesitated.', 'The priest hesitated.']), REQUIREMENTS)
    assert result['status'] == 'mixed_word_support'
    assert not result['eligible_as_word_retention_reference']
    assert compare_transcript_bounds(result['bounds'], result['bounds'])['status'] == 'uncertain'


def test_cached_success_does_not_override_actual_observed_word_errors():
    correct = requested_transcript_status(content(['The priest hesitated.']*3), REQUIREMENTS)
    wrong = content(['The priesthood hesitated.']*3)
    wrong['asr']['observed_error_bounds'] = correct['bounds']
    with pytest.raises(ValueError, match='Stored recognition bounds'):
        requested_transcript_status(wrong, REQUIREMENTS)


@pytest.mark.parametrize('evidence', [{}, {'asr': {'required': False}}, content(['The priest hesitated.'])])
def test_missing_recognition_is_not_treated_as_zero_error(evidence):
    with pytest.raises(ValueError): requested_transcript_status(evidence, REQUIREMENTS)


def test_requests_without_literal_speech_do_not_acquire_an_invented_transcript_target():
    result = requested_transcript_status({'asr': {'required': False}}, {'sources': [{'constraints': []}]})
    assert result['status'] == 'not_requested' and result['absolute_word_check_passed']
    assert result['bounds'] is None
