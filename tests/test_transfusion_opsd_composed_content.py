import pytest
from stable_audio_tools.training.transfusion_opsd.event_composed_content import (
    transcript_consensus, compare_content_evidence,
)


def test_reference_blind_majority_normalizes_case_and_punctuation():
    result=transcript_consensus(['The priest hesitated.', 'the priest hesitated', 'The priest hesitated it.'])
    assert result['status']=='agreed' and result['normalized_text']=='the priest hesitated'


def test_repeated_words_are_not_silently_removed():
    result=transcript_consensus(['The priest priest hesitated.', 'the priest priest hesitated', 'The priest hesitated.'])
    assert result['normalized_text']=='the priest priest hesitated'


def test_no_majority_means_no_selected_transcript():
    result=transcript_consensus(['one', 'two', 'three'])
    assert result['status']=='ambiguous' and result['normalized_text'] is None


def test_empty_consensus_is_not_a_correct_transcript_claim():
    result=transcript_consensus(['', '', 'noise'])
    assert result['status']=='agreed' and result['normalized_text']==''


def evidence(error, *, present=True, clap=.3):
    return dict(presence={'source_presence_failure':0 if present else 1},content_view={'admissible':True},
        clap_source_mean=clap,asr={'required':True,'requested_transcript_error':error})


def test_ambiguous_asr_cannot_appear_as_zero_wer():
    result=compare_content_evidence(evidence(None),evidence({'wer':0.}))
    assert not result['passed'] and result['wer_delta'] is None and result['uncertain']==['asr_consensus']


def test_absent_source_vetoes_even_higher_clap_and_correct_asr():
    result=compare_content_evidence(evidence({'wer':0.},present=False,clap=.5),evidence({'wer':0.}))
    assert not result['passed'] and result['failures']==['source_presence']


def test_transcript_scope_cannot_change_during_comparison():
    after=evidence(None);after['asr']['required']=False
    with pytest.raises(ValueError):compare_content_evidence(after,evidence({'wer':0.}))
