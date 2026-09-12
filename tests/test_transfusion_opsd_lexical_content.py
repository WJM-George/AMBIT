import copy

import pytest

from stable_audio_tools.training.transfusion_opsd.lexical_content_evidence import (
    WORD_RULE, lexical_error_rates, lexical_transcript_bounds, lexical_observation,
    compare_lexical_native_content,
)


@pytest.mark.parametrize('hypothesis', ["'Nothing,' said North.", '“Nothing,” said North.',
    "\"'Nothing,' said North.", 'NOTHING — said North!'])
def test_boundary_quotations_do_not_create_words(hypothesis):
    assert lexical_error_rates(hypothesis, 'Nothing, said North.')['word_edits'] == 0


@pytest.mark.parametrize(('hypothesis', 'reference'), [('I can go.', "I can't go."),
    ('I cant go.', "I can't go."), ('Leave now.', 'Do not leave now.'),
    ('Bring the blue cup.', 'Bring the red cup.'), ('Go go now.', 'Go now.')])
def test_real_lexical_changes_remain_errors(hypothesis, reference):
    assert lexical_error_rates(hypothesis, reference)['word_edits'] > 0


def test_internal_curly_apostrophe_is_a_straight_apostrophe_equivalent():
    assert lexical_error_rates('We can’t leave at six o’clock.', "We can't leave at six o'clock.")['wer'] == 0


def test_empty_hypothesis_and_repeated_insertions_keep_actual_error_counts():
    assert lexical_error_rates('', 'Leave now.')['wer'] == 1.
    assert lexical_error_rates('Go go go go.', 'Go.')['wer'] == 3.
    with pytest.raises(ValueError):
        lexical_error_rates('Anything', "' , \"")


def test_three_views_keep_disagreement_and_do_not_choose_best_asr():
    value = lexical_transcript_bounds(['Go now.', 'Go now.', 'Go later.'], 'Go now.')
    assert value['lower_wer'] == 0. and value['upper_wer'] == .5
    assert not value['all_requested_words_supported'] and value['word_rule'] == WORD_RULE
    with pytest.raises(ValueError):
        lexical_transcript_bounds(['Go now.'], 'Go now.')


def test_upgrade_preserves_legacy_hypotheses_without_mutating_input():
    texts = ["'Nothing,' said North."] * 3
    record = dict(content=dict(contract='level_canonical_native_foa_clap_content_v1',
        asr=dict(required=True, decodes=[dict(text=x) for x in texts], observed_error_bounds=dict(upper_wer=2/3))),
        independent_ctc=[dict(view=kind, text='NOTHING SAID NORTH', error=dict(wer=0.)) for kind in ['whole_w', 'canonical_event_w']])
    original = copy.deepcopy(record)
    requirements = dict(sources=[dict(constraints=[dict(op='transcript', value='Nothing, said North.')])])
    upgraded = lexical_observation(record, requirements)
    assert record == original
    assert upgraded['legacy_word_observations']['asr'] == original['content']['asr']
    assert upgraded['content']['asr']['observed_error_bounds']['all_requested_words_supported']
    assert upgraded['content']['asr']['decodes'] == original['content']['asr']['decodes']
    with pytest.raises(ValueError):
        lexical_observation(upgraded, requirements)


def test_legacy_and_new_content_cannot_be_silently_compared():
    with pytest.raises(ValueError):
        compare_lexical_native_content(dict(contract='level_canonical_native_foa_clap_content_v1'),
            dict(contract='level_canonical_native_foa_clap_content_lexical_v2', word_rule=WORD_RULE),
            maximum_semantic_drop=.04)
