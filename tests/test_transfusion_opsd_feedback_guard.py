import pytest

from stable_audio_tools.training.transfusion_opsd.event_feedback_guard import (
    compare_feedback_guard, FEEDBACK_BRANCHES, PreupdateBranchReferences,
)
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore


def scores(similarity, asr=0.):
    return [RewardScore(.9, {'clap_content_cost/source_0': 1-similarity, 'asr_wer': asr})] * 2


def test_released_reference_cannot_hide_loss_from_the_actual_initial_policy():
    result = compare_feedback_guard(scores(.5), scores(.55), initial_policy=scores(.6),
        tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
    assert result['protected_reference'] and not result['protected_initial_policy']
    assert not result['protected']
    assert result['initial_policy_comparison']['utility_gain']['mean'] == pytest.approx(-.05)


def test_small_declared_fluctuations_remain_allowed_against_both_references():
    result = compare_feedback_guard(scores(.5), scores(.596), initial_policy=scores(.6),
        tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
    assert result['protected']


def test_better_semantics_cannot_buy_a_worse_transcript_than_initial_policy():
    result = compare_feedback_guard(scores(.5, .1), scores(.7, .05), initial_policy=scores(.6, 0.),
        tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
    assert result['protected_reference'] and not result['protected_initial_policy']
    assert not result['protected']


def test_guard_cannot_drop_a_noise_or_protection_dimension():
    with pytest.raises(ValueError, match='aligned noise pairs'):
        compare_feedback_guard(scores(.5), scores(.5)[:1], tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
    with pytest.raises(ValueError, match='coverage'):
        compare_feedback_guard(scores(.5), [RewardScore(.9, {'clap_content_cost/source_0': .5})]*2,
            tolerances={'asr_wer': 0.}, semantic_tolerance=.005)


def test_resumed_dit_gain_does_not_appear_as_a_loss_in_the_unchanged_ar_branch():
    references = PreupdateBranchReferences()
    references.capture('Anew_D0', 'request', [11, 12], scores(.5))
    references.capture('Anew_Dnew', 'request', [11, 12], scores(.7))
    correct = compare_feedback_guard(scores(.5), scores(.5),
        initial_policy=references.lookup('Anew_D0', 'request', [11, 12]),
        tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
    mixed = compare_feedback_guard(scores(.5), scores(.5),
        initial_policy=references.lookup('Anew_Dnew', 'request', [11, 12]),
        tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
    assert correct['protected'] and not mixed['protected']


@pytest.mark.parametrize('branch', FEEDBACK_BRANCHES)
def test_each_branch_keeps_its_own_preupdate_semantic_and_transcript_advantages(branch):
    references = PreupdateBranchReferences()
    references.capture(branch, 'request', [11, 12], scores(.7, 0.))
    before = references.lookup(branch, 'request', [11, 12])
    for after in (scores(.6, 0.), scores(.8, .1)):
        result = compare_feedback_guard(scores(.5, .2), after, initial_policy=before,
            tolerances={'asr_wer': 0.}, semantic_tolerance=.005)
        assert result['protected_reference'] and not result['protected_initial_policy']


def test_branch_capture_cannot_be_replaced_mutated_or_mispaired():
    references = PreupdateBranchReferences()
    original = scores(.6)
    references.capture('Anew_D0', 'request', [11, 12], original)
    original[0].costs['asr_wer'] = 1.
    assert references.lookup('Anew_D0', 'request', [11, 12])[0].costs['asr_wer'] == 0.
    with pytest.raises(ValueError, match='cannot be reset'):
        references.capture('Anew_D0', 'request', [11, 12], scores(.4))
    for branch, key, seeds in [('Anew_Dnew', 'request', [11, 12]),
            ('Anew_D0', 'other', [11, 12]), ('Anew_D0', 'request', [12, 11])]:
        with pytest.raises(ValueError, match='matching'):
            references.lookup(branch, key, seeds)
