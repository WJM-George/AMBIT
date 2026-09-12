import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_common_repair_task import EventCommonRepairTask
from stable_audio_tools.training.transfusion_opsd.event_task_quality import select_query_teacher
from stable_audio_tools.training.transfusion_opsd.joint_repair import select_joint_repair_teacher
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore
from stable_audio_tools.training.transfusion_opsd.joint_repair_distillation import repair_teacher_distribution
from stable_audio_tools.training.transfusion_opsd.common_teacher_mixture import positive_context_mixture


TOLERANCES = {'requested_sector_failure': .01, 'asr_wer': 0.,
              'direction_unobservable_fraction': .01}


def task(kind='requested_sector_v1'):
    return EventCommonRepairTask(kind=kind, tolerances=TOLERANCES,
        minimum_gain=.01, utility_tolerance=.01 if kind == 'requested_sector_v1' else .005,
        content_tolerance=.005 if kind == 'requested_sector_v1' else None)


def score(sector=.2, content=.3, wer=0., unknown=0.):
    return RewardScore(.99, {'requested_sector_failure': sector, 'asr_wer': wer,
        'direction_unobservable_fraction': unknown, 'clap_content_cost/source_0': 1. - content})


def variant(profile, raw, fraction=0.):
    return dict(score=profile.quality(raw), repair_fraction=fraction, certified=True)


def test_direction_gain_does_not_require_an_unrelated_content_gain():
    values = [score(), score(.1, .298)]
    assert not select_query_teacher(values, legal=[True, True],
        tolerances=TOLERANCES)['qualified']
    profile = task()
    result = profile.select([[variant(profile, raw)] for raw in values],
        legal=[True, True], repair_penalty=.01)
    assert result['qualified'] and result['action'] == 1
    assert result['improving_plans'][0]['utility_gain'] == pytest.approx(.1)
    assert 'semantic_gain' not in result['improving_plans'][0]


@pytest.mark.parametrize('bad', [score(.1, .29), score(.1, wer=.1), score(.1, unknown=.02)])
def test_space_cannot_buy_content_transcript_or_observability_regressions(bad):
    profile = task()
    result = profile.select([[variant(profile, score())], [variant(profile, bad)]],
        legal=[True, True], repair_penalty=.01)
    assert not result['qualified'] and result['action'] == 0


def test_equal_repair_of_original_removes_unnecessary_plan_compensation():
    profile = task()
    rows = [[variant(profile, score()), variant(profile, score(.04), .5)],
            [variant(profile, score(.08)), variant(profile, score(.05), .5)]]
    raw = profile.select_raw([score(), score(.08)], rows, legal=[True, True])
    joint = profile.select(rows, legal=[True, True], repair_penalty=.01)
    assert raw['action'] == 1 and raw['qualified']
    assert joint['action'] == 0 and joint['trial'] == 1 and not joint['qualified']


def test_source_content_is_not_averaged_away_or_removed():
    profile = task()
    base = score()
    base = RewardScore(base.utility, {**base.costs, 'clap_content_cost/source_1': .7})
    bad = score(.1)
    bad = RewardScore(bad.utility, {**bad.costs, 'clap_content_cost/source_0': .71,
                                   'clap_content_cost/source_1': .69})
    rows = [[variant(profile, base)], [variant(profile, bad)]]
    assert not profile.select(rows, legal=[True, True], repair_penalty=.01)['qualified']
    with pytest.raises(ValueError, match='source content'):
        profile.certificate(score(.1), limits=profile.limits(base))


def test_fixed_original_limits_cannot_be_reset_by_a_repaired_output():
    profile = task()
    limits = profile.limits(score())
    reference = profile.certificate(score(), limits=limits)
    assert profile.certificate(score(.1, .298), limits=limits).improves(reference)
    assert not profile.certificate(score(.1, .29), limits=limits).improves(reference)
    assert limits['clap_content_cost/source_0'] == pytest.approx(.705)


def test_semantic_profile_preserves_existing_teacher_selection():
    profile = task('semantic_v1')
    values = [score(content=.3), score(content=.34)]
    rows = [[variant(profile, raw)] for raw in values]
    expected = select_joint_repair_teacher(rows, legal=[True, True], tolerances=TOLERANCES,
        minimum_gain=.01, semantic_tolerance=.005, repair_penalty=.01)
    assert profile.select(rows, legal=[True, True], repair_penalty=.01) == expected
    assert profile.select_raw(values, rows, legal=[True, True]) == select_query_teacher(
        values, legal=[True, True], tolerances=TOLERANCES)


def test_spatial_gradient_stops_inside_the_requested_sector():
    class Direction:
        def soft_sector_cost(self, audio):
            return (.8 - audio[0]).clamp_min(0.)
    class Content:
        def differentiable_content_cost(self, audio, requirements):
            return audio[1].square()
    value = torch.tensor([.95, .5], requires_grad=True)
    objective = task().latent_objective(value, decode=lambda x: x, scorer=Content(),
        requirements={}, directional_reward=Direction(), diagnostics=[])
    gradient, = torch.autograd.grad(objective, value)
    assert torch.equal(gradient, torch.zeros_like(value))


def test_unsupported_task_and_missing_protection_are_errors():
    with pytest.raises(ValueError, match='supported'):
        EventCommonRepairTask(kind='unknown', tolerances=TOLERANCES,
            minimum_gain=.01, utility_tolerance=.01)
    with pytest.raises(ValueError, match='coverage'):
        task().quality(RewardScore(0., {'clap_content_cost/source_0': .7,
                                     'requested_sector_failure': .2}))


def test_spatial_ar_soft_teacher_and_dit_marginal_share_verified_outcomes():
    profile = task()
    rows = [[variant(profile, score(.2))],
            [variant(profile, score(.3)), variant(profile, score(.15), .25)],
            [variant(profile, score(.14))]]
    q, evidence = repair_teacher_distribution(rows, legal=[True] * 3,
        initial_logits=torch.zeros(3), mode='joint_repair',
        tolerances=profile.score_tolerances(rows[0][0]['score']),
        minimum_gain=profile.minimum_gain, semantic_tolerance=profile.utility_tolerance)
    evidence = profile.label_distillation(evidence)
    selection = profile.select(rows, legal=[True] * 3, repair_penalty=.01)
    assert evidence['selected_joint_teacher'] == selection
    assert selection['action'] == 2
    terms, _ = positive_context_mixture(selection, q, legal=[True] * 3)
    assert [term['action'] for term in terms] == [1]
    assert terms[0]['weight'] == float(q[1])
    assert terms[0]['weight'] > .1
    assert 'utility_gain' in evidence['candidates'][1]
