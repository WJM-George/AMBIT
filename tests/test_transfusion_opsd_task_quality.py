import pytest

from stable_audio_tools.training.transfusion_opsd.event_task_quality import (
    SemanticWithinRequestReward, request_quality_change, select_query_teacher)
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore


def score(content, sector=0., wer=0.):
    return RewardScore(.99, {'clap_content_cost/source_0': 1. - content,
        'requested_sector_failure': sector, 'asr_wer': wer})


def test_semantic_teacher_respects_requested_region_and_exact_speech_protection():
    tolerances = {'requested_sector_failure': .01, 'asr_wer': 0.}
    values = [score(.3), score(.34, .005), score(.4, .02), score(.5, 0., .1)]
    result = select_query_teacher(values, legal=[True] * 4, tolerances=tolerances)
    assert result['qualified'] and result['action'] == 1
    assert not select_query_teacher([score(.3), score(.305)], legal=[True] * 2,
        tolerances=tolerances)['qualified']
    with pytest.raises(ValueError, match='coverage'):
        request_quality_change(values[0], RewardScore(0, {'asr_wer': 0}), tolerances=tolerances)


def test_local_targets_cannot_move_reference_limits():
    base = score(.3)
    reward = SemanticWithinRequestReward(lambda value: value, base,
        tolerances={'requested_sector_failure': .01, 'asr_wer': 0.})
    assert reward(score(.34, .005)).improves(reward(base))
    assert not reward(score(.4, .02)).improves(reward(base))
    assert not reward(score(.5, 0., .1)).improves(reward(base))


def test_refresh_teacher_compares_against_actual_current_action():
    tolerances = {'requested_sector_failure': .01, 'asr_wer': 0.}
    values = [score(.3), score(.36), score(.34)]
    current = select_query_teacher(values, legal=[True] * 3, tolerances=tolerances, baseline_action=1)
    assert current['action'] == 1 and not current['qualified']
    values[0] = score(.4)
    repaired = select_query_teacher(values, legal=[True] * 3, tolerances=tolerances, baseline_action=1)
    assert repaired['qualified'] and repaired['action'] == 0
