import math

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_directional_reward import FixedReferenceDirectionalReward
from stable_audio_tools.training.transfusion_opsd.event_rewards import EventRequestFoaRewardV2
from stable_audio_tools.training.transfusion_opsd.rewards import UnobservableAudio
from test_transfusion_opsd_event import codec, example


def field(samples, *, angle=45., diffuse=1., direct=True):
    t = torch.arange(samples, dtype=torch.float32) / 1024
    w = .03 * torch.sin(2 * math.pi * 7 * t)
    direction = torch.tensor([math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.])
    xyz = direction[:, None] * w * math.sqrt(2) * direct
    xyz = xyz + diffuse * .03 * torch.stack([torch.sin(2 * math.pi * f * t) for f in [11, 13, 17]])
    return torch.stack([w, xyz[1], xyz[2], xyz[0]])[None]


def test_more_coherence_is_not_mislabeled_as_better_angle(codec):
    plan, _, request = example(codec)
    samples = round(plan['duration_sec'] * 44100)
    wet, dry = field(samples, diffuse=1.5), field(samples, diffuse=.2)
    reward = FixedReferenceDirectionalReward(request, plan, reference_audio=wet, model_num_samples=samples)
    old = EventRequestFoaRewardV2(request, plan, model_num_samples=samples)
    assert old(dry).utility > old(wet).utility + .05
    assert reward(dry).utility == pytest.approx(reward(wet).utility, abs=2e-4)
    assert torch.equal(wet[:, 0], dry[:, 0])
    wrong = field(samples, angle=-45., diffuse=1.5).requires_grad_(True)
    assert reward(wrong).utility < reward(wet).utility - .3
    reward.differentiable(wrong).backward()
    assert torch.isfinite(wrong.grad).all() and wrong.grad.norm() > 0
    assert reward.soft_sector_cost(field(samples, angle=30.)).item() == pytest.approx(0., abs=1e-6)
    assert reward.soft_sector_cost(wrong).item() > .5


def test_silence_or_removing_direction_cannot_hide_failed_frames(codec):
    plan, _, request = example(codec)
    samples = round(plan['duration_sec'] * 44100)
    baseline = field(samples)
    reward = FixedReferenceDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples)
    for broken in [torch.zeros_like(baseline), field(samples, direct=False)]:
        score = reward(broken)
        assert score.utility < .01
        assert score.costs['direction_unobservable_fraction'] > .99
        assert score.costs['requested_sector_failure'] > .99
    with pytest.raises(UnobservableAudio, match='reliable'):
        FixedReferenceDirectionalReward(request, plan, reference_audio=torch.zeros_like(baseline), model_num_samples=samples)
