import copy
import random

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from stable_audio_tools.training.transfusion_opsd.objectives import (
    RewardScore, DiffusionTargets, forward_kl, legal_log_probs, sample_ar,
    euler_rollout, resume_from_clean, clean_prediction, build_diffusion_targets, diffusion_loss,
)
from stable_audio_tools.training.transfusion_opsd.shared_step import (
    joint_adam_step, project_shared_displacement,
)


class TinyCodec:
    bos_id, eos_id, vocab_size, fingerprint = 0, 3, 4, "tiny-codec-test-only"

    def allowed_next_ids(self, ids):
        return {1, 2} if len(ids) == 1 else {3}


def test_legal_forward_kl_is_dense_detached_and_padding_safe():
    student = torch.tensor([[[.2, -.7, 10.], [float("nan")] * 3]], requires_grad=True)
    teacher = torch.tensor([[[1., 2., -100.], [float("nan")] * 3]], requires_grad=True)
    allowed = torch.tensor([[[True, True, False], [False] * 3]])
    loss = forward_kl(student, teacher, allowed, temperature=.8, token_weights=torch.tensor([[1., 0.]]))
    lp, lq = (student[0, 0, :2] / .8).log_softmax(-1), (teacher[0, 0, :2].detach() / .8).log_softmax(-1)
    torch.testing.assert_close(loss, (lq.exp() * (lq - lp)).sum())
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    assert student.grad[0, 0, 2] == 0 and not student.grad[0, 1].count_nonzero()
    assert student.grad[0, 0, 0] != 0 and student.grad[0, 0, 1] != 0


def test_rollout_logs_exact_untruncated_probabilities_and_incompleteness():
    codec = TinyCodec()
    logits = torch.tensor([0., 1., -1., 0.])
    fn = lambda ids: logits[None, None].expand(1, ids.shape[1], -1)
    result = sample_ar(fn, codec, seed=42, device=torch.device("cpu"), temperature=.8)
    expected = legal_log_probs(logits, torch.tensor([False, True, True, False]), .8)
    assert result.finished and result.token_ids[-1] == codec.eos_id
    assert result.log_probs[0] == float(expected[result.token_ids[1]])
    assert result.log_probs[1] == 0.
    assert result.legal_ids[0] == (1, 2)
    partial = sample_ar(fn, codec, seed=42, device=torch.device("cpu"), max_tokens=2)
    assert not partial.finished and len(partial.log_probs) == 1


def test_euler_resume_preserves_nonuniform_schedule_and_padding():
    z = torch.tensor([[[.5, 1., 999.], [-.7, .2, 999.]]])
    mask = torch.tensor([[True, True, False]])
    velocity = lambda x, t: .2 * x + t[:, None, None]
    trace = euler_rollout(velocity, z, mask, [1., .8, .3, .1, 0.])
    for query in range(4):
        time = torch.tensor([trace.times[query]])
        clean = clean_prediction(trace.states[query], time, velocity(trace.states[query], time))
        resumed = resume_from_clean(trace, query, clean, velocity)
        torch.testing.assert_close(resumed, trace.states[-1], atol=2e-7, rtol=1e-6)
        assert not resumed[..., -1].count_nonzero()


def test_diffusion_targets_validate_actual_suffix_not_just_anchor_reward():
    anchor = torch.tensor([[[.2, 99.]]])
    mask = torch.tensor([[True, False]])
    reward = lambda z: -(z[..., 0] - 1).square().sum()
    score = lambda z: RewardScore(float(reward(z)), {"protected": 0.})
    target = build_diffusion_targets(anchor, mask, differentiable_reward=reward,
        score_clean=score, score_suffix=score, radius=.5, target_steps=2,
        target_mode="positive")
    assert target is not None
    assert target.positive[0, 0, 0] > .2 and target.positive[0, 0, 1] == 0
    assert not target.positive.requires_grad
    # Same decoded reward gain, opposite effect through a nonlinear executor.
    backwards = lambda z: RewardScore(float(-reward(z)), {"protected": 0.})
    rejected = build_diffusion_targets(anchor, mask, differentiable_reward=reward,
        score_clean=score, score_suffix=backwards, radius=.5)
    assert rejected is None


def test_zero_reward_gradient_cannot_create_a_fake_teacher():
    z = torch.ones(1, 1, 2)
    score = lambda value: RewardScore(0., {})
    assert build_diffusion_targets(z, torch.ones(1, 2, dtype=torch.bool),
        differentiable_reward=lambda value: value.sum() * 0,
        score_clean=score, score_suffix=score) is None


def test_positive_configuration_fits_the_actual_certified_target():
    anchor = torch.tensor([[[.2, 99.]]])
    mask = torch.tensor([[True, False]])
    calls = []
    def reward(value):
        return -(value[..., 0] - 1).square().sum()
    def differentiable(value):
        calls.append(value.detach().clone())
        return reward(value)
    score = lambda value: RewardScore(float(reward(value)), {"protected": 0.})
    # Omitting target_mode exercises the initial configuration's public default.
    targets = build_diffusion_targets(anchor, mask, differentiable_reward=differentiable,
        score_clean=score, score_suffix=score, radius=.5, target_steps=2)
    assert targets is not None and len(calls) == 2
    assert score(targets.positive).improves(score(targets.anchor))
    time = torch.tensor([.25])
    initial_v = torch.zeros_like(anchor, requires_grad=True)
    initial_loss = diffusion_loss(targets.anchor, time, initial_v, targets)
    initial_loss.backward()
    assert initial_v.grad[..., 0] > 0 and initial_v.grad[..., 1] == 0
    exact_v = (targets.anchor - targets.positive) / .25
    torch.testing.assert_close(diffusion_loss(targets.anchor, time, exact_v, targets), torch.tensor(0.))
    assert not targets.positive.requires_grad


def test_rejected_positive_target_does_not_spend_negative_search_budget():
    anchor = torch.tensor([[[.2]]])
    calls = []
    def reward(value):
        return -(value - 1).square().sum()
    def differentiable(value):
        calls.append(value.detach().clone())
        return reward(value)
    clean = lambda value: RewardScore(float(reward(value)), {})
    # The clean-output improvement degrades the actual continuation.
    suffix = lambda value: RewardScore(float(-reward(value)), {})
    assert build_diffusion_targets(anchor, torch.ones(1, 1, dtype=torch.bool),
        differentiable_reward=differentiable, score_clean=clean, score_suffix=suffix,
        target_steps=2) is None
    assert len(calls) == 2


def test_positive_target_pushes_velocity_toward_better_clean_output():
    anchor = torch.ones(1, 1, 2)
    targets = DiffusionTargets(anchor, anchor + .1, torch.ones(1, 2, dtype=torch.bool))
    velocity = torch.zeros_like(anchor, requires_grad=True)
    loss = diffusion_loss(anchor, torch.tensor([.25]), velocity, targets)
    loss.backward()
    assert (velocity.grad > 0).all()  # descending v increases z_0 = z_t - t*v
    assert targets.positive.grad is None


def test_projection_uses_adam_geometry_not_raw_gradient_geometry():
    ga, gd = torch.tensor([1., 2.]), torch.tensor([-2., 1.])
    raw = torch.tensor([1., -3.])
    assert ga @ raw < 0 and gd @ raw < 0
    actual = torch.tensor([10., -3.])
    assert ga @ actual > 0
    result, metrics = project_shared_displacement([actual], [ga], [gd], [torch.tensor([10., 1.])])
    torch.testing.assert_close(result[0], torch.tensor([50 / 7, -25 / 7]))
    assert metrics["ar_dot"] <= 1e-6 and metrics["dit_dot"] < 0
    opposed, _ = project_shared_displacement([actual], [ga], [-ga], [torch.ones(2)])
    assert abs(float(ga @ opposed[0])) < 1e-6


class Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.ar = nn.Parameter(torch.tensor(0.))
        self.shared = nn.Parameter(torch.zeros(2))
        self.dit = nn.Parameter(torch.tensor(0.))
        self.register_buffer("calls", torch.zeros(()))

    def partition(self):
        return {"ar_private": [("ar", self.ar)], "shared": [("shared", self.shared)],
                "dit_private": [("dit", self.dit)]}

    def losses(self):
        self.calls.add_(1)
        # Exercise deterministic replay and rejected-step RNG restoration.
        torch.rand(()); random.random(); np.random.rand()
        return ((self.ar + self.shared.sum() - 2).square(),
                (self.dit - self.shared[0] + self.shared[1] - 1).square(), None)


def _assert_tree_equal(a, b):
    if isinstance(a, Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_tree_equal(x, y)
    else:
        assert a == b


def test_commit_and_rejection_are_atomic_including_warm_adam_and_scheduler():
    model = Pair()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    accepted = joint_adam_step(model, model.partition(), optimizer, model.losses,
                              scheduler=scheduler, validate_execution=lambda: True)
    assert accepted.committed and accepted.ar_after < accepted.ar_before and accepted.dit_after < accepted.dit_before
    assert model.ar != 0 and model.dit != 0 and model.shared.count_nonzero()
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    rng = torch.get_rng_state().clone()
    rejected = joint_adam_step(model, model.partition(), optimizer, model.losses,
                              scheduler=scheduler, validate_execution=lambda: False)
    assert not rejected.committed
    _assert_tree_equal(before, (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(RuntimeError, match="scorer failure"):
        joint_adam_step(model, model.partition(), optimizer, model.losses,
            validate_execution=lambda: (_ for _ in ()).throw(RuntimeError("scorer failure")))
    _assert_tree_equal(before, (model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))


def test_cost_coverage_cannot_disappear_and_quality_cannot_buy_count_regression():
    baseline = RewardScore(.2, {"count": 0.})
    assert not RewardScore(.9, {"count": 1.}).improves(baseline)
    with pytest.raises(ValueError, match="coverage"):
        RewardScore(.9, {}).improves(baseline)
