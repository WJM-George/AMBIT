from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.execution_sample_distillation import execution_sample_teacher
from stable_audio_tools.training.transfusion_opsd.native_execution_teacher import NativeDecisionSite
from stable_audio_tools.training.transfusion_opsd.native_reward_execution_teacher import (
    RewardExecutionObservation, build_native_reward_execution_teacher, native_reward_free_objective,
)


def test_distribution_variational_identity_and_stopped_inputs():
    rewards = torch.tensor([0., .4, .8], dtype=torch.float64, requires_grad=True)
    prior = torch.tensor([.5, .3, .2], dtype=torch.float64, requires_grad=True)
    teacher = execution_sample_teacher(rewards, temperature=.2, prior=prior)
    q = teacher['probabilities']
    alternative = torch.tensor([.25, .5, .25], dtype=torch.float64)
    objective = lambda a: (a * rewards.detach()).sum() - .2 * (a * (a.log() - prior.detach().log())).sum()
    torch.testing.assert_close(objective(q) - objective(alternative), .2 * (alternative * (alternative.log() - q.log())).sum())
    assert teacher['mean_reward_target'] > teacher['mean_reward_before']
    assert all(not x.requires_grad for x in teacher.values())


def test_constant_reward_preserves_prior_and_common_offset_changes_nothing():
    prior = torch.tensor([.2, .3, .5], dtype=torch.float64)
    t = execution_sample_teacher(torch.ones(3), temperature=.2, prior=prior)
    torch.testing.assert_close(t['probabilities'], prior)
    a = execution_sample_teacher(torch.tensor([.2, .4], dtype=torch.float64), temperature=.1)
    b = execution_sample_teacher(torch.tensor([.2, .4], dtype=torch.float64) + 4, temperature=.1)
    torch.testing.assert_close(a['probabilities'], b['probabilities'])


def test_tiny_sample_probability_does_not_make_target_kl_nan():
    teacher = execution_sample_teacher(torch.tensor([0., 1.]), temperature=.0001)
    assert all(torch.isfinite(x).all() for x in teacher.values())
    torch.testing.assert_close(teacher['target_kl'], torch.tensor(2., dtype=torch.float64).log())


@pytest.mark.parametrize('rewards,temp,prior', [
    ([float('nan'), 1.], .1, None), ([0., 1.], 0., None), ([0., 1.], .1, [0., 1.]),
])
def test_invalid_execution_weights_rejected(rewards, temp, prior):
    with pytest.raises(ValueError):
        execution_sample_teacher(torch.tensor(rewards), temperature=temp,
            prior=None if prior is None else torch.tensor(prior))


def test_actual_native_taught_site_receives_credit_and_other_site_retains_zero_gradient():
    logits = torch.nn.Parameter(torch.tensor([[2., 1., 0.], [2., 1., 0.]]))
    class Native(torch.nn.Module):
        def forward(self, tokens, mask, context, context_mask):
            return logits[tokens[:, -1], None, :]
    bundle = SimpleNamespace(ar=Native(), encode_event_requests=lambda rows, device: (torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool)))
    decisions = tuple(SimpleNamespace(prefix=(i,), legal_ids=(0, 1, 2), selected_token=0,
        teacher_logits=logits[i].detach().clone()) for i in range(2))
    observation = SimpleNamespace(sample_id='case', request='say the requested words')
    proposal = SimpleNamespace(free_decisions=decisions, observation=observation)
    policy = SimpleNamespace(bundle=bundle, device=torch.device('cpu'))
    teacher = build_native_reward_execution_teacher(decisions[0].teacher_logits,
        site=NativeDecisionSite.from_request('case', observation.request, (0,), (0, 1, 2)),
        executor_fingerprint='1'*64, observer_fingerprint='2'*64, evidence_sha256='3'*64,
        reference_token=0, paired_seeds=(17, 18), temperature=.1,
        observations=[RewardExecutionObservation(token, seed, .2 if token == 0 else .9, 'passed')
            for token in (0, 1) for seed in (17, 18)])
    rows, _ = native_reward_free_objective(policy, proposal, teachers=(teacher,),
        round_executor_fingerprint='1'*64, observer_fingerprint='2'*64)
    sum(x['positive'] + x['retained_kl'] + x['retained_margin'] for x in rows).backward()
    assert logits.grad[0, 1] < 0 and logits.grad[0, 0] > 0
    assert logits.grad[1].count_nonzero() == 0
    with pytest.raises(ValueError):
        native_reward_free_objective(policy, proposal, teachers=(teacher,),
            round_executor_fingerprint='4'*64, observer_fingerprint='2'*64)
