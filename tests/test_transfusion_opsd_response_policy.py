from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_audio_tools.training.transfusion_opsd.event_feedback_policy import EventFeedbackPolicy
from stable_audio_tools.training.transfusion_opsd.event_policy import EventBehavior
from stable_audio_tools.training.transfusion_opsd.event_response_policy import EventResponsePolicy
from stable_audio_tools.training.transfusion_opsd.execution_query import ExecutionQuery


class ToyBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.codec = None
        self.calls = 0

    def render_condition(self, observation, plan):
        return SimpleNamespace(mask=torch.ones(1, 8, dtype=torch.bool), model_num_samples=8, action=plan.get('action', 0))

    def schedule(self, steps, frames):
        return torch.linspace(1., 0., steps+1).tolist()

    def velocity_function(self, condition, *, differentiable):
        def velocity(z, time):
            self.calls += 1
            return torch.ones_like(z) * condition.action * .01
        return velocity

    def decode_for_reward(self, latent, samples):
        return torch.zeros(1, 4, samples)


class RecordingFeedbackPolicy(EventFeedbackPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.bundle = ToyBundle()
        self.latent_channels = 2
        self.executors = []

    def propose(self, observations):
        return [SimpleNamespace(observation=obs, plan={}) for obs in observations]

    def feedback_act(self, proposal, query, *, use_feedback=True, executor=None):
        self.executors.append(executor)
        return EventBehavior(proposal, (0,), proposal.plan, None)


def test_frozen_branch_forwards_the_prefix_executor_to_feedback():
    policy, frozen = RecordingFeedbackPolicy(), ToyBundle()
    policy.generate_feedback_audio(SimpleNamespace(), seed=10, query_index=50, model_version=0, executor=frozen)
    assert policy.executors == [frozen]
    assert policy.bundle.calls == 0
    assert frozen.calls == 101  # 100 native steps plus current clean prediction.


def test_optional_native_trace_keeps_the_actual_prefix_and_conditioning():
    policy = RecordingFeedbackPolicy()
    behavior, query, audio, context = policy.generate_feedback_audio(SimpleNamespace(), seed=10,
        query_index=50, model_version=0, return_trace=True)
    assert len(context['trace'].states) == 101
    assert torch.equal(context['trace'].states[50], query.state)
    assert context['original_velocity'] is context['selected_velocity']
    assert policy.bundle.calls == 101


def test_response_observations_are_fixed_and_executor_specific(monkeypatch):
    import stable_audio_tools.training.transfusion_opsd.event_response_policy as module
    policy = EventResponsePolicy.__new__(EventResponsePolicy)
    nn.Module.__init__(policy)
    policy.bundle = ToyBundle()
    policy.latent_channels, policy.temporal_bins = 2, 4
    plans = tuple({'action': i} for i in range(7))
    monkeypatch.setattr(module, 'refinement_candidates', lambda *args: (plans, torch.ones(7, dtype=torch.bool)))
    monkeypatch.setattr(module, 'completion_geometry', lambda plan: torch.full((1, 8), float(plan['action'])))
    proposal = SimpleNamespace(plan={'sources': [{}]}, observation=SimpleNamespace(), trace=[])
    query = ExecutionQuery(torch.ones(1, 2, 8), torch.ones(1, 2, 8), torch.tensor([.5]),
        torch.ones(1, 8, dtype=torch.bool), 3)
    frozen = ToyBundle()
    prepared = policy.prepare_feedback_query(proposal, query, executor=frozen)
    assert frozen.calls == 7 and policy.bundle.calls == 0
    assert prepared.model_version == 3
    assert prepared.response_features.shape == (1, 7, 17)
    assert prepared.query_features.shape == (1, 33)
    assert policy.prepare_feedback_query(proposal, prepared, executor=frozen) is prepared
    assert frozen.calls == 7
    with pytest.raises(ValueError, match='different executor'):
        policy.prepare_feedback_query(proposal, prepared, executor=policy.bundle)
    with pytest.raises(ValueError, match='detached'):
        replace(prepared, response_features=prepared.response_features.clone().requires_grad_())


def test_full_restore_rejects_a_different_response_input_mode(tmp_path):
    from stable_audio_tools.training.transfusion_opsd.event_trainer import EventOPSDTrainer

    class SmallPolicy(nn.Module):
        contract = 'test_response_policy'

        def __init__(self, mode):
            super().__init__()
            self.bundle = SimpleNamespace(qwen_runtime='torch_reference', dit_runtime='fp32')
            self.runtime_contract = {'response_input_mode': mode}
            self.shared = nn.Parameter(torch.ones(2))
            self.ar = nn.Parameter(torch.ones(2))
            self.dit = nn.Parameter(torch.ones(2))
            self.completion_head = nn.Linear(2, 2)

        def dependency_parameters(self):
            return {'shared': [('shared', self.shared)], 'dit_private': [('dit', self.dit)],
                'ar_private': [('ar', self.ar)] + [('completion_head.'+name, value)
                    for name, value in self.completion_head.named_parameters()]}

    identity = {'initialization': 'small CPU interface test', 'data': {}, 'protocol': {}}
    source = EventOPSDTrainer(SmallPolicy('current_response'), identity=identity)
    sum(value.square().sum() for value in source.policy.parameters()).backward()
    source.optimizer.step()
    source.commits = source.fit_attempts = 1
    path = tmp_path/'candidate.pt'
    source.save_candidate(path, collection_state={'next_collection_version': 1})
    restored = EventOPSDTrainer(SmallPolicy('current_response'), identity=identity)
    assert restored.restore_candidate(path) == {'next_collection_version': 1}
    assert restored.commits == 1 and len(restored.optimizer.state) == len(source.optimizer.state) > 0
    for key, value in source.policy.state_dict().items():
        assert torch.equal(value, restored.policy.state_dict()[key])
    wrong = EventOPSDTrainer(SmallPolicy('query_moments'), identity=identity)
    with pytest.raises(ValueError, match='policy_runtime_contract'):
        wrong.restore_candidate(path)
    assert wrong.commits == 0 and not wrong.optimizer.state
