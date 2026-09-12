"""Full EVENT proposal plus a learned finite numeric-completion decision stage."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import GenerationObservation
from .event_completion import EventCompletionHead, completion_geometry, completion_candidates, apply_completion_actions


@dataclass(frozen=True)
class EventProposal:
    observation: GenerationObservation
    tokens: tuple[int, ...]
    trace: list[dict]
    plan: dict


@dataclass(frozen=True)
class EventBehavior:
    proposal: EventProposal
    actions: tuple[int, ...]
    plan: dict
    seed: int | None


class EventPlanningPolicy(nn.Module):
    contract = 'event_generation_execution_coupled_opsd_policy_v1'

    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
        self.completion_head = EventCompletionHead().to(bundle.device)
        # The selected inventory decoder retains but does not call this fallback.
        # Keep its full state; do not create fictitious optimizer ownership.
        self.bundle.copy_pointer.requires_grad_(False)
        self.eval()

    @property
    def device(self):
        return self.bundle.device

    def dependency_parameters(self):
        result = {key: [('bundle.' + name, value) for name, value in items]
            for key, items in self.bundle.dependency_parameters().items()}
        result['ar_private'] += [('completion_head.' + name, value)
            for name, value in self.completion_head.named_parameters() if value.requires_grad]
        return result

    @torch.no_grad()
    def propose(self, observations):
        tokens, traces = self.bundle.generate_released(observations)
        return [EventProposal(observation, tuple(row), trace,
            self.bundle.codec.decode(row, sample_id=observation.sample_id))
            for observation, row, trace in zip(observations, tokens, traces)]

    def decision_forward(self, proposal):
        output = self.bundle.planning_forward([proposal.observation], [proposal.tokens], [proposal.trace])
        geometry = completion_geometry(proposal.plan)[None].to(self.device)
        output['completion'] = self.completion_head(output['source_queries'], output['qualitative'], geometry)
        output['completion_legal'] = torch.stack([completion_candidates(self.bundle.codec, proposal.plan,
            proposal.trace, slot)[1] for slot in range(len(proposal.plan['sources']))])[None].to(self.device)
        return output

    @torch.no_grad()
    def act(self, proposal, *, seed=None):
        output = self.decision_forward(proposal)
        logits = output['completion'][0].double().masked_fill(~output['completion_legal'][0], -torch.inf)
        if seed is None:
            actions = logits.argmax(-1).tolist()
        else:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            actions = [int(torch.multinomial(row.softmax(-1).cpu(), 1, generator=generator)) for row in logits]
        plan = apply_completion_actions(self.bundle.codec, proposal.plan, proposal.trace, actions)
        return EventBehavior(proposal, tuple(actions), plan, seed)

    @torch.no_grad()
    def generate(self, observations):
        return [self.act(proposal) for proposal in self.propose(observations)]

    def frozen_copy(self):
        result = type(self)(self.bundle.frozen_copy())
        result.completion_head.load_state_dict(self.completion_head.state_dict(), strict=True)
        return result.eval().requires_grad_(False)


def load_event_policy(release_path, *, device, qwen_runtime, dit_runtime='native_bf16'):
    from .generation_event import load_event_generation
    bundle, receipt = load_event_generation(release_path, device=device, qwen_runtime=qwen_runtime, dit_runtime=dit_runtime)
    policy = EventPlanningPolicy(bundle)
    return policy, {**receipt, 'policy_contract': policy.contract,
        'completion_initialization': 'greedy action 0 is exactly the released numeric completion',
        'completion_observation': 'actual EVENT AR source states, predicted qualitative distributions and proposed geometry',
        'retained_inactive_literal_pointer': 'full checkpoint state, frozen while inventory decoder is active'}
