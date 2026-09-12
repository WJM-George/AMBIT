"""A learnable seven-choice EVENT refiner with native decision provenance.

The complete released EVENT policy is loaded first. The new output head has a
bounded initialization that always selects the original plan. Teacher queries
are not treated as existing student actions until this policy is used.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .event_policy import EventPlanningPolicy, EventProposal, EventBehavior
from .event_refinement import refinement_candidates, REFINEMENT_ACTIONS


def install_refinement_output(head, *, initialization_seed=18322):
    # Use a local generator; do not reseed global CUDA generators when creating
    # this CPU-initialized output module. Preserve the global CPU RNG as well.
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device='cpu').manual_seed(initialization_seed)
        output = nn.Linear(head.width, len(REFINEMENT_ACTIONS))
        nn.init.normal_(output.weight, std=.0001, generator=generator)
        with torch.no_grad():
            output.bias.zero_()
            output.bias[0] = .1
    bound = output.weight.detach().abs().sum(-1)
    if not bool((output.bias[0] - bound[0] > output.bias[1:] + bound[1:]).all()):
        raise ValueError('expanded refiner did not retain an exact initial greedy action')
    head.output = output.to(device=next(head.parameters()).device)


@dataclass(frozen=True)
class NativeFreeDecision:
    prefix: tuple[int, ...]
    legal_ids: tuple[int, ...]
    selected_token: int
    teacher_logits: torch.Tensor


@dataclass(frozen=True)
class RefinementProposal(EventProposal):
    free_decisions: tuple[NativeFreeDecision, ...]


class EventRefinementPolicy(EventPlanningPolicy):
    contract = 'event_generation_seven_choice_refinement_opsd_policy_v2'

    def __init__(self, bundle):
        from .event_experiment import state_fingerprint
        super().__init__(bundle)
        self.initial_native_policy_fingerprint = state_fingerprint(self)
        install_refinement_output(self.completion_head)
        self.eval()

    @torch.no_grad()
    def propose(self, observations):
        captured = [[] for _ in observations]
        def hook(module, args, output):
            tokens, mask = args[:2]
            for index in range(tokens.shape[0]):
                length = int(mask[index].sum())
                prefix = tuple(tokens[index, :length].tolist())
                if prefix[-1] == self.bundle.codec.eos_id:
                    continue
                legal = tuple(sorted(self.bundle.codec.allowed_next_ids(prefix)))
                if len(legal) > 1:
                    captured[index].append((prefix, legal, output[index, length - 1, list(legal)].detach().float().clone()))
        handle = self.bundle.ar.register_forward_hook(hook)
        try:
            tokens, traces = self.bundle.generate_released(observations)
        finally:
            handle.remove()
        proposals = []
        for observation, row, trace, decisions in zip(observations, tokens, traces, captured):
            free = []
            for prefix, legal, logits in decisions:
                if tuple(row[:len(prefix)]) != prefix:
                    raise ValueError('recorded native AR decision was not on the actual generated prefix')
                selected = int(row[len(prefix)])
                if selected != legal[int(logits.argmax())]:
                    raise ValueError('recorded logits do not explain the actual native free decision')
                free.append(NativeFreeDecision(prefix, legal, selected, logits))
            if not free:
                raise ValueError('full EVENT planner did not expose its native room/duration/gain choices')
            proposals.append(RefinementProposal(observation, tuple(row), trace,
                self.bundle.codec.decode(row, sample_id=observation.sample_id), tuple(free)))
        return proposals

    def decision_forward(self, proposal):
        output = super().decision_forward(proposal)
        output['completion_legal'] = torch.stack([refinement_candidates(self.bundle.codec,
            proposal.plan, proposal.trace, slot)[1] for slot in range(len(proposal.plan['sources']))])[None].to(self.device)
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
        plan = copy.deepcopy(proposal.plan)
        for slot, action in enumerate(actions):
            candidates, legal = refinement_candidates(self.bundle.codec, proposal.plan, proposal.trace, slot)
            if not legal[action]:
                raise ValueError('refinement sampled an illegal/duplicate choice')
            plan['sources'][slot] = copy.deepcopy(candidates[action]['sources'][slot])
        return EventBehavior(proposal, tuple(actions), plan, seed)

    def native_retention_loss(self, proposal):
        """Self-KL on the actual free decisions, not every forced text token.

Each stored prefix came from a real native generation forward. Restricting
normalization to its legal alternatives avoids diluting sparse room/duration/
gain retention over the much larger forced serialization vocabulary.
        """
        context, context_mask = self.bundle.encode_event_requests([proposal.observation.request], device=self.device)
        terms = []
        for decision in proposal.free_decisions:
            tokens = torch.tensor([decision.prefix], dtype=torch.long, device=self.device)
            output = self.bundle.ar(tokens, torch.ones_like(tokens, dtype=torch.bool), context, context_mask)
            student = output[0, -1, list(decision.legal_ids)].double().log_softmax(-1)
            teacher = decision.teacher_logits.detach().to(self.device).double().log_softmax(-1)
            terms.append((teacher.exp() * (teacher - student)).sum().float())
        return torch.stack(terms).mean()

    def frozen_copy(self):
        result, _ = load_refinement_policy(self.bundle.release_path, device=self.device,
            qwen_runtime=self.bundle.qwen_runtime, dit_runtime=self.bundle.dit_runtime)
        result.load_state_dict(self.state_dict(), strict=True)
        return result.eval().requires_grad_(False)


def load_refinement_policy(release_path, *, device, qwen_runtime='torch_reference', dit_runtime='fp32'):
    from .generation_event import load_event_generation
    bundle, receipt = load_event_generation(release_path, device=device, qwen_runtime=qwen_runtime, dit_runtime=dit_runtime)
    policy = EventRefinementPolicy(bundle)
    return policy, {**receipt, 'policy_contract': policy.contract, 'action_names': list(REFINEMENT_ACTIONS),
        'initial_native_policy_fingerprint': policy.initial_native_policy_fingerprint,
        'initial_greedy': 'analytically unchanged action 0; actual full EVENT parity still required',
        'native_retention': 'self-teacher on actual native free-decision prefixes and legal alternatives'}
