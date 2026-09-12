"""Native Editing token decisions for the common OPSD condition surrogate.

The rollout prefix remains the student's. Counterfactual suffixes are held
fixed, so this is finite conditional support, not the full plan distribution.
Schema legality and task admissibility are separate required checks. No target
audio or source annotation enters the recomputed student logits.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EditingTokenDecision:
    prefix: tuple[int, ...]
    choices: tuple[int, ...]
    hard_index: int
    plans: tuple[dict, ...]

    def student_logits(self, adapter, observation):
        tokens = torch.tensor([self.prefix], device=adapter.device, dtype=torch.long)
        full = adapter.student_logits(observation, tokens)[0, -1].float()
        return full[list(self.choices)]


def editing_token_decision(adapter, observation, rollout_tokens, *, position,
                           choice_ids, admissible):
    """Validate explicit local alternatives at a supplied native rollout token.

    ``admissible(plan)`` must check requested and preserved facts, in addition
    to the codec grammar checked here. It is training-side validation only.
    This helper does not sample, choose a teacher or change inference.
    """
    from ...models.sceneplan_transfusion_editing_pipeline import (
        _align_decoded_sceneplan_to_audio_duration,
    )

    if isinstance(rollout_tokens, torch.Tensor):
        rollout_tokens = rollout_tokens.detach().cpu().flatten().tolist()
    tokens = tuple(int(x) for x in rollout_tokens)
    choices = tuple(int(x) for x in choice_ids)
    if (isinstance(position, bool) or not 0 < position < len(tokens)
            or len(choices) < 2 or len(set(choices)) != len(choices)
            or tokens[position] not in choices or not callable(admissible)):
        raise ValueError('Require a native prefix, distinct choices and explicit admissibility.')
    prefix = tokens[:position]
    allowed = set(adapter.allowed_next_ids(observation, list(prefix)))
    if not set(choices) <= allowed:
        raise ValueError('A local choice is outside the actual native token grammar.')
    plans = []
    for choice in choices:
        candidate = list(tokens)
        candidate[position] = choice
        # A legal next token can still invalidate the held suffix.
        for index in range(position + 1, len(candidate)):
            if candidate[index] not in adapter.allowed_next_ids(observation, candidate[:index]):
                raise ValueError('The counterfactual choice invalidates a later native token.')
        plan = adapter.codec.decode(candidate, sample_id=observation.sample_id)
        plan = _align_decoded_sceneplan_to_audio_duration(
            plan, observation.model_num_samples / 44100,
        )
        if admissible(plan) is not True:
            raise ValueError('A schema-legal plan violates task or preservation constraints.')
        plans.append(plan)
    return EditingTokenDecision(prefix, choices, choices.index(tokens[position]), tuple(plans))
