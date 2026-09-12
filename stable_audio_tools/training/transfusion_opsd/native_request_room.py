"""Explicit request-room supervision on the current native student prefix."""
from __future__ import annotations

import torch

from ...data.model_sceneplan_codec_v3 import ROOM_TOKENS


def requested_room(requirements):
    values = {x['value'] for x in requirements.get('scene', ()) if x['op'] == 'room'}
    if len(values) > 1 or not values.issubset(ROOM_TOKENS):
        raise ValueError('Request room constraints must be consistent and supported.')
    return next(iter(values)) if values else None


def native_requested_room_loss(policy, proposal, requirements):
    """Recompute differentiable room logits; no sampled logit is reused.

    The caller supplies a current, actually sampled proposal. The target comes
    only from an explicit request requirement, never the completed ScenePlan.
    Absent room requirements provide no categorical supervision.
    """
    room = requested_room(requirements)
    if room is None:
        return None
    codec = policy.bundle.codec
    marker = codec._tid('<room>')
    decisions = [x for x in proposal.free_decisions if x.prefix[-1] == marker]
    if len(decisions) != 1:
        raise ValueError('Expected one actual native room decision.')
    decision = decisions[0]
    if tuple(proposal.tokens[:len(decision.prefix)]) != decision.prefix:
        raise ValueError('Room supervision must use the current sampled prefix.')
    support = tuple(sorted(codec._tid(x) for x in ROOM_TOKENS.values()))
    if decision.legal_ids != support or set(codec.allowed_next_ids(decision.prefix)) != set(support):
        raise ValueError('Room logits must span the complete native legal support.')
    context, context_mask = policy.bundle.encode_event_requests(
        [proposal.observation.request], device=policy.device)
    tokens = torch.tensor([decision.prefix], dtype=torch.long, device=policy.device)
    output = policy.bundle.ar(tokens, torch.ones_like(tokens, dtype=torch.bool), context, context_mask)
    logits = output[0, -1, list(support)]
    target = support.index(codec._tid(ROOM_TOKENS[room]))
    if not torch.isfinite(logits).all():
        raise ValueError('Native room logits must be finite.')
    loss = torch.nn.functional.cross_entropy(logits[None].float(), logits.new_tensor([target], dtype=torch.long))
    return loss, dict(room=room, prefix=list(decision.prefix), target_token=support[target],
                     selected_token=support[int(logits.detach().argmax())],
                     correct=int(logits.detach().argmax()) == target,
                     explicit_request_target=True, loss=float(loss.detach()))
