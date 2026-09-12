"""Evidence-conditioned self-teachers on actual EVENT decision modules.

This is an OPSD adaptation: a verified request changes a detached current-model
distribution analytically, instead of pretending forced tokens decided the
inventory or adding annotations to the student's text input.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import torch

from ...models.sceneplan_generation_ar_qualitative_head import ATTRIBUTES
from ...models.sceneplan_generation_ar_source_inventory import KINDS
from .objectives import forward_kl, EulerTrace


@dataclass(frozen=True)
class EventTeacher:
    heads: dict
    completion: torch.Tensor
    completion_legal: torch.Tensor
    token_anchor: torch.Tensor
    evidence: dict


def _detached(values):
    return {name: value.detach().clone() for name, value in values.items()}


def build_event_task_teacher(output, proposal, request_reward, *, logit_bonus=2.):
    """Only traceable, verified decisions receive a task-evidence logit bonus.

    Unspecified radial/timing/angle values have no invented exact answer. An
    unverified content match is not made into an incorrect semantic label.
    """
    if not 0 < logit_bonus <= 10:
        raise ValueError('task evidence bonus must be bounded and positive')
    score = request_reward.evaluate(proposal.plan)
    if not score['request_constraints_joint']:
        raise ValueError('task teacher requires a verified current proposal, including source binding')
    inventory = _detached(output['inventory'])
    qualitative = _detached(output['qualitative'])
    count = len(proposal.plan['sources'])
    inventory['count'][0, count - 1] += logit_bonus
    by_key = {row['key']: row for row in request_reward.requirements['sources']}
    bindings = {value: key for key, value in score['assignment'].items()}
    changed = []
    events = [event for event in proposal.trace if event['field'] != 'transcript']
    for slot, source in enumerate(proposal.plan['sources']):
        inventory['kind'][0, slot, KINDS.index(source['kind'])] += logit_bonus
        event = events[slot]
        # Literal-span evidence is already independently verified against this
        # source's requested content. Event-scope endpoints have no unique label.
        for key, endpoint in [('start', event['start']), ('end', event['end'])]:
            inventory[key][0, slot, 0, endpoint] += logit_bonus
        if source['kind'] == 'speech':
            transcript = next(e for e in proposal.trace if e['generated_source_slot'] == slot and e['field'] == 'transcript')
            for key, endpoint in [('start', transcript['start']), ('end', transcript['end'])]:
                inventory[key][0, slot, 1, endpoint] += logit_bonus
        reference = by_key[bindings[source['source_id']]]
        targets = {}
        for constraint in reference['constraints']:
            op = constraint['op']
            if op == 'motion':
                targets['motion'] = constraint['value']
            elif op == 'compass':
                for point in ('start', 'end') if constraint['point'] == 'both' else (constraint['point'],):
                    targets[point] = constraint['value']
            elif op == 'time_phase':
                targets[{'onset_sec': 'onset', 'offset_sec': 'offset'}[constraint['field']]] = constraint['value']
            elif op == 'distance_change':
                targets['radial'] = constraint['value']
        for name, target in targets.items():
            qualitative[name][0, slot, ATTRIBUTES[name].index(target)] += logit_bonus
            changed.append({'source_id': source['source_id'], 'attribute': name, 'requested_category': target})
    return EventTeacher({'inventory': inventory, 'qualitative': qualitative},
        output['completion'].detach().double().clone(), output['completion_legal'].detach().clone(),
        output['token_logits'].detach().clone(), {'source_assignment': copy.deepcopy(score['assignment']),
            'request_category_evidence': changed, 'logit_bonus': logit_bonus,
            'event_scope_teacher': 'unchanged detached current distribution; no unique endpoint claimed',
            'student_input_contains_annotations': False, 'hidden_numeric_target': False})


def event_ar_loss(output, teacher, *, completion_weight=1., task_weight=1., anchor_weight=.05):
    """Actual head/completion KL, with token logits used only for retention."""
    losses = []
    for family in ('inventory', 'qualitative'):
        terms = []
        for name, target in teacher.heads[family].items():
            student = output[family][name]
            legal = torch.isfinite(target)
            terms.append(forward_kl(student, target, legal, temperature=1.))
        losses.append(torch.stack(terms).mean())
    task = torch.stack(losses).mean()
    legal = teacher.completion_legal
    lp = output['completion'].double().masked_fill(~legal, -torch.inf).log_softmax(-1)
    lq = teacher.completion.detach().double().masked_fill(~legal, -torch.inf).log_softmax(-1)
    completion = (lq.exp() * (lq.masked_fill(~legal, 0.) - lp.masked_fill(~legal, 0.))).sum(-1).mean().float()
    # Retain the old AR representation without claiming these forced token
    # logits were responsible for inventory/qualitative decisions.
    anchor = (output['token_logits'].float() - teacher.token_anchor).square().mean()
    return task_weight * task + completion_weight * completion + anchor_weight * anchor


@torch.no_grad()
def native_continue_from_query(trace, query_index, velocity, *, return_trace=False):
    """Continue a real diffusion prefix under a possibly revised condition.

    The input is the already generated query state, not a fresh noise draw or
    a clean prediction treated as a trajectory. This is an inference-schedule
    intervention and must be declared separately from full-plan rerendering.
    """
    from ...inference.sampling import sample_discrete_euler
    if not 0 <= query_index < len(trace.times) - 1:
        raise ValueError('condition revision must precede the native endpoint')
    state = trace.states[query_index]
    times = torch.tensor(trace.times[query_index:], device=state.device, dtype=torch.float32)
    states = []
    final = sample_discrete_euler(velocity, state.clone(), times, disable_tqdm=True,
        callback=(lambda values: states.append(values['x'].detach().clone())) if return_trace else None)
    if return_trace:
        return EulerTrace(tuple([*trace.states[:query_index], *states, final]), trace.times, trace.mask)
    return final


@torch.no_grad()
def native_resume_from_clean(trace, query_index, clean, velocity, *, anchor=None):
    """Perturb the recorded successor, then run the unchanged native suffix.

    Reconstructing the entire velocity through (z - clean) / t introduces a
    floating-point round trip even when clean is unchanged. Native BF16 VAE
    decoding can amplify that error. Applying only the clean displacement to
    the recorded successor makes the zero intervention exactly the baseline.
    The trace and anchor must come from the same frozen native query.
    """
    from ...inference.sampling import sample_discrete_euler
    if not 0 <= query_index < len(trace.times) - 1:
        raise ValueError('clean intervention must precede the native endpoint')
    z = trace.states[query_index]
    if clean.shape != z.shape:
        raise ValueError('clean intervention geometry differs from its actual query')
    times = torch.tensor(trace.times[query_index:], device=z.device, dtype=torch.float32)
    if anchor is None:
        from .objectives import clean_prediction
        time = times[0].expand(z.shape[0])
        anchor = clean_prediction(z, time, velocity(z, time)).detach()
    if anchor.shape != z.shape:
        raise ValueError('native intervention anchor geometry differs from its query')
    successor = trace.states[query_index + 1] + ((times[0] - times[1]) / times[0]) * (clean - anchor)
    if len(times) == 2:
        return successor
    return sample_discrete_euler(velocity, successor, times[1:], disable_tqdm=True)
