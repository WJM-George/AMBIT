"""Execution teachers on complete native distributions at visited prefixes.

A finite set M of complete decision paths receives a conditional teacher q_M.
Its original mass m is retained: Q(path)=m*q_M(path) on M, and Q=pi_ref outside.
The resulting conditional distributions are constructed on the prefix trie of
M. The training loss covers these visited prefixes, not every possible prefix
of the native policy. Retention elsewhere and actual output checks remain
necessary. No reward or favorable label is assigned to an unmeasured path.
"""
from dataclasses import dataclass
import json
import math

import torch

from .native_stochastic_policy import (
    _qualitative_inputs, kind_assignments, pointer_module, timing_pair_logits,
)


@dataclass
class DecisionDistribution:
    site: str
    logits: torch.Tensor  # Already divided by the actual sampling temperature.
    selected: int


@dataclass
class PrefixTarget:
    site: str
    probabilities: torch.Tensor
    prefix_mass: float


@dataclass
class PrefixTeacher:
    targets: dict
    measured_mass: float
    normalizer: float
    report: dict


def native_distributions(policy, sample):
    """Expose exactly the distributions used by sample/rescore_native_plan.

    Ordered copy spans are joint distributions over legal low<=high endpoints,
    not two independent endpoint losses. The adapter is differentiable and does
    not replace the released decoder or change ordinary generation.
    """
    if sample.failure or sample.plan is None:
        raise ValueError('A complete observed path is required for this teacher.')
    bundle = policy.bundle
    context, mask = bundle.encode_event_requests([sample.observation.request], device=bundle.device)
    alignment = pointer_module.encode_character_alignment(bundle.prompt_conditioner.tokenizer,
        [sample.observation.request], mask, device=bundle.device)
    with torch.autocast(device_type=context.device.type, enabled=False):
        inventory = bundle.source_inventory(context, mask, alignment)
        qualitative = _qualitative_inputs(bundle, context, mask, alignment, sample.inventory)
    result = []
    for row in sample.decisions:
        family, kind, selected = row['family'], row['kind'], row['selected']
        site = {k: v for k, v in row.items() if k not in ('selected', 'old_log_prob')}
        if family == 'inventory':
            if kind == 'count':
                logits = inventory['count'][0]
            elif kind == 'kinds':
                logits = torch.stack([sum(inventory['kind'][0, slot, k] for slot, k in enumerate(a))
                                      for a in kind_assignments(row['count'])])
            else:
                slot, field = row['slot'], row['field']
                endpoints = alignment.endpoint_mask[0].nonzero().flatten()
                low, high = torch.triu_indices(len(endpoints), len(endpoints), device=context.device)
                start = inventory['start'][0, slot, field].double()
                end = inventory['end'][0, slot, field].double()
                logits = start[endpoints[low]] + end[endpoints[high]]
                matches = ((endpoints[low] == selected[0]) & (endpoints[high] == selected[1])).nonzero().flatten()
                if len(matches) != 1:
                    raise ValueError('The sampled ordered span is outside the complete native support.')
                selected = int(matches[0])
                site['supported_endpoints'] = endpoints.tolist()
        elif family == 'qualitative':
            slot = row['slot']
            if kind == 'static_direction':
                logits = qualitative['start'][0, slot].double() + qualitative['end'][0, slot].double()
            elif kind == 'timing':
                logits = timing_pair_logits(qualitative['onset'][0, slot].double(),
                                           qualitative['offset'][0, slot].double(), row['frames'])
            else:
                logits = qualitative[kind][0, slot]
        elif family == 'token':
            token = torch.tensor([row['prefix']], dtype=torch.long, device=bundle.device)
            logits = bundle.ar(token, torch.ones_like(token, dtype=torch.bool), context, mask)[0, -1, row['support']]
        else:
            raise ValueError('Unknown native decision family.')
        result.append(DecisionDistribution(json.dumps(site, sort_keys=True, separators=(',', ':')),
                                          logits.double()/sample.temperature, int(selected)))
    return result


def path_nodes(paths, *, verify_repeated=False):
    nodes, leaves, path_probabilities = {}, [], []
    for path in paths:
        history, logp = (), None
        if not path:
            raise ValueError('An observed path cannot be empty.')
        for step in path:
            if (step.logits.ndim != 1 or not torch.isfinite(step.logits).all()
                    or not 0 <= step.selected < step.logits.numel()):
                raise ValueError('Require a finite complete categorical support and a legal choice.')
            if history in nodes:
                previous = nodes[history]
                if previous.site != step.site or previous.logits.shape != step.logits.shape:
                    raise ValueError('The same decision history must have the same native support.')
                if verify_repeated and not torch.allclose(previous.logits.detach(), step.logits.detach(), atol=1e-8, rtol=0):
                    raise ValueError('Repeated native prefix logits differ.')
            else:
                nodes[history] = step
            item = step.logits.log_softmax(0)[step.selected]
            logp = item if logp is None else logp + item
            history += ((step.site, step.selected),)
        leaves.append(history)
        path_probabilities.append(logp)
    if len(set(leaves)) != len(leaves) or any(leaf in nodes for leaf in leaves):
        raise ValueError('Use distinct complete paths; a path cannot terminate at an internal prefix.')
    return nodes, leaves, torch.stack(path_probabilities)


@torch.no_grad()
def build_prefix_teacher(paths, conditional_logits=None, *, normalization='measured_mass'):
    """Construct stopped full-support teachers, retaining the mass of M.

    Dividing the chain objective by m matches the initial value and gradient
    of the old conditional-path KL. It strengthens later off-candidate
    constraints and is an explicit objective choice, not free efficiency.
    Reference-only coverage may instead use the sum of visited prefix masses.
    """
    nodes, leaves, logps = path_nodes(paths, verify_repeated=True)
    logps = logps.detach().double().cpu()
    mass = float(logps.logsumexp(0).exp())
    if not math.isfinite(mass) or not 0 < mass <= 1+1e-8:
        raise ValueError('The complete measured paths must have finite nonzero probability mass <=1.')
    prior = logps.softmax(0)
    q = prior if conditional_logits is None else conditional_logits.detach().double().cpu().softmax(0)
    if q.shape != prior.shape or not torch.isfinite(q).all():
        raise ValueError('The conditional teacher must match the distinct complete paths.')
    changes = mass*(q-prior)
    prefix_logps, prefix_delta, children_delta = {(): 0.}, {}, {}
    for leaf, change in zip(leaves, changes.tolist()):
        for length, (site, selected) in enumerate(leaf):
            key = leaf[:length]
            prefix_delta[key] = prefix_delta.get(key, 0.) + change
            child = children_delta.setdefault(key, {})
            child[selected] = child.get(selected, 0.) + change
            raw = nodes[key].logits.detach().double().cpu()
            nxt = leaf[:length+1]
            lp = prefix_logps[key] + float(raw.log_softmax(0)[selected])
            if nxt in prefix_logps and abs(prefix_logps[nxt]-lp) > 1e-8:
                raise ValueError('Inconsistent reference path probabilities.')
            prefix_logps[nxt] = lp
    targets = {}
    minimum = 1.
    for key, node in nodes.items():
        a = math.exp(prefix_logps[key])
        occupancy = a + prefix_delta[key]
        if occupancy < -1e-12:
            raise ValueError('Teacher prefix mass became negative.')
        if occupancy <= 0:
            continue
        reference = node.logits.detach().double().cpu().softmax(0)
        numerator = a*reference
        for selected, change in children_delta[key].items():
            numerator[selected] += change
        if float(numerator.min()) < -1e-12 or abs(float(numerator.sum())-occupancy) > 1e-8:
            raise ValueError('The full teacher distribution is not a valid probability measure.')
        probability = numerator.clamp_min(0)/occupancy
        probability /= probability.sum()
        minimum = min(minimum, float(probability.min()))
        targets[key] = PrefixTarget(node.site, probability, occupancy)
    if normalization == 'measured_mass':
        normalizer = mass
    elif normalization == 'prefix_occupancy':
        normalizer = sum(t.prefix_mass for t in targets.values())
    else:
        raise ValueError('Declare measured_mass or prefix_occupancy normalization.')
    return PrefixTeacher(targets, mass, normalizer, dict(paths=len(paths), nodes=len(targets),
        measured_mass=mass, normalizer=normalizer, normalization=normalization,
        reference_conditional=prior.tolist(), teacher_conditional=q.tolist(),
        largest_complete_support=max(t.probabilities.numel() for t in targets.values()),
        minimum_teacher_probability=minimum,
        scope='Full local distributions at the measured-path prefix trie; unvisited prefixes rely on separate retention.'))


def prefix_teacher_loss(paths, teacher):
    nodes, _, logps = path_nodes(paths)
    if not set(teacher.targets).issubset(nodes):
        raise ValueError('Student paths lost a teacher prefix.')
    terms = []
    for key, target in teacher.targets.items():
        node = nodes[key]
        if node.site != target.site or node.logits.shape != target.probabilities.shape:
            raise ValueError('Student and teacher decision supports differ.')
        q = target.probabilities.to(node.logits.device).detach()
        # Zero-q alternatives remain inside the STUDENT softmax denominator.
        kl = (torch.special.xlogy(q, q)-q*node.logits.log_softmax(0)).sum()
        terms.append(kl*(target.prefix_mass/teacher.normalizer))
    return torch.stack(terms).sum().float(), logps
