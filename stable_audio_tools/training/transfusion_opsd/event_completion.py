"""Learn numeric completion choices inside the EVENT planner's own categories.

This is an explicit second planning stage after the full EVENT proposal. It
does not reinterpret the probability of forced serialization tokens. Only raw
student states, predicted categories, and the student's proposal enter it.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from ...models.sceneplan_generation_ar_qualitative_head import ATTRIBUTES, CENTERS

COMPLETION_CONTRACT = 'event_angular_completion_refiner_v1'
ANGLE_OFFSETS = (0., -10., 10.)
CATEGORY_HALF_WIDTH = 22.


class EventCompletionHead(nn.Module):
    """Finite decision policy with a provably unchanged initial greedy action."""

    def __init__(self, hidden_dim=1024, width=128, *, initialization_seed=18321):
        super().__init__()
        self.hidden_dim, self.width = hidden_dim, width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.hidden = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
            self.attributes = nn.Linear(sum(map(len, ATTRIBUTES.values())), width)
            self.geometry = nn.Linear(8, width)
            self.output = nn.Linear(width, len(ANGLE_OFFSETS))
            nn.init.normal_(self.output.weight, std=.0001)
            with torch.no_grad():
                self.output.bias.copy_(torch.tensor([.1, 0., 0.]))
        bound = self.output.weight.detach().abs().sum(-1)
        assert bool((self.output.bias[0] - bound[0] > self.output.bias[1:] + bound[1:]).all())

    def forward(self, source_states, qualitative_logits, geometry):
        if source_states.shape[:-1] != geometry.shape[:-1] or geometry.shape[-1] != 8:
            raise ValueError('completion states and actual proposed geometry must align')
        if set(qualitative_logits) != set(ATTRIBUTES):
            raise ValueError('completion requires all actual predicted qualitative distributions')
        probabilities = torch.cat([qualitative_logits[key].float().softmax(-1) for key in ATTRIBUTES], -1)
        value = self.hidden(source_states.float()) + self.attributes(probabilities) + self.geometry(geometry.float())
        return self.output(value.tanh())


def _wrap(angle):
    return (float(angle) + 180.) % 360. - 180.


def _source_events(traces):
    events = [event for event in traces if event['field'] != 'transcript']
    if [event['generated_source_slot'] for event in events] != list(range(len(events))):
        raise ValueError('completion requires the actual ordered EVENT proposal slots')
    return events


def completion_geometry(plan):
    """Proposal features; no annotation, target-plan or executed-audio input."""
    rows = []
    for source in plan['sources']:
        trajectory = source['trajectory']
        if trajectory['type'] == 'static':
            first = last = trajectory['position']
        elif trajectory['type'] == 'linear':
            first, last = trajectory['start'], trajectory['end']
        else:
            raise ValueError('completion scope is the current static/linear EVENT planner')
        a, b = math.radians(first['azimuth_deg']), math.radians(last['azimuth_deg'])
        rows.append([math.sin(a), math.cos(a), math.sin(b), math.cos(b),
            source['activity']['onset_sec'] / plan['duration_sec'],
            source['activity']['offset_sec'] / plan['duration_sec'],
            math.log(first['distance_m']), math.log(last['distance_m'])])
    return torch.tensor(rows, dtype=torch.float32)


def completion_candidates(codec, plan, traces, source_slot):
    """Angular alternatives bounded by the student's predicted compass cells.

    Request admissibility is a separate training/evaluation-side check. This
    function must never be given those labels as a replacement for predictions.
    """
    events = _source_events(traces)
    if len(events) != len(plan['sources']) or not 0 <= source_slot < len(events):
        raise ValueError('completion decision and actual proposal source differ')
    labels = events[source_slot]['qualitative_control']['labels']
    original = plan['sources'][source_slot]['trajectory']
    if original['type'] != labels['motion']:
        raise ValueError('proposal trajectory and its predicted motion differ')
    points = [('position', 'start')] if original['type'] == 'static' else [('start', 'start'), ('end', 'end')]
    candidates = [copy.deepcopy(plan)]
    for offset in ANGLE_OFFSETS[1:]:
        candidate = copy.deepcopy(plan)
        trajectory = candidate['sources'][source_slot]['trajectory']
        for point, attribute in points:
            center = CENTERS[ATTRIBUTES[attribute].index(labels[attribute])]
            delta = _wrap(original[point]['azimuth_deg'] - center)
            if abs(delta) > CATEGORY_HALF_WIDTH:
                raise ValueError('base completion is already outside its predicted category')
            value = max(-CATEGORY_HALF_WIDTH, min(CATEGORY_HALF_WIDTH, delta + offset))
            trajectory[point]['azimuth_deg'] = _wrap(center + value)
        projected = codec.project_plan(candidate)
        # Numeric completion cannot change text, duration, source binding or timing.
        protected_before, protected_after = copy.deepcopy(plan), copy.deepcopy(projected)
        protected_before['sources'][source_slot].pop('trajectory')
        protected_after['sources'][source_slot].pop('trajectory')
        if protected_before != protected_after:
            raise ValueError('codec projection changed a protected proposal field')
        candidates.append(projected)
    legal = []
    seen = set()
    import json
    for candidate in candidates:
        signature = json.dumps(candidate, sort_keys=True)
        legal.append(signature not in seen)
        seen.add(signature)
    return tuple(candidates), torch.tensor(legal, dtype=torch.bool)


def apply_completion_actions(codec, plan, traces, actions):
    if len(actions) != len(plan['sources']):
        raise ValueError('one explicit completion action is required per actual source')
    result = copy.deepcopy(plan)
    for slot, action in enumerate(actions):
        candidates, legal = completion_candidates(codec, plan, traces, slot)
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(candidates) or not legal[action]:
            raise ValueError('completion action outside its distinct legal support')
        result['sources'][slot] = copy.deepcopy(candidates[action]['sources'][slot])
    return result
