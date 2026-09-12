"""Request-checkable teacher queries around an actual EVENT planning proposal.

This is a candidate catalog, not a claim that the current three-action student
already predicts these seven choices. A learned policy and native-greedy
improvement checks must precede training/promotion of the expanded catalog.
"""
from __future__ import annotations

import copy
import json

import torch

from ...models.sceneplan_generation_ar_qualitative_head import ATTRIBUTES, CENTERS
from .event_completion import CATEGORY_HALF_WIDTH, completion_candidates

REFINEMENT_ACTIONS = ('baseline', 'angle_minus_10', 'angle_plus_10',
    'angle_minus_5', 'angle_plus_5', 'distance_times_0p8', 'distance_times_1p25')


def refinement_candidates(codec, plan, traces, source_slot):
    """Keep source identity/text/activity/duration and predicted compass cells.

The student proposal supplies every geometric value. No requirement or target
plan is read here; requested constraints are checked independently per query.
Both endpoints receive the same distance scale, preserving relative radial
motion before codec projection. The external verifier checks the final plan.
    """
    initial, _ = completion_candidates(codec, plan, traces, source_slot)
    values = list(initial)
    events = [trace for trace in traces if trace['field'] != 'transcript']
    labels = events[source_slot]['qualitative_control']['labels']
    source = plan['sources'][source_slot]
    trajectory = source['trajectory']
    points = [('position', 'start')] if trajectory['type'] == 'static' else [('start', 'start'), ('end', 'end')]
    wrap = lambda value: (float(value) + 180.) % 360. - 180.
    for offset in [-5., 5.]:
        candidate = copy.deepcopy(plan)
        target = candidate['sources'][source_slot]['trajectory']
        for point, attribute in points:
            center = CENTERS[ATTRIBUTES[attribute].index(labels[attribute])]
            delta = wrap(trajectory[point]['azimuth_deg'] - center)
            target[point]['azimuth_deg'] = wrap(center + min(CATEGORY_HALF_WIDTH, max(-CATEGORY_HALF_WIDTH, delta + offset)))
        values.append(codec.project_plan(candidate))
    for multiplier in [.8, 1.25]:
        candidate = copy.deepcopy(plan)
        target = candidate['sources'][source_slot]['trajectory']
        for point, _ in points:
            target[point]['distance_m'] *= multiplier
        values.append(codec.project_plan(candidate))
    signatures, legal = set(), []
    protected = copy.deepcopy(plan)
    protected['sources'][source_slot].pop('trajectory')
    for candidate in values:
        other = copy.deepcopy(candidate)
        other['sources'][source_slot].pop('trajectory')
        if other != protected:
            raise ValueError('a refinement query changed protected planning fields')
        signature = json.dumps(candidate, sort_keys=True)
        legal.append(signature not in signatures)
        signatures.add(signature)
    assert len(values) == len(REFINEMENT_ACTIONS)
    return tuple(values), torch.tensor(legal, dtype=torch.bool)
