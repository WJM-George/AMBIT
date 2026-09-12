"""Finite execution queries against the policy's actual greedy action.

These checks constrain empirical means over declared paired noises. They are
neither per-output guarantees nor selection-corrected population bounds.
"""
from __future__ import annotations

import math

import numpy as np

from .expected_execution_teacher import _summary


def paired_execution_change(before, after, *, min_gain=1e-6, require_gain=True):
    if len(before) != len(after) or len(before) < 2:
        raise ValueError('execution comparison requires at least two aligned noise pairs')
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError('minimum gain must be finite and nonnegative')
    keys = before[0].costs.keys()
    if any(score.costs.keys() != keys for score in [*before, *after]):
        raise ValueError('every execution must retain exactly the same protection metrics')
    gain = _summary(np.asarray([b.utility - a.utility for a, b in zip(before, after)]))
    costs = {key: _summary(np.asarray([b.costs[key] - a.costs[key]
        for a, b in zip(before, after)])) for key in keys}
    utility_okay = gain['mean'] > min_gain if require_gain else gain['mean'] >= -1e-12
    return {'utility_gain': gain, 'cost_changes': costs,
        'empirical_protected_gain': bool(utility_okay and all(v['mean'] <= 1e-12 for v in costs.values())),
        'requires_positive_gain': require_gain,
        'meaning': 'paired empirical means; per-noise changes and descriptive intervals retained'}


def select_greedy_execution_teacher(construction, validation, *, legal, baseline_action, min_gain=1e-6):
    legal = tuple(bool(value) for value in legal)
    if (not legal or not 0 <= baseline_action < len(legal) or not legal[baseline_action]
            or any(len(row) != len(legal) for row in [*construction, *validation])):
        raise ValueError('finite execution support must contain the actual baseline action')
    # Inspect construction first. Validation cannot choose or switch actions.
    actions = {}
    for action, allowed in enumerate(legal):
        if allowed and action != baseline_action:
            actions[action] = paired_execution_change([row[baseline_action] for row in construction],
                [row[action] for row in construction], min_gain=min_gain)
    feasible = [a for a, evidence in actions.items() if evidence['empirical_protected_gain']]
    chosen = max(feasible, key=lambda a: (actions[a]['utility_gain']['mean'], -a)) if feasible else None
    check = None if chosen is None else paired_execution_change([row[baseline_action] for row in validation],
        [row[chosen] for row in validation], min_gain=min_gain)
    return {'baseline_action': baseline_action, 'chosen_on_construction': chosen,
        'qualified': chosen is not None and check['empirical_protected_gain'],
        'construction': actions, 'validation_of_selected_action': check,
        'selection': 'highest construction gain among feasible actions, then validate without reselection'}


def qualify_fixed_execution_teacher(before, after, *, first_indices, second_indices, min_gain=1e-6):
    """Validate a previously fixed action on two disjoint complete noise groups.

    This function never selects an action. Missing observations cannot silently
    reduce the denominator, and every group must satisfy the same protection.
    """
    first, second = tuple(first_indices), tuple(second_indices)
    if (len(before) != len(after) or min(len(first), len(second)) < 2
            or sorted(first + second) != list(range(len(before)))):
        raise ValueError('two disjoint noise groups must cover every recorded pair exactly once')
    if any(value is None for value in [*before, *after]):
        return {'qualified': False, 'reason': 'missing_or_unobservable_pair_kept_in_denominator'}
    checks = {name: paired_execution_change([before[i] for i in indices],
        [after[i] for i in indices], min_gain=min_gain)
        for name, indices in [('pooled', range(len(before))), ('half_0', first), ('half_1', second)]}
    return {'qualified': all(value['empirical_protected_gain'] for value in checks.values()), 'checks': checks,
        'selection': 'action fixed before these noises; no reselection',
        'meaning': 'finite paired empirical means, not a population guarantee'}
