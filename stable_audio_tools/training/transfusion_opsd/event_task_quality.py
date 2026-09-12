"""Semantic improvement inside the request's allowed spatial/time region.

This is an explicit experiment profile. It does not reinterpret previous
spatial-utility results or turn CLAP similarity into absolute correctness.
"""
import math

from .objectives import RewardScore


def semantic_similarity(score):
    values = [value for key, value in score.costs.items() if key.startswith('clap_content_cost/')]
    if not values:
        raise ValueError('semantic scoring must cover the requested source')
    return 1. - sum(values) / len(values)


def request_quality_change(before, after, *, tolerances):
    if before.costs.keys() != after.costs.keys():
        raise ValueError('request quality comparison cannot drop protection coverage')
    changes = {key: after.costs[key] - value for key, value in before.costs.items()
        if not key.startswith('clap_content_cost/')}
    if changes.keys() != tolerances.keys() or any(not math.isfinite(x) or x < 0 for x in tolerances.values()):
        raise ValueError('declare a finite tolerance for every spatial/time/ASR protection')
    return {'semantic_gain': semantic_similarity(after) - semantic_similarity(before),
        'cost_changes': changes,
        'protected': all(value <= tolerances[key] + 1e-12 for key, value in changes.items())}


def select_query_teacher(scores, *, legal, tolerances, minimum_semantic_gain=.01, baseline_action=0):
    """Training-side execution queries; keep the original action if none helps."""
    if (len(scores) != len(legal) or type(baseline_action) is not int
            or not 0 <= baseline_action < len(legal) or not legal[baseline_action]
            or not math.isfinite(minimum_semantic_gain) or minimum_semantic_gain <= 0):
        raise ValueError('query teacher requires the actual native action and an explicit meaningful gain')
    if any(value is None for value in scores):
        return {'action': baseline_action, 'qualified': False, 'reason': 'unobservable_query_kept_in_denominator'}
    evidence = {action: request_quality_change(scores[baseline_action], score, tolerances=tolerances)
        for action, score in enumerate(scores) if legal[action]}
    feasible = [action for action, value in evidence.items()
        if action != baseline_action and value['protected'] and value['semantic_gain'] > minimum_semantic_gain]
    selected = max(feasible, key=lambda action: (evidence[action]['semantic_gain'], -action)) if feasible else baseline_action
    return {'action': selected, 'qualified': bool(feasible), 'baseline_action': baseline_action, 'candidates': evidence}


class SemanticWithinRequestReward:
    """Semantic utility and explicit slack measured against native reference.

    Costs are violations of fixed reference limits. A local target cannot use
    its own output to move those limits, or earn a gain by losing a constraint.
    Keep raw scores separately in collection and all actual-output reports.
    """
    profile = 'event_semantic_within_requested_region_v1_experimental'

    def __init__(self, raw_reward, reference_score, *, tolerances):
        request_quality_change(reference_score, reference_score, tolerances=tolerances)
        self.raw_reward = raw_reward
        self.limits = {key: reference_score.costs[key] + tolerance for key, tolerance in tolerances.items()}

    def __call__(self, audio):
        raw = self.raw_reward(audio)
        return RewardScore(semantic_similarity(raw),
            {key: max(0., raw.costs[key] - limit) for key, limit in self.limits.items()})
