"""Keep released capability and the actual pre-update policy comparable."""
import copy
import math

from .event_task_quality import request_quality_change, semantic_similarity
from .greedy_execution_teacher import paired_execution_change
from .objectives import RewardScore


FEEDBACK_BRANCHES = ('A0_Dnew', 'Anew_D0', 'Anew_Dnew')


class PreupdateBranchReferences:
    """Keep each comparison paired with its own pre-update executor branch.

    A0 denotes the fixed original plan here. After a resumed DiT update,
    Aprev/D0 and Aprev/Dprev need different references even if their AR
    decisions happen to match. References cannot be replaced during fitting.
    """

    def __init__(self):
        self._values = {}

    @staticmethod
    def _key(branch, sample_id, seeds):
        if branch not in FEEDBACK_BRANCHES or not isinstance(sample_id, str) or not sample_id:
            raise ValueError('declare a known execution branch and request')
        seeds = tuple(seeds)
        if len(seeds) < 2 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError('branch references require distinct aligned noise pairs')
        return (branch, sample_id), seeds

    def capture(self, branch, sample_id, seeds, scores):
        key, seeds = self._key(branch, sample_id, seeds)
        if key in self._values:
            raise ValueError('pre-update branch references cannot be reset during fitting')
        if len(scores) != len(seeds):
            raise ValueError('branch reference scores must cover every declared noise')
        self._values[key] = seeds, copy.deepcopy(tuple(scores))

    def lookup(self, branch, sample_id, seeds):
        key, seeds = self._key(branch, sample_id, seeds)
        if key not in self._values or self._values[key][0] != seeds:
            raise ValueError('the matching pre-update branch/noise reference is missing')
        return copy.deepcopy(self._values[key][1])


def compare_feedback_guard(reference, candidate, *, tolerances, semantic_tolerance,
        initial_policy=None):
    """Paired scores share the same frozen reward reference and noise order.

    A valid native proposal can differ between execution contexts. Comparing
    with both references prevents pre-existing advantages from being counted
    as learning, or a loss relative to the initial policy from being hidden
    by a weaker released-plan reference. This is still a finite-sample guard.
    """
    if not math.isfinite(semantic_tolerance) or semantic_tolerance < 0:
        raise ValueError('declare a finite nonnegative semantic tolerance')

    def quality(score):
        return RewardScore(semantic_similarity(score), {key: value for key, value in score.costs.items()
            if not key.startswith('clap_content_cost/')})

    def compare(before):
        if len(before) != len(candidate) or len(before) < 2:
            raise ValueError('audio guards require at least two aligned noise pairs')
        for old, new in zip(before, candidate):
            request_quality_change(old, new, tolerances=tolerances)
        change = paired_execution_change([quality(score) for score in before],
            [quality(score) for score in candidate], require_gain=False)
        protected = (change['utility_gain']['mean'] >= -semantic_tolerance
            and all(value['mean'] <= tolerances[key]+1e-12 for key, value in change['cost_changes'].items()))
        return change, protected

    comparison, protected_reference = compare(reference)
    initial_comparison, protected_initial = (compare(initial_policy) if initial_policy is not None else (None, True))
    return {'comparison': comparison, 'initial_policy_comparison': initial_comparison,
        'protected_reference': protected_reference, 'protected_initial_policy': protected_initial,
        'protected': protected_reference and protected_initial}
