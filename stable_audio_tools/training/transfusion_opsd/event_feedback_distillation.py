"""Fixed execution-side categorical targets, separate from student features."""
import math

import torch

from .event_task_quality import select_query_teacher


def query_teacher_distribution(scores, *, legal, initial_logits, tolerances,
        objective, minimum_semantic_gain=.01, temperature=.01, smoothing=1e-4,
        baseline_action=0, neutral_query_target='behavior', neutral_semantic_tolerance=.005):
    """Preserve the actual action when complete protected improvement is absent.

    An unobservable query returns no target. Callers must still count it in
    their original query denominator. All scores and logits are detached
    teacher information; they must never be concatenated to student features.
    """
    logits = torch.as_tensor(initial_logits).detach().double()
    allowed = torch.as_tensor(legal, device=logits.device, dtype=torch.bool)
    if logits.ndim != 1 or allowed.shape != logits.shape or not bool(torch.isfinite(logits[allowed]).all()):
        raise ValueError('a query target needs finite initial logits on its actual legal support')
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= smoothing < 1:
        raise ValueError('declare a finite positive temperature and valid smoothing')
    if objective not in ('winner_cross_entropy', 'protected_soft_teacher'):
        raise ValueError('unknown fixed execution teacher objective')
    if neutral_query_target not in ('behavior', 'initial_policy', 'protected_policy'):
        raise ValueError('declare behavior, initial_policy or protected_policy for observable neutral queries')
    if not math.isfinite(neutral_semantic_tolerance) or neutral_semantic_tolerance < 0:
        raise ValueError('declare a finite nonnegative neutral semantic tolerance')
    teacher = select_query_teacher(scores, legal=allowed.tolist(), tolerances=tolerances,
        minimum_semantic_gain=minimum_semantic_gain, baseline_action=baseline_action)
    if any(score is None for score in scores):
        return None
    if not teacher['qualified'] and neutral_query_target in ('initial_policy', 'protected_policy'):
        # No observed improvement is not new evidence that the greedy action
        # deserves probability one. Keep the detached current distribution;
        # its initial KL gradient is zero, and later drift is still penalized.
        # No smoothing: that would itself introduce a nonzero initial target.
        support = allowed.clone()
        if neutral_query_target == 'protected_policy':
            # Observing no useful improvement may still reveal harmful
            # alternatives. Retain relative probabilities only among the
            # candidates that satisfy the already declared task tolerances.
            for action, evidence in teacher['candidates'].items():
                support[action] = evidence['protected'] and evidence['semantic_gain'] >= -neutral_semantic_tolerance
        return logits.masked_fill(~support, -torch.inf).softmax(-1)
    if objective == 'winner_cross_entropy':
        target = torch.zeros_like(logits)
        alternatives = allowed.clone()
        alternatives[teacher['action']] = False
        if bool(alternatives.any()):
            target[alternatives] = smoothing / int(alternatives.sum())
            target[teacher['action']] = 1 - smoothing
        else:
            target[teacher['action']] = 1.
        return target
    support = torch.zeros_like(allowed)
    support[baseline_action] = True
    gains = torch.zeros_like(logits)
    for action, evidence in teacher['candidates'].items():
        if evidence['protected'] and evidence['semantic_gain'] > minimum_semantic_gain:
            support[action] = True
            gains[action] = evidence['semantic_gain']
    weights = (logits + gains / temperature).masked_fill(~support, -torch.inf)
    target = weights.softmax(-1)
    return (1 - smoothing) * target + smoothing * allowed / allowed.sum()
