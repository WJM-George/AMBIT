"""Plan distillation from measured raw or equally repaired execution outcomes."""
import math

import torch

from .joint_repair import select_joint_repair_teacher


def repair_teacher_distribution(variants, *, legal, initial_logits, mode, tolerances,
        minimum_gain=.01, temperature=.01, smoothing=1e-4,
        semantic_tolerance=.005, repair_penalty=.01):
    """Matched split/joint targets; future outcome values stay training-side.

    The split control sees raw plan outcomes. The joint target sees each plan's
    certified cost-adjusted outcome and compares to an equally repaired original.
    Both use the same prior, protection rules and soft-target construction.
    """
    if mode not in ('raw_plan', 'joint_repair'):
        raise ValueError('declare raw_plan or joint_repair supervision')
    logits = torch.as_tensor(initial_logits).detach().double()
    allowed = torch.as_tensor(legal, device=logits.device, dtype=torch.bool)
    if logits.ndim != 1 or logits.shape != allowed.shape or len(variants) != len(logits):
        raise ValueError('teacher observations must align with the actual action support')
    if not bool(torch.isfinite(logits[allowed]).all()) or not math.isfinite(temperature) or temperature <= 0 or not 0 <= smoothing < 1:
        raise ValueError('finite prior and valid temperature/smoothing required')
    selection = select_joint_repair_teacher(variants, legal=allowed.tolist(), tolerances=tolerances,
        minimum_gain=minimum_gain, semantic_tolerance=semantic_tolerance, repair_penalty=repair_penalty)
    choices = [0] * len(variants) if mode == 'raw_plan' else [
        choice['trial'] if choice is not None else 0 for choice in selection['best_per_plan']]
    original = variants[0][0]['score']
    reference = variants[0][choices[0]]
    penalty = 0. if mode == 'raw_plan' else repair_penalty
    reference_value = reference['score'].utility - penalty * reference['repair_fraction'] ** 2
    safe, improved = torch.zeros_like(allowed), torch.zeros_like(allowed)
    improved[0] = True
    gains = torch.zeros_like(logits)
    evidence = []
    for action, available in enumerate(allowed.tolist()):
        if not available:
            evidence.append(None)
            continue
        trial = variants[action][choices[action]]
        score = trial['score']
        protected = all(score.costs[k] <= min(original.costs[k], reference['score'].costs[k]) + tol + 1e-12
            for k, tol in tolerances.items())
        eligible = mode == 'raw_plan' or selection['best_per_plan'][action] is not None
        semantic_gain = score.utility - reference['score'].utility
        value_gain = score.utility - penalty * trial['repair_fraction'] ** 2 - reference_value
        safe[action] = eligible and protected and semantic_gain >= -semantic_tolerance and value_gain >= -semantic_tolerance
        qualified = eligible and protected and semantic_gain > minimum_gain and value_gain > minimum_gain
        improved[action] = qualified or action == 0
        if qualified:
            gains[action] = value_gain
        evidence.append(dict(action=action, trial=choices[action], protected=protected,
            eligible=eligible, semantic_gain=semantic_gain, value_gain=value_gain, qualified=qualified))
    qualified = bool(improved[1:].any())
    if qualified:
        target = (logits + gains / temperature).masked_fill(~improved, -torch.inf).softmax(-1)
        target = (1 - smoothing) * target + smoothing * allowed / allowed.sum()
    else:
        target = logits.masked_fill(~safe, -torch.inf).softmax(-1)
    if not bool(torch.isfinite(target).all()) or not bool(safe[0]):
        raise ValueError('the original plan must remain a finite protected reference')
    return target, dict(mode=mode, qualified=qualified, original_repair_trial=choices[0],
        candidates=evidence, selected_joint_teacher=selection,
        target_kind='improvement' if qualified else 'protected_distribution_retention',
        outcome_values_are_student_inputs=False)
