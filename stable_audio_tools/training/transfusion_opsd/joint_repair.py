"""Bounded repair proposals and equal-budget cross-plan teacher selection.

The generic core consumes verified execution scores. Task adapters must run
the actual continuation and certify each proposed positive target first.
"""
from __future__ import annotations

import math

import torch

from .objectives import valid_mask


def propose_bounded_repair(anchor, mask, *, objective, absolute_budget, steps):
    """A detached local proposal; the caller supplies one common plan budget.

    Preserve the exact anchor outside valid frames, including at zero repair.
    A proposal is not qualified by this function.
    """
    if not math.isfinite(absolute_budget) or absolute_budget <= 0 or type(steps) is not int or steps < 1:
        raise ValueError('positive finite absolute budget and integer steps required')
    full = valid_mask(anchor, mask)
    if not torch.isfinite(anchor).all():
        raise ValueError('finite anchor required')
    base = anchor.detach().float()
    value = base.clone()
    for _ in range(steps):
        with torch.enable_grad():
            leaf = value.detach().requires_grad_(True)
            utility = objective(leaf)
            if utility.numel() != 1 or not torch.isfinite(utility).all():
                raise ValueError('finite scalar repair objective required')
            gradient, = torch.autograd.grad(utility.sum(), leaf)
        gradient = gradient.detach().masked_fill(~full, 0)
        if not torch.isfinite(gradient).all():
            raise ValueError('non-finite repair gradient')
        step = (absolute_budget / steps) * gradient / gradient.norm().clamp_min(1e-12)
        delta = (value + step - base).masked_fill(~full, 0)
        value = (base + delta * (absolute_budget / delta.norm().clamp_min(1e-12)).clamp(max=1)).detach()
    return value


def select_joint_repair_teacher(variants, *, legal, tolerances, minimum_gain,
        semantic_tolerance, repair_penalty):
    """Select plan + certified repair relative to the equally repaired original.

    variants[action][trial] has score (RewardScore), certified (bool), and
    repair_fraction in [0,1]. Trial zero is the unmodified plan outcome.
    Utility and cost meanings belong to the task adapter. Here utility is
    maximized; protection costs are minimized.
    """
    if len(variants) != len(legal) or not legal or not legal[0] or not variants[0]:
        raise ValueError('complete support and a legal original plan required')
    if any(not math.isfinite(x) or x < 0 for x in [semantic_tolerance, repair_penalty, *tolerances.values()]) or not math.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError('finite protection, repair cost and positive improvement threshold required')
    original = variants[0][0]['score']
    if variants[0][0]['repair_fraction'] != 0 or not variants[0][0]['certified']:
        raise ValueError('unmodified original must be the zero-cost certified reference')

    def protected(score, reference):
        if score.costs.keys() != reference.costs.keys() or score.costs.keys() != tolerances.keys():
            raise ValueError('repair selection cannot drop protection coverage')
        return all(score.costs[k] <= reference.costs[k] + tol + 1e-12 for k, tol in tolerances.items())

    chosen, evidence = [], []
    for action, trials in enumerate(variants):
        if legal[action] and (not trials or trials[0]['repair_fraction'] != 0 or not trials[0]['certified']):
            raise ValueError('every legal plan requires its unmodified zero-cost outcome')
        checked = []
        for index, trial in enumerate(trials):
            fraction, score = trial['repair_fraction'], trial['score']
            if not math.isfinite(fraction) or not 0 <= fraction <= 1 + 1e-6:
                raise ValueError('every repair must fit the common absolute budget')
            okay = protected(score, original)
            eligible = bool(legal[action] and trial['certified'] and okay
                and score.utility >= original.utility - semantic_tolerance)
            checked.append(dict(trial=index, eligible=eligible,
                utility=score.utility, repair_fraction=fraction,
                penalized_utility=score.utility - repair_penalty * fraction ** 2))
        feasible = [x for x in checked if x['eligible']]
        best = max(feasible, key=lambda x: (x['penalized_utility'], -x['repair_fraction'], -x['trial'])) if feasible else None
        chosen.append(best)
        evidence.append(checked)
    reference = chosen[0]
    if reference is None:
        raise ValueError('original no-repair reference must remain available')
    ref_score = variants[0][reference['trial']]['score']
    candidates = []
    for action, choice in enumerate(chosen):
        if not action or choice is None:
            continue
        score = variants[action][choice['trial']]['score']
        gain = score.utility - ref_score.utility
        credit = choice['penalized_utility'] - reference['penalized_utility']
        if protected(score, ref_score) and gain > minimum_gain and credit > minimum_gain:
            candidates.append(dict(action=action, semantic_gain=gain, ar_credit=credit, trial=choice['trial']))
    winner = max(candidates, key=lambda x: (x['ar_credit'], x['semantic_gain'], -x['action'])) if candidates else None
    action = winner['action'] if winner else 0
    return dict(action=action, trial=chosen[action]['trial'], qualified=winner is not None,
        baseline_repair_trial=reference['trial'], best_per_plan=chosen, trials=evidence,
        improving_plans=candidates, ar_credit=winner['ar_credit'] if winner else 0.,
        baseline_gets_equal_repair_budget=True)
