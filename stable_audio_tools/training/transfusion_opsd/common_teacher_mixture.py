"""Conditional repair supervision weighted by a frozen plan-teacher marginal.

This generic helper knows neither audio metrics nor a Transfusion architecture.
It keeps the probability mass of outcomes without positive repairs in the
query denominator; it does not renormalize a few surviving repair targets.
"""
import math

import torch


def best_positive_contexts(selection):
    result = []
    for action, chosen in enumerate(selection['best_per_plan']):
        if chosen is None:
            continue
        trial = chosen['trial']
        if type(trial) is not int or trial < 0 or not chosen['eligible']:
            raise ValueError('each selected plan variant must be eligible and indexed')
        if trial:
            evidence = selection['trials'][action][trial]
            if not evidence['eligible'] or evidence['trial'] != trial:
                raise ValueError('positive context must retain its verified selection evidence')
            result.append((action, trial))
    return tuple(result)


def positive_context_mixture(selection, probabilities, *, legal):
    """Use the same frozen AR teacher probabilities for DiT positive targets.

    The caller still verifies target provenance and actual loss correspondence.
    Unrepaired probability mass supplies no positive-repair loss. Capability
    preservation is enforced separately; this is not a full generative KL.
    """
    q = torch.as_tensor(probabilities).detach().double().cpu()
    allowed = torch.as_tensor(legal, dtype=torch.bool).cpu()
    if (q.ndim != 1 or q.shape != allowed.shape
            or len(selection['best_per_plan']) != len(q)
            or not torch.isfinite(q).all() or (q < 0).any()
            or not math.isclose(float(q.sum()), 1., abs_tol=1e-6, rel_tol=0.)
            or (q[~allowed] != 0).any()):
        raise ValueError('a finite normalized legal plan-teacher marginal is required')
    terms = [dict(action=action, trial=trial, weight=float(q[action]))
        for action, trial in best_positive_contexts(selection) if q[action] > 0]
    mass = sum(term['weight'] for term in terms)
    return terms, dict(mode='ar_teacher_positive_marginal_v1', positive_probability_mass=mass,
        unrepaired_probability_mass=max(0., 1. - mass), renormalized_positive_targets=False,
        future_outcomes_in_student_inputs=False)
