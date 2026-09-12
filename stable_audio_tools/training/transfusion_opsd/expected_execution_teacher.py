"""Noise-marginal execution supervision on a finite actual AR decision.

The fitting target solves an empirical expected-utility problem. Independent
paired noises check expected utility and EACH expected cost separately. This
is not the v1 every-noise guarantee and not a population guarantee; intervals
and per-noise regressions are always reported. Actual parameter updates need
their own execution guards and an independent final evaluation.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def _arrays(rows, count):
    if len(rows) < 2 or any(len(row) != count for row in rows):
        raise ValueError('at least two complete paired execution noises are required')
    keys = sorted(rows[0][0].costs)
    if any(sorted(score.costs) != keys for row in rows for score in row):
        raise ValueError('execution protection coverage differs across actions or noises')
    utility = np.asarray([[score.utility for score in row] for row in rows], dtype=np.float64)
    costs = {key: np.asarray([[score.costs[key] for score in row] for row in rows], dtype=np.float64) for key in keys}
    if not np.isfinite(utility).all() or any(not np.isfinite(values).all() for values in costs.values()):
        raise ValueError('execution evidence must be finite')
    return utility, costs


def _summary(values):
    from scipy.stats import t
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    radius = float(t.ppf(.975, len(values) - 1)) * se
    return {'mean': mean, 'paired_noise_count': len(values), 'per_noise': values.tolist(),
        'descriptive_t_interval_95': [mean - radius, mean + radius],
        'interval_is_not_selection_or_multiple_comparison_corrected': True}


def expected_execution_teacher(logits, legal, construction, *, temperature=.1, strength=.5, min_gain=1e-6):
    from scipy.optimize import linprog, minimize
    if (logits.ndim != 1 or legal.shape != logits.shape or legal.dtype != torch.bool
            or int(legal.sum()) < 2 or not 0 < strength < 1
            or not math.isfinite(temperature) or temperature <= 0 or min_gain <= 0):
        raise ValueError('invalid finite execution teacher support or parameters')
    u, costs = _arrays(construction, logits.numel())
    legal = legal.to(logits.device)
    base = logits.detach().double().masked_fill(~legal, -torch.inf).softmax(-1)
    if not torch.isfinite(base).all() or not (base[legal] > 0).all():
        raise ValueError('finite positive probabilities are required on legal actions')
    ids = legal.nonzero().flatten().tolist()
    p = base.cpu().numpy()[ids]
    means = u.mean(0)[ids]
    tilt = (base.log() + torch.tensor(u.mean(0), device=base.device) / temperature).softmax(-1)
    reference = ((1 - strength) * base + strength * tilt).cpu().numpy()[ids]
    gain = max(2 * min_gain, min_gain + 1e-9)
    cost_means = np.asarray([value.mean(0)[ids] for value in costs.values()], dtype=np.float64).reshape(-1, len(ids))
    variable_costs = cost_means[np.ptp(cost_means, axis=1) > 1e-12]
    matrix = np.vstack((-means[None], variable_costs))
    rhs = np.r_[-means @ p - gain, variable_costs @ p]
    center = matrix.mean(1)
    scale = np.maximum(np.max(np.abs(matrix - center[:, None]), axis=1), 1e-8)
    matrix, rhs = (matrix - center[:, None]) / scale[:, None], (rhs - center) / scale
    lower = (1 - strength) * p
    bounds = [(float(value), 1.) for value in lower]
    feasible = linprog(np.zeros(len(ids)), A_ub=matrix, b_ub=rhs,
        A_eq=np.ones((1, len(ids))), b_eq=[1.], bounds=bounds, method='highs',
        options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    evidence = {'contract': 'paired_noise_marginal_execution_teacher_v2',
        'construction_noises': len(construction), 'validation_used_in_solver': False,
        'aggregation': 'expected utility and every expected cost over paired construction noises',
        'every_noise_protection_claimed': False, 'population_guarantee_claimed': False,
        'feasibility_status': feasible.message, 'temperature': temperature, 'strength': strength}
    if not feasible.success:
        return None, base, None, {**evidence, 'reason': 'no_feasible_expected_teacher'}
    fit = minimize(lambda q: float(np.sum(q * np.log(np.maximum(q, 1e-300) / reference))), feasible.x,
        jac=lambda q: np.log(np.maximum(q, 1e-300) / reference) + 1., method='SLSQP', bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda q: q.sum() - 1., 'jac': lambda q: np.ones_like(q)},
            {'type': 'ineq', 'fun': lambda q: rhs - matrix @ q, 'jac': lambda q: -matrix}],
        options={'ftol': 1e-13, 'maxiter': 200})
    evidence.update(solver_status=fit.message, solver_iterations=int(fit.nit))
    q = np.asarray(fit.x)
    if (not fit.success or abs(q.sum() - 1.) > 1e-10 or (q < lower - 1e-12).any()
            or means @ (q - p) < gain - 1e-12 or (cost_means @ (q - p) > 1e-12).any()):
        return None, base, None, {**evidence, 'reason': 'expected_teacher_numerical_check_failed'}
    target = torch.zeros_like(base)
    target[ids] = torch.as_tensor(q, dtype=base.dtype, device=base.device)
    teacher = logits.detach().double().clone()
    teacher[legal] = target[legal].log()
    target = teacher.masked_fill(~legal, -torch.inf).softmax(-1)
    return teacher, base, target, {**evidence, 'reason': 'expected_construction_teacher_built'}


def qualify_expected_teacher(base, target, construction, validation, *, min_gain=1e-6):
    if base.ndim != 1 or target.shape != base.shape or len(validation) < 2:
        raise ValueError('independent complete paired validation is required')
    for probabilities in (base, target):
        if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
                or abs(float(probabilities.sum()) - 1.) > 1e-10):
            raise ValueError('teacher probabilities must be finite and normalized')
    delta = (target.detach().double() - base.detach().double()).cpu().numpy()
    evidence = {'contract': 'empirical_paired_expectation_check_v2', 'splits': {},
        'every_noise_protection_claimed': False, 'population_guarantee_claimed': False}
    okay, keys = True, None
    for split, rows in [('construction', construction), ('validation', validation)]:
        utility, costs = _arrays(rows, len(delta))
        if keys is not None and set(costs) != keys:
            raise ValueError('validation omitted a construction cost')
        keys = set(costs)
        gain = _summary(utility @ delta)
        changes = {key: _summary(values @ delta) for key, values in costs.items()}
        accepted = gain['mean'] > min_gain and all(value['mean'] <= 1e-12 for value in changes.values())
        evidence['splits'][split] = {'gain': gain, 'cost_changes': changes, 'accepted': accepted,
            'per_noise_cost_regressions': {key: int((np.asarray(value['per_noise']) > 1e-12).sum()) for key, value in changes.items()}}
        okay &= accepted
    evidence['reason'] = 'independent_expected_teacher_qualified' if okay else 'expected_gain_or_cost_not_reproduced'
    return bool(okay), evidence
