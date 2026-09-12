"""Construct an execution teacher inside measured request/content constraints.

Only construction seeds enter this finite convex problem. Qualification on
independent seeds remains mandatory and may reject the resulting teacher.
"""
from __future__ import annotations

import numpy as np
import torch

def constrained_event_teacher(logits, legal, construction, *, ar_temperature=1.,
        temperature=.1, strength=.5, min_gain=1e-6):
    from scipy.optimize import linprog, minimize
    if logits.ndim != 1 or legal.shape != logits.shape or len(construction) < 2:
        raise ValueError('EVENT teacher requires one actual decision and independent construction noises')
    count = logits.numel()
    if any(len(row) != count for row in construction):
        raise ValueError('every construction noise must measure the complete finite decision support')
    keys = set(construction[0][0].costs)
    if any(set(score.costs) != keys for row in construction for score in row):
        raise ValueError('teacher protection coverage changed across actions or seeds')
    ids = legal.nonzero().flatten().tolist()
    if len(ids) < 2 or not 0 < strength < 1 or min_gain <= 0:
        raise ValueError('protected teacher requires two legal choices and a positive bounded mixture/gain')
    utilities = [[score.utility for score in row] for row in construction]
    means = np.mean(utilities, axis=0).tolist()
    if min(ar_temperature, temperature) <= 0 or not np.isfinite([ar_temperature, temperature]).all():
        raise ValueError('teacher temperatures must be finite and positive')
    # The EVENT completion loss and sampling stage use this same FP64 finite
    # distribution. Avoid certifying ideal probabilities then fitting rounded
    # logits with a different distribution on a tight protection boundary.
    base = (logits.detach().double() / ar_temperature).masked_fill(~legal, -torch.inf).softmax(-1)
    scores = torch.tensor(means, dtype=torch.float64, device=logits.device)
    tilt = (base.log() + scores / temperature).softmax(-1)
    tilted = (1. - strength) * base + strength * tilt
    p = base.detach().cpu().numpy()[ids].astype(np.float64)
    reference = tilted.detach().cpu().numpy()[ids].astype(np.float64)
    u = np.asarray(utilities, dtype=np.float64)[:, ids]
    c = np.asarray([[score.costs[key] for score in row]
        for row in construction for key in sorted(keys)], dtype=np.float64)[:, ids]
    # Rows constant across actions cannot be affected by a normalized teacher.
    c = c[np.ptp(c, axis=1) > 1e-12]
    lower = (1. - strength) * p
    bounds = [(float(value), 1.) for value in lower]
    matrix = np.vstack((-u, c))
    construction_gain = max(2 * min_gain, min_gain + 1e-9)
    rhs = np.r_[-u @ p - construction_gain, c @ p]
    scales = np.maximum(np.max(np.abs(matrix - matrix.mean(1, keepdims=True)), axis=1), 1e-8)
    # Subtract constants using sum(q)=1 before scaling to keep small observed
    # action differences well conditioned even when absolute scores are large.
    center = matrix.mean(1)
    scaled_matrix = (matrix - center[:, None]) / scales[:, None]
    scaled_rhs = (rhs - center) / scales
    feasible = linprog(np.zeros(len(ids)), A_ub=scaled_matrix, b_ub=scaled_rhs,
        A_eq=np.ones((1, len(ids))), b_eq=[1.], bounds=bounds, method='highs',
        options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    evidence = {'method': 'KL projection of execution-tilted distribution onto construction constraints',
        'construction_noise_count': len(construction), 'validation_data_used_in_solver': False,
        'base': base.tolist(), 'unconstrained': tilted.tolist(), 'strength': strength,
        'minimum_gain_per_construction_seed': construction_gain, 'feasibility_status': feasible.message}
    if not feasible.success:
        return None, base, None, {**evidence, 'reason': 'no_feasible_protected_construction_teacher'}
    objective = lambda q: float(np.sum(q * np.log(np.maximum(q, 1e-300) / reference)))
    derivative = lambda q: np.log(np.maximum(q, 1e-300) / reference) + 1.
    fit = minimize(objective, feasible.x, jac=derivative, method='SLSQP', bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda q: q.sum() - 1., 'jac': lambda q: np.ones_like(q)},
            {'type': 'ineq', 'fun': lambda q: scaled_rhs - scaled_matrix @ q,
                'jac': lambda q: -scaled_matrix}], options={'ftol': 1e-13, 'maxiter': 200})
    evidence.update(solver_status=fit.message, solver_iterations=int(fit.nit))
    if not fit.success:
        return None, base, None, {**evidence, 'reason': 'protected_teacher_solver_failed'}
    q = np.asarray(fit.x, dtype=np.float64)
    if (abs(q.sum() - 1.) > 1e-10 or (q < lower - 1e-12).any()
            or (u @ (q - p) < construction_gain - 1e-12).any() or (c @ (q - p) > 1e-12).any()):
        return None, base, None, {**evidence, 'reason': 'protected_teacher_solution_did_not_certify'}
    target = torch.zeros_like(base)
    target[ids] = torch.as_tensor(q, device=base.device, dtype=base.dtype)
    teacher = logits.detach().clone().to(torch.float64)
    teacher[legal] = target[legal].log() * ar_temperature
    target = (teacher / ar_temperature).masked_fill(~legal, -torch.inf).softmax(-1)
    evidence.update(reason='protected_construction_teacher_built', target=target.tolist(),
        construction_expected_gains=(u @ (q - p)).tolist(),
        construction_cost_changes=(c @ (q - p)).tolist())
    return teacher.detach(), base, target, evidence
