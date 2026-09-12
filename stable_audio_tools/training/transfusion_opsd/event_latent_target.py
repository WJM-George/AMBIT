"""First-order content-aware latent search, followed by unchanged certification.

The gradient construction is a local proposal, not a guarantee about decoded
content, ASR or the denoising suffix. All those checks still decide acceptance.
"""
from __future__ import annotations

import torch


def content_aware_latent_objective(latent, *, decode, spatial_reward, content_cost, diagnostics=None):
    """Keep ascent useful for spatial utility while decreasing one content cost.

    For g=grad(U), h=grad(C), use g-beta*h. If they conflict, beta removes the
    cost-increasing component and adds a bounded interior margin. The margin
    consumes at most 25% of the remaining first-order utility gain. Numerical
    and nonlinear failures remain subject to actual clean/suffix rejection.
    """
    return protected_latent_objective(latent, decode=decode, utility=spatial_reward,
        protected_cost=content_cost, diagnostics=diagnostics)


def protected_latent_objective(latent, *, decode, utility, protected_cost, diagnostics=None):
    """Utility ascent with a locally nonincreasing protected differentiable cost.

    The caller declares the roles (e.g. semantic utility and sector deficit).
    Real decoded/suffix measurements still decide target acceptance.
    """
    if not latent.requires_grad:
        raise ValueError('latent search requires an explicit differentiable query')
    waveform = decode(latent)
    utility = utility(waveform)
    cost = protected_cost(waveform)
    gu, = torch.autograd.grad(utility, latent, retain_graph=True)
    gc, = torch.autograd.grad(cost, latent, retain_graph=True)
    if not torch.isfinite(gu).all() or not torch.isfinite(gc).all():
        raise ValueError('non-finite spatial/content latent gradient')
    uu = gu.double().square().sum()
    cc = gc.double().square().sum()
    uc = (gu.double() * gc.double()).sum()
    beta = torch.zeros((), dtype=torch.float64, device=latent.device)
    if cc > 1e-24:
        beta = .25 * (uu / cc).sqrt()
        if uc > 0:
            remaining = (uu - uc.square() / cc).clamp_min(0.)
            interior = torch.minimum(beta, .25 * remaining / uc.clamp_min(1e-24))
            beta = uc / cc + interior
    if diagnostics is not None:
        diagnostics.append({'utility': float(utility.detach()), 'protected_cost': float(cost.detach()),
            'utility_gradient_norm': float(uu.sqrt()), 'protected_cost_gradient_norm': float(cc.sqrt()),
            'gradient_dot': float(uc), 'protected_cost_multiplier': float(beta),
            'first_order_utility_gain': float(uu - beta * uc),
            'first_order_protected_cost_change': float(uc - beta * cc)})
    return utility - beta.detach().to(cost.dtype) * cost
