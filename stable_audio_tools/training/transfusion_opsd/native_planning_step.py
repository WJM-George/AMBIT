"""Limit planner-affecting update groups against an actual planning check.

The DiT-private proposal retains its full step while the planner/shared groups
backtrack. This finite-development constraint is not an audio-quality or unseen
request guarantee. The caller must immediately evaluate actual generation.
"""
from __future__ import annotations

import copy
import math

import torch


def native_planning_constrained_step(optimizer, check, *, constrained_groups=('ar_body','shared'),
        factors=(1., .5, .25, .125, .0625, .03125, .015625, 0.)):
    if (not callable(check) or not factors or factors[0]!=1 or factors[-1]!=0
            or any(not math.isfinite(x) or not 0<=x<=1 for x in factors)
            or any(a<=b for a,b in zip(factors,factors[1:]))):
        raise ValueError('Declare a check and strictly decreasing finite factors from one to zero.')
    names={group.get('name') for group in optimizer.param_groups}
    if not set(constrained_groups).issubset(names):
        raise ValueError('Planner-affecting optimizer groups are missing.')
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in parameters})!=len(parameters):
        raise ValueError('Optimizer groups must have disjoint parameter ownership.')
    initial=check()
    if not initial['passed']:
        raise ValueError('The pre-update native planning state already violates the declared constraint.')
    before=[p.detach().clone() for p in parameters]
    # Copy only optimizer tensors/metadata, never a model with scoped closures.
    optimizer_before=copy.deepcopy(optimizer.state_dict())
    constrained=[group.get('name') in constrained_groups for group in optimizer.param_groups for _ in group['params']]
    attempts=[]
    try:
        optimizer.step()
        proposal=[p.detach().clone() if selected else None for p,selected in zip(parameters,constrained)]
        for factor in factors:
            with torch.no_grad():
                for p,base,trial,selected in zip(parameters,before,proposal,constrained):
                    if selected:
                        p.copy_(trial if factor==1 else base if factor==0 else torch.lerp(base,trial,factor))
            evidence=check()
            attempts.append(dict(factor=factor,**evidence))
            if evidence['passed']:
                return dict(accepted=True,planner_shared_factor=factor,dit_private_factor=1.,attempts=attempts,
                    optimizer_moments_advanced=True,scope='Declared native planning constraints only; actual output validation remains required.')
    except BaseException:
        with torch.no_grad():
            for p,base in zip(parameters,before):p.copy_(base)
        optimizer.load_state_dict(optimizer_before)
        raise
    with torch.no_grad():
        for p,base in zip(parameters,before):p.copy_(base)
    optimizer.load_state_dict(optimizer_before)
    return dict(accepted=False,planner_shared_factor=0.,dit_private_factor=0.,attempts=attempts,
        optimizer_moments_advanced=False,scope='No proposal met the declared native planning constraints.')
