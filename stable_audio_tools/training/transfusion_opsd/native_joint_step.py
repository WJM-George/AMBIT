"""Prefer a full shared/DiT update when a smaller private-AR step is feasible.

The existing transactional step owns parameter and optimizer restoration.
This ordering prevents an oversized private planner proposal from needlessly
shrinking a compatible shared execution update. Feasibility is defined by the
caller's actual native planning check, never by this helper's name.
"""
from .native_planning_step import native_planning_constrained_step


def native_private_first_joint_step(optimizer, check, *,
        factors=(1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625, 0.)):
    """Try private AR rescaling with full shared/DiT, then joint rescaling.

    A failed first pass restores parameters and Adam moments before the second
    proposal. Successful fractional steps retain the existing convention of
    advancing the proposal's optimizer moments. This is a finite feasibility
    procedure, not a policy-improvement or audio-preservation guarantee.
    """
    first = native_planning_constrained_step(optimizer, check,
        constrained_groups=('ar_body',), factors=factors)
    if first['accepted']:
        return dict(accepted=True, ar_private_factor=first['planner_shared_factor'],
            shared_factor=1., dit_private_factor=1., optimizer_moments_advanced=True,
            passes=[dict(name='private_AR_only', result=first)])
    second = native_planning_constrained_step(optimizer, check,
        constrained_groups=('ar_body','shared'), factors=factors)
    return dict(accepted=second['accepted'], ar_private_factor=second['planner_shared_factor'],
        shared_factor=second['planner_shared_factor'], dit_private_factor=second['dit_private_factor'],
        optimizer_moments_advanced=second['optimizer_moments_advanced'],
        passes=[dict(name='private_AR_only',result=first),dict(name='AR_and_shared',result=second)])
