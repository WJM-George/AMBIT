"""Bound an Adam proposal using actual native behavior in either Transfusion task.

Gradients are computed by the caller's joint OPSD objectives, including any
declared decision surrogate. A finite, predeclared list of group scales limits
the representable displacement; it does not change inference or manufacture
teacher labels. Adam moments describe the original proposed gradient. The
accepted parameter displacement can differ from an unmodified Adam step.

This guard only establishes the supplied finite behavior checks. Real generated
audio and refreshed execution feedback remain separate acceptance requirements.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Mapping, Sequence

import torch

from .shared_step import _cpu_clone, _rng_state, _restore_rng


@dataclass(frozen=True)
class GuardedJointUpdate:
    accepted: bool
    scales: dict[str, float] | None
    trials: list[dict]
    actual_displacements: dict
    optimizer_proposals: int = 1


def guarded_joint_adam_update(module, optimizer, *, trial_scales: Sequence[Mapping[str,float]],
                             validate_native: Callable[[],dict], scheduler=None) -> GuardedJointUpdate:
    """Apply one gradient proposal, at most the declared native-only trials.

All state copies are in memory. If no trial passes (or validation raises),
restore optimizer moments, parameters, buffers, scheduler and global RNG. On
success advance the optimizer/scheduler once and discard validation RNG effects.
Every nonempty parameter group gets an explicit positive scale, including DiT.
The validator must be read-only and return a bool ``passed`` plus evidence.
"""
    if not isinstance(optimizer,torch.optim.AdamW) or not callable(validate_native):
        raise TypeError('Use an AdamW proposal and an actual native behavior validator.')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size()!=1:
        raise ValueError('This bounded guard is a single-device development implementation.')
    groups=optimizer.param_groups;names=[g.get('name') for g in groups]
    if any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
        raise ValueError('Optimizer groups need unique explicit names.')
    params=[p for g in groups for p in g['params']]
    if not params or len({id(p) for p in params})!=len(params):
        raise ValueError('Shared parameters must occur exactly once in the optimizer.')
    if len({p.device for p in params})!=1 or any(p.dtype not in (torch.float32,torch.float64) for p in params):
        raise ValueError('Guard actual representable FP32/FP64 master-parameter displacements.')
    scales=[dict(x) for x in trial_scales]
    if not scales or len(scales)>8:
        raise ValueError('Declare one to eight finite trials before fitting.')
    for scale in scales:
        if set(scale)!=set(names) or any(not math.isfinite(s) or not 0<s<=1 for s in scale.values()):
            raise ValueError('Every group needs an explicit positive scale at most one.')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise ValueError('Nonfinite joint gradient cannot be proposed.')
    before=[p.detach().clone() for p in params]
    buffers={n:b.detach().clone() for n,b in module.named_buffers()}
    old_opt=_cpu_clone(optimizer.state_dict());old_rng=_rng_state()
    old_scheduler=_cpu_clone(scheduler.state_dict()) if scheduler is not None else None
    trials=[]

    @torch.no_grad()
    def restore_buffers():
        current=dict(module.named_buffers())
        if current.keys()!=buffers.keys():raise RuntimeError('Native validator changed registered buffers.')
        for n,b in current.items():b.copy_(buffers[n])

    @torch.no_grad()
    def rollback():
        for p,value in zip(params,before):p.copy_(value)
        restore_buffers();optimizer.load_state_dict(old_opt)
        if scheduler is not None:scheduler.load_state_dict(old_scheduler)
        optimizer.zero_grad(set_to_none=True);_restore_rng(old_rng)

    try:
        optimizer.step()
        proposed=[p.detach().clone() for p in params];after_proposal_rng=_rng_state()
        if any(not torch.isfinite(p).all() for p in proposed):
            raise FloatingPointError('Nonfinite Adam parameter proposal.')
        for scale in scales:
            with torch.no_grad():
                index=0
                for group in groups:
                    factor=scale[group['name']]
                    for p in group['params']:
                        # Preserve the exact original Adam proposal for factor=1.
                        p.copy_(proposed[index] if factor==1 else before[index]+factor*(proposed[index]-before[index]))
                        index+=1
                restore_buffers();_restore_rng(after_proposal_rng)
                evidence=validate_native()
            if not isinstance(evidence,dict) or type(evidence.get('passed')) is not bool:
                raise TypeError('Native validator must return an explicit bool passed field.')
            trials.append(dict(scales=scale,validation=evidence))
            if evidence['passed']:
                stats={};index=0
                with torch.no_grad():
                    for group in groups:
                        squares=0.;maximum=0.;changed=0;number=0
                        for p in group['params']:
                            d=p.double()-before[index].double();index+=1
                            squares+=float(d.square().sum());maximum=max(maximum,float(d.abs().max()))
                            changed+=int(d.count_nonzero());number+=p.numel()
                        stats[group['name']]=dict(L2=math.sqrt(squares),RMS=math.sqrt(squares/number) if number else 0.,
                            maximum=maximum,changed_elements=changed,parameters=number)
                restore_buffers();_restore_rng(after_proposal_rng)
                if scheduler is not None:scheduler.step()
                return GuardedJointUpdate(True,scale,trials,stats)
        rollback()
        return GuardedJointUpdate(False,None,trials,{})
    except BaseException:
        rollback();raise
