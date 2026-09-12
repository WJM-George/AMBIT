"""A finite joint Adam block, accepted only after actual native behavior checks.

The caller fixes observations, teacher targets and objectives for a small block.
Unlike a first-step-only rejection, later internal steps can see nonzero
reference-retention loss. This is a hypothesis about learnability, not a
retention guarantee. Every inner step updates shared parameters once. Report
all attempted inner steps; an accepted block is not one ordinary Adam step.
Snapshots stay in memory. Existing one-step experiments remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .shared_step import _cpu_clone, _rng_state, _restore_rng


@dataclass(frozen=True)
class GuardedJointBlock:
    accepted: bool
    scales: dict | None
    trials: list[dict]
    actual_displacements: dict
    attempted_optimizer_steps: int
    accepted_optimizer_steps: int
    inner_evidence: list[dict]


def guarded_joint_adam_block(module, optimizer, *, backward_step, inner_steps,
                             trial_scales, validate_native):
    """Compute 1--4 joint steps, then test fixed scales of the entire block.

``backward_step(i)`` computes and accumulates the declared joint gradients and
returns detached diagnostic data; this function owns zero_grad and step.
It must not mutate optimizer groups or parameters itself. ``validate_native``
is read-only and returns an explicit bool ``passed``. The optimizer moments
describe all unscaled inner steps, even if the final displacement is scaled.
On rejection/exception restore parameters, buffers, optimizer and global RNG.
"""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('Use the actual joint AdamW optimizer.')
    if (type(inner_steps) is not int or not 1 <= inner_steps <= 4
            or not callable(backward_step) or not callable(validate_native)):
        raise ValueError('Declare a bounded block and both callbacks.')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('This development block is single-device only.')
    groups = optimizer.param_groups
    names = [g.get('name') for g in groups]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Joint optimizer groups need unique names.')
    params = [p for g in groups for p in g['params']]
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError('Register shared parameters exactly once.')
    if len({p.device for p in params}) != 1 or any(p.dtype not in (torch.float32, torch.float64) for p in params):
        raise ValueError('Use one device and actual FP32/FP64 master parameters.')
    scales = [dict(s) for s in trial_scales]
    if not 1 <= len(scales) <= 8:
        raise ValueError('Declare a finite scale list before training.')
    for s in scales:
        if set(s) != set(names) or any(not math.isfinite(v) or not 0 < v <= 1 for v in s.values()):
            raise ValueError('Every parameter group needs a positive scale at most one.')
    # CPU copies avoid keeping a second full model on the GPU during backward.
    before = [_cpu_clone(p) for p in params]
    buffers = {n: _cpu_clone(b) for n, b in module.named_buffers()}
    old_opt, old_rng = _cpu_clone(optimizer.state_dict()), _rng_state()
    evidence, trials, attempted = [], [], 0

    @torch.no_grad()
    def restore_buffers():
        current = dict(module.named_buffers())
        if current.keys() != buffers.keys():
            raise RuntimeError('A callback changed registered buffers.')
        for n, b in current.items():
            b.copy_(buffers[n])

    @torch.no_grad()
    def rollback():
        for p, value in zip(params, before):
            p.copy_(value)
        restore_buffers()
        optimizer.load_state_dict(old_opt)
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(old_rng)

    try:
        for i in range(inner_steps):
            optimizer.zero_grad(set_to_none=True)
            record = backward_step(i)
            if not isinstance(record, dict):
                raise TypeError('The joint backward callback must report its objectives.')
            if not any(p.grad is not None for p in params) or any(
                    p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError('Missing or nonfinite joint gradient.')
            optimizer.step()
            attempted += 1
            if any(not torch.isfinite(p).all() for p in params):
                raise FloatingPointError('Nonfinite internal parameter proposal.')
            evidence.append(record)
        optimizer.zero_grad(set_to_none=True)
        proposed = [_cpu_clone(p) for p in params]
        after_rng = _rng_state()
        for scale in scales:
            with torch.no_grad():
                index = 0
                for g in groups:
                    factor = scale[g['name']]
                    for p in g['params']:
                        value = proposed[index] if factor == 1 else before[index] + factor * (proposed[index] - before[index])
                        p.copy_(value)
                        index += 1
                restore_buffers()
                _restore_rng(after_rng)
                check = validate_native()
            if not isinstance(check, dict) or type(check.get('passed')) is not bool:
                raise TypeError('Return an actual native passed bool and its evidence.')
            trials.append(dict(scales=scale, validation=check))
            if check['passed']:
                stats, index = {}, 0
                with torch.no_grad():
                    for g in groups:
                        squares = maximum = 0.
                        changed = number = 0
                        for p in g['params']:
                            d = p.detach().cpu().double() - before[index].double()
                            index += 1
                            squares += float(d.square().sum())
                            maximum = max(maximum, float(d.abs().max()))
                            changed += int(d.count_nonzero())
                            number += p.numel()
                        stats[g['name']] = dict(L2=math.sqrt(squares), RMS=math.sqrt(squares / number) if number else 0., maximum=maximum, changed_elements=changed, parameters=number)
                restore_buffers()
                _restore_rng(after_rng)
                return GuardedJointBlock(True, scale, trials, stats, attempted, attempted, evidence)
        rollback()
        return GuardedJointBlock(False, None, trials, {}, attempted, 0, evidence)
    except BaseException:
        rollback()
        raise
