"""Project the *actual Adam displacement* and atomically accept both routes.

Two first-order halfspaces alone do not guarantee task improvement. The full
step must additionally lower both fixed distillation objectives and pass an
actual execution callback. Rejected updates restore weights, buffers, Adam
moments, scheduler and global RNG. Collection IDs live outside this transaction.
"""
from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def _dot(left, right):
    return sum(float((a.detach().double() * b.detach().double()).sum())
               for a, b in zip(left, right))


def project_shared_displacement(displacement, grad_ar, grad_dit, metric):
    """P^-1 metric projection onto g_AR.d <= 0 and g_DiT.d <= 0.

    Lists of tensors avoid a second concatenated copy of the full 300M stack.
    P is a positive diagonal metric, usually Adam's current preconditioner.
    """
    if not (len(displacement) == len(grad_ar) == len(grad_dit) == len(metric)):
        raise ValueError("shared displacement/gradient/metric lengths differ")
    if not displacement:
        raise ValueError("a Transfusion bundle requires shared parameters")
    for d, a, b, p in zip(displacement, grad_ar, grad_dit, metric):
        if not (d.shape == a.shape == b.shape == p.shape):
            raise ValueError("projection geometry mismatch")
        if not all(torch.isfinite(v).all() for v in (d, a, b, p)) or not (p > 0).all():
            raise ValueError("projection requires finite tensors and a positive metric")
    gradients = [grad_ar, grad_dit]
    pg = [[p * g for p, g in zip(metric, row)] for row in gradients]
    gram = torch.tensor([[_dot(row, prow) for prow in pg] for row in gradients], dtype=torch.float64)
    violation = torch.tensor([_dot(row, displacement) for row in gradients], dtype=torch.float64)
    tolerance = 1e-10 * max(1e-16, float(violation.abs().max()))
    best = None
    for size in range(3):
        for active in itertools.combinations(range(2), size):
            multiplier = torch.zeros(2, dtype=torch.float64)
            if active:
                idx = list(active)
                sub = gram[idx][:, idx]
                solution = torch.linalg.pinv(sub, rtol=1e-12, hermitian=True) @ violation[idx]
                if (solution < -1e-12).any():
                    continue
                multiplier[idx] = solution.clamp_min(0)
            remaining = violation - gram @ multiplier
            if (remaining > tolerance).any():
                continue
            cost = float(multiplier @ gram @ multiplier) / 2
            if best is None or cost < best[0]:
                best = (cost, multiplier)
    if best is None:
        # Zero is always feasible, including rank-deficient opposing gradients.
        result = [torch.zeros_like(d) for d in displacement]
    else:
        multiplier = best[1].tolist()
        result = [d - multiplier[0] * a - multiplier[1] * b for d, a, b in zip(displacement, *pg)]
    return result, {"ar_dot": _dot(grad_ar, result), "dit_dot": _dot(grad_dit, result),
                    "shared_step_norm": _dot(result, result) ** 0.5,
                    "projection_fallback": best is None}


def _cpu_clone(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _rng_state():
    return (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            random.getstate(), np.random.get_state())


def _restore_rng(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
    random.setstate(state[2])
    np.random.set_state(state[3])


def checkpoint_rng_state():
    """Encode NumPy RNG keys as a tensor so weights_only=True remains usable."""
    cpu, cuda, python, numpy = _rng_state()
    return {"torch": cpu, "cuda": cuda, "python": python,
            "numpy": [numpy[0], torch.from_numpy(numpy[1].astype(np.int64)), *numpy[2:]]}


def decode_checkpoint_rng(state):
    if not isinstance(state, dict) or set(state) != {"torch", "cuda", "python", "numpy"}:
        raise ValueError("candidate lacks a complete RNG checkpoint")
    if state["cuda"] is not None and len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("resume requires the same visible CUDA RNG device count")
    numpy = state["numpy"]
    return (state["torch"], state["cuda"], state["python"],
            (numpy[0], numpy[1].cpu().numpy().astype(np.uint32), *numpy[2:]))


@dataclass(frozen=True)
class JointStepResult:
    committed: bool
    reason: str
    ar_before: float | None = None
    dit_before: float | None = None
    ar_after: float | None = None
    dit_after: float | None = None
    step_scale: float = 0.0
    shared: dict | None = None


def joint_adam_step(module, partition, optimizer, loss_closure, *, validate_execution,
                    scheduler=None, min_loss_gain: float = 1e-8) -> JointStepResult:
    """Propose once, backtrack the whole update, commit once or restore all.

    loss_closure returns (pure_AR_loss, pure_DiT_loss, optional_auxiliary_loss).
    The callback must recompute actual candidate behavior, not reuse old scores.
    This implementation is serial/single-device and rejects distributed use.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("Transfusion OPSD does not implement distributed transactions")
    if not isinstance(optimizer, torch.optim.AdamW) or not callable(validate_execution):
        raise ValueError("an AdamW optimizer and actual execution validator are required")
    groups = {key: tuple(partition[key]) for key in ("ar_private", "shared", "dit_private")}
    named = [pair for values in groups.values() for pair in values]
    params = [p for _, p in named]
    if any(not values for values in groups.values()) or len({id(p) for p in params}) != len(params):
        raise ValueError("all three disjoint parameter groups must be nonempty")
    if {id(p) for p in params} != {id(p) for p in module.parameters() if p.requires_grad}:
        raise ValueError("partition does not cover every trainable parameter")
    if len({p.device for p in params}) != 1 or any(p.dtype not in (torch.float32, torch.float64) for p in params):
        raise ValueError("OPSD requires one device and FP32/FP64 master weights")
    opt_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(opt_params) != len(params) or {id(p) for p in opt_params} != {id(p) for p in params}:
        raise ValueError("optimizer ownership differs from the declared partition")
    shared_ids = {id(p) for _, p in groups["shared"]}
    for group in optimizer.param_groups:
        if any(id(p) in shared_ids for p in group["params"]) and group["weight_decay"] != 0:
            raise ValueError("shared weight decay must be zero")
    before_params = {name: p.detach().cpu().clone() for name, p in module.named_parameters()}
    before_buffers = {name: p.detach().cpu().clone() for name, p in module.named_buffers()}
    before_opt = _cpu_clone(optimizer.state_dict())
    before_scheduler = _cpu_clone(scheduler.state_dict()) if scheduler is not None else None
    before_rng = _rng_state()

    @torch.no_grad()
    def restore_tensors():
        for name, p in module.named_parameters():
            p.copy_(before_params[name])
        for name, p in module.named_buffers():
            p.copy_(before_buffers[name])

    def rollback():
        restore_tensors()
        optimizer.load_state_dict(before_opt)
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.load_state_dict(before_scheduler)
        _restore_rng(before_rng)

    try:
        optimizer.zero_grad(set_to_none=True)
        ar_loss, dit_loss, auxiliary = loss_closure()
        if any(loss.ndim != 0 or not torch.isfinite(loss) for loss in (ar_loss, dit_loss)):
            rollback()
            return JointStepResult(False, "nonfinite_loss")
        ga = torch.autograd.grad(ar_loss, params, retain_graph=True, allow_unused=True)
        gd = torch.autograd.grad(dit_loss, params, retain_graph=auxiliary is not None, allow_unused=True)
        gx = (torch.autograd.grad(auxiliary, params, allow_unused=True) if auxiliary is not None
              else (None,) * len(params))
        if any(g is not None and not torch.isfinite(g).all() for row in (ga, gd, gx) for g in row):
            rollback()
            return JointStepResult(False, "nonfinite_gradient")
        a0, d0 = float(ar_loss.detach()), float(dit_loss.detach())
        del ar_loss, dit_loss, auxiliary
        index = {id(p): i for i, p in enumerate(params)}
        shared_index = [index[id(p)] for _, p in groups["shared"]]
        zeros = lambda g, p: torch.zeros_like(p) if g is None else g.detach()
        sa = [zeros(ga[i], params[i]) for i in shared_index]
        sd = [zeros(gd[i], params[i]) for i in shared_index]
        na, nd = max(_dot(sa, sa) ** .5, 1e-12), max(_dot(sd, sd) ** .5, 1e-12)
        for label, values in groups.items():
            for _, p in values:
                i = index[id(p)]
                if label == "shared":
                    p.grad = .5 * (zeros(ga[i], p) / na + zeros(gd[i], p) / nd)
                    if gx[i] is not None and gx[i].count_nonzero():
                        raise ValueError("source-caption auxiliary reached shared parameters")
                else:
                    own, other = (ga[i], gd[i]) if label == "ar_private" else (gd[i], ga[i])
                    if other is not None and other.count_nonzero():
                        raise ValueError("route dependency escaped its parameter partition")
                    p.grad = zeros(own, p) + zeros(gx[i], p)
        optimizer.step()  # proposal includes moments and private weight decay
        raw = [(p.detach() - before_params[name].to(p.device)).clone() for name, p in named]
        metric_by_id = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                if id(p) in shared_ids:
                    state = optimizer.state[p]
                    variance = state["max_exp_avg_sq"] if group.get("amsgrad") else state["exp_avg_sq"]
                    denom = (variance / (1 - group["betas"][1] ** float(state["step"]))).sqrt() + group["eps"]
                    metric_by_id[id(p)] = (group["lr"] / denom).clamp_min(1e-16)
        projected, diagnostics = project_shared_displacement(
            [raw[i] for i in shared_index], sa, sd,
            [metric_by_id[id(params[i])] for i in shared_index])
        for i, direction in zip(shared_index, projected):
            raw[i] = direction
        after_fit_rng = _rng_state()
        diagnostics['backtracking'] = []
        for alpha in (1., .5, .25):
            restore_tensors()
            with torch.no_grad():
                for (name, p), direction in zip(named, raw):
                    p.copy_(before_params[name].to(p.device) + alpha * direction)
                # Verify the representable displacement after floating-point copy.
                actual = [params[i] - before_params[named[i][0]].to(params[i].device) for i in shared_index]
                norm = _dot(actual, actual) ** .5
                actual_ar_dot, actual_dit_dot = _dot(sa, actual), _dot(sd, actual)
                trial = {'scale': alpha, 'actual_ar_dot': actual_ar_dot, 'actual_dit_dot': actual_dit_dot}
                diagnostics['backtracking'].append(trial)
                if actual_ar_dot > 1e-6 * na * norm or actual_dit_dot > 1e-6 * nd * norm:
                    trial['reason'] = 'representable_shared_step_not_feasible'
                    continue
                _restore_rng(before_rng)
                la, ld, _ = loss_closure()
                a1, d1 = float(la), float(ld)
                lower = (torch.isfinite(la) and torch.isfinite(ld)
                         and a1 < a0 - min_loss_gain and d1 < d0 - min_loss_gain)
                trial.update(ar_loss=a1, dit_loss=d1, both_losses_lower=bool(lower))
                execution_okay = bool(validate_execution()) if lower else False
                trial['execution_accepted'] = execution_okay
                if lower and execution_okay:
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(after_fit_rng)
                    return JointStepResult(True, "candidate_step_accepted", a0, d0, a1, d1, alpha, diagnostics)
        rollback()
        return JointStepResult(False, "joint_loss_or_execution_rejected", a0, d0, shared=diagnostics)
    except BaseException:
        rollback()
        raise
