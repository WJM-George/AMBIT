"""Detached OPSD teachers, with the repo's noise-at-one RF convention.

References: Zhao et al., arXiv:2601.18734v3; Zhou et al., arXiv:2608.24646.
Only certified positive clean targets are supported. Certification and fixed
fitting normalizers are Transfusion adaptations, not an exact reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import math

import torch
from torch import Tensor


def legal_log_probs(logits: Tensor, allowed: Tensor, temperature: float) -> Tensor:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if logits.shape != allowed.shape or allowed.dtype != torch.bool:
        raise ValueError("legal mask must be bool and match logits")
    if not allowed.any(-1).all() or not torch.isfinite(logits[allowed]).all():
        raise ValueError("every active prefix needs finite logits and legal actions")
    return (logits.float() / temperature).masked_fill(~allowed, -torch.inf).log_softmax(-1)


def forward_kl(
    student_logits: Tensor, teacher_logits: Tensor, allowed: Tensor, *,
    temperature: float = 0.8, token_weights: Tensor | None = None,
    pointwise_clip: float | None = None,
) -> Tensor:
    """Full legal-vocabulary KL(q_teacher || p_student) on student prefixes.

    Both distributions have identical grammar support and temperature. Weights
    apply to positions, not vocabulary entries. Zero-weight positions (padding)
    are excluded *before* normalization. Teacher logits never receive gradients.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must align at student prefixes")
    if token_weights is None:
        token_weights = torch.ones_like(student_logits[..., 0])
    weights = token_weights.detach().to(student_logits.device).float()
    if weights.shape != student_logits.shape[:-1] or not torch.isfinite(weights).all():
        raise ValueError("invalid token weights")
    if (weights < 0).any() or not (weights > 0).any():
        raise ValueError("token weights need positive total mass")
    active = weights > 0
    legal = allowed[active]
    lp = legal_log_probs(student_logits[active], legal, temperature)
    lq = legal_log_probs(teacher_logits.detach()[active], legal, temperature)
    # Avoid 0 * (-inf - -inf), including in autograd's unselected branches.
    delta = lq.masked_fill(~legal, 0) - lp.masked_fill(~legal, 0)
    contributions = lq.exp() * delta
    if pointwise_clip is not None:
        if not math.isfinite(pointwise_clip) or pointwise_clip <= 0:
            raise ValueError("pointwise clipping threshold must be positive")
        contributions = contributions.clamp(max=pointwise_clip)
    return (contributions.sum(-1) * weights[active]).sum() / weights[active].sum()


@dataclass(frozen=True)
class ARRollout:
    token_ids: tuple[int, ...]  # includes BOS and, when finished, EOS
    legal_ids: tuple[tuple[int, ...], ...]  # one set per sampled action
    log_probs: tuple[float, ...]
    finished: bool
    seed: int
    temperature: float
    behavior_version: int

    def legal_mask(self, vocab_size: int, device: torch.device) -> Tensor:
        mask = torch.zeros((1, len(self.legal_ids), vocab_size), device=device, dtype=torch.bool)
        for index, ids in enumerate(self.legal_ids):
            mask[0, index, list(ids)] = True
        return mask


@torch.no_grad()
def sample_ar(
    logits_fn: Callable[[Tensor], Tensor], codec, *, seed: int,
    device: torch.device, temperature: float = 0.8, max_tokens: int = 1024,
    behavior_version: int = 0,
    allowed_fn: Callable[[Sequence[int]], set[int]] | None = None,
) -> ARRollout:
    """Untruncated categorical sampling; never repairs/relabels a rollout."""
    if max_tokens < 2 or seed < 0:
        raise ValueError("invalid rollout budget or seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids, supports, logs = [int(codec.bos_id)], [], []
    allowed_fn = codec.allowed_next_ids if allowed_fn is None else allowed_fn
    while len(ids) < max_tokens:
        legal = tuple(sorted(allowed_fn(ids)))
        if not legal:
            break
        logits = logits_fn(torch.tensor([ids], dtype=torch.long, device=device))[0, -1]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[list(legal)] = True
        logp = legal_log_probs(logits, mask, temperature)
        token = int(torch.multinomial(logp.exp().cpu(), 1, generator=generator))
        supports.append(legal)
        logs.append(float(logp[token]))
        ids.append(token)
        if token == codec.eos_id:
            break
    return ARRollout(tuple(ids), tuple(supports), tuple(logs), ids[-1] == codec.eos_id,
                     seed, temperature, behavior_version)


@dataclass(frozen=True)
class RewardScore:
    utility: float
    costs: Mapping[str, float]

    def __post_init__(self):
        if not all(math.isfinite(float(x)) for x in (self.utility, *self.costs.values())):
            raise ValueError("reward and protection costs must be finite; missing is not success")

    def improves(self, baseline: "RewardScore", *, min_gain: float = 0.0) -> bool:
        if self.costs.keys() != baseline.costs.keys():
            raise ValueError("protection metric coverage changed during comparison")
        return self.utility > baseline.utility + min_gain and all(
            self.costs[key] <= baseline.costs[key] for key in self.costs
        )


def valid_mask(value: Tensor, mask: Tensor) -> Tensor:
    if value.ndim != 3 or mask.shape != (value.shape[0], value.shape[2]):
        raise ValueError("expected [B,C,T] and [B,T]")
    if mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError("every row needs a nonempty boolean valid-frame mask")
    return mask[:, None].expand_as(value)


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    full = valid_mask(value, mask)
    return value.masked_fill(~full, 0).sum((1, 2)) / full.sum((1, 2))


def clean_prediction(z: Tensor, time: Tensor, velocity: Tensor) -> Tensor:
    if z.shape != velocity.shape or time.shape != (z.shape[0],):
        raise ValueError("RF query geometry mismatch")
    if not torch.isfinite(time).all() or not ((time > 0) & (time <= 1)).all():
        raise ValueError("RF clean prediction requires 0 < t <= 1")
    return z.float() - time[:, None, None] * velocity.float()


@dataclass(frozen=True)
class EulerTrace:
    states: tuple[Tensor, ...]
    times: tuple[float, ...]
    mask: Tensor


@torch.no_grad()
def euler_rollout(
    velocity_fn: Callable[[Tensor, Tensor], Tensor], initial: Tensor,
    mask: Tensor, times: Sequence[float],
) -> EulerTrace:
    schedule = tuple(float(t) for t in times)
    if (len(schedule) < 2 or schedule[-1] != 0 or schedule[0] > 1
            or not all(math.isfinite(t) for t in schedule)
            or not all(a > b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError("Euler schedule must strictly decrease from <=1 to zero")
    full = valid_mask(initial, mask)
    z = initial.detach().float().masked_fill(~full, 0)
    states = [z.clone()]
    for start, end in zip(schedule, schedule[1:]):
        t = torch.full((z.shape[0],), start, device=z.device)
        velocity = velocity_fn(z, t).float()
        if velocity.shape != z.shape or not torch.isfinite(velocity[full]).all():
            raise ValueError("invalid RF velocity")
        z = (z - (start - end) * velocity).masked_fill(~full, 0)
        states.append(z.clone())
    return EulerTrace(tuple(states), schedule, mask.detach().clone())


@torch.no_grad()
def resume_from_clean(
    trace: EulerTrace, query_index: int, clean: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Replace exactly one Euler prediction, then run the unchanged suffix."""
    if not 0 <= query_index < len(trace.times) - 1:
        raise ValueError("query must precede the endpoint")
    z = trace.states[query_index]
    if clean.shape != z.shape:
        raise ValueError("clean target and query must have identical geometry")
    t, next_t = trace.times[query_index:query_index + 2]
    successor = (z - (t - next_t) * ((z - clean) / t)).masked_fill(
        ~valid_mask(z, trace.mask), 0)
    if next_t == 0:
        return successor
    return euler_rollout(velocity_fn, successor, trace.mask, trace.times[query_index + 1:]).states[-1]


@dataclass(frozen=True)
class DiffusionTargets:
    anchor: Tensor
    positive: Tensor
    mask: Tensor


def build_diffusion_targets(
    anchor: Tensor, mask: Tensor, *,
    differentiable_reward: Callable[[Tensor], Tensor],
    score_clean: Callable[[Tensor], RewardScore],
    score_suffix: Callable[[Tensor], RewardScore],
    radius: float = 0.1, target_steps: int = 2, min_gain: float = 0.0,
    target_mode: str = "positive",
) -> DiffusionTargets | None:
    """Certified reward ascent on a detached clean latent through a frozen VAE.

    This serial pilot handles one query at a time. Positive targets must improve
    both the clean-output score and the actual Euler suffix, including every
    declared protection cost. No gradient is taken through the rollout DiT.
    Negative/reflected targets were retired; legacy configurations fail closed.
    """
    if target_mode != "positive":
        raise ValueError("only certified positive targets are supported; legacy objectives are retired")
    if anchor.shape[0] != 1 or not 0 < radius <= 1 or target_steps < 1:
        raise ValueError("one query, 0 < radius <= 1, and positive target_steps required")
    full = valid_mask(anchor, mask)
    base = anchor.detach().float().masked_fill(~full, 0)
    if not torch.isfinite(base).all():
        raise ValueError("non-finite clean anchor")
    budget = radius * base.norm()
    if not budget > 0:
        return None
    with torch.no_grad():
        clean_before, suffix_before = score_clean(base), score_suffix(base)

    def refine() -> Tensor:
        value = base.clone()
        for _ in range(target_steps):
            with torch.enable_grad():
                leaf = value.detach().requires_grad_(True)
                reward = differentiable_reward(leaf)
                if reward.numel() != 1 or not torch.isfinite(reward).all():
                    raise ValueError("reward must be one finite differentiable scalar per query")
                gradient, = torch.autograd.grad(reward.sum(), leaf)
            gradient = gradient.masked_fill(~full, 0)
            if not torch.isfinite(gradient).all():
                raise ValueError("reward gradient is non-finite")
            value = value.detach() + (budget / target_steps) * gradient / gradient.norm().clamp_min(1e-12)
            delta = value - base
            value = (base + delta * (budget / delta.norm().clamp_min(1e-12)).clamp(max=1)).detach()
        return value

    positive = refine()
    accepted = None
    with torch.no_grad():
        for alpha in (1.0, 0.5, 0.25):
            proposal = base + alpha * (positive - base)
            if (score_clean(proposal).improves(clean_before, min_gain=min_gain)
                    and score_suffix(proposal).improves(suffix_before, min_gain=min_gain)):
                accepted = proposal.detach()
                break
    if accepted is None:
        return None
    return DiffusionTargets(base, accepted, mask.detach())


def diffusion_loss(z: Tensor, time: Tensor, velocity: Tensor, targets: DiffusionTargets) -> Tensor:
    prediction = clean_prediction(z.detach(), time.detach(), velocity)
    anchor = targets.anchor.detach()
    target = targets.positive.detach()
    normalizer = masked_mean((anchor - target).abs(), targets.mask).clamp_min(1e-5)
    return (masked_mean((prediction - target).square(), targets.mask) / normalizer).mean()
