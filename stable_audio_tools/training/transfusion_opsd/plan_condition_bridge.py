"""Explicit planner-state -> native diffusion conditioning, without sampling gradients.

The same bridge is used for training and normal sampling. It does not replace
the discrete ScenePlan, certify request adherence, or connect output decision
heads which are not upstream of the supplied hidden states.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class PlanBridgeConfig:
    plan_dim: int = 1024
    condition_dim: int = 768
    width: int = 128
    maximum_relative_rms: float = .05
    reference_rms_floor: float = 1e-6
    initialization_seed: int = 82909001

    def __post_init__(self):
        for name in ('plan_dim', 'condition_dim', 'width', 'initialization_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('maximum_relative_rms', 'reference_rms_floor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive fixed scalar')


class PlanConditionBridge(nn.Module):
    """Attend from each native condition token to causal planner hidden states.

    Native conditions are detached only inside this auxiliary branch. Their
    existing diffusion path remains trainable. The detached reference norm
    bounds every valid token's residual; padded condition tokens stay zero.
    """
    contract = 'bounded_native_plan_hidden_condition_bridge_v1_experimental'

    def __init__(self, config=PlanBridgeConfig()):
        super().__init__()
        self.config = config
        # Construct on CPU without advancing the caller's RNG or CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.initialization_seed)
            self.plan_norm = nn.LayerNorm(config.plan_dim)
            self.condition_norm = nn.LayerNorm(config.condition_dim)
            self.query = nn.Linear(config.condition_dim, config.width, bias=False)
            self.key = nn.Linear(config.plan_dim, config.width, bias=False)
            self.value = nn.Linear(config.plan_dim, config.width, bias=False)
            self.output = nn.Linear(config.width, config.condition_dim, bias=False)
            nn.init.zeros_(self.output.weight)

    def receipt(self):
        return dict(contract=self.contract, config=asdict(self.config),
            parameters=sum(p.numel() for p in self.parameters()),
            initialization='zero residual; first update can train output before upstream planner gradients appear',
            scope='positive native conditioning only; negative CFG conditioning unchanged')

    def forward(self, hidden, plan_mask, condition, condition_mask, *, stop_plan_gradient=False):
        if not isinstance(stop_plan_gradient, bool):
            raise ValueError('stop_plan_gradient must explicitly be boolean')
        if (hidden.ndim != 3 or condition.ndim != 3 or hidden.shape[0] != condition.shape[0]
                or hidden.shape[-1] != self.config.plan_dim or condition.shape[-1] != self.config.condition_dim):
            raise ValueError('planner and native condition dimensions differ from the declared bridge')
        for value, mask in ((hidden, plan_mask), (condition, condition_mask)):
            if (not value.is_floating_point() or mask.dtype != torch.bool or mask.shape != value.shape[:2]
                    or value.device != mask.device or not mask.any(-1).all() or not torch.isfinite(value).all()):
                raise ValueError('bridge inputs require finite features and nonempty aligned boolean masks')
        if hidden.device != condition.device or hidden.device != self.output.weight.device:
            raise ValueError('bridge inputs and parameters must share a device')
        # Preserve a direct, ordinary autograd edge; no straight-through estimator.
        source = hidden.detach() if stop_plan_gradient else hidden
        reference = condition.detach()
        dtype = self.output.weight.dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            source = self.plan_norm(source.to(dtype))
            query = self.query(self.condition_norm(reference.to(dtype)))
            keys, values = self.key(source), self.value(source)
            scores = query @ keys.transpose(-1, -2) / math.sqrt(self.config.width)
            scores = scores.masked_fill(~plan_mask[:, None, :], -torch.inf)
            raw = self.output(scores.softmax(-1) @ values)
            limit = (reference.to(dtype).square().mean(-1, keepdim=True).sqrt()
                .clamp_min(self.config.reference_rms_floor) * self.config.maximum_relative_rms)
            # Smooth zero-preserving bound with a finite derivative at zero.
            residual = raw / (1 + raw.square().mean(-1, keepdim=True) / limit.square()).sqrt()
            return residual.masked_fill(~condition_mask[..., None], 0.)


def native_plan_hidden_states(bundle, observation, token_ids):
    """Replay a complete sampled plan through the released causal AR forward.

    This function accepts no audio, time, reward, teacher state, or future
    observation. Model-produced token IDs remain fixed for this replay.
    """
    if not token_ids or token_ids[0] != bundle.codec.bos_id or token_ids[-1] != bundle.codec.eos_id:
        raise ValueError('bridge requires a complete sampled native plan')
    context, context_mask = bundle.encode_event_requests([observation.request], device=bundle.device)
    tokens = torch.as_tensor([token_ids], device=bundle.device, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    captured = []
    handle = bundle.ar.ar_adapter.output_norm.register_forward_hook(
        lambda module, args, output: captured.append(output))
    try:
        bundle.ar(tokens, mask, context, context_mask)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[:2] != tokens.shape:
        raise ValueError('native AR did not expose one aligned complete planning forward')
    return captured[0], mask


def bridged_velocity_function(bundle, condition, proposal, bridge, *, mode, differentiable):
    """Build one native closure; reuse its request/plan features while sampling.

    'stopped' cuts the new interface only. Shared parameters still receive
    gradients through the original diffusion computation.
    """
    if mode not in ('off', 'stopped', 'connected'):
        raise ValueError('declare off, stopped, or connected planner interface')
    if condition.plan != proposal.plan:
        raise ValueError('planner features must describe the plan actually executed')
    if mode == 'off':
        return bundle.velocity_function(condition, differentiable=differentiable)
    if not isinstance(bridge, PlanConditionBridge):
        raise ValueError('a registered PlanConditionBridge is required')
    enabled = nullcontext() if differentiable else torch.no_grad()
    with enabled:
        hidden, plan_mask = native_plan_hidden_states(bundle, proposal.observation, proposal.tokens)
        inputs = bundle.native_conditioning(condition, differentiable=differentiable)
        residual = bridge(hidden, plan_mask, inputs['cross_attn_cond'], inputs['cross_attn_mask'].bool(),
            stop_plan_gradient=(mode == 'stopped'))
        inputs['external_cross_attn_residual'] = residual

    def velocity(z, t):
        enabled = nullcontext() if differentiable else torch.no_grad()
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if bundle.device.type == 'cuda' else nullcontext()
        with enabled, amp:
            return bundle.diffusion.model(z, t, **inputs, cfg_scale=bundle.cfg_scale,
                batch_cfg=True, rescale_cfg=True, scale_phi=bundle.cfg_rescale_phi,
                apg_scale=0., cfg_dropout_prob=0., padding_mask=condition.mask)
    velocity.plan_hidden = hidden
    velocity.condition_residual = residual
    return velocity
