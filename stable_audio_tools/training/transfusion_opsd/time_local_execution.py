"""Experimental time-local parameter increments for a shared AR/DiT model.

The registered student parameters remain the optimizer's unique leaves. AR uses
them normally. A DiT forward at t > maximum_trainable_time substitutes detached
anchor values; a later forward uses the current student. This is a changed
execution parameterization, requiring the anchor AND threshold at inference.
It is not equivalent to an ordinary time-independent candidate checkpoint.

Both conditioning and velocity evaluation are inside the functional call. This
matters when the conditioning projections share updated parameters with AR.
The first prototype accepts homogeneous gate decisions per microbatch.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.func import functional_call


class _ExecutionForward(nn.Module):
    def __init__(self, bundle, native_factory, conditional_factory):
        super().__init__()
        self.bundle = bundle
        self.native_factory = native_factory
        self.conditional_factory = conditional_factory

    def forward(self, state, time, condition, conditional=False):
        if conditional:
            velocity = self.conditional_factory(self.bundle, condition)
        else:
            velocity = self.native_factory(condition, differentiable=torch.is_grad_enabled())
        return velocity(state, time)


class TimeLocalExecution:
    """Hold an immutable parameter anchor without registering extra trainables.

    Construct after loading the common anchor and moving the bundle to its final
    device. Keep this helper external to the bundle's Module tree. The caller
    may use ``velocity_function`` for inference/repair and
    ``conditional_velocity_function`` for paired RF. Never silently route one
    of those paths around the gate. No parameter data is copied in a forward.
    Registered parameter objects are explicitly restored, including when the
    same module is reachable through multiple AR/DiT alias paths.
    """

    def __init__(self, bundle, *, maximum_trainable_time, conditional_factory=None):
        cutoff = float(maximum_trainable_time)
        if not math.isfinite(cutoff) or not 0 < cutoff < 1:
            raise ValueError('A finite flow-time cutoff strictly inside (0, 1) is required.')
        if conditional_factory is None:
            from .native_paired_rf import paired_rf_velocity_function
            conditional_factory = paired_rf_velocity_function
        self.cutoff = cutoff
        self.forward_module = _ExecutionForward(bundle, bundle.velocity_function, conditional_factory)
        self.current = {name: p for name, p in self.forward_module.named_parameters() if p.requires_grad}
        if not self.current:
            raise ValueError('Time-local execution requires registered trainable student parameters.')
        self.anchor = {name: p.detach().clone() for name, p in self.current.items()}
        active_ids = {id(p) for p in self.current.values()}
        self._restore_slots = [(module, name, parameter)
            for module in self.forward_module.modules()
            for name, parameter in module._parameters.items()
            if parameter is not None and id(parameter) in active_ids]
        self.anchor_calls = 0
        self.student_calls = 0

    def _velocity(self, state, time, condition, *, conditional):
        if (time.ndim != 1 or time.numel() != state.shape[0]
                or not torch.isfinite(time).all() or not ((time >= 0) & (time <= 1)).all()):
            raise ValueError('Require finite per-example RF times in [0, 1].')
        active = time <= self.cutoff
        if bool(active.any()) != bool(active.all()):
            raise ValueError('Use homogeneous-gate microbatches; mixed gate decisions are not implemented.')
        if bool(active.all()):
            self.student_calls += 1
            return self.forward_module(state, time, condition, conditional)
        self.anchor_calls += 1
        try:
            result = functional_call(self.forward_module, self.anchor,
                (state, time, condition, conditional), tie_weights=True, strict=False)
        finally:
            # With multiple paths to the SAME submodule, installed functional
            # call restoration can leave an anchor tensor in a registered slot.
            # Restore unique owner/slot references, never tensor contents, so
            # the optimizer and the AR alias retain their original leaves.
            for module, name, parameter in self._restore_slots:
                module._parameters[name] = parameter
        # Preserve a legal zero-gradient backward for frozen-time paired RF.
        # Upstream gradients to state/condition remain those of the anchor.
        if torch.is_grad_enabled():
            leaf = next(iter(self.current.values()))
            result = result + leaf.reshape(-1)[0] * 0.
        return result

    def velocity_function(self, condition, *, differentiable):
        def velocity(state, time):
            with torch.set_grad_enabled(differentiable and torch.is_grad_enabled()):
                return self._velocity(state, time, condition, conditional=False)
        return velocity

    def conditional_velocity_function(self, condition):
        return lambda state, time: self._velocity(state, time, condition, conditional=True)

    def receipt(self):
        return dict(parameterization='anchor_plus_flow_time_gated_student_increment_v1',
            maximum_trainable_time=self.cutoff,
            anchor_parameter_tensors=len(self.anchor),
            anchor_parameter_bytes=sum(p.numel() * p.element_size() for p in self.anchor.values()),
            registered_student_leaves_unchanged=True, extra_optimizer_parameters=0,
            high_noise_anchor_calls=self.anchor_calls, low_noise_student_calls=self.student_calls,
            conditioning_inside_gate=True, mixed_gate_microbatches_supported=False,
            inference_requires_anchor_and_threshold=True,
            effectiveness_proven=False)
