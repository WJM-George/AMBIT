"""Optional source-gain condition for the existing 256-channel Editing control.

Install after strict legacy checkpoint loading. A zero-initialized 32-parameter
projection preserves the original output. It must be trained and acoustically
validated before promotion. Runtime gains come only from the desired new plan.
"""
from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def attach_desired_gain_tracks(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy compiled metadata and add gain tracks without copying the reference."""
    plan = metadata['model_sceneplan']
    controls = metadata['sceneplan_44']
    events = torch.as_tensor(controls['source_event_frame_ids'])
    valid = torch.as_tensor(controls['frame_valid_mask'], dtype=torch.bool, device=events.device)
    if events.ndim != 2 or events.shape[0] != 4 or valid.shape != events.shape[1:]:
        raise ValueError('Gain tracks require aligned four-slot event and valid-frame controls')
    gains = torch.zeros(events.shape, dtype=torch.float32, device=events.device)
    seen = set()
    for source in plan['sources']:
        sid = str(source['source_id'])
        if sid not in {'source_0', 'source_1', 'source_2', 'source_3'} or sid in seen:
            raise ValueError('Gain conditioning requires unique persistent source slots')
        seen.add(sid); slot = int(sid[-1]); value = float(source['gain_db'])
        if not -24. <= value <= 12.:
            raise ValueError('Desired gain_db is outside the frozen ScenePlan range')
        gains[slot] = ((events[slot] == slot + 1) & valid).to(gains.dtype) * value
    return {**metadata, 'sceneplan_44': {**controls, 'source_gain_db_frames': gains}}


def _forward_with_gain(self, controls, device=None):
    original = self._editing_gain_original_forward(controls, device=device)
    encoded = original[0]
    if encoded.ndim != 3 or encoded.shape[1] != 256:
        raise ValueError('The gain adapter requires the existing [B,256,T] control interface')
    if len(controls) != encoded.shape[0]:
        raise ValueError('Control batch and gain batch differ')
    gain_rows = []
    for value in controls:
        events = torch.as_tensor(value['source_event_frame_ids'], device=encoded.device)
        valid = torch.as_tensor(value['frame_valid_mask'], device=encoded.device, dtype=torch.bool)
        if bool(value.get('cfg_unknown', False)):
            # Do not retain gain information when the structured condition is unknown.
            gains = torch.zeros_like(events, dtype=self.gain_projection_weight.dtype)
        else:
            if 'source_gain_db_frames' not in value:
                raise ValueError('Known gain-conditioned input requires complete desired gain tracks')
            gains = torch.as_tensor(value['source_gain_db_frames'], device=encoded.device,
                                    dtype=self.gain_projection_weight.dtype)
            if gains.shape != events.shape or not bool(torch.isfinite(gains).all()):
                raise ValueError('Gain track shape or finite-value contract failed')
            active = events.gt(0) & valid.unsqueeze(0)
            if bool((gains.masked_select(~active) != 0).any()):
                raise ValueError('Inactive and padded frames must have zero gain tracks')
            if bool(((gains < -24.) | (gains > 12.)).any()):
                raise ValueError('Gain tracks exceed the frozen ScenePlan range')
            gains = torch.where(active, gains / 24., torch.zeros_like(gains))
            if self.editing_gain_mode == 'zero':
                gains = torch.zeros_like(gains)
        gain_rows.append(F.pad(gains, (0, encoded.shape[-1] - gains.shape[-1])))
    gain_input = torch.stack(gain_rows).unsqueeze(-1)  # [B,4,T,1]
    trajectory_delta = F.linear(gain_input, self.gain_projection_weight).to(encoded.dtype)
    delta = torch.cat([torch.zeros_like(trajectory_delta), trajectory_delta], dim=-1)
    delta = delta.permute(0, 1, 3, 2).reshape_as(encoded)
    return [encoded + delta, *original[1:]]


def install_local_gain_adapter(local_conditioner: nn.Module, *, mode: str = 'provided') -> nn.Module:
    """Add a zero projection after loading a legacy ScenePlan44LocalConditioner.

    ``zero`` is a matched ablation: identical module and parameter count, with
    the gain input set to zero. No legacy parameter or buffer is overwritten.
    """
    if mode not in {'provided', 'zero'}:
        raise ValueError('Gain mode must be provided or zero')
    if hasattr(local_conditioner, 'gain_projection_weight'):
        raise ValueError('Gain adapter is already installed')
    encoder = local_conditioner.encoder
    if encoder.output_dim != 256 or encoder.event_embedding_dim != 32 or encoder.trajectory_embedding_dim != 32:
        raise ValueError('Gain adapter requires four existing 32+32 source blocks')
    weight = encoder.trajectory_encoder[0].weight
    # torch.zeros consumes no RNG state, preserving the matched initialization.
    local_conditioner.register_parameter('gain_projection_weight', nn.Parameter(weight.new_zeros((32, 1))))
    local_conditioner.editing_gain_mode = mode
    local_conditioner._editing_gain_original_forward = local_conditioner.forward
    local_conditioner.forward = MethodType(_forward_with_gain, local_conditioner)
    return local_conditioner


def _metadata_with_gain(self, *args, **kwargs):
    return [attach_desired_gain_tracks(row) for row in self._editing_gain_original_metadata(*args, **kwargs)]


def install_pipeline_gain_metadata(pipeline):
    """Attach desired gains wherever the pipeline builds source/new-plan inputs."""
    if hasattr(pipeline, '_editing_gain_original_metadata'):
        raise ValueError('Pipeline gain metadata is already installed')
    pipeline._editing_gain_original_metadata = pipeline._conditioning_metadata
    pipeline._conditioning_metadata = MethodType(_metadata_with_gain, pipeline)
    return pipeline
