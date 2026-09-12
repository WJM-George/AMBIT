"""Student-visible summaries of an actual continuous-generation query.

No audio convention, transcript, reward, candidate rollout or random seed is
part of the features. A clean prediction uses only the current query velocity;
it must not be replaced with a subsequently completed teacher output.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExecutionQuery:
    state: torch.Tensor
    clean: torch.Tensor
    time: torch.Tensor
    mask: torch.Tensor
    model_version: int

    def __post_init__(self):
        z = self.state
        if (z.ndim != 3 or self.clean.shape != z.shape or self.mask.shape != (z.shape[0], z.shape[2])
                or self.mask.dtype != torch.bool or self.time.shape != (z.shape[0],)
                or not self.mask.any(-1).all() or type(self.model_version) is not int or self.model_version < 0):
            raise ValueError('execution feedback requires aligned actual query tensors and a model version')
        if any(value.requires_grad or not torch.isfinite(value).all() for value in (z, self.clean, self.time)):
            raise ValueError('on-policy query observations must be finite and detached')
        if not ((self.time > 0) & (self.time <= 1)).all():
            raise ValueError('feedback must precede the final diffusion endpoint')


def execution_query_features(query, *, bins=4):
    """Masked temporal moments, invariant to padding and independent of seeds."""
    if type(bins) is not int or bins < 1 or (query.mask.sum(-1) < bins).any():
        raise ValueError('each temporal bin needs an observed query frame')
    mask = query.mask.to(query.state.device)
    position = mask.long().cumsum(-1) - 1
    groups = (position * bins // mask.sum(-1, keepdim=True)).clamp(0, bins - 1)
    summaries = []
    for value in (query.state, query.clean):
        for index in range(bins):
            keep = (mask & (groups == index))[:, None]
            count = keep.sum(-1).clamp_min(1)
            mean = value.float().masked_fill(~keep, 0.).sum(-1) / count
            variance = ((value.float() - mean[..., None]).square().masked_fill(~keep, 0.).sum(-1) / count)
            summaries.extend((mean, variance.clamp_min(0.).sqrt()))
    return torch.cat([*summaries, query.time.to(query.state.device).float()[:, None]], -1)
