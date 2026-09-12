"""Current-query conditional responses and a shared candidate decision head.

The generic interface has no audio reward, transcript, future suffix or seed
input. A domain adapter supplies legal alternatives and plan descriptors.
"""
import torch
from torch import nn

from .execution_query import ExecutionQuery, execution_query_features


def execution_response_features(query, clean_predictions, *, bins=4):
    """Compare each candidate's current clean prediction to candidate zero.

    Candidate zero is recomputed in the same forward environment as the other
    alternatives. Normalize response direction and retain relative magnitude;
    padding contributes to neither moments nor amplitude.
    """
    values = clean_predictions
    if (values.ndim != 4 or values.shape[0] != query.state.shape[0]
            or values.shape[2:] != query.state.shape[1:] or values.shape[1] < 1
            or values.requires_grad or not bool(torch.isfinite(values).all())):
        raise ValueError('conditional responses require aligned detached current-query clean predictions')
    keep = query.mask[:, None].to(values.device)
    count = keep.sum(-1).clamp_min(1) * values.shape[2]
    def rms(value):
        return (value.float().square().masked_fill(~keep, 0.).sum((1, 2)) / count[:, 0]).sqrt()
    base = rms(values[:, 0]).clamp_min(1e-6)
    result = []
    for index in range(values.shape[1]):
        delta = values[:, index] - values[:, 0]
        size = rms(delta)
        normalized = delta / size.clamp_min(1e-6)[:, None, None]
        response = ExecutionQuery(normalized.detach(), normalized.detach(), query.time,
            query.mask, query.model_version)
        moments = execution_query_features(response, bins=bins)[:, :2 * bins * values.shape[2]]
        result.append(torch.cat([moments, (size/base).log1p()[:, None]], -1))
    return torch.stack(result, 1)


class CandidateResponseHead(nn.Module):
    """One scorer shared by all candidates, with initial original-action bias."""

    original_action_prior = .1

    def __init__(self, context_dim, attribute_dim, query_dim, response_dim, action_dim,
            *, width=128, initialization_seed=18325):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, width))
            self.attributes = nn.Linear(attribute_dim, width)
            self.query = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, width))
            self.response = nn.Sequential(nn.LayerNorm(response_dim), nn.Linear(response_dim, width))
            self.base_plan = nn.Linear(action_dim, width)
            self.action_change = nn.Linear(action_dim, width)
            self.output = nn.Linear(width, 1)
            nn.init.normal_(self.output.weight, std=.0001)
            nn.init.zeros_(self.output.bias)
        if 2 * self.output.weight.detach().abs().sum() >= self.original_action_prior:
            raise ValueError('candidate scorer must initially preserve the original action for every query')

    @torch.no_grad()
    def calibrate_initial_action_margin(self, *, margin=.02):
        """Keep learned features but start from the original greedy behavior.

        Every candidate shares the output bias and a tanh-bounded hidden
        state, so its score difference is at most 2 * ||output.weight||_1.
        This is an initialization transform before optimizer construction,
        not an inference-time veto. All parameters remain trainable.
        """
        import math
        if not math.isfinite(margin) or not 0 < margin < self.original_action_prior:
            raise ValueError('initial decision margin must be positive and below the original-action prior')
        before = 2 * float(self.output.weight.double().abs().sum())
        if not math.isfinite(before):
            raise ValueError('the response output weights must be finite')
        bound = (self.original_action_prior - margin) * (1 - 1e-6)
        scale = min(1., bound / before) if before else 1.
        self.output.weight.mul_(scale)
        after = 2 * float(self.output.weight.double().abs().sum())
        if after > self.original_action_prior - margin:
            raise ValueError('representable initialization failed the declared decision margin')
        return {'contract': 'bounded_original_action_initialization_v1', 'requested_margin': margin,
            'before_pairwise_bound': before, 'after_pairwise_bound': after, 'output_weight_scale': scale,
            'original_action_prior': self.original_action_prior,
            'guaranteed_initial_margin': self.original_action_prior - after,
            'scope': 'initial greedy decision only; no capability or learning guarantee'}

    def forward(self, context, attributes, query, responses, action_descriptors, legal, *,
            input_mode='current_response'):
        if input_mode not in ('request_only', 'query_moments', 'current_response'):
            raise ValueError('unknown current-query response control')
        if responses.shape[:2] != action_descriptors.shape[:2] or legal.shape != responses.shape[:2]:
            raise ValueError('conditional responses must correspond to the actual legal alternatives')
        if not bool(legal[:, 0].all()):
            raise ValueError('the original action must remain available')
        if input_mode == 'request_only':
            query = torch.zeros_like(query)
        if input_mode != 'current_response':
            responses = torch.zeros_like(responses)
        shared = self.context(context.float()) + self.attributes(attributes.float()) + self.query(query.float())
        shared = shared + self.base_plan(action_descriptors[:, 0].float())
        changes = action_descriptors.float() - action_descriptors[:, :1].float()
        hidden = shared[:, None] + self.response(responses.float()) + self.action_change(changes)
        logits = self.output(hidden.tanh())[..., 0]
        prior = torch.zeros_like(logits)
        prior[:, 0] = self.original_action_prior
        return (logits + prior).masked_fill(~legal, -torch.inf)
