"""Audio-only event readout and permutation-invariant structured supervision.

Experimental auxiliary module, not an enabled CLAP trainer or AR input. One
assignment binds every field of each event; trajectories cannot be matched
independently of content. Exact enumeration is bounded by four source slots.
"""
from itertools import permutations
import math

import torch
from torch import nn
import torch.nn.functional as F

from stable_audio_tools.data.model_sceneplan import MAX_DURATION_SEC, MAX_SOURCES
from .data import MAX_KEYFRAMES

DEFAULT_WEIGHTS = {'kind': 1., 'content': 1., 'motion': .5, 'frame_count': .25,
                   'activity': 1., 'frame_time': 1., 'direction': 1.,
                   'log_distance': .25, 'gain': .1}
ASSIGNMENTS = {n: torch.tensor(list(permutations(range(MAX_SOURCES), n)), dtype=torch.long)
               for n in range(1, MAX_SOURCES + 1)}


class AudioEventReadout(nn.Module):
    """Learned source queries see only the unpooled CLAP audio sequence."""

    def __init__(self, *, audio_width=384, width=192, heads=6, layers=2, text_dim=1024):
        super().__init__()
        self.input_projection = nn.Linear(audio_width, width)
        self.queries = nn.Parameter(torch.randn(MAX_SOURCES, width) * .02)
        layer = nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0.,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))
        sizes = {'kind_logits': 4, 'motion_logits': 3, 'frame_count_logits': MAX_KEYFRAMES,
                 'content': text_dim, 'activity_sec': 2, 'gain_db': 1,
                 'keyframe_time_sec': MAX_KEYFRAMES, 'direction_xyz': 3 * MAX_KEYFRAMES,
                 'log_distance': MAX_KEYFRAMES}
        self.output_heads = nn.ModuleDict({key: nn.Linear(width, size) for key, size in sizes.items()})

    def forward(self, sequence, sequence_mask):
        if sequence.ndim != 3 or sequence_mask.shape != sequence.shape[:2] or sequence_mask.dtype != torch.bool:
            raise ValueError('Event readout requires masked audio sequence features')
        if not bool(sequence_mask.any(-1).all()) or not bool(torch.isfinite(sequence).all()):
            raise ValueError('Missing or invalid audio sequence')
        memory = self.input_projection(sequence)
        hidden = self.decoder(self.queries[None].expand(sequence.shape[0], -1, -1), memory,
                              memory_key_padding_mask=~sequence_mask)
        out = {key: head(hidden).float() for key, head in self.output_heads.items()}
        out['activity_sec'] = out['activity_sec'].sigmoid() * MAX_DURATION_SEC
        out['keyframe_time_sec'] = out['keyframe_time_sec'].sigmoid() * MAX_DURATION_SEC
        out['gain_db'] = out['gain_db'].squeeze(-1).sigmoid() * 36 - 24
        out['direction_xyz'] = out['direction_xyz'].reshape(sequence.shape[0], MAX_SOURCES, MAX_KEYFRAMES, 3)
        return out


def _pair_ce(logits, labels):
    values = -logits.float().log_softmax(-1)
    return values[:, :, None, :].expand(-1, -1, MAX_SOURCES, -1).gather(
        -1, labels[:, None, :, None].expand(-1, MAX_SOURCES, -1, 1)).squeeze(-1)


def event_objective(prediction, targets, content_features, *, weights=None):
    """Return example sums and means for matched content/time/space supervision.

    Ground-truth content features must come from a frozen text teacher. All
    labels are detached. The matching cost includes the change from empty to
    occupied kind loss, so its assignment minimizes the same complete loss.
    Empty slots receive kind supervision and no fabricated geometry targets.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if weights.keys() != DEFAULT_WEIGHTS.keys() or any(not math.isfinite(float(w)) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError('Invalid event objective weights')
    device = prediction['kind_logits'].device
    t = {key: value.detach().to(device) for key, value in targets.items()}
    truth = content_features.detach().to(device=device, dtype=torch.float32)
    presence = t['presence']; counts = presence.sum(-1)
    batch = presence.shape[0]
    if presence.dtype != torch.bool or presence.shape != (batch, MAX_SOURCES):
        raise ValueError('Event targets require four masked slots')
    if not bool(((counts >= 1) & (counts <= MAX_SOURCES)).all()):
        raise ValueError('Each native audio scene has one to four events')
    if not torch.equal(presence, torch.arange(MAX_SOURCES, device=device)[None] < counts[:, None]):
        raise ValueError('Target events must be packed before padding')
    if not torch.equal(counts, t['source_count']):
        raise ValueError('Source count disagrees with event masks')
    frame_mask = t['keyframe_mask']; frame_counts = frame_mask.sum(-1)
    if bool((frame_counts[presence] < 1).any()) or bool(frame_mask[~presence].any()):
        raise ValueError('Invalid event keyframe masks')
    if truth.shape != prediction['content'].shape or truth.shape[:2] != presence.shape:
        raise ValueError('Content supervision must align with event slots')
    if any(not bool(torch.isfinite(value).all()) for value in (*prediction.values(), truth)):
        raise ValueError('Non-finite event prediction or text target')
    if not bool((truth[presence].norm(dim=-1) > 0).all()):
        raise ValueError('Real events need nonzero content teacher features')
    pair = {}
    pair['kind'] = _pair_ce(prediction['kind_logits'], t['kind'])
    pair['motion'] = _pair_ce(prediction['motion_logits'], t['motion'].clamp_min(0))
    pair['frame_count'] = _pair_ce(prediction['frame_count_logits'], (frame_counts - 1).clamp_min(0))
    pair['content'] = 1 - torch.einsum('bsd,btd->bst', F.normalize(prediction['content'].float(), dim=-1), F.normalize(truth, dim=-1))
    for name, key, scale in [('activity', 'activity_sec', MAX_DURATION_SEC), ('gain', 'gain_db', 36.)]:
        difference = (prediction[key][:, :, None].float() - t[key][:, None].float()) / scale
        value = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none')
        pair[name] = value.mean(-1) if name == 'activity' else value
    mask = frame_mask[:, None].float()
    denominator = frame_counts[:, None].clamp_min(1)
    pred_direction = F.normalize(prediction['direction_xyz'].float(), dim=-1)
    direction = 1 - (pred_direction[:, :, None] * t['direction_xyz'][:, None].float()).sum(-1)
    pair['direction'] = (direction * mask).sum(-1) / denominator
    differences = {
        'frame_time': (prediction['keyframe_time_sec'][:, :, None].float() - t['keyframe_time_sec'][:, None].float()) / MAX_DURATION_SEC,
        'log_distance': prediction['log_distance'][:, :, None].float() - t['position_spherical'][:, None, :, :, 2].float().clamp_min(1e-12).log(),
    }
    for name, difference in differences.items():
        value = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none')
        pair[name] = (value * mask).sum(-1) / denominator
    empty_cost = -prediction['kind_logits'].float().log_softmax(-1)[..., 0]
    cost = weights['kind'] * (pair['kind'] - empty_cost[:, :, None]) / MAX_SOURCES
    for name, value in pair.items():
        if name != 'kind':
            cost = cost + weights[name] * value / counts[:, None, None]
    target_to_prediction = torch.full((batch, MAX_SOURCES), -1, device=device, dtype=torch.long)
    assigned_kind = torch.zeros((batch, MAX_SOURCES), device=device, dtype=torch.long)
    for n in range(1, MAX_SOURCES + 1):
        rows = (counts == n).nonzero().flatten()
        if not rows.numel():
            continue
        assignments = ASSIGNMENTS[n].to(device)
        columns = torch.arange(n, device=device)
        alternatives = cost.detach()[rows][:, assignments, columns].sum(-1)
        selected = assignments[alternatives.argmin(-1)]
        target_to_prediction[rows[:, None], columns] = selected
        assigned_kind[rows[:, None], selected] = t['kind'][rows, :n]
    components = {'kind': F.cross_entropy(prediction['kind_logits'].float().transpose(1, 2), assigned_kind, reduction='none').mean(-1)}
    for name, value in pair.items():
        if name == 'kind':
            continue
        matched = value.gather(1, target_to_prediction.clamp_min(0)[:, None]).squeeze(1)
        components[name] = (matched * presence).sum(-1) / counts
    per_example = sum(weights[name] * value for name, value in components.items())
    return {'loss': per_example.mean(), 'loss_sum': per_example.sum(), 'example_count': batch,
        'per_example': per_example, 'components_per_example': components,
        'target_to_prediction': target_to_prediction.detach()}
