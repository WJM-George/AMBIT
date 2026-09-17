"""One source assignment binds edit-object and post-edit field supervision.

All ground truth enters here, after the audio/request-only forward pass.
Post-edit events are never independently rematched to hide an identity swap.
"""
import torch
import torch.nn.functional as F

from scripts.t2a.experiments.clap_scene_supervision_v1.events import event_objective, DEFAULT_WEIGHTS
from stable_audio_tools.data.model_sceneplan import MAX_DURATION_SEC
from .data import FIELDS

DEFAULT_LOSS_WEIGHTS = {'source': .1, 'binding': .1, 'preservation': .1, 'modification': .1, 'target_presence': .1}


@torch.no_grad()
def content_targets(supervision, teacher, device):
    # Deduplicate within this batch only. No uncheckpointed cross-step cache.
    texts = list(dict.fromkeys(text for rows in (
        supervision['source_content_texts'], supervision['post_content_texts'])
        for row in rows for text in row if text))
    vectors = teacher(texts, device)
    lookup = {text: i for i, text in enumerate(texts)}
    batch = len(supervision['source_content_texts'])
    source = vectors.new_zeros(batch, 4, vectors.shape[-1])
    target = vectors.new_zeros(batch, 5, vectors.shape[-1])
    for dst, key in ((source, 'source_content_texts'), (target, 'post_content_texts')):
        for i, row in enumerate(supervision[key]):
            for j, text in enumerate(row):
                if text:
                    dst[i, j] = vectors[lookup[text]]
    return source.detach(), target.detach()


def _ce(logits, target):
    return F.cross_entropy(logits.float().transpose(1, 2), target.long(), reduction='none')


def paired_post_components(prediction, targets, content):
    frames = targets['keyframe_mask'].sum(-1)
    mask = targets['keyframe_mask'].float()
    components = {
        'kind': _ce(prediction['kind_logits'], targets['kind']),
        'content': 1 - (F.normalize(prediction['content'].float(), dim=-1) * F.normalize(content.float(), dim=-1)).sum(-1),
        'motion': _ce(prediction['motion_logits'], targets['motion'].clamp_min(0)),
        'frame_count': _ce(prediction['frame_count_logits'], (frames - 1).clamp_min(0)),
    }
    for field, key, scale in (('activity', 'activity_sec', MAX_DURATION_SEC), ('gain', 'gain_db', 36.)):
        difference = (prediction[key].float() - targets[key].float()) / scale
        value = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none')
        components[field] = value.mean(-1) if field == 'activity' else value
    direction = 1 - (F.normalize(prediction['direction_xyz'].float(), dim=-1) * targets['direction_xyz'].float()).sum(-1)
    components['direction'] = (direction * mask).sum(-1) / frames.clamp_min(1)
    for field, difference in (
        ('frame_time', (prediction['keyframe_time_sec'].float() - targets['keyframe_time_sec'].float()) / MAX_DURATION_SEC),
        ('log_distance', prediction['log_distance'].float() - targets['position_spherical'][..., 2].float().clamp_min(1e-12).log()),
    ):
        value = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none')
        components[field] = (value * mask).sum(-1) / frames.clamp_min(1)
    return components


def structured_objective(outputs, supervision, source_content, post_content, *, weights=None):
    weights = DEFAULT_LOSS_WEIGHTS if weights is None else weights
    if weights.keys() != DEFAULT_LOSS_WEIGHTS.keys():
        raise ValueError('Structured loss weights must be explicitly complete')
    source = event_objective(outputs['source'], supervision['source_targets'], source_content)
    mapping = source['target_to_prediction']
    device = mapping.device
    batch = len(mapping)
    presence = supervision['source_targets']['presence'].to(device)
    rows = torch.arange(batch, device=device)[:, None].expand_as(mapping)
    assigned_actions = torch.zeros(batch, 5, device=device, dtype=torch.long)
    assigned_kind = torch.zeros_like(assigned_actions)
    actions = supervision['actions'].to(device)
    post = supervision['post_targets']
    # Only occupied OLD sources are assigned. A removed source keeps its old
    # assignment, receiving REMOVE and post-edit EMPTY at that very slot.
    assigned_actions[rows[presence], mapping[presence]] = actions[:, :4][presence]
    assigned_kind[rows[presence], mapping[presence]] = post['kind'][:, :4][presence]
    assigned_actions[:, 4] = actions[:, 4]
    assigned_kind[:, 4] = post['kind'][:, 4]
    object_labels = supervision['edit_target'].to(device)
    expanded_mapping = torch.cat((mapping, torch.full((batch, 1), 4, device=device, dtype=torch.long)), dim=1)
    object_slots = expanded_mapping.gather(1, object_labels[:, None]).squeeze(1)
    if bool((object_slots < 0).any()):
        raise ValueError('Edited source is not present in the audio supervision')
    target_ce = F.cross_entropy(outputs['target_logits'].float(), object_slots, reduction='none')
    action_ce = _ce(outputs['operation_logits'], assigned_actions).mean(-1)
    target_presence_ce = _ce(outputs['target']['kind_logits'], assigned_kind).mean(-1)
    indices = expanded_mapping.clamp_min(0)
    aligned = {}
    for key, value in outputs['target'].items():
        shape = (*indices.shape, *((1,) * (value.ndim - 2)))
        aligned[key] = value.gather(1, indices.reshape(shape).expand(-1, -1, *value.shape[2:]))
    components = paired_post_components(aligned, post, post_content)
    values = torch.stack([components[k] * DEFAULT_WEIGHTS[k] for k in FIELDS], dim=-1)
    valid = post['presence'].bool()[..., None].expand_as(values)
    preserve_mask = valid & supervision['preserve_fields'].bool()
    modify_mask = valid & ~supervision['preserve_fields'].bool()
    preservation = (values * preserve_mask).sum((1, 2)) / preserve_mask.sum((1, 2)).clamp_min(1)
    modification = (values * modify_mask).sum((1, 2)) / modify_mask.sum((1, 2)).clamp_min(1)
    pieces = {'source': source['per_example'], 'binding': target_ce + action_ce,
        'preservation': preservation, 'modification': modification, 'target_presence': target_presence_ce}
    per_example = sum(float(weights[k]) * value for k, value in pieces.items())
    metrics = {name: value.detach().sum() for name, value in pieces.items()}
    metrics.update(edit_target_correct=(outputs['target_logits'].argmax(-1) == object_slots).float().sum(),
        operation_correct=(outputs['operation_logits'].argmax(-1) == assigned_actions).float().mean(-1).sum(),
        source_count_correct=((outputs['source']['kind_logits'].argmax(-1) != 0).sum(-1)
            == supervision['source_targets']['source_count']).float().sum())
    return {'loss_sum': per_example.sum(), 'per_example': per_example,
        'example_count': batch, 'metrics_sums': metrics, 'assignment': mapping,
        'edit_object_slots': object_slots, 'post_kind_by_predicted_slot': assigned_kind}
