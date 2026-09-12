"""Explicit event-supervision training fork around the unchanged CLAP44 core."""
from contextlib import nullcontext
import copy

import torch
from torch import distributed as dist, nn

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import (
    CLAP44Config, EditingCLAP44, binding_negative_loss, clap44_objective,
)
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import optimizer_and_scheduler
from .events import AudioEventReadout, event_objective


class SceneEventTrainingModel(nn.Module):
    """The detached control trains the same readout without changing its encoder."""

    def __init__(self, encoder_config, readout_config, readout_seed):
        super().__init__()
        self.encoder = EditingCLAP44(CLAP44Config(**encoder_config))
        # CPU-only construction inside a local RNG scope, before moving the model.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(readout_seed))
            self.readout = AudioEventReadout(**readout_config)

    def forward(self, latent, mask, semantic, scene, negatives, *, encoder_aux_enabled):
        audio = self.encoder.encode_audio(latent, mask, return_sequence=True)
        text = self.encoder.encode_text_features(semantic, scene)
        negative = self.encoder.encode_text_features(negatives, negatives)['scene']
        sequence = audio['sequence'] if encoder_aux_enabled else audio['sequence'].detach()
        events = self.readout(sequence, audio['sequence_mask'])
        return audio, text, negative, events


def build_optimizers(model, contract, parent):
    """Inherit every encoder moment/schedule; the new head has its own clock."""
    encoder_opt, encoder_sched = optimizer_and_scheduler(model.encoder, contract['native_config']['training'])
    encoder_opt.load_state_dict(copy.deepcopy(parent['optimizer']))
    encoder_sched.load_state_dict(copy.deepcopy(parent['scheduler']))
    options = contract['readout_optimizer']
    readout_opt = torch.optim.AdamW(model.readout.parameters(), lr=float(options['learning_rate']),
                                    betas=tuple(options['betas']), weight_decay=float(options['weight_decay']))
    readout_sched = torch.optim.lr_scheduler.LambdaLR(readout_opt, lambda _: 1.)
    return {'encoder': encoder_opt, 'readout': readout_opt}, {'encoder': encoder_sched, 'readout': readout_sched}


def content_teacher_targets(batch, teacher, device):
    texts = [text for metadata in batch['event_metadata'] for text in metadata['content_texts']]
    features = teacher(texts, device)
    if features.requires_grad:
        raise RuntimeError('Event content teacher must be frozen')
    result = features.new_zeros((len(batch['labels']), 4, features.shape[-1]))
    offset = 0
    for i, metadata in enumerate(batch['event_metadata']):
        count = len(metadata['content_texts'])
        result[i, :count] = features[offset:offset + count]
        offset += count
    if offset != features.shape[0]:
        raise RuntimeError('Content features do not align with actual scene events')
    return result


def training_step(model, teacher, optimizers, schedulers, batch, *, contract, device, observer=None):
    core = getattr(model, 'module', model)
    training = contract['native_config']['training']
    labels = batch['labels']; count = len(labels)
    if count != 2 * contract['data']['pairs_per_rank']:
        raise RuntimeError('Native CLAP and event loss require complete equal-sized rank batches')
    texts = [row['semantic_text'] for row in labels] + [row['scene_text'] for row in labels]
    texts += batch['negative_scene_texts']
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    with autocast:
        features = teacher(texts, device)
        if features.requires_grad:
            raise RuntimeError('Global CLAP text teacher must be frozen')
        content = content_teacher_targets(batch, teacher, device)
        audio, text, negative, events = model(
            batch['latent'].to(device, non_blocking=True), batch['mask'].to(device, non_blocking=True),
            features[:count], features[count:2 * count], features[2 * count:],
            encoder_aux_enabled=contract['method']['encoder_aux_enabled'])
        native = clap44_objective(core.encoder, audio, text, labels,
                                  scene_weight=float(training['scene_weight']), include_edit_negatives=True)
        binding = binding_negative_loss(audio['scene'], text['scene'], negative, batch['negative_owners'])
        baseline = native['loss'] + float(training['binding_weight']) * binding
        event = event_objective(events, batch['event_targets'], content, weights=contract['method']['event_weights'])
        # Each native rank has the same number of views. DDP averages these
        # per-view means; different source counts do not change rank weighting.
        if event['example_count'] != count:
            raise RuntimeError('Event loss denominator differs from the actual audio-view count')
        total = baseline + float(contract['method']['event_weight']) * event['loss']
    if not bool(torch.isfinite(total)):
        raise RuntimeError('Non-finite CLAP event training loss')
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    total.backward()
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in core.parameters()):
        raise RuntimeError('Missing or non-finite encoder/readout gradient')
    if any(parameter.grad is not None for parameter in teacher.conditioner.model.parameters()):
        raise RuntimeError('Frozen Qwen unexpectedly received a gradient')
    if observer is not None:
        observer(core)
    # Independent clipping keeps the control encoder's native gradient scale.
    norms = {key: torch.nn.utils.clip_grad_norm_(getattr(core, key).parameters(), 1., error_if_nonfinite=True)
             for key in optimizers}
    learning_rates = {key: [group['lr'] for group in value.param_groups] for key, value in optimizers.items()}
    for key in optimizers:
        optimizers[key].step(); schedulers[key].step()
    metrics = {'native_loss': baseline.detach(), 'semantic_loss': native['semantic'].detach(),
               'scene_loss': native['scene'].detach(), 'binding_loss': binding.detach(),
               'event_loss': event['loss'].detach(), 'total_loss': total.detach()}
    metrics.update({f'event_{key}': value.detach().mean() for key, value in event['components_per_example'].items()})
    metrics.update({f'{key}_grad_norm': value.detach() for key, value in norms.items()})
    local = {key: float(value) for key, value in metrics.items()}
    if dist.is_initialized():
        global_values = torch.stack(list(metrics.values()))
        dist.all_reduce(global_values)
        global_values /= dist.get_world_size()
        global_metrics = {key: float(value) for key, value in zip(metrics, global_values)}
    else:
        global_metrics = local
    return {'local': local, 'global': global_metrics, 'learning_rates_used': learning_rates,
            'event_views': count, 'event_count': int(batch['event_targets']['presence'].sum()),
            'assignment': event['target_to_prediction'].cpu().tolist()}
