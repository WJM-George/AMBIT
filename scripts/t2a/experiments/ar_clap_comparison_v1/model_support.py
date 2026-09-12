"""Common AR-only computation and named state transfer for three CLAP arms.

This module does not launch training or select an encoder. The latent-only arm
keeps the real source-audio prefix and drops only the CLAP semantic bridge.
Retained weights and AdamW states come from the same AR-only parent. This is
an adaptation comparison, not a comparison of training from scratch.
"""
import copy
import json
from pathlib import Path
import types

import torch

from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_instruction_t200_v2.ar_only import GROUP_NAMES, ce_sum
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import (
    ar_specific_state, load_ar_specific,
)


def forward(self, *, source_foa_latent, source_attention_mask, plan_input_ids,
            plan_attention_mask, raw_edit_requests):
    encoder = self.ar.source_clap_model
    features = None if encoder is None else encoder.source_features(
        source_foa_latent, source_attention_mask)
    context, mask = self.ar.encode_edit_instructions(
        raw_edit_requests, device=source_foa_latent.device)
    return self.ar(source_foa_latent, source_attention_mask, plan_input_ids,
                   plan_attention_mask, context, mask,
                   source_clap_features=features, return_source_contrastive_query=False)


def configure(module, cfg):
    """Select only parameters reached by AR CE, including native latent-only."""
    groups, counts = native.optimizer_groups(
        module, ar_lr=cfg['ar_lr'], shared_lr=cfg['shared_lr'], dit_lr=cfg['dit_lr'])
    names = {id(p): name for name, p in module.named_parameters()}
    old_names = [[names[id(p)] for p in g['params']] for g in groups]
    ar = module.ar
    allowed_modules = [ar.source_audio_adapter, ar.plan_adapter,
                       ar.shared_transformer.layers, ar.editing_dit.to_cond_embed,
                       ar.instruction_conditioner]
    if ar.source_clap_model is not None:
        if ar.source_semantic_mode != 'clap44_audio_caption_aux':
            raise ValueError('CLAP arm must use the native CLAP44 feature bridge')
        bridge = ar.source_semantic_bridge
        allowed_modules += [bridge.global_projection, bridge.sequence_projection]
    elif ar.source_semantic_mode != 'latent_only' or list(ar.source_semantic_bridge.parameters()):
        raise ValueError('No-CLAP arm requires the native parameter-free latent-only bridge')
    allowed = {id(p) for item in allowed_modules for p in item.parameters() if p.requires_grad}
    allowed.update((id(ar.source_audio_type_embedding), id(ar.plan_type_embedding)))
    module.requires_grad_(False)
    for p in module.parameters():
        p.requires_grad_(id(p) in allowed)
    if ar.source_clap_model is not None:
        assert not any(p.requires_grad for p in ar.source_clap_model.parameters())
        assert not any(p.requires_grad for p in ar.source_semantic_bridge.source_to_caption.parameters())
    assert not any(p.requires_grad for p in ar.instruction_conditioner.model.parameters())
    selected = [{**group, 'group_name': name,
                 'params': [p for p in group['params'] if p.requires_grad]}
                for name, group in zip(GROUP_NAMES, groups, strict=True)]
    assert all(g['params'] for g in selected)
    assert {id(p) for g in selected for p in g['params']} == allowed
    new_names = [[names[id(p)] for p in g['params']] for g in selected]
    assert len(new_names[1]) == 180 and len(new_names[2]) == 6
    parameters = dict(module.named_parameters())
    audit = {
        'old_group_names': [g['group_name'] for g in groups],
        'old_parameter_names': old_names, 'new_group_names': list(GROUP_NAMES),
        'new_parameter_names': new_names, 'native_counts': counts,
        'trainable_parameters': {g['group_name']: sum(p.numel() for p in g['params']) for g in selected},
        'removed_parameter_names': sorted(set(sum(old_names, [])) - set(sum(new_names, []))),
        'parameter_shapes': {name: list(parameters[name].shape) for name in sum(old_names, [])},
        'CLAP_features_enabled': ar.source_clap_model is not None,
        'source_audio_prefix_retained': True,
    }
    module.forward = types.MethodType(forward, module)
    return selected, audit


def restore_common_weights(module, parent):
    """Load native state strictly after explicitly omitting absent CLAP bridge keys."""
    contract = parent['run_contract']
    if (contract['training_mode'] != 'ar_pretrain'
            or contract.get('training_objective') != 'AR_CE_ONLY'
            or contract['variant'] != 'global_and_sequence'):
        raise ValueError('Three-arm adaptation requires the common AR-only CLAP parent')
    saved = parent['editing_ar_specific_state_dict']
    expected = set(ar_specific_state(module.ar))
    omitted = sorted(set(saved) - expected)
    assert expected <= set(saved), 'New AR adapter keys have no parent weights'
    if module.ar.source_clap_model is None:
        assert omitted and all(name.startswith('source_semantic_bridge.') for name in omitted)
    else:
        assert not omitted
    selected = {key: saved[key] for key in saved if key in expected}
    encoder = module.ar.source_clap_model
    frozen_before = None if encoder is None else fingerprint(encoder.state_dict())
    module.diffusion.load_state_dict(parent['diffusion_state_dict'], strict=True)
    load_ar_specific(module.ar, selected)
    after = {'diffusion': fingerprint(module.diffusion.state_dict()),
             'AR_adapters': fingerprint(ar_specific_state(module.ar))}
    assert after['diffusion'] == fingerprint(parent['diffusion_state_dict'])
    assert after['AR_adapters'] == fingerprint(selected)
    assert frozen_before == (None if encoder is None else fingerprint(encoder.state_dict()))
    return {'common_weights_exact': True, 'fingerprints': after,
            'omitted_absent_CLAP_bridge_keys': omitted,
            'external_encoder_preserved': True, 'parent_payload_mutated': False}


def optimizer_from_ar_only_parent(parent, destination_audit):
    """Reuse an AR-only parent's moments by name; never reproject as an AR+RF state."""
    contract = parent['run_contract']
    if contract.get('training_objective') != 'AR_CE_ONLY':
        raise ValueError('This transfer accepts an already AR-only optimizer')
    reference = contract['optimizer_scope']
    path = Path(reference['audit']).resolve(strict=True)
    assert file_sha256(path) == reference['audit_sha256']
    original_audit = json.loads(path.read_text())
    original = parent['optimizer']; original_groups = original['param_groups']
    assert [g['group_name'] for g in original_groups] == original_audit['new_group_names'] == list(GROUP_NAMES)
    assert destination_audit['new_group_names'] == list(GROUP_NAMES)
    by_name = {}
    for group, names in zip(original_groups, original_audit['new_parameter_names'], strict=True):
        for index, name in zip(group['params'], names, strict=True):
            assert name not in by_name and index in original['state']
            state = original['state'][index]
            assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
            by_name[name] = state
    assert len(by_name) == len(original['state'])
    requested = sum(destination_audit['new_parameter_names'], [])
    assert len(set(requested)) == len(requested) and set(requested) <= set(by_name)
    removed = sorted(set(by_name) - set(requested))
    if destination_audit['CLAP_features_enabled']:
        assert not removed
    else:
        assert removed and all(name.startswith('ar.source_semantic_bridge.') for name in removed)
    projected = {'state': {}, 'param_groups': []}; index = 0
    for old, names in zip(original_groups, destination_audit['new_parameter_names'], strict=True):
        group = copy.deepcopy({k: v for k, v in old.items() if k != 'params'})
        group['params'] = []
        for name in names:
            state = by_name[name]; shape = destination_audit['parameter_shapes'][name]
            assert list(state['exp_avg'].shape) == list(state['exp_avg_sq'].shape) == shape
            projected['state'][index] = state
            group['params'].append(index); index += 1
        projected['param_groups'].append(group)
    selected = {name: by_name[name] for name in requested}
    reloaded = {name: projected['state'][i] for i, name in enumerate(requested)}
    assert fingerprint(selected) == fingerprint(reloaded)
    unchanged = original_audit['new_parameter_names'] == destination_audit['new_parameter_names']
    if unchanged:
        assert fingerprint(projected) == fingerprint(original)
    return original if unchanged else projected, {
        'scope': 'Explicit adaptation fork; retained AR-only AdamW state and hyperparameters are unchanged.',
        'selected_states_exact': True, 'selected_state_fingerprint': fingerprint(selected)['sha256'],
        'parent_optimizer_returned_unchanged': unchanged,
        'omitted_absent_CLAP_bridge_parameters': removed,
        'selected_parameters': len(requested),
        'scheduler_RNG_data_not_modified_by_this_function': True,
    }
