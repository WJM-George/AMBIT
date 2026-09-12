"""AR-only objective and audited projection of an existing native optimizer.

The native AR forward is retained. RF-exclusive weights and the unused
contrastive query head are frozen; CLAP and the Qwen backbone stay frozen.
"""
import copy
import types

import torch
import torch.nn.functional as F

from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint

GROUP_NAMES = ('editing_ar_adapters', 'shared_transformer_blocks', 'editing_ar_instruction_projections')


def forward(self, *, source_foa_latent, source_attention_mask, plan_input_ids,
            plan_attention_mask, raw_edit_requests):
    features = self.ar.source_clap_model.source_features(source_foa_latent, source_attention_mask)
    context, mask = self.ar.encode_edit_instructions(raw_edit_requests, device=source_foa_latent.device)
    return self.ar(source_foa_latent, source_attention_mask, plan_input_ids,
                   plan_attention_mask, context, mask, source_clap_features=features,
                   return_source_contrastive_query=False)


def ce_sum(logits, labels):
    # Identical to the native _losses AR term; no RF target enters this loss.
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                           ignore_index=-100, reduction='sum')


def configure(module, cfg):
    native_groups, native_counts = native.optimizer_groups(
        module, ar_lr=cfg['ar_lr'], shared_lr=cfg['shared_lr'], dit_lr=cfg['dit_lr'])
    names = {id(p): name for name, p in module.named_parameters()}
    old_names = [[names[id(p)] for p in group['params']] for group in native_groups]
    ar = module.ar
    allowed_modules = [ar.source_audio_adapter, ar.plan_adapter,
                       ar.source_semantic_bridge.global_projection,
                       ar.source_semantic_bridge.sequence_projection,
                       ar.shared_transformer.layers, ar.editing_dit.to_cond_embed,
                       ar.instruction_conditioner]
    allowed = {id(p) for item in allowed_modules for p in item.parameters() if p.requires_grad}
    allowed.update((id(ar.source_audio_type_embedding), id(ar.plan_type_embedding)))
    module.requires_grad_(False)
    for p in module.parameters():
        p.requires_grad_(id(p) in allowed)
    assert not any(p.requires_grad for p in ar.source_semantic_bridge.source_to_caption.parameters())
    assert not any(p.requires_grad for p in ar.source_clap_model.parameters())
    assert not any(p.requires_grad for p in ar.instruction_conditioner.model.parameters())
    groups = []
    for name, original in zip(GROUP_NAMES, native_groups, strict=True):
        groups.append({**original, 'group_name': name,
                       'params': [p for p in original['params'] if p.requires_grad]})
    assert all(g['params'] for g in groups)
    assert {id(p) for g in groups for p in g['params']} == allowed
    new_names = [[names[id(p)] for p in g['params']] for g in groups]
    parameters = dict(module.named_parameters())
    # Pin the actual P10 architecture. A future conditional/shared parameter
    # needs an explicit reachability audit, not automatic permission to train.
    assert len(new_names[1]) == 180 and len(new_names[2]) == 6
    counts = {g['group_name']: sum(p.numel() for p in g['params']) for g in groups}
    audit = {'old_group_names': [g['group_name'] for g in native_groups],
             'old_parameter_names': old_names, 'new_parameter_names': new_names,
             'new_group_names': list(GROUP_NAMES), 'native_counts': native_counts,
             'trainable_parameters': counts,
             'removed_parameter_names': sorted(set(sum(old_names, [])) - allowed_names(new_names)),
             'parameter_shapes': {n: list(parameters[n].shape) for n in sum(old_names, [])}}
    module.forward = types.MethodType(forward, module)
    return groups, audit


def allowed_names(groups):
    return set(name for group in groups for name in group)


def project_optimizer(original, audit):
    """Keep every selected AdamW tensor and scalar; reindex by canonical name."""
    old_groups = original['param_groups']
    assert [g['group_name'] for g in old_groups] == audit['old_group_names']
    old_by_name = {}
    for group, names in zip(old_groups, audit['old_parameter_names'], strict=True):
        for index, name in zip(group['params'], names, strict=True):
            assert name not in old_by_name and index in original['state']
            state = original['state'][index]
            assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
            assert list(state['exp_avg'].shape) == list(state['exp_avg_sq'].shape) == audit['parameter_shapes'][name]
            old_by_name[name] = state
    assert len(old_by_name) == len(original['state'])
    result = {'state': {}, 'param_groups': []}; index = 0; selected_before = {}; selected_after = {}
    for old, names, group_name in zip(old_groups, audit['new_parameter_names'], GROUP_NAMES, strict=True):
        group = copy.deepcopy({k: v for k, v in old.items() if k != 'params'})
        group.update(group_name=group_name, params=[])
        for name in names:
            state = old_by_name[name]
            result['state'][index] = state  # Reuse the exact immutable loaded tensors.
            selected_before[name] = state; selected_after[name] = result['state'][index]
            group['params'].append(index); index += 1
        result['param_groups'].append(group)
    before = fingerprint(selected_before); after = fingerprint(selected_after)
    assert before == after
    receipt = {'selected_states_exact': True, 'selected_tensor_elements': before['tensor_elements'],
               'selected_state_fingerprint': before['sha256'], 'selected_parameters': index,
               'removed_parameters': len(old_by_name) - index,
               'removed_parameter_names': audit['removed_parameter_names'],
               'hyperparameters_preserved': True,
               'scope': 'Explicit optimizer projection for a new AR-only fork; not bitwise continuation of the former AR+RF objective.'}
    return result, receipt
