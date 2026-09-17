"""Read-only multi-parent initialization of one shared AR/Editing-DiT model."""
import json
from pathlib import Path

import torch

from scripts.t2a.experiments.ar_factual_clap_v1 import integration
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash
from scripts.t2a.experiments.ar_instruction_t200_v3 import policy
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import ar_specific_state
from . import model


def protected_identity(reference, *, full_hash):
    p = Path(reference['path']).resolve(strict=True)
    stat = p.stat()
    expected = reference['sha256']
    if full_hash and sha(p) != expected:
        raise RuntimeError('Protected original Editing DiT50k checksum changed')
    return {'path': str(p), 'sha256': expected, 'size': stat.st_size,
        'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def build(cfg):
    policy.check_selection(cfg['base_AR_configuration'])
    reference = cfg['protected_DiT50k']
    before = protected_identity(reference, full_hash=True)
    encoder = cfg['base_AR_configuration']['clap_dependency']['checkpoint']
    module, codec, ar_parent, encoder_identity, ar_transfer = integration.build_from_ar_parent(
        cfg['AR_adapter_initialization'], encoder, preflight_path=cfg['preflight'])
    parent_adapter = fingerprint(ar_specific_state(module.ar))
    # The source file is memory-mapped privately and never written. copy_ puts
    # values into the independently allocated model, not the mapped file.
    original = torch.load(reference['path'], map_location='cpu', weights_only=True, mmap=True)
    if original['schema'] != 'editing_full1M_fit_v1' or original['optimizer_steps'] != 50000:
        raise RuntimeError('Expected the actual completed Editing DiT50k checkpoint')
    state = original['state']
    from stable_audio_tools.models.conditioners import ScenePlan44LocalConditioner
    from stable_audio_tools.models.sceneplan_editing_gain_adapter import install_local_gain_adapter
    local = [m for m in module.diffusion.conditioner.conditioners.values() if isinstance(m, ScenePlan44LocalConditioner)]
    if len(local) != 1:
        raise RuntimeError('Editing DiT needs its trained local gain conditioner')
    install_local_gain_adapter(local[0], mode='provided')
    parameters = dict(module.diffusion.named_parameters())
    if set(parameters) != set(state) or len(state) != 201:
        raise RuntimeError('Editing DiT initialization must cover every one of its201 parameters')
    with torch.no_grad():
        for name, value in state.items():
            dest = parameters[name]
            if dest.shape != value.shape or dest.dtype != value.dtype:
                raise RuntimeError(f'Editing DiT parameter shape/dtype changed: {name}')
            dest.copy_(value)
            if not torch.equal(dest, value):
                raise RuntimeError(f'Editing DiT parameter copy was not exact: {name}')
    if fingerprint(ar_specific_state(module.ar)) != parent_adapter:
        raise RuntimeError('Editing DiT initialization overwrote AR-specific learned adapters')
    shared = module.diffusion.model.model.transformer
    if module.ar.shared_transformer is not shared:
        raise RuntimeError('AR and Editing DiT do not share the same Transformer object')
    clap_payload = torch.load(encoder['path'], map_location='cpu', weights_only=True, mmap=True)
    model.attach(module, clap_payload['contract']['readout'], clap_payload['readout']['model'])
    module.planning_pretrain = False
    # A new joint objective has multiple weight parents; optimizer/scheduler
    # start at new update0. This is explicitly not a resume of either parent.
    module.requires_grad_(True)
    module.ar.source_clap_model.requires_grad_(False).eval()
    module.ar.instruction_conditioner.model.requires_grad_(False).eval()
    module.ar.source_semantic_bridge.source_to_caption.requires_grad_(False)
    module.ar.activation_checkpointing = bool(cfg['performance']['AR_activation_checkpointing'])
    groups, scope = optimizer_groups(module, cfg)
    after = protected_identity(reference, full_hash=False)
    if before != after:
        raise RuntimeError('Original checkpoint metadata changed while loading')
    provenance = {'schema': model.CONTRACT, 'protected_DiT50k': before,
        'all201_DiT_parameters_copied_exactly': True,
        'AR_adapters_preserved_sha256': parent_adapter['sha256'],
        'AR_adapter_initialization': cfg['AR_adapter_initialization'],
        'CLAP_checkpoint_sha256': encoder_identity['sha256'],
        'CLAP_readout_same_checkpoint': True,
        'shared_transformer_same_object': True,
        'shared_parameter_count': sum(p.numel() for p in shared.layers.parameters()),
        'new_joint_optimizer_clock': 0, 'parent_optimizers_transferred': False,
        'frozen_Qwen_sha256': state_hash(module.ar.instruction_conditioner.model),
        'frozen_CLAP_sha256': state_hash(module.ar.source_clap_model),
        'frozen_dependency_loader': ar_transfer['encoder_dependency'],
        'optimizer_scope': scope, 'quality_gate_passed': False}
    return module, codec, groups, provenance


def optimizer_groups(module, cfg):
    ar = module.ar
    shared = list(ar.shared_transformer.layers.parameters())
    structure = list(ar.source_structure.parameters()) + list(ar.plan_adapter.slot_residual.parameters())
    occupied = {id(p) for p in shared + structure}
    specific = []
    for submodule in (ar.source_audio_adapter, ar.plan_adapter, ar.source_semantic_bridge):
        for p in submodule.parameters():
            if p.requires_grad and id(p) not in occupied:
                specific.append(p); occupied.add(id(p))
    specific += [ar.source_audio_type_embedding, ar.plan_type_embedding]
    occupied.update(id(p) for p in specific)
    remaining = [p for p in module.diffusion.parameters() if p.requires_grad and id(p) not in occupied]
    sets = {'AR_adapters': specific, 'shared_Transformer': shared,
        'Editing_DiT_adapters_and_conditioning': remaining, 'structured_heads': structure}
    names = {id(p): n for n, p in module.named_parameters()}
    flat = [p for values in sets.values() for p in values]
    if len({id(p) for p in flat}) != len(flat) or {id(p) for p in flat} != {id(p) for p in module.parameters() if p.requires_grad}:
        raise RuntimeError('Every trainable parameter must belong to exactly one optimizer group')
    groups = [{'params': values, 'lr': cfg['learning_rates'][name],
        'initial_lr': cfg['learning_rates'][name], 'group_name': name} for name, values in sets.items()]
    scope = {name: {'parameters': sum(p.numel() for p in values),
        'names': [names[id(p)] for p in values]} for name, values in sets.items()}
    return groups, scope


def assert_separate_output(output, protected):
    output = Path(output).resolve()
    original = Path(protected['path']).resolve(strict=True)
    # Reject the entire checkpoint archive subtree containing the protected file.
    protected_root = original.parents[2]
    if output == original or output.is_relative_to(protected_root):
        raise RuntimeError('New joint checkpoints must use an independent run directory')
    return output
