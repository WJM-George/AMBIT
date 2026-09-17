"""Load factual CLAP as an explicit frozen Editing AR dependency.

The encoder's real training contract is preserved. Its fresh update clock and
20k weight initialization are not reinterpreted as a native CLAP resume. This
module prepares model transfers; it does not select an encoder, publish an AR
run, copy optimizer states, or authorize training from interface evidence.
"""
import json
from pathlib import Path

import torch

from scripts.t2a.experiments.clap_factual50k_v1.evaluation import (
    load_factual_encoder, native_configuration_view,
)
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import (
    file_sha256, load_clap44_checkpoint,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import (
    ar_specific_state, codec_artifact_sha256, load_ar_specific, load_joint_checkpoint,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import verify_frozen_qwen_runtime

SCHEMA = 'editing_ar_frozen_factual_clap_dependency_v1'


def load_source_encoder(reference, *, preflight_path, device='cpu'):
    """Return a frozen model, its unchanged identity, and an AR dependency record."""
    path = Path(reference['path']).resolve(strict=True)
    sidecar = path.parent / 'TRAIN_CONTRACT.json'
    if file_sha256(sidecar) != reference['contract_sha256']:
        raise RuntimeError('Selected CLAP training contract changed')
    kind = reference['format']
    if kind == 'factual50k':
        model, identity = load_factual_encoder(path, expected_sha256=reference['sha256'], device=device)
        configuration = native_configuration_view(identity)
        clock = {'kind': 'fresh_new_updates', 'new_updates': identity['new_updates'],
                 'initialization_step': identity['initialization_step']}
    elif kind == 'native_clap44':
        model, identity = load_clap44_checkpoint(path, expected_sha256=reference['sha256'], device=device)
        configuration = identity['contract']
        clock = {'kind': 'native_training_step', 'native_step': identity['step']}
    else:
        raise ValueError(f'Unsupported explicit CLAP format: {kind}')
    if identity['step'] != reference['step']:
        raise RuntimeError('Selected CLAP update count changed')
    preflight = json.loads(Path(preflight_path).read_text())
    if (configuration['preflight_sha256'] != file_sha256(preflight_path)
            or preflight.get('status') != 'PASS'
            or [preflight['indices'][s]['rows'] for s in ('train', 'validation', 'test')]
            != [1000000, 20000, 5000]):
        raise RuntimeError('CLAP and AR preflight populations differ')
    if kind == 'factual50k':
        data = identity['contract']['data']
        if (data['index_sha256'] != preflight['indices']['train']['sha256']
                or data['rows'] != 1000000):
            raise RuntimeError('Factual CLAP and AR training indices differ')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('AR source CLAP must be frozen and in evaluation mode')
    record = {
        'schema': SCHEMA, 'format': kind,
        'checkpoint': {key: identity[key] for key in ('path', 'sha256', 'step')},
        'training_contract_path': str(sidecar),
        'training_contract_sha256': reference['contract_sha256'],
        'training_contract_schema': identity['contract']['schema'],
        'training_contract_returned_without_relabeling': True,
        'training_clock': clock, 'preflight_sha256': configuration['preflight_sha256'],
        'model_config': configuration['config']['model'],
        'text_files': configuration['text_files'], 'frontend_files': configuration['frontend_files'],
        'encoder_state_fingerprint': fingerprint(model.state_dict()),
        'AR_inputs': ['source_foa_latent', 'raw_edit_request'],
        'CLAP_inputs': ['source_foa_latent', 'source_attention_mask'],
        'factual_descriptions_are_training_supervision_only': True,
        'CLAP_readout_optimizer_scheduler_RNG_data_not_transferred_to_AR': True,
        'quality_gate_passed': False, 'independent_test_used': False,
        'loader_sha256': file_sha256(__file__),
    }
    return model, identity, record


def restore_ar_weights(module, parent):
    """Restore native AR weights while leaving the external frozen CLAP intact."""
    if parent['run_contract']['training_mode'] != 'ar_pretrain':
        raise RuntimeError('This transfer requires a native P10-descended AR parent')
    if parent['run_contract']['variant'] != 'global_and_sequence':
        raise RuntimeError('Both trained CLAP feature projections are required')
    before_clap = fingerprint(module.ar.source_clap_model.state_dict())
    module.diffusion.load_state_dict(parent['diffusion_state_dict'], strict=True)
    load_ar_specific(module.ar, parent['editing_ar_specific_state_dict'])
    checks = {
        'diffusion': (module.diffusion.state_dict(), parent['diffusion_state_dict']),
        'AR_adapters': (ar_specific_state(module.ar), parent['editing_ar_specific_state_dict']),
    }
    fingerprints = {}
    for key, (loaded, saved) in checks.items():
        fingerprints[key] = fingerprint(loaded)
        if fingerprints[key] != fingerprint(saved):
            raise RuntimeError(f'Native AR transfer changed {key} weights')
    if fingerprint(module.ar.source_clap_model.state_dict()) != before_clap:
        raise RuntimeError('AR parent overwrote the separately selected frozen CLAP')
    return fingerprints


def build_from_ar_parent(parent_reference, encoder_reference, *, preflight_path):
    """Construct the real native AR on CPU with explicit external encoder provenance.

    Returns the original read-only parent payload for subsequent declared state
    transfer. No checkpoint, optimizer, scheduler or RNG is created or mutated.
    """
    parent, identity = load_joint_checkpoint(parent_reference['checkpoint'],
        verify_sources=True, require_latest=False)
    if identity != parent_reference:
        raise RuntimeError('AR parent differs from the bound checkpoint identity')
    contract = parent['run_contract']
    preflight = json.loads(Path(preflight_path).read_text())
    for split in ('train', 'validation'):
        if contract['indices'][split]['sha256'] != preflight['indices'][split]['sha256']:
            raise RuntimeError('AR parent data differs from the selected CLAP population')
    model_path = Path(contract['model_config'])
    if file_sha256(model_path) != contract['model_config_sha256']:
        raise RuntimeError('AR parent model configuration changed')
    if codec_artifact_sha256(contract['codec']) != contract['codec_sha256']:
        raise RuntimeError('AR parent codec changed')
    base = contract['base_selection']
    if base['initialization'] != 'P10_EMA' or file_sha256(base['checkpoint']) != base['sha256']:
        raise RuntimeError('AR parent P10 initialization changed')
    if verify_frozen_qwen_runtime(contract['frozen_qwen_runtime']['root']) != contract['frozen_qwen_runtime']:
        raise RuntimeError('AR parent frozen instruction backbone changed')
    encoder, encoder_identity, dependency = load_source_encoder(encoder_reference,
        preflight_path=preflight_path)
    codec = native.ModelScenePlanCodecV4(contract['codec'])
    with torch.random.fork_rng(devices=[]):
        module = native.build_model(native.load_config(model_path), base['checkpoint'],
            codec, encoder, 'global_and_sequence', 0., 'ar_pretrain')
    weights = restore_ar_weights(module, parent)
    module.eval()
    transfer = {'schema': 'editing_ar_factual_clap_model_transfer_v1',
        'parent': identity, 'encoder_dependency': dependency,
        'native_AR_weight_fingerprints': weights,
        'external_encoder_preserved': True, 'parent_payload_mutated': False,
        'optimizer_scheduler_RNG_data_transfer_performed': False,
        'run_or_checkpoint_published': False, 'quality_gate_passed': False,
        'scope': 'Model loading only; effect acceptance, short adaptation and actual three-rank AR restart remain separate.'}
    return module, codec, parent, encoder_identity, transfer
