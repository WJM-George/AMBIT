"""Load a new joint candidate with one shared AR and audio-rendering backbone."""
from pathlib import Path

from stable_audio_tools.models.sceneplan_editing_gain_adapter import install_pipeline_gain_metadata
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import load_joint_checkpoint, load_ar_specific
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import ScenePlanTransfusionEditingCLAP44Pipeline
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import _load_frozen_foa_vae
from . import initialization, model


def load_candidate(checkpoint, *, device='cpu', load_audio_autoencoder=True):
    payload, identity = load_joint_checkpoint(checkpoint)
    contract = payload['run_contract']
    if contract.get('structured_architecture') != model.CONTRACT or contract['training_mode'] != 'joint':
        raise RuntimeError('Expected the structured shared AR/Editing DiT joint candidate')
    cfg = contract['recipe']
    module, codec, groups, provenance = initialization.build(cfg)
    if provenance != contract['initialization'] or codec.fingerprint != contract['codec_fingerprint']:
        raise RuntimeError('Joint candidate initialization or codec identity changed')
    module.diffusion.load_state_dict(payload['diffusion_state_dict'], strict=True)
    load_ar_specific(module.ar, payload['editing_ar_specific_state_dict'])
    del payload, groups
    if module.ar.shared_transformer is not module.diffusion.model.model.transformer:
        raise RuntimeError('AR and audio renderer lost their shared Transformer')
    vae = vae_identity = None
    if load_audio_autoencoder:
        vae, vae_identity = _load_frozen_foa_vae(device)
    pipeline = ScenePlanTransfusionEditingCLAP44Pipeline(diffusion=module.diffusion,
        editing_ar=module.ar, codec=codec, audio_autoencoder=vae).requires_grad_(False).to(device).eval()
    install_pipeline_gain_metadata(pipeline)
    return pipeline, {'checkpoint': str(Path(checkpoint).resolve()), **identity,
        'shared_transformer_same_object': True, 'audio_renderer_uses_this_joint_checkpoint': True,
        'protected_original_DiT50k_is_initialization_only': True,
        'CLAP_checkpoint': cfg['base_AR_configuration']['clap_dependency']['checkpoint'],
        'frozen_vae': vae_identity, 'runtime_inputs': ['source_foa_audio', 'raw_edit_instruction'],
        'old_sceneplan_input': False, 'quality_gate_passed': False}
