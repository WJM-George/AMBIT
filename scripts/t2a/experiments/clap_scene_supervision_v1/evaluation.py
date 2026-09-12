"""Frozen encoder loading with explicit provenance for the two-optimizer format.

The core loader returns the actual training contract. The separate native-metric
adapter exposes a labeled configuration view for the unchanged legacy evaluator;
it also retains the full original contract. Neither entry selects an AR encoder
or declares quality acceptance.
"""
from pathlib import Path

import torch

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44
from . import state


def load_event_encoder(path, *, expected_sha256, device='cpu'):
    path = Path(path).resolve(strict=True)
    contract = state.read(path.parent / 'TRAIN_CONTRACT.json')
    if contract.get('schema') != state.SCHEMA:
        raise RuntimeError('This loader requires an actual scene-event training checkpoint')
    for source, expected in contract['source_sha256'].items():
        if state.sha(source) != expected:
            raise RuntimeError(f'Checkpoint training source changed: {source}')
    payload, manifest = state.load_checkpoint(path, contract)
    if manifest['sha256'] != expected_sha256:
        raise RuntimeError('Frozen encoder differs from the explicitly selected file identity')
    with torch.random.fork_rng(devices=[]):
        model = EditingCLAP44(CLAP44Config(**contract['native_config']['model']))
    model.load_state_dict(payload['encoder']['model'], strict=True)
    if state.fingerprint(model.state_dict()) != state.fingerprint(payload['encoder']['model']):
        raise RuntimeError('Frozen encoder tensors differ from the trained state')
    identity = {'path': str(path), 'sha256': manifest['sha256'], 'step': payload['step'],
                'new_updates': payload['new_updates'], 'contract': contract,
                'contract_sha256': manifest['contract_sha256'], 'loader_sha256': state.sha(__file__),
                'encoder_state_fingerprint': state.fingerprint(payload['encoder']['model']),
                'quality_gate_passed': False}
    return model.to(device).eval().requires_grad_(False), identity


def native_configuration_view(identity):
    contract = identity['contract']; parent = contract['parent']
    parent_contract_path = Path(parent['path']).parent / 'TRAIN_CONTRACT.json'
    if state.sha(parent_contract_path) != parent['contract_sha256']:
        raise RuntimeError('Native parent configuration provenance changed')
    native = state.read(parent_contract_path)
    if native['config'] != contract['native_config'] or native['train_index_sha256'] != contract['data']['index_sha256']:
        raise RuntimeError('Evaluation configuration is not inherited from this native parent')
    if native['m2d_used'] is not False or native['test_split_used_for_training_or_selection'] is not False:
        raise RuntimeError('Foreign data/teacher provenance')
    return {'schema': 'editing_clap44_event_native_metrics_configuration_view_v1',
            'configuration_view_only': True, 'training_contract_schema': contract['schema'],
            'training_contract_sha256': identity['contract_sha256'],
            'preflight_sha256': native['preflight_sha256'], 'text_files': native['text_files'],
            'frontend_files': native['frontend_files'], 'config': contract['native_config'],
            'source_sha256': contract['source_sha256'],
            'm2d_used': False, 'test_split_used_for_training_or_selection': False}


def native_metric_loader(checkpoint):
    """Return a loader hook, bound to one file, for the unchanged native metrics."""
    selected = Path(checkpoint['path']).resolve(strict=True)

    def load(path, *, device='cpu'):
        if Path(path).resolve(strict=True) != selected:
            raise RuntimeError('Native metrics attempted to load another checkpoint')
        model, identity = load_event_encoder(selected, expected_sha256=checkpoint['sha256'], device=device)
        view = native_configuration_view(identity)
        return model, {**identity, 'training_contract': identity['contract'], 'contract': view,
                       'contract_view_scope': 'Configuration adapter only; the actual two-clock training contract is retained separately.'}
    return load
