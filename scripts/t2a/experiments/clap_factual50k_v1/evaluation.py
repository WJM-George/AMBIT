"""Frozen encoder extraction from the explicit fresh three-rank lineage."""
from pathlib import Path
import torch

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44
from . import checkpoint as state


def load_factual_encoder(path, *, expected_sha256, device='cpu'):
    path = Path(path).resolve(strict=True)
    contract = state.read(path.parent / 'TRAIN_CONTRACT.json')
    if contract.get('schema') != state.SCHEMA:
        raise RuntimeError('Expected the fresh factual50k three-rank checkpoint schema')
    for source, digest in contract['source_sha256'].items():
        if state.sha(source) != digest: raise RuntimeError(f'Training source changed: {source}')
    payload, manifest = state.load(path, contract)
    if manifest['sha256'] != expected_sha256:
        raise RuntimeError('Frozen encoder file differs from the selected identity')
    with torch.random.fork_rng(devices=[]):
        model = EditingCLAP44(CLAP44Config(**contract['native_config']['model']))
    model.load_state_dict(payload['encoder']['model'], strict=True)
    fingerprint = state.fingerprint(payload['encoder']['model'])
    if state.fingerprint(model.state_dict()) != fingerprint:
        raise RuntimeError('Frozen encoder differs from the complete saved state')
    identity = {'path': str(path), 'sha256': manifest['sha256'], 'step': payload['step'],
        'new_updates': payload['new_updates'], 'initialization_step': payload['initialization_step'],
        'contract': contract, 'contract_sha256': manifest['contract_sha256'],
        'loader_sha256': state.sha(__file__), 'encoder_state_fingerprint': fingerprint,
        'quality_gate_passed': False}
    return model.to(device).eval().requires_grad_(False), identity


def native_configuration_view(identity):
    contract = identity['contract']
    if contract['schema'] != state.SCHEMA or contract['physical_gpus'] != [5, 6, 7] or contract['world_size'] != 3:
        raise RuntimeError('Wrong full factual training lineage')
    parent = contract['initialization']
    native_path = Path(parent['checkpoint']).parent / 'TRAIN_CONTRACT.json'
    if state.sha(native_path) != parent['contract_sha256'] or parent['step'] != 20000:
        raise RuntimeError('Warm-start native20k provenance changed')
    native = state.read(native_path)
    for key in ('model', 'text', 'frontend'):
        if native['config'][key] != contract['native_config'][key]:
            raise RuntimeError(f'Native metric {key} configuration changed')
    if native['train_index_sha256'] != contract['data']['index_sha256'] or contract['data']['rows'] != 1000000:
        raise RuntimeError('Training population changed')
    if native['m2d_used'] or native['test_split_used_for_training_or_selection']:
        raise RuntimeError('Foreign teacher or test provenance')
    if contract['raw_edit_requests_consumed'] or contract['independent_test_used']:
        raise RuntimeError('Editing request or sealed test is not factual CLAP supervision')
    if contract['method']['text_view'] != 'factual_templates200' or contract['max_new_updates'] != 50000:
        raise RuntimeError('Factual text view or fresh update budget changed')
    return {'schema': 'clap44_factual50k_native_metrics_configuration_view_v1',
        'configuration_view_only': True, 'actual_full_training_contract_preserved': True,
        'training_contract_schema': contract['schema'], 'training_contract_sha256': identity['contract_sha256'],
        'preflight_sha256': native['preflight_sha256'], 'text_files': native['text_files'],
        'frontend_files': native['frontend_files'], 'config': contract['native_config'],
        'source_sha256': contract['source_sha256'], 'm2d_used': False,
        'test_split_used_for_training_or_selection': False,
        'declared_training_changes': 'Factual200 scene text and explicit scene1/binding5/pair5/eventkind1 auxiliary recipe; fresh three-rank optimizer/scheduler/RNG/data lineage.'}


def native_metric_loader(checkpoint):
    selected = Path(checkpoint['path']).resolve(strict=True)
    def load(path, *, device='cpu'):
        if Path(path).resolve(strict=True) != selected: raise RuntimeError('Unexpected checkpoint selection')
        model, identity = load_factual_encoder(selected, expected_sha256=checkpoint['sha256'], device=device)
        if identity['step'] != checkpoint['step']: raise RuntimeError('New update count changed')
        view = native_configuration_view(identity)
        return model, {**identity, 'training_contract': identity['contract'], 'contract': view,
            'contract_view_scope': 'Metric configuration adapter only. The actual fresh optimizer lineage and original20k initialization remain explicit.'}
    return load
