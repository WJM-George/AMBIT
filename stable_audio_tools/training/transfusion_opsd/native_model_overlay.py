"""Load a model-only native overlay over an exactly identified frozen parent.

The manifest binds all updated parameter aliases and the full reconstructed
policy fingerprint. It does not contain optimizer state or certify quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .event_experiment import state_fingerprint
from .event_native_policy import load_native_event_policy
from .provenance import sha256_file


def load_native_model_overlay(release_path, manifest_path, *, expected_sha256, device):
    if not expected_sha256 or sha256_file(manifest_path) != expected_sha256:
        raise ValueError('Native overlay manifest differs from the declared identity.')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['contract'] != 'parent_bound_native_model_overlay_v1':
        raise ValueError('Unsupported native model overlay contract.')
    tensors = manifest['tensors']
    if sha256_file(tensors['path']) != tensors['sha256']:
        raise ValueError('Native overlay tensors differ from the bound file.')
    parent = manifest['parent']
    policy, receipt = load_native_event_policy(release_path, device=device,
        parent_path=parent['path'], parent_sha256=parent['sha256'])
    if state_fingerprint(policy) != manifest['initial_fingerprint']:
        raise ValueError('Loaded native parent has a different full policy fingerprint.')
    named = dict(policy.named_parameters(remove_duplicate=False))
    aliases = manifest['aliases']
    if not aliases or any(key not in named for key in aliases):
        raise ValueError('Overlay canonical parameters are absent from this native architecture.')
    if len({id(named[key]) for key in aliases}) != len(aliases):
        raise ValueError('Overlay canonical parameter ownership is not disjoint.')
    for key, names in aliases.items():
        if not names or key not in names or any(name not in named or named[name] is not named[key] for name in names):
            raise ValueError('Native shared-parameter aliases differ from the saved overlay.')
    with safe_open(tensors['path'], framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(aliases):
            raise ValueError('Overlay tensors and canonical parameter manifest disagree.')
        with torch.no_grad():
            for key in aliases:
                value = handle.get_tensor(key)
                if value.shape != named[key].shape or value.dtype != named[key].dtype:
                    raise ValueError('Overlay shape or dtype differs from the native parameter.')
                named[key].copy_(value)
    if state_fingerprint(policy) != manifest['reconstructed_fingerprint']:
        raise ValueError('Parent plus overlay does not match the evaluated native model.')
    return policy, dict(native_parent=receipt, overlay=dict(path=str(manifest_path), sha256=expected_sha256),
        policy_fingerprint=manifest['reconstructed_fingerprint'], optimizer_loaded=False,
        whole_opsd_validated=manifest['whole_opsd_validated'], promotion_status=manifest['promotion_status'])
