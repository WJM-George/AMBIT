import pytest
import torch
from torch import nn

from stable_audio_tools.training.transfusion_opsd.event_native_policy import NativeEventPolicy, import_legacy_parent


def fixture():
    bundle = nn.Module()
    bundle.copy_pointer = nn.Linear(2, 2)
    bundle.native_head = nn.Linear(2, 3)
    policy = NativeEventPolicy(bundle)
    source = {name: torch.full_like(value, .25) for name, value in policy.state_dict().items()}
    source['completion_head.retired.weight'] = torch.ones(4, 2)
    return policy, source


def test_complete_native_import_omits_only_inactive_refiner_and_preserves_parent():
    policy, source = fixture()
    keys = set(source)
    receipt = import_legacy_parent(policy, {'model': source})
    assert receipt['omitted_keys'] == ['completion_head.retired.weight']
    assert set(source) == keys and not hasattr(policy, 'completion_head')
    assert all(torch.equal(value, source[name]) for name, value in policy.state_dict().items())
    assert not any(p.requires_grad for p in policy.bundle.copy_pointer.parameters())
    assert all(p.requires_grad for p in policy.bundle.native_head.parameters())


def test_unknown_parent_subtree_is_not_silently_dropped():
    policy, source = fixture(); source['another_active_head.weight'] = torch.ones(2, 2)
    with pytest.raises(ValueError, match='unknown'):
        import_legacy_parent(policy, {'model': source})


def test_missing_native_head_is_not_silently_initialized():
    policy, source = fixture(); del source['bundle.native_head.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        import_legacy_parent(policy, {'model': source})
