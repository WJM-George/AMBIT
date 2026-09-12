import copy

import pytest
import torch

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44
from stable_audio_tools.training.transfusion_opsd.native_latent_clap import (
    FrozenNativeLatentCLAP, native_feature_distillation, requested_semantic_text,
)


def test_native_input_gradient_and_frozen_observer_survive_training_mode():
    torch.manual_seed(34)
    model = FrozenNativeLatentCLAP(EditingCLAP44(CLAP44Config(width=12, heads=3, layers=1)))
    model.train()
    clean = torch.randn(1, 64, 8, requires_grad=True)
    mask = torch.tensor([[True] * 4 + [False] * 4])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    output = model(clean, mask)
    teacher = {k: torch.randn_like(v, requires_grad=True) for k, v in output.items()}
    native_feature_distillation(output, teacher, semantic_weight=1., scene_weight=.5).sum().backward()
    assert clean.grad[:, :, :4].norm() > 0 and torch.isfinite(clean.grad).all()
    assert clean.grad[:, :, 4:].count_nonzero() == 0
    assert not model.training and not model.encoder.training
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(x.grad is None for x in teacher.values())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_semantic_retention_does_not_preserve_wrong_scene():
    student = dict(semantic=torch.tensor([[1., 1.]], requires_grad=True),
                   scene=torch.tensor([[0., 1.]], requires_grad=True))
    teacher = dict(semantic=torch.tensor([[1., 0.]], requires_grad=True),
                   scene=torch.tensor([[1., 0.]], requires_grad=True))
    loss = native_feature_distillation(student, teacher, semantic_weight=1., scene_weight=0.).mean()
    loss.backward()
    assert student['semantic'].grad.norm() > 0 and student['scene'].grad is None
    assert all(x.grad is None for x in teacher.values())


def test_request_content_is_independent_of_unspecified_or_spatial_values():
    request = {'sources': [{'kind': 'speech', 'core': 'a low male voice', 'constraints': [
        {'op': 'transcript', 'value': 'Please come here.'}, {'op': 'compass', 'value': 'left'}]}]}
    changed = copy.deepcopy(request)
    changed['sources'][0]['constraints'][-1]['value'] = 'right'
    assert requested_semantic_text(request) == requested_semantic_text(changed)
    assert requested_semantic_text(request) == '1 audible sources. Speech, a low male voice, saying "Please come here.".'
    changed['sources'][0]['constraints'][0]['value'] = 'Stop there.'
    assert requested_semantic_text(request) != requested_semantic_text(changed)
    with pytest.raises(ValueError):
        requested_semantic_text({'sources': [{'kind': 'speech', 'core': 'voice', 'constraints': []}]})


def test_native_heads_reject_mismatched_teacher_and_invalid_weights():
    with pytest.raises(ValueError):
        native_feature_distillation({'semantic': torch.ones(2, 3)}, {'semantic': torch.ones(1, 3)},
                                    semantic_weight=1., scene_weight=0.)
    with pytest.raises(ValueError):
        native_feature_distillation({}, {}, semantic_weight=0., scene_weight=0.)
