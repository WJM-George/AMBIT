import copy

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_request_support import (
    request_conditioned_kl, requested_duration_support,
)


def test_known_incompatible_mass_has_a_gradient_and_teacher_is_stopped():
    student = torch.tensor([3., 0., -1.], dtype=torch.float64, requires_grad=True)
    reference = student.detach().clone().requires_grad_(True)
    compatible = torch.tensor([False, True, True])
    loss = request_conditioned_kl(student, reference, compatible)
    target = reference.detach().masked_fill(~compatible, -torch.inf).softmax(-1)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(probs=target),
        torch.distributions.Categorical(logits=student.detach()))
    assert torch.allclose(loss.double(), expected, atol=1e-6, rtol=0)
    loss.backward()
    assert reference.grad is None
    assert student.grad[0] > 0 and (student.grad[1:] < 0).all()
    assert torch.allclose(student.grad, student.detach().softmax(-1) - target)
    assert (student.detach() - student.grad).softmax(-1)[1:].sum() > student.detach().softmax(-1)[1:].sum()


def test_full_compatible_support_is_an_exact_retention_fixed_point():
    logits = torch.tensor([[1., 3., -2.], [.2, -.7, 2.3]], requires_grad=True)
    loss = request_conditioned_kl(logits, logits.detach().clone(), torch.ones_like(logits, dtype=torch.bool))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_gradient_matches_full_support_kl_after_student_moves():
    torch.manual_seed(31)
    logits = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    reference = torch.randn_like(logits)
    compatible = torch.tensor([[True, False, True, False, False], [False, True, False, True, True]])
    assert torch.autograd.gradcheck(lambda x: request_conditioned_kl(x, reference, compatible).double(),
                                   (logits,), eps=1e-3, atol=1e-4, rtol=1e-3)


class FrameCodec:
    bos_id = 1
    token_to_id = {'<duration_frames>': 5}
    frame_ids = tuple(range(1000, 1649))

    def allowed_next_ids(self, prefix):
        assert tuple(prefix) == (1, 5)
        return set(self.frame_ids[1:])


def requirements():
    return {'schema': 'generation_ar_natural_requirements_v2', 'scene': [], 'relations': [],
            'sources': [{'key': 'source_0', 'kind': 'sound', 'core': 'A bell.',
                         'evidence': 'A bell.', 'constraints': []}]}


def test_only_explicit_scene_evidence_constrains_duration_and_keeps_tolerance():
    codec = FrameCodec(); legal = sorted(codec.allowed_next_ids((1, 5)))
    req = requirements()
    assert requested_duration_support(codec, 'A bell.', req, (1, 5), legal) is None
    request = 'A bell. The scene lasts 5 seconds.'
    req['scene'] = [{'op': 'numeric', 'field': 'duration_sec', 'value': 5.,
                     'evidence': 'The scene lasts 5 seconds.'}]
    allowed = requested_duration_support(codec, request, req, (1, 5), legal)
    seconds = [codec.frame_ids.index(x) * 1024 / 44100 for x in legal]
    assert sum(allowed) > 1
    assert all(ok == (abs(s - 5) <= .25 + 1e-9) for ok, s in zip(allowed, seconds))
    bad = copy.deepcopy(req); bad['scene'][0]['evidence'] = 'Unstated duration'
    with pytest.raises(ValueError, match='evidence'):
        requested_duration_support(codec, request, bad, (1, 5), legal)


def test_contradictory_request_support_and_empty_targets_are_rejected():
    codec = FrameCodec(); req = requirements(); request = 'A bell. Use 3 seconds and 11 seconds.'
    req['scene'] = [{'op': 'numeric', 'field': 'duration_sec', 'value': n, 'evidence': request}
                    for n in (3., 11.)]
    with pytest.raises(ValueError, match='no executable'):
        requested_duration_support(codec, request, req, (1, 5), sorted(codec.allowed_next_ids((1, 5))))
    with pytest.raises(ValueError, match='nonempty'):
        request_conditioned_kl(torch.zeros(3), torch.zeros(3), torch.zeros(3, dtype=torch.bool))
