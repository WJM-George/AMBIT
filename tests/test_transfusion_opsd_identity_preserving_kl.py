import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.identity_preserving_kl import identity_preserving_kl


@pytest.mark.parametrize('scale,support', [(1.,8), (4.,64), (8.,256)])
def test_identity_has_exact_zero_gradient_and_no_adam_update(scale, support):
    generator = torch.Generator().manual_seed(42)
    student = torch.nn.Parameter(torch.randn(32, support, generator=generator)*scale)
    teacher = student.detach().clone()
    optimizer = torch.optim.AdamW([student], lr=1e-4, weight_decay=0., foreach=False)
    loss = identity_preserving_kl(student, teacher)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(student.grad).item() == 0
    optimizer.step()
    assert torch.equal(student, teacher)


def test_value_and_analytic_gradient_match_masked_kl_with_stopped_teacher():
    student = torch.tensor([[.2, -1., 4., -.5], [3., -.1, .7, 2.]], requires_grad=True)
    teacher = torch.tensor([[.4, 1., 2., .1], [-.5, 2., .4, -.8]], requires_grad=True)
    mask = torch.tensor([[True,False,True,True],[True,True,False,True]])
    temperature = .7
    lp = (student.double()/temperature).masked_fill(~mask, -torch.inf).log_softmax(-1)
    lq = (teacher.detach().double()/temperature).masked_fill(~mask, -torch.inf).log_softmax(-1)
    expected = (lq.exp()*(lq.masked_fill(~mask,0)-lp.masked_fill(~mask,0))).sum(-1).mean()
    loss = identity_preserving_kl(student, teacher, mask, temperature=temperature)
    torch.testing.assert_close(loss.double(), expected.detach(), rtol=1e-6, atol=1e-8)
    loss.backward()
    torch.testing.assert_close(student.grad.double(), (lp.exp()-lq.exp()).detach()/temperature/2, rtol=1e-6, atol=1e-8)
    assert teacher.grad is None and torch.count_nonzero(student.grad[~mask]).item()==0


def test_first_and_second_derivatives():
    # Test the per-row function in double precision, before the public FP32
    # scalar cast used by the native mixed-precision trainer.
    from stable_audio_tools.training.transfusion_opsd.identity_preserving_kl import _StoppedTeacherKL
    student = torch.tensor([[.2,-.3,.7]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.6,.1,-.2]], dtype=torch.float64)
    allowed = torch.ones_like(student, dtype=torch.bool)
    function = lambda x: _StoppedTeacherKL.apply(x, teacher, allowed, 1.2).mean()
    assert torch.autograd.gradcheck(function, (student,))
    assert torch.autograd.gradgradcheck(function, (student,))


def test_empty_support_is_rejected():
    with pytest.raises(ValueError):
        identity_preserving_kl(torch.ones(2,3), torch.ones(2,3), torch.zeros(2,3,dtype=torch.bool))
