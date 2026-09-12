from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_paired_rf import (
    paired_rf_example, paired_rf_loss, paired_rf_velocity_function,
)


def test_paired_flow_reconstructs_endpoints_and_clean_without_dividing_at_zero():
    clean = torch.tensor([[[2., -1.]], [[3., 4.]], [[-2., 5.]]], requires_grad=True)
    noise = torch.tensor([[[1., 2.]], [[-1., 2.]], [[1., 3.]]], requires_grad=True)
    time = torch.tensor([0., .4, 1.], requires_grad=True)
    mask = torch.ones(3, 2, dtype=torch.bool)
    state, target = paired_rf_example(clean, noise, time, mask)
    torch.testing.assert_close(state - time.detach()[:, None, None]*target, clean.detach())
    torch.testing.assert_close(state[0], clean[0])
    torch.testing.assert_close(state[-1], noise[-1])
    assert not state.requires_grad and not target.requires_grad


def test_masked_pair_loss_retains_each_example_weight_and_only_student_gradient():
    mask = torch.tensor([[True, False, False], [True, True, True]])
    target = torch.randn(2, 2, 3, requires_grad=True)
    student = (target.detach()+torch.tensor([1., 3.])[:, None, None]).requires_grad_(True)
    loss = paired_rf_loss(student, target, mask)
    torch.testing.assert_close(loss, torch.tensor(5.))
    loss.backward()
    assert target.grad is None and torch.count_nonzero(student.grad[0, :, 1:]) == 0
    assert student.grad[1].norm() > student.grad[0].norm()


@pytest.mark.parametrize('time,mask', [(torch.tensor([-0.1]), torch.tensor([[True]])),
    (torch.tensor([1.1]), torch.tensor([[True]])), (torch.tensor([.5]), torch.tensor([[False]]))])
def test_pair_rejects_invalid_time_or_empty_support(time, mask):
    with pytest.raises(ValueError):
        paired_rf_example(torch.zeros(1, 2, 1), torch.ones(1, 2, 1), time, mask)


def test_paired_retention_does_not_use_inference_cfg_or_negative_context():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.kw = None
        def forward(self, z, t, **kwargs):
            assert kwargs['batch_cfg'] is True
            self.kw = kwargs
            return z*self.weight+kwargs['cross_attn_cond']
    model = Model()
    condition_parameter = torch.nn.Parameter(torch.ones(()))
    def conditioner(rows, device):
        assert rows == ['positive']
        return dict(cross_attn_cond=condition_parameter)
    diffusion = SimpleNamespace(model=model, conditioner=conditioner, get_conditioning_inputs=lambda x: x)
    bundle = SimpleNamespace(diffusion=diffusion, device=torch.device('cpu'), cfg_scale=3., cfg_rescale_phi=.4)
    condition = SimpleNamespace(positive=['positive'], negative=['forbidden'], mask=torch.ones(1, 2, dtype=torch.bool))
    velocity = paired_rf_velocity_function(bundle, condition)
    loss = paired_rf_loss(velocity(torch.ones(1, 2, 2), torch.tensor([.5])), torch.zeros(1, 2, 2), condition.mask)
    loss.backward()
    assert model.weight.grad != 0 and condition_parameter.grad != 0
    assert model.kw['cfg_scale'] == 1 and not model.kw['rescale_cfg'] and model.kw['cfg_dropout_prob'] == 0
    assert bundle.cfg_scale == 3 and bundle.cfg_rescale_phi == .4
    with pytest.raises(ValueError):
        velocity(torch.ones(1, 2, 3), torch.tensor([.5]))
