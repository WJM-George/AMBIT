import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_decision_condition import hard_native_condition


def inputs(value):
    return dict(cross_attn_cond=value, cross_attn_mask=torch.ones(1, 2, dtype=torch.bool),
                negative_cross_attn_cond=None)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_exact_native_forward_and_only_one_encoder_gradient(dtype):
    c = torch.tensor([[.25, -1.]], dtype=dtype, requires_grad=True)
    other = torch.tensor([[2., 3.]], dtype=dtype, requires_grad=True)
    logits = torch.tensor([.7, -.3], requires_grad=True)
    hard, alternatives = inputs(c), [inputs(c), inputs(other)]
    connected = hard_native_condition(hard, alternatives, logits, hard_index=0,
        differentiable_keys=['cross_attn_cond'])['cross_attn_cond']
    assert torch.equal(connected, c)
    loss = connected.float().square().sum()
    dc, do, dl = torch.autograd.grad(loss, (c, other, logits), allow_unused=True)
    torch.testing.assert_close(dc, 2*c)
    assert do is None
    assert torch.isfinite(dl).all() and dl.abs().sum() > 0


def test_logit_gradient_matches_declared_st_surrogate_not_argmax_derivative():
    c, other = torch.tensor([[1., 2.]], dtype=torch.float64), torch.tensor([[3., 5.]], dtype=torch.float64)
    logits = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    out = hard_native_condition(inputs(c), [inputs(c), inputs(other)], logits,
        hard_index=0, differentiable_keys=['cross_attn_cond'], temperature=.7)['cross_attn_cond']
    actual, = torch.autograd.grad(out.square().sum(), logits)
    s = (logits.detach()/.7).softmax(0)
    d = torch.stack([(2*c*c).sum(), (2*c*other).sum()])
    expected = s*(d-(s*d).sum())/.7
    torch.testing.assert_close(actual, expected)
    # Finite execution differences contain curvature absent from the local ST estimate.
    assert (other.square().sum()-c.square().sum()) != (2*c*(other-c)).sum()


def test_detached_control_keeps_hard_encoder_and_executor_paths_only():
    c = torch.ones(1, 2, requires_grad=True)
    other = torch.full((1, 2), 2., requires_grad=True)
    logits = torch.zeros(2, requires_grad=True)
    executor = torch.tensor(3., requires_grad=True)
    out = hard_native_condition(inputs(c), [inputs(c), inputs(other)], logits,
        hard_index=0, differentiable_keys=['cross_attn_cond'], connect_decisions=False)['cross_attn_cond']
    dc, do, dl, de = torch.autograd.grad((executor*out).sum(), (c, other, logits, executor), allow_unused=True)
    assert do is None and dl is None and de.item() == 2
    torch.testing.assert_close(dc, torch.full_like(c, 3.))


@pytest.mark.parametrize('change', ['mask', 'shape', 'missing', 'hard', 'negative'])
def test_native_mask_geometry_and_hard_membership_must_match(change):
    hard = inputs(torch.ones(1, 2))
    alt = inputs(torch.zeros(1, 2))
    if change == 'mask':
        alt['cross_attn_mask'][0, 1] = False
    if change == 'shape':
        alt['cross_attn_cond'] = torch.zeros(1, 3)
    if change == 'missing':
        del alt['negative_cross_attn_cond']
    if change == 'negative':
        alt['negative_cross_attn_cond'] = torch.zeros(1)
    with pytest.raises(ValueError):
        hard_native_condition(hard, [hard, alt], torch.zeros(2), hard_index=int(change == 'hard'),
            differentiable_keys=['cross_attn_cond'])
