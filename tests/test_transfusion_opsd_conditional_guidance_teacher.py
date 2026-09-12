import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.conditional_guidance_teacher import conditional_clean_target


def inputs():
    return (torch.ones(2, 3, 4), torch.tensor([.5, .25]), torch.zeros(2, 3, 4),
        torch.ones(2, 3, 4), torch.zeros(2, 3, 4), torch.ones(2, 4, dtype=torch.bool))


def test_flow_sign_and_per_example_bound():
    result = conditional_clean_target(*inputs(), relative_radius=.1)
    torch.testing.assert_close(result['positive'], torch.full((2, 3, 4), .9))
    torch.testing.assert_close(result['actual_norm'], result['radius'])
    torch.testing.assert_close(result['fraction'], torch.tensor([.2, .4]))


def test_all_teacher_values_stop_gradient():
    args = [x.requires_grad_() if x.is_floating_point() else x for x in inputs()]
    result = conditional_clean_target(*args)
    assert all(not value.requires_grad for value in result.values())
    assert all(x.grad is None for x in args)


def test_padding_does_not_set_budget_or_receive_a_repair():
    state, time, native, positive, negative, mask = inputs()
    state[:, :, 2:] = 1000
    positive[:, :, 2:] = 10000
    mask[:, 2:] = False
    result = conditional_clean_target(state, time, native, positive, negative, mask, relative_radius=.1)
    torch.testing.assert_close(result['positive'][:, :, :2], torch.full((2, 3, 2), .9))
    assert torch.equal(result['positive'][:, :, 2:], state[:, :, 2:])
    torch.testing.assert_close(result['radius'], torch.full((2,), .1 * 6 ** .5))


def test_zero_weight_or_identical_condition_is_an_exact_noop():
    args = inputs()
    assert torch.equal(conditional_clean_target(*args, weight=0)['positive'], args[0])
    state, time, native, positive, _, mask = args
    assert torch.equal(conditional_clean_target(state, time, native, positive, positive, mask)['positive'], state)


@pytest.mark.parametrize('kind', ['time', 'nan', 'shape', 'empty_mask'])
def test_invalid_teacher_inputs_fail(kind):
    args = list(inputs())
    if kind == 'time': args[1][0] = 0
    if kind == 'nan': args[3][0, 0, 0] = float('nan')
    if kind == 'shape': args[4] = args[4][:1]
    if kind == 'empty_mask': args[5][0] = False
    with pytest.raises(ValueError): conditional_clean_target(*args)
