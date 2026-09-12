import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.execution_query import ExecutionQuery
from stable_audio_tools.training.transfusion_opsd.execution_response import execution_response_features, CandidateResponseHead


def test_response_baseline_is_zero_and_padding_cannot_change_features():
    state = torch.randn(1, 3, 8)
    mask = torch.ones(1, 8, dtype=torch.bool)
    query = ExecutionQuery(state, state, torch.tensor([.5]), mask, 0)
    values = torch.stack([state, state + .2, state - .3], 1)
    features = execution_response_features(query, values)
    assert torch.equal(features[:, 0], torch.zeros_like(features[:, 0]))
    padded = ExecutionQuery(torch.nn.functional.pad(state, (0, 8), value=500.),
        torch.nn.functional.pad(state, (0, 8), value=-500.), query.time,
        torch.nn.functional.pad(mask, (0, 8), value=False), 0)
    more = torch.nn.functional.pad(values, (0, 8), value=1000.)
    assert torch.allclose(features, execution_response_features(padded, more), atol=1e-6)


def test_response_observation_cannot_carry_live_teacher_gradients():
    state = torch.ones(1, 2, 8)
    query = ExecutionQuery(state, state, torch.tensor([.5]), torch.ones(1, 8, dtype=torch.bool), 0)
    with pytest.raises(ValueError, match='detached'):
        execution_response_features(query, state[:, None].requires_grad_())


def test_candidate_head_initially_preserves_baseline_and_shares_nonbaseline_scoring():
    head = CandidateResponseHead(8, 4, 5, 9, 3, width=16)
    inputs = [torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 5),
        torch.randn(2, 7, 9), torch.randn(2, 7, 3), torch.ones(2, 7, dtype=torch.bool)]
    for mode in ('request_only', 'query_moments', 'current_response'):
        assert torch.equal(head(*inputs, input_mode=mode).argmax(-1), torch.zeros(2, dtype=torch.long))
    permutation = torch.tensor([0, 4, 6, 1, 5, 2, 3])
    permuted = inputs[:3] + [value[:, permutation] for value in inputs[3:]]
    assert torch.allclose(head(*permuted), head(*inputs)[:, permutation], atol=1e-6)


def test_query_only_control_cannot_see_responses():
    head = CandidateResponseHead(8, 4, 5, 9, 3, width=16)
    args = [torch.randn(2, 8), torch.randn(2, 4), torch.randn(2, 5),
        torch.randn(2, 7, 9), torch.randn(2, 7, 3), torch.ones(2, 7, dtype=torch.bool)]
    before = head(*args, input_mode='query_moments')
    args[3] *= 20.
    assert torch.equal(before, head(*args, input_mode='query_moments'))


def test_calibrated_initialization_keeps_features_and_can_learn_new_actions():
    head = CandidateResponseHead(8, 4, 5, 9, 3, width=16)
    with torch.no_grad():
        head.output.weight.uniform_(-.1, .1)
    before = {name: value.clone() for name, value in head.state_dict().items()}
    receipt = head.calibrate_initial_action_margin(margin=.02)
    assert receipt['output_weight_scale'] < 1
    assert receipt['guaranteed_initial_margin'] >= .02
    for name, value in head.state_dict().items():
        if name != 'output.weight':
            assert torch.equal(before[name], value)
    args = [torch.randn(32, 8), torch.randn(32, 4), torch.randn(32, 5),
        torch.randn(32, 7, 9), torch.randn(32, 7, 3), torch.ones(32, 7, dtype=torch.bool)]
    for mode in ('request_only', 'query_moments', 'current_response'):
        logits = head(*args, input_mode=mode)
        assert bool((logits[:, 0] - logits[:, 1:].max(-1).values >= .02-1e-6).all())
    loss = -head(*args).log_softmax(-1)[:, 5].mean()
    loss.backward()
    assert all(parameter.requires_grad for parameter in head.parameters())
    assert head.output.weight.grad.norm() > 0 and head.context[1].weight.grad.norm() > 0


def test_initial_margin_rejects_invalid_or_nonfinite_bounds():
    head = CandidateResponseHead(8, 4, 5, 9, 3, width=16)
    for margin in (0., .1, -.1, float('nan')):
        with pytest.raises(ValueError, match='initial decision margin'):
            head.calibrate_initial_action_margin(margin=margin)
