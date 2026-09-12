from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_audio_tools.models.diffusion import DiTWrapper
from stable_audio_tools.training.transfusion_opsd.plan_condition_bridge import (
    PlanBridgeConfig, PlanConditionBridge, native_plan_hidden_states,
)


def inputs():
    g = torch.Generator().manual_seed(31)
    hidden = torch.randn(2, 5, 10, generator=g, requires_grad=True)
    condition = torch.randn(2, 4, 12, generator=g, requires_grad=True)
    p_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    c_mask = torch.tensor([[True, True, False, False], [True] * 4])
    return hidden, p_mask, condition, c_mask


def bridge():
    return PlanConditionBridge(PlanBridgeConfig(plan_dim=10, condition_dim=12, width=8))


def activate(model):
    with torch.no_grad():
        model.output.weight.copy_(torch.randn(model.output.weight.shape,
            generator=torch.Generator().manual_seed(43)) * .03)


def test_zero_initialization_preserves_rng_and_has_finite_first_update():
    rng = torch.random.get_rng_state().clone()
    model = bridge()
    assert torch.equal(rng, torch.random.get_rng_state())
    h, pm, c, cm = inputs()
    result = model(h, pm, c, cm)
    assert torch.count_nonzero(result) == 0
    result.sum().backward()
    assert torch.isfinite(model.output.weight.grad).all()
    assert torch.count_nonzero(model.output.weight.grad) > 0
    assert torch.count_nonzero(h.grad) == 0
    assert c.grad is None


def test_identical_forward_values_with_only_the_new_planning_gradient_stopped():
    model = bridge(); activate(model)
    h, pm, c, cm = inputs()
    connected = model(h, pm, c, cm)
    connected.square().sum().backward()
    assert torch.count_nonzero(h.grad) > 0 and c.grad is None
    assert torch.count_nonzero(h.grad[0, 3:]) == 0
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    h.grad = None; model.zero_grad(set_to_none=True)
    stopped = model(h, pm, c, cm, stop_plan_gradient=True)
    assert torch.equal(connected, stopped)
    stopped.square().sum().backward()
    assert h.grad is None
    for name, p in model.named_parameters():
        assert torch.equal(gradients[name], p.grad)


def test_bounded_residual_ignores_padded_planner_states_and_condition_tokens():
    model = bridge(); activate(model)
    h, pm, c, cm = inputs()
    result = model(h, pm, c, cm)
    limit = c.detach().square().mean(-1).sqrt().clamp_min(model.config.reference_rms_floor) * model.config.maximum_relative_rms
    assert (result.square().mean(-1).sqrt() <= limit + 1e-7).all()
    assert torch.count_nonzero(result[~cm]) == 0
    changed = h.detach().clone(); changed[~pm] = 9999.
    assert torch.equal(result, model(changed, pm, c, cm))


def test_native_dit_cfg_keeps_forward_and_connects_only_upstream_planner_parameters():
    torch.manual_seed(41)
    model = bridge(); activate(model)
    renderer = DiTWrapper(diffusion_objective='rectified_flow', io_channels=4,
        embed_dim=64, cond_token_dim=12, depth=1, num_heads=2,
        zero_init_branch_outputs=False, activation_checkpointing=False).eval()
    planner = nn.Embedding(11, 10)
    output_head = nn.Linear(10, 11)  # Untied decision output is not in this path.
    tokens = torch.tensor([[1, 2, 3]])
    h = planner(tokens); pm = torch.ones(1, 3, dtype=torch.bool)
    c = torch.randn(1, 4, 12, requires_grad=True); cm = torch.ones(1, 4, dtype=torch.bool)
    x = torch.randn(1, 4, 6); time = torch.tensor([.3])
    kwargs = dict(cross_attn_cond=c, cross_attn_mask=cm,
        negative_cross_attn_cond=torch.randn(1, 1, 12), negative_cross_attn_mask=torch.ones(1, 1, dtype=torch.bool),
        cfg_scale=3., apg_scale=0.)
    baseline = renderer(x, time, **kwargs)
    zero = renderer(x, time, external_cross_attn_residual=torch.zeros_like(c), **kwargs)
    assert torch.equal(baseline, zero)
    linked = renderer(x, time, external_cross_attn_residual=model(h, pm, c, cm), **kwargs)
    stopped = renderer(x, time, external_cross_attn_residual=model(h, pm, c, cm, stop_plan_gradient=True), **kwargs)
    assert torch.equal(linked, stopped) and not torch.equal(linked, baseline)
    linked.square().mean().backward()
    assert torch.isfinite(planner.weight.grad).all() and planner.weight.grad.abs().sum() > 0
    assert output_head.weight.grad is None and c.grad.abs().sum() > 0
    planner.zero_grad(set_to_none=True); renderer.zero_grad(set_to_none=True); c.grad = None
    stopped.square().mean().backward()
    assert planner.weight.grad is None and c.grad.abs().sum() > 0
    assert renderer.model.transformer.project_out.weight.grad.abs().sum() > 0


def test_native_hidden_replay_removes_hook_on_failure():
    class AR(nn.Module):
        def __init__(self):
            super().__init__(); self.ar_adapter = SimpleNamespace(output_norm=nn.LayerNorm(10))
        def forward(self, *args):
            self.ar_adapter.output_norm(torch.ones(1, 3, 10))
            raise RuntimeError('forward interrupted')
    ar = AR()
    bundle = SimpleNamespace(codec=SimpleNamespace(bos_id=1, eos_id=2), device=torch.device('cpu'), ar=ar,
        encode_event_requests=lambda texts, device: (torch.ones(1, 2, 10), torch.ones(1, 2, dtype=torch.bool)))
    with pytest.raises(RuntimeError, match='interrupted'):
        native_plan_hidden_states(bundle, SimpleNamespace(request='test'), (1, 3, 2))
    assert not ar.ar_adapter.output_norm._forward_hooks


@pytest.mark.parametrize('bad', [0., -1., float('inf'), float('nan'), True])
def test_invalid_bound_is_rejected(bad):
    with pytest.raises(ValueError): PlanBridgeConfig(maximum_relative_rms=bad)


def test_empty_plan_mask_is_rejected():
    h, pm, c, cm = inputs(); pm[0] = False
    with pytest.raises(ValueError, match='nonempty'):
        bridge()(h, pm, c, cm)
