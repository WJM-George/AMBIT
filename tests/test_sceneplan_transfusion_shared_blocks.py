from __future__ import annotations

import pytest
import torch

from stable_audio_tools.models.transformer import ContinuousTransformer


def _tiny_shared_stack() -> ContinuousTransformer:
    torch.manual_seed(42)
    return ContinuousTransformer(
        dim=64,
        depth=2,
        dim_in=5,
        dim_out=7,
        dim_heads=32,
        cross_attend=True,
        cond_token_dim=64,
        causal=False,
        zero_init_branch_outputs=False,
    ).eval()


def test_ar_override_is_causal_and_does_not_mutate_dit_mode() -> None:
    stack = _tiny_shared_stack()
    audio = torch.randn(2, 6, 5)
    audio_mask = torch.ones(2, 6, dtype=torch.bool)
    context = torch.randn(2, 4, 64)
    context_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        dit_before = stack(
            audio,
            context=context,
            context_mask=context_mask,
            padding_mask=audio_mask,
            use_checkpointing=False,
        )
        plan_hidden = torch.randn(2, 6, 64)
        ar_before = stack(
            plan_hidden,
            context=context,
            context_mask=context_mask,
            padding_mask=audio_mask,
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=True,
            use_checkpointing=False,
        )
        changed = plan_hidden.clone()
        changed[:, 4:] += 100.0
        ar_after = stack(
            changed,
            context=context,
            context_mask=context_mask,
            padding_mask=audio_mask,
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=True,
            use_checkpointing=False,
        )
        dit_after = stack(
            audio,
            context=context,
            context_mask=context_mask,
            padding_mask=audio_mask,
            use_checkpointing=False,
        )

    torch.testing.assert_close(ar_before[:, :4], ar_after[:, :4])
    assert not torch.allclose(ar_before[:, 4:], ar_after[:, 4:])
    assert torch.equal(dit_before, dit_after)
    assert all(not layer.self_attn.causal for layer in stack.layers)
    assert all(not layer.cross_attn.causal for layer in stack.layers)


def test_complete_request_context_remains_visible_to_early_plan_tokens() -> None:
    stack = _tiny_shared_stack()
    plan_hidden = torch.randn(1, 5, 64)
    plan_mask = torch.ones(1, 5, dtype=torch.bool)
    context = torch.randn(1, 4, 64)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    changed_context = context.clone()
    changed_context[:, -1] += 100.0

    with torch.no_grad():
        first = stack(
            plan_hidden,
            context=context,
            context_mask=context_mask,
            padding_mask=plan_mask,
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=True,
            use_checkpointing=False,
        )
        second = stack(
            plan_hidden,
            context=changed_context,
            context_mask=context_mask,
            padding_mask=plan_mask,
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=True,
            use_checkpointing=False,
        )

    assert not torch.allclose(first[:, 0], second[:, 0])


def test_frozen_stack_backpropagates_only_to_discrete_input() -> None:
    stack = _tiny_shared_stack().requires_grad_(False)
    plan_hidden = torch.randn(1, 5, 64, requires_grad=True)
    context = torch.randn(1, 4, 64)
    output = stack(
        plan_hidden,
        context=context,
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        padding_mask=torch.ones(1, 5, dtype=torch.bool),
        skip_input_projection=True,
        skip_output_projection=True,
        self_attention_causal=True,
        use_checkpointing=False,
    )
    output.square().mean().backward()

    assert plan_hidden.grad is not None
    assert torch.isfinite(plan_hidden.grad).all()
    assert float(plan_hidden.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in stack.parameters())


def test_skipped_projection_rejects_non_hidden_width() -> None:
    stack = _tiny_shared_stack()
    with pytest.raises(ValueError, match="hidden width 64"):
        stack(
            torch.randn(1, 2, 63),
            skip_input_projection=True,
            skip_output_projection=True,
            use_checkpointing=False,
        )
