from __future__ import annotations

import torch

from stable_audio_tools.training.counterfactual_pair_auxiliary import (
    CounterfactualPairDeltaLoss,
)


def _example(value: float, pair_id: str, role: str):
    return {
        "metadata": {
            "curriculum_pair_id": pair_id,
            "curriculum_source_family_id": "family",
            "curriculum_pair_index": pair_id,
            "curriculum_role": role,
        },
        "modalities": {"foa_latent": torch.full((2, 5), value)},
    }


def test_same_noise_boundary_pair_delta_has_nonzero_gradient():
    auxiliary = CounterfactualPairDeltaLoss(
        {"weight": 0.5, "anchor_pairs_per_rank": 1, "anchor_time": 0.0},
        target_modality="foa_latent",
    )
    examples = [_example(1.0, "a", "base"), _example(3.0, "a", "control_swapped")]
    times = torch.tensor([[0.4, 0.6], [0.2, 0.8]])
    mask = torch.ones(5)
    prepared_times, states = auxiliary.prepare(
        examples=examples,
        times=times,
        target_index=1,
        modality_count=2,
        target_masks=[mask, mask.clone()],
        base_states=None,
        step=0,
    )
    assert prepared_times[:, 1].tolist() == [0.0, 0.0]
    assert torch.equal(states[0][1], states[1][1])
    base = torch.zeros(2, 5, requires_grad=True)
    swapped = torch.zeros(2, 5, requires_grad=True)
    loss = auxiliary(
        [base, swapped],
        [states[0][1], states[1][1]],
        [prepared_times[0, 1], prepared_times[1, 1]],
        target_latents=[
            examples[0]["modalities"]["foa_latent"],
            examples[1]["modalities"]["foa_latent"],
        ],
    )
    loss.backward()
    assert torch.isfinite(loss) and float(loss) > 0.0
    assert base.grad is not None and float(base.grad.abs().sum()) > 0.0
    assert swapped.grad is not None and float(swapped.grad.abs().sum()) > 0.0
    assert auxiliary.last_metrics["anchor_example_fraction"] == 1.0


def test_pair_rotation_and_incomplete_pair_fail_closed():
    auxiliary = CounterfactualPairDeltaLoss(
        {"anchor_pairs_per_rank": 1}, target_modality="foa_latent"
    )
    examples = [
        _example(0.0, "a", "base"),
        _example(1.0, "a", "control_swapped"),
        _example(2.0, "b", "base"),
        _example(3.0, "b", "control_swapped"),
    ]
    times = torch.rand(4, 3)
    first, _ = auxiliary.prepare(
        examples=examples,
        times=times,
        target_index=2,
        modality_count=3,
        target_masks=None,
        base_states=None,
        step=0,
    )
    second, _ = auxiliary.prepare(
        examples=examples,
        times=times,
        target_index=2,
        modality_count=3,
        target_masks=None,
        base_states=None,
        step=1,
    )
    assert first[:2, 2].eq(0.0).all() and not first[2:, 2].eq(0.0).all()
    assert second[2:, 2].eq(0.0).all() and not second[:2, 2].eq(0.0).all()
    try:
        auxiliary.prepare(
            examples=examples[:-1],
            times=times[:-1],
            target_index=2,
            modality_count=3,
            target_masks=None,
            base_states=None,
            step=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete counterfactual pair was accepted")


def test_multitime_anchor_matches_conditions_at_one_noised_state():
    auxiliary = CounterfactualPairDeltaLoss(
        {
            "weight": 1.0,
            "anchor_pairs_per_rank": 1,
            "anchor_times": [0.0, 0.5],
            "same_noised_state_at_anchor": True,
            "same_state_tolerance": 1.0e-6,
        },
        target_modality="foa_latent",
    )
    examples = [
        _example(1.0, "a", "base"),
        _example(3.0, "a", "control_swapped"),
    ]
    times = torch.tensor([[0.1], [0.9]])
    prepared_times, states = auxiliary.prepare(
        examples=examples,
        times=times,
        target_index=0,
        modality_count=1,
        target_masks=[torch.ones(5), torch.ones(5)],
        base_states=None,
        step=1,
    )
    assert prepared_times[:, 0].tolist() == [0.5, 0.5]

    targets = [example["modalities"]["foa_latent"] for example in examples]
    noised = [
        target * prepared_times[index, 0]
        + states[index][0] * (1.0 - prepared_times[index, 0])
        for index, target in enumerate(targets)
    ]
    assert torch.allclose(noised[0], noised[1], atol=1.0e-6, rtol=0.0)

    exact_flows = [
        (target - states[index][0]).detach().requires_grad_()
        for index, target in enumerate(targets)
    ]
    loss = auxiliary(
        exact_flows,
        noised,
        [prepared_times[0, 0], prepared_times[1, 0]],
        target_latents=targets,
    )
    loss.backward()
    assert float(loss) < 1.0e-10
    assert torch.isclose(auxiliary.last_metrics["anchor_time"], torch.tensor(0.5))
    assert torch.isclose(
        auxiliary.last_metrics["target_delta_rms"], torch.tensor(4.0)
    )
    assert torch.isclose(
        auxiliary.last_metrics["delta_rms_ratio"], torch.tensor(1.0)
    )
    assert torch.isclose(
        auxiliary.last_metrics["delta_cosine"], torch.tensor(1.0)
    )
    assert auxiliary.last_metrics["same_noised_state"] == 1.0
    assert auxiliary.last_metrics["same_state_error_max_abs"] < 1.0e-6


def test_same_state_anchor_rejects_singular_endpoint():
    try:
        CounterfactualPairDeltaLoss(
            {
                "anchor_times": [0.0, 1.0],
                "same_noised_state_at_anchor": True,
            },
            target_modality="foa_latent",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("same-state anchor accepted singular t=1")
