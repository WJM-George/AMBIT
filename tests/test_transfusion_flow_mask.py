from __future__ import annotations

import unittest

import torch


def _tiny_transfusion():
    from transfusion_pytorch import Transfusion

    return Transfusion(
        num_text_tokens=16,
        transformer={
            "dim": 16,
            "depth": 1,
            "dim_head": 8,
            "heads": 2,
            "time_cond_dim": 16,
            "use_flex_attn": False,
        },
        dim_latent=2,
        channel_first_latent=True,
        modality_default_shape=(4,),
        modality_num_dim=1,
        model_output_clean=False,
        text_loss_weight=0.0,
        velocity_consistency_loss_weight=0.0,
        prob_uncond=0.0,
    )


class TransfusionFlowMaskTests(unittest.TestCase):
    def test_callable_late_residual_is_exact_noop_at_zero_init(self):
        torch.manual_seed(5)
        model = _tiny_transfusion().eval()
        target = torch.randn(2, 4)
        sample = [[torch.tensor([1]), (0, target)]]
        times = torch.tensor([[0.5]])

        baseline, _ = model(
            sample,
            times=times,
            return_loss=False,
            return_embed=True,
        )
        calls = []

        def zero_residual(tokens):
            calls.append(tuple(tokens.shape))
            return torch.zeros_like(tokens)

        unchanged, _ = model(
            sample,
            times=times,
            return_loss=False,
            return_embed=True,
            modality_layer_residuals=[[(zero_residual, 1)]],
        )
        self.assertEqual(calls, [(4, 16)])
        self.assertTrue(torch.equal(unchanged, baseline))

        changed, _ = model(
            sample,
            times=times,
            return_loss=False,
            return_embed=True,
            modality_layer_residuals=[[(lambda tokens: torch.ones_like(tokens), 1)]],
        )
        self.assertFalse(torch.equal(changed, baseline))

    def test_identity_training_coupling_has_exact_zero_flow_target(self):
        torch.manual_seed(11)
        model = _tiny_transfusion().train()
        for parameter in model.parameters():
            parameter.data.zero_()
        target = torch.randn(2, 4)
        sample = [[torch.tensor([1]), (0, target)]]
        times = torch.tensor([[0.37]])

        _, identity = model(
            sample,
            times=times,
            modality_flow_masks=[[torch.zeros(4)]],
            return_breakdown=True,
        )
        _, ordinary = model(
            sample,
            times=times,
            return_breakdown=True,
        )

        self.assertLess(float(identity.flow[0]), 1.0e-8)
        self.assertGreater(float(ordinary.flow[0]), 0.1)

    def test_zero_sampling_mask_preserves_initial_gaussian_latent(self):
        torch.manual_seed(17)
        model = _tiny_transfusion().eval()
        initial = torch.randn(4, 2)
        prompt = [
            torch.tensor([model.meta_id]),
            model.char_tokenizer("4", device=model.device),
            torch.tensor([model.som_ids[0]]),
        ]
        sample = model.sample(
            prompt=prompt,
            max_length=3,
            fixed_modality_shape=(4,),
            init_modality_noise=initial,
            modality_steps=4,
            cfg_scale=1.0,
            modality_flow_mask=torch.zeros(4),
            return_unprocessed_modalities=True,
        )
        generated = [item[1] for item in sample if isinstance(item, tuple)][-1]
        torch.testing.assert_close(generated, initial.transpose(0, 1))

    def test_cfg_can_remove_conditional_only_late_residual(self):
        torch.manual_seed(19)
        model = _tiny_transfusion().eval()
        initial = torch.randn(4, 2)
        prompt = [
            torch.tensor([model.meta_id]),
            model.char_tokenizer("4", device=model.device),
            torch.tensor([model.som_ids[0]]),
        ]

        shared_calls = []
        conditional_calls = []

        def shared_residual(tokens):
            shared_calls.append(tuple(tokens.shape))
            return torch.ones_like(tokens)

        def conditional_residual(tokens):
            conditional_calls.append(tuple(tokens.shape))
            return torch.ones_like(tokens)

        common = {
            "prompt": prompt,
            "cfg_uncond_prompt": prompt,
            "max_length": 3,
            "fixed_modality_shape": (4,),
            "init_modality_state": initial,
            "modality_steps": 3,
            "cfg_scale": 2.0,
            "return_unprocessed_modalities": True,
        }
        model.sample(
            **common,
            modality_layer_residual=(shared_residual, 1),
        )
        model.sample(
            **common,
            modality_layer_residual=(conditional_residual, 1),
            cfg_uncond_uses_modality_layer_residual=False,
        )

        self.assertGreater(len(conditional_calls), 0)
        self.assertEqual(len(shared_calls), 2 * len(conditional_calls))
        self.assertTrue(all(shape == (4, 16) for shape in shared_calls))
        self.assertTrue(all(shape == (4, 16) for shape in conditional_calls))

    def test_flow_element_weights_match_normalized_weighted_mse(self):
        model = _tiny_transfusion().train()
        for parameter in model.parameters():
            parameter.data.zero_()
        target = torch.tensor([[0.5, -0.5, 1.0, -1.0], [2.0, 1.5, -2.0, -1.5]])
        weights = torch.tensor([[1.0], [3.0]])
        sample = [[torch.tensor([1]), (0, target)]]
        times = torch.tensor([[0.5]])

        torch.manual_seed(29)
        noise = torch.randn_like(target)
        expected = ((target - noise).square() * weights).sum() / weights.expand_as(
            target
        ).sum()
        torch.manual_seed(29)
        _, breakdown = model(
            sample,
            times=times,
            modality_flow_element_weights=[[weights]],
            return_breakdown=True,
        )
        torch.testing.assert_close(breakdown.flow[0], expected)

    def test_explicit_flow_base_state_replaces_gaussian_coupling(self):
        model = _tiny_transfusion().train()
        for parameter in model.parameters():
            parameter.data.zero_()
        target = torch.tensor([[0.5, -0.5, 1.0, -1.0], [2.0, 1.5, -2.0, -1.5]])
        base = torch.tensor([[-0.5, 0.25, 0.0, 1.0], [1.0, -0.5, -1.0, 0.5]])
        sample = [[torch.tensor([1]), (0, target)]]

        _, breakdown = model(
            sample,
            times=torch.tensor([[0.3]]),
            modality_flow_base_states=[[base]],
            return_breakdown=True,
        )
        torch.testing.assert_close(
            breakdown.flow[0], (target - base).square().mean()
        )

    def test_auxiliary_flow_hook_observes_exact_noised_state_and_time(self):
        model = _tiny_transfusion().train()
        target = torch.randn(2, 4)
        sample = [[torch.tensor([1]), (0, target)]]
        time = torch.tensor([[0.3]])
        observed = {}

        def auxiliary(modality_id, predicted, noised, times):
            observed["modality_id"] = modality_id
            observed["predicted"] = predicted
            observed["noised"] = noised
            observed["times"] = times
            return predicted[0].square().mean() * 0.25

        _, breakdown = model(
            sample,
            times=time,
            modality_flow_auxiliary_loss_fn=auxiliary,
            return_breakdown=True,
        )
        self.assertEqual(observed["modality_id"], 0)
        self.assertEqual(len(observed["predicted"]), 1)
        self.assertEqual(len(observed["noised"]), 1)
        self.assertEqual(len(observed["times"]), 1)
        torch.testing.assert_close(observed["times"][0], time[0, 0])
        self.assertIsNotNone(breakdown.auxiliary_flow)
        torch.testing.assert_close(
            breakdown.auxiliary_flow[0],
            observed["predicted"][0].square().mean() * 0.25,
        )

    def test_primary_flow_transform_owns_exact_flow_loss(self):
        model = _tiny_transfusion().train()
        for parameter in model.parameters():
            parameter.data.zero_()
        target = torch.randn(2, 4)
        base = torch.randn(2, 4)
        exact_flow = target - base
        sample = [[torch.tensor([1]), (0, target)]]
        observed = {}

        def transform(modality_id, predicted, noised, times):
            observed["modality_id"] = modality_id
            observed["predicted"] = predicted
            observed["noised"] = noised
            observed["times"] = times
            return [exact_flow]

        _, breakdown = model(
            sample,
            times=torch.tensor([[0.4]]),
            modality_flow_base_states=[[base]],
            modality_pred_flow_transform_fn=transform,
            return_breakdown=True,
        )
        self.assertEqual(observed["modality_id"], 0)
        self.assertEqual(len(observed["predicted"]), 1)
        self.assertEqual(len(observed["noised"]), 1)
        self.assertEqual(len(observed["times"]), 1)
        self.assertLess(float(breakdown.flow[0]), 1.0e-8)

    def test_sampling_flow_transform_is_applied_to_conditional_field(self):
        model = _tiny_transfusion().eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        initial = torch.zeros(2, 4)
        prompt = [
            torch.tensor([model.meta_id]),
            model.char_tokenizer("4", device=model.device),
            torch.tensor([model.som_ids[0]]),
        ]
        calls = []

        def transform(modality_id, predicted, state, time, conditional):
            calls.append((modality_id, float(time), bool(conditional)))
            return torch.ones_like(state)

        sample = model.sample(
            prompt=prompt,
            max_length=3,
            fixed_modality_shape=(4,),
            init_modality_state=initial,
            modality_steps=3,
            cfg_scale=1.0,
            modality_flow_transform_fn=transform,
            return_unprocessed_modalities=True,
        )
        generated = [item[1] for item in sample if isinstance(item, tuple)][-1]
        self.assertTrue(calls)
        self.assertTrue(all(item[0] == 0 and item[2] for item in calls))
        torch.testing.assert_close(generated, torch.ones_like(initial))

    def test_native_channel_first_sampling_state_is_accepted(self):
        model = _tiny_transfusion().eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        initial = torch.randn(2, 4)
        prompt = [
            torch.tensor([model.meta_id]),
            model.char_tokenizer("4", device=model.device),
            torch.tensor([model.som_ids[0]]),
        ]
        sample = model.sample(
            prompt=prompt,
            max_length=3,
            fixed_modality_shape=(4,),
            init_modality_state=initial,
            modality_steps=3,
            cfg_scale=1.0,
            return_unprocessed_modalities=True,
        )
        generated = [item[1] for item in sample if isinstance(item, tuple)][-1]
        torch.testing.assert_close(generated, initial)


if __name__ == "__main__":
    unittest.main()
