import math
import unittest

import torch

from stable_audio_tools.models.dit_moe import CPEMoEFeedForward


class CPEMoETest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 12, 8, requires_grad=True)
        valid = torch.ones(2, 12, dtype=torch.bool)
        valid[1, -2:] = False
        audio = torch.zeros_like(valid)
        audio[0, 3:10] = True
        audio[1, 5:10] = True
        context = valid & ~audio
        time = torch.randn(2, 8)
        return hidden, audio, context, time, valid

    def test_zero_init_preserves_dense_shared_path(self):
        module = CPEMoEFeedForward(8, num_experts=4, top_k=2, chunk_size=4)
        hidden, audio, context, time, valid = self._inputs()
        result = module(
            hidden,
            audio_mask=audio,
            context_mask=context,
            time_condition=time,
            valid_mask=valid,
        )
        expected = module.shared(hidden) * valid[..., None]
        self.assertTrue(torch.allclose(result.hidden_states, expected))
        # 7 frames -> 4+3, 5 frames -> 4+1.
        self.assertEqual(result.routing["chunk_length"].tolist(), [4, 3, 4, 1])
        self.assertEqual(tuple(result.routing["topk_indices"].shape), (4, 2))
        self.assertTrue(torch.isfinite(result.auxiliary_loss))

    def test_auxiliary_and_expert_outputs_receive_gradients(self):
        module = CPEMoEFeedForward(8, num_experts=4, top_k=2, chunk_size=4)
        hidden, audio, context, time, valid = self._inputs()
        result = module(
            hidden,
            audio_mask=audio,
            context_mask=context,
            time_condition=time,
            valid_mask=valid,
        )
        (result.hidden_states.square().mean() + result.auxiliary_loss).backward()
        expert_grad = sum(
            float(expert.out_proj.weight.grad.abs().sum())
            for expert in module.experts
            if expert.out_proj.weight.grad is not None
        )
        router_grad = float(module.prior_router[-1].weight.grad.abs().sum())
        self.assertGreater(expert_grad, 0.0)
        self.assertGreater(router_grad, 0.0)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_padding_cannot_be_routed(self):
        module = CPEMoEFeedForward(8)
        hidden, audio, context, time, valid = self._inputs()
        audio[1, -1] = True
        with self.assertRaisesRegex(ValueError, "padded"):
            module(
                hidden,
                audio_mask=audio,
                context_mask=context,
                time_condition=time,
                valid_mask=valid,
            )

    def test_no_audio_uses_only_shared_path(self):
        module = CPEMoEFeedForward(8)
        hidden, audio, context, time, valid = self._inputs()
        audio.zero_()
        result = module(
            hidden,
            audio_mask=audio,
            context_mask=context,
            time_condition=time,
            valid_mask=valid,
        )
        self.assertEqual(result.routing["topk_indices"].numel(), 0)
        self.assertEqual(float(result.auxiliary_loss), 0.0)

    def test_activity_change_restarts_the_four_frame_chunk(self):
        module = CPEMoEFeedForward(8, chunk_size=4)
        hidden = torch.randn(1, 8, 8)
        audio = torch.ones(1, 8, dtype=torch.bool)
        valid = torch.ones_like(audio)
        frame_ids = torch.zeros(1, 4, 8, dtype=torch.long)
        frame_ids[0, 0, :3] = 1
        frame_ids[0, 1, 3:] = 2
        result = module(
            hidden,
            audio_mask=audio,
            time_condition=torch.randn(1, 8),
            frame_source_ids=frame_ids,
            valid_mask=valid,
        )
        self.assertEqual(result.routing["chunk_start"].tolist(), [0, 3, 7])
        self.assertEqual(result.routing["chunk_length"].tolist(), [3, 4, 1])

    def test_unselected_experts_stay_in_the_autograd_graph_for_ddp(self):
        module = CPEMoEFeedForward(
            8, num_experts=4, top_k=1, chunk_size=4
        )
        hidden, audio, context, time, valid = self._inputs()
        with torch.no_grad():
            for router in (
                module.delta.prior_router,
                module.delta.evidence_router,
            ):
                router[-1].weight.zero_()
                router[-1].bias.copy_(torch.tensor([12.0, -12.0, -12.0, -12.0]))
        result = module(
            hidden,
            audio_mask=audio,
            context_mask=context,
            time_condition=time,
            valid_mask=valid,
        )
        result.hidden_states.sum().backward()
        for expert in module.experts:
            for parameter in expert.parameters():
                self.assertIsNotNone(parameter.grad)

    def test_router_probabilities_stay_fp32_under_mixed_precision(self):
        module = CPEMoEFeedForward(8, num_experts=4, top_k=2).to(
            torch.bfloat16
        )
        hidden, audio, context, time, valid = self._inputs()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = module(
                hidden.to(torch.bfloat16),
                audio_mask=audio,
                context_mask=context,
                time_condition=time.to(torch.bfloat16),
                valid_mask=valid,
            )
        self.assertEqual(result.routing["probabilities"].dtype, torch.float32)
        self.assertEqual(result.routing["topk_weights"].dtype, torch.float32)

    def test_bfloat16_autocast_scatter_matches_fp32_residual_dtype(self):
        module = CPEMoEFeedForward(8, num_experts=4, top_k=2)
        hidden, audio, context, time, valid = self._inputs()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = module(
                hidden,
                audio_mask=audio,
                context_mask=context,
                time_condition=time,
                valid_mask=valid,
            )
        self.assertEqual(result.hidden_states.dtype, hidden.dtype)
        self.assertEqual(result.routing["probabilities"].dtype, torch.float32)
        (result.hidden_states.square().mean() + result.auxiliary_loss).backward()
        self.assertIsNotNone(module.experts[0].out_proj.weight.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_router_v2_bounds_and_collapse_diagnostics(self):
        module = CPEMoEFeedForward(
            8,
            num_experts=4,
            top_k=2,
            router_temperature=0.7,
            conflict_gate_min=0.05,
            conflict_gate_max=0.95,
            load_balance_weight=0.001,
            conflict_logit_l2_loss_weight=0.0001,
        )
        hidden, audio, context, time, valid = self._inputs()
        result = module(
            hidden,
            audio_mask=audio,
            context_mask=context,
            time_condition=time,
            valid_mask=valid,
        )
        gate = result.routing["conflict_gate"]
        self.assertTrue(bool((gate >= 0.05).all()))
        self.assertTrue(bool((gate <= 0.95).all()))
        for name in (
            "router_max_probability",
            "top1_weight_mean",
            "conflict_gate_saturation_fraction",
            "conflict_logit_l2",
            "prior_evidence_top1_agreement",
        ):
            self.assertIn(name, result.routing)
            self.assertTrue(torch.isfinite(result.routing[name]))
        self.assertGreaterEqual(float(result.routing["router_max_probability"]), 0.25)
        self.assertGreaterEqual(float(result.routing["top1_weight_mean"]), 0.5)
        self.assertTrue(torch.isfinite(result.auxiliary_loss))

    def test_router_v2_rejects_invalid_temperature_and_gate_bounds(self):
        with self.assertRaisesRegex(ValueError, "router_temperature"):
            CPEMoEFeedForward(8, router_temperature=0.0)
        with self.assertRaisesRegex(ValueError, "gate bounds"):
            CPEMoEFeedForward(
                8,
                conflict_gate_min=0.9,
                conflict_gate_max=0.1,
            )

    def test_router_v3_entropy_and_conflict_losses_reach_router_parameters(self):
        module = CPEMoEFeedForward(
            8,
            num_experts=4,
            top_k=2,
            load_balance_weight=0.0,
            router_entropy_loss_weight=1.0,
            router_entropy_target=0.9,
            conflict_logit_l2_loss_weight=1.0,
            conflict_gate_min=0.25,
            conflict_gate_max=0.75,
        )
        hidden, audio, context, time, valid = self._inputs()
        result = module(
            hidden,
            audio_mask=audio,
            context_mask=context,
            time_condition=time,
            valid_mask=valid,
        )
        result.auxiliary_loss.backward()
        router_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in module.named_parameters()
            if any(
                component in name
                for component in ("prior_router", "evidence_router")
            )
            and parameter.grad is not None
        )
        conflict_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in module.named_parameters()
            if "conflict_gate" in name and parameter.grad is not None
        )
        self.assertGreater(router_gradient, 0.0)
        self.assertGreater(conflict_gradient, 0.0)

    def test_router_entropy_target_rejects_degenerate_values(self):
        for target in (0.0, -0.1, math.log(4.0)):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "router_entropy_target"):
                    CPEMoEFeedForward(
                        8,
                        num_experts=4,
                        router_entropy_target=target,
                    )


if __name__ == "__main__":
    unittest.main()
