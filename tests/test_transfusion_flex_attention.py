from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention requires CUDA")
class TransfusionFlexAttentionTests(unittest.TestCase):
    def test_variable_layout_matches_dense_attention(self):
        """FlexAttention must not broadcast sample zero's modality mask."""

        from transfusion_pytorch.transfusion import Transformer

        torch.manual_seed(1234)
        dense = Transformer(
            dim=64,
            depth=1,
            dim_head=32,
            heads=2,
            time_cond_dim=64,
            use_flex_attn=False,
        ).cuda()
        flex = Transformer(
            dim=64,
            depth=1,
            dim_head=32,
            heads=2,
            time_cond_dim=64,
            use_flex_attn=True,
        ).cuda()
        flex.load_state_dict(dense.state_dict())

        x_dense = torch.randn(
            2, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        x_flex = x_dense.detach().clone().requires_grad_(True)
        # Deliberately different modality spans for the two samples.
        positions = torch.tensor(
            [[[0, 8, 20], [1, 36, 18]], [[0, 5, 11], [1, 24, 31]]],
            device="cuda",
        )
        times = torch.rand(2, 64, device="cuda", dtype=torch.float32)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_dense = dense(x_dense, times=times, modality_positions=positions)
            out_flex = flex(x_flex, times=times, modality_positions=positions)
        torch.testing.assert_close(out_flex, out_dense, rtol=3e-2, atol=3e-2)

        gradient = torch.randn_like(out_dense)
        out_dense.backward(gradient)
        out_flex.backward(gradient)
        torch.testing.assert_close(x_flex.grad, x_dense.grad, rtol=6e-2, atol=6e-2)

    def test_dynamic_objective_layouts_do_not_fall_back(self):
        """Planner (one block) and renderer (three blocks) share Flex safely."""

        from transfusion_pytorch.transfusion import Transformer

        previous_suppress_errors = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = False
        try:
            model = Transformer(
                dim=64,
                depth=1,
                dim_head=32,
                heads=2,
                time_cond_dim=64,
                use_flex_attn=True,
            ).cuda()

            def run(sequence_length: int, modality_count: int) -> None:
                inputs = torch.randn(
                    4,
                    sequence_length,
                    64,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                positions = torch.zeros(
                    4, modality_count, 3, device="cuda", dtype=torch.long
                )
                for index in range(modality_count):
                    positions[:, index, 0] = index
                    positions[:, index, 1] = 5 + index * 10
                    positions[:, index, 2] = 8
                times = torch.rand(4, sequence_length, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        inputs,
                        times=times,
                        modality_positions=positions,
                    )
                    loss = output.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

            # The second length makes Dynamo generalize the sequence axis; the
            # third call then changes the modality-layout width exactly as the
            # Spatial-CoT objective sequence does in one optimizer step.
            run(64, 1)
            run(72, 1)
            run(80, 3)
        finally:
            torch._dynamo.config.suppress_errors = previous_suppress_errors


if __name__ == "__main__":
    unittest.main()
