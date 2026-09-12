from __future__ import annotations

import unittest

import torch


class TransfusionCheckpointingTests(unittest.TestCase):
    def test_layer_checkpointing_matches_full_autograd(self):
        from transfusion_pytorch.transfusion import Transformer

        torch.manual_seed(7)
        full = Transformer(
            dim=32,
            depth=4,
            dim_head=16,
            heads=2,
            time_cond_dim=32,
            use_flex_attn=False,
            checkpoint_layers=False,
        )
        checkpointed = Transformer(
            dim=32,
            depth=4,
            dim_head=16,
            heads=2,
            time_cond_dim=32,
            use_flex_attn=False,
            checkpoint_layers=True,
        )
        checkpointed.load_state_dict(full.state_dict())
        full.train()
        checkpointed.train()

        full_input = torch.randn(2, 48, 32, requires_grad=True)
        checkpointed_input = full_input.detach().clone().requires_grad_(True)
        times = torch.rand(2, 48)
        positions = torch.tensor(
            [[[0, 8, 16], [1, 30, 10]], [[0, 5, 11], [1, 24, 18]]]
        )

        full_output = full(
            full_input, times=times, modality_positions=positions
        )
        checkpointed_output = checkpointed(
            checkpointed_input, times=times, modality_positions=positions
        )
        torch.testing.assert_close(full_output, checkpointed_output)

        gradient = torch.randn_like(full_output)
        full_output.backward(gradient)
        checkpointed_output.backward(gradient)
        torch.testing.assert_close(full_input.grad, checkpointed_input.grad)
        for (full_name, full_parameter), (
            checkpointed_name,
            checkpointed_parameter,
        ) in zip(full.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(full_name, checkpointed_name)
            torch.testing.assert_close(
                full_parameter.grad, checkpointed_parameter.grad
            )


if __name__ == "__main__":
    unittest.main()
