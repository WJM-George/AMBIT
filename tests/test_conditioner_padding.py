from __future__ import annotations

import unittest

import torch

from stable_audio_tools.models.conditioners import Conditioner, MultiConditioner


class _EchoConditioner(Conditioner):
    def __init__(self):
        super().__init__(1, 1)

    def forward(self, values, device):
        return values, device


class ConditionerPaddingTests(unittest.TestCase):
    def test_identity_output_projection_is_exact_and_trainable(self):
        conditioner = Conditioner(
            4,
            4,
            project_out=True,
            project_out_init="identity",
        )
        values = torch.randn(2, 3, 4)

        projected = conditioner.proj_out(values)

        self.assertTrue(torch.equal(projected, values))
        self.assertTrue(conditioner.proj_out.weight.requires_grad)
        self.assertTrue(
            torch.equal(conditioner.proj_out.weight, torch.eye(4))
        )

    def test_identity_output_projection_rejects_non_square_shape(self):
        with self.assertRaisesRegex(ValueError, "square"):
            Conditioner(4, 8, project_out_init="identity")

    def test_zero_padding_preserves_embedding_dtype(self):
        conditioner = Conditioner(4, 4, padding_mode="zero")
        embeddings = torch.ones(2, 3, 4, dtype=torch.bfloat16)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

        padded = conditioner.apply_padding(embeddings, mask)

        self.assertEqual(padded.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(padded[:, 2], torch.zeros_like(padded[:, 2])))

    def test_learned_padding_preserves_embedding_dtype(self):
        conditioner = Conditioner(4, 4, padding_mode="learned")
        embeddings = torch.ones(1, 2, 4, dtype=torch.bfloat16)
        mask = torch.tensor([[1, 0]], dtype=torch.bool)

        padded = conditioner.apply_padding(embeddings, mask)

        self.assertEqual(padded.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(padded[:, 0], embeddings[:, 0]))

    def test_multi_conditioner_unwraps_only_singletons_per_item(self):
        conditioner = MultiConditioner(
            {"prompt": _EchoConditioner()},
            default_keys={"prompt": "text"},
        )
        values, device = conditioner(
            [
                {"prompt": ["keep", "both"]},
                {"text": ["unwrap-me"]},
                {"prompt": "direct"},
            ],
            "cpu",
        )["prompt"]

        self.assertEqual(values, [["keep", "both"], "unwrap-me", "direct"])
        self.assertEqual(device, "cpu")


if __name__ == "__main__":
    unittest.main()
