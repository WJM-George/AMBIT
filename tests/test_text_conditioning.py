from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from stable_audio_tools.models.conditioners import QwenTextConditioner
from stable_audio_tools.data.text_conditioning import (
    build_batch_caption_region_ids,
    build_caption_region_ids,
    build_explicit_speech_region_ids,
    build_source_region_ids,
    collect_conditioner_tokenizers,
    find_speech_quote_regions,
)
from stable_audio_tools.training.transfusion import _read_caption


class TextRegionTests(unittest.TestCase):
    def test_source_summary_basis_can_be_token_local(self):
        contextual = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        token_local = torch.tensor([[[0.0, 1.0], [0.0, 2.0]]])

        self.assertIs(
            QwenTextConditioner._source_summary_basis(
                contextual, token_local, mode="contextual"
            ),
            contextual,
        )
        self.assertIs(
            QwenTextConditioner._source_summary_basis(
                contextual, token_local, mode="token_local"
            ),
            token_local,
        )
        with self.assertRaisesRegex(RuntimeError, "were not computed"):
            QwenTextConditioner._source_summary_basis(
                contextual, None, mode="token_local"
            )

    def test_token_local_summary_uses_qwen_input_embeddings(self):
        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tokens = torch.nn.Embedding(4, 2)
                with torch.no_grad():
                    self.tokens.weight.copy_(
                        torch.tensor(
                            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]]
                        )
                    )

            def get_input_embeddings(self):
                return self.tokens

            def forward(self, input_ids, **_):
                return SimpleNamespace(
                    last_hidden_state=self.tokens(input_ids) + 10.0
                )

        conditioner = QwenTextConditioner.__new__(QwenTextConditioner)
        torch.nn.Module.__init__(conditioner)
        conditioner.__dict__["model"] = FakeBackbone().eval()
        conditioner.proj_out = torch.nn.Identity()
        conditioner.padding_mode = "zero"
        conditioner.enable_grad = False
        conditioner.caption_region_embedding = False
        conditioner.source_region_embedding = True
        conditioner.source_region_num_slots = 2
        conditioner.source_region_scale = 1.0
        conditioner.source_summary_mode = "token_local"
        conditioner.source_region_embed = torch.nn.Embedding(3, 2, padding_idx=0)
        with torch.no_grad():
            conditioner.source_region_embed.weight.zero_()

        _, _, summaries, present = conditioner(
            [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.tensor([1, 1, 1]),
                    "source_region_ids": torch.tensor([1, 1, 2]),
                }
            ],
            torch.device("cpu"),
            return_source_summaries=True,
        )

        self.assertTrue(
            torch.allclose(
                summaries,
                torch.tensor([[[0.5, 1.0], [3.0, 3.0]]]),
            )
        )
        self.assertEqual(present.tolist(), [[True, True]])

    def test_persistent_source_and_exact_transcript_regions(self):
        text = 'Scene: dog; speech saying "hello".'
        offsets = torch.tensor(
            [[0, 5], [7, 10], [12, 18], [19, 25], [26, 27], [27, 32], [32, 33]]
        )
        mask = torch.ones(offsets.shape[0], dtype=torch.bool)
        source_regions = [
            {
                "source_id": "source_0",
                "source_slot": 0,
                "start": 7,
                "end": 10,
            },
            {
                "source_id": "source_1",
                "source_slot": 1,
                "start": 12,
                "end": 33,
            },
        ]
        transcript_regions = [
            {
                "source_id": "source_1",
                "source_slot": 1,
                "start": 27,
                "end": 32,
            }
        ]
        source_ids = build_source_region_ids(
            text, offsets, mask, source_regions, max_sources=4
        )
        speech_ids = build_explicit_speech_region_ids(
            text, offsets, mask, source_regions, transcript_regions
        )
        self.assertEqual(source_ids.tolist(), [0, 1, 2, 2, 2, 2, 2])
        self.assertEqual(speech_ids.tolist(), [0, 0, 1, 1, 1, 2, 1])

    def test_noncontiguous_source_slot_keeps_persistent_id(self):
        text = "Only a sliding door"
        offsets = torch.tensor([[0, 4], [5, 6], [7, 14], [15, 19]])
        mask = torch.ones(4, dtype=torch.bool)
        ids = build_source_region_ids(
            text,
            offsets,
            mask,
            [
                {
                    "source_id": "source_1",
                    "source_slot": 1,
                    "start": 7,
                    "end": 19,
                }
            ],
            max_sources=4,
        )
        self.assertEqual(ids.tolist(), [0, 0, 2, 2])

    def test_source_region_truncation_fails_closed(self):
        text = "source zero and source one"
        offsets = torch.tensor([[0, 6], [7, 11]])
        mask = torch.ones(2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "truncated"):
            build_source_region_ids(
                text,
                offsets,
                mask,
                [
                    {
                        "source_id": "source_0",
                        "source_slot": 0,
                        "start": 0,
                        "end": 11,
                    },
                    {
                        "source_id": "source_1",
                        "source_slot": 1,
                        "start": 16,
                        "end": len(text),
                    },
                ],
            )

    def test_tts_caption_regions(self):
        text = 'A woman says "hello" from the left'
        spans = find_speech_quote_regions(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[1][0] : spans[1][1]], "hello")
        self.assertEqual(spans[1][2], 2)

    def test_spatial_librispeech_caption_regions(self):
        text = 'A voice is in front. Spoken words: "turn around"'
        spans = find_speech_quote_regions(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[0][0] : spans[0][1]].strip(), "Spoken words:")
        self.assertEqual(text[spans[1][0] : spans[1][1]], "turn around")

    def test_unknown_caption_remains_untyped(self):
        text = 'A dog barks near a person called "Sam"'
        offsets = torch.tensor([[0, 1], [2, 5], [31, 36]])
        mask = torch.ones(3, dtype=torch.bool)
        ids = build_caption_region_ids(text, offsets, mask)
        self.assertTrue(torch.equal(ids, torch.zeros_like(ids)))

    def test_batch_matches_single_item_builder(self):
        texts = ['A speaker says "one"', 'Spoken words: "two"']
        offsets = torch.tensor(
            [
                [[0, 9], [10, 14], [15, 20], [20, 21]],
                [[0, 6], [7, 13], [14, 19], [19, 20]],
            ]
        )
        mask = torch.ones(2, 4, dtype=torch.bool)
        batch = build_batch_caption_region_ids(texts, offsets, mask)
        singles = torch.stack(
            [
                build_caption_region_ids(texts[i], offsets[i], mask[i])
                for i in range(2)
            ]
        )
        self.assertTrue(torch.equal(batch, singles))


class TokenizerDiscoveryTests(unittest.TestCase):
    def test_discovers_dit_and_transfusion_conditioners(self):
        tokenizer_a = object()
        tokenizer_b = object()
        dit_conditioner = SimpleNamespace(
            tokenizer=tokenizer_a,
            max_length=128,
            caption_region_embedding=True,
            caption_region_strategy="speech_quote_v1",
        )
        transfusion_conditioner = SimpleNamespace(
            tokenizer=tokenizer_b,
            max_length=64,
            caption_region_embedding=False,
        )
        model = SimpleNamespace(
            conditioner=SimpleNamespace(conditioners={"prompt": dit_conditioner}),
            qwen_conditioner=transfusion_conditioner,
        )
        config = {"model": {"text": {"metadata_key": "scene_prompt"}}}
        specs = collect_conditioner_tokenizers(model, config)

        self.assertIs(specs["prompt"][0], tokenizer_a)
        self.assertTrue(specs["prompt"][2]["enabled"])
        self.assertIs(specs["scene_prompt"][0], tokenizer_b)
        self.assertIsNone(specs["scene_prompt"][2])

    def test_transfusion_prefers_worker_tokens_for_qwen_prefix(self):
        tokenized = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
        }
        metadata = {
            "prompt": tokenized,
            "prompt_text": 'A speaker says "hello"',
        }
        self.assertIs(
            _read_caption(metadata, ("prompt",), allow_tokenized=True),
            tokenized,
        )
        self.assertEqual(
            _read_caption(metadata, ("prompt",), allow_tokenized=False),
            metadata["prompt_text"],
        )


if __name__ == "__main__":
    unittest.main()
