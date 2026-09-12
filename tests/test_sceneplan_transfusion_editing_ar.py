from __future__ import annotations

import inspect
import unittest

import torch
from torch import nn

from stable_audio_tools.data.sceneplan_transfusion_editing_ar_dataset import (
    EDITING_AR_SELECT_COLUMNS,
    collate_editing_ar,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import (
    _reject_old_sceneplan_keys,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (
    EDITING_DIT_RUNTIME_SELECT_COLUMNS,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (
    EditingScenePlanAdapter,
    ScenePlanTransfusionEditingAR,
    editing_ar_prefix_attention_allowed,
    editing_ar_prefix_attention_bias,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (
    EditingARSourceSemanticBridge,
    multi_positive_source_caption_infonce,
    sha256_group_id,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (
    ScenePlanTransfusionEditingPipeline,
    _align_decoded_sceneplan_to_audio_duration,
)
from stable_audio_tools.models.transformer import ContinuousTransformer


class EditingARLatestContractTests(unittest.TestCase):
    def test_old_sceneplan_is_absent_from_ar_reader_and_forward(self) -> None:
        self.assertFalse(
            any("old_sceneplan" in value.lower() for value in EDITING_AR_SELECT_COLUMNS)
        )
        self.assertNotIn(
            "old_sceneplan_zlib", EDITING_DIT_RUNTIME_SELECT_COLUMNS
        )
        parameters = inspect.signature(ScenePlanTransfusionEditingAR.forward).parameters
        self.assertNotIn("old_sceneplan", parameters)
        self.assertNotIn("source_sceneplan", parameters)
        self.assertNotIn("source_caption", parameters)
        self.assertNotIn("caption_embedding", parameters)
        for method_name in (
            "generate_new_sceneplans",
            "sample_edited_latents",
            "edit_latents",
        ):
            pipeline_parameters = inspect.signature(
                getattr(ScenePlanTransfusionEditingPipeline, method_name)
            ).parameters
            self.assertNotIn("old_sceneplan", pipeline_parameters)
            self.assertNotIn("source_sceneplan", pipeline_parameters)

    def test_collator_fails_closed_on_every_old_plan_alias(self) -> None:
        for key in (
            "old_sceneplan",
            "old-plan",
            "source_sceneplan",
            "source-plan",
            "previous_sceneplan",
            "previous-plan",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                RuntimeError, "old ScenePlan is forbidden"
            ):
                collate_editing_ar([{key: {}}], pad_id=0)

    def test_collator_honors_canonical_source_prefix_envelope(self) -> None:
        row = {
            "pair_id": "pair-0",
            "pair_ordinal": 0,
            "raw_edit_request": "move the sound",
            "source_foa_latent": torch.randn(64, 6),
            "source_attention_mask": torch.tensor(
                [True, True, False, False, False, False]
            ),
            "source_valid_frames": 2,
            "source_prefix_frames": 4,
            "target_token_ids": torch.tensor([1, 7, 2]),
            "target_loss_group_ids": torch.tensor([0, 0, 0]),
        }
        batch = collate_editing_ar([row], pad_id=0)
        self.assertEqual(tuple(batch["source_foa_latent"].shape), (1, 64, 4))
        self.assertEqual(batch["source_valid_frames"].tolist(), [2])
        self.assertEqual(batch["source_prefix_frames"].tolist(), [4])
        self.assertEqual(
            batch["source_attention_mask"].tolist(),
            [[True, True, False, False]],
        )

    def test_collator_rejects_prefix_shorter_than_valid_audio(self) -> None:
        row = {
            "pair_id": "pair-0",
            "pair_ordinal": 0,
            "raw_edit_request": "move the sound",
            "source_foa_latent": torch.randn(64, 6),
            "source_attention_mask": torch.tensor(
                [True, True, True, False, False, False]
            ),
            "source_valid_frames": 3,
            "source_prefix_frames": 2,
            "target_token_ids": torch.tensor([1, 7, 2]),
            "target_loss_group_ids": torch.tensor([0, 0, 0]),
        }
        with self.assertRaisesRegex(RuntimeError, "prefix envelope"):
            collate_editing_ar([row], pad_id=0)

    def test_joint_reader_fails_closed_on_old_sceneplan(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "old ScenePlan is forbidden"):
            _reject_old_sceneplan_keys(
                {"old_sceneplan": {"sources": []}}, where="unit-test"
            )

    def test_prefix_lm_relation_is_exact(self) -> None:
        allowed = editing_ar_prefix_attention_allowed(3, 4)
        expected = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(allowed, expected))
        bias = editing_ar_prefix_attention_bias(
            2, 3, 4, device="cpu", dtype=torch.float32
        )
        self.assertEqual(tuple(bias.shape), (2, 1, 7, 7))
        self.assertTrue(torch.equal(torch.isfinite(bias[0, 0]), expected))

    def test_m2d_audio_bridge_is_exactly_zero_initialized(self) -> None:
        torch.manual_seed(42)
        bridge = EditingARSourceSemanticBridge(
            mode="m2d_audio_caption_aux",
            hidden_dim=16,
            semantic_dim=768,
        ).eval()
        hidden = torch.randn(3, 7, 16)
        embedding = torch.nn.functional.normalize(torch.randn(3, 768), dim=-1)
        injected = bridge.inject(hidden, embedding)
        self.assertTrue(torch.equal(injected, hidden))
        self.assertEqual(
            int(sum(parameter.numel() for parameter in bridge.parameters())),
            768 * 16 + 16 + 16 * 2 + 16 * 768,
        )
        with torch.no_grad():
            bridge.audio_to_hidden.weight[0, 0] = 1.0
        changed = bridge.inject(hidden, embedding)
        self.assertFalse(torch.equal(changed, hidden))

    def test_m2d_ablation_modes_fail_closed(self) -> None:
        hidden = torch.randn(2, 3, 8)
        embedding = torch.nn.functional.normalize(torch.randn(2, 768), dim=-1)
        latent_only = EditingARSourceSemanticBridge(
            mode="latent_only", hidden_dim=8
        )
        with self.assertRaisesRegex(ValueError, "must not consume M2D"):
            latent_only.inject(hidden, embedding)
        m2d = EditingARSourceSemanticBridge(mode="m2d_audio", hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "requires a source M2D"):
            m2d.inject(hidden, None)

    def test_source_caption_infonce_treats_duplicates_as_positives(self) -> None:
        torch.manual_seed(42)
        query = torch.randn(4, 768, requires_grad=True)
        target = torch.randn(4, 768, requires_grad=True)
        caption_hashes = ["1" * 64, "1" * 64, "2" * 64, "3" * 64]
        source_hashes = ["4" * 64, "5" * 64, "6" * 64, "6" * 64]
        caption_ids = torch.tensor(
            [sha256_group_id(value) for value in caption_hashes],
            dtype=torch.int64,
        )
        source_ids = torch.tensor(
            [sha256_group_id(value) for value in source_hashes],
            dtype=torch.int64,
        )
        loss, metrics = multi_positive_source_caption_infonce(
            query,
            target,
            caption_ids,
            source_ids,
            gather_distributed=False,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(float(query.grad.abs().sum()), 0.0)
        self.assertIsNone(target.grad)
        self.assertEqual(float(metrics["mean_positives_per_row"]), 2.0)

    def test_source_caption_query_cannot_read_instruction_or_plan_target(self) -> None:
        """The caption auxiliary is formed before shared/cross attention."""

        class ContextAwareTransformer(nn.Module):
            def forward(self, hidden, *, context, **_kwargs):
                # Make the downstream plan logits observably instruction-aware;
                # the separately returned contrastive query must remain invariant.
                return hidden + context[:, :1, :]

        class EditingDitHarness(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = ContextAwareTransformer()

        torch.manual_seed(42)
        ar = ScenePlanTransfusionEditingAR.__new__(
            ScenePlanTransfusionEditingAR
        )
        nn.Module.__init__(ar)
        ar.editing_dit = EditingDitHarness()
        ar.instruction_conditioner = nn.Identity()
        ar.activation_checkpointing = False
        ar.pad_id = 0
        ar.vocab_size = 32
        ar.source_audio_adapter = nn.Linear(64, 1024, bias=False)
        ar.source_audio_type_embedding = nn.Parameter(torch.zeros(1024))
        ar.plan_type_embedding = nn.Parameter(torch.zeros(1024))
        ar.plan_adapter = EditingScenePlanAdapter(
            vocab_size=ar.vocab_size, hidden_dim=1024, pad_id=ar.pad_id
        )
        ar.source_semantic_bridge = EditingARSourceSemanticBridge(
            mode="m2d_audio_caption_aux", hidden_dim=1024
        )
        # Step zero is intentionally latent-only. Give the bridge a nonzero
        # test weight so this test also proves that M2D can affect the query.
        with torch.no_grad():
            ar.source_semantic_bridge.audio_to_hidden.weight.normal_(std=0.01)
        ar.eval()

        source = torch.randn(2, 64, 5)
        source_mask = torch.ones(2, 5, dtype=torch.bool)
        plan_a = torch.tensor([[1, 2, 3], [1, 4, 5]])
        plan_b = torch.tensor([[1, 9, 10], [1, 11, 12]])
        plan_mask = torch.ones(2, 3, dtype=torch.bool)
        context_a = torch.randn(2, 2, 1024)
        context_b = torch.randn(2, 2, 1024) * 5.0
        context_mask = torch.ones(2, 2, dtype=torch.bool)
        semantic = torch.nn.functional.normalize(torch.randn(2, 768), dim=-1)

        with torch.no_grad():
            logits_a, query_a = ar(
                source,
                source_mask,
                plan_a,
                plan_mask,
                context_a,
                context_mask,
                source_m2d_audio_embedding=semantic,
                return_source_contrastive_query=True,
            )
            logits_b, query_b = ar(
                source,
                source_mask,
                plan_b,
                plan_mask,
                context_b,
                context_mask,
                source_m2d_audio_embedding=semantic,
                return_source_contrastive_query=True,
            )
            _, query_changed_source = ar(
                source + 0.5,
                source_mask,
                plan_a,
                plan_mask,
                context_a,
                context_mask,
                source_m2d_audio_embedding=semantic,
                return_source_contrastive_query=True,
            )
            _, query_changed_m2d = ar(
                source,
                source_mask,
                plan_a,
                plan_mask,
                context_a,
                context_mask,
                source_m2d_audio_embedding=semantic.roll(1, dims=0),
                return_source_contrastive_query=True,
            )

        self.assertTrue(torch.equal(query_a, query_b))
        self.assertFalse(torch.equal(logits_a, logits_b))
        self.assertFalse(torch.equal(query_a, query_changed_source))
        self.assertFalse(torch.equal(query_a, query_changed_m2d))

    def test_future_teacher_tokens_cannot_leak_through_shared_blocks(self) -> None:
        torch.manual_seed(42)
        transformer = ContinuousTransformer(
            16,
            2,
            dim_in=16,
            dim_out=16,
            dim_heads=4,
            cross_attend=False,
            rotary_pos_emb=False,
            zero_init_branch_outputs=False,
        ).eval()
        source_frames = 3
        plan_tokens = 4
        hidden = torch.randn(1, source_frames + plan_tokens, 16)
        changed = hidden.clone()
        changed[:, -1] = torch.randn_like(changed[:, -1]) * 7.0
        bias = editing_ar_prefix_attention_bias(
            1,
            source_frames,
            plan_tokens,
            device="cpu",
            dtype=hidden.dtype,
        )
        mask = torch.ones(1, source_frames + plan_tokens, dtype=torch.bool)
        with torch.no_grad():
            original_output = transformer(
                hidden,
                padding_mask=mask,
                self_attention_bias=bias,
                self_attention_causal=False,
                use_checkpointing=False,
            )
            changed_output = transformer(
                changed,
                padding_mask=mask,
                self_attention_bias=bias,
                self_attention_causal=False,
                use_checkpointing=False,
            )
        # Source states and every earlier plan logit state are invariant to a
        # changed future teacher token. The changed position itself may differ.
        self.assertTrue(
            torch.allclose(
                original_output[:, :-1], changed_output[:, :-1], atol=1e-6, rtol=0
            )
        )
        self.assertFalse(
            torch.allclose(
                original_output[:, -1], changed_output[:, -1], atol=1e-6, rtol=0
            )
        )

    def test_prefix_bias_masks_padded_source_keys_with_varlen_available(self) -> None:
        """The additive Editing bias, not the bypassed varlen path, owns padding."""

        torch.manual_seed(43)
        transformer = ContinuousTransformer(
            16,
            2,
            dim_in=16,
            dim_out=16,
            dim_heads=4,
            cross_attend=False,
            rotary_pos_emb=False,
            zero_init_branch_outputs=False,
        ).eval()
        source_frames = 4
        plan_tokens = 3
        hidden = torch.randn(1, source_frames + plan_tokens, 16)
        changed = hidden.clone()
        # This is an interior invalid source key, which a length-only mask
        # cannot accidentally handle correctly.
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 11.0
        padding_mask = torch.tensor(
            [[True, False, True, True, True, True, True]], dtype=torch.bool
        )
        bias = editing_ar_prefix_attention_bias(
            1,
            source_frames,
            plan_tokens,
            device="cpu",
            dtype=hidden.dtype,
            key_padding_mask=padding_mask,
        )
        self.assertTrue(torch.isneginf(bias[0, 0, :, 1]).all())
        with torch.no_grad():
            original_output = transformer(
                hidden,
                padding_mask=padding_mask,
                self_attention_bias=bias,
                self_attention_causal=False,
                use_checkpointing=False,
            )
            changed_output = transformer(
                changed,
                padding_mask=padding_mask,
                self_attention_bias=bias,
                self_attention_causal=False,
                use_checkpointing=False,
            )
        # Every valid query is invariant to arbitrary content in the invalid
        # source slot. The invalid query itself is outside the loss contract.
        self.assertTrue(
            torch.allclose(
                original_output[:, padding_mask[0]],
                changed_output[:, padding_mask[0]],
                atol=1e-6,
                rtol=0,
            )
        )

    def test_batched_constrained_decode_handles_different_eos_steps(self) -> None:
        class StubAR(ScenePlanTransfusionEditingAR):
            def __init__(self) -> None:
                nn.Module.__init__(self)
                self.pad_id = 0
                self.vocab_size = 16

            def encode_edit_instructions(self, instructions, *, device):
                return (
                    torch.zeros(len(instructions), 1, 1024, device=device),
                    torch.ones(len(instructions), 1, dtype=torch.bool, device=device),
                )

            def forward(
                self,
                source_foa_latent,
                source_attention_mask,
                plan_input_ids,
                plan_attention_mask,
                instruction_context,
                instruction_attention_mask,
            ):
                return torch.zeros(
                    plan_input_ids.shape[0],
                    plan_input_ids.shape[1],
                    self.vocab_size,
                    device=plan_input_ids.device,
                )

        class StubCodec:
            bos_id = 1
            eos_id = 2
            token_to_id = {
                "<source_begin>": 8,
                **{f"<source_slot_{slot}>": 12 + slot for slot in range(4)},
            }

            @staticmethod
            def allowed_next_ids(prefix, *, fixed_duration_sec=None):
                target = (
                    [1, 5, 2]
                    if fixed_duration_sec == 1.0
                    else [1, 6, 7, 2]
                )
                return {target[len(prefix)]}

        source = torch.randn(2, 64, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        generated = StubAR().generate_batch(
            source,
            mask,
            ["short", "long"],
            codec=StubCodec(),
            fixed_duration_sec=[1.0, 2.0],
            max_plan_tokens=8,
        )
        self.assertEqual([value.tolist() for value in generated], [[1, 5, 2], [1, 6, 7, 2]])

    def test_decoded_plan_is_reanchored_to_exact_audio_duration(self) -> None:
        frame_seconds = 1024 / 44_100
        decoded_duration = 3 * frame_seconds
        exact_duration = 2_600 / 44_100
        decoded = {
            "sample_id": "edited_000000",
            "duration_sec": decoded_duration,
            "room": {"type": "dry"},
            "sources": [
                {
                    "source_id": "source_0",
                    "kind": "sound",
                    "description": "a short test sound",
                    "activity": {
                        "onset_sec": 0.0,
                        "offset_sec": decoded_duration,
                    },
                    "trajectory": {
                        "type": "keyframed",
                        "keyframes": [
                            {
                                "time_sec": 0.0,
                                "position": {
                                    "azimuth_deg": 0.0,
                                    "elevation_deg": 0.0,
                                    "distance_m": 1.0,
                                },
                            },
                            {
                                "time_sec": decoded_duration,
                                "position": {
                                    "azimuth_deg": 90.0,
                                    "elevation_deg": 10.0,
                                    "distance_m": 2.0,
                                },
                            },
                        ],
                    },
                    "gain_db": 0.0,
                }
            ],
        }
        aligned = _align_decoded_sceneplan_to_audio_duration(
            decoded, exact_duration
        )
        canonical_duration = round(exact_duration, 6)
        self.assertEqual(aligned["duration_sec"], canonical_duration)
        self.assertEqual(
            aligned["sources"][0]["activity"]["offset_sec"], canonical_duration
        )
        self.assertEqual(
            aligned["sources"][0]["trajectory"]["keyframes"][-1]["time_sec"],
            canonical_duration,
        )
        self.assertEqual(decoded["duration_sec"], decoded_duration)

    def test_decoded_plan_rejects_wrong_duration_frame(self) -> None:
        decoded = {
            "sample_id": "edited_000000",
            "duration_sec": 4 * 1024 / 44_100,
            "room": {"type": "dry"},
            "sources": [
                {
                    "source_id": "source_0",
                    "kind": "sound",
                    "description": "a short test sound",
                    "activity": {
                        "onset_sec": 0.0,
                        "offset_sec": 1024 / 44_100,
                    },
                    "trajectory": {
                        "type": "static",
                        "position": {
                            "azimuth_deg": 0.0,
                            "elevation_deg": 0.0,
                            "distance_m": 1.0,
                        },
                    },
                    "gain_db": 0.0,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "duration frame differs"):
            _align_decoded_sceneplan_to_audio_duration(decoded, 2_600 / 44_100)


if __name__ == "__main__":
    unittest.main()
