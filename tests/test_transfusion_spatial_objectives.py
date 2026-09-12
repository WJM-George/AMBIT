from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from stable_audio_tools.models.conditioners import QwenTextConditioner
from stable_audio_tools.data.spatial_plan_codec import (
    SpatialPlanCodec,
    create_codec_artifact,
)
from stable_audio_tools.data.scene_plan import semantic_caption_from_scene_plan
from stable_audio_tools.models.transfusion_spatial import (
    _SourceRegionalFrameAdapter,
    TransfusionSpatialWrapper,
    _SoftTextEmbed,
)
from stable_audio_tools.training.transfusion import TransfusionSpatialTrainingWrapper


class _FakeQwen(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.calls = 0

    def forward(
        self,
        captions,
        device,
        *,
        return_source_summaries: bool = False,
    ):
        self.calls += 1
        embeddings = self.bias.to(device).view(1, 1, -1).expand(len(captions), 3, -1) + 0.1
        mask = torch.ones(len(captions), 3, dtype=torch.bool, device=device)
        if return_source_summaries:
            slots = int(
                getattr(self, "source_region_embed").num_embeddings - 1
            )
            ramp = torch.linspace(
                -1.0,
                1.0,
                embeddings.shape[-1],
                device=device,
            )
            slot_summaries = torch.stack(
                [torch.roll(ramp, shifts=slot) for slot in range(slots)]
            )
            summaries = slot_summaries.unsqueeze(0).expand(
                len(captions), -1, -1
            ).clone()
            source_mask = torch.ones(
                len(captions), slots, dtype=torch.bool, device=device
            )
            return embeddings, mask, summaries, source_mask
        return embeddings, mask


class _RecordingPlannerCore(nn.Module):
    def __init__(self, wrapper: TransfusionSpatialWrapper):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.text_embed = wrapper._soft_text_embed
        self.num_modalities = wrapper.model.num_modalities
        self.sos_id = wrapper.model.sos_id
        self.vocab_size = wrapper.num_text_tokens
        self.times_shapes = []

    def forward(self, _samples, *, times=None, **_kwargs):
        self.times_shapes.append(None if times is None else tuple(times.shape))
        return torch.zeros(
            (1, 1, self.vocab_size),
            device=self.anchor.device,
        )


def _fake_init_qwen(self, _text_config, transformer_dim: int):
    self.qwen_conditioner = _FakeQwen(transformer_dim)
    if _text_config.get("source_region_embedding", False):
        slots = int(_text_config.get("source_region_num_slots", 4))
        self.qwen_conditioner.source_region_embed = nn.Embedding(
            slots + 1,
            transformer_dim,
            padding_idx=0,
        )
        nn.init.zeros_(self.qwen_conditioner.source_region_embed.weight)
    self._soft_text_embed = _SoftTextEmbed(
        self.model.text_embed,
        self.model.sos_id,
        self.model.null_text_id,
    )
    self.model.text_embed = self._soft_text_embed


def _tiny_config(codec_path: str) -> dict:
    return {
        "sample_rate": 44_100,
        "audio_channels": 4,
        "model": {
            "text": {
                "mode": "qwen_prefix",
                "num_text_tokens": 4096,
                "tie_text_embeddings": True,
                "ar_target": {
                    "type": "spatial_plan",
                    "codec_path": codec_path,
                },
            },
            "modalities": [
                {
                    "id": "spatial_traj",
                    "dim_latent": 4,
                    "channel_first_latent": False,
                    "modality_num_dim": 1,
                    "add_pos_emb": True,
                    "default_shape": [4],
                    "source": {"type": "foa_intensity_trajectory"},
                },
                {
                    "id": "foa_latent",
                    "dim_latent": 8,
                    "channel_first_latent": True,
                    "modality_num_dim": 1,
                    "add_pos_emb": True,
                    "default_shape": [4],
                    "source": {"type": "pretransform_latent"},
                },
            ],
            "transfusion": {
                "modality_order": ["spatial_traj", "foa_latent"],
                "modality_time_policy": "upstream_default",
                "transformer": {
                    "dim": 32,
                    "depth": 2,
                    "time_cond_dim": 16,
                    "dim_head": 8,
                    "heads": 4,
                    "ff_expansion_factor": 2,
                    "use_flex_attn": False,
                    "unet_skips": True,
                },
                "text_loss_weight": 0.0,
                "velocity_consistency_loss_weight": 0.0,
                "reconstruction_loss_weight": 0.0,
                "prob_uncond": 0.0,
                "model_output_clean": True,
            },
        },
    }


class TransfusionSpatialObjectiveTests(unittest.TestCase):
    def test_global_stratified_logit_normal_biases_toward_noise(self):
        kwargs = {
            "batch_size": 8192,
            "device": torch.device("cpu"),
            "minimum": 0.0,
            "maximum": 1.0,
            "global_rank": 0,
            "world_size": 1,
            "schedule_step": 17,
            "seed": 20260813,
            "logit_mean": -0.77,
            "logit_std": 1.0,
            "uniform_mix": 0.1,
        }
        torch.manual_seed(91)
        first = (
            TransfusionSpatialTrainingWrapper
            ._global_stratified_logit_normal_time_column(**kwargs)
        )
        torch.manual_seed(91)
        replay = (
            TransfusionSpatialTrainingWrapper
            ._global_stratified_logit_normal_time_column(**kwargs)
        )

        torch.testing.assert_close(first, replay, rtol=0.0, atol=0.0)
        self.assertTrue(bool(((first > 0.0) & (first < 1.0)).all()))
        self.assertGreater(float((first < 0.5).float().mean()), 0.70)
        self.assertLess(float(first.mean()), 0.40)
        self.assertGreater(float(first.mean()), 0.32)

    def test_logit_normal_time_mapping_preserves_target_range(self):
        unit = torch.linspace(0.0, 1.0, 4097)
        mapped = TransfusionSpatialTrainingWrapper._logit_normal_time_from_unit(
            unit,
            minimum=0.2,
            maximum=0.8,
            logit_mean=-0.77,
            logit_std=1.0,
            uniform_mix=0.1,
        )
        self.assertTrue(bool(((mapped >= 0.2) & (mapped <= 0.8)).all()))
        self.assertTrue(bool(torch.isfinite(mapped).all()))

    def test_qwen_source_summary_pooling_preserves_exact_slots(self):
        embeddings = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [100.0, 100.0],
                ]
            ]
        )
        source_ids = torch.tensor([[1, 1, 2, 0]])
        attention = torch.tensor([[True, True, True, False]])
        summaries, present = QwenTextConditioner._pool_source_region_summaries(
            embeddings,
            source_ids,
            attention,
            num_slots=3,
        )
        self.assertTrue(torch.equal(summaries[0, 0], torch.tensor([2.0, 3.0])))
        self.assertTrue(torch.equal(summaries[0, 1], torch.tensor([5.0, 6.0])))
        self.assertTrue(torch.equal(summaries[0, 2], torch.zeros(2)))
        self.assertTrue(torch.equal(present, torch.tensor([[True, True, False]])))

    def test_source_regional_frame_adapter_preserves_slot_assignment(self):
        adapter = _SourceRegionalFrameAdapter(2, 2, 2)
        with torch.no_grad():
            adapter.weight[0, 0, 0] = 1.0
            adapter.weight[1, 0, 1] = 1.0
        first = torch.zeros(2, 2, 3)
        first[0, 0] = 1.0
        swapped = torch.zeros_like(first)
        swapped[1, 0] = 1.0
        first_output = adapter(first)
        swapped_output = adapter(swapped)
        self.assertFalse(torch.equal(first_output, swapped_output))
        self.assertTrue(torch.equal(first_output[:, 0], torch.ones(3)))
        self.assertTrue(torch.equal(swapped_output[:, 1], torch.ones(3)))

    def test_source_semantic_geometry_fusion_is_zero_init_and_binding_sensitive(self):
        adapter = _SourceRegionalFrameAdapter(
            2,
            2,
            2,
            semantic_rank=1,
        )
        tracks = torch.zeros(2, 2, 3)
        tracks[:, 0] = 1.0
        semantics = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        initial = adapter(tracks, source_semantics=semantics)
        self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))

        with torch.no_grad():
            adapter.semantic_down.weight.copy_(torch.tensor([[1.0, 0.0]]))
            adapter.semantic_track_weight.zero_()
            adapter.semantic_track_weight[0, 0, 0] = 1.0
            adapter.semantic_track_weight[1, 0, 0] = 2.0
            adapter.semantic_out.weight.copy_(torch.tensor([[1.0], [0.5]]))

        aligned = adapter(tracks, source_semantics=semantics)
        swapped = adapter(tracks, source_semantics=semantics.flip(0))
        self.assertFalse(torch.allclose(aligned, swapped))
        masked = adapter(
            tracks,
            source_semantics=semantics,
            source_semantic_mask=torch.tensor([True, False]),
        )
        self.assertFalse(torch.allclose(aligned, masked))

    def test_shared_source_executor_is_id_permutation_invariant_but_binding_sensitive(self):
        torch.manual_seed(41)
        adapter = _SourceRegionalFrameAdapter(
            2,
            3,
            4,
            semantic_rank=2,
            source_aggregation="permutation_equivariant",
        )
        with torch.no_grad():
            adapter.semantic_out.weight.normal_(std=0.2)

        tracks = torch.randn(2, 3, 5)
        tracks[:, 0] = 1.0
        semantics = torch.randn(2, 4)
        present = torch.tensor([True, True])

        aligned = adapter(
            tracks,
            source_semantics=semantics,
            source_semantic_mask=present,
        )
        renamed = adapter(
            tracks.flip(0),
            source_semantics=semantics.flip(0),
            source_semantic_mask=present.flip(0),
        )
        torch.testing.assert_close(renamed, aligned)

        misbound = adapter(
            tracks,
            source_semantics=semantics.flip(0),
            source_semantic_mask=present,
        )
        self.assertFalse(torch.allclose(misbound, aligned))

    def test_renderer_routes_source_summaries_into_zero_init_fusion(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            config = _tiny_config(str(codec_path))
            config["model"]["text"].update(
                {
                    "source_region_embedding": True,
                    "source_region_num_slots": 2,
                }
            )
            config["model"]["transfusion"]["source_region_target_residual"] = {
                "enabled": True,
                "target_modality": "foa_latent",
                "max_sources": 2,
                "init": "zero",
                "semantic_fusion": {
                    "enabled": True,
                    "rank": 4,
                },
            }
            with patch.object(
                TransfusionSpatialWrapper,
                "_init_qwen_prefix",
                _fake_init_qwen,
            ):
                wrapper = TransfusionSpatialWrapper(config)

            tracks = torch.zeros(2, 8, 4)
            tracks[:, 0] = 1.0
            tracks[:, 1] = 1.0
            sample = wrapper.build_renderer_sample(
                torch.tensor([1, 2, 3]),
                {
                    "spatial_traj": torch.randn(4, 4),
                    "foa_latent": torch.randn(8, 4),
                },
                semantic_caption={
                    "text": "source_0 is speech; source_1 is music",
                    "source_regions": [],
                },
                source_region_tracks=tracks,
            )
            loss = wrapper.forward_renderer(
                [sample],
                times=torch.tensor([[1.0, 0.5]]),
                return_breakdown=False,
            )
            loss.backward()
            adapter = wrapper.model.frame_aligned_source_region_adapter
            self.assertIsNotNone(adapter.semantic_out.weight.grad)
            self.assertGreater(
                float(adapter.semantic_out.weight.grad.abs().sum()), 0.0
            )

    def test_frame_aligned_anchor_adapter_is_zero_init_and_receives_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            config = _tiny_config(str(codec_path))
            config["model"]["transfusion"]["anchor_target_residual"] = {
                "enabled": True,
                "control_modality": "spatial_traj",
                "target_modality": "foa_latent",
                "init": "zero",
                "bias": False,
            }
            with patch.object(
                TransfusionSpatialWrapper,
                "_init_qwen_prefix",
                _fake_init_qwen,
            ):
                wrapper = TransfusionSpatialWrapper(config)

            anchor = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.5],
                    [-1.0, 0.0, 0.0, 0.5],
                    [0.0, -1.0, 0.0, 0.5],
                ]
            )
            target = torch.randn(8, 4)
            sample = wrapper.build_renderer_sample(
                torch.tensor([1, 2, 3]),
                {"spatial_traj": anchor, "foa_latent": target},
            )
            residuals = wrapper.build_frame_aligned_anchor_residuals([sample])
            self.assertIsNotNone(residuals)
            projected = residuals[0][1]
            self.assertTrue(torch.equal(projected, torch.zeros_like(projected)))

            loss = wrapper.forward_renderer(
                [sample],
                times=torch.tensor([[1.0, 0.5]]),
                return_breakdown=False,
            )
            loss.backward()
            adapter = wrapper.model.frame_aligned_anchor_adapter
            self.assertIsNotNone(adapter.weight.grad)
            self.assertTrue(torch.isfinite(adapter.weight.grad).all())
            self.assertGreater(float(adapter.weight.grad.abs().sum()), 0.0)

    def test_parameter_selection_can_expose_only_last_transformer_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            training = TransfusionSpatialTrainingWrapper(
                wrapper,
                optimizer_configs={
                    "transfusion": {
                        "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                    }
                },
                use_ema=False,
                parameter_selection={
                    "mode": "modality_interfaces",
                    "modality_ids": ["spatial_traj"],
                    "include_input_projection": False,
                    "include_position_mlp": False,
                    "include_output_projection": False,
                    "include_boundary_tokens": False,
                    "include_transformer_last_n_layers": 1,
                },
            )

            first_layer = wrapper.model.transformer.layers[0]
            last_layer = wrapper.model.transformer.layers[-1]
            self.assertTrue(all(not p.requires_grad for p in first_layer.parameters()))
            self.assertTrue(all(p.requires_grad for p in last_layer.parameters()))
            self.assertTrue(wrapper.model.transformer.norm.gamma.requires_grad)
            self.assertFalse(wrapper.qwen_conditioner.bias.requires_grad)
            self.assertEqual(
                training.parameter_selection_report["transformer_last_n_layers"],
                1,
            )

    def test_parameter_selection_targets_one_output_modality(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            training = TransfusionSpatialTrainingWrapper(
                wrapper,
                optimizer_configs={
                    "transfusion": {
                        "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                    }
                },
                use_ema=False,
                parameter_selection={
                    "mode": "modality_interfaces",
                    "modality_ids": ["spatial_traj"],
                    "include_input_projection": False,
                    "include_position_mlp": False,
                    "include_output_projection": True,
                    "output_modality_ids": ["foa_latent"],
                    "include_boundary_tokens": False,
                },
            )

            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in wrapper.model.model_to_latent_projs[
                        0
                    ].parameters()
                )
            )
            self.assertTrue(
                all(
                    parameter.requires_grad
                    for parameter in wrapper.model.model_to_latent_projs[
                        1
                    ].parameters()
                )
            )
            self.assertEqual(
                training.parameter_selection_report["output_modality_ids"],
                ["foa_latent"],
            )

    def test_parameter_selection_can_train_native_dense_kv_lora_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            adapter = nn.Module()
            adapter.semantic_down = nn.Linear(8, 3, bias=False)
            adapter.semantic_track_weight = nn.Parameter(
                torch.ones(2, 5, 3)
            )
            adapter.condition_token_out = nn.Linear(6, 4, bias=False)
            adapter.dense_kv_lora_down = nn.Parameter(
                torch.ones(2, 8, 3)
            )
            adapter.dense_kv_lora_up = nn.Parameter(
                torch.zeros(2, 3, 16)
            )
            adapter.existing_parameter = nn.Parameter(torch.ones(7))
            wrapper.model.frame_aligned_source_region_adapter = adapter
            expected = {
                id(adapter.semantic_track_weight),
                *(id(parameter) for parameter in adapter.semantic_down.parameters()),
                *(
                    id(parameter)
                    for parameter in adapter.condition_token_out.parameters()
                ),
                id(adapter.dense_kv_lora_down),
                id(adapter.dense_kv_lora_up),
            }
            training = TransfusionSpatialTrainingWrapper(
                wrapper,
                optimizer_configs={
                    "transfusion": {
                        "optimizer": {
                            "type": "AdamW",
                            "config": {"lr": 1.0e-4},
                        },
                        "parameter_group_lr_scales": {
                            "source_region": 10.0
                        },
                    }
                },
                use_ema=False,
                parameter_selection={
                    "mode": "modality_interfaces",
                    "modality_ids": ["spatial_traj"],
                    "include_input_projection": False,
                    "include_position_mlp": False,
                    "include_output_projection": False,
                    "include_boundary_tokens": False,
                    "include_source_dense_condition_kv_lora_adapter": True,
                },
            )

            self.assertFalse(adapter.existing_parameter.requires_grad)
            optimizer = training.configure_optimizers()[0]
            self.assertEqual(len(optimizer.param_groups), 1)
            group = optimizer.param_groups[0]
            self.assertEqual(group["group_name"], "source_region")
            self.assertAlmostEqual(group["lr"], 1.0e-3)
            self.assertEqual(
                {id(parameter) for parameter in group["params"]}, expected
            )

    def test_parameter_selection_can_train_registered_dense_joint_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(8, 16),
                        nn.SiLU(),
                        nn.Linear(16, 8),
                    )
                    for _ in range(2)
                ]
            )
            wrapper.model.dense_joint_late_blocks = blocks
            expected = {id(parameter) for parameter in blocks.parameters()}
            training = TransfusionSpatialTrainingWrapper(
                wrapper,
                optimizer_configs={
                    "transfusion": {
                        "optimizer": {
                            "type": "AdamW",
                            "config": {"lr": 5.0e-6},
                        },
                        "parameter_group_lr_scales": {
                            "dense_joint": 10.0
                        },
                    }
                },
                use_ema=False,
                parameter_selection={
                    "mode": "modality_interfaces",
                    "modality_ids": ["spatial_traj"],
                    "include_input_projection": False,
                    "include_position_mlp": False,
                    "include_output_projection": False,
                    "include_boundary_tokens": False,
                    "include_dense_joint_late_blocks": True,
                },
            )

            optimizer = training.configure_optimizers()[0]
            self.assertEqual(len(optimizer.param_groups), 1)
            group = optimizer.param_groups[0]
            self.assertEqual(group["group_name"], "dense_joint")
            self.assertAlmostEqual(group["lr"], 5.0e-5)
            self.assertEqual(
                {id(parameter) for parameter in group["params"]}, expected
            )
            self.assertEqual(
                training.parameter_selection_report[
                    "dense_joint_optimizer_parameters"
                ],
                sum(parameter.numel() for parameter in blocks.parameters()),
            )

    def test_source_region_interfaces_can_use_a_higher_lr_group(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            config = _tiny_config(str(codec_path))
            config["model"]["text"].update(
                {
                    "source_region_embedding": True,
                    "source_region_num_slots": 2,
                }
            )
            config["model"]["transfusion"]["source_region_target_residual"] = {
                "enabled": True,
                "target_modality": "foa_latent",
                "max_sources": 2,
                "init": "zero",
            }
            with patch.object(
                TransfusionSpatialWrapper,
                "_init_qwen_prefix",
                _fake_init_qwen,
            ):
                wrapper = TransfusionSpatialWrapper(config)
            training = TransfusionSpatialTrainingWrapper(
                wrapper,
                optimizer_configs={
                    "transfusion": {
                        "optimizer": {
                            "type": "AdamW",
                            "config": {"lr": 1.0e-4, "weight_decay": 0.01},
                        },
                        "parameter_group_lr_scales": {"source_region": 5.0},
                    }
                },
                use_ema=False,
                parameter_selection={
                    "mode": "modality_interfaces",
                    "modality_ids": ["spatial_traj"],
                    "include_input_projection": True,
                    "include_position_mlp": False,
                    "include_output_projection": False,
                    "include_boundary_tokens": False,
                    "include_source_region_embedding": True,
                    "include_frame_aligned_source_region_adapter": True,
                },
            )

            optimizer = training.configure_optimizers()[0]
            groups = {group["group_name"]: group for group in optimizer.param_groups}
            self.assertEqual(set(groups), {"base", "source_region"})
            self.assertAlmostEqual(groups["base"]["lr"], 1.0e-4)
            self.assertAlmostEqual(groups["source_region"]["lr"], 5.0e-4)
            expected_source_ids = {
                id(parameter)
                for parameter in (
                    *wrapper.qwen_conditioner.source_region_embed.parameters(),
                    *wrapper.model.frame_aligned_source_region_adapter.parameters(),
                )
            }
            actual_source_ids = {
                id(parameter) for parameter in groups["source_region"]["params"]
            }
            self.assertEqual(actual_source_ids, expected_source_ids)
            self.assertTrue(
                expected_source_ids.isdisjoint(
                    id(parameter) for parameter in groups["base"]["params"]
                )
            )
    def test_creation_edit_balancing_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            training = TransfusionSpatialTrainingWrapper(
                wrapper,
                optimizer_configs={
                    "transfusion": {
                        "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                    }
                },
                use_ema=False,
                example_selection={
                    "mode": "creation_edit_balanced",
                    "context_presence_key": "previous_foa_present",
                    "creation_repeat": 3,
                    "edit_repeat": 1,
                },
            )
            examples = [
                {"metadata": {"previous_foa_present": False, "turn": 0}},
                {"metadata": {"previous_foa_present": True, "turn": 1}},
                {"metadata": {"previous_foa_present": True, "turn": 2}},
                {"metadata": {"previous_foa_present": True, "turn": 3}},
            ]
            selected = training._select_training_examples(examples)
            self.assertEqual(len(selected), 6)
            self.assertEqual(
                sum(not item["metadata"]["previous_foa_present"] for item in selected),
                3,
            )
            self.assertEqual(
                sum(item["metadata"]["previous_foa_present"] for item in selected),
                3,
            )

    def test_edit_delta_ce_is_normalized_per_edit_sample(self):
        token_loss = torch.tensor([[1.0, 9.0], [1.0, 1.0]])
        token_weights = torch.ones_like(token_loss)
        edit_mask = torch.tensor([[False, True], [False, False]])
        total, breakdown = TransfusionSpatialWrapper._combine_planner_token_losses(
            token_loss,
            token_weights,
            edit_mask,
            edit_delta_ce_weight=0.5,
        )
        self.assertAlmostEqual(float(breakdown["base_ce"]), 3.0)
        self.assertAlmostEqual(float(breakdown["edit_ce"]), 9.0)
        self.assertAlmostEqual(float(breakdown["edit_sample_fraction"]), 0.5)
        self.assertAlmostEqual(float(breakdown["edit_token_fraction"]), 0.25)
        self.assertAlmostEqual(float(total), 7.5)

    def test_renderer_semantic_caption_is_compiled_without_spatial_controls(self):
        caption = semantic_caption_from_scene_plan(
            {
                "scene": {
                    "room": {"type": "cathedral", "rt60_s": 2.5},
                    "sources": [
                        {
                            "event": {"label": "Speech"},
                            "content": {"transcript": "turn left"},
                            "motion": {
                                "type": "static",
                                "keyframes": [
                                    {
                                        "position": {
                                            "azimuth_deg": -90.0,
                                            "distance_m": 1.0,
                                        }
                                    }
                                ],
                            },
                        },
                        {
                            "event": {"label": "Bird"},
                            "content": {"transcript": None},
                        },
                    ],
                }
            }
        )
        self.assertIn('speech saying "turn left"', caption)
        self.assertIn("Bird", caption)
        self.assertNotIn("azimuth", caption)
        self.assertNotIn("cathedral", caption)
        self.assertNotIn("2.5", caption)

    def _make_wrapper(self, codec_path: str):
        with patch.object(
            TransfusionSpatialWrapper,
            "_init_qwen_prefix",
            _fake_init_qwen,
        ):
            return TransfusionSpatialWrapper(_tiny_config(codec_path))

    def test_route_warmstart_prefers_ema_and_skips_changed_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            target = wrapper.state_dict()
            transformer_key = "model.transformer.norm.gamma"
            transformer_value = target[transformer_key]
            qwen_value = target["qwen_conditioner.bias"]
            plan_row_before = wrapper._soft_text_embed.fallback.weight[0].detach().clone()
            source_embedding = torch.full(
                (256 + 3 + 2 * 2 + 129, transformer_value.numel()), 7.0
            )
            source = {
                "transfusion.model.transformer.norm.gamma": torch.full_like(
                    transformer_value, 2.0
                ),
                "transfusion_ema.ema_model.transformer.norm.gamma": torch.full_like(
                    transformer_value, 3.0
                ),
                "transfusion.qwen_conditioner.bias": torch.full_like(qwen_value, 4.0),
                "text_conditioner_ema.shadow_00000": torch.full_like(qwen_value, 5.0),
                "transfusion_ema.ema_model.text_embed.fallback.weight": source_embedding,
                # A T1/T2 vocabulary change must be reported and skipped.
                "transfusion_ema.ema_model.to_text_logits.weight": torch.zeros(3, 3),
            }
            source_config = {
                "model": {
                    "text": {"mode": "qwen_prefix"},
                    "modalities": [
                        {"id": "spatial_traj"},
                        {"id": "foa_latent"},
                    ],
                }
            }

            report = wrapper.load_pretrained_route_state_dict(
                source,
                source_model_config=source_config,
                source_text_conditioner_ema_names=["bias"],
            )

            self.assertTrue(
                torch.equal(
                    wrapper.state_dict()[transformer_key],
                    torch.full_like(transformer_value, 3.0),
                )
            )
            self.assertTrue(
                torch.equal(
                    wrapper.state_dict()["qwen_conditioner.bias"],
                    torch.full_like(qwen_value, 5.0),
                )
            )
            self.assertEqual(report["loaded_from"]["ema_core"], 1)
            self.assertEqual(report["loaded_from"]["conditioner_ema"], 1)
            self.assertEqual(
                report["modality_mapping"], {"spatial_traj": 0, "foa_latent": 1}
            )
            self.assertEqual(report["text_rows_loaded"]["plan_vocab"], 0)
            self.assertEqual(report["text_rows_loaded"]["control"], 136)
            self.assertTrue(
                torch.equal(wrapper._soft_text_embed.fallback.weight[0], plan_row_before)
            )
            self.assertTrue(
                torch.equal(
                    wrapper._soft_text_embed.fallback.weight[wrapper.model.sos_id],
                    torch.full_like(plan_row_before, 7.0),
                )
            )
            self.assertTrue(
                any(
                    item["target_key"] == "model.to_text_logits.weight"
                    for item in report["shape_mismatches"]
                )
            )

            remapped = wrapper._route_warmstart_candidates(
                "model.latent_to_model_projs.0.weight",
                prefer_ema=True,
                source_modality_ids=["foa_latent", "spatial_traj"],
            )
            self.assertEqual(
                remapped[0][0],
                "transfusion_ema.ema_model.latent_to_model_projs.1.weight",
            )

    def test_route_warmstart_does_not_alias_new_modality_control_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            target = wrapper.state_dict()
            transformer_key = "model.transformer.norm.gamma"
            embedding = wrapper._soft_text_embed.fallback.weight
            new_spatial_som = wrapper.num_text_tokens + 3
            foa_som = new_spatial_som + 1
            spatial_row_before = embedding[new_spatial_som].detach().clone()

            source_embedding = torch.full_like(embedding, 7.0)
            source = {
                "transfusion_ema.ema_model.transformer.norm.gamma": torch.full_like(
                    target[transformer_key], 3.0
                ),
                "transfusion_ema.ema_model.text_embed.fallback.weight": source_embedding,
                "transfusion_ema.ema_model.to_text_logits.weight": source_embedding,
            }
            source_config = {
                "model": {
                    "text": {
                        "mode": "qwen_prefix",
                        "ar_target": {"type": "spatial_plan"},
                    },
                    "modalities": [
                        {"id": "source_tracks"},
                        {"id": "foa_latent"},
                    ],
                }
            }

            report = wrapper.load_pretrained_route_state_dict(
                source,
                source_model_config=source_config,
            )

            self.assertTrue(report["modality_contract_changed"])
            self.assertTrue(torch.equal(embedding[0], source_embedding[0]))
            self.assertTrue(torch.equal(embedding[foa_som], source_embedding[foa_som]))
            self.assertTrue(torch.equal(embedding[new_spatial_som], spatial_row_before))
            self.assertEqual(
                report["modality_mapping"],
                {"spatial_traj": None, "foa_latent": 1},
            )

    def test_time_condition_width_is_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))

            transformer = wrapper.model.transformer
            self.assertEqual(transformer.time_cond_dim, 16)
            self.assertEqual(transformer.to_time_cond[1].out_features, 16)
            self.assertEqual(
                transformer.layers[0][1].to_film.in_features,
                16,
            )

    def test_planner_generation_uses_training_time_condition_width(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            core = _RecordingPlannerCore(wrapper)
            trajectory = torch.zeros(4, 4)

            plan = wrapper.generate_plan(
                "a source in a room",
                core_model=core,
                context_modalities={"spatial_traj": trajectory},
                temperature=0.0,
            )

            self.assertTrue(core.times_shapes)
            self.assertEqual(
                set(core.times_shapes),
                {(1, wrapper.model.num_modalities)},
            )
            decoded = wrapper.get_plan_codec().decode(plan)
            self.assertAlmostEqual(decoded["audio"]["duration_sec"], 10.05)

    def test_planner_and_renderer_are_causally_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            codec = SpatialPlanCodec(codec_path)
            plan_ids = codec.encode(
                {
                    "audio": {"duration_sec": 1.0},
                    "mix": {"type": "single"},
                    "scene": {"room": {}, "sources": []},
                }
            )["input_ids"]

            planner_loss = wrapper.forward_planner(
                ["a source on the left", "a source on the right"],
                [plan_ids, plan_ids],
            )
            self.assertTrue(torch.isfinite(planner_loss))
            qwen_calls = wrapper.qwen_conditioner.calls
            self.assertEqual(qwen_calls, 1)
            self.assertIs(
                wrapper.model.to_text_logits.weight,
                wrapper._soft_text_embed.fallback.weight,
            )

            trajectory = torch.full((4, 4), 2.0)
            latent = torch.full((8, 4), 3.0)
            samples = [
                wrapper.build_renderer_sample(
                    plan_ids,
                    {"spatial_traj": trajectory, "foa_latent": latent},
                )
            ]
            projection_inputs = {}

            def capture(name):
                def hook(_module, args):
                    projection_inputs[name] = args[0].detach().clone()

                return hook

            handles = [
                wrapper.model.latent_to_model_projs[0].register_forward_pre_hook(
                    capture("trajectory")
                ),
                wrapper.model.latent_to_model_projs[1].register_forward_pre_hook(
                    capture("latent")
                ),
            ]
            try:
                _, breakdown = wrapper.forward_renderer(
                    samples,
                    # trajectory is pure noise; latent is an exact clean prefix.
                    times=torch.tensor([[0.0, 1.0]]),
                    return_breakdown=True,
                )
            finally:
                for handle in handles:
                    handle.remove()

            self.assertEqual(wrapper.qwen_conditioner.calls, qwen_calls)
            self.assertFalse(
                torch.equal(projection_inputs["trajectory"].squeeze(0), trajectory)
            )
            self.assertTrue(
                torch.equal(projection_inputs["latent"].squeeze(0), latent)
            )
            total = planner_loss + sum(breakdown.flow)
            total.backward()
            self.assertIsNotNone(wrapper.qwen_conditioner.bias.grad)

    def test_context_planner_reports_edit_delta_breakdown(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            wrapper = self._make_wrapper(str(codec_path))
            codec = SpatialPlanCodec(codec_path)
            plan = codec.encode(
                {
                    "audio": {"duration_sec": 1.0},
                    "mix": {"type": "single"},
                    "scene": {"room": {"type": "office"}, "sources": []},
                }
            )
            plan["edit_token_mask"] = torch.zeros_like(
                plan["input_ids"], dtype=torch.bool
            )
            plan["edit_token_mask"][3] = True
            loss, breakdown = wrapper.forward_planner(
                ["change the duration"],
                [plan],
                loss_group_weights={
                    "grammar": 0.2,
                    "semantic": 1.0,
                    "room": 1.0,
                },
                edit_delta_ce_weight=0.5,
                return_loss_breakdown=True,
                context_modalities=[{"spatial_traj": torch.zeros(4, 4)}],
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(breakdown["base_ce"]))
            self.assertTrue(torch.isfinite(breakdown["edit_ce"]))
            self.assertAlmostEqual(
                float(loss),
                float(breakdown["base_ce"] + 0.5 * breakdown["edit_ce"]),
                places=5,
            )

    def test_joint_coupled_sampling_matches_all_modality_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            codec_path = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "single", "office"] * 16,
                text_vocab_size=512,
            )
            config = copy.deepcopy(_tiny_config(str(codec_path)))
            config["route_id"] = "continuous_traj"
            config["model"]["text"].pop("ar_target")
            config["model"]["text"]["num_text_tokens"] = 256
            config["model"]["text"]["tie_text_embeddings"] = False
            config["model"]["transfusion"]["modality_time_policy"] = "shared_uniform"
            config["model"]["transfusion"]["sampling"] = {
                "mode": "joint_coupled",
                "modality_steps": 3,
                "cfg_scale": 1.0,
            }
            with patch.object(
                TransfusionSpatialWrapper,
                "_init_qwen_prefix",
                _fake_init_qwen,
            ):
                wrapper = TransfusionSpatialWrapper(config)

            sample = wrapper.generate("a moving source")
            modalities = {
                item[0]: item[1]
                for item in sample
                if isinstance(item, tuple)
            }
            self.assertEqual(tuple(modalities[0].shape), (4, 4))
            self.assertEqual(tuple(modalities[1].shape), (8, 4))
            self.assertTrue(torch.isfinite(modalities[0]).all())
            self.assertTrue(torch.isfinite(modalities[1]).all())
            with self.assertRaisesRegex(ValueError, "does not accept fixed"):
                wrapper.generate(
                    "a moving source",
                    provided_modalities={"spatial_traj": torch.zeros(4, 4)},
                )


if __name__ == "__main__":
    unittest.main()
