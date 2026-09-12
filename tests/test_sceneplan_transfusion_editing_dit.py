from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from stable_audio_tools.configuration import load_config, validate_t2a_config
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (
    make_editing_dit_cfg_unknown_metadata,
    verify_editing_source_latent_shards,
    verify_editing_target_latent_shards,
)
from stable_audio_tools.models.conditioners import MultiConditioner
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
EDITING_CONFIG = (
    REPO_ROOT
    / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    / "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_v1.json"
)
EDITING_FULL_CONFIG = (
    REPO_ROOT
    / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    / "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json"
)
EDITING_DATASET_CONFIG = (
    REPO_ROOT
    / "stable_audio_tools/configs/dataset_configs/"
    / "sceneplan_transfusion_editing_v1_pilot_validation.json"
)


class _ExpandedEditingCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Target input is [noisy target:2, plan:3, source reference:2].
        self.input_concat_dim = 5
        self.preprocess_conv = nn.Conv1d(7, 7, 1, bias=False)
        self.transformer = nn.Module()
        self.transformer.project_in = nn.Linear(7, 11, bias=False)
        self.transformer.shared = nn.Linear(11, 11, bias=False)


class _Route(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ExpandedEditingCore()


class EditingDiTContractTests(unittest.TestCase):
    def test_source_shard_audit_hashes_live_external_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "source.safetensors"
            shard.write_bytes(b"frozen-source-latents")
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            target = root / "target.safetensors"
            target.write_bytes(b"frozen-target-latents")
            target_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            index = root / "editing.sqlite"
            connection = sqlite3.connect(index)
            try:
                connection.execute(
                    "CREATE TABLE pairs("
                    "source_latent_path TEXT,"
                    "source_latent_shard_sha256 TEXT,"
                    "target_latent_path TEXT,"
                    "target_latent_shard_sha256 TEXT)"
                )
                connection.executemany(
                    "INSERT INTO pairs VALUES(?,?,?,?)",
                    [
                        (str(shard), digest, str(target), target_digest),
                        (str(shard), digest, str(target), target_digest),
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            audit = verify_editing_source_latent_shards(index)
            self.assertEqual(audit["source_latent_shards"], 1)
            self.assertEqual(audit["source_pair_rows"], 2)
            self.assertTrue(
                audit["source_latent_shards_exhaustively_verified"]
            )
            self.assertEqual(
                len(audit["source_latent_shard_inventory_sha256"]), 64
            )
            target_audit = verify_editing_target_latent_shards(index)
            self.assertEqual(target_audit["target_latent_shards"], 1)
            self.assertEqual(target_audit["target_pair_rows"], 2)
            self.assertTrue(
                target_audit["target_latent_shards_exhaustively_verified"]
            )

            shard.write_bytes(b"mutated-source-latents")
            with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
                verify_editing_source_latent_shards(index)
            target.write_bytes(b"mutated-target-latents")
            with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
                verify_editing_target_latent_shards(index)

    def test_full_config_preserves_384_channel_route(self):
        model = load_config(EDITING_FULL_CONFIG)
        diffusion = model["model"]["diffusion"]
        self.assertEqual(
            diffusion["input_concat_ids"],
            ["sceneplan_44", "source_foa_latent"],
        )
        self.assertEqual(diffusion["config"]["input_concat_dim"], 320)
        self.assertEqual(
            model["training"]["optimizer_configs"]["diffusion"]["optimizer"][
                "config"
            ]["lr"],
            5e-5,
        )

    def test_resolved_config_is_exact_384_channel_p10_suffix_expansion(self):
        model = load_config(EDITING_CONFIG)
        dataset = load_config(EDITING_DATASET_CONFIG)
        validate_t2a_config(model, dataset)

        diffusion = model["model"]["diffusion"]
        self.assertEqual(
            diffusion["input_concat_ids"],
            ["sceneplan_44", "source_foa_latent"],
        )
        self.assertEqual(diffusion["config"]["io_channels"], 64)
        self.assertEqual(diffusion["config"]["input_concat_dim"], 320)
        self.assertEqual(
            64 + diffusion["config"]["input_concat_dim"], 384
        )
        self.assertEqual(diffusion["config"]["depth"], 15)
        self.assertEqual(diffusion["config"]["embed_dim"], 1024)
        self.assertEqual(diffusion["config"]["num_heads"], 16)
        self.assertTrue(
            diffusion["config"]["require_explicit_negative_input_concat"]
        )
        self.assertEqual(
            model["training"]["sceneplan_speech_active_loss_weight"], 1.0
        )
        self.assertFalse(
            model["training"]["sceneplan_sound_temporal_difference_loss"][
                "enabled"
            ]
        )

    def test_cfg_unknowns_new_plan_but_keeps_exact_source_tensor(self):
        source = torch.randn(64, 9)
        positive = {
            "prompt": {"input_ids": torch.ones(3, dtype=torch.long)},
            "sceneplan_44": {
                "source_event_frame_ids": torch.zeros(4, 9, dtype=torch.int8)
            },
            "source_foa_latent": source,
        }
        negative = make_editing_dit_cfg_unknown_metadata(positive)
        self.assertTrue(negative["prompt"]["cfg_unknown"])
        self.assertTrue(negative["sceneplan_44"]["cfg_unknown"])
        self.assertIs(negative["source_foa_latent"], source)

    def test_direct_preencoded_source_latent_needs_no_dummy_conditioner(self):
        conditioner = MultiConditioner(
            {}, pre_encoded_keys=["source_foa_latent"]
        )
        source_0 = torch.randn(64, 9)
        source_1 = torch.randn(64, 9)
        output = conditioner(
            [
                {"source_foa_latent": source_0},
                {"source_foa_latent": [source_1]},
            ],
            "cpu",
        )
        self.assertEqual(tuple(output["source_foa_latent"][0].shape), (2, 64, 9))
        torch.testing.assert_close(
            output["source_foa_latent"][0][0], source_0
        )
        torch.testing.assert_close(
            output["source_foa_latent"][0][1], source_1
        )

    def test_condition_order_is_plan_then_aligned_source_reference(self):
        wrapper = ConditionedDiffusionModelWrapper.__new__(
            ConditionedDiffusionModelWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.cross_attn_cond_ids = []
        wrapper.global_cond_ids = []
        wrapper.input_concat_ids = ["sceneplan_44", "source_foa_latent"]
        wrapper.local_add_cond_ids = []
        wrapper.modular_local_cond_ids = []
        wrapper.prepend_cond_ids = []
        wrapper.gate = False

        plan = torch.full((2, 3, 7), 3.0)
        source = torch.full((2, 2, 7), 7.0)
        inputs = wrapper.get_conditioning_inputs(
            {
                "sceneplan_44": [plan, None],
                "source_foa_latent": [source, None],
            }
        )
        concatenated = inputs["input_concat_cond"]
        torch.testing.assert_close(concatenated[:, :3], plan)
        torch.testing.assert_close(concatenated[:, 3:], source)

    def test_p10_320_prefix_is_copied_and_only_source_suffix_is_zero(self):
        wrapper = ConditionedDiffusionModelWrapper.__new__(
            ConditionedDiffusionModelWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.model = _Route()
        wrapper.conditioner = nn.Module()
        wrapper.io_channels = 2
        wrapper.diffusion_objective = "rectified_flow"

        # Source P10 analogue is [noisy target:2, plan:3] = five columns.
        source_project = torch.arange(55, dtype=torch.float32).reshape(11, 5)
        source_preprocess = torch.arange(
            25, dtype=torch.float32
        ).reshape(5, 5, 1)
        source_shared = torch.full((11, 11), 0.125)
        state = {
            "diffusion_ema.ema_model.model.transformer.project_in.weight": (
                source_project
            ),
            "diffusion_ema.ema_model.model.preprocess_conv.weight": (
                source_preprocess
            ),
            "diffusion_ema.ema_model.model.transformer.shared.weight": (
                source_shared
            ),
        }
        source_config = {
            "model": {
                "io_channels": 2,
                "diffusion": {
                    "diffusion_objective": "rectified_flow",
                    "config": {"io_channels": 2, "input_concat_dim": 3},
                },
            }
        }
        report = wrapper.load_pretrained_route_state_dict(
            state, source_model_config=source_config, prefer_ema=True
        )

        project = wrapper.model.model.transformer.project_in.weight
        preprocess = wrapper.model.model.preprocess_conv.weight
        torch.testing.assert_close(project[:, :5], source_project)
        torch.testing.assert_close(project[:, 5:], torch.zeros_like(project[:, 5:]))
        expected_preprocess = torch.zeros_like(preprocess)
        expected_preprocess[:5, :5] = source_preprocess
        torch.testing.assert_close(preprocess, expected_preprocess)
        torch.testing.assert_close(
            wrapper.model.model.transformer.shared.weight, source_shared
        )
        self.assertEqual(report["missing"], [])
        self.assertEqual(len(report["partial_expansions"]), 2)
        self.assertEqual(
            report["modality_mapping"],
            "exact_name_and_shape_plus_trained_prefix_input_expansion",
        )
        for expansion in report["partial_expansions"]:
            self.assertEqual(expansion["copied_prefix_channels"], 5)
            self.assertEqual(expansion["zero_initialized_suffix_channels"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
