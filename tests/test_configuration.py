from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from stable_audio_tools.configuration import (
    ConfigError,
    load_config,
    validate_t2a_config,
)
from stable_audio_tools.training.factory import create_demo_callback_from_config


class DemoCallbackFactoryTests(unittest.TestCase):
    def test_demo_is_opt_in(self):
        base = {"model_type": "transfusion_spatial", "training": {}}
        self.assertIsNone(create_demo_callback_from_config(base))

        base["training"]["demo"] = {}
        self.assertIsNone(create_demo_callback_from_config(base))

        base["training"]["demo"] = {"enabled": False}
        self.assertIsNone(create_demo_callback_from_config(base))


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = (
    REPO_ROOT / "stable_audio_tools" / "configs" / "model_configs" / "txt2audio"
)
DATA_ROOT = (
    REPO_ROOT / "stable_audio_tools" / "configs" / "dataset_configs" / "vae_v2_dataset"
)


class ConfigLoadingTests(unittest.TestCase):
    def test_extends_deep_merges_dicts_and_replaces_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.json").write_text(
                json.dumps(
                    {
                        "model": {"dim": 512, "nested": {"a": 1, "b": 2}},
                        "values": [1, 2],
                    }
                ),
                encoding="utf-8",
            )
            (root / "child.json").write_text(
                json.dumps(
                    {
                        "extends": "base.json",
                        "model": {"nested": {"b": 3}},
                        "values": [9],
                    }
                ),
                encoding="utf-8",
            )
            resolved = load_config(root / "child.json")

        self.assertEqual(resolved["model"]["dim"], 512)
        self.assertEqual(resolved["model"]["nested"], {"a": 1, "b": 3})
        self.assertEqual(resolved["values"], [9])
        self.assertNotIn("extends", resolved)

    def test_extends_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.json").write_text('{"extends": "b.json"}', encoding="utf-8")
            (root / "b.json").write_text('{"extends": "a.json"}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "inheritance cycle"):
                load_config(root / "a.json")


class T2AConfigTests(unittest.TestCase):
    def test_all_canonical_t2a_configs_validate(self):
        dit_dataset = load_config(
            DATA_ROOT / "t2a_preencoded_v2_construct_expansion.json"
        )
        transfusion_dataset = load_config(
            DATA_ROOT / "t2a_preencoded_v2_transfusion.json"
        )
        spatial_chat_dataset = load_config(
            DATA_ROOT / "t2a_spatial_cot_1m_families.json"
        )
        cases = (
            (MODEL_ROOT / "t2a" / "dit" / "t5_1b.json", dit_dataset),
            (MODEL_ROOT / "t2a" / "dit" / "qwen35_0p8b_1b.json", dit_dataset),
            (MODEL_ROOT / "t2a" / "dit" / "qwen35_0p8b_300m.json", dit_dataset),
            (
                MODEL_ROOT
                / "t2a"
                / "transfusion"
                / "qwen35_0p8b_continuous_traj_300m.json",
                transfusion_dataset,
            ),
            (
                MODEL_ROOT
                / "t2a"
                / "transfusion"
                / "qwen35_0p8b_continuous_traj_v2.json",
                transfusion_dataset,
            ),
            (
                MODEL_ROOT
                / "t2a"
                / "spatial_cot"
                / "qwen35_0p8b_spatial_chat_500m.json",
                spatial_chat_dataset,
            ),
        )
        for model_path, dataset in cases:
            with self.subTest(model=model_path.name):
                summary = validate_t2a_config(load_config(model_path), dataset)
                self.assertEqual(summary["latent_channels"], 64)
                self.assertEqual(summary["latent_length"], 432)
                self.assertEqual(summary["audio_channels"], 4)

    def test_spatial_cot_dense_joint_base_is_exact_fixed10(self):
        model = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "bootstrap"
            / "qwen35_0p8b_spatial_chat_dense_joint_source_binding_base.json"
        )
        dataset = load_config(
            DATA_ROOT
            / "t2a_spatial_cot_source_binding_counterfactual_fixed48_v2.json"
        )
        summary = validate_t2a_config(model, dataset)
        window = model["fixed_audio_window"]
        self.assertEqual(model["sample_rate"], 44_100)
        self.assertEqual(model["sample_size"], 442_368)
        self.assertEqual(summary["latent_length"], 432)
        self.assertEqual(window["num_samples"], 442_368)
        self.assertEqual(window["latent_frames"], 432)
        self.assertTrue(
            model["model"]["transfusion"]["activity_identity_flow"][
                "enabled"
            ]
        )

        for key, value in (
            ("num_samples", 441_000),
            ("latent_frames", 431),
            ("downsampling_ratio", 1_000),
            ("sample_rate", 48_000),
        ):
            with self.subTest(key=key):
                candidate = copy.deepcopy(model)
                candidate["fixed_audio_window"][key] = value
                with self.assertRaisesRegex(ConfigError, "fixed_audio_window"):
                    validate_t2a_config(candidate, dataset)

    def test_spatial_cot_native_kv_lora_base_is_valid_fixed10(self):
        model = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "bootstrap"
            / "qwen35_0p8b_spatial_chat_dense_joint_source_binding_base.json"
        )
        dataset = load_config(
            DATA_ROOT
            / "t2a_spatial_cot_source_binding_counterfactual_fixed48_v2.json"
        )
        summary = validate_t2a_config(model, dataset)
        renderer = model["model"]["transfusion"][
            "dense_dit_residual_renderer"
        ]
        selection = model["training"]["parameter_selection"]
        self.assertEqual(summary["latent_length"], 432)
        self.assertEqual(renderer["kv_lora_context_dim"], 1024)
        self.assertEqual(renderer["kv_lora_rank"], 32)
        self.assertTrue(
            selection["include_source_dense_condition_kv_lora_adapter"]
        )

        invalid = copy.deepcopy(model)
        invalid["model"]["transfusion"]["dense_dit_residual_renderer"][
            "kv_lora_rank"
        ] = 2048
        with self.assertRaisesRegex(ConfigError, "K/V LoRA"):
            validate_t2a_config(invalid, dataset)

    def test_spatial_cot_dense_joint_contract_is_fail_closed(self):
        model = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "bootstrap"
            / "qwen35_0p8b_spatial_chat_dense_joint_source_binding_base.json"
        )
        dataset = load_config(
            DATA_ROOT
            / "t2a_spatial_cot_source_binding_counterfactual_fixed48_v2.json"
        )
        renderer = model["model"]["transfusion"][
            "dense_dit_residual_renderer"
        ]
        renderer.update(
            {
                "layer_start_index": 0,
                "layer_count": 15,
                "kv_lora_layer_start_index": 9,
                "kv_lora_layer_count": 6,
                "joint_late_blocks": {
                    "enabled": True,
                    "start_index": 0,
                    "count": 15,
                },
            }
        )
        model["training"]["parameter_selection"][
            "include_dense_joint_late_blocks"
        ] = True
        model["training"]["optimizer_configs"]["transfusion"][
            "parameter_group_lr_scales"
        ]["dense_joint"] = 10.0
        validate_t2a_config(model, dataset)

        removed_source_resolved = copy.deepcopy(model)
        removed_source_resolved["model"]["transfusion"][
            "dense_dit_residual_renderer"
        ]["source_resolved_vector_field"] = {
            "enabled": True,
            "composition": "scene_plus_source_deltas",
            "scene_prompt": "A spatial audio recording.",
            "source_prompt_prefix": "A spatial audio recording contains ",
        }
        with self.assertRaisesRegex(ConfigError, "removed route keys"):
            validate_t2a_config(removed_source_resolved, dataset)

        invalid_range = copy.deepcopy(model)
        invalid_range["model"]["transfusion"][
            "dense_dit_residual_renderer"
        ]["joint_late_blocks"]["count"] = 14
        with self.assertRaisesRegex(ConfigError, "non-empty suffix"):
            validate_t2a_config(invalid_range, dataset)

        invalid_selection = copy.deepcopy(model)
        invalid_selection["training"]["parameter_selection"][
            "include_dense_joint_late_blocks"
        ] = False
        with self.assertRaisesRegex(ConfigError, "explicitly"):
            validate_t2a_config(invalid_selection, dataset)

        invalid_scale = copy.deepcopy(model)
        invalid_scale["training"]["optimizer_configs"]["transfusion"][
            "parameter_group_lr_scales"
        ]["dense_joint"] = 0.0
        with self.assertRaisesRegex(ConfigError, "dense_joint.*LR scale"):
            validate_t2a_config(invalid_scale, dataset)

        singular_anchor = copy.deepcopy(model)
        pair = singular_anchor["training"]["renderer_counterfactual_pair_loss"]
        pair["anchor_times"] = [0.0, 1.0]
        pair["same_noised_state_at_anchor"] = True
        with self.assertRaisesRegex(ConfigError, "less than 1"):
            validate_t2a_config(singular_anchor, dataset)

        removed_composition = copy.deepcopy(model)
        removed_composition["model"]["transfusion"][
            "dense_dit_residual_renderer"
        ]["composition"] = "dense_with_source_binding_hybrid"
        with self.assertRaisesRegex(ConfigError, "retained source-binding"):
            validate_t2a_config(removed_composition, dataset)

        removed_renderer_key = copy.deepcopy(model)
        removed_renderer_key["model"]["transfusion"][
            "dense_dit_residual_renderer"
        ]["explicit_source_token_layer_count"] = 15
        with self.assertRaisesRegex(ConfigError, "removed route keys"):
            validate_t2a_config(removed_renderer_key, dataset)

        removed_selection_key = copy.deepcopy(model)
        removed_selection_key["training"]["parameter_selection"][
            "include_source_dense_explicit_token_adapter"
        ] = True
        with self.assertRaisesRegex(ConfigError, "removed or unknown"):
            validate_t2a_config(removed_selection_key, dataset)

    def test_transfusion_routes_are_explicit_and_config_driven(self):
        continuous = load_config(
            MODEL_ROOT
            / "t2a"
            / "transfusion"
            / "qwen35_0p8b_continuous_traj_300m.json"
        )
        spatial_cot = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "qwen35_0p8b_spatial_chat_500m.json"
        )
        self.assertEqual(continuous["route_id"], "continuous_traj")
        self.assertNotIn("objectives", continuous["training"])
        self.assertEqual(spatial_cot["route_id"], "spatial_cot")
        self.assertEqual(
            {item["type"] for item in spatial_cot["training"]["objectives"]},
            {"qwen_prefix_to_plan", "plan_to_modalities"},
        )
        self.assertTrue(spatial_cot["model"]["text"]["tie_text_embeddings"])
        self.assertEqual(
            continuous["model"]["transfusion"]["sampling"]["mode"],
            "joint_coupled",
        )
        self.assertEqual(
            spatial_cot["model"]["transfusion"]["sampling"]["mode"],
            "sequential",
        )

        self.assertEqual(spatial_cot["variant_id"], "spatial_chat_v1_500m")
        self.assertEqual(
            spatial_cot["model"]["transfusion"]["transformer"]["depth"], 24
        )
        self.assertTrue(
            spatial_cot["model"]["transfusion"]["transformer"]["use_flex_attn"]
        )
        self.assertEqual(
            [item["id"] for item in spatial_cot["training"]["objectives"]],
            ["state_planner", "understanding", "renderer"],
        )
        self.assertEqual(
            spatial_cot["training"]["diffusion_forcing_options"][
                "context_clean_probability"
            ],
            0.5,
        )
        self.assertEqual(
            spatial_cot["training"]["diffusion_forcing_options"][
                "target_time_sampling"
            ],
            "global_stratified",
        )
        validate_t2a_config(
            spatial_cot,
            load_config(DATA_ROOT / "t2a_spatial_cot_1m_families.json"),
        )

    def test_diffusion_forcing_options_fail_closed(self):
        model = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "qwen35_0p8b_spatial_chat_500m.json"
        )
        dataset = load_config(DATA_ROOT / "t2a_spatial_cot_1m_families.json")

        invalid_probability = copy.deepcopy(model)
        invalid_probability["training"]["diffusion_forcing_options"][
            "context_clean_probability"
        ] = 1.1
        with self.assertRaisesRegex(ConfigError, "context_clean_probability"):
            validate_t2a_config(invalid_probability, dataset)

        invalid_sampling = copy.deepcopy(model)
        invalid_sampling["training"]["diffusion_forcing_options"][
            "target_time_sampling"
        ] = "randomish"
        with self.assertRaisesRegex(ConfigError, "target_time_sampling"):
            validate_t2a_config(invalid_sampling, dataset)

        high_noise = copy.deepcopy(model)
        high_noise["training"]["diffusion_forcing_options"].update(
            {
                "target_time_sampling": "global_stratified_logit_normal",
                "target_time_logit_mean": -0.77,
                "target_time_logit_std": 1.0,
                "target_time_uniform_mix": 0.1,
            }
        )
        validate_t2a_config(high_noise, dataset)

        invalid_logit_std = copy.deepcopy(high_noise)
        invalid_logit_std["training"]["diffusion_forcing_options"][
            "target_time_logit_std"
        ] = 0.0
        with self.assertRaisesRegex(ConfigError, "target_time_logit_std"):
            validate_t2a_config(invalid_logit_std, dataset)

        invalid_uniform_mix = copy.deepcopy(high_noise)
        invalid_uniform_mix["training"]["diffusion_forcing_options"][
            "target_time_uniform_mix"
        ] = 1.0
        with self.assertRaisesRegex(ConfigError, "target_time_uniform_mix"):
            validate_t2a_config(invalid_uniform_mix, dataset)

        misplaced_logit_options = copy.deepcopy(model)
        misplaced_logit_options["training"]["diffusion_forcing_options"][
            "target_time_logit_mean"
        ] = -0.77
        with self.assertRaisesRegex(ConfigError, "require"):
            validate_t2a_config(misplaced_logit_options, dataset)

    def test_source_semantic_fusion_contract_is_fail_closed(self):
        model = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "bootstrap"
            / "qwen35_0p8b_spatial_chat_dense_joint_source_binding_base.json"
        )
        dataset = load_config(
            DATA_ROOT
            / "t2a_spatial_cot_source_binding_counterfactual_fixed48_v2.json"
        )
        validate_t2a_config(model, dataset)
        fusion = model["model"]["transfusion"][
            "source_region_target_residual"
        ]["semantic_fusion"]
        self.assertEqual(fusion["rank"], 64)
        self.assertFalse(
            model["training"]["parameter_selection"][
                "include_qwen_projection"
            ]
        )

        invalid_rank = copy.deepcopy(model)
        invalid_rank["model"]["transfusion"][
            "source_region_target_residual"
        ]["semantic_fusion"]["rank"] = 0
        with self.assertRaisesRegex(ConfigError, "rank must be a positive"):
            validate_t2a_config(invalid_rank, dataset)

        missing_regions = copy.deepcopy(model)
        missing_regions["model"]["text"]["source_region_embedding"] = False
        with self.assertRaisesRegex(ConfigError, "source_region_embedding"):
            validate_t2a_config(missing_regions, dataset)

        invalid_summary_mode = copy.deepcopy(model)
        invalid_summary_mode["model"]["text"]["source_summary_mode"] = "mixed"
        with self.assertRaisesRegex(ConfigError, "source_summary_mode"):
            validate_t2a_config(invalid_summary_mode, dataset)

        invalid_aggregation = copy.deepcopy(model)
        invalid_aggregation["model"]["transfusion"][
            "source_region_target_residual"
        ]["aggregation"] = "slot_classifier"
        with self.assertRaisesRegex(ConfigError, "aggregation"):
            validate_t2a_config(invalid_aggregation, dataset)

        token_local = copy.deepcopy(model)
        token_local["model"]["text"]["source_summary_mode"] = "token_local"
        validate_t2a_config(token_local, dataset)

        token_local_without_fusion = copy.deepcopy(token_local)
        token_local_without_fusion["model"]["transfusion"][
            "source_region_target_residual"
        ]["semantic_fusion"]["enabled"] = False
        with self.assertRaisesRegex(ConfigError, "requires source semantic fusion"):
            validate_t2a_config(token_local_without_fusion, dataset)

    def test_transfusion_rejects_train_sampling_time_mismatch(self):
        continuous = load_config(
            MODEL_ROOT
            / "t2a"
            / "transfusion"
            / "qwen35_0p8b_continuous_traj_300m.json"
        )
        continuous = copy.deepcopy(continuous)
        continuous["model"]["transfusion"]["sampling"]["mode"] = "sequential"
        with self.assertRaisesRegex(ConfigError, "clean trajectory prefix"):
            validate_t2a_config(continuous)

        spatial_cot = load_config(
            MODEL_ROOT
            / "t2a"
            / "spatial_cot"
            / "qwen35_0p8b_spatial_chat_500m.json"
        )
        spatial_cot = copy.deepcopy(spatial_cot)
        spatial_cot["model"]["transfusion"]["sampling"]["mode"] = "joint_coupled"
        spatial_cot["model"]["transfusion"]["modality_time_policy"] = "shared_uniform"
        with self.assertRaisesRegex(ConfigError, "requires sampling.mode=sequential"):
            validate_t2a_config(spatial_cot)

    def test_transfusion_v2_requires_its_causal_training_contract(self):
        v2 = load_config(
            MODEL_ROOT
            / "t2a"
            / "transfusion"
            / "qwen35_0p8b_continuous_traj_v2.json"
        )
        self.assertEqual(v2["route_id"], "continuous_traj")
        self.assertEqual(v2["variant_id"], "causal_flow_v2")
        self.assertEqual(
            v2["model"]["transfusion"]["transformer"]["time_cond_dim"],
            1024,
        )

        missing_schedule = copy.deepcopy(v2)
        missing_schedule["training"].pop("flow_schedule")
        with self.assertRaisesRegex(ConfigError, "requires training.flow_schedule"):
            validate_t2a_config(missing_schedule)

        inconsistent_teacher = copy.deepcopy(v2)
        inconsistent_teacher["training"]["use_velocity_consistency"] = True
        with self.assertRaisesRegex(ConfigError, "requires use_velocity_consistency=false"):
            validate_t2a_config(inconsistent_teacher)

    def test_latent_crop_mismatch_is_rejected(self):
        model = load_config(MODEL_ROOT / "t2a" / "dit" / "qwen35_0p8b_300m.json")
        dataset = load_config(
            DATA_ROOT / "t2a_preencoded_v2_construct_expansion.json"
        )
        dataset = copy.deepcopy(dataset)
        dataset["latent_crop_length"] = 431
        with self.assertRaisesRegex(ConfigError, "latent_crop_length"):
            validate_t2a_config(model, dataset)

    def test_incompatible_cross_attention_head_counts_are_rejected(self):
        model = load_config(
            MODEL_ROOT / "t2a" / "dit" / "qwen35_0p8b_300m.json"
        )
        model = copy.deepcopy(model)
        model["model"]["diffusion"]["config"]["project_cond_tokens"] = False
        with self.assertRaisesRegex(ConfigError, "query heads"):
            validate_t2a_config(model)

    def test_transfusion_requires_trajectory_provider(self):
        model = load_config(
            MODEL_ROOT
            / "t2a"
            / "transfusion"
            / "qwen35_0p8b_continuous_traj_300m.json"
        )
        dataset = load_config(
            DATA_ROOT / "t2a_preencoded_v2_construct_expansion.json"
        )
        with self.assertRaisesRegex(ConfigError, "custom metadata provider"):
            validate_t2a_config(model, dataset)


if __name__ == "__main__":
    unittest.main()
