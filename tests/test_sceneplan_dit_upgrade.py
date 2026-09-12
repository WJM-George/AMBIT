from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
from torch import nn

from stable_audio_tools.configuration import load_config, validate_t2a_config
from stable_audio_tools.models.dit import DiffusionTransformer
from stable_audio_tools.models.diffusion import (
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
)
from stable_audio_tools.models.sceneplan_alignment import (
    ScenePlanSoftBlockAlignment,
)
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
DIT_CONFIG_ROOT = (
    REPO_ROOT
    / "stable_audio_tools"
    / "configs"
    / "model_configs"
    / "txt2audio"
    / "t2a"
    / "dit"
)
SCENEPLAN_DATASET = (
    REPO_ROOT
    / "stable_audio_tools"
    / "configs"
    / "dataset_configs"
    / "sceneplan_v2_speech_expansion_noalign_15s_v1_train_semantic_v2.json"
)


class ScenePlanSoftBlockMaskTests(unittest.TestCase):
    def test_only_positive_matching_source_roles_receive_local_bias(self):
        builder = ScenePlanSoftBlockAlignment(max_sources=4)
        embeddings = torch.randn(2, 6, 8)
        token_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        token_aux = {
            # Role 0 is global-only; role -1 is CFG-unknown.  Neither is local.
            "event_source_ids": torch.tensor(
                [[0, 1, 2, -1, 0, 4], [-1, 0, 0, 0, 0, 0]]
            ),
            "speech_source_ids": torch.tensor(
                [[0, 0, 0, -1, 3, 0], [-1, 0, 0, 0, 0, 0]]
            ),
        }
        frame_ids = torch.zeros(2, 4, 4, dtype=torch.long)
        frame_ids[0, 0, 0] = 1
        frame_ids[0, 1, 1] = 2
        frame_ids[0, 2, 1] = 3
        frame_ids[0, 3, 2] = 4
        # The unknown CFG branch carries -1 on every valid source slot.
        frame_ids[1, :, :3] = -1
        frame_valid = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool
        )

        event_mask, speech_mask, metrics = builder(
            embeddings,
            token_mask,
            token_aux,
            {
                "source_event_frame_ids": frame_ids,
                "frame_valid_mask": frame_valid,
            },
            query_frames=4,
        )

        expected_event = torch.zeros(2, 1, 4, 6, dtype=torch.bool)
        expected_event[0, 0, 0, 1] = True
        expected_event[0, 0, 1, 2] = True
        expected_event[0, 0, 2, 5] = True
        expected_speech = torch.zeros_like(expected_event)
        expected_speech[0, 0, 1, 4] = True
        self.assertTrue(torch.equal(event_mask, expected_event))
        self.assertTrue(torch.equal(speech_mask, expected_speech))
        self.assertEqual(int(event_mask[1].count_nonzero()), 0)
        self.assertEqual(int(speech_mask[1].count_nonzero()), 0)
        self.assertTrue(torch.isfinite(metrics["active_frame_fraction"]))


class ScenePlanDiTUpgradeTests(unittest.TestCase):
    @staticmethod
    def _base_args() -> dict:
        return {
            "io_channels": 2,
            "input_concat_dim": 4,
            "embed_dim": 8,
            "cond_token_dim": 8,
            "project_cond_tokens": False,
            "depth": 3,
            "num_heads": 2,
            "timestep_features_dim": 8,
            "diffusion_objective": "rectified_flow",
            "activation_checkpointing": False,
            "zero_init_branch_outputs": False,
            # Tiny head dimensions are below the production RoPE minimum.
            "rotary_pos_emb": False,
        }

    @classmethod
    def _model(cls, *, soft_block: bool, chunk_moe: bool):
        arguments = cls._base_args()
        if soft_block:
            arguments["sceneplan_soft_block_attention"] = {
                "enabled": True,
                "layer_start": 0,
                "layer_count": 3,
                "max_bias": 4.0,
                "max_sources": 4,
            }
        if chunk_moe:
            arguments["sceneplan_chunk_moe"] = {
                "enabled": True,
                "layer_start": 2,
                "layer_count": 1,
                "num_experts": 2,
                "top_k": 1,
                "chunk_size": 2,
                "expert_inner_dim": 8,
                "router_hidden_dim": 8,
                "max_sources": 4,
                "boundary_aware": True,
                "sparse_scale": 0.5,
                "load_balance_loss_weight": 0.01,
            }
        return DiffusionTransformer(**arguments)

    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch, frames, tokens = 2, 7, 6
        x = torch.randn(batch, 2, frames)
        timestep = torch.tensor([0.2, 0.8])
        condition = torch.randn(batch, tokens, 8)
        condition_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        input_concat = torch.randn(batch, 4, frames)
        event_ids = torch.tensor(
            [[0, 1, 2, 0, -1, 0], [0, 3, 0, 4, 0, 0]]
        )
        speech_ids = torch.tensor(
            [[0, 0, 0, 1, -1, 0], [0, 0, 3, 0, 0, 0]]
        )
        frame_ids = torch.zeros(batch, 4, frames, dtype=torch.long)
        frame_ids[0, 0, :3] = 1
        frame_ids[0, 1, 3:6] = 2
        frame_ids[1, 2, :4] = 3
        frame_ids[1, 3, 4:] = 4
        frame_valid = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        frame_ids *= frame_valid[:, None]
        kwargs = {
            "cross_attn_cond": condition,
            "cross_attn_cond_mask": condition_mask,
            "cross_attn_aux": {
                "event_source_ids": event_ids,
                "speech_source_ids": speech_ids,
            },
            "input_concat_cond": input_concat,
            "input_concat_aux": {
                "source_event_frame_ids": frame_ids,
                "frame_valid_mask": frame_valid,
            },
            "padding_mask": frame_valid,
        }
        return x, timestep, kwargs

    def setUp(self):
        torch.manual_seed(7)

    def test_all_ablation_arms_are_bit_exact_at_upcycle_step_zero(self):
        dense = self._model(soft_block=False, chunk_moe=False).eval()
        x, timestep, kwargs = self._inputs()
        with torch.no_grad():
            expected = dense(
                x,
                timestep,
                cross_attn_cond=kwargs["cross_attn_cond"],
                cross_attn_cond_mask=kwargs["cross_attn_cond_mask"],
                input_concat_cond=kwargs["input_concat_cond"],
                padding_mask=kwargs["padding_mask"],
            )

        for soft_block, chunk_moe in ((True, False), (False, True), (True, True)):
            with self.subTest(soft_block=soft_block, chunk_moe=chunk_moe):
                upgraded = self._model(
                    soft_block=soft_block, chunk_moe=chunk_moe
                ).eval()
                incompatible = upgraded.load_state_dict(
                    dense.state_dict(), strict=False
                )
                self.assertFalse(incompatible.unexpected_keys)
                self.assertTrue(incompatible.missing_keys)
                self.assertTrue(
                    all(
                        "sceneplan_" in name
                        for name in incompatible.missing_keys
                    )
                )
                with torch.no_grad():
                    actual = upgraded(x, timestep, **kwargs)
                self.assertTrue(torch.equal(actual, expected))

    def test_combined_route_backpropagates_and_reports_exact_chunks(self):
        dense = self._model(soft_block=False, chunk_moe=False)
        upgraded = self._model(soft_block=True, chunk_moe=True)
        upgraded.load_state_dict(dense.state_dict(), strict=False)
        x, timestep, kwargs = self._inputs()

        output, info = upgraded(
            x, timestep, return_moe_info=True, **kwargs
        )
        gate_metrics = upgraded.sceneplan_soft_block_gate_metrics()
        self.assertEqual(float(gate_metrics["active_layers"]), 3.0)
        self.assertEqual(float(gate_metrics["bias_abs_max"]), 0.0)
        loss = output.square().mean() + info["auxiliary_loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(info["auxiliary_loss"]))
        # Batch 0: (2,1) + (2,1); batch 1: (2,2) + (1).
        self.assertEqual(float(info["chunk_count"]), 7.0)
        self.assertEqual(float(info["routed_chunk_evaluations"]), 7.0)
        self.assertEqual(float(info["active_layers"]), 1.0)
        self.assertEqual(tuple(info["expert_dispatch"].shape), (2,))
        self.assertAlmostEqual(
            float(info["expert_dispatch"].sum()), 1.0, places=6
        )
        for name in (
            "router_max_probability",
            "top1_weight_mean",
            "conflict_gate_saturation_fraction",
            "conflict_logit_l2",
            "prior_evidence_top1_agreement",
        ):
            self.assertIn(name, info)
            self.assertTrue(torch.isfinite(info[name]))

        event_gate_grad = sum(
            float(layer.sceneplan_event_bias_gate.grad.abs())
            for layer in upgraded.transformer.layers
        )
        speech_gate_grad = sum(
            float(layer.sceneplan_speech_bias_gate.grad.abs())
            for layer in upgraded.transformer.layers
        )
        expert_output_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in upgraded.named_parameters()
            if "sceneplan_moe.experts" in name
            and ".out_proj." in name
            and parameter.grad is not None
        )
        router_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in upgraded.named_parameters()
            if "sceneplan_moe" in name
            and any(
                component in name
                for component in (
                    "prior_router",
                    "evidence_router",
                    "prior_alignment",
                    "conflict_gate",
                )
            )
            and parameter.grad is not None
        )
        self.assertGreater(event_gate_grad, 0.0)
        self.assertGreater(speech_gate_grad, 0.0)
        self.assertGreater(expert_output_grad, 0.0)
        self.assertGreater(router_grad, 0.0)

    def test_cfg_requires_and_accepts_explicit_unknown_source_roles(self):
        model = self._model(soft_block=True, chunk_moe=True).eval()
        x, timestep, kwargs = self._inputs()
        negative_condition = torch.randn(x.shape[0], 1, 8)
        negative_mask = torch.ones(x.shape[0], 1, dtype=torch.bool)
        negative_token_aux = {
            "event_source_ids": -torch.ones(x.shape[0], 1, dtype=torch.long),
            "speech_source_ids": -torch.ones(x.shape[0], 1, dtype=torch.long),
        }
        valid = kwargs["padding_mask"]
        negative_frame_ids = torch.where(
            valid[:, None],
            -torch.ones(x.shape[0], 4, x.shape[-1], dtype=torch.long),
            torch.zeros(x.shape[0], 4, x.shape[-1], dtype=torch.long),
        )
        negative_frame_aux = {
            "source_event_frame_ids": negative_frame_ids,
            "frame_valid_mask": valid.clone(),
        }

        with self.assertRaisesRegex(ValueError, "explicit negative caption"):
            model(x, timestep, cfg_scale=2.0, **kwargs)

        with torch.no_grad():
            output = model(
                x,
                timestep,
                cfg_scale=2.0,
                negative_cross_attn_cond=negative_condition,
                negative_cross_attn_mask=negative_mask,
                negative_cross_attn_aux=negative_token_aux,
                negative_input_concat_cond=torch.zeros_like(
                    kwargs["input_concat_cond"]
                ),
                negative_input_concat_aux=negative_frame_aux,
                **kwargs,
            )
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(output).all())

    def test_production_router_auxiliaries_are_not_detached(self):
        arguments = self._base_args()
        arguments["sceneplan_chunk_moe"] = {
            "enabled": True,
            "layer_start": 2,
            "layer_count": 1,
            "num_experts": 2,
            "top_k": 1,
            "chunk_size": 2,
            "expert_inner_dim": 8,
            "router_hidden_dim": 8,
            "max_sources": 4,
            "boundary_aware": True,
            "sparse_scale": 0.5,
            "load_balance_loss_weight": 0.0,
            "router_entropy_loss_weight": 1.0,
            "router_entropy_target": 0.5,
            "conflict_logit_l2_loss_weight": 1.0,
            "conflict_gate_min": 0.25,
            "conflict_gate_max": 0.75,
        }
        model = DiffusionTransformer(**arguments)
        x, timestep, kwargs = self._inputs()
        _, info = model(x, timestep, return_moe_info=True, **kwargs)
        self.assertTrue(info["auxiliary_loss"].requires_grad)
        self.assertTrue(torch.isfinite(info["router_entropy_loss"]))
        info["auxiliary_loss"].backward()
        router_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if "sceneplan_moe" in name
            and any(
                component in name
                for component in ("prior_router", "evidence_router")
            )
            and parameter.grad is not None
        )
        conflict_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if "sceneplan_moe.conflict_gate" in name
            and parameter.grad is not None
        )
        self.assertGreater(router_gradient, 0.0)
        self.assertGreater(conflict_gradient, 0.0)

    def test_training_wrapper_discovers_the_nested_dit_upgrade(self):
        arguments = self._base_args()
        arguments.pop("diffusion_objective")
        arguments["sceneplan_soft_block_attention"] = {
            "enabled": True,
            "layer_start": 0,
            "layer_count": 3,
            "max_bias": 4.0,
            "max_sources": 4,
        }
        arguments["sceneplan_chunk_moe"] = {
            "enabled": True,
            "layer_start": 2,
            "layer_count": 1,
            "num_experts": 2,
            "top_k": 1,
            "chunk_size": 2,
            "expert_inner_dim": 8,
            "router_hidden_dim": 8,
            "max_sources": 4,
        }
        route = DiTWrapper(diffusion_objective="rectified_flow", **arguments)
        model = ConditionedDiffusionModelWrapper(
            route,
            nn.Identity(),
            io_channels=2,
            sample_rate=44_100,
            min_input_length=1,
            diffusion_objective="rectified_flow",
            cross_attn_cond_ids=["prompt"],
            input_concat_ids=["sceneplan_44"],
        )
        training = DiffusionCondTrainingWrapper(
            model,
            lr=1.0e-4,
            use_ema=False,
            pre_encoded=True,
            cfg_dropout_prob=0.0,
            sceneplan_cfg_dropout={
                "mode": "independent",
                "caption_unknown_prob": 0.15,
                "structured_unknown_prob": 0.15,
            },
        )
        self.assertTrue(training.sceneplan_chunk_moe_enabled)
        self.assertTrue(training.sceneplan_soft_block_enabled)
        self.assertIs(training._sceneplan_dit_model(), route.model)

    def test_model_only_ema_warmstart_leaves_only_upgrade_parameters_new(self):
        dense_arguments = self._base_args()
        dense_arguments.pop("diffusion_objective")
        upgraded_arguments = copy.deepcopy(dense_arguments)
        upgraded_arguments["sceneplan_soft_block_attention"] = {
            "enabled": True,
            "layer_start": 0,
            "layer_count": 3,
            "max_sources": 4,
        }
        upgraded_arguments["sceneplan_chunk_moe"] = {
            "enabled": True,
            "layer_start": 2,
            "layer_count": 1,
            "num_experts": 2,
            "top_k": 1,
            "chunk_size": 2,
            "expert_inner_dim": 8,
            "router_hidden_dim": 8,
            "max_sources": 4,
        }

        def wrap(route):
            return ConditionedDiffusionModelWrapper(
                route,
                nn.Identity(),
                io_channels=2,
                sample_rate=44_100,
                min_input_length=1,
                diffusion_objective="rectified_flow",
                cross_attn_cond_ids=["prompt"],
                input_concat_ids=["sceneplan_44"],
            )

        source = wrap(
            DiTWrapper(
                diffusion_objective="rectified_flow", **dense_arguments
            )
        )
        destination = wrap(
            DiTWrapper(
                diffusion_objective="rectified_flow", **upgraded_arguments
            )
        )
        ema_state = {
            "diffusion_ema.ema_model." + name[len("model.") :]: value.clone()
            for name, value in source.state_dict().items()
            if name.startswith("model.")
        }
        report = destination.load_pretrained_route_state_dict(
            ema_state, prefer_ema=True
        )
        self.assertGreater(report["loaded_from"]["dit_ema"], 0)
        self.assertFalse(report["shape_mismatches"])
        self.assertTrue(report["missing"])
        self.assertTrue(
            all("sceneplan_" in name for name in report["missing"])
        )
        self.assertTrue(
            torch.equal(
                source.model.model.transformer.layers[0].ff.ff[2].weight,
                destination.model.model.transformer.layers[0].ff.ff[2].weight,
            )
        )


class ScenePlanDiTUpgradeConfigTests(unittest.TestCase):
    def test_all_four_upgrade_configs_resolve_against_canonical_p10_data(self):
        dataset = load_config(SCENEPLAN_DATASET)
        names = (
            "qwen35_0p8b_300m_model_sceneplan_44_upgrade_v12_base.json",
            "qwen35_0p8b_300m_model_sceneplan_44_softblock_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_softblock_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv2_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv2_softblock_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv3_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv3_softblock_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv4_v12.json",
            "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv4_softblock_v12.json",
        )
        for name in names:
            with self.subTest(name=name):
                model = load_config(DIT_CONFIG_ROOT / name)
                summary = validate_t2a_config(model, dataset)
                self.assertEqual(summary["latent_channels"], 64)
                self.assertEqual(summary["latent_length"], 648)
                self.assertEqual(summary["audio_channels"], 4)
                self.assertEqual(
                    model["model"]["diffusion"]["config"]["depth"], 15
                )

    def test_production_combined_config_builds_the_expected_meta_model(self):
        config = load_config(
            DIT_CONFIG_ROOT
            / "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_routerv4_softblock_v12.json"
        )
        diffusion = config["model"]["diffusion"]
        with torch.device("meta"):
            model = DiffusionTransformer(
                diffusion_objective=diffusion["diffusion_objective"],
                **diffusion["config"],
            )
        moe_layers = [
            index
            for index, layer in enumerate(model.transformer.layers)
            if layer.sceneplan_moe is not None
        ]
        soft_block_layers = [
            index
            for index, layer in enumerate(model.transformer.layers)
            if layer.sceneplan_event_bias_gate is not None
        ]
        moe_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "sceneplan_moe" in name
        )
        self.assertEqual(moe_layers, [11, 12, 13, 14])
        self.assertEqual(soft_block_layers, list(range(15)))
        self.assertEqual(moe_parameters, 73_535_524)
        self.assertEqual(
            tuple(
                model.transformer.layers[11].ff.ff[0].proj.weight.shape
            ),
            (8192, 1024),
        )
        self.assertEqual(
            tuple(model.transformer.layers[11].ff.ff[2].weight.shape),
            (1024, 4096),
        )
        moe = model.transformer.layers[11].sceneplan_moe
        self.assertAlmostEqual(moe.router_temperature, 0.7)
        self.assertAlmostEqual(moe.conflict_gate_min, 0.25)
        self.assertAlmostEqual(moe.conflict_gate_max, 0.75)
        self.assertAlmostEqual(
            model.transformer.sceneplan_chunk_moe_load_balance_weight,
            0.01,
        )
        self.assertAlmostEqual(
            model.transformer.sceneplan_chunk_moe_router_entropy_weight,
            0.02,
        )
        self.assertAlmostEqual(
            model.transformer.sceneplan_chunk_moe_router_entropy_target,
            0.9,
        )
        self.assertAlmostEqual(
            model.transformer.sceneplan_chunk_moe_conflict_logit_l2_weight,
            0.01,
        )

    def test_upgrade_rejects_the_legacy_monotonic_alignment_route(self):
        dataset = load_config(SCENEPLAN_DATASET)
        model = load_config(
            DIT_CONFIG_ROOT
            / "qwen35_0p8b_300m_model_sceneplan_44_chunkmoe_v12.json"
        )
        candidate = copy.deepcopy(model)
        candidate["model"]["diffusion"]["config"][
            "sceneplan_frame_text_alignment"
        ] = {"enabled": True}
        with self.assertRaisesRegex(
            ValueError, "no-align ScenePlan attention/MoE"
        ):
            validate_t2a_config(candidate, dataset)


if __name__ == "__main__":
    unittest.main()
