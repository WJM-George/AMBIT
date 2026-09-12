from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from stable_audio_tools.training.ema import EMA, TrainableParameterEMA
from stable_audio_tools.training.transfusion import TransfusionSpatialTrainingWrapper


class _TinyTransfusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.qwen_conditioner = nn.Embedding(3, 2, padding_idx=0)
        self.modality_ids = ["spatial_traj", "foa_latent"]


class TrainableParameterEMATests(unittest.TestCase):
    def test_global_stratified_times_cover_the_complete_ddp_batch(self):
        columns = [
            TransfusionSpatialTrainingWrapper._global_stratified_time_column(
                batch_size=4,
                device="cpu",
                minimum=0.0,
                maximum=1.0,
                global_rank=rank,
                world_size=2,
                schedule_step=17,
                seed=20260810,
            )
            for rank in range(2)
        ]
        bins = torch.floor(torch.cat(columns) * 8).long().tolist()
        self.assertEqual(sorted(bins), list(range(8)))

    def test_diffusion_forcing_puts_mass_on_the_clean_context_boundary(self):
        wrapper = TransfusionSpatialTrainingWrapper(
            _TinyTransfusion(),
            optimizer_configs={
                "transfusion": {
                    "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                }
            },
            use_ema=False,
            use_velocity_consistency=False,
            diffusion_forcing_options={
                "context_clean_probability": 1.0,
                "target_time_sampling": "global_stratified",
                "stratified_seed": 20260810,
            },
        )
        examples = [
            {"metadata": {"has_context": present}}
            for present in (True, True, False, True, False, True, True, True)
        ]
        times, target, scale = wrapper._flow_stage(
            {
                "type": "diffusion_forcing",
                "order": ["spatial_traj", "foa_latent"],
                "target_modality": "foa_latent",
                "context_modalities": ["spatial_traj"],
                "context_presence_key": "has_context",
            },
            batch_size=len(examples),
            device="cpu",
            examples=examples,
            schedule_step=5,
        )

        self.assertEqual(target, 1)
        self.assertEqual(scale, 1.0)
        self.assertTrue(torch.equal(times[:, 0], torch.ones(len(examples))))
        self.assertEqual(
            sorted(torch.floor(times[:, 1] * len(examples)).long().tolist()),
            list(range(len(examples))),
        )
        self.assertEqual(
            float(
                wrapper._last_flow_stage_metrics[
                    "boundary_anchor_fraction_present"
                ]
            ),
            1.0,
        )

    def test_diffusion_forcing_legacy_context_has_no_clean_atom(self):
        wrapper = TransfusionSpatialTrainingWrapper(
            _TinyTransfusion(),
            optimizer_configs={
                "transfusion": {
                    "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                }
            },
            use_ema=False,
            use_velocity_consistency=False,
        )
        examples = [{"metadata": {"has_context": True}} for _ in range(64)]
        times, _, _ = wrapper._flow_stage(
            {
                "type": "diffusion_forcing",
                "order": ["spatial_traj", "foa_latent"],
                "target_modality": "foa_latent",
                "context_modalities": ["spatial_traj"],
                "context_presence_key": "has_context",
            },
            batch_size=len(examples),
            device="cpu",
            examples=examples,
        )
        self.assertTrue(torch.all(times[:, 0] < 1.0))

    def test_sequential_flow_schedule_alternates_and_cleans_prefix(self):
        wrapper = TransfusionSpatialTrainingWrapper(
            _TinyTransfusion(),
            optimizer_configs={
                "transfusion": {
                    "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                }
            },
            use_ema=False,
            use_velocity_consistency=False,
            flow_schedule={
                "type": "sequential_alternating",
                "order": ["spatial_traj", "foa_latent"],
                "clean_prefix_time_range": [0.9, 1.0],
            },
        )
        schedule = wrapper.flow_schedule

        first_times, first_target, first_scale = wrapper._flow_stage(
            schedule, batch_size=8, device="cpu", schedule_step=0
        )
        second_times, second_target, second_scale = wrapper._flow_stage(
            schedule, batch_size=8, device="cpu", schedule_step=1
        )

        self.assertEqual(first_target, 0)
        self.assertEqual(second_target, 1)
        self.assertEqual(first_scale, 2.0)
        self.assertEqual(second_scale, 2.0)
        self.assertTrue(torch.all((first_times >= 0.0) & (first_times <= 1.0)))
        self.assertTrue(torch.all(second_times[:, 0] >= 0.9))
        self.assertTrue(torch.all(second_times[:, 0] <= 1.0))

    def test_tracks_only_trainable_parameters_and_restores_context(self):
        module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        module[1].requires_grad_(False)
        ema = TrainableParameterEMA(
            module,
            beta=0.5,
            update_after_step=0,
            update_every=1,
            power=1.0,
        )
        self.assertEqual(
            ema.parameter_names,
            ("0.weight", "0.bias"),
        )
        self.assertNotIn("_online_model_ref", dict(ema.named_modules()))

        ema.update()
        with torch.no_grad():
            module[0].weight.add_(2.0)
        ema.update()
        with torch.no_grad():
            module[0].weight.add_(2.0)
            online = module[0].weight.detach().clone()
        ema.update()
        shadow = dict(ema.named_shadows())["0.weight"]
        self.assertFalse(torch.equal(shadow, online))

        with ema.apply_to():
            self.assertTrue(torch.equal(module[0].weight, shadow))
        self.assertTrue(torch.equal(module[0].weight, online))

    def test_standard_ema_tracks_bfloat16(self):
        module = nn.Linear(2, 2).to(torch.bfloat16)
        ema = EMA(module, update_after_step=0, update_every=1)
        self.assertIn("weight", ema.parameter_names)

    def test_transfusion_exports_text_conditioner_ema(self):
        wrapper = TransfusionSpatialTrainingWrapper(
            _TinyTransfusion(),
            optimizer_configs={
                "transfusion": {
                    "optimizer": {"type": "AdamW", "config": {"lr": 1e-4}}
                }
            },
            ema_beta=0.5,
            ema_power=1.0,
            ema_update_every=1,
            ema_update_after_step=0,
            use_velocity_consistency=False,
        )
        self.assertEqual(wrapper.text_conditioner_ema.parameter_names, ("weight",))

        wrapper.on_before_zero_grad()
        with torch.no_grad():
            wrapper.transfusion.qwen_conditioner.weight.add_(2.0)
        wrapper.on_before_zero_grad()
        with torch.no_grad():
            wrapper.transfusion.qwen_conditioner.weight.add_(2.0)
            online = wrapper.transfusion.qwen_conditioner.weight.detach().clone()
        wrapper.on_before_zero_grad()

        shadow = dict(wrapper.text_conditioner_ema.named_shadows())["weight"]
        self.assertFalse(torch.equal(shadow, online))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            wrapper.export_model(str(path))
            saved = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]

        self.assertTrue(torch.equal(saved["qwen_conditioner.weight"], shadow.cpu()))
        self.assertTrue(
            torch.equal(wrapper.transfusion.qwen_conditioner.weight, online)
        )


if __name__ == "__main__":
    unittest.main()
