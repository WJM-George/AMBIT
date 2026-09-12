#!/usr/bin/env python3
"""Regression gates for semantic cross-attention plus direct ScenePlan 4+4."""

from __future__ import annotations

import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan import (
    compile_model_44_controls,
    compile_model_semantic_caption,
    make_sceneplan_cfg_dropout_metadata,
    make_sceneplan_cfg_unknown_metadata,
    tokenize_model_semantic_caption,
)
from stable_audio_tools.configuration import (
    ConfigError,
    load_config,
    validate_training_configs,
)
from stable_audio_tools.models.dit import DiffusionTransformer
from stable_audio_tools.models.diffusion import (
    ConditionedDiffusionModelWrapper,
    DiTWrapper,
)
from stable_audio_tools.models.sceneplan_conditioning import ScenePlan44Conditioner
from stable_audio_tools.models.sceneplan_alignment import (
    ScenePlanFrameTextAlignment,
)
from stable_audio_tools.data.speech_alignment import check_alignment_items
from stable_audio_tools.data.resumable_dataloader import ResumableDataLoader
from stable_audio_tools.inference.sampling import sample_diffusion
from stable_audio_tools.training.diffusion import DiffusionCondTrainingWrapper
from stable_audio_tools.training.ema import EMA
from scripts.t2a.data.build_sceneplan_speech_timing_index import (
    _refined_centers,
)


class CharacterTokenizer:
    """Tiny offset-preserving tokenizer used only by these contract tests."""

    def __call__(
        self,
        text,
        *,
        truncation=False,
        padding=False,
        max_length=None,
        return_offsets_mapping=False,
        add_special_tokens=True,
    ):
        ids = [10 + (ord(char) % 97) for char in text]
        attention = [1] * len(ids)
        offsets = [(index, index + 1) for index in range(len(ids))]
        if padding == "max_length":
            if max_length is None or len(ids) > max_length:
                raise ValueError("test tokenizer input exceeds max_length")
            amount = max_length - len(ids)
            ids += [0] * amount
            attention += [0] * amount
            offsets += [(0, 0)] * amount
        output = {"input_ids": ids, "attention_mask": attention}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output


class ResumableDataLoaderTests(unittest.TestCase):
    def test_resume_yields_the_next_unconsumed_batch(self):
        dataset = TensorDataset(torch.arange(12))
        original = ResumableDataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            in_order=True,
        )
        iterator = iter(original)
        torch.testing.assert_close(next(iterator)[0], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(next(iterator)[0], torch.tensor([3, 4, 5]))
        saved = original.state_dict()
        self.assertEqual(saved["batches_yielded"], 2)

        resumed = ResumableDataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            in_order=True,
        )
        resumed.load_state_dict(saved)
        resumed.assert_resume_state_loaded()
        torch.testing.assert_close(
            next(iter(resumed))[0], torch.tensor([6, 7, 8])
        )

    def test_mid_epoch_guard_rejects_a_legacy_checkpoint(self):
        loader = ResumableDataLoader(
            TensorDataset(torch.arange(4)),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            in_order=True,
        )
        with self.assertRaisesRegex(RuntimeError, "no resumable DataLoader cursor"):
            loader.assert_resume_state_loaded()


def example_sceneplan():
    return {
        "sample_id": "unit_0001",
        "duration_sec": 2.0,
        "room": {"type": "moderate"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": "A small dog barks twice with a sharp dry timbre.",
                "activity": {"onset_sec": 0.0, "offset_sec": 1.0},
                "trajectory": {
                    "type": "linear",
                    "start": {
                        "azimuth_deg": -90.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                    "end": {
                        "azimuth_deg": 90.0,
                        "elevation_deg": 0.0,
                        "distance_m": 2.0,
                    },
                },
                "gain_db": -4.0,
            },
            {
                "source_id": "source_2",
                "kind": "speech",
                "speaker_description": (
                    "An adult female narrator with a warm measured voice"
                ),
                "transcript": "There she stood.",
                "activity": {"onset_sec": 0.5, "offset_sec": 1.5},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": 30.0,
                        "elevation_deg": 10.0,
                        "distance_m": 1.2,
                    },
                },
                "gain_db": 0.0,
            },
        ],
    }


class SemanticCaptionTests(unittest.TestCase):
    def test_speaker_and_who_says_are_one_event_transcript_is_speech(self):
        caption = compile_model_semantic_caption(example_sceneplan())
        tokenized = tokenize_model_semantic_caption(
            caption, CharacterTokenizer(), max_length=512
        )
        text = caption["text"]
        event = tokenized["event_source_ids"]
        speech = tokenized["speech_source_ids"]
        who = text.index("who says")
        transcript = text.index("There she stood.")
        self.assertEqual(int(event[who]), 3)
        self.assertEqual(int(speech[who]), 0)
        self.assertEqual(int(event[transcript]), 0)
        self.assertEqual(int(speech[transcript]), 3)
        self.assertFalse(bool(np.any((event > 0) & (speech > 0))))
        self.assertEqual(int(event[text.index("moderately")]), 0)

    def test_gain_is_provenance_only_not_a_control_feature(self):
        first = compile_model_44_controls(
            example_sceneplan(), model_num_samples=88_200, latent_frames_valid=87
        )
        changed = copy.deepcopy(example_sceneplan())
        changed["sources"][0]["gain_db"] = 8.0
        changed["sources"][1]["gain_db"] = -12.0
        second = compile_model_44_controls(
            changed, model_num_samples=88_200, latent_frames_valid=87
        )
        np.testing.assert_array_equal(
            first["source_event_frame_ids"], second["source_event_frame_ids"]
        )
        np.testing.assert_allclose(
            first["source_trajectory_features"],
            second["source_trajectory_features"],
        )


class LocalConditionerTests(unittest.TestCase):
    def test_direct_blocks_distinguish_known_inactive_and_unknown(self):
        module = ScenePlan44Conditioner(
            event_embedding_dim=3,
            trajectory_embedding_dim=2,
            output_dim=20,
        )
        event = torch.zeros(1, 4, 5, dtype=torch.long)
        event[0, 0, :3] = 1
        trajectory = torch.zeros(1, 4, 5, 5)
        trajectory[0, 0, :3, 1] = 1.0
        known = module(
            source_event_frame_ids=event,
            source_trajectory_features=trajectory,
        )
        self.assertEqual(tuple(known.shape), (1, 20, 5))
        self.assertEqual(float(known[..., 4].abs().sum()), 0.0)

        unknown_event = torch.full_like(event, -1)
        unknown = module(
            source_event_frame_ids=unknown_event,
            source_trajectory_features=torch.zeros_like(trajectory),
        )
        self.assertGreater(float(unknown.abs().sum()), 0.0)
        self.assertGreater(float((known - unknown).abs().sum()), 0.0)

    def test_swapped_source_id_fails_closed(self):
        module = ScenePlan44Conditioner()
        event = torch.zeros(1, 4, 2, dtype=torch.long)
        event[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "wrong source slot"):
            module(
                source_event_frame_ids=event,
                source_trajectory_features=torch.zeros(1, 4, 2, 5),
            )


class FrameTextAlignmentTests(unittest.TestCase):
    @staticmethod
    def _inputs(*, unknown_caption=False, unknown_structured=False):
        # token 0 describes sound source 1; token 2 describes the formal
        # speech source 2; tokens 3 and 4 are its two lexical transcript
        # tokens.  The remaining token is neutral template text.
        token_embeddings = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        token_mask = torch.ones(1, 6, dtype=torch.bool)
        event_ids = torch.tensor([[1, 0, 2, 0, 0, 0]], dtype=torch.long)
        speech_ids = torch.tensor([[0, 0, 0, 2, 2, 0]], dtype=torch.long)
        lexical = torch.tensor(
            [[False, False, False, True, True, False]], dtype=torch.bool
        )
        if unknown_caption:
            event_ids.fill_(-1)
            speech_ids.fill_(-1)
            lexical.zero_()
        token_aux = {
            "event_source_ids": event_ids,
            "speech_source_ids": speech_ids,
            "speech_lexical_mask": lexical,
        }
        if not unknown_caption:
            token_aux.update(
                speech_duration_target_fraction=torch.tensor(
                    [[0.0, 0.0, 0.0, 0.25, 0.75, 0.0]]
                ),
                speech_duration_target_mask=lexical.clone(),
            )

        frame_event_ids = torch.zeros(1, 4, 8, dtype=torch.long)
        frame_event_ids[0, 0, :3] = 1
        frame_event_ids[0, 1, 3:8] = 2
        if unknown_structured:
            frame_event_ids.fill_(-1)
        frame_aux = {
            "source_event_frame_ids": frame_event_ids,
            "frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        }
        return token_embeddings, token_mask, token_aux, frame_aux

    def test_bias_separates_event_semantics_and_monotonic_transcript(self):
        module = ScenePlanFrameTextAlignment(
            token_dim=4,
            duration_hidden_dim=8,
            duration_dropout=0.0,
        )
        embeddings, mask, token_aux, frame_aux = self._inputs()
        bias, info = module(
            embeddings, mask, token_aux, frame_aux, query_frames=8
        )
        self.assertEqual(tuple(bias.shape), (1, 1, 8, 6))
        self.assertGreater(float(bias[0, 0, 1, 0]), 0.0)
        self.assertEqual(float(bias[0, 0, 5, 0]), 0.0)
        self.assertGreater(float(bias[0, 0, 5, 2]), 0.0)
        # Zero initialization is deliberately the uniform-duration baseline.
        torch.testing.assert_close(
            info["predicted_fractions"][0, 3:5], torch.tensor([0.5, 0.5])
        )
        self.assertGreater(float(info["duration_kl"]), 0.0)
        torch.testing.assert_close(
            info["duration_kl"], info["duration_uniform_kl"]
        )
        # Earlier lexical text must prefer an earlier speech frame and the
        # later text a later frame.
        self.assertGreater(float(bias[0, 0, 4, 3]), float(bias[0, 0, 7, 3]))
        self.assertGreater(float(bias[0, 0, 7, 4]), float(bias[0, 0, 4, 4]))

    def test_cfg_unknown_is_zero_not_known_inactive(self):
        module = ScenePlanFrameTextAlignment(
            token_dim=4, duration_hidden_dim=8, duration_dropout=0.0
        )
        values = self._inputs(
            unknown_caption=True, unknown_structured=True
        )
        bias, info = module(*values, query_frames=8)
        self.assertEqual(int(torch.count_nonzero(bias)), 0)
        self.assertEqual(int(info["duration_teacher_rows"]), 0)

    def test_duration_head_can_beat_uniform_teacher_baseline(self):
        torch.manual_seed(20260824)
        module = ScenePlanFrameTextAlignment(
            token_dim=4, duration_hidden_dim=8, duration_dropout=0.0
        )
        embeddings, mask, token_aux, frame_aux = self._inputs()
        optimizer = torch.optim.AdamW(
            module.duration_predictor.parameters(), lr=0.05
        )
        with torch.no_grad():
            _, initial = module(
                embeddings, mask, token_aux, frame_aux, query_frames=8
            )
            uniform = float(initial["duration_uniform_kl"])
        for _ in range(80):
            _, info = module(
                embeddings, mask, token_aux, frame_aux, query_frames=8
            )
            optimizer.zero_grad(set_to_none=True)
            info["duration_kl"].backward()
            optimizer.step()
        _, final = module(
            embeddings, mask, token_aux, frame_aux, query_frames=8
        )
        self.assertLess(float(final["duration_kl"]), uniform * 0.05)

    def test_attention_bias_supports_a_complete_backward_pass(self):
        torch.manual_seed(20260825)
        module = ScenePlanFrameTextAlignment(
            token_dim=4, duration_hidden_dim=8, duration_dropout=0.0
        )
        embeddings, mask, token_aux, frame_aux = self._inputs()
        embeddings = embeddings.requires_grad_(True)
        bias, info = module(
            embeddings, mask, token_aux, frame_aux, query_frames=8
        )
        loss = bias.square().mean() + info["duration_kl"]
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        # The duration head deliberately starts with a zero final projection,
        # so its first backward pass need not reach the Qwen embeddings yet.
        # It must nevertheless traverse the complete score graph without an
        # autograd version error and open the duration head itself.
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertIsNotNone(module.speech_bias_raw.grad)
        self.assertGreater(float(module.speech_bias_raw.grad.abs()), 0.0)
        duration_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.duration_predictor.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(duration_grad, 0.0)

    def test_dit_route_returns_alignment_diagnostics(self):
        model = DiffusionTransformer(
            io_channels=2,
            embed_dim=8,
            cond_token_dim=4,
            input_concat_dim=3,
            depth=1,
            num_heads=2,
            timestep_embed_dim=8,
            input_concat_cond_cfg=True,
            require_explicit_negative_input_concat=True,
            activation_checkpointing=False,
            rotary_pos_emb=False,
            sceneplan_frame_text_alignment={
                "enabled": True,
                "duration_hidden_dim": 8,
                "duration_dropout": 0.0,
            },
        )
        embeddings, mask, token_aux, frame_aux = self._inputs()
        output, info = model(
            torch.randn(1, 2, 8),
            torch.full((1,), 0.5),
            cross_attn_cond=embeddings,
            cross_attn_cond_mask=mask,
            cross_attn_aux=token_aux,
            input_concat_cond=torch.randn(1, 3, 8),
            input_concat_aux=frame_aux,
            return_alignment_info=True,
        )
        self.assertEqual(tuple(output.shape), (1, 2, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIn("duration_kl", info)


class SpeechTimingFinalizerTests(unittest.TestCase):
    def test_terminal_timestamp_grid_clip_is_bounded_not_silently_ignored(self):
        allowed = check_alignment_items(
            [{"text": "word", "start_sec": 0.80, "end_sec": 1.12}],
            1.0,
            nominal_endpoint_grid_overshoot_sec=0.16,
        )
        rejected = check_alignment_items(
            [{"text": "word", "start_sec": 0.80, "end_sec": 1.24}],
            1.0,
            nominal_endpoint_grid_overshoot_sec=0.16,
        )
        invalid_start = check_alignment_items(
            [{"text": "word", "start_sec": 1.20, "end_sec": 1.24}],
            1.0,
            nominal_endpoint_grid_overshoot_sec=0.16,
        )
        bounded_terminal_suffix = check_alignment_items(
            [
                {"text": "and", "start_sec": 0.80, "end_sec": 1.01},
                {"text": "then", "start_sec": 1.01, "end_sec": 1.01},
            ],
            1.0,
            nominal_endpoint_grid_overshoot_sec=0.16,
        )
        self.assertTrue(allowed["in_bounds"])
        self.assertFalse(allowed["raw_in_bounds"])
        self.assertAlmostEqual(allowed["endpoint_grid_overshoot_sec"], 0.12)
        self.assertTrue(allowed["endpoint_grid_overshoot_within_nominal_bound"])
        self.assertTrue(rejected["in_bounds"])
        self.assertFalse(rejected["endpoint_grid_overshoot_within_nominal_bound"])
        self.assertFalse(invalid_start["in_bounds"])
        self.assertTrue(bounded_terminal_suffix["in_bounds"])
        self.assertFalse(bounded_terminal_suffix["raw_in_bounds"])

    def test_zero_width_word_precedes_first_subtoken_of_next_word(self):
        def item(text, start, end, char_start, char_end):
            return {
                "text": text,
                "absolute_start_sec": start,
                "absolute_end_sec": end,
                "transcript_char_start": char_start,
                "transcript_char_end": char_end,
            }

        # This is the production edge case caught by the 1k gate: a
        # zero-width word at the same grid boundary as a following word which
        # splits into multiple Qwen subtokens.  The old midpoint-based repair
        # put "the" after the first "f" subtoken.
        items = [
            item(" of", 2.44, 2.60, 0, 2),
            item(" the", 2.60, 2.60, 3, 6),
            item(" f", 2.60, 3.16, 7, 8),
            item("ountains", 2.60, 3.16, 8, 16),
        ]
        centers = _refined_centers(
            items, activity_onset=2.40, activity_offset=3.20
        )
        self.assertTrue(
            all(right > left for left, right in zip(centers, centers[1:]))
        )
        self.assertGreater(centers[1], centers[0])
        self.assertLess(centers[1], centers[2])


class CfgTests(unittest.TestCase):
    @staticmethod
    def _row():
        return {
            "prompt": {"input_ids": torch.ones(4, dtype=torch.long)},
            "sceneplan_44": {
                "source_event_frame_ids": torch.zeros(4, 3)
            },
        }

    def test_metadata_unknown_is_non_mutating(self):
        row = self._row()
        unknown = make_sceneplan_cfg_unknown_metadata(row)
        self.assertNotIn("cfg_unknown", row["prompt"])
        self.assertNotIn("cfg_unknown", row["sceneplan_44"])
        self.assertIs(unknown["prompt"]["cfg_unknown"], True)
        self.assertIs(unknown["sceneplan_44"]["cfg_unknown"], True)

    def test_branch_metadata_can_drop_caption_or_structured_independently(self):
        row = self._row()
        caption_only = make_sceneplan_cfg_dropout_metadata(
            row, caption_unknown=True, structured_unknown=False
        )
        structured_only = make_sceneplan_cfg_dropout_metadata(
            row, caption_unknown=False, structured_unknown=True
        )
        self.assertIs(caption_only["prompt"]["cfg_unknown"], True)
        self.assertNotIn("cfg_unknown", caption_only["sceneplan_44"])
        self.assertNotIn("cfg_unknown", structured_only["prompt"])
        self.assertIs(structured_only["sceneplan_44"]["cfg_unknown"], True)
        self.assertNotIn("cfg_unknown", row["prompt"])
        self.assertNotIn("cfg_unknown", row["sceneplan_44"])

    def test_training_dropout_exposes_all_four_independent_states(self):
        wrapper = DiffusionCondTrainingWrapper.__new__(
            DiffusionCondTrainingWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.sceneplan_cfg_dropout_mode = "independent"
        wrapper.sceneplan_caption_unknown_prob = 0.15
        wrapper.sceneplan_structured_unknown_prob = 0.15
        wrapper._sceneplan_cfg_running_counts = [0, 0, 0, 0, 0]
        rows = [self._row() for _ in range(4)]
        draws = torch.tensor(
            [
                [0.10, 0.10],  # both unknown
                [0.10, 0.90],  # caption unknown only
                [0.90, 0.10],  # structured unknown only
                [0.90, 0.90],  # fully conditioned
            ]
        )
        with patch(
            "stable_audio_tools.training.diffusion.torch.rand",
            return_value=draws,
        ):
            output, stats = wrapper._apply_sceneplan_independent_cfg_dropout(
                rows
            )
        states = [
            (
                bool(value["prompt"].get("cfg_unknown", False)),
                bool(value["sceneplan_44"].get("cfg_unknown", False)),
            )
            for value in output
        ]
        self.assertEqual(
            states,
            [(True, True), (True, False), (False, True), (False, False)],
        )
        self.assertEqual(stats["caption_unknown_fraction"], 0.5)
        self.assertEqual(stats["structured_unknown_fraction"], 0.5)
        self.assertEqual(stats["joint_unknown_fraction"], 0.25)
        self.assertEqual(stats["full_condition_fraction"], 0.25)

    def test_independent_15_percent_empirical_distribution(self):
        wrapper = DiffusionCondTrainingWrapper.__new__(
            DiffusionCondTrainingWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.sceneplan_cfg_dropout_mode = "independent"
        wrapper.sceneplan_caption_unknown_prob = 0.15
        wrapper.sceneplan_structured_unknown_prob = 0.15
        wrapper._sceneplan_cfg_running_counts = [0, 0, 0, 0, 0]
        torch.manual_seed(20260821)
        _, stats = wrapper._apply_sceneplan_independent_cfg_dropout(
            [self._row() for _ in range(20_000)]
        )
        self.assertAlmostEqual(
            stats["caption_unknown_fraction"], 0.15, delta=0.01
        )
        self.assertAlmostEqual(
            stats["structured_unknown_fraction"], 0.15, delta=0.01
        )
        self.assertAlmostEqual(
            stats["joint_unknown_fraction"], 0.0225, delta=0.005
        )
        self.assertAlmostEqual(
            stats["full_condition_fraction"], 0.7225, delta=0.0125
        )

    def test_model_config_forbids_joint_or_double_cfg_dropout(self):
        repo = Path(__file__).resolve().parents[3]
        model_path = repo / (
            "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44.json"
        )
        dataset_path = repo / (
            "stable_audio_tools/configs/dataset_configs/"
            "sceneplan_44_overfit10.json"
        )
        model = load_config(model_path)
        dataset = load_config(dataset_path)
        validate_training_configs(model, dataset)

        joint = copy.deepcopy(model)
        joint["training"]["sceneplan_cfg_dropout"]["mode"] = "joint"
        with self.assertRaisesRegex(ConfigError, "must be independent"):
            validate_training_configs(joint, dataset)

        double_dropout = copy.deepcopy(model)
        double_dropout["training"]["cfg_dropout_prob"] = 0.1
        with self.assertRaisesRegex(ConfigError, "requires cfg_dropout_prob=0"):
            validate_training_configs(double_dropout, dataset)

    def test_preflight_candidate_requires_immutable_speech_timing_sidecar(self):
        repo = Path(__file__).resolve().parents[3]
        model = load_config(
            repo
            / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44_preflight_candidate.json"
        )
        dataset = load_config(
            repo
            / "stable_audio_tools/configs/dataset_configs/"
            "sceneplan_v2_train.json"
        )
        with self.assertRaisesRegex(ConfigError, "speech timing sidecar"):
            validate_training_configs(model, dataset)
        dataset.update(
            speech_timing_index_path="/immutable/teacher.sqlite",
            speech_timing_index_sha256="a" * 64,
            expected_speech_timing_rows=500_000,
            require_speech_timing=True,
        )
        validate_training_configs(model, dataset)

        validation_dataset = load_config(
            repo
            / "stable_audio_tools/configs/dataset_configs/"
            "sceneplan_v2_validation.json"
        )
        with self.assertRaisesRegex(ConfigError, "speech timing sidecar"):
            validate_training_configs(model, validation_dataset)
        validate_training_configs(
            model,
            validation_dataset,
            allow_missing_speech_timing=True,
        )
        partially_configured_validation = copy.deepcopy(validation_dataset)
        partially_configured_validation["require_speech_timing"] = False
        with self.assertRaisesRegex(ConfigError, "teacher-free validation"):
            validate_training_configs(
                model,
                partially_configured_validation,
                allow_missing_speech_timing=True,
            )

        invalid = copy.deepcopy(model)
        invalid["training"]["sceneplan_sound_transient_loss"] = {
            "enabled": True
        }
        with self.assertRaisesRegex(ConfigError, "rejected transient"):
            validate_training_configs(invalid, dataset)

    def test_multi_index_sceneplan_rows_are_explicit_and_exact(self):
        repo = Path(__file__).resolve().parents[3]
        model = load_config(
            repo
            / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44.json"
        )
        dataset = load_config(
            repo
            / "stable_audio_tools/configs/dataset_configs/"
            "sceneplan_v2_train.json"
        )
        dataset["expected_num_samples"] = 1_120_000
        dataset["datasets"] = [
            {
                "id": "frozen_base",
                "path": "/immutable/base.sqlite",
                "num_samples": 1_100_000,
                "weight": 1.0,
            },
            {
                "id": "frozen_sound_delta",
                "path": "/immutable/sound.sqlite",
                "num_samples": 20_000,
                "weight": 1.0,
            },
        ]
        validate_training_configs(model, dataset)

        wrong_total = copy.deepcopy(dataset)
        wrong_total["datasets"][1]["num_samples"] = 19_999
        with self.assertRaisesRegex(ConfigError, "do not sum"):
            validate_training_configs(model, wrong_total)

        weighted = copy.deepcopy(dataset)
        weighted["datasets"][1]["weight"] = 2.0
        with self.assertRaisesRegex(ConfigError, "weight=1.0"):
            validate_training_configs(model, weighted)

    def test_multi_index_loader_concatenates_frozen_children(self):
        from stable_audio_tools.data.dataset import create_dataloader_from_config

        calls = []

        class FakeScenePlanDataset(torch.utils.data.Dataset):
            def __init__(self, index_path, **kwargs):
                calls.append((index_path, kwargs))
                self.rows = int(kwargs["expected_num_samples"])

            def __len__(self):
                return self.rows

            def __getitem__(self, index):
                return torch.tensor(index), {"index": index}

        config = {
            "dataset_type": "sceneplan_v2_preencoded",
            "expected_num_samples": 5,
            "latent_crop_length": 432,
            "caption_max_tokens": 512,
            "random_crop": False,
            "require_complete": True,
            "datasets": [
                {"id": "base", "path": "/base.sqlite", "num_samples": 3},
                {"id": "delta", "path": "/delta.sqlite", "num_samples": 2},
            ],
        }
        with patch(
            "stable_audio_tools.data.sceneplan_v2_dataset.ScenePlanV2Dataset",
            FakeScenePlanDataset,
        ):
            loader = create_dataloader_from_config(
                config,
                batch_size=1,
                sample_size=442_368,
                sample_rate=44_100,
                audio_channels=4,
                num_workers=0,
                shuffle=False,
                tokenizers={"prompt": (CharacterTokenizer(), 512)},
            )
        self.assertIsInstance(loader.dataset, torch.utils.data.ConcatDataset)
        self.assertEqual(len(loader.dataset), 5)
        self.assertEqual([call[0] for call in calls], ["/base.sqlite", "/delta.sqlite"])
        self.assertEqual(
            [call[1]["expected_num_samples"] for call in calls], [3, 2]
        )

    def test_multi_index_loader_passes_audited_ordinal_range(self):
        from stable_audio_tools.data.dataset import create_dataloader_from_config

        calls = []

        class FakeScenePlanDataset(torch.utils.data.Dataset):
            def __init__(self, index_path, **kwargs):
                calls.append((index_path, kwargs))
                self.rows = int(kwargs["expected_num_samples"])

            def __len__(self):
                return self.rows

            def __getitem__(self, index):
                return torch.tensor(index), {"index": index}

        config = {
            "dataset_type": "sceneplan_v2_preencoded",
            "expected_num_samples": 8,
            "latent_crop_length": 432,
            "caption_max_tokens": 512,
            "random_crop": False,
            "require_complete": True,
            "datasets": [
                {"id": "full", "path": "/same.sqlite", "num_samples": 5},
                {
                    "id": "speech",
                    "path": "/same.sqlite",
                    "index_num_samples": 5,
                    "num_samples": 3,
                    "ordinal_range": [0, 3],
                },
            ],
        }
        with patch(
            "stable_audio_tools.data.sceneplan_v2_dataset.ScenePlanV2Dataset",
            FakeScenePlanDataset,
        ):
            loader = create_dataloader_from_config(
                config,
                batch_size=1,
                sample_size=442_368,
                sample_rate=44_100,
                audio_channels=4,
                num_workers=0,
                shuffle=False,
                tokenizers={"prompt": (CharacterTokenizer(), 512)},
            )
        self.assertEqual(len(loader.dataset), 8)
        self.assertIsNone(calls[0][1]["ordinal_start"])
        self.assertIsNone(calls[0][1]["ordinal_stop"])
        self.assertEqual(calls[1][1]["ordinal_start"], 0)
        self.assertEqual(calls[1][1]["ordinal_stop"], 3)

        wrong = copy.deepcopy(config)
        wrong["datasets"][1]["ordinal_range"] = [0, 4]
        model = load_config(
            REPO_ROOT
            / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44.json"
        )
        with self.assertRaisesRegex(ConfigError, "length does not match"):
            validate_training_configs(model, wrong)

    def test_explicit_negative_text_and_local_reach_cfg_batch(self):
        model = DiffusionTransformer(
            io_channels=2,
            embed_dim=8,
            cond_token_dim=4,
            input_concat_dim=3,
            depth=1,
            num_heads=2,
            timestep_embed_dim=8,
            input_concat_cond_cfg=True,
            require_explicit_negative_input_concat=True,
            activation_checkpointing=False,
        )
        captured = {}

        def fake_forward(this, x, t, **kwargs):
            captured.update(kwargs)
            return x

        model._forward = types.MethodType(fake_forward, model)
        positive_local = torch.ones(1, 3, 4)
        negative_local = torch.full((1, 3, 4), -7.0)
        model(
            torch.ones(1, 2, 4),
            torch.full((1,), 0.5),
            cross_attn_cond=torch.ones(1, 3, 4),
            cross_attn_cond_mask=torch.ones(1, 3, dtype=torch.bool),
            negative_cross_attn_cond=torch.full((1, 1, 4), 2.0),
            negative_cross_attn_mask=torch.ones(1, 1, dtype=torch.bool),
            input_concat_cond=positive_local,
            negative_input_concat_cond=negative_local,
            cfg_scale=2.0,
        )
        torch.testing.assert_close(
            captured["input_concat_cond"][1:], negative_local
        )
        self.assertEqual(tuple(captured["cross_attn_cond"].shape), (2, 3, 4))
        self.assertEqual(
            captured["cross_attn_cond_mask"][1].tolist(), [True, False, False]
        )

    def test_dit_wrapper_rescale_boolean_reaches_numeric_phi(self):
        class Capture(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = None

            def forward(self, x, t, **kwargs):
                self.phi = kwargs["scale_phi"]
                return x

        wrapper = DiTWrapper.__new__(DiTWrapper)
        nn.Module.__init__(wrapper)
        wrapper.diffusion_objective = "rectified_flow"
        wrapper.model = Capture()
        x = torch.randn(1, 2, 3)
        wrapper(x, torch.full((1,), 0.5), rescale_cfg=True)
        self.assertEqual(wrapper.model.phi, 0.4)

        wrapper(
            x,
            torch.full((1,), 0.5),
            rescale_cfg=True,
            scale_phi=0.25,
        )
        self.assertEqual(wrapper.model.phi, 0.25)

    def test_unified_sampler_resolves_rescale_boolean(self):
        class Capture(nn.Module):
            def __init__(self):
                super().__init__()
                self.phis = []

            def forward(self, x, t, **kwargs):
                self.phis.append(kwargs["scale_phi"])
                return torch.zeros_like(x)

        model = Capture()
        output = sample_diffusion(
            model=model,
            noise=torch.randn(1, 2, 4),
            cond_inputs={},
            diffusion_objective="rectified_flow",
            steps=2,
            cfg_scale=2.0,
            rescale_cfg=True,
            sampler_type="euler",
            decode=False,
            disable_tqdm=True,
        )
        self.assertEqual(tuple(output.shape), (1, 2, 4))
        self.assertTrue(model.phis)
        self.assertEqual(set(model.phis), {0.4})


class SpeechWeightTests(unittest.TestCase):
    def test_supervision_mask_is_not_a_condition_and_keeps_three_to_one(self):
        wrapper = DiffusionCondTrainingWrapper.__new__(DiffusionCondTrainingWrapper)
        nn.Module.__init__(wrapper)
        wrapper.sceneplan_speech_active_loss_weight = 3.0
        metadata = [
            {
                "sceneplan_44": {
                    "speech_active_frame_mask": torch.tensor(
                        [1, 1, 0, 0], dtype=torch.bool
                    )
                }
            }
        ]
        base = torch.ones(1, 2, 4)
        weighted, metrics = wrapper._apply_sceneplan_speech_active_weight(
            base, torch.ones(1, 4, dtype=torch.bool), metadata
        )
        self.assertAlmostEqual(float(weighted.mean()), 1.0, places=6)
        self.assertAlmostEqual(
            float(weighted[0, 0, 0] / weighted[0, 0, 3]), 3.0, places=6
        )
        self.assertAlmostEqual(
            float(metrics["train/speech_active_fraction"]), 0.5, places=6
        )


class SoundTemporalDifferenceTests(unittest.TestCase):
    class MinimalDiffusion(nn.Module):
        def __init__(self, objective="rectified_flow"):
            super().__init__()
            self.model = nn.Linear(2, 2)
            self.conditioner = nn.Identity()
            self.diffusion_objective = objective
            self.mask_padding_attention = False
            self.use_effective_length_for_schedule = False

    @staticmethod
    def _optimizer_config():
        return {
            "diffusion": {
                "optimizer": {
                    "type": "AdamW",
                    "config": {"lr": 1.0e-4},
                }
            }
        }

    def test_enabled_auxiliary_constructs_for_rectified_flow(self):
        wrapper = DiffusionCondTrainingWrapper(
            self.MinimalDiffusion(),
            use_ema=False,
            optimizer_configs=self._optimizer_config(),
            sceneplan_sound_temporal_difference_loss={
                "enabled": True,
                "scope": "sound_only",
                "weight": 0.05,
                "lags": [1, 2, 4],
                "smooth_l1_beta": 0.1,
            },
        )
        self.assertEqual(wrapper.diffusion_objective, "rectified_flow")
        self.assertTrue(wrapper.sceneplan_sound_temporal_difference_enabled)

    def test_enabled_auxiliary_rejects_a_non_flow_objective(self):
        with self.assertRaisesRegex(ValueError, "rectified-flow objective"):
            DiffusionCondTrainingWrapper(
                self.MinimalDiffusion(objective="v"),
                use_ema=False,
                optimizer_configs=self._optimizer_config(),
                sceneplan_sound_temporal_difference_loss={"enabled": True},
            )

    @staticmethod
    def _wrapper():
        wrapper = DiffusionCondTrainingWrapper.__new__(
            DiffusionCondTrainingWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.sceneplan_sound_temporal_difference_enabled = True
        wrapper.sceneplan_sound_temporal_difference_lags = (1, 2, 4)
        wrapper.sceneplan_sound_temporal_difference_beta = 0.1
        wrapper.sceneplan_sound_temporal_difference_weight = 0.05
        return wrapper

    @staticmethod
    def _metadata():
        return [
            {
                "model_sceneplan": {
                    "sources": [{"kind": "sound"}]
                }
            },
            {
                "model_sceneplan": {
                    "sources": [{"kind": "music"}]
                }
            },
        ]

    def test_clean_prediction_is_zero_and_music_rows_are_excluded(self):
        torch.manual_seed(7)
        wrapper = self._wrapper()
        clean = torch.randn(2, 3, 12)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0.25, 0.75])
        noised = (
            (1.0 - timesteps)[:, None, None] * clean
            + timesteps[:, None, None] * noise
        )
        velocity = noise - clean
        # A large error on the music row must not contribute to the sound-only
        # auxiliary.
        prediction = velocity.clone()
        prediction[1] += torch.randn_like(prediction[1]) * 100.0
        loss, metrics = wrapper._sceneplan_sound_temporal_difference_auxiliary(
            output=prediction,
            noised_inputs=noised,
            timesteps=timesteps,
            clean_target=clean,
            loss_mask=torch.ones(2, 12, dtype=torch.bool),
            metadata=self._metadata(),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(
            float(metrics["train/sound_temporal_difference_eligible_fraction"]),
            0.5,
            places=6,
        )

    def test_wrong_sound_derivatives_produce_gradient_without_reweighting_mse(self):
        torch.manual_seed(8)
        wrapper = self._wrapper()
        clean = torch.randn(2, 3, 12)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0.4, 0.6])
        noised = (
            (1.0 - timesteps)[:, None, None] * clean
            + timesteps[:, None, None] * noise
        )
        prediction = (noise - clean).detach().clone().requires_grad_(True)
        prediction.data[0, :, 5:] += 1.5
        base_mse = torch.nn.functional.mse_loss(prediction, noise - clean)
        base_before = base_mse.detach().clone()
        auxiliary, _ = wrapper._sceneplan_sound_temporal_difference_auxiliary(
            output=prediction,
            noised_inputs=noised,
            timesteps=timesteps,
            clean_target=clean,
            loss_mask=torch.ones(2, 12, dtype=torch.bool),
            metadata=self._metadata(),
        )
        self.assertGreater(float(auxiliary), 0.0)
        auxiliary.backward()
        self.assertGreater(float(prediction.grad[0].abs().sum()), 0.0)
        torch.testing.assert_close(base_mse.detach(), base_before)


class EmaThroughputTests(unittest.TestCase):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(7, 11)
            self.second = nn.Linear(11, 5)
            self.register_buffer("running", torch.arange(5, dtype=torch.float32))

    class ScalarReferenceEMA(EMA):
        @torch.no_grad()
        def copy_params_from_model_to_ema(self):
            for (_, target), (_, source) in zip(
                self.get_params_iter(self.ema_model),
                self.get_params_iter(self.model),
            ):
                target.copy_(source)
            for (_, target), (_, source) in zip(
                self.get_buffers_iter(self.ema_model),
                self.get_buffers_iter(self.model),
            ):
                target.copy_(source)

        @torch.no_grad()
        def update_moving_average(self, ma_model, current_model):
            decay = self.get_current_decay()
            for (name, source), (_, target) in zip(
                self.get_params_iter(current_model),
                self.get_params_iter(ma_model),
            ):
                if name in self.ignore_names or any(
                    name.startswith(prefix)
                    for prefix in self.ignore_startswith_names
                ):
                    continue
                if name in self.param_or_buffer_names_no_ema:
                    target.copy_(source)
                else:
                    target.lerp_(source, 1.0 - decay)
            for (name, source), (_, target) in zip(
                self.get_buffers_iter(current_model),
                self.get_buffers_iter(ma_model),
            ):
                if name in self.ignore_names or any(
                    name.startswith(prefix)
                    for prefix in self.ignore_startswith_names
                ):
                    continue
                if name in self.param_or_buffer_names_no_ema:
                    target.copy_(source)
                else:
                    target.lerp_(source, 1.0 - decay)

    def test_foreach_ema_is_numerically_equal_to_scalar_reference(self):
        torch.manual_seed(1234)
        foreach_model = self.Model()
        scalar_model = copy.deepcopy(foreach_model)
        kwargs = {
            "beta": 0.9999,
            "update_after_step": 1,
            "update_every": 1,
            "param_or_buffer_names_no_ema": {"second.bias"},
        }
        foreach_ema = EMA(foreach_model, **kwargs)
        scalar_ema = self.ScalarReferenceEMA(scalar_model, **kwargs)

        with patch(
            "stable_audio_tools.training.ema.torch._foreach_lerp_",
            wraps=torch._foreach_lerp_,
        ) as foreach_lerp:
            for update_index in range(9):
                with torch.no_grad():
                    for left, right in zip(
                        foreach_model.parameters(), scalar_model.parameters()
                    ):
                        delta = torch.full_like(left, 0.01 * (update_index + 1))
                        left.add_(delta)
                        right.add_(delta)
                    foreach_model.running.add_(0.25)
                    scalar_model.running.add_(0.25)
                foreach_ema.update()
                scalar_ema.update()

        self.assertGreater(foreach_lerp.call_count, 0)
        self.assertEqual(foreach_ema._step_value, scalar_ema._step_value)
        self.assertEqual(foreach_ema._initted_value, scalar_ema._initted_value)
        for left, right in zip(
            foreach_ema.ema_model.state_dict().values(),
            scalar_ema.ema_model.state_dict().values(),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_checkpoint_restores_python_step_mirrors(self):
        torch.manual_seed(4321)
        original = EMA(
            self.Model(), update_after_step=1, update_every=1
        )
        for _ in range(6):
            with torch.no_grad():
                for parameter in original.model.parameters():
                    parameter.add_(0.03)
            original.update()

        resumed = EMA(self.Model(), update_after_step=1, update_every=1)
        resumed.load_state_dict(copy.deepcopy(original.state_dict()))
        self.assertEqual(resumed._step_value, int(resumed.step.item()))
        self.assertEqual(resumed._initted_value, bool(resumed.initted.item()))

        with torch.no_grad():
            for left, right in zip(
                original.model.parameters(), resumed.model.parameters()
            ):
                left.add_(0.07)
                right.add_(0.07)
        original.update()
        resumed.update()
        self.assertEqual(original._step_value, resumed._step_value)
        for left, right in zip(
            original.ema_model.state_dict().values(),
            resumed.ema_model.state_dict().values(),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


class AudioPriorWarmStartTests(unittest.TestCase):
    class ExpandedInputCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_concat_dim = 3
            self.preprocess_conv = nn.Conv1d(5, 5, 1, bias=False)
            self.transformer = nn.Module()
            self.transformer.project_in = nn.Linear(5, 7, bias=False)
            self.transformer.shared = nn.Linear(7, 7, bias=False)

    class Route(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AudioPriorWarmStartTests.ExpandedInputCore()

    class PromptRoles(nn.Module):
        def __init__(self):
            super().__init__()
            self.event_role_embed = nn.Embedding(6, 7, padding_idx=1)
            self.speech_role_embed = nn.Embedding(6, 7, padding_idx=1)

    class Conditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.conditioners = nn.ModuleDict(
                {"prompt": AudioPriorWarmStartTests.PromptRoles()}
            )

    def test_audio_prefix_is_preserved_and_new_44_columns_start_zero(self):
        wrapper = ConditionedDiffusionModelWrapper.__new__(
            ConditionedDiffusionModelWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.model = self.Route()
        wrapper.conditioner = nn.Module()
        wrapper.io_channels = 2
        wrapper.diffusion_objective = "rectified_flow"

        source_project = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        source_preprocess = torch.arange(
            4, dtype=torch.float32
        ).reshape(2, 2, 1)
        source_shared = torch.full((7, 7), 0.125)
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
                    "config": {"io_channels": 2},
                },
            }
        }
        report = wrapper.load_pretrained_route_state_dict(
            state,
            source_model_config=source_config,
            prefer_ema=True,
        )

        project = wrapper.model.model.transformer.project_in.weight
        preprocess = wrapper.model.model.preprocess_conv.weight
        torch.testing.assert_close(project[:, :2], source_project)
        torch.testing.assert_close(project[:, 2:], torch.zeros_like(project[:, 2:]))
        torch.testing.assert_close(
            preprocess[:2, :2], source_preprocess
        )
        expected_preprocess = torch.zeros_like(preprocess)
        expected_preprocess[:2, :2] = source_preprocess
        torch.testing.assert_close(preprocess, expected_preprocess)
        torch.testing.assert_close(
            wrapper.model.model.transformer.shared.weight, source_shared
        )
        self.assertEqual(report["loaded"], 3)
        self.assertEqual(len(report["partial_expansions"]), 2)
        self.assertEqual(report["missing"], [])
        self.assertEqual(
            report["modality_mapping"],
            "exact_name_and_shape_plus_audio_prefix_input_expansion",
        )

    def test_zero_initialized_control_columns_open_after_one_optimizer_step(self):
        """The behavior-preserving expansion must not strand the 4+4 route.

        DiffusionTransformer applies ``preprocess_conv(x) + x`` before its
        input projection.  Consequently the raw structured features reach the
        zeroed new projection columns, those columns receive a gradient on the
        first update, and gradients reach the upstream ScenePlan conditioner
        from the second update onward.
        """

        torch.manual_seed(17)
        wrapper = ConditionedDiffusionModelWrapper.__new__(
            ConditionedDiffusionModelWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.model = self.Route()
        wrapper.conditioner = nn.Module()
        wrapper.io_channels = 2
        wrapper.diffusion_objective = "rectified_flow"

        state = {
            "diffusion_ema.ema_model.model.transformer.project_in.weight": (
                torch.randn(7, 2)
            ),
            "diffusion_ema.ema_model.model.preprocess_conv.weight": (
                torch.randn(2, 2, 1)
            ),
            "diffusion_ema.ema_model.model.transformer.shared.weight": (
                torch.randn(7, 7)
            ),
        }
        source_config = {
            "model": {
                "io_channels": 2,
                "diffusion": {
                    "diffusion_objective": "rectified_flow",
                    "config": {"io_channels": 2},
                },
            }
        }
        wrapper.load_pretrained_route_state_dict(
            state,
            source_model_config=source_config,
            prefer_ema=True,
        )

        core = wrapper.model.model
        optimizer = torch.optim.SGD(core.parameters(), lr=0.1)
        audio = torch.randn(2, 2, 5)
        controls = torch.randn(2, 3, 5, requires_grad=True)

        def objective() -> torch.Tensor:
            features = torch.cat([audio, controls], dim=1)
            features = core.preprocess_conv(features) + features
            projected = core.transformer.project_in(features.transpose(1, 2))
            return projected.square().mean()

        objective().backward()
        first_projection_gradient = (
            core.transformer.project_in.weight.grad[:, 2:].abs().sum()
        )
        self.assertGreater(float(first_projection_gradient), 0.0)
        # Exact behavior preservation means no upstream control gradient before
        # the first projection update; this is intentional and lasts one step.
        torch.testing.assert_close(controls.grad, torch.zeros_like(controls.grad))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        controls.grad = None

        objective().backward()
        self.assertGreater(float(controls.grad.abs().sum()), 0.0)

    def test_caption_cue_and_quote_roles_expand_to_four_source_rows(self):
        wrapper = ConditionedDiffusionModelWrapper.__new__(
            ConditionedDiffusionModelWrapper
        )
        nn.Module.__init__(wrapper)
        wrapper.model = self.Route()
        wrapper.conditioner = self.Conditioner()
        wrapper.io_channels = 2
        wrapper.diffusion_objective = "rectified_flow"

        cue = torch.arange(7, dtype=torch.float32) + 10
        quote = torch.arange(7, dtype=torch.float32) + 30
        source_roles = torch.stack([torch.zeros(7), cue, quote])
        state = {
            "diffusion_ema.ema_model.model.transformer.project_in.weight": (
                torch.randn(7, 2)
            ),
            "diffusion_ema.ema_model.model.preprocess_conv.weight": (
                torch.randn(2, 2, 1)
            ),
            "diffusion_ema.ema_model.model.transformer.shared.weight": (
                torch.randn(7, 7)
            ),
            "conditioner_ema.shadow_00000": source_roles,
        }
        source_config = {
            "model": {
                "io_channels": 2,
                "diffusion": {
                    "diffusion_objective": "rectified_flow",
                    "config": {"io_channels": 2},
                },
            }
        }
        report = wrapper.load_pretrained_route_state_dict(
            state,
            source_model_config=source_config,
            source_conditioner_ema_names=[
                "conditioners.prompt.caption_region_embed.weight"
            ],
            prefer_ema=True,
        )

        event = wrapper.conditioner.conditioners["prompt"].event_role_embed.weight
        speech = wrapper.conditioner.conditioners["prompt"].speech_role_embed.weight
        torch.testing.assert_close(event[:2], torch.zeros_like(event[:2]))
        torch.testing.assert_close(speech[:2], torch.zeros_like(speech[:2]))
        torch.testing.assert_close(event[2:], cue.expand(4, -1))
        torch.testing.assert_close(speech[2:], quote.expand(4, -1))
        self.assertEqual(len(report["semantic_role_expansions"]), 2)
        self.assertEqual(report["missing"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
