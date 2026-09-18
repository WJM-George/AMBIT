#!/usr/bin/env python3
"""CPU checks for Editing Transfusion route C. Does not import or launch OPSD."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.models.sceneplan_transfusion_editing_event_condition import (
    caption_roles_from_metadata,
    replace_editing_prompt_event_spans,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_event_head import (
    EditingEventHead,
    EventHeadConfig,
    gather_plan_states,
    inplace_caption_condition,
    plan_readout_positions,
    pool_qwen_event_targets,
)


class TinyCodec:
    pad_id, bos_id, eos_id = 0, 1, 2
    token_to_id = {
        "<source_begin>": 3,
        "<source_end>": 4,
        **{f"<source_slot_{i}>": 5 + i for i in range(4)},
        "<kind_speech>": 9,
        "<kind_sound>": 10,
        "<kind_music>": 11,
    }


PLAN = [1, 3, 7, 10, 20, 4, 3, 5, 11, 21, 4, 2]


class EventHeadChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)

    def test_source_end_identity(self):
        ids = torch.tensor([PLAN + [0, 0]])
        mask = ids != 0
        positions = plan_readout_positions(TinyCodec(), ids, mask)
        self.assertEqual(positions.tolist(), [[11, 10, -1, 5, -1]])
        hidden = torch.arange(ids.numel() * 16).reshape(1, -1, 16).float()
        values, valid = gather_plan_states(hidden, positions)
        torch.testing.assert_close(values[0, 3], hidden[0, 5])
        self.assertTrue(torch.equal(values[~valid], torch.zeros_like(values[~valid])))

    def test_inplace_replaces_event_spans_only(self):
        caption = torch.arange(8 * 6).reshape(1, 8, 6).float()
        caption_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]]).bool()
        event_ids = torch.tensor([[0, 1, 1, 0, 2, 2, 0, 0]])
        speech_ids = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0]])
        vectors = torch.zeros(1, 5, 6)
        vectors[0, 1] = 10
        vectors[0, 2] = 20
        mask = torch.tensor([[True, True, True, False, False]])
        values, valid = inplace_caption_condition(
            caption, caption_mask, event_ids, speech_ids, vectors, mask
        )
        torch.testing.assert_close(values[0, 0], caption[0, 0])
        torch.testing.assert_close(values[0, 1], vectors[0, 1])
        torch.testing.assert_close(values[0, 3], caption[0, 3])
        torch.testing.assert_close(values[0, 4], vectors[0, 2])
        self.assertEqual(float(values[0, 7].abs().sum()), 0.0)
        self.assertTrue(torch.equal(valid, caption_mask))

    def test_teacher_excludes_speech_from_event_pool(self):
        value = torch.arange(7 * 6).reshape(1, 7, 6).float()
        event = torch.tensor([[0, 1, 1, 0, 0, 0, 0]])
        speech = torch.tensor([[0, 0, 0, 1, 1, 1, 0]])
        valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0]]).bool()
        pooled, mask = pool_qwen_event_targets(value, valid, event, speech, allow_speech=True)
        torch.testing.assert_close(pooled[0, 0], value[0, 0])
        torch.testing.assert_close(pooled[0, 1], value[0, 1:3].mean(0))
        self.assertEqual(mask.tolist(), [[True, True, False, False, False]])

    def test_prompt_replacement_keeps_scene_and_speech(self):
        caption = torch.randn(1, 6, 8)
        caption_mask = torch.ones(1, 6).bool()
        event_ids = torch.tensor([[0, 1, 1, 0, 0, 0]])
        speech_ids = torch.tensor([[0, 0, 0, 1, 1, 0]])
        vectors = torch.zeros(1, 5, 8)
        vectors[0, 1] = 3
        mask = torch.tensor([[True, True, False, False, False]])
        metadata = [
            {
                "prompt": {
                    "attention_mask": caption_mask[0],
                    "event_source_ids": event_ids[0],
                    "speech_source_ids": speech_ids[0],
                }
            }
        ]
        roles = caption_roles_from_metadata(metadata, 6, caption.device)
        self.assertTrue(torch.equal(roles["event_source_ids"], event_ids))
        updated = replace_editing_prompt_event_spans(
            {"prompt": (caption, caption_mask), "sceneplan_44": torch.ones(1)},
            metadata,
            vectors,
            mask,
            max_events=4,
        )
        replaced, valid = updated["prompt"]
        torch.testing.assert_close(replaced[0, 0], caption[0, 0])
        torch.testing.assert_close(replaced[0, 1], vectors[0, 1])
        torch.testing.assert_close(replaced[0, 3], caption[0, 3])
        self.assertIn("sceneplan_44", updated)
        self.assertTrue(torch.equal(valid, caption_mask))

    def test_identity_adapter_at_init(self):
        head = EditingEventHead(EventHeadConfig(hidden_dim=16, condition_dim=6, head_dim=24, adapter_dim=8))
        values = torch.randn(2, 5, 6)
        mask = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]).bool()
        torch.testing.assert_close(head.adapt(values, mask), values.masked_fill(~mask[..., None], 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
