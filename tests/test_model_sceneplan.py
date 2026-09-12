import copy
import unittest

import numpy as np
import torch

from stable_audio_tools.data.model_sceneplan import (
    compile_model_44_controls,
    compile_model_renderer_caption,
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
    validate_model_sceneplan,
)
from stable_audio_tools.models.sceneplan_conditioning import ScenePlan44Conditioner


def _position(azimuth: float, distance: float = 1.2):
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": 0.0,
        "distance_m": distance,
    }


def _plan():
    return {
        "sample_id": "train_000001",
        "duration_sec": 10.0,
        "room": {"type": "moderate"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": "A passenger car engine accelerates with a bright whine.",
                "activity": {"onset_sec": 0.0, "offset_sec": 8.0},
                "trajectory": {
                    "type": "linear",
                    "start": _position(-90.0, 1.1),
                    "end": _position(90.0, 2.0),
                },
                "gain_db": -3.0,
            },
            {
                "source_id": "source_2",
                "kind": "speech",
                "speaker_description": "an English audiobook narrator",
                "transcript": 'She replied, "I understand."',
                "activity": {"onset_sec": 2.0, "offset_sec": 5.0},
                "trajectory": {
                    "type": "static",
                    "position": _position(45.0),
                },
                "gain_db": 0.0,
            },
        ],
    }


class ModelScenePlanTests(unittest.TestCase):
    def test_compact_plan_compiles_complete_caption_and_gain_control(self):
        plan = _plan()
        validate_model_sceneplan(plan)
        caption = compile_model_renderer_caption(plan)
        self.assertEqual(caption["compiler_version"], 5)
        self.assertTrue(
            caption["text"].startswith("In a moderately reverberant room, ")
        )
        self.assertIn('She replied, "I understand."', caption["text"])
        self.assertIn("mixed at -3.00 dB", caption["text"])
        self.assertEqual(len(caption["source_semantic_regions"]), 2)
        self.assertEqual(len(caption["source_motion_activity_regions"]), 2)
        self.assertEqual(len(caption["speaker_info_regions"]), 1)
        self.assertEqual(len(caption["transcript_regions"]), 1)
        transcript = caption["transcript_regions"][0]
        self.assertEqual(
            caption["text"][transcript["start"] : transcript["end"]],
            plan["sources"][1]["transcript"],
        )

        controls = compile_model_44_controls(plan)
        frames = int(np.ceil(10.0 * 44_100 / 1024))
        self.assertEqual(
            controls["source_present_mask"].tolist(), [1, 0, 1, 0]
        )
        self.assertEqual(
            controls["source_event_frame_ids"].shape, (4, frames)
        )
        features = controls["source_trajectory_features"]
        self.assertEqual(features.shape, (4, frames, 5))
        active = controls["source_event_frame_ids"][0].astype(bool)
        self.assertTrue(np.all(features[0, active, 4] > 0.0))
        self.assertFalse(controls["source_event_frame_ids"][1].any())
        self.assertFalse(controls["source_event_frame_ids"][3].any())

    def test_model_plan_rejects_renderer_lineage_and_unstable_sources(self):
        plan = _plan()
        plan["lineage"] = {"renderer_backend": "should_not_be_here"}
        with self.assertRaisesRegex(ValueError, "top-level"):
            validate_model_sceneplan(plan)

        duplicate = _plan()
        duplicate["sources"][1]["source_id"] = "source_0"
        with self.assertRaisesRegex(ValueError, "unique persistent IDs"):
            validate_model_sceneplan(duplicate)

        asset = _plan()
        asset["sources"][0]["asset_ref"] = {"asset_id": "not-model-state"}
        with self.assertRaisesRegex(ValueError, "fields do not match"):
            validate_model_sceneplan(asset)

    def test_model_plan_rejects_second_formal_speech_source(self):
        plan = _plan()
        second = copy.deepcopy(plan["sources"][1])
        second["source_id"] = "source_3"
        plan["sources"].append(second)
        with self.assertRaisesRegex(ValueError, "at most one formal speech"):
            validate_model_sceneplan(plan)

    def test_semantic_caption_is_separate_from_44_dit_input_concat(self):
        plan = _plan()
        caption = compile_model_semantic_caption_v2(plan)
        self.assertNotIn("active 0.00", caption["text"])
        self.assertNotIn("mixed at", caption["text"])
        controls = compile_model_44_controls(plan)
        features = controls["source_trajectory_features"]
        conditioner = ScenePlan44Conditioner(
            event_embedding_dim=4,
            trajectory_embedding_dim=4,
            output_dim=32,
        )
        result = conditioner(
            source_event_frame_ids=torch.as_tensor(
                controls["source_event_frame_ids"][None]
            ),
            source_trajectory_features=torch.as_tensor(features[None]),
        )
        self.assertEqual(
            result.shape,
            (1, 32, features.shape[1]),
        )
        self.assertTrue(torch.isfinite(result).all())

    def test_semantic_caption_v2_uses_explicit_roles_not_protocol_quotes(self):
        plan = _plan()
        plan["sources"][1]["transcript"] = "I understand completely."
        caption = compile_model_semantic_caption_v2(plan)
        self.assertEqual(caption["compiler_version"], 2)
        self.assertIn("who says: I understand completely.", caption["text"])
        self.assertNotIn('"I understand completely."', caption["text"])

        event = caption["event_regions"][1]
        speech = caption["speech_regions"][0]
        self.assertTrue(
            caption["text"][event["start"] : event["end"]].endswith("who says")
        )
        self.assertEqual(
            caption["text"][speech["start"] : speech["end"]],
            "I understand completely.",
        )
        self.assertLessEqual(event["end"], speech["start"])


if __name__ == "__main__":
    unittest.main()
