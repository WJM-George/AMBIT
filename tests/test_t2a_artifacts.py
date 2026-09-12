from __future__ import annotations

import copy
import json
import pickle
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from stable_audio_tools.data.scene_plan import (
    DSL_VERSION,
    crop_scene_plan,
    serialize_spatial_dsl,
)
from stable_audio_tools.data.spatial_plan_codec import (
    CodecError,
    SpatialPlanCodec,
    create_codec_artifact,
)
from stable_audio_tools.data.spatial_conversation_metadata import (
    SpatialConversationMetadata,
)
from stable_audio_tools.data.t2a_artifacts import (
    IndexedJsonlStore,
    ShardedTensorStore,
    normalize_audio_path,
    stable_sample_id,
)
from scripts.t2a.data.upgrade_scene_plan_v1_1 import upgrade_record


def _touch_ready(root: Path) -> None:
    (root / "READY").write_text("{}\n", encoding="utf-8")


class ArtifactIdentityTests(unittest.TestCase):
    def test_sample_id_uses_normalized_absolute_audio_path(self):
        first = "/tmp/t2a/a/../source.flac"
        second = "/tmp/t2a/source.flac"
        self.assertEqual(normalize_audio_path(first), second)
        self.assertEqual(stable_sample_id(first), stable_sample_id(second))
        self.assertTrue(stable_sample_id(first).startswith("t2a_"))


class ShardedTensorStoreTests(unittest.TestCase):
    def test_reads_tensor_and_reopens_after_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shards").mkdir()
            audio_path = normalize_audio_path(root / "source.flac")
            sample_id = stable_sample_id(audio_path)
            expected = torch.arange(12, dtype=torch.float16).reshape(3, 4)
            save_file({sample_id: expected}, str(root / "shards" / "traj.safetensors"))

            connection = sqlite3.connect(root / "index.sqlite")
            connection.execute(
                "CREATE TABLE samples ("
                "sample_id TEXT PRIMARY KEY, audio_path TEXT NOT NULL UNIQUE, "
                "latent_relpath TEXT NOT NULL UNIQUE, shard TEXT NOT NULL, "
                "tensor_key TEXT NOT NULL, num_frames INTEGER NOT NULL, "
                "channels INTEGER NOT NULL, dtype TEXT NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    audio_path,
                    "0/sample.npy",
                    "shards/traj.safetensors",
                    sample_id,
                    3,
                    4,
                    "float16",
                ),
            )
            connection.commit()
            connection.close()
            _touch_ready(root)

            store = ShardedTensorStore(root)
            found, entry = store.get_tensor(
                audio_path, expected_frames=3, expected_channels=4
            )
            self.assertTrue(torch.equal(found, expected))
            self.assertEqual(entry["sample_id"], sample_id)

            reopened = pickle.loads(pickle.dumps(store))
            found_again, _ = reopened.get_tensor(audio_path)
            self.assertTrue(torch.equal(found_again, expected))
            store.close()
            reopened.close()


class IndexedJsonlStoreTests(unittest.TestCase):
    def test_reads_exact_record_and_reopens_after_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shards").mkdir()
            audio_path = normalize_audio_path(root / "source.flac")
            sample_id = stable_sample_id(audio_path)
            record = {
                "sample_id": sample_id,
                "audio": {"path": audio_path},
                "caption": "A moving source.",
            }
            payload = json.dumps(
                record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            (root / "shards" / "records.jsonl").write_bytes(payload + b"\n")

            connection = sqlite3.connect(root / "index.sqlite")
            connection.execute(
                "CREATE TABLE samples ("
                "sample_id TEXT PRIMARY KEY, audio_path TEXT NOT NULL UNIQUE, "
                "shard TEXT NOT NULL, byte_offset INTEGER NOT NULL, "
                "byte_length INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                (sample_id, audio_path, "shards/records.jsonl", 0, len(payload)),
            )
            connection.commit()
            connection.close()
            _touch_ready(root)

            store = IndexedJsonlStore(root)
            self.assertEqual(store.get(audio_path), record)
            reopened = pickle.loads(pickle.dumps(store))
            self.assertEqual(reopened.get(audio_path), record)
            store.close()
            reopened.close()


class ScenePlanCropTests(unittest.TestCase):
    def test_dynamic_crop_uses_shortest_azimuth_arc(self):
        plan = {
            "audio": {"format": "foa", "duration_sec": 10.0},
            "room": {},
            "scene": {
                "sources": [
                    {
                        "source_id": "s0",
                        "event": {"description": "moving source"},
                        "position": {},
                        "motion": {
                            "type": "linear",
                            "keyframes": [
                                {
                                    "t_norm": 0.0,
                                    "position": {
                                        "azimuth_deg": 170.0,
                                        "elevation_deg": 0.0,
                                        "distance_m": 1.0,
                                    },
                                },
                                {
                                    "t_norm": 1.0,
                                    "position": {
                                        "azimuth_deg": -170.0,
                                        "elevation_deg": 20.0,
                                        "distance_m": 3.0,
                                    },
                                },
                            ],
                        },
                    }
                ]
            },
        }
        cropped = crop_scene_plan(plan, [0.25, 0.75])
        keyframes = cropped["scene"]["sources"][0]["motion"]["keyframes"]
        self.assertAlmostEqual(keyframes[0]["position"]["azimuth_deg"], 175.0)
        self.assertAlmostEqual(keyframes[1]["position"]["azimuth_deg"], -175.0)
        self.assertAlmostEqual(cropped["audio"]["duration_sec"], 5.0)
        self.assertEqual(
            cropped["training_crop"]["source_timestamps"], [0.25, 0.75]
        )
        self.assertTrue(serialize_spatial_dsl(cropped).startswith(f"scene<{DSL_VERSION}>"))

    def test_keyframed_crop_and_activity_are_projected(self):
        plan = {
            "audio": {"duration_sec": 10.0},
            "scene": {
                "sources": [
                    {
                        "activity": {
                            "onset_sec": 1.0,
                            "offset_sec": 9.0,
                            "quality": "source_annotation",
                            "render_window_sec": [0.0, 10.0],
                        },
                        "motion": {
                            "type": "keyframed",
                            "keyframes": [
                                {"t_norm": 0.0, "position": {"azimuth_deg": -90.0, "elevation_deg": 0.0, "distance_m": 1.0}},
                                {"t_norm": 0.5, "position": {"azimuth_deg": 0.0, "elevation_deg": 10.0, "distance_m": 2.0}},
                                {"t_norm": 1.0, "position": {"azimuth_deg": 90.0, "elevation_deg": 20.0, "distance_m": 3.0}},
                            ],
                        },
                    }
                ]
            },
        }
        cropped = crop_scene_plan(plan, [0.25, 0.75])
        source = cropped["scene"]["sources"][0]
        self.assertEqual(source["motion"]["type"], "keyframed")
        self.assertEqual([frame["t_norm"] for frame in source["motion"]["keyframes"]], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(source["motion"]["keyframes"][0]["position"]["azimuth_deg"], -45.0)
        self.assertAlmostEqual(source["motion"]["keyframes"][-1]["position"]["azimuth_deg"], 45.0)
        self.assertEqual(source["activity"]["onset_sec"], 0.0)
        self.assertEqual(source["activity"]["offset_sec"], 5.0)
        self.assertEqual(source["activity"]["render_window_sec"], [0.0, 5.0])


class SpatialPlanCodecTests(unittest.TestCase):
    def test_edit_token_mask_is_sparse_and_marks_deletion_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "music", "studio", "single"] * 32,
                text_vocab_size=512,
            )
            codec = SpatialPlanCodec(artifact)
            previous = {
                "audio": {"duration_sec": 10.0},
                "mix": {"type": "single"},
                "scene": {
                    "room": {"type": "studio"},
                    "sources": [
                        {
                            "source_id": "source_0",
                            "event": {"label": "speech"},
                            "acoustics": {"gain_db": -2.5},
                            "motion": {"type": "static"},
                        },
                        {
                            "source_id": "source_1",
                            "event": {"label": "music"},
                            "acoustics": {"gain_db": -1.0},
                            "motion": {"type": "static"},
                        },
                    ],
                },
            }
            gain_edit = copy.deepcopy(previous)
            gain_edit["scene"]["sources"][0]["acoustics"]["gain_db"] = -8.5
            previous_tokens = codec.encode(previous)
            gain_tokens = codec.encode(gain_edit)
            gain_mask = codec.edit_token_mask(previous_tokens, gain_tokens)
            self.assertEqual(gain_mask.dtype, torch.bool)
            self.assertEqual(gain_mask.numel(), gain_tokens["input_ids"].numel())
            self.assertEqual(int(gain_mask.sum()), 1)

            removal = copy.deepcopy(previous)
            removal["scene"]["sources"].pop()
            removal_tokens = codec.encode(removal)
            removal_mask = codec.edit_token_mask(previous_tokens, removal_tokens)
            self.assertGreater(int(removal_mask.sum()), 0)
            self.assertEqual(removal_mask.numel(), removal_tokens["input_ids"].numel())

    def test_roundtrip_and_every_prefix_is_accepted_by_fsm(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = create_codec_artifact(
                Path(directory) / "codec",
                ["speech", "office", "My Lord, 世界!", "front-left"] * 16,
                text_vocab_size=512,
            )
            codec = SpatialPlanCodec(artifact)
            plan = {
                "audio": {"duration_sec": 10.0},
                "mix": {"type": "single"},
                "scene": {
                    "room": {
                        "type": "office",
                        "description": "an office room",
                        "rt60_s": 0.42,
                        "dimensions_m": [4.0, 5.0, 3.0],
                        "quality": "exact",
                    },
                    "sources": [
                        {
                            "source_id": "s0",
                            "event": {"label": "speech", "category": "speech"},
                            "content": {"transcript": "My Lord, 世界!", "speaker_id": "9017"},
                            "activity": {"onset_sec": None, "offset_sec": None, "quality": "not_annotated"},
                            "motion": {
                                "type": "linear",
                                "timing_quality": "renderer_defined",
                                "keyframes": [
                                    {"t_norm": 0.0, "position": {"azimuth_deg": -45.0, "elevation_deg": 0.0, "distance_m": 3.7, "direction": "front-left", "geometry_quality": "exact"}},
                                    {"t_norm": 1.0, "position": {"azimuth_deg": 45.0, "elevation_deg": 10.0, "distance_m": 2.0, "direction": "front-right", "geometry_quality": "exact"}},
                                ],
                            },
                        }
                    ],
                },
            }
            encoded = codec.encode(plan, max_tokens=1024)
            token_ids = encoded["input_ids"]
            decoded = codec.decode(token_ids)
            self.assertEqual(
                decoded["scene"]["sources"][0]["content"]["transcript"],
                "My Lord, 世界!",
            )
            self.assertEqual(decoded["scene"]["sources"][0]["event"]["label"], "speech")
            self.assertAlmostEqual(
                decoded["scene"]["sources"][0]["motion"]["keyframes"][0]["position"]["azimuth_deg"],
                -45.0,
            )
            for index, token_id in enumerate(token_ids.tolist()):
                self.assertIn(token_id, codec.allowed_next_ids(token_ids[:index]))
            self.assertEqual(codec.allowed_next_ids(token_ids), set())

    def test_fixed_duration_and_noncanonical_text_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = create_codec_artifact(
                Path(directory) / "codec",
                ["toco", "studio", "single"] * 64,
                text_vocab_size=512,
            )
            codec = SpatialPlanCodec(artifact)
            plan = {
                "audio": {"duration_sec": 10.031020408163266},
                "mix": {"type": "toco"},
                "scene": {"room": {"type": "studio"}, "sources": []},
            }
            encoded = codec.encode(plan, max_tokens=1024)["input_ids"]
            duration_prefix = encoded[:3]
            self.assertEqual(
                codec.allowed_next_ids(
                    duration_prefix,
                    fixed_duration_sec=10.031020408163266,
                ),
                {int(encoded[3])},
            )
            wrong_duration = codec._quantize("seconds", 1.0)
            self.assertNotEqual(wrong_duration, int(encoded[3]))
            with self.assertRaises(CodecError):
                codec.allowed_next_ids(
                    [*duration_prefix.tolist(), wrong_duration],
                    fixed_duration_sec=10.031020408163266,
                )

            values = encoded.tolist()
            mix_index = values.index(codec.token_to_id["<mix_type>"])
            text_begin = mix_index + 1
            self.assertEqual(values[text_begin], codec.token_to_id["<text_begin>"])
            text_end = values.index(codec.token_to_id["<text_end>"], text_begin + 1)
            byte_pieces = [
                codec.text_processor.piece_to_id(f"<0x{value:02X}>")
                for value in b"toco"
            ]
            self.assertTrue(all(piece >= 0 for piece in byte_pieces))
            noncanonical = torch.tensor(
                [
                    *values[: text_begin + 1],
                    *(codec.text_offset + piece for piece in byte_pieces),
                    *values[text_end:],
                ],
                dtype=torch.long,
            )
            self.assertFalse(torch.equal(noncanonical, encoded))
            self.assertEqual(codec.decode(noncanonical), codec.decode(encoded))
            canonical = codec.canonicalize(noncanonical, max_tokens=1024)[
                "input_ids"
            ]
            self.assertTrue(torch.equal(canonical, encoded))


class ScenePlanMigrationTests(unittest.TestCase):
    def test_tts_label_is_speech_and_renderer_extent_is_not_strong_activity(self):
        record = {
            "schema": "stable_audio_tools.spatial_scene_plan",
            "schema_version": 1,
            "dataset_id": "spatial_speech_tts_sdb",
            "audio": {"duration_sec": 4.8},
            "scene": {
                "sources": [
                    {
                        "event": {"label": "My Lord the Duke", "category": "speech"},
                        "content": {"transcript": "My Lord the Duke"},
                        "activity": {"onset_sec": None, "offset_sec": None, "quality": "unknown"},
                        "motion": {"type": "static", "keyframes": [{"t_norm": 0.0, "position": {}}]},
                    }
                ]
            },
            "supervision": {"source_activity_time": False, "motion": False},
            "provenance": {},
        }
        upgraded = upgrade_record(record)
        source = upgraded["scene"]["sources"][0]
        self.assertEqual(upgraded["schema_version"], "1.1")
        self.assertEqual(source["event"]["label"], "speech")
        self.assertEqual(source["content"]["transcript"], "My Lord the Duke")
        self.assertIsNone(source["activity"]["onset_sec"])
        self.assertEqual(source["activity"]["quality"], "not_annotated")
        self.assertEqual(source["activity"]["render_window_sec"], [0.0, 4.8])
        self.assertFalse(upgraded["supervision"]["source_activity_time"])


class SpatialConversationPromptTests(unittest.TestCase):
    def test_room_metrics_are_inserted_before_source_details(self):
        prompt = "Room: a treated studio. source_0: Bird at the front."
        plan = {
            "scene": {
                "room": {
                    "rt60_s": 0.325,
                    "dimensions_m": [9.7, 11.8, 4.2],
                }
            }
        }
        enriched = SpatialConversationMetadata._planner_prompt_with_room_metrics(
            prompt, plan
        )
        self.assertEqual(
            enriched,
            "Room: a treated studio; RT60=0.325s; "
            "dimensions=9.7x11.8x4.2m. source_0: Bird at the front.",
        )


if __name__ == "__main__":
    unittest.main()
