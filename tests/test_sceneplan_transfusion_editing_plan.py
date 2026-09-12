from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from stable_audio_tools.data.model_sceneplan_codec_v4 import (
    ModelScenePlanCodecV4,
    create_model_sceneplan_codec_v4_artifact,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
from stable_audio_tools.data.sceneplan_transfusion_editing_ar_dataset import (
    EDITING_AR_SELECT_COLUMNS,
    ScenePlanTransfusionEditingARDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (
    ScenePlanTransfusionEditingDataset,
    compile_editing_dit_plan_condition,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import (
    ScenePlanTransfusionEditingJointDataset,
    collate_editing_joint,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_plan import (
    canonicalize_editing_plan,
    editing_ar_allowed_next_ids,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (
    ScenePlanTransfusionEditingAR,
)


@pytest.fixture(scope="module")
def codec(tmp_path_factory):
    root = tmp_path_factory.mktemp("editing-plan-codec") / "codec"
    create_model_sceneplan_codec_v4_artifact(root)
    return ModelScenePlanCodecV4(root)


def plan():
    def sound(slot, description, onset, azimuth):
        return {
            "source_id": f"source_{slot}", "kind": "sound",
            "description": description, "gain_db": 0.0,
            "activity": {"onset_sec": onset, "offset_sec": 1.8},
            "trajectory": {"type": "static", "position": {
                "azimuth_deg": azimuth, "elevation_deg": 0.0, "distance_m": 2.0,
            }},
        }
    return {
        "sample_id": "target", "duration_sec": 2.0, "room": {"type": "moderate"},
        "sources": [sound(1, "A dog barking.", 0.9, 90.0),
                    sound(3, "Rain falling.", 0.1, -90.0)],
    }


@pytest.mark.parametrize("slots", list(itertools.permutations(range(4), 2)))
def test_random_persistent_slots_produce_identical_targets(codec, slots):
    original = plan()
    expected, _ = canonicalize_editing_plan(original, codec=codec)
    renamed = copy.deepcopy(original)
    for source, slot in zip(renamed["sources"], slots):
        source["source_id"] = f"source_{slot}"
    renamed["sources"].sort(key=lambda source: source["source_id"])
    observed, _ = canonicalize_editing_plan(renamed, codec=codec)
    assert observed == expected
    assert torch.equal(codec.encode(observed)["input_ids"], codec.encode(expected)["input_ids"])
    assert original == plan()
    assert canonicalize_editing_plan(observed, codec=codec)[0] == observed


def test_order_uses_only_codec_representable_fields(codec):
    original = plan()
    original["sources"][0]["activity"]["onset_sec"] = 0.90001
    original["sources"][1]["activity"]["onset_sec"] = 0.90002
    canonical, _ = canonicalize_editing_plan(original, codec=codec)
    projected, _ = canonicalize_editing_plan(codec.project_plan(original), codec=codec)
    assert torch.equal(codec.encode(canonical)["input_ids"], codec.encode(projected)["input_ids"])
    # Reordering must preserve original, unquantized acoustic controls.
    before = {source["description"]: source for source in original["sources"]}
    for source in canonical["sources"]:
        expected = dict(before[source["description"]], source_id=source["source_id"])
        assert source == expected


class CharacterTokenizer:
    def __call__(self, text, *, padding, max_length=None, **kwargs):
        ids = [ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        attention = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids += [0] * pad
            offsets += [(0, 0)] * pad
            attention += [0] * pad
        return {"input_ids": ids, "attention_mask": attention, "offset_mapping": offsets}


class BaseDataset(ScenePlanTransfusionEditingDataset):
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.caption_max_tokens = 512
        self.source = torch.arange(64 * 648, dtype=torch.float32).reshape(64, 648).half()
        self.target = self.source.clone()
        mask = torch.arange(648) < 87
        original = plan()
        self.metadata = {
            "pair_id": "pair", "pair_ordinal": 0, "operation": "event_remove",
            "raw_edit_request": "Remove the bell.",
            "editing_ar_target_model_sceneplan": original,
            "source_foa_latent": self.source, "padding_mask": [mask], "audio": self.target,
            "latent_frames_valid": 87, "latent_bucket_frames": 432,
            "latent_crop_length": 648, "model_num_samples": 88200,
            "seconds_total": 2.0, "seconds_start": 0.0,
            "edited_source_ids": ("source_0",),
            "unchanged_source_ids": ("source_1", "source_3"),
            **compile_editing_dit_plan_condition(
                original, tokenizer=self.tokenizer, model_num_samples=88200,
                latent_frames_valid=87, latent_crop_length=648,
            ),
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.target, self.metadata


def test_joint_labels_text_and_frame_controls_follow_the_same_sources(codec):
    base = BaseDataset()
    original = copy.deepcopy(base.metadata["model_sceneplan"])
    target, metadata, ar_row = ScenePlanTransfusionEditingJointDataset(base, codec=codec)[0]
    canonical, mapping = canonicalize_editing_plan(original, codec=codec)
    assert target is base.target
    assert metadata["source_foa_latent"] is base.source
    assert base.metadata["model_sceneplan"] == original
    assert "editing_persistent_target_sceneplan" not in base.metadata
    assert metadata["editing_persistent_target_sceneplan"] == original
    assert metadata["model_sceneplan"] == metadata["editing_ar_target_model_sceneplan"] == canonical
    assert torch.equal(ar_row["target_token_ids"], codec.encode(canonical)["input_ids"])
    assert metadata["edited_source_ids"] == ()  # removed old slot 0 is not new slot 0
    assert metadata["editing_persistent_edited_source_ids"] == ("source_0",)
    assert set(metadata["unchanged_source_ids"]) == {"source_0", "source_1"}
    assert metadata["editing_model_source_id_map"] == mapping
    expected = compile_editing_dit_plan_condition(
        canonical, tokenizer=base.tokenizer, model_num_samples=88200,
        latent_frames_valid=87, latent_crop_length=648,
    )
    assert metadata["prompt_text"] == expected["prompt_text"]
    for section in ("prompt", "sceneplan_44"):
        for key, value in expected[section].items():
            assert torch.equal(metadata[section][key], value)
    for old, new in mapping.items():
        old_slot, new_slot = int(old[-1]), int(new[-1])
        assert torch.equal(
            metadata["sceneplan_44"]["source_trajectory_features"][new_slot],
            base.metadata["sceneplan_44"]["source_trajectory_features"][old_slot],
        )
        original_active = base.metadata["sceneplan_44"]["source_event_frame_ids"][old_slot].ne(0)
        assert torch.equal(
            metadata["sceneplan_44"]["source_event_frame_ids"][new_slot],
            original_active.to(torch.int8) * (new_slot + 1),
        )
    batch = collate_editing_joint([(target, metadata, ar_row)], pad_id=codec.pad_id)
    assert batch["ar"]["source_foa_latent"].shape == (1, 64, 432)


def test_ar_only_and_joint_readers_encode_identical_targets(codec, tmp_path):
    base = BaseDataset()
    source_file = tmp_path / "source.safetensors"
    save_file({"source": base.source[:, :87].contiguous()}, str(source_file))
    values = {
        "pair_ordinal": 0, "pair_id": "pair", "operation": "event_remove",
        "raw_edit_request": "Remove the bell.",
        "new_sceneplan_zlib": zlib.compress(json.dumps(plan()).encode()),
        "new_sceneplan_sha256": sha256_json(plan()),
        "source_latent_path": str(source_file), "source_latent_key": "source",
        "source_latent_tensor_sha256": "unused", "latent_frames_valid": 87,
        "latent_bucket_frames": 432,
    }
    reader = object.__new__(ScenePlanTransfusionEditingARDataset)
    reader._length, reader.row_ordinals = 1, None
    reader.codec, reader.max_plan_tokens, reader.latent_crop_length = codec, 1024, 648
    reader.verify_tensor_hashes_on_access = False
    reader._db = lambda: SimpleNamespace(execute=lambda *args: SimpleNamespace(
        fetchone=lambda: tuple(values[key] for key in EDITING_AR_SELECT_COLUMNS)
    ))
    observed = reader[0]
    expected = ScenePlanTransfusionEditingJointDataset(base, codec=codec)[0][2]
    assert torch.equal(observed["target_token_ids"], expected["target_token_ids"])
    assert "editing_model_source_id_map" not in observed


def test_free_decode_cannot_choose_random_slots_even_when_logits_prefer_them(codec):
    canonical, _ = canonicalize_editing_plan(plan(), codec=codec)
    target = codec.encode(canonical)["input_ids"].tolist()
    random_slot = codec.token_to_id["<source_slot_3>"]

    class StubAR(ScenePlanTransfusionEditingAR):
        def __init__(self):
            nn.Module.__init__(self)
            self.pad_id = codec.pad_id

        def encode_edit_instructions(self, instructions, *, device):
            return torch.zeros(1, 1, 1024), torch.ones(1, 1, dtype=torch.bool)

        def forward(self, source, source_mask, plan_ids, plan_mask, context, context_mask):
            length = plan_ids.shape[1]
            logits = torch.zeros(1, length, 4096)
            logits[0, -1, target[length]] = 5.0
            logits[0, -1, random_slot] = 100.0
            return logits

    generated = StubAR().generate_batch(
        torch.zeros(1, 64, 3), torch.ones(1, 3, dtype=torch.bool), ["edit"],
        codec=codec, max_plan_tokens=512,
    )[0]
    assert generated.tolist() == target
    for index in range(1, len(target)):
        assert target[index] in editing_ar_allowed_next_ids(codec, target[:index])


def test_audio_evaluation_compares_plans_in_the_ar_output_vocabulary(codec):
    from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_audio_end_to_end import (
        _plan_metrics,
    )

    original = plan()
    original["sources"][0]["gain_db"] = 0.37
    canonical, _ = canonicalize_editing_plan(original, codec=codec)
    tokens = codec.encode(canonical)["input_ids"].tolist()
    decoded = codec.decode(tokens, sample_id=canonical["sample_id"])
    assert decoded != canonical  # continuous truth need not lie on codec bins
    metrics = _plan_metrics(
        codec=codec, target_plan=canonical, predicted_plan=decoded,
        target_tokens=tokens, predicted_tokens=tokens,
    )
    for field in ("activity_onset_exact", "activity_offset_exact", "gain_exact"):
        assert metrics[f"plan_ordered_{field}"] == 1.0

    incorrect = copy.deepcopy(decoded)
    incorrect["sources"][0]["gain_db"] += 2.0
    incorrect_tokens = codec.encode(incorrect)["input_ids"].tolist()
    metrics = _plan_metrics(
        codec=codec, target_plan=canonical, predicted_plan=incorrect,
        target_tokens=tokens, predicted_tokens=incorrect_tokens,
    )
    assert metrics["plan_ordered_gain_exact"] == 0.5
