import json

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_clap_text_cache import (
    load_native_semantic_cache, save_native_text_cache,
)


def bank(tmp_path):
    checkpoint = dict(path="/native/update-030000.pt", sha256="a" * 64,
                      step=30000, new_updates=30000, initialization_step=20000)
    raw = torch.randn(2, 1024, requires_grad=True)
    semantic = torch.nn.functional.normalize(torch.randn(2, 512, requires_grad=True), dim=-1)
    labels = ["engine", "speech"]
    path = save_native_text_cache(tmp_path / "bank", labels=labels, raw_features=raw,
        semantic_features=semantic, checkpoint=checkpoint, text_encoder_provenance={"frozen": True})
    return path, checkpoint, labels, semantic


def test_native_checkpoint_projection_roundtrip_stops_teacher_gradients(tmp_path):
    path, checkpoint, labels, semantic = bank(tmp_path)
    result = load_native_semantic_cache(path, checkpoint=checkpoint, expected_labels=labels, device="cpu")
    assert torch.equal(result["speech"], semantic[1:2].detach())
    assert not any(value.requires_grad for value in result.values())


def test_same_text_and_geometry_cannot_reuse_other_checkpoint_projection(tmp_path):
    path, checkpoint, labels, _ = bank(tmp_path)
    stale = dict(checkpoint, path="/native/update-020000.pt", sha256="b" * 64,
                 step=20000, new_updates=20000)
    with pytest.raises(ValueError, match="different CLAP checkpoint"):
        load_native_semantic_cache(path, checkpoint=stale, expected_labels=labels, device="cpu")


def test_native_cache_rejects_swapped_text_assignment(tmp_path):
    path, checkpoint, labels, _ = bank(tmp_path)
    with pytest.raises(ValueError, match="labels or order differ"):
        load_native_semantic_cache(path, checkpoint=checkpoint, expected_labels=labels[::-1], device="cpu")


def test_native_cache_rejects_changed_feature_payload(tmp_path):
    path, checkpoint, labels, _ = bank(tmp_path)
    feature_path = json.loads(path.read_text())["features"]["path"]
    with open(feature_path, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="feature file changed"):
        load_native_semantic_cache(path, checkpoint=checkpoint, expected_labels=labels, device="cpu")
