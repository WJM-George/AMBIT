from __future__ import annotations

import inspect
import math

import torch

from scripts.t2a.eval import sceneplan_transfusion_editing_content_metrics as content


def _source(source_id: str, description: str) -> dict:
    return {
        "source_id": source_id,
        "kind": "sound",
        "description": description,
        "activity": {"onset_sec": 0.0, "offset_sec": 1.0},
        "trajectory": {
            "type": "static",
            "position": {
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "distance_m": 1.0,
            },
        },
        "gain_db": 0.0,
    }


def test_edit_caption_uses_removed_old_source_and_added_new_source() -> None:
    old = {"sources": [_source("source_0", "old bell")]}
    new = {"sources": [_source("source_1", "new dog bark")]}
    assert (
        content.edited_source_caption(old, new, ["source_0"], "event_removal")
        == "old bell"
    )
    assert (
        content.edited_source_caption(old, new, ["source_1"], "event_addition")
        == "new dog bark"
    )


def test_clap_views_cover_both_ends_without_padding_long_audio() -> None:
    samples = 12 * 44_100
    audio = torch.zeros(4, samples)
    audio[0, : samples // 2] = torch.linspace(-1.0, 0.25, samples // 2)
    audio[0, samples // 2 :] = torch.linspace(0.5, 1.0, samples - samples // 2)
    views = content.clap_full_duration_views(audio)
    assert tuple(views.shape) == (2, content.CLAP_WINDOW_SAMPLES)
    assert torch.isfinite(views).all()
    assert not torch.equal(views[0], views[1])


def test_short_clap_view_is_padded_to_exact_window() -> None:
    audio = torch.randn(4, 44_100)
    views = content.clap_full_duration_views(audio)
    assert tuple(views.shape) == (1, content.CLAP_WINDOW_SAMPLES)
    assert math.isclose(float(views[0, -1]), 0.0, abs_tol=1e-8)


def test_removed_speech_recall_is_bounded_and_duplicate_aware() -> None:
    reference = "go go now"
    assert content._token_recall("go now", reference) == 2 / 3
    assert content._token_recall("go go go now", reference) == 1.0
    assert content._token_recall("silence", reference) == 0.0


def test_whisper_model_bin_is_bound_by_exact_sha256_not_only_size() -> None:
    assert content.WHISPER_MODEL_BIN_SHA256 == (
        "b79368e19b6623813609431a6e5ee309a71506701ebc49fd7820e692dec7c5f5"
    )
    source = inspect.getsource(content.verify_independent_content_metric_assets)
    assert '"model.bin": WHISPER_MODEL_BIN_SHA256' in source
    assert "if observed != expected" in source
    assert "model size changed" in source
