from __future__ import annotations

from scripts.t2a.eval.diagnostics.prepare_sceneplan_sound_transient_heldout import (
    _source_parent_key,
)


def test_vggsound_segments_share_parent_recording_key() -> None:
    left = "sound:vggsound:vggsound_IvQtJCFTSlY_000352"
    right = "sound:vggsound:vggsound_IvQtJCFTSlY_000417"
    assert _source_parent_key(left) == _source_parent_key(right)
    assert _source_parent_key(left) == "sound:vggsound:vggsound_IvQtJCFTSlY"


def test_nonsegmented_assets_keep_exact_parent_key() -> None:
    for asset_id in (
        "sound:audioset:audioset_2p8bLH6fbUM",
        "sound:audiocaps:audiocaps_85797",
        "sound:fsd50k:fsd50k_dev_398706",
    ):
        assert _source_parent_key(asset_id) == asset_id

