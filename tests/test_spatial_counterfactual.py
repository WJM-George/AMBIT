from __future__ import annotations

import copy

import numpy as np
import soundfile as sf
import torch

from stable_audio_tools.data.spatial_counterfactual import (
    swap_recipe_source_controls,
    swap_scene_plan_source_controls,
)
from stable_audio_tools.data.spatial_edit_recipe import (
    _resolved_playback_window,
    recipe_fingerprint,
)
from stable_audio_tools.data.spatial_story import (
    compile_source_tracks,
    source_tracks_to_mixture_trajectory,
)


def _plan() -> dict:
    def source(source_id: str, label: str, onset: float, offset: float, az: float, gain: float):
        return {
            "source_id": source_id,
            "event": {"category": "audio", "label": label},
            "content": {"transcript": None},
            "activity": {
                "onset_sec": onset,
                "offset_sec": offset,
                "quality": "paired_edit_exact",
            },
            "motion": {
                "type": "static",
                "time_basis": "full_clip",
                "keyframes": [
                    {
                        "t_norm": onset / 4.0,
                        "position": {
                            "azimuth_deg": az,
                            "elevation_deg": 0.0,
                            "distance_m": 1.5,
                            "geometry_quality": "exact",
                        },
                    }
                ],
            },
            "acoustics": {"gain_db": gain},
        }

    return {
        "audio": {"duration_sec": 4.0},
        "scene": {
            "sources": [
                source("source_0", "bell", 0.25, 1.25, -70.0, -3.0),
                source("source_1", "engine", 1.0, 3.5, 110.0, 2.0),
            ]
        },
    }


def _write_dry(path, *, sample_rate: int, frequency: float) -> dict:
    time = np.arange(4 * sample_rate, dtype=np.float32) / sample_rate
    audio = (0.2 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    stat = path.stat()
    return {
        "path": str(path),
        "native_sample_rate": sample_rate,
        "native_num_frames": len(audio),
        "crop_start_native_sample": 0,
        "crop_num_native_samples": len(audio),
        "file_size_bytes": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
    }


def _playback(dry: dict, *, duration: float) -> dict:
    return _resolved_playback_window(
        dry,
        activity_duration_sec=duration,
        preserve_source_start=False,
        source_loudness={
            "policy": "rms_target_peak_limited",
            "policy_version": 1,
            "target_mono_rms": 0.05,
            "peak_ceiling": 0.95,
            "max_gain_db": 30.0,
            "minimum_input_rms": 1.0e-4,
        },
    )


def _recipe(tmp_path) -> dict:
    plan = _plan()
    sources = []
    for index, plan_source in enumerate(plan["scene"]["sources"]):
        dry = _write_dry(
            tmp_path / f"source_{index}.wav",
            sample_rate=8_000,
            frequency=220.0 + 110.0 * index,
        )
        activity = copy.deepcopy(plan_source["activity"])
        duration = activity["offset_sec"] - activity["onset_sec"]
        sources.append(
            {
                "source_id": plan_source["source_id"],
                "source_uid": f"uid_{index}",
                "event": copy.deepcopy(plan_source["event"]),
                "content": copy.deepcopy(plan_source["content"]),
                "dry_audio": dry,
                "playback": _playback(dry, duration=duration),
                "activity": activity,
                "motion": {
                    **copy.deepcopy(plan_source["motion"]),
                    "time_basis": "activity_window",
                },
                "gain_db": plan_source["acoustics"]["gain_db"],
            }
        )
    recipe = {"sources": sources, "audio": {"sample_rate": 8_000}}
    recipe["recipe_fingerprint"] = recipe_fingerprint(recipe)
    return recipe


def test_scene_plan_control_swap_is_same_anchor_and_keeps_semantics():
    plan = _plan()
    swapped = swap_scene_plan_source_controls(plan, "source_0", "source_1")
    original_sources = plan["scene"]["sources"]
    swapped_sources = swapped["scene"]["sources"]

    assert swapped_sources[0]["event"] == original_sources[0]["event"]
    assert swapped_sources[1]["content"] == original_sources[1]["content"]
    assert swapped_sources[0]["activity"] == original_sources[1]["activity"]
    assert swapped_sources[1]["motion"] == original_sources[0]["motion"]
    assert swapped_sources[0]["acoustics"] == original_sources[1]["acoustics"]
    assert plan == _plan()  # input remains immutable

    tracks = compile_source_tracks(plan, num_frames=64)["tracks"]
    swapped_tracks = compile_source_tracks(swapped, num_frames=64)["tracks"]
    anchor = source_tracks_to_mixture_trajectory(
        tracks, fourth_component="received_level"
    )
    swapped_anchor = source_tracks_to_mixture_trajectory(
        swapped_tracks, fourth_component="received_level"
    )
    assert torch.equal(anchor, swapped_anchor)
    assert torch.equal(
        tracks.sort(dim=0).values, swapped_tracks.sort(dim=0).values
    )


def test_recipe_control_swap_refreshes_playback_and_keeps_dry_sources(tmp_path):
    recipe = _recipe(tmp_path)
    immutable_reference = copy.deepcopy(recipe)
    swapped = swap_recipe_source_controls(recipe, "source_0", "source_1")
    original = {source["source_id"]: source for source in recipe["sources"]}
    result = {source["source_id"]: source for source in swapped["sources"]}

    for source_id in original:
        assert result[source_id]["event"] == original[source_id]["event"]
        assert result[source_id]["content"] == original[source_id]["content"]
        assert result[source_id]["dry_audio"] == original[source_id]["dry_audio"]
    assert result["source_0"]["activity"] == original["source_1"]["activity"]
    assert result["source_1"]["motion"] == original["source_0"]["motion"]
    assert result["source_0"]["gain_db"] == original["source_1"]["gain_db"]
    assert result["source_0"]["playback"]["num_native_samples"] == 20_000
    assert result["source_1"]["playback"]["num_native_samples"] == 8_000
    assert swapped["recipe_fingerprint"] != recipe["recipe_fingerprint"]
    assert recipe == immutable_reference  # input remains immutable


def test_control_swap_rejects_missing_or_identical_source_ids(tmp_path):
    plan = _plan()
    recipe = _recipe(tmp_path)
    for source_a, source_b in (("source_0", "source_0"), ("source_0", "missing")):
        try:
            swap_scene_plan_source_controls(plan, source_a, source_b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ScenePlan pair was accepted")
        try:
            swap_recipe_source_controls(recipe, source_a, source_b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid recipe pair was accepted")

