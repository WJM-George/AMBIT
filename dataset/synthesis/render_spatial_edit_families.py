#!/usr/bin/env python3
"""Render deterministic Spatial-CoT edit families into a new FOA branch.

The input is produced by
``scripts/t2a/data/materialize_spatial_cot_recipe_shard.py``. Every
unique source render signature is cached once as a PCM24 FLAC pre-mix track.
Unchanged sources therefore decode from the exact same cached file in adjacent
turns.  Each turn is independently summed from those dry-source-derived tracks;
the previous turn's FOA is never an audio input.  All turn mixtures share one
family-level master gain.

This script never writes beside or over the reference FOA.  ``--output-root``
must name a separate branch on the selected storage volume.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SYNTHESIS_ROOT = Path(__file__).resolve().parent
if str(SYNTHESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNTHESIS_ROOT))

from synthesize_foa_pyroom import (  # noqa: E402
    az_el_dist_to_xyz,
    room_rirs,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    FOA_LAYOUT,
    RECIPE_SCHEMA,
    STATE_RENDER_INPUT,
    source_render_signature,
    unique_source_render_jobs,
    validate_edit_family,
)
from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


OUTPUT_MARKER = ".spatial_cot_render_root.json"
READY_MARKER = "READY"
RETAINED_PROFILE = "retained_flac_pcm24"
TRANSIENT_PROFILE = "transient_flac_pcm24"
STORAGE_PROFILES = (RETAINED_PROFILE, TRANSIENT_PROFILE)


def _prepare_output_root(output_root: Path, storage_profile: str) -> None:
    """Claim only an empty or already-recognized Spatial-CoT render root."""

    if output_root.exists():
        names = {path.name for path in output_root.iterdir()}
        recognized = {
            name
            for name in names
            if name == OUTPUT_MARKER
            or name == READY_MARKER
            or name == "families"
            or name.startswith("render_manifest.shard")
        }
        if names - recognized:
            raise SystemExit(
                "output-root contains non-Spatial-CoT data and will not be claimed: "
                f"{output_root}"
            )
        marker = output_root / OUTPUT_MARKER
        if marker.is_file():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            existing_profile = str(
                existing.get("storage_profile") or RETAINED_PROFILE
            )
            if existing_profile != storage_profile:
                raise SystemExit(
                    "render storage profile changed for existing output-root: "
                    f"{existing_profile} != {storage_profile}"
                )
    else:
        output_root.mkdir(parents=True)
    atomic_write_json(
        output_root / OUTPUT_MARKER,
        {
            "schema": "stable_audio_tools.spatial_cot_render_root",
            "schema_version": 1,
            "output_root": str(output_root),
            "storage_profile": storage_profile,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_families(root: Path) -> Iterator[dict[str, Any]]:
    shards = sorted((root / "shards").glob("*.jsonl")) if root.is_dir() else [root]
    if not shards:
        raise FileNotFoundError(f"no recipe JSONL found under {root}")
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _atomic_write_pcm24(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        sf.write(str(temporary), audio.T, sample_rate, subtype="PCM_24")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _flac_pcm24_memory_roundtrip(
    audio: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Apply the retained FLAC/PCM24 quantization without filesystem I/O."""

    buffer = io.BytesIO()
    sf.write(
        buffer,
        audio.T,
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )
    buffer.seek(0)
    decoded, decoded_rate = sf.read(
        buffer, always_2d=True, dtype="float32"
    )
    if int(decoded_rate) != int(sample_rate) or decoded.shape[1] != 4:
        raise RuntimeError("in-memory PCM24 track round-trip changed layout")
    return decoded.T.astype(np.float32, copy=False)


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return signal.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(
        signal,
        int(target_rate) // divisor,
        int(source_rate) // divisor,
    ).astype(np.float32)


def _load_activity_signal(
    source: dict[str, Any],
    *,
    sample_rate: int,
    num_samples: int,
) -> np.ndarray:
    dry = source["dry_audio"]
    playback = source["playback"]
    absolute_start = int(dry["crop_start_native_sample"]) + int(
        playback["offset_in_crop_native_sample"]
    )
    audio, native_rate = sf.read(
        dry["path"],
        start=absolute_start,
        stop=absolute_start + int(playback["num_native_samples"]),
        always_2d=True,
        dtype="float32",
    )
    if audio.shape[1] != 1:
        raise RuntimeError(
            "ScenePlan renderer accepts canonical dry mono only; refusing "
            f"{audio.shape[1]}-channel source {dry['path']}"
        )
    if dry.get("input_audio_domain") not in (None, "dry_mono"):
        raise RuntimeError(
            f"source is not declared dry_mono: {dry.get('input_audio_domain')!r}"
        )
    if int(dry.get("spatialization_passes_before_scene", 0)) != 0:
        raise RuntimeError("refusing to spatialize an already-spatialized source")
    if int(native_rate) != int(dry["native_sample_rate"]):
        raise RuntimeError(
            f"dry source sample-rate changed: {native_rate} != {dry['native_sample_rate']}"
        )
    if len(audio) != int(playback["num_native_samples"]):
        raise RuntimeError(
            "dry playback window changed: "
            f"{len(audio)} != {playback['num_native_samples']}"
        )
    mono = audio[:, 0].astype(np.float32, copy=False)
    mono = (
        mono * float(playback["loudness"]["gain_linear"])
    ).astype(np.float32, copy=False)
    mono = _resample(mono, int(native_rate), int(sample_rate))
    activity = source["activity"]
    start = max(0, min(num_samples, int(round(float(activity["onset_sec"]) * sample_rate))))
    stop = max(start, min(num_samples, int(round(float(activity["offset_sec"]) * sample_rate))))
    signal = np.zeros(num_samples, dtype=np.float32)
    available = stop - start
    if len(mono) > available:
        raise RuntimeError(
            "complete dry source does not fit its declared activity window; "
            f"refusing silent truncation ({len(mono)} > {available} samples)"
        )
    if len(mono):
        signal[start : start + len(mono)] = mono
    return signal


def _align_rirs_to_direct_arrival(
    rirs: list[np.ndarray],
    *,
    source_xyz: tuple[float, float, float],
    microphone_xyz: tuple[float, float, float],
    sample_rate: int,
    sound_speed_m_s: float = 343.0,
) -> list[np.ndarray]:
    """Remove the source-distance-dependent geometric propagation delay.

    Pyroom RIRs contain propagation delay. Without removing that common delay,
    a complete utterance placed at the end of a scene loses its last phoneme
    when the convolution is cropped to the scene extent. Physical distance is
    still represented by RIR amplitude and reflections. Pyroom's fixed
    81-sample fractional-delay kernel retains its 40-sample algorithmic group
    delay; v2 records that constant instead of pretending sub-millisecond
    sample-zero alignment.
    """
    distance = float(
        np.linalg.norm(
            np.asarray(source_xyz, dtype=np.float64)
            - np.asarray(microphone_xyz, dtype=np.float64)
        )
    )
    delay = max(0, int(round(distance / sound_speed_m_s * sample_rate)))
    aligned = []
    for rir in rirs:
        value = np.asarray(rir, dtype=np.float32)
        if delay >= len(value):
            raise RuntimeError(
                f"computed direct delay {delay} exceeds RIR length {len(value)}"
            )
        aligned.append(value[delay:].copy())
    return aligned


def _position_xyz(position: dict[str, Any], room: dict[str, Any]) -> tuple[float, float, float]:
    azimuth = math.radians(float(position.get("azimuth_deg") or 0.0))
    elevation = math.radians(float(position.get("elevation_deg") or 0.0))
    distance = max(0.3, float(position.get("distance_m") or 1.0))
    return az_el_dist_to_xyz(
        tuple(room["microphone_xyz_m"]), azimuth, elevation, distance
    )


def _activity_signal_key(
    source: dict[str, Any],
    *,
    sample_rate: int,
    num_samples: int,
) -> tuple[Any, ...]:
    """Identify the exact mono activity signal before spatial rendering."""

    dry = source["dry_audio"]
    playback = source["playback"]
    activity = source["activity"]
    return (
        str(dry["path"]),
        int(dry["native_sample_rate"]),
        int(dry["crop_start_native_sample"]),
        int(playback["offset_in_crop_native_sample"]),
        int(playback["num_native_samples"]),
        float(playback["loudness"]["gain_linear"]),
        float(activity["onset_sec"]),
        float(activity["offset_sec"]),
        int(sample_rate),
        int(num_samples),
    )


def _keyframe_weights(
    keyframes: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    sample_rate: int,
    num_samples: int,
) -> np.ndarray:
    """Piecewise-linear weights [K,T] over the source activity window."""

    keyframes = sorted(keyframes, key=lambda item: float(item.get("t_norm", 0.0)))
    positions = np.asarray(
        [min(1.0, max(0.0, float(item.get("t_norm", 0.0)))) for item in keyframes],
        dtype=np.float32,
    )
    if len(positions) == 1:
        return np.ones((1, num_samples), dtype=np.float32)
    activity = source["activity"]
    start = float(activity["onset_sec"])
    stop = float(activity["offset_sec"])
    seconds = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    denominator = max(1e-6, stop - start)
    normalized = np.clip((seconds - start) / denominator, 0.0, 1.0)
    weights = np.zeros((len(keyframes), num_samples), dtype=np.float32)
    for index in range(len(keyframes) - 1):
        left, right = float(positions[index]), float(positions[index + 1])
        if right <= left:
            continue
        mask = (normalized >= left) & (normalized <= right)
        fraction = np.zeros_like(normalized)
        fraction[mask] = (normalized[mask] - left) / (right - left)
        weights[index, mask] = 1.0 - fraction[mask]
        weights[index + 1, mask] = fraction[mask]
    weights[0, normalized <= positions[0]] = 1.0
    weights[-1, normalized >= positions[-1]] = 1.0
    return weights


def _convolve_foa_keyframes(
    mono: np.ndarray,
    rir_keyframes: list[list[np.ndarray]],
    *,
    num_samples: int,
) -> list[np.ndarray]:
    """FFT the mono signal once and render all four-channel RIR banks."""

    from scipy.fft import irfft, next_fast_len, rfft

    max_rir = max(len(rir) for bank in rir_keyframes for rir in bank)
    fft_size = next_fast_len(len(mono) + max_rir - 1)
    signal_fft = rfft(mono, n=fft_size, workers=1)
    rendered: list[np.ndarray] = []
    for rirs in rir_keyframes:
        bank = np.zeros((4, max_rir), dtype=np.float32)
        for channel, impulse in enumerate(rirs):
            bank[channel, : len(impulse)] = impulse
        bank_fft = rfft(bank, n=fft_size, axis=-1, workers=1)
        audio = irfft(
            bank_fft * signal_fft[None, :],
            n=fft_size,
            axis=-1,
            workers=1,
        )
        rendered.append(audio[:, :num_samples].astype(np.float32, copy=False))
    return rendered


def _render_track(
    recipe: dict[str, Any],
    source: dict[str, Any],
    *,
    mono: np.ndarray | None = None,
) -> np.ndarray:
    audio = recipe["audio"]
    sample_rate = int(audio["sample_rate"])
    num_samples = int(audio["num_samples"])
    if mono is None:
        mono = _load_activity_signal(
            source, sample_rate=sample_rate, num_samples=num_samples
        )
    declared_tail = int(audio.get("render_tail_samples", 0))
    if declared_tail < 0 or declared_tail >= num_samples:
        raise RuntimeError(f"invalid declared render tail: {declared_tail}")
    transcript = str((source.get("content") or {}).get("transcript") or "").strip()
    if transcript:
        activity_stop = int(
            round(float(source["activity"]["offset_sec"]) * sample_rate)
        )
        available_tail = num_samples - activity_stop
        if available_tail < declared_tail:
            raise RuntimeError(
                "complete speech does not leave its declared room tail: "
                f"{available_tail} < {declared_tail} samples"
            )
    room = recipe["room"]
    room_dim = tuple(map(float, room["dimensions_m"]))
    rt60 = float(room["rt60_s"])
    max_order = int(room["max_order"])
    microphone = tuple(map(float, room["microphone_xyz_m"]))
    keyframes = source["motion"]["keyframes"]

    rir_keyframes = []
    for keyframe in keyframes:
        source_xyz = _position_xyz(keyframe.get("position") or {}, room)
        rir_keyframes.append(
            _align_rirs_to_direct_arrival(
                room_rirs(
                    room_dim,
                    rt60,
                    max_order,
                    microphone,
                    source_xyz,
                    sample_rate,
                ),
                source_xyz=source_xyz,
                microphone_xyz=microphone,
                sample_rate=sample_rate,
            )
        )
    rendered_at_keyframes = _convolve_foa_keyframes(
        mono,
        rir_keyframes,
        num_samples=num_samples,
    )
    if len(rendered_at_keyframes) == 1:
        output = rendered_at_keyframes[0]
    else:
        weights = _keyframe_weights(
            keyframes,
            source=source,
            sample_rate=sample_rate,
            num_samples=num_samples,
        )
        output = np.zeros((4, num_samples), dtype=np.float32)
        for index, rendered in enumerate(rendered_at_keyframes):
            if rendered.shape[1] < num_samples:
                rendered = np.pad(rendered, ((0, 0), (0, num_samples - rendered.shape[1])))
            output += rendered[:, :num_samples] * weights[index][None, :]
    if declared_tail:
        fade_samples = min(declared_tail, max(1, int(round(0.020 * sample_rate))))
        output[:, -fade_samples:] *= np.linspace(
            1.0, 0.0, fade_samples, endpoint=True, dtype=np.float32
        )[None, :]
    return output


def _store_track(
    track_root: Path,
    track_id: str,
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    path = track_root / f"{track_id}.flac"
    metadata_path = track_root / f"{track_id}.json"
    peak = float(np.max(np.abs(audio)))
    storage_scale = min(1.0, 0.95 / peak) if peak > 0 else 1.0
    _atomic_write_pcm24(path, audio * storage_scale, sample_rate)
    metadata = {
        "track_id": track_id,
        "path": str(path),
        "sample_rate": int(sample_rate),
        "channels": 4,
        "num_samples": int(audio.shape[1]),
        "channel_layout": FOA_LAYOUT,
        "storage_scale": storage_scale,
        "restore_gain": 1.0 / storage_scale,
        "sha256": _sha256(path),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _load_track(metadata: dict[str, Any]) -> np.ndarray:
    audio, sample_rate = sf.read(
        metadata["path"], always_2d=True, dtype="float32"
    )
    if int(sample_rate) != int(metadata["sample_rate"]) or audio.shape[1] != 4:
        raise RuntimeError(f"cached track layout changed: {metadata['path']}")
    return (audio.T * float(metadata["restore_gain"])).astype(np.float32)


def _track_metadata(track_root: Path, track_id: str) -> dict[str, Any] | None:
    path = track_root / f"{track_id}.json"
    audio_path = track_root / f"{track_id}.flac"
    if not path.is_file() or not audio_path.is_file():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("track_id") != track_id or metadata.get("path") != str(audio_path):
        raise RuntimeError(f"invalid cached track metadata: {path}")
    return metadata


def render_family(
    family: dict[str, Any],
    output_root: Path,
    *,
    storage_profile: str = RETAINED_PROFILE,
    minimum_rms: float = 1.0e-4,
    minimum_active_100ms_fraction: float = 0.005,
    active_frame_rms_threshold: float = 1.0e-4,
) -> dict[str, Any]:
    if storage_profile not in STORAGE_PROFILES:
        raise ValueError(f"unknown storage profile: {storage_profile}")
    validate_edit_family(family, require_outputs=False)
    family_id = str(family["family_id"])
    family_root = output_root / "families" / family_id
    manifest_path = family_root / "family.json"
    if manifest_path.is_file():
        rendered = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_edit_family(rendered, require_outputs=True)
        existing_profile = str(
            (rendered.get("render_summary") or {}).get("storage_profile")
            or RETAINED_PROFILE
        )
        if existing_profile != storage_profile:
            raise RuntimeError(
                f"render profile mismatch for {family_id}: "
                f"{existing_profile} != {storage_profile}"
            )
        return rendered

    track_root = family_root / "source_tracks"
    track_metadata: dict[str, dict[str, Any]] = {}
    # A persistent source commonly appears in several consecutive turns. The
    # previous implementation decoded its PCM24 FLAC once per turn, creating
    # avoidable filesystem traffic under high process counts. Decode each
    # stored track exactly once per family and reuse the same float32 array.
    # Loading after `_store_track` (instead of reusing the pre-quantized render)
    # preserves the existing PCM24 quantization and therefore the exact FOA
    # mixing contract.
    track_audio: dict[str, np.ndarray] = {}
    signal_cache: dict[tuple[Any, ...], np.ndarray] = {}
    for job in unique_source_render_jobs(family):
        track_id = job["track_id"]
        sample_rate = int(job["recipe"]["audio"]["sample_rate"])
        num_samples = int(job["recipe"]["audio"]["num_samples"])
        signal_key = _activity_signal_key(
            job["source"],
            sample_rate=sample_rate,
            num_samples=num_samples,
        )
        mono = signal_cache.get(signal_key)
        if mono is None:
            mono = _load_activity_signal(
                job["source"],
                sample_rate=sample_rate,
                num_samples=num_samples,
            )
            signal_cache[signal_key] = mono
        if storage_profile == RETAINED_PROFILE:
            cached = _track_metadata(track_root, track_id)
            if cached is None:
                track = _render_track(
                    job["recipe"], job["source"], mono=mono
                )
                cached = _store_track(
                    track_root,
                    track_id,
                    track,
                    sample_rate,
                )
            track_metadata[track_id] = cached
            track_audio[track_id] = _load_track(cached)
        else:
            track = _render_track(
                job["recipe"], job["source"], mono=mono
            )
            peak = float(np.max(np.abs(track)))
            storage_scale = min(1.0, 0.95 / peak) if peak > 0 else 1.0
            quantized = _flac_pcm24_memory_roundtrip(
                track * storage_scale,
                sample_rate,
            )
            restore_gain = 1.0 / storage_scale
            track_audio[track_id] = (
                quantized * restore_gain
            ).astype(np.float32, copy=False)
            track_metadata[track_id] = {
                "track_id": track_id,
                "path": None,
                "sample_rate": sample_rate,
                "channels": 4,
                "num_samples": int(track.shape[1]),
                "channel_layout": FOA_LAYOUT,
                "storage_scale": storage_scale,
                "restore_gain": restore_gain,
                "sha256": None,
                "retained": False,
            }

    raw_mixes: list[np.ndarray] = []
    per_turn_refs: list[list[dict[str, Any]]] = []
    for recipe in family["recipes"]:
        mix = np.zeros((4, int(recipe["audio"]["num_samples"])), dtype=np.float32)
        refs = []
        for source in recipe["sources"]:
            track_id = source_render_signature(recipe, source)
            metadata = track_metadata[track_id]
            track = track_audio[track_id]
            gain = 10.0 ** (float(source.get("gain_db", 0.0)) / 20.0)
            mix += track[:, : mix.shape[1]] * gain
            refs.append(
                {
                    "source_id": source["source_id"],
                    "track_id": track_id,
                    "path": metadata["path"],
                    "gain_db": float(source.get("gain_db", 0.0)),
                    "retained": storage_profile == RETAINED_PROFILE,
                }
            )
        raw_mixes.append(mix)
        per_turn_refs.append(refs)

    family_peak = max(float(np.max(np.abs(mix))) for mix in raw_mixes)
    peak_target = float(family["recipes"][0]["mix"]["family_peak_target"])
    master_gain = peak_target / family_peak if family_peak > 0 else 1.0
    rendered = copy.deepcopy(family)
    state_signal_stats = []
    for index, (recipe, raw_mix, refs) in enumerate(
        zip(rendered["recipes"], raw_mixes, per_turn_refs)
    ):
        output = family_root / "audio" / f"{recipe['turn_id']}_WYZX_4ch.flac"
        final_mix = np.clip(raw_mix * master_gain, -1.0, 1.0).astype(
            np.float32, copy=False
        )
        if not np.isfinite(final_mix).all():
            raise RuntimeError(f"non-finite FOA state: {family_id}/{recipe['turn_id']}")
        state_peak = float(np.max(np.abs(final_mix)))
        state_rms = float(
            np.sqrt(np.mean(np.square(final_mix, dtype=np.float64)))
        )
        block = max(1, int(recipe["audio"]["sample_rate"]) // 10)
        frame_rms = [
            float(
                np.sqrt(
                    np.mean(
                        np.square(final_mix[:, start : start + block], dtype=np.float64)
                    )
                )
            )
            for start in range(0, final_mix.shape[1], block)
        ]
        active_fraction = float(
            np.mean(
                np.asarray(frame_rms, dtype=np.float64)
                >= float(active_frame_rms_threshold)
            )
        )
        if state_rms < float(minimum_rms):
            raise RuntimeError(
                f"FOA state is effectively silent: {family_id}/{recipe['turn_id']} "
                f"rms={state_rms:.9g} minimum={float(minimum_rms):.9g}"
            )
        if active_fraction < float(minimum_active_100ms_fraction):
            raise RuntimeError(
                f"FOA state has no audible activity: {family_id}/{recipe['turn_id']} "
                f"active_100ms_fraction={active_fraction:.9g} "
                f"minimum={float(minimum_active_100ms_fraction):.9g}"
            )
        _atomic_write_pcm24(
            output,
            final_mix,
            int(recipe["audio"]["sample_rate"]),
        )
        signal_stats = {
            "peak": state_peak,
            "rms": state_rms,
            "active_100ms_fraction": active_fraction,
            "active_frame_rms_threshold": float(active_frame_rms_threshold),
        }
        state_signal_stats.append(signal_stats)
        recipe["mix"]["family_master_gain_linear"] = master_gain
        recipe["outputs"] = {
            "foa_path": str(output),
            "foa_sha256": _sha256(output),
            "source_track_refs": refs,
            "reference_audio": family.get("reference_audio"),
            "state_input": STATE_RENDER_INPUT,
            "uses_previous_foa": False,
            "signal_stats": signal_stats,
        }
        rendered["turns"][index]["after"]["audio_ref"] = str(output)
        if index:
            rendered["turns"][index]["before"]["audio_ref"] = rendered[
                "recipes"
            ][index - 1]["outputs"]["foa_path"]
    rendered["render_status"] = "rendered"
    rendered["render_summary"] = {
        "family_peak_before_master": family_peak,
        "family_master_gain_linear": master_gain,
        "peak_target": peak_target,
        "unique_source_tracks": len(track_metadata),
        "state_input": STATE_RENDER_INPUT,
        "uses_previous_foa": False,
        "independent_state_mix": True,
        "storage_profile": storage_profile,
        "source_tracks_retained": storage_profile == RETAINED_PROFILE,
        "foa_container": "flac",
        "foa_subtype": "PCM_24",
        "output_root": str(output_root),
        "minimum_rms": float(minimum_rms),
        "minimum_active_100ms_fraction": float(minimum_active_100ms_fraction),
        "active_frame_rms_threshold": float(active_frame_rms_threshold),
        "state_rms_min": min(item["rms"] for item in state_signal_stats),
        "state_active_100ms_fraction_min": min(
            item["active_100ms_fraction"] for item in state_signal_stats
        ),
    }
    validate_edit_family(rendered, require_outputs=True)
    atomic_write_json(manifest_path, rendered)
    return rendered


def _render_family_job(
    payload: tuple[dict[str, Any], str, str, float, float, float],
) -> dict[str, Any]:
    """Process-pool entrypoint; each family owns a disjoint output directory."""

    (
        family,
        output_root,
        storage_profile,
        minimum_rms,
        minimum_active_fraction,
        active_frame_rms_threshold,
    ) = payload
    return render_family(
        family,
        Path(output_root),
        storage_profile=storage_profile,
        minimum_rms=minimum_rms,
        minimum_active_100ms_fraction=minimum_active_fraction,
        active_frame_rms_threshold=active_frame_rms_threshold,
    )


def _warm_process_pool_imports() -> None:
    """Import heavy CPU kernels once before Linux forks render workers."""

    import pyroomacoustics  # noqa: F401
    from scipy import signal  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel family render processes within this recipe shard.",
    )
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--minimum-rms", type=float, default=1.0e-4)
    parser.add_argument(
        "--minimum-active-100ms-fraction", type=float, default=0.005
    )
    parser.add_argument(
        "--active-frame-rms-threshold", type=float, default=1.0e-4
    )
    parser.add_argument(
        "--storage-profile",
        choices=STORAGE_PROFILES,
        default=RETAINED_PROFILE,
        help=(
            "retained_flac_pcm24 keeps FOA and source-track FLACs; "
            "transient_flac_pcm24 keeps only FOA FLAC staging and quantizes "
            "source tracks in memory"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard < args.num_shards:
        raise SystemExit("require 0 <= shard < num-shards")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if (
        args.minimum_rms <= 0.0
        or not 0.0 <= args.minimum_active_100ms_fraction <= 1.0
        or args.active_frame_rms_threshold <= 0.0
    ):
        raise SystemExit("invalid signal audibility thresholds")
    recipe_root = args.recipe_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if (
        output_root == recipe_root
        or output_root in recipe_root.parents
        or recipe_root in output_root.parents
    ):
        raise SystemExit("output-root must be a separate render branch")
    if not args.dry_run:
        _prepare_output_root(output_root, args.storage_profile)
    free_probe = output_root if output_root.exists() else output_root.parent
    while not free_probe.exists() and free_probe != free_probe.parent:
        free_probe = free_probe.parent
    free_gb = shutil.disk_usage(free_probe).free / (1024**3)
    if not args.dry_run and free_gb < args.min_free_gb:
        raise SystemExit(
            f"output volume has only {free_gb:.1f} GiB free; "
            f"minimum is {args.min_free_gb:.1f} GiB"
        )

    families: list[dict[str, Any]] = []
    for index, family in enumerate(_iter_families(recipe_root)):
        if index % args.num_shards != args.shard:
            continue
        validate_edit_family(family, require_outputs=False)
        families.append(family)
        if args.limit is not None and len(families) >= args.limit:
            break

    selected = len(families)
    rendered_count = 0
    manifest = output_root / f"render_manifest.shard{args.shard:03d}.jsonl"
    # Family manifests are the resumable canonical outputs. Rebuild this small
    # summary on every invocation so an interrupted retry cannot accumulate
    # duplicate rows before publishing READY.
    sink = None if args.dry_run else manifest.open("w", encoding="utf-8")
    try:
        if args.dry_run:
            for family in families:
                print(
                    json.dumps(
                        {
                            "family_id": family["family_id"],
                            "turns": len(family["recipes"]),
                            "unique_source_tracks": len(unique_source_render_jobs(family)),
                            "reference_audio": family.get("reference_audio"),
                        },
                        sort_keys=True,
                    )
                )
        else:
            payloads = [
                (
                    family,
                    str(output_root),
                    args.storage_profile,
                    float(args.minimum_rms),
                    float(args.minimum_active_100ms_fraction),
                    float(args.active_frame_rms_threshold),
                )
                for family in families
            ]
            if args.workers == 1:
                results = map(_render_family_job, payloads)
            else:
                _warm_process_pool_imports()
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(args.workers, max(1, selected))
                )
                results = executor.map(_render_family_job, payloads, chunksize=1)
            try:
                for result in results:
                    sink.write(
                        json.dumps(
                            {
                                "family_id": result["family_id"],
                                "family_manifest": str(
                                    output_root
                                    / "families"
                                    / result["family_id"]
                                    / "family.json"
                                ),
                                "turns": len(result["recipes"]),
                                "reference_audio": result.get("reference_audio"),
                                **result["render_summary"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    sink.flush()
                    rendered_count += 1
            finally:
                if args.workers != 1:
                    executor.shutdown(wait=True, cancel_futures=True)
    finally:
        if sink is not None:
            sink.close()
    if not args.dry_run:
        if rendered_count != selected:
            raise RuntimeError(
                f"render completion mismatch: rendered={rendered_count} "
                f"selected={selected}"
            )
        if args.num_shards == 1:
            atomic_write_json(
                output_root / READY_MARKER,
                {
                    "schema": "stable_audio_tools.spatial_cot_render_ready",
                    "schema_version": 1,
                    "families": rendered_count,
                    "render_manifest": str(manifest),
                    "storage_profile": args.storage_profile,
                },
            )
    print(
        f"[spatial-edit-render] selected={selected} rendered={rendered_count} "
        f"dry_run={args.dry_run} output={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
