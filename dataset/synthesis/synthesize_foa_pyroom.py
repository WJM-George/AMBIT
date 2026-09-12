#!/usr/bin/env python3
"""Synthesize first-order ambisonics (FOA) from mono dry audio via pyroomacoustics.

Use case: augment datasets like AudioCaps (mono + caption jsonl) into 4-channel FOA
for the 4ch VAE / spatial-audio training path. Channel layout matches dataset_4ch:
ACN/SN3D **[W, Y, Z, X]** (pyroom ``get_mn_in_acn_order(1)``).

Pipeline per clip:
  1. Load mono wav (from jsonl ``audio_path``).
  2. Random shoebox room (size, RT60, source position around listener).
  3. Coincident FOA mic at room center with real spherical-harmonic directivities.
  4. ISM RIR + simulate -> 4ch; joint peak-normalize; write ``*_WYZX_4ch.flac``.
  5. Append manifest jsonl (resumable, shardable).

This is **synthetic** spatial audio (not YouTube FOA like Sphere360). Pair with real FOA
data (Sphere360, Spatial LibriSpeech) for diversity.

Setup (project standard: uv, no manual venv activate):
    cd /home/tanhe/dataset_storage/stable-audio-tools
    uv sync --extra spatial

Build input jsonl from AudioCaps parquet (one-time):
    uv run python dataset/synthesis/synthesize_foa_pyroom.py \\
        --export-audiocaps-jsonl \\
        --audiocaps-root /mnt/sdd/audio_dataset/datasets/audiocaps/snapshot \\
        --split train --out /mnt/sdc/audio_dataset_tmp/audiocaps_train.jsonl

Synthesize FOA (full jsonl):
    uv run python dataset/synthesis/synthesize_foa_pyroom.py \\
        --input /mnt/sdc/audio_dataset_tmp/audiocaps_train.jsonl \\
        --out-dir /mnt/sdc/audio_dataset_tmp/audiocaps_foa/train \\
        --manifest /mnt/sdc/audio_dataset_tmp/audiocaps_foa/train_manifest.jsonl \\
        --jobs 8

Quick sample (~1000 clips only):
    uv run python dataset/synthesis/synthesize_foa_pyroom.py \\
        --input /mnt/sdc/audio_dataset_tmp/audiocaps_train.jsonl \\
        --out-dir /mnt/sdc/audio_dataset_tmp/audiocaps_foa/sample1k \\
        --manifest /mnt/sdc/audio_dataset_tmp/audiocaps_foa/sample1k_manifest.jsonl \\
        --num 1000 --jobs 8

Shard across machines/GPUs (CPU-bound; scale with --jobs):
    uv run python dataset/synthesis/synthesize_foa_pyroom.py --input ... --num-shards 4 --shard 0 ...
"""

## 设置一下，研究一下怎么搞pyroom的设置。
## 输入和输出是什么对比
## 有binaural和FOA的转化库
## 训不动整体，就冻结中间训练两边。VAE的数据占比和分配。stable audio是怎么训练的，定长？
## VAE目标就是能用就行，让DiT好学。
## 冻结哪部分，问问AI。
## 数据怎么造： 要训练VAE，要多样化数据，造的方式：三维，波形上比较明显，多组环境/多组方位/多组距离/多组角度
## 以平面和上下为主来造。1.环境 2.距离 3.方位 4.声源（audio music speech，要混均匀）5.造一些声源是动的。
## audiocaps,audioset,fsdkaggle,picoaudio,vggsound,musiccaps,spatiallibrispeech
## 造数据，训练VAE，看看哪个设置是最好的，开始做实验。
## captioning可以用文本模型refine caption。
## 咸鱼上找数据,soundspace可以合成视频
## T2A先做


from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import soundfile as sf

# pyroomacoustics imported inside worker to keep main process light

FOA_DEGREE = 1
NUM_SLOTS = 4
DEFAULT_FS = 48000
N3D_TO_SN3D_FIRST_ORDER_GAIN = 1.0 / np.sqrt(3.0)
DEFAULT_PRIMARY = Path(os.environ.get("AUDIO_DATASET_PRIMARY_ROOT", "/mnt/sdd/audio_dataset"))


def pyroom_n3d_to_sn3d_foa(audio: np.ndarray) -> np.ndarray:
    """Convert Pyroom's ACN real-SH response to ACN/SN3D `[W,Y,Z,X]`.

    ``RealSphericalHarmonicsDirectivity`` uses orthonormal/N3D-like scaling:
    a first-order channel has a ``sqrt(3)`` larger ratio to W than SN3D.  The
    former renderer labelled those signals SN3D without this conversion.  V2
    makes the normalization explicit at the first boundary after Pyroom.
    """
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim < 1 or value.shape[0] != NUM_SLOTS:
        raise ValueError(f"expected Pyroom FOA with leading shape 4, got {value.shape}")
    result = value.copy()
    result[1:4] *= N3D_TO_SN3D_FIRST_ORDER_GAIN
    return result


def pyroom_n3d_to_sn3d_rirs(rirs: list[np.ndarray]) -> list[np.ndarray]:
    if len(rirs) != NUM_SLOTS:
        raise ValueError(f"expected four Pyroom RIRs, found {len(rirs)}")
    return [
        np.asarray(rir, dtype=np.float32)
        * (1.0 if channel == 0 else N3D_TO_SN3D_FIRST_ORDER_GAIN)
        for channel, rir in enumerate(rirs)
    ]


@dataclass
class RoomConfig:
    """Randomized shoebox + source placement for one clip."""

    room_dim: tuple[float, float, float]  # L, W, H (m)
    rt60: float
    max_order: int
    mic_pos: tuple[float, float, float]
    src_pos: tuple[float, float, float]


def _random_room_config(rng: random.Random, rt60_range: tuple[float, float],
                       room_range: tuple[float, float]) -> RoomConfig:
    import pyroomacoustics as pra

    L = rng.uniform(*room_range)
    W = rng.uniform(*room_range)
    H = rng.uniform(2.2, min(4.0, room_range[1]))
    room_dim = (L, W, H)
    rt60 = rng.uniform(*rt60_range)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    max_order = int(min(max(max_order, 3), 15))

    # Listener (FOA array) near center, elevated ~1.5 m
    mic = (
        rng.uniform(0.35 * L, 0.65 * L),
        rng.uniform(0.35 * W, 0.65 * W),
        rng.uniform(1.2, min(1.8, H - 0.3)),
    )
    # Source: 0.8–3.5 m away, not too close to walls
    for _ in range(32):
        src = (
            rng.uniform(0.5, L - 0.5),
            rng.uniform(0.5, W - 0.5),
            rng.uniform(1.0, min(2.0, H - 0.2)),
        )
        dist = np.linalg.norm(np.subtract(src, mic))
        if 0.8 <= dist <= 3.5:
            return RoomConfig(room_dim, rt60, max_order, mic, src)
    return RoomConfig(room_dim, rt60, max_order, mic, src)


def _resample_mono(signal: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return signal.astype(np.float32, copy=False)
    import torch
    import torchaudio.functional as F

    x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)
    y = F.resample(x, sr_in, sr_out)
    return y.squeeze(0).numpy()


def _joint_peak_normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(audio))
    if m > 0:
        audio = audio * (peak / m)
    return audio


def synthesize_foa(
    mono: np.ndarray,
    sr: int,
    cfg: RoomConfig,
    *,
    target_fs: int = DEFAULT_FS,
) -> tuple[np.ndarray, int]:
    """Return float32 array [4, T] at target_fs."""
    import pyroomacoustics as pra
    from pyroomacoustics.directivities.harmonics import (
        RealSphericalHarmonicsDirectivity,
        get_mn_in_acn_order,
    )

    mono = _resample_mono(mono, sr, target_fs)
    sr = target_fs
    if mono.ndim != 1:
        mono = mono.reshape(-1)

    e_absorption, _ = pra.inverse_sabine(cfg.rt60, cfg.room_dim)
    room = pra.ShoeBox(
        list(cfg.room_dim),
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=cfg.max_order,
    )

    mic_xyz = np.array([[cfg.mic_pos[0]], [cfg.mic_pos[1]], [cfg.mic_pos[2]]])
    all_m, all_n = get_mn_in_acn_order(FOA_DEGREE)
    for m, n in zip(all_m, all_n):
        room.add_microphone(
            mic_xyz,
            directivity=RealSphericalHarmonicsDirectivity(int(m), int(n)),
        )

    room.add_source(list(cfg.src_pos), signal=mono)
    room.compute_rir()
    room.simulate()

    out = pyroom_n3d_to_sn3d_foa(room.mic_array.signals)
    out = _joint_peak_normalize(out, peak=0.9)
    return out, sr


# ===========================================================================
# Spatial engine: az/el/distance placement, room RIRs, static & dynamic (moving)
# sources, multi-source mixing, and human-readable spatial words.
#
# Conventions (consistent with dataset/indexing/build_spatial_prompts.py + the FOA layout
# [W, Y, Z, X], ACN/SN3D):
#   * azimuth az: radians, 0 = front (+x), +pi/2 = LEFT (+y), -pi/2 = RIGHT.
#   * elevation el: radians, + = up (+z).
#   * FOA gains (SN3D): W=1, Y=sin(az)cos(el), Z=sin(el), X=cos(az)cos(el).
# ===========================================================================

_DIR8 = ["front", "front-left", "left", "rear-left", "behind",
         "rear-right", "right", "front-right"]


def azimuth_words(az_rad: float) -> str:
    a = (np.degrees(az_rad) + 360.0) % 360.0
    return _DIR8[int(((a + 22.5) % 360.0) // 45.0)]


def elevation_words(el_rad: float) -> str:
    el = np.degrees(el_rad)
    if el > 20:
        return "above"
    if el < -20:
        return "below"
    return "level"


def distance_words(dist_m: float) -> str:
    if dist_m < 1.2:
        return "very close"
    if dist_m < 2.2:
        return "nearby"
    if dist_m < 3.5:
        return "a short distance away"
    return "far away"


def reverb_words(rt60: float) -> str:
    if rt60 < 0.3:
        return "fairly dry room"
    if rt60 < 0.6:
        return "moderately reverberant room"
    return "highly reverberant room"


# ---------------------------------------------------------------------------
# Room archetypes: instead of one uniform random box, draw from a library of
# realistic acoustic spaces so the FOA corpus spans dry booths -> cathedrals ->
# outdoors. Each archetype carries dim/RT60/order ranges, a human description,
# and per-category weights (music gets more halls/studios, speech more rooms,
# general audio more outdoors/varied). RT60 ranges are kept physically valid for
# the size (Sabine), and outdoor uses max_order=0 (free field).
# ---------------------------------------------------------------------------

ROOM_ARCHETYPES: dict[str, dict] = {
    "vocal_booth":  {"L": (1.8, 2.8), "W": (1.6, 2.6), "H": (2.0, 2.6),
                     "rt60": (0.12, 0.25), "order": (6, 10),
                     "desc": "a small dry vocal booth",
                     "w": {"audio": 0.5, "music": 0.6, "speech": 1.4}},
    "studio":       {"L": (4.0, 7.0), "W": (3.0, 5.0), "H": (2.6, 3.2),
                     "rt60": (0.18, 0.35), "order": (8, 12),
                     "desc": "a treated recording studio",
                     "w": {"audio": 0.8, "music": 1.6, "speech": 1.0}},
    "small_room":   {"L": (3.0, 5.0), "W": (3.0, 4.5), "H": (2.4, 3.0),
                     "rt60": (0.25, 0.45), "order": (8, 12),
                     "desc": "a small room",
                     "w": {"audio": 1.2, "music": 0.8, "speech": 1.4}},
    "living_room":  {"L": (4.0, 7.0), "W": (3.5, 6.0), "H": (2.5, 3.2),
                     "rt60": (0.30, 0.55), "order": (8, 12),
                     "desc": "a furnished living room",
                     "w": {"audio": 1.2, "music": 1.0, "speech": 1.2}},
    "office":       {"L": (4.0, 8.0), "W": (4.0, 7.0), "H": (2.6, 3.2),
                     "rt60": (0.40, 0.70), "order": (8, 12),
                     "desc": "an office room",
                     "w": {"audio": 1.0, "music": 0.5, "speech": 1.2}},
    "classroom":    {"L": (7.0, 12.0), "W": (6.0, 10.0), "H": (3.0, 4.0),
                     "rt60": (0.50, 0.90), "order": (8, 12),
                     "desc": "a classroom",
                     "w": {"audio": 0.9, "music": 0.6, "speech": 1.1}},
    "bathroom_tiled": {"L": (2.0, 3.5), "W": (2.0, 3.5), "H": (2.4, 3.0),
                       "rt60": (0.60, 1.10), "order": (10, 14),
                       "desc": "a bright tiled bathroom",
                       "w": {"audio": 1.0, "music": 0.6, "speech": 0.8}},
    "corridor":     {"L": (8.0, 20.0), "W": (1.5, 2.5), "H": (2.6, 3.5),
                     "rt60": (0.60, 1.30), "order": (8, 12),
                     "desc": "a long corridor",
                     "w": {"audio": 1.1, "music": 0.5, "speech": 0.9}},
    "gymnasium":    {"L": (18.0, 30.0), "W": (12.0, 22.0), "H": (6.0, 10.0),
                     "rt60": (1.20, 2.00), "order": (6, 10),
                     "desc": "a large gymnasium",
                     "w": {"audio": 1.0, "music": 0.9, "speech": 0.6}},
    "concert_hall": {"L": (15.0, 28.0), "W": (12.0, 22.0), "H": (8.0, 14.0),
                     "rt60": (1.40, 2.40), "order": (6, 10),
                     "desc": "a concert hall",
                     "w": {"audio": 0.5, "music": 1.8, "speech": 0.5}},
    "cathedral":    {"L": (20.0, 40.0), "W": (14.0, 28.0), "H": (12.0, 22.0),
                     "rt60": (2.20, 3.80), "order": (5, 8),
                     "desc": "a vast reverberant cathedral",
                     "w": {"audio": 0.6, "music": 1.3, "speech": 0.5}},
    "outdoor":      {"L": (8.0, 16.0), "W": (8.0, 16.0), "H": (6.0, 10.0),
                     "rt60": (0.30, 0.50), "order": (0, 0),  # free field (no reflections)
                     "desc": "outdoors in the open",
                     "w": {"audio": 1.6, "music": 1.0, "speech": 0.8}},
}


def sample_room(rng: random.Random, category: Optional[str] = None) -> dict:
    """Draw a room archetype (category-weighted) and instantiate its parameters."""
    names = list(ROOM_ARCHETYPES.keys())
    weights = [ROOM_ARCHETYPES[n]["w"].get(category, 1.0) if category else 1.0 for n in names]
    name = rng.choices(names, weights=weights, k=1)[0]
    a = ROOM_ARCHETYPES[name]
    L = rng.uniform(*a["L"])
    W = rng.uniform(*a["W"])
    H = rng.uniform(*a["H"])
    rt60 = rng.uniform(*a["rt60"])
    max_order = rng.randint(*a["order"])
    mic = (rng.uniform(0.4 * L, 0.6 * L), rng.uniform(0.4 * W, 0.6 * W),
           rng.uniform(1.2, min(1.8, H - 0.4)))
    return {"dim": (L, W, H), "rt60": rt60, "max_order": max_order, "mic": mic,
            "type": name, "desc": a["desc"], "free_field": name == "outdoor"}


def room_words(room: dict) -> str:
    """Human room description, e.g. 'a concert hall (highly reverberant)'."""
    desc = room.get("desc") or reverb_words(room.get("rt60", 0.4))
    if room.get("type") == "outdoor":
        return desc
    return f"{desc} ({reverb_words(room.get('rt60', 0.4))})"


def foa_gains(az: float, el: float) -> np.ndarray:
    """SN3D/ACN order-1 encoding gains for [W, Y, Z, X]."""
    ce = np.cos(el)
    return np.array([1.0, np.sin(az) * ce, np.sin(el), np.cos(az) * ce], dtype=np.float32)


def az_el_dist_to_xyz(mic: tuple[float, float, float], az: float, el: float,
                      dist: float) -> tuple[float, float, float]:
    ce = np.cos(el)
    return (mic[0] + dist * ce * np.cos(az),
            mic[1] + dist * ce * np.sin(az),
            mic[2] + dist * np.sin(el))


def clamp_src_in_room(xyz, room_dim, margin: float = 0.3):
    return tuple(float(np.clip(c, margin, d - margin)) for c, d in zip(xyz, room_dim))


def _build_room(room_dim, rt60, max_order, mic_pos, fs):
    import pyroomacoustics as pra
    from pyroomacoustics.directivities.harmonics import (
        RealSphericalHarmonicsDirectivity, get_mn_in_acn_order,
    )
    try:
        e_absorption, _ = pra.inverse_sabine(rt60, list(room_dim))
    except ValueError:
        # RT60 too short for the room volume (Sabine over-absorption); clamp.
        e_absorption = 0.99
    room = pra.ShoeBox(list(room_dim), fs=fs, materials=pra.Material(e_absorption),
                       max_order=int(max_order))
    mic_xyz = np.array([[mic_pos[0]], [mic_pos[1]], [mic_pos[2]]])
    all_m, all_n = get_mn_in_acn_order(FOA_DEGREE)
    for m, n in zip(all_m, all_n):
        room.add_microphone(mic_xyz, directivity=RealSphericalHarmonicsDirectivity(int(m), int(n)))
    return room


def room_rirs(room_dim, rt60, max_order, mic_pos, src_xyz, fs) -> list[np.ndarray]:
    """Return the 4 FOA-mic impulse responses for a source at src_xyz."""
    room = _build_room(room_dim, rt60, max_order, mic_pos, fs)
    room.add_source(list(clamp_src_in_room(src_xyz, room_dim)))
    room.compute_rir()
    return pyroom_n3d_to_sn3d_rirs(
        [np.asarray(room.rir[m][0], dtype=np.float32) for m in range(NUM_SLOTS)]
    )


def _conv(sig: np.ndarray, ir: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import fftconvolve
        return fftconvolve(sig, ir)[: len(sig) + len(ir) - 1].astype(np.float32)
    except Exception:  # noqa: BLE001
        return np.convolve(sig, ir).astype(np.float32)


def synthesize_foa_static_at(mono, sr, room_dim, rt60, max_order, mic_pos, src_xyz):
    """Single static source via full room simulation -> [4, T]."""
    room = _build_room(room_dim, rt60, max_order, mic_pos, sr)
    room.add_source(list(clamp_src_in_room(src_xyz, room_dim)), signal=mono.astype(np.float32))
    room.compute_rir()
    room.simulate()
    return pyroom_n3d_to_sn3d_foa(room.mic_array.signals)


def synthesize_foa_dynamic_room(mono, sr, room_dim, rt60, max_order, mic_pos,
                                waypoints_xyz) -> np.ndarray:
    """Moving source: render the clip through each waypoint's RIR, then cross-fade
    along time with a linear (tent) basis so only two adjacent waypoints overlap.
    Realistic room movement -> [4, T]."""
    mono = mono.astype(np.float32)
    n = len(mono)
    K = len(waypoints_xyz)
    rendered = []
    for wp in waypoints_xyz:
        rirs = room_rirs(room_dim, rt60, max_order, mic_pos, wp, sr)
        chans = np.stack([_conv(mono, ir)[:n] for ir in rirs], axis=0)  # [4, n] (trimmed)
        if chans.shape[1] < n:
            chans = np.pad(chans, ((0, 0), (0, n - chans.shape[1])))
        rendered.append(chans)
    pos = np.linspace(0.0, K - 1, n)
    out = np.zeros((NUM_SLOTS, n), dtype=np.float32)
    for k in range(K):
        w = np.clip(1.0 - np.abs(pos - k), 0.0, 1.0).astype(np.float32)
        out += w[None, :] * rendered[k]
    return out


def synthesize_foa_dynamic_pan(mono, sr, traj) -> np.ndarray:
    """Anechoic moving pan from a trajectory list of (az, el, dist). Cheap, very
    clear motion (no reverb). -> [4, T]."""
    mono = mono.astype(np.float32)
    n = len(mono)
    K = len(traj)
    idx = np.linspace(0.0, K - 1, n)
    k0 = np.clip(np.floor(idx).astype(int), 0, K - 1)
    k1 = np.clip(k0 + 1, 0, K - 1)
    fr = (idx - k0).astype(np.float32)
    az = np.array([t[0] for t in traj], np.float32)
    el = np.array([t[1] for t in traj], np.float32)
    di = np.array([t[2] for t in traj], np.float32)
    az_t = az[k0] * (1 - fr) + az[k1] * fr
    el_t = el[k0] * (1 - fr) + el[k1] * fr
    di_t = np.clip(di[k0] * (1 - fr) + di[k1] * fr, 0.5, None)
    atten = 1.0 / di_t
    ce = np.cos(el_t)
    out = np.stack([
        mono * atten,                       # W
        mono * np.sin(az_t) * ce * atten,   # Y
        mono * np.sin(el_t) * atten,        # Z
        mono * np.cos(az_t) * ce * atten,   # X
    ], axis=0).astype(np.float32)
    return out


def mix_foa(foas: list[np.ndarray]) -> np.ndarray:
    """Sum a list of [4, T_i] FOA signals (zero-padded to the longest)."""
    n = max(f.shape[1] for f in foas)
    acc = np.zeros((NUM_SLOTS, n), dtype=np.float32)
    for f in foas:
        acc[:, : f.shape[1]] += f
    return acc


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return data[:, 0].astype(np.float32), int(sr)


def _worker(args: tuple) -> dict[str, Any]:
    """Process one jsonl record; return manifest row or error dict."""
    record, out_dir, target_fs, seed, rt60_range, room_range = args
    rng = random.Random(seed)
    clip_id = record.get("id") or record.get("clip_id") or Path(record["audio_path"]).stem
    audio_path = Path(record["audio_path"])
    out_path = out_dir / f"{clip_id}_WYZX_4ch.flac"

    if out_path.exists() and out_path.stat().st_size > 1024:
        return {
            "id": clip_id,
            "status": "skipped",
            "foa_path": str(out_path),
            "caption": record.get("caption", ""),
        }

    try:
        mono, sr = _load_mono(audio_path)
        cfg = _random_room_config(rng, rt60_range, room_range)
        foa, out_sr = synthesize_foa(mono, sr, cfg, target_fs=target_fs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), foa.T, out_sr, subtype="PCM_24")
        return {
            "id": clip_id,
            "status": "ok",
            "foa_path": str(out_path),
            "audio_path": str(audio_path),
            "caption": record.get("caption", ""),
            "sample_rate": out_sr,
            "channels": 4,
            "channel_layout": "WYZX_ACN_SN3D",
            "duration_sec": float(foa.shape[1] / out_sr),
            "room": asdict(cfg),
        }
    except Exception as exc:  # noqa: BLE001
        return {"id": clip_id, "status": "error", "error": str(exc)[:300]}


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(manifest):
        if row.get("status") in ("ok", "skipped"):
            done.add(row["id"])
    return done


def export_audiocaps_jsonl(
    root: Path,
    split: str,
    out_jsonl: Path,
    *,
    wav_dir: Path | None,
    limit: int | None,
) -> int:
    """Write jsonl with extracted wav paths from AudioCaps parquet shards.

    Uses pyarrow + soundfile only (no HuggingFace ``datasets`` / ``torchcodec``).
    Audio is stored as embedded bytes in each parquet row.
    """
    import pyarrow.parquet as pq

    shards = sorted(root.glob(f"data/{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards for split={split} under {root}/data")

    wav_dir = wav_dir or (out_jsonl.parent / f"{split}_mono_wav")
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    columns = ["audiocap_id", "youtube_id", "start_time", "caption", "audio"]
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as sink:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    if limit is not None and n >= limit:
                        return n
                    clip_id = f"audiocaps_{row['audiocap_id']}"
                    wav_path = wav_dir / f"{clip_id}.wav"
                    if not wav_path.exists():
                        blob = row["audio"]["bytes"]
                        data, sr = sf.read(io.BytesIO(blob), dtype="float32")
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        sf.write(str(wav_path), data, int(sr))
                    sink.write(json.dumps({
                        "id": clip_id,
                        "audio_path": str(wav_path),
                        "caption": row["caption"],
                        "audiocap_id": int(row["audiocap_id"]),
                        "youtube_id": row.get("youtube_id"),
                        "start_time": int(row.get("start_time", 0)),
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 500 == 0:
                        logging.info("export %d ...", n)
    return n


def _resolve_num(args: argparse.Namespace) -> int | None:
    """--num wins over deprecated --limit."""
    if args.num is not None:
        return args.num
    return args.limit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="Input jsonl: each line needs audio_path; optional id, caption.")
    p.add_argument("--out-dir", type=Path, help="Directory for *_WYZX_4ch.flac outputs.")
    p.add_argument("--manifest", type=Path, help="Output manifest jsonl (append, resumable).")
    p.add_argument("--export-audiocaps-jsonl", action="store_true",
                   help="Only export AudioCaps parquet -> mono wav + input jsonl; then exit.")
    p.add_argument("--audiocaps-root", type=Path, default=DEFAULT_PRIMARY / "datasets/audiocaps/snapshot")
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--out", type=Path, default=None, help="With --export-audiocaps-jsonl: output jsonl path.")
    p.add_argument("--wav-dir", type=Path, default=None, help="Mono wav dir for AudioCaps export.")
    p.add_argument("--target-fs", type=int, default=DEFAULT_FS)
    p.add_argument("--rt60-min", type=float, default=0.25)
    p.add_argument("--rt60-max", type=float, default=0.65)
    p.add_argument("--room-min", type=float, default=3.0, help="Min room length/width (m).")
    p.add_argument("--room-max", type=float, default=8.0)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="Base seed; per-item seed = seed + index.")
    p.add_argument(
        "--num",
        type=int,
        default=None,
        help="Max items to process this run (export rows or FOA clips). E.g. --num 1000 for a 1k sample.",
    )
    p.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)  # alias of --num
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cap = _resolve_num(args)

    if args.export_audiocaps_jsonl:
        out_jsonl = args.out or Path(f"/mnt/sdc/audio_dataset_tmp/audiocaps_{args.split}.jsonl")
        n = export_audiocaps_jsonl(
            args.audiocaps_root.resolve(), args.split, out_jsonl,
            wav_dir=args.wav_dir, limit=cap,
        )
        logging.info("Exported %d rows -> %s", n, out_jsonl)
        return

    if not args.input or not args.out_dir or not args.manifest:
        sys.exit("Synthesis requires --input, --out-dir, and --manifest (or use --export-audiocaps-jsonl).")

    records = list(iter_jsonl(args.input.resolve()))
    if args.num_shards > 1:
        records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard]

    manifest = args.manifest.resolve()
    if args.num_shards > 1:
        manifest = manifest.with_name(f"{manifest.stem}.shard{args.shard}{manifest.suffix}")

    done = load_done(manifest)
    todo = []
    for i, rec in enumerate(records):
        cid = rec.get("id") or rec.get("clip_id") or Path(rec["audio_path"]).stem
        if cid not in done:
            todo.append((i, rec, cid))
        if cap is not None and len(todo) >= cap:
            break

    logging.info(
        "input=%s cap=%s todo=%d (done=%d, jsonl_rows=%d) -> %s",
        args.input, cap if cap is not None else "all", len(todo), len(done), len(records), manifest,
    )
    if not todo:
        logging.info("Nothing to do.")
        return

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rt60_range = (args.rt60_min, args.rt60_max)
    room_range = (args.room_min, args.room_max)

    ok = skip = err = 0
    with manifest.open("a", encoding="utf-8") as sink, ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for idx, rec, cid in todo:
            seed = args.seed + idx
            fut = pool.submit(
                _worker,
                (rec, out_dir, args.target_fs, seed, rt60_range, room_range),
            )
            futures[fut] = cid

        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            st = row.get("status")
            if st == "ok":
                ok += 1
            elif st == "skipped":
                skip += 1
            else:
                err += 1
                logging.warning("%s: %s", row.get("id"), row.get("error", st))
            if n % 100 == 0 or n == len(todo):
                logging.info("  %d/%d ok=%d skip=%d err=%d", n, len(todo), ok, skip, err)

    logging.info("DONE ok=%d skip=%d err=%d -> %s", ok, skip, err, manifest)


if __name__ == "__main__":
    main()
