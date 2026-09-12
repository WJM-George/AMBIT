#!/usr/bin/env python3
"""Render planned LibriTTS/HiFi-TTS speech clips into synthetic FOA FLAC.

This script is intentionally shard/micro-batch oriented:
  parquet audio bytes -> temporary source cache -> pyroom FOA render -> validate -> delete cache.

It reuses the existing stable-audio-tools FOA engine and writes 48 kHz, 4-channel,
24-bit FLAC with WYZX_ACN_SN3D layout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


REPO_ROOT = Path("/home/tanhe/dataset_storage/stable-audio-tools")
SYNTH_DIR = REPO_ROOT / "dataset/synthesis"
if str(SYNTH_DIR) not in sys.path:
    sys.path.insert(0, str(SYNTH_DIR))

from synthesize_foa_pyroom import (  # noqa: E402
    _joint_peak_normalize,
    az_el_dist_to_xyz,
    mix_foa,
    synthesize_foa_dynamic_pan,
    synthesize_foa_dynamic_room,
    synthesize_foa_static_at,
)


DATASET_ROOTS = {
    "libritts": Path("/mnt/sdc/speech_dataset/mythicinfinity__libritts"),
    "hifi_tts": Path("/mnt/sdc/speech_dataset/MikhailT__hifi-tts"),
}

SOURCE_INDEX: dict[tuple[str, str], dict[str, Any]] | None = None

DIR_TO_AZ_DEG = {
    "front": 0.0,
    "front-left": 45.0,
    "left": 90.0,
    "rear-left": 135.0,
    "behind": 180.0,
    "rear-right": -135.0,
    "right": -90.0,
    "front-right": -45.0,
}
ELEV_TO_DEG = {"level": 0.0, "above": 30.0, "below": -30.0}
ROOM_ARCHETYPES = {
    "vocal_booth": {"L": (1.8, 2.8), "W": (1.6, 2.6), "H": (2.0, 2.6), "rt60": (0.12, 0.25), "order": (6, 10)},
    "studio": {"L": (4.0, 7.0), "W": (3.0, 5.0), "H": (2.6, 3.2), "rt60": (0.18, 0.35), "order": (8, 12)},
    "small_room": {"L": (3.0, 5.0), "W": (3.0, 4.5), "H": (2.4, 3.0), "rt60": (0.25, 0.45), "order": (8, 12)},
    "living_room": {"L": (4.0, 7.0), "W": (3.5, 6.0), "H": (2.5, 3.2), "rt60": (0.30, 0.55), "order": (8, 12)},
    "office": {"L": (4.0, 8.0), "W": (4.0, 7.0), "H": (2.6, 3.2), "rt60": (0.40, 0.70), "order": (8, 12)},
    "classroom": {"L": (7.0, 12.0), "W": (6.0, 10.0), "H": (3.0, 4.0), "rt60": (0.50, 0.90), "order": (8, 12)},
    "bathroom_tiled": {"L": (2.0, 3.5), "W": (2.0, 3.5), "H": (2.4, 3.0), "rt60": (0.60, 1.10), "order": (10, 14)},
    "corridor": {"L": (8.0, 20.0), "W": (1.5, 2.5), "H": (2.6, 3.5), "rt60": (0.60, 1.30), "order": (8, 12)},
    "gymnasium": {"L": (18.0, 30.0), "W": (12.0, 22.0), "H": (6.0, 10.0), "rt60": (1.20, 2.00), "order": (6, 10)},
    "concert_hall": {"L": (15.0, 28.0), "W": (12.0, 22.0), "H": (8.0, 14.0), "rt60": (1.40, 2.40), "order": (6, 10)},
    "cathedral": {"L": (20.0, 40.0), "W": (14.0, 28.0), "H": (12.0, 22.0), "rt60": (2.20, 3.80), "order": (5, 8)},
    "outdoor": {"L": (8.0, 16.0), "W": (8.0, 16.0), "H": (6.0, 10.0), "rt60": (0.30, 0.50), "order": (0, 0)},
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_plan_rows(paths: list[Path], *, start: int, count: int, partition: str | None, shard: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for path in paths:
        for row in iter_jsonl(path):
            if partition and row.get("partition") != partition:
                continue
            if shard is not None and int(row.get("shard", -1)) != shard:
                continue
            if idx < start:
                idx += 1
                continue
            if len(rows) >= count:
                return rows
            rows.append(row)
            idx += 1
    return rows


def apply_output_root(rows: list[dict[str, Any]], output_root: Path | None) -> list[dict[str, Any]]:
    """Redirect planned FOA paths into an alternate root for smoke tests.

    Production runs should normally leave this unset so the planned sdb/sdc paths
    are used. Smoke tests pass an isolated root to avoid polluting final audio dirs.
    """
    if output_root is None:
        return rows
    audio_root = output_root / "audio"
    out = []
    for row in rows:
        nr = dict(row)
        nr["foa_path"] = str(audio_root / f"{row['id']}_WYZX_4ch.flac")
        out.append(nr)
    return out


def audio_key_for_row(dataset: str, row: dict[str, Any]) -> str | None:
    if dataset == "libritts":
        return row.get("id")
    audio = row.get("audio") or {}
    path = audio.get("path") or row.get("file") or ""
    stem = Path(path).stem
    return f"hifi_tts_{stem}" if stem else None


def extension_from_audio(audio: dict[str, Any], fallback: str | None = None) -> str:
    suffix = Path((audio or {}).get("path") or fallback or "").suffix.lower()
    if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        return suffix
    data = bytes((audio or {}).get("bytes") or b"")
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"RIFF":
        return ".wav"
    return ".bin"


def load_source_index(path: Path | None) -> dict[tuple[str, str], dict[str, Any]] | None:
    if path is None:
        return None
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[(row["source_dataset"], row["source_id"])] = row
    return index


def extract_sources_for_batch(rows: list[dict[str, Any]], cache_dir: Path, source_index: dict[tuple[str, str], dict[str, Any]] | None = None) -> dict[str, Path]:
    if source_index is not None:
        return extract_sources_for_batch_indexed(rows, cache_dir, source_index)

    needed_by_ds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        needed_by_ds[row["source_dataset"]][row["source_id"]] = row

    found: dict[str, Path] = {}
    for dataset, needed in needed_by_ds.items():
        root = DATASET_ROOTS[dataset]
        parquet_files = sorted(root.rglob("*.parquet"))
        remaining = set(needed)
        cols = ["audio"]
        if dataset == "libritts":
            cols += ["id", "speaker_id", "path", "text_normalized"]
        else:
            cols += ["speaker", "file", "duration", "text_normalized"]
        out_dir = cache_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for pf in parquet_files:
            if not remaining:
                break
            table = pq.read_table(pf, columns=cols)
            for item in table.to_pylist():
                key = audio_key_for_row(dataset, item)
                if key not in remaining:
                    continue
                audio = item.get("audio") or {}
                data = audio.get("bytes")
                if not data:
                    continue
                plan = needed[key]
                ext = extension_from_audio(audio, item.get("file") or item.get("path"))
                out_path = out_dir / f"{plan['id']}__{key}{ext}"
                out_path.write_bytes(bytes(data))
                found[plan["id"]] = out_path
                remaining.remove(key)
            del table
        if remaining:
            raise RuntimeError(f"missing {dataset} source ids: {sorted(remaining)[:10]}")
    return found


def extract_sources_for_batch_indexed(rows: list[dict[str, Any]], cache_dir: Path, source_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Path]:
    """Extract source audio by grouping needed rows by parquet file.

    This still reads the audio column for a file, but it only touches parquet
    files containing the current batch rather than scanning the full dataset.
    """
    by_file: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["source_dataset"], row["source_id"])
        hit = source_index.get(key)
        if hit is None:
            raise RuntimeError(f"source index miss: {key}")
        item = dict(row)
        item["_row_index"] = int(hit["row_index"])
        item["_parquet_path"] = hit["parquet_path"]
        by_file[Path(hit["parquet_path"])].append(item)

    found: dict[str, Path] = {}
    by_group: dict[tuple[Path, int | None], list[dict[str, Any]]] = defaultdict(list)
    for parquet_path, file_rows in by_file.items():
        for r in file_rows:
            rg = r.get("row_group")
            by_group[(parquet_path, int(rg) if rg is not None else None)].append(r)

    parquet_handles: dict[Path, pq.ParquetFile] = {}
    for (parquet_path, row_group), group_rows in by_group.items():
        if row_group is None:
            wanted = {int(r["_row_index"]): r for r in group_rows}
            table = pq.read_table(parquet_path, columns=["audio"])
            audio_rows = table.to_pylist()
        else:
            pf = parquet_handles.get(parquet_path)
            if pf is None:
                pf = pq.ParquetFile(parquet_path)
                parquet_handles[parquet_path] = pf
            wanted = {int(r["row_in_group"]): r for r in group_rows}
            table = pf.read_row_group(row_group, columns=["audio"])
            audio_rows = table.to_pylist()
        for row_index, plan in wanted.items():
            audio = audio_rows[row_index].get("audio") or {}
            data = audio.get("bytes")
            if not data:
                raise RuntimeError(f"empty audio bytes for {plan['id']} in {parquet_path}:{row_index}")
            out_dir = cache_dir / plan["source_dataset"]
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = extension_from_audio(audio, audio.get("path"))
            out_path = out_dir / f"{plan['id']}__{plan['source_id']}{ext}"
            out_path.write_bytes(bytes(data))
            found[plan["id"]] = out_path
        del table, audio_rows
    return found


def pos_to_tuple(pos: dict[str, Any]) -> tuple[float, float, float]:
    az = math.radians(DIR_TO_AZ_DEG.get(pos.get("dir", "front"), 0.0))
    el = math.radians(ELEV_TO_DEG.get(pos.get("elev", "level"), 0.0))
    dist = float(pos.get("dist_m") or 2.0)
    return az, el, dist


def room_to_engine_room(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    info = row.get("room") or {}
    rtype = info.get("type") or "small_room"
    arch = ROOM_ARCHETYPES.get(rtype, ROOM_ARCHETYPES["small_room"])
    L = rng.uniform(*arch["L"])
    W = rng.uniform(*arch["W"])
    H = rng.uniform(*arch["H"])
    rt60 = rng.uniform(*arch["rt60"])
    max_order = rng.randint(*arch["order"])
    mic = (rng.uniform(0.4 * L, 0.6 * L), rng.uniform(0.4 * W, 0.6 * W), rng.uniform(1.2, min(1.8, H - 0.4)))
    return {
        "type": rtype,
        "desc": info.get("desc") or rtype,
        "dim": (L, W, H),
        "rt60": rt60,
        "max_order": max_order,
        "mic": mic,
        "free_field": rtype == "outdoor",
    }


def load_mono(path: Path, sr_out: int, clip_len: int, rng: random.Random) -> np.ndarray | None:
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception:
        return None
    mono = data.mean(axis=1)
    if sr != sr_out:
        try:
            from scipy.signal import resample_poly
            g = math.gcd(int(sr), int(sr_out))
            mono = resample_poly(mono, sr_out // g, sr // g).astype(np.float32)
        except Exception:
            n_out = int(round(len(mono) * sr_out / sr))
            mono = np.interp(np.linspace(0, 1, n_out, endpoint=False), np.linspace(0, 1, len(mono), endpoint=False), mono).astype(np.float32)
    if len(mono) >= clip_len:
        start = rng.randint(0, len(mono) - clip_len)
        mono = mono[start:start + clip_len]
    else:
        mono = np.pad(mono, (0, clip_len - len(mono)))
    return mono.astype(np.float32)


def waypoints(mic, start, end, k: int):
    az0, el0, d0 = start
    az1, el1, d1 = end
    daz = ((az1 - az0 + np.pi) % (2 * np.pi)) - np.pi
    pts = []
    for j in range(k):
        t = j / max(1, k - 1)
        az = az0 + daz * t
        el = el0 + (el1 - el0) * t
        d = d0 + (d1 - d0) * t
        pts.append(az_el_dist_to_xyz(mic, az, el, d))
    return pts


def render_one(task: tuple[dict[str, Any], str, int, float, float, int, str, int]) -> dict[str, Any]:
    row, src_path, target_fs, min_seconds, max_seconds, seed, dynamic_mode, waypoints_count = task
    rng = random.Random(seed)
    out_path = Path(row["foa_path"])
    if out_path.exists() and out_path.stat().st_size > 1024:
        return {
            "id": row["id"], "status": "skipped", "foa_path": str(out_path),
            "category": "speech", "mix_type": row.get("mix_type", "single"),
            "n_sources": 1, "sample_rate": target_fs,
            "channel_layout": "WYZX_ACN_SN3D",
            "dynamic_mode": row.get("dynamic_mode"), "caption": row.get("caption"),
            "text": row.get("text"), "normalized_text": row.get("normalized_text"),
            "source_dataset": row.get("source_dataset"), "source_id": row.get("source_id"),
            "speaker": row.get("speaker"), "split": row.get("split"),
            "partition": row.get("partition"), "shard": row.get("shard"),
            "room": row.get("room"), "sources": row.get("sources") or [],
        }
    clip_len = int(rng.uniform(min_seconds, max_seconds) * target_fs)
    mono = load_mono(Path(src_path), target_fs, clip_len, rng)
    if mono is None or float(np.max(np.abs(mono))) < 1e-6:
        return {"id": row["id"], "status": "error", "error": "empty_or_unreadable_source", "source_audio_path": src_path}

    room = room_to_engine_room(row, rng)
    src = (row.get("sources") or [{}])[0]
    start = pos_to_tuple(src.get("start") or {})
    motion = src.get("motion") or ("dynamic" if row.get("dynamic_mode") else "static")
    try:
        if motion == "dynamic" and src.get("end"):
            end = pos_to_tuple(src.get("end") or {})
            if dynamic_mode == "pan":
                foa = synthesize_foa_dynamic_pan(mono, target_fs, [start, end])
            else:
                wps = waypoints(room["mic"], start, end, waypoints_count)
                foa = synthesize_foa_dynamic_room(
                    mono,
                    target_fs,
                    room["dim"],
                    min(float(room["rt60"]), 0.9),
                    min(int(room["max_order"]), 8),
                    room["mic"],
                    wps,
                )
            dyn = dynamic_mode
        else:
            src_xyz = az_el_dist_to_xyz(room["mic"], *start)
            foa = synthesize_foa_static_at(mono, target_fs, room["dim"], room["rt60"], room["max_order"], room["mic"], src_xyz)
            dyn = None
        foa = foa[:, :clip_len]
        if foa.shape[1] < clip_len:
            foa = np.pad(foa, ((0, 0), (0, clip_len - foa.shape[1])))
        foa = mix_foa([foa])
        foa = _joint_peak_normalize(foa, peak=0.9)
        foa = np.clip(foa, -1.0, 1.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), foa.T, target_fs, subtype="PCM_24")
        return {
            "id": row["id"],
            "status": "ok",
            "foa_path": str(out_path),
            "category": "speech",
            "mix_type": row.get("mix_type", "single"),
            "n_sources": 1,
            "sample_rate": target_fs,
            "duration_sec": round(float(foa.shape[1]) / target_fs, 3),
            "channel_layout": "WYZX_ACN_SN3D",
            "dynamic_mode": dyn,
            "caption": row.get("caption"),
            "text": row.get("text"),
            "normalized_text": row.get("normalized_text"),
            "source_dataset": row.get("source_dataset"),
            "source_id": row.get("source_id"),
            "speaker": row.get("speaker"),
            "split": row.get("split"),
            "partition": row.get("partition"),
            "shard": row.get("shard"),
            "room": {"type": row.get("room", {}).get("type"), "desc": row.get("room", {}).get("desc"), "reverb": row.get("room", {}).get("reverb")},
            "sources": row.get("sources") or [],
        }
    except Exception as exc:
        return {"id": row["id"], "status": "error", "error": repr(exc)[:1000], "source_audio_path": src_path, "foa_path": str(out_path)}


def ffprobe(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels,duration,bits_per_raw_sample", "-of", "json", str(path)],
        text=True,
        capture_output=True,
        timeout=20,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()[:500]}
    data = json.loads(proc.stdout)
    stream = (data.get("streams") or [{}])[0]
    return {
        "ok": True,
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration_sec": float(stream.get("duration") or 0),
        "bits_per_raw_sample": stream.get("bits_per_raw_sample"),
    }


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--caption-out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/mnt/sdc/speech_dataset/spatial_speech_foa_tts_v1_work"))
    ap.add_argument("--output-root", type=Path, default=None,
                    help="Optional root that overrides each row's planned foa_path; intended for smoke tests.")
    ap.add_argument("--source-index", type=Path, default=None,
                    help="Optional jsonl built by build_tts_parquet_source_index.py.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--partition", choices=["sdb", "sdc"], default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target-fs", type=int, default=48000)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--dynamic-mode", choices=["room", "pan"], default="room")
    ap.add_argument("--dynamic-waypoints", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=150.0)
    ap.add_argument("--validate-samples", type=int, default=20)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.caption_out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_plan_rows(args.plan, start=args.start, count=args.num, partition=args.partition, shard=args.shard)
    rows = apply_output_root(rows, args.output_root)
    if not rows:
        raise RuntimeError("no plan rows selected")

    done = set()
    if args.manifest.exists():
        for r in iter_jsonl(args.manifest):
            if r.get("status") in {"ok", "skipped"}:
                done.add(r.get("id"))

    run_summary = {
        "phase": "render_tts_speech_foa",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "selected_rows": len(rows),
        "start": args.start,
        "num": args.num,
        "partition": args.partition,
        "shard": args.shard,
        "micro_batch": args.micro_batch,
        "jobs": args.jobs,
        "min_free_gb": args.min_free_gb,
        "batches": [],
    }
    progress_path = args.work_dir / f"render_progress_{args.partition or 'all'}_{args.start}_{args.num}.json"
    ok = skipped = err = 0
    validation_rows: list[dict[str, Any]] = []
    source_index = load_source_index(args.source_index)

    with args.manifest.open("a", encoding="utf-8") as manifest_sink, args.caption_out.open("a", encoding="utf-8") as cap_sink:
        for b0 in range(0, len(rows), args.micro_batch):
            batch = [r for r in rows[b0:b0 + args.micro_batch] if r.get("id") not in done]
            if not batch:
                continue
            roots = {Path(r["foa_path"]).anchor + Path(r["foa_path"]).parts[1] if Path(r["foa_path"]).is_absolute() else str(Path.cwd()) for r in batch}
            for r in batch:
                if free_gb(Path(r["foa_path"]).parent.parent) < args.min_free_gb:
                    raise RuntimeError(f"free space below threshold near {r['foa_path']}")

            cache_dir = Path(tempfile.mkdtemp(prefix=f"tts_foa_src_{b0:07d}_", dir=str(args.work_dir)))
            cache_bytes = 0
            batch_summary = {"batch_start": b0, "batch_rows": len(batch), "cache_dir": str(cache_dir)}
            try:
                source_paths = extract_sources_for_batch(batch, cache_dir, source_index)
                cache_bytes = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
                tasks = []
                for i, row in enumerate(batch):
                    tasks.append((row, str(source_paths[row["id"]]), args.target_fs, args.min_seconds, args.max_seconds, args.seed + args.start + b0 + i, args.dynamic_mode, args.dynamic_waypoints))
                with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                    futs = [pool.submit(render_one, task) for task in tasks]
                    for fut in as_completed(futs):
                        rec = fut.result()
                        manifest_sink.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                        manifest_sink.flush()
                        st = rec.get("status")
                        if st in {"ok", "skipped"}:
                            done.add(rec.get("id"))
                            if st == "ok":
                                ok += 1
                            else:
                                skipped += 1
                            cap_rec = {
                                "id": rec.get("id"),
                                "foa_path": rec.get("foa_path"),
                                "audio_path": rec.get("foa_path"),
                                "caption": rec.get("caption"),
                                "stage": "template",
                                "category": "speech",
                                "mix_type": rec.get("mix_type"),
                                "n_sources": rec.get("n_sources"),
                                "source_dataset": rec.get("source_dataset"),
                                "speaker": rec.get("speaker"),
                                "dynamic_mode": rec.get("dynamic_mode"),
                                "sample_rate": rec.get("sample_rate"),
                                "duration_sec": rec.get("duration_sec"),
                                "channel_layout": rec.get("channel_layout"),
                                "text": rec.get("text"),
                                "normalized_text": rec.get("normalized_text"),
                            }
                            cap_sink.write(json.dumps(cap_rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                            cap_sink.flush()
                            if len(validation_rows) < args.validate_samples:
                                probe = ffprobe(Path(rec["foa_path"]))
                                validation_rows.append({"id": rec.get("id"), "foa_path": rec.get("foa_path"), "caption": rec.get("caption"), "probe": probe})
                        else:
                            err += 1
                batch_summary.update({"ok_total": ok, "skipped_total": skipped, "err_total": err, "cache_bytes": cache_bytes})
            finally:
                shutil.rmtree(cache_dir, ignore_errors=True)
                batch_summary["cache_deleted"] = not cache_dir.exists()
                run_summary["batches"].append(batch_summary)
            progress = dict(run_summary)
            progress.update({
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "ok": ok,
                "skipped": skipped,
                "errors": err,
                "done_total": ok + skipped + err,
                "remaining_estimate": max(0, len(rows) - (ok + skipped + err)),
                "manifest": str(args.manifest),
                "caption_out": str(args.caption_out),
            })
            progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"PROGRESS partition={args.partition or 'all'} done={ok + skipped + err}/{len(rows)} "
                f"ok={ok} skipped={skipped} err={err} cache_deleted={batch_summary.get('cache_deleted')}",
                flush=True,
            )

    run_summary.update({
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "ok": ok,
        "skipped": skipped,
        "errors": err,
        "dataset_counts": dict(Counter(r.get("source_dataset") for r in rows)),
        "motion_counts": dict(Counter("dynamic" if r.get("dynamic_mode") else "static" for r in rows)),
        "validation_samples": validation_rows,
        "manifest": str(args.manifest),
        "caption_out": str(args.caption_out),
        "progress": str(progress_path),
    })
    summary_path = args.work_dir / f"render_summary_{args.partition or 'all'}_{args.start}_{args.num}.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print(f"SUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()
