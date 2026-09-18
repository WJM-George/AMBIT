#!/usr/bin/env python3
"""Orchestrate a category-balanced spatial (FOA) dataset with pyroom.

Consumes the source pools from ``build_source_index.py`` and synthesizes ~200k
4-channel FOA clips with the engine in ``synthesize_foa_pyroom.py``:

  * Category balance (default): audio 80k, music 70k, speech 70k.
  * Mixing:  single-source 80%, 2-source 15%, multi-source (3-4) 5%.
  * Motion:  static 70%, dynamic (moving source) 30%   [per source].
  * Placement: full-circle azimuth + emphasis on the horizontal plane AND up/down
    elevation (so the VAE sees clear left/right/front/back AND above/below cues).
  * Room: randomized shoebox + RT60 (reverberant variety) for static sources;
    dynamic sources use a moving trajectory (room RIR cross-fade, or cheap pan).

Each clip writes:
  <out-dir>/<id>_WYZX_4ch.flac           ACN/SN3D [W,Y,Z,X]
  and a row in <manifest> with FULL spatial metadata (room, per-source label,
  azimuth/elevation/distance, motion + trajectory) that the caption refiner
  (dataset/captioning/refine_caption.py) turns into a spatial text caption.

Resumable (skip done ids), shardable (--num-shards/--shard), CPU-parallel (--jobs).

Run:
    uv run python dataset/synthesis/build_spatial_dataset.py \
        --sources-dir ${AMBIT_DATA_ROOT}/spatial_sources \
        --out-dir ${AMBIT_DATA_ROOT}/spatial_foa/audio \
        --manifest ${AMBIT_DATA_ROOT}/spatial_foa/manifest.jsonl \
        --audio 80000 --music 60000 --speech 60000 \
        --jobs 24

Clip count is driven ENTIRELY by --audio + --music + --speech (here 200000).
--total is only an optional sanity check (warns if it disagrees with the sum).

Smoke test (100 clips):
    uv run python dataset/synthesis/build_spatial_dataset.py ... --num 100 --jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from synthesize_foa_pyroom import (  # noqa: E402
    azimuth_words, elevation_words, distance_words, reverb_words, room_words,
    sample_room, az_el_dist_to_xyz, synthesize_foa_static_at,
    synthesize_foa_dynamic_room, synthesize_foa_dynamic_pan, mix_foa,
    _joint_peak_normalize,
)

LOG = logging.getLogger("spatial_dataset")
DEFAULT_FS = 48000


# ---------------------------------------------------------------- placement RNG

def _sample_elevation(rng: random.Random) -> float:
    """Emphasis: plane (small el) + meaningful up/down. Returns radians."""
    u = rng.random()
    if u < 0.55:
        deg = rng.uniform(-12, 12)        # plane
    elif u < 0.80:
        deg = rng.choice([1, -1]) * rng.uniform(12, 35)   # mid
    else:
        deg = rng.choice([1, -1]) * rng.uniform(35, 70)   # strong up/down
    return np.radians(deg)


def _sample_placement(rng: random.Random) -> tuple[float, float, float]:
    az = rng.uniform(-np.pi, np.pi)
    el = _sample_elevation(rng)
    dist = rng.uniform(0.8, 4.0)
    return az, el, dist


def _sample_end_placement(rng: random.Random, start) -> tuple[float, float, float]:
    """A noticeably different endpoint for a moving source."""
    az0, el0, d0 = start
    daz = rng.uniform(np.pi / 4, np.pi) * rng.choice([1, -1])
    az1 = ((az0 + daz + np.pi) % (2 * np.pi)) - np.pi
    el1 = _sample_elevation(rng)
    d1 = float(np.clip(d0 + rng.uniform(-1.5, 1.5), 0.8, 4.0))
    return az1, el1, d1


# Room sampling now lives in the engine (synthesize_foa_pyroom.sample_room),
# which draws from a library of acoustic archetypes (booth -> cathedral ->
# outdoor) with per-category weighting. Use that instead of a uniform box.


def _angle_words(az, el, dist) -> dict:
    return {
        "az_deg": round(float(np.degrees(az)), 1),
        "el_deg": round(float(np.degrees(el)), 1),
        "dist_m": round(float(dist), 2),
        "dir": azimuth_words(az),
        "elev": elevation_words(el),
        "dist_word": distance_words(dist),
    }


# --------------------------------------------------------------------- planning

def _load_pool(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plan_clips(pools: dict[str, list[dict]], targets: dict[str, int],
               pair_frac: float, multi_frac: float, dynamic_frac: float,
               rng: random.Random, id_prefix: str = "sp",
               start_index: int = 0) -> list[dict]:
    all_rows = [r for rows in pools.values() for r in rows]
    if not all_rows:
        raise RuntimeError("All source pools are empty. Run build_source_index.py first.")
    active_rows = [
        r
        for cat, rows in pools.items()
        if targets.get(cat, 0) > 0
        for r in rows
    ]
    extra_rows = active_rows or all_rows

    specs: list[dict] = []
    order = []
    for cat, n in targets.items():
        order += [cat] * n
    rng.shuffle(order)

    for i, cat in enumerate(order):
        pool = pools.get(cat) or all_rows
        if not pool:
            pool = all_rows
        u = rng.random()
        if u < multi_frac:
            n_src = rng.randint(3, 4)
        elif u < multi_frac + pair_frac:
            n_src = 2
        else:
            n_src = 1
        srcs = []
        # Primary source from the clip category; extras from active target pools.
        # This keeps speech out of synthetic audio/music runs when --speech 0.
        primary = rng.choice(pool)
        chosen = [primary] + [rng.choice(extra_rows) for _ in range(n_src - 1)]
        for s in chosen:
            motion = "dynamic" if rng.random() < dynamic_frac else "static"
            start = _sample_placement(rng)
            end = _sample_end_placement(rng, start) if motion == "dynamic" else None
            srcs.append({
                "path": s["path"], "dataset": s.get("dataset", ""),
                "category": s.get("category", cat), "label": s.get("label", ""),
                "motion": motion, "start": start, "end": end,
            })
        specs.append({
            "id": f"{id_prefix}_{start_index + i:07d}", "category": cat,
            "mix_type": {1: "single", 2: "pair"}.get(n_src, "multi"),
            "n_sources": n_src, "room": sample_room(rng, cat), "sources": srcs,
            "seed": rng.randint(0, 2**31 - 1),
        })
    return specs


# ----------------------------------------------------------------------- worker

def _load_mono(path: str, sr_out: int, clip_len: int, rng: random.Random) -> Optional[np.ndarray]:
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")  # [T, C]
    except Exception:  # noqa: BLE001
        return None
    mono = data.mean(axis=1)
    if sr != sr_out:
        try:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(int(sr), int(sr_out))
            mono = resample_poly(mono, sr_out // g, sr // g).astype(np.float32)
        except Exception:  # noqa: BLE001
            n_out = int(round(len(mono) * sr_out / sr))
            mono = np.interp(np.linspace(0, 1, n_out, endpoint=False),
                             np.linspace(0, 1, len(mono), endpoint=False), mono).astype(np.float32)
    if len(mono) >= clip_len:
        start = rng.randint(0, len(mono) - clip_len)
        mono = mono[start:start + clip_len]
    else:
        mono = np.pad(mono, (0, clip_len - len(mono)))
    return mono.astype(np.float32)


def _waypoints(mic, start, end, k: int):
    az0, el0, d0 = start
    az1, el1, d1 = end
    daz = ((az1 - az0 + np.pi) % (2 * np.pi)) - np.pi  # shortest arc
    pts = []
    for j in range(k):
        t = j / (k - 1)
        az = az0 + daz * t
        el = el0 + (el1 - el0) * t
        d = d0 + (d1 - d0) * t
        pts.append(az_el_dist_to_xyz(mic, az, el, d))
    return pts


def render_clip(spec: dict, sr: int, min_sec: float, max_sec: float,
                dynamic_mode: str, waypoints: int, out_dir: str) -> dict:
    rng = random.Random(spec["seed"])
    out_path = Path(out_dir) / f"{spec['id']}_WYZX_4ch.flac"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return {"id": spec["id"], "status": "skipped", "foa_path": str(out_path)}

    clip_len = int(rng.uniform(min_sec, max_sec) * sr)
    room = spec["room"]
    foas, src_meta = [], []
    for s in spec["sources"]:
        mono = _load_mono(s["path"], sr, clip_len, rng)
        if mono is None or float(np.max(np.abs(mono))) < 1e-5:
            continue
        if s["motion"] == "static":
            src_xyz = az_el_dist_to_xyz(room["mic"], *s["start"])
            foa = synthesize_foa_static_at(mono, sr, room["dim"], room["rt60"],
                                           room["max_order"], room["mic"], src_xyz)
            foa = foa[:, :clip_len]
            if foa.shape[1] < clip_len:
                foa = np.pad(foa, ((0, 0), (0, clip_len - foa.shape[1])))
            move = None
        else:
            if dynamic_mode == "pan":
                traj = [s["start"], s["end"]]
                foa = synthesize_foa_dynamic_pan(mono, sr, traj)
            else:
                wps = _waypoints(room["mic"], s["start"], s["end"], waypoints)
                # Cap RT60/order for moving sources: long-tail halls make huge
                # per-waypoint RIRs (K of them) -> keep dynamic synthesis affordable.
                dyn_rt60 = min(room["rt60"], 0.9)
                foa = synthesize_foa_dynamic_room(mono, sr, room["dim"], dyn_rt60,
                                                  min(room["max_order"], 8), room["mic"], wps)
            move = {"from": azimuth_words(s["start"][0]), "to": azimuth_words(s["end"][0])}
        foas.append(foa)
        meta = {"path": s["path"], "dataset": s["dataset"],
                "category": s["category"], "label": s["label"],
                "motion": s["motion"], "start": _angle_words(*s["start"])}
        if s["end"] is not None:
            meta["end"] = _angle_words(*s["end"])
            meta["move"] = move
        src_meta.append(meta)

    if not foas:
        return {"id": spec["id"], "status": "error", "error": "no usable sources"}

    foa = mix_foa(foas)
    foa = _joint_peak_normalize(foa, peak=0.9)
    foa = np.clip(foa, -1.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), foa.T, sr, subtype="PCM_24")

    return {
        "id": spec["id"], "status": "ok", "foa_path": str(out_path),
        "category": spec["category"], "mix_type": spec["mix_type"],
        "n_sources": len(src_meta), "sample_rate": sr,
        "duration_sec": round(foa.shape[1] / sr, 3),
        "channel_layout": "WYZX_ACN_SN3D",
        "dynamic_mode": dynamic_mode if any(s["motion"] == "dynamic" for s in spec["sources"]) else None,
        "room": {"type": room.get("type"), "desc": room.get("desc"),
                 "dim": [round(x, 2) for x in room["dim"]],
                 "rt60": round(room["rt60"], 3), "reverb": reverb_words(room["rt60"]),
                 "free_field": bool(room.get("free_field"))},
        "sources": src_meta,
        "spatial_caption": _template_caption(spec["category"], src_meta, room),
    }


def _template_caption(category: str, sources: list[dict], room: dict) -> str:
    """Deterministic seed caption (content + space); refiner polishes it later."""
    parts = []
    for s in sources:
        lab = (s.get("label") or s.get("category") or "a sound").strip().rstrip(".")
        st = s["start"]
        where = f"from the {st['dir']}"
        if st["elev"] != "level":
            where += f" and {st['elev']}"
        where += f", {st['dist_word']}"
        if s["motion"] == "dynamic" and s.get("move"):
            where = f"moving from the {s['move']['from']} to the {s['move']['to']}"
        parts.append(f"{lab} {where}")
    joined = "; ".join(parts)
    where_room = "Recorded outdoors" if room.get("free_field") else f"Recorded in {room_words(room)}"
    return f"{joined}. {where_room}."


# ------------------------------------------------------------------------- main

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--total", type=int, default=None,
                    help="Optional sanity check ONLY. Clip count = audio+music+speech; "
                         "if --total is set and != that sum, a warning is logged.")
    ap.add_argument("--audio", type=int, default=80000)
    ap.add_argument("--music", type=int, default=60000)
    ap.add_argument("--speech", type=int, default=60000)
    ap.add_argument("--id-prefix", default="sp",
                    help="Prefix for generated clip IDs. Default keeps legacy sp_0000000 IDs.")
    ap.add_argument("--pair-frac", type=float, default=0.15)
    ap.add_argument("--multi-frac", type=float, default=0.05)
    ap.add_argument("--dynamic-frac", type=float, default=0.30)
    ap.add_argument("--dynamic-mode", choices=["room", "pan"], default="room")
    ap.add_argument("--dynamic-waypoints", type=int, default=4)
    ap.add_argument("--target-fs", type=int, default=DEFAULT_FS)
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--num", type=int, default=None, help="Cap clips this run (testing).")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempt-rounds", type=int, default=1,
                    help="Retry planning extra unique clips until requested ok+skipped count is reached.")
    args = ap.parse_args()

    pools = {
        "audio": _load_pool(args.sources_dir / "sources_audio.jsonl"),
        "music": _load_pool(args.sources_dir / "sources_music.jsonl"),
        "speech": _load_pool(args.sources_dir / "sources_speech.jsonl"),
    }
    LOG.info("source pools: audio=%d music=%d speech=%d",
             len(pools["audio"]), len(pools["music"]), len(pools["speech"]))

    targets = {"audio": args.audio, "music": args.music, "speech": args.speech}
    requested_total = sum(targets.values())
    if args.total is not None and args.total != requested_total:
        LOG.warning("--total=%d ignored: per-category counts sum to %d "
                    "(audio=%d music=%d speech=%d).", args.total, requested_total,
                    targets["audio"], targets["music"], targets["speech"])
    LOG.info("target clip count = %d (audio=%d music=%d speech=%d)",
             requested_total, targets["audio"], targets["music"], targets["speech"])
    rng = random.Random(args.seed)
    specs = plan_clips(pools, targets, args.pair_frac, args.multi_frac,
                       args.dynamic_frac, rng, id_prefix=args.id_prefix)

    if args.num_shards > 1:
        specs = [s for i, s in enumerate(specs) if i % args.num_shards == args.shard]
    if args.num is not None:
        specs = specs[: args.num]

    manifest = args.manifest
    if args.num_shards > 1:
        manifest = manifest.with_name(f"{manifest.stem}.shard{args.shard}{manifest.suffix}")
    done = set()
    done_by_category_existing = {cat: 0 for cat in targets}
    if manifest.exists():
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") in ("ok", "skipped"):
                        done.add(r["id"])
                        cat = r.get("category")
                        if cat in done_by_category_existing:
                            done_by_category_existing[cat] += 1
                except json.JSONDecodeError:
                    continue
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    ok = skip = err = 0
    total_done = len(done)
    next_index = requested_total
    round_idx = 0
    target_this_run = len(specs)
    target_by_category = {cat: 0 for cat in targets}
    for s in specs:
        if s.get("category") in target_by_category:
            target_by_category[s["category"]] += 1
    done_by_category = dict(done_by_category_existing)
    while round_idx < max(args.max_attempt_rounds, 1):
        todo = [s for s in specs if s["id"] not in done]
        LOG.info("planned=%d todo=%d (done=%d) -> %s", len(specs), len(todo), len(done), args.out_dir)
        if not todo:
            LOG.info("Nothing to do.")
            break
        round_ok = round_skip = round_err = 0
        with manifest.open("a", encoding="utf-8") as sink, \
                ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(render_clip, s, args.target_fs, args.min_seconds, args.max_seconds,
                            args.dynamic_mode, args.dynamic_waypoints, str(args.out_dir)): s["id"]
                for s in todo
            }
            for n, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                st = row.get("status")
                if st in ("ok", "skipped"):
                    done.add(row["id"])
                    total_done += 1
                    cat = row.get("category")
                    if cat in done_by_category:
                        done_by_category[cat] += 1
                ok += st == "ok"
                skip += st == "skipped"
                err += st not in ("ok", "skipped")
                round_ok += st == "ok"
                round_skip += st == "skipped"
                round_err += st not in ("ok", "skipped")
                if n % 200 == 0 or n == len(todo):
                    LOG.info("  %d/%d ok=%d skip=%d err=%d", n, len(todo), ok, skip, err)
        if total_done >= target_this_run:
            break
        shortfall = target_this_run - total_done
        if shortfall <= 0:
            break
        round_idx += 1
        if round_idx >= args.max_attempt_rounds:
            break
        LOG.info("retry round %d: shortfall=%d after ok=%d skip=%d err=%d",
                 round_idx, shortfall, round_ok, round_skip, round_err)
        retry_targets = {
            cat: max(0, int(target) - int(done_by_category.get(cat, 0)))
            for cat, target in target_by_category.items()
        }
        if sum(retry_targets.values()) != shortfall:
            LOG.warning("category shortfall %s sums to %d but total shortfall is %d",
                        retry_targets, sum(retry_targets.values()), shortfall)
        rng = random.Random(args.seed + 1000003 * round_idx)
        retry_start_index = next_index
        if args.num_shards > 1:
            retry_start_index = requested_total * (1 + args.shard + (round_idx - 1) * args.num_shards)
        specs = plan_clips(pools, retry_targets, args.pair_frac, args.multi_frac,
                           args.dynamic_frac, rng, id_prefix=args.id_prefix,
                           start_index=retry_start_index)
        next_index = retry_start_index + len(specs)

    LOG.info("DONE ok=%d skip=%d err=%d -> %s (manifest %s)", ok, skip, err, args.out_dir, manifest)


if __name__ == "__main__":
    main()
