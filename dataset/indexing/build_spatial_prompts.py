#!/usr/bin/env python3
"""Build text prompts for spatial-audio datasets from their NATIVE metadata.

No audio/vision model needed: Spatial LibriSpeech and MRSDrama both ship transcription +
spatial information. This emits a captions jsonl per dataset that
``dataset_4ch.load_caption_map`` consumes during pre-encode, baking the prompt into each
latent's metadata (the Stage-2 text condition).

SLS (FOA): metadata.parquet -> per-sample description (voice + direction + distance +
reverb) with the spoken words appended. Key = zero-padded sample_id (matches <sid>.flac).
  Azimuth convention (verified against the FOA Y channel): positive = LEFT, negative =
  RIGHT, +/-180 deg = BEHIND, ~0 = FRONT.

MRSDrama (binaural): each scene's data.json already has English ``textual_prompt`` (speaker
position) + ``scene_prompt`` (room). Key = ABSOLUTE wav path (segment_*.wav repeats across
scenes, so path-based matching is mandatory).

Run (CPU only, fast):
    uv run python dataset/indexing/build_spatial_prompts.py sls \
        --parquet /mnt/sdb/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet \
        --out /mnt/sdb/audio_dataset/datasets/spatial_librispeech/sls_prompts.jsonl

    uv run python dataset/indexing/build_spatial_prompts.py mrsdrama \
        --root /mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot \
        --out /mnt/sdd/audio_dataset/datasets/mrsdrama/mrsdrama_prompts.jsonl

Then point the dataset entries in local_4ch_preencode.json at these files via "captions".
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


# --------------------------------------------------------------------------- SLS

_DIR8 = ["front", "front-left", "left", "rear-left", "behind",
         "rear-right", "right", "front-right"]


def azimuth_words(az_rad: float) -> str:
    """8-way horizontal direction. az>0 = left, az<0 = right, +/-pi = behind (SLS convention)."""
    a = (math.degrees(az_rad) + 360.0) % 360.0
    idx = int(((a + 22.5) % 360.0) // 45.0)
    return _DIR8[idx]


def elevation_words(el_rad: float) -> str:
    el = math.degrees(el_rad)
    if el > 20:
        return "above"
    if el < -20:
        return "below"
    return ""


def reverb_words(t30_ms_list) -> str:
    vals = [x for x in t30_ms_list if isinstance(x, (int, float)) and x == x]
    if not vals:
        return "room"
    t30 = statistics.median(vals) / 1000.0
    if t30 < 0.3:
        return "fairly dry room"
    if t30 < 0.6:
        return "moderately reverberant room"
    return "highly reverberant room"


def build_sls(args) -> int:
    import pyarrow.parquet as pq

    cols = [
        "sample_id", "speech/azimuth", "speech/elevation", "speech/distance",
        "speech/librispeech_metadata/transcription",
        "speech/librispeech_metadata/reader_sex", "acoustics/t30_ms",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    pf = pq.ParquetFile(args.parquet)
    with out.open("w", encoding="utf-8") as sink:
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            for r in batch.to_pylist():
                sid = r["sample_id"]
                sex = {"M": "male", "F": "female"}.get(
                    r.get("speech/librispeech_metadata/reader_sex"), "")
                horiz = azimuth_words(r["speech/azimuth"])
                vert = elevation_words(r["speech/elevation"])
                where = horiz + (f" and {vert}" if vert else "")
                loc = "coming from behind" if horiz == "behind" and not vert else f"coming from the {where}"
                dist = r.get("speech/distance")
                rev = reverb_words(r.get("acoustics/t30_ms") or [])
                voice = f"a {sex} voice speaking" if sex else "a voice speaking"
                dist_str = f", about {dist:.1f} meters away" if isinstance(dist, (int, float)) else ""
                prompt = f"{voice}, {loc}{dist_str}, in a {rev}."
                trans = (r.get("speech/librispeech_metadata/transcription") or "").strip()
                if trans and not args.no_text:
                    prompt += f' Spoken words: "{trans}".'
                sink.write(json.dumps({"id": f"{sid:06d}", "caption": prompt},
                                      ensure_ascii=False) + "\n")
                n += 1
                if n % 20000 == 0:
                    print(f"[sls] {n} ...")
    print(f"[sls] wrote {n} prompts -> {out}")
    return n


# ---------------------------------------------------------------------- MRSDrama

def build_mrsdrama(args) -> int:
    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_files = sorted(root.glob("*/data.json"))
    if not data_files:
        raise FileNotFoundError(f"No */data.json under {root}")

    n = missing = 0
    with out.open("w", encoding="utf-8") as sink:
        for dj in data_files:
            try:
                rows = json.loads(dj.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for r in rows:
                wp = r.get("wav_path")
                if not wp:
                    continue
                # wav_path is relative to the snapshot root and includes the scene dir.
                abs_wav = Path(__import__("os").path.normpath(root / wp.lstrip("./")))
                tp = (r.get("textual_prompt") or "").strip()
                sp = (r.get("scene_prompt") or "").strip()
                parts = []
                if tp:
                    parts.append(tp if tp.endswith(".") else tp + ".")
                if sp:
                    parts.append("Scene: " + (sp if sp.endswith(".") else sp + "."))
                if args.include_text:
                    raw = "".join(r.get("raw_txt") or []).strip()
                    if raw:
                        parts.append(f'Spoken words: "{raw}".')
                caption = " ".join(parts).strip()
                if not caption:
                    missing += 1
                    continue
                sink.write(json.dumps({"path": str(abs_wav), "caption": caption},
                                      ensure_ascii=False) + "\n")
                n += 1
                if n % 10000 == 0:
                    print(f"[mrsdrama] {n} ...")
    print(f"[mrsdrama] wrote {n} prompts ({missing} segments had no text) -> {out}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sls", help="Build SLS prompts from metadata.parquet")
    ps.add_argument("--parquet", default="/mnt/sdb/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet")
    ps.add_argument("--out", default="/mnt/sdb/audio_dataset/datasets/spatial_librispeech/sls_prompts.jsonl")
    ps.add_argument("--no-text", action="store_true", help="Omit the spoken-words transcription.")
    ps.set_defaults(func=build_sls)

    pm = sub.add_parser("mrsdrama", help="Build MRSDrama prompts from per-scene data.json")
    pm.add_argument("--root", default="/mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot")
    pm.add_argument("--out", default="/mnt/sdd/audio_dataset/datasets/mrsdrama/mrsdrama_prompts.jsonl")
    pm.add_argument("--include-text", action="store_true",
                    help="Append the (Chinese) raw_txt as spoken words. Off by default (T5 is English-centric).")
    pm.set_defaults(func=build_mrsdrama)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
