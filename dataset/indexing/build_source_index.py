#!/usr/bin/env python3
"""Build a unified, category-balanced index of MONO dry sources for pyroom synthesis.

The spatial dataset we synthesize needs *dry mono* sources tagged with a category
(audio / music / speech) and a native text label. Our raw datasets are
heterogeneous (parquet w/ embedded audio, loose wav/flac, zips), so this script
normalizes them into three JSONL pools:

    <out>/sources_audio.jsonl
    <out>/sources_music.jsonl
    <out>/sources_speech.jsonl

Each row: {"id", "path", "category", "dataset", "label"}
  * path  : a readable audio file on disk (mono not required; synthesis downmixes).
            For parquet/zip datasets the audio is extracted to <cache>/<dataset>/.
  * label : the native caption / class label (used later by the caption refiner).

Datasets & how they map to categories:
  audiocaps  (parquet, caption)         -> audio    [/mnt/sdd .../audiocaps]
  musiccaps  (wav + csv caption)        -> music    [/mnt/sdd .../musiccaps]
  audioset   (parquet, human_labels)    -> routed by label (music/speech/audio) [/mnt/sdb]
  fsd50k     (wav + csv labels)         -> routed by label (music/audio; speech excluded) [/mnt/sdd]
  picoaudio  (zip + json caption)       -> audio    [/mnt/sdd] (needs unzip)
  vggsound   (extracted wav + csv label)-> routed by label (music/audio) [/mnt/sdd]
                                            (run dataset/indexing/extract_vggsound.py first)

NOTE: Spatial LibriSpeech is intentionally NOT indexed here. SLS is used DIRECTLY as
real FOA (referenced in configs/dataset_configs/construct_dataset/*), NOT downmixed +
re-synthesized. It is the speech category of the constructed dataset.

Run (build the synth pools; SLS is added directly downstream, not here):
    cd /home/tanhe/dataset_storage/stable-audio-tools
    uv run python dataset/indexing/build_source_index.py \
        --out /mnt/sdd/audio_dataset/spatial_sources \
        --cache /mnt/sdd/audio_dataset/source_wav_cache \
        --datasets audiocaps,musiccaps,audioset,fsd50k,picoaudio,vggsound \
        --audioset-max 120000 --vggsound-max 80000

Then feed the pools to dataset/synthesis/build_spatial_dataset.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import soundfile as sf

LOG = logging.getLogger("source_index")

# Category routing for dry sources. In this script "audio" is the legacy file/API
# name for the non-music, non-speech sound-event pool. V2 reports it as "sound".
_MUSIC_TERMS = {
    "music", "musical instrument", "singing", "song", "melody", "tune",
    "accordion", "acoustic guitar", "bagpipes", "banjo", "bass", "bass drum",
    "bass guitar", "bassoon", "bongo", "bowed string", "brass instrument",
    "bugle", "castanets", "cello", "clarinet", "congas", "cornet", "cymbal",
    "didgeridoo", "djembe", "double bass", "drum", "electric guitar",
    "electronic organ", "erhu", "flute", "french horn", "glockenspiel", "gong",
    "guitar", "guiro", "harmonica", "hammond organ", "harp", "harpsichord",
    "hi-hat", "keyboard (musical)", "mallet percussion", "mandolin", "marimba",
    "oboe", "orchestra", "organ", "percussion", "piano", "saxophone",
    "shofar", "singing bowl", "sitar", "snare drum", "steel guitar", "steelpan",
    "synthesizer", "tabla", "tambourine", "tapping guitar", "theremin",
    "timbales", "timpani", "tympani", "trombone", "trumpet", "tuning fork", "ukulele",
    "vibraphone", "violin", "washboard", "wind instrument", "xylophone", "zither",
    "choir",
}
_SPEECH_TERMS = {
    "speech", "conversation", "narration", "male speech", "female speech",
    "child speech", "monologue", "speech synthesizer", "babbling",
}
_STRICT_SPEECH_TERMS = _SPEECH_TERMS | {
    "human voice", "voice", "vocal", "vocals", "singing", "choir", "chant",
    "lyrics", "talking", "talk", "spoken", "shout", "shouting", "scream",
    "screaming", "whisper", "whispering", "laughter", "laugh", "crying",
    "baby cry", "children shouting", "crowd", "crowd cheering",
}


def _route_audioset(human_labels: list[str]) -> str:
    return _route_labels_to_category(human_labels)


def _has_speech_like_text(parts: list[str]) -> bool:
    low = " ".join(str(x or "").replace("_", " ").lower() for x in parts)
    return any(t in low for t in _STRICT_SPEECH_TERMS)


def _route_labels_to_category(labels: list[str], extra_text: list[str] | None = None) -> str:
    """Route source metadata to the legacy categories: audio(sound), music, speech."""
    parts = list(labels or []) + list(extra_text or [])
    if _has_speech_like_text(parts):
        return "speech"
    low = [str(x).replace("_", " ").lower() for x in (labels or [])]
    if any(any(t in lab for t in _MUSIC_TERMS) for lab in low):
        return "music"
    return "audio"


# --------------------------------------------------------------------- adapters

def adapt_audiocaps(jsonl_path: Path, cache: Path, limit: Optional[int],
                    parquet_root: Optional[Path]) -> Iterator[dict]:
    """Prefer the already-extracted jsonl (audio_path + caption); else parquet."""
    n = 0
    if jsonl_path and jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ap = r.get("audio_path")
                if not ap or not Path(ap).exists():
                    continue
                label = r.get("caption", "")
                yield {"id": r.get("id", Path(ap).stem), "path": ap,
                       "category": _route_labels_to_category([label]), "dataset": "audiocaps",
                       "label": label}
                n += 1
                if limit and n >= limit:
                    return
    # Fallback (or top-up) from parquet when the jsonl had no usable wavs.
    if n == 0 and parquet_root and parquet_root.exists():
        LOG.info("audiocaps jsonl had no on-disk wavs; extracting from parquet")
        yield from _extract_parquet(
            parquet_root, "data/train-*.parquet", cache / "audiocaps",
            audio_col="audio", id_col="audiocap_id", label_col="caption",
            category="route", limit=limit, dataset="audiocaps",
        )


def adapt_musiccaps(audio_dir: Path, csv_path: Path, limit: Optional[int]) -> Iterator[dict]:
    captions: dict[str, str] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                captions[row["ytid"]] = row.get("caption", "")
    n = 0
    for wav in sorted(audio_dir.glob("*.wav")):
        ytid = wav.stem
        label = captions.get(ytid, "")
        cat = "speech" if _has_speech_like_text([label]) else "music"
        yield {"id": f"musiccaps_{ytid}", "path": str(wav), "category": cat,
               "dataset": "musiccaps", "label": label}
        n += 1
        if limit and n >= limit:
            return


def adapt_sls(ambi_dir: Path, prompts_jsonl: Path, limit: Optional[int]) -> Iterator[dict]:
    labels: dict[str, str] = {}
    if prompts_jsonl.exists():
        with prompts_jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    labels[str(r.get("id"))] = r.get("caption", "")
    n = 0
    for flac in sorted(ambi_dir.glob("*.flac")):
        sid = flac.stem  # zero-padded id
        # SLS prompts already bake in the ORIGINAL geometry ("coming from the
        # front, 2m away, reverberant room"). We RE-spatialize with pyroom, so
        # keep only the CONTENT (voice + spoken words); drop the stale location.
        raw = _sls_content_only(labels.get(sid, ""))
        yield {"id": f"sls_{sid}", "path": str(flac), "category": "speech",
               "dataset": "spatial_librispeech", "label": raw}
        n += 1
        if limit and n >= limit:
            return


def _sls_content_only(prompt: str) -> str:
    """Strip SLS's original spatial words; keep the voice + spoken words only."""
    if not prompt:
        return "a person speaking"
    voice = prompt.split(",", 1)[0].strip() if "," in prompt else "a person speaking"
    spoken = ""
    if "Spoken words:" in prompt:
        spoken = "Spoken words:" + prompt.split("Spoken words:", 1)[1]
    return (f"{voice}. {spoken}".strip() if spoken else voice)


def adapt_mrsdrama(prompts_jsonl: Path, limit: Optional[int]) -> Iterator[dict]:
    if not prompts_jsonl.exists():
        LOG.warning("mrsdrama prompts not found: %s", prompts_jsonl)
        return
    n = 0
    with prompts_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            p = r.get("path")
            if not p or not Path(p).exists():
                continue
            yield {"id": f"mrsdrama_{Path(p).stem}", "path": p, "category": "speech",
                   "dataset": "mrsdrama", "label": r.get("caption", "")}
            n += 1
            if limit and n >= limit:
                return


def adapt_audioset(parquet_root: Path, cache: Path, limit: Optional[int]) -> Iterator[dict]:
    """AudioSet parquet -> wav cache; category routed by human_labels."""
    yield from _extract_parquet(
        parquet_root, "data/*/*.parquet", cache / "audioset",
        audio_col="audio", id_col="video_id", label_col="human_labels",
        category="route", limit=limit, dataset="audioset",
    )


def _read_fsd50k_clip_info(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): (v or {}) for k, v in data.items()}


def _fsd50k_info_text(info: dict) -> list[str]:
    fields = [
        info.get("title", ""),
        info.get("description", ""),
        " ".join(str(x) for x in info.get("tags", []) or []),
    ]
    return [str(x) for x in fields if x]


def adapt_fsd50k(root: Path, limit: Optional[int], include_eval: bool = False) -> Iterator[dict]:
    """FSD50K wavs + weak labels; route to sound/music/speech raw pools.

    FSD50K labels are AudioSet ontology names stored as underscore-separated strings.
    Final V2 pools filter speech/vocal out of sound/music; the raw index keeps the
    speech-like rows visible for accounting instead of silently dropping them.
    """
    labels_root = root / "labels"
    clips_root = root / "clips"
    meta_root = root / "metadata"
    clip_info = {}
    clip_info.update(_read_fsd50k_clip_info(meta_root / "dev_clips_info_FSD50K.json"))
    if include_eval:
        clip_info.update(_read_fsd50k_clip_info(meta_root / "eval_clips_info_FSD50K.json"))

    splits = [("dev", labels_root / "dev.csv")]
    if include_eval:
        splits.append(("eval", labels_root / "eval.csv"))

    n = 0
    for split, csv_path in splits:
        if not csv_path.exists():
            LOG.warning("FSD50K labels missing: %s", csv_path)
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = str(row.get("fname", "")).strip()
                if not fid:
                    continue
                wav = clips_root / split / f"{fid}.wav"
                if not wav.exists():
                    continue
                labels = [
                    x.strip().replace("_", " ")
                    for x in str(row.get("labels", "")).split(",")
                    if x.strip()
                ]
                info_text = _fsd50k_info_text(clip_info.get(fid, {}))
                cat = _route_labels_to_category(labels, info_text)
                label = ", ".join(labels)
                yield {
                    "id": f"fsd50k_{split}_{fid}",
                    "path": str(wav),
                    "category": cat,
                    "dataset": "fsd50k",
                    "label": label,
                }
                n += 1
                if limit and n >= limit:
                    return


def adapt_picoaudio(snapshot: Path, cache: Path, limit: Optional[int]) -> Iterator[dict]:
    """PicoAudio: audio under audio_data.zip (must be unzipped) + json captions."""
    audio_root = snapshot / "data"
    if not audio_root.exists():
        LOG.warning("PicoAudio audio not extracted. Unzip %s into %s first.",
                    snapshot / "audio_data.zip", snapshot)
        return
    caps: dict[str, str] = {}
    train_json = snapshot / "meta_data" / "train.json"
    if train_json.exists():
        with train_json.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                caps[r["filepath"]] = r.get("frequencyCaption") or r.get("onoffCaption", "")
    n = 0
    for rel, label in caps.items():
        p = snapshot / rel
        if not p.exists():
            continue
        yield {"id": f"pico_{p.stem}", "path": str(p), "category": _route_labels_to_category([label]),
               "dataset": "picoaudio", "label": label}
        n += 1
        if limit and n >= limit:
            return


def adapt_vggsound(audio_dir: Path, csv_path: Path, limit: Optional[int]) -> Iterator[dict]:
    """VGGSound mono wavs (from extract_vggsound.py) + csv labels; routed by label.

    csv columns (no header): ytid, start_sec, label, split. Extracted wav stems are
    ``<ytid>_<start_sec:06d>`` (matching the mp4 names inside the tarballs), so we key
    the label map the same way. Instrument labels (piano/guitar/...) route to music,
    most others to audio; speech is left to the real-FOA SLS track, not synthesized.
    """
    if not audio_dir.exists():
        LOG.warning("vggsound audio not found: %s (run extract_vggsound.py first)", audio_dir)
        return
    labels: dict[str, str] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    ytid, start, label = row[0].strip(), row[1].strip(), row[2].strip()
                    try:
                        labels[f"{ytid}_{int(start):06d}"] = label
                    except ValueError:
                        continue
    n = 0
    for wav in sorted(audio_dir.glob("*.wav")):
        label = labels.get(wav.stem, "")
        cat = _route_labels_to_category([label]) if label else "audio"
        yield {"id": f"vggsound_{wav.stem}", "path": str(wav), "category": cat,
               "dataset": "vggsound", "label": label}
        n += 1
        if limit and n >= limit:
            return


def _extract_parquet(root: Path, glob: str, out_dir: Path, *, audio_col: str,
                     id_col: str, label_col: str, category: str,
                     limit: Optional[int], dataset: str,
                     fixed_category: Optional[str] = None) -> Iterator[dict]:
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(root.glob(glob))
    if not shards:
        LOG.warning("No parquet shards for %s under %s/%s", dataset, root, glob)
        return
    n = 0
    cols = [audio_col, id_col, label_col]
    for shard in shards:
        pf = pq.ParquetFile(str(shard))
        for batch in pf.iter_batches(batch_size=64, columns=cols):
            for row in batch.to_pylist():
                rid = str(row[id_col])
                label = row[label_col]
                if isinstance(label, list):
                    label_str = ", ".join(str(x) for x in label)
                    cat = _route_labels_to_category(label) if category == "route" else (fixed_category or "audio")
                else:
                    label_str = str(label or "")
                    cat = _route_labels_to_category([label_str]) if category == "route" else (fixed_category or "audio")
                wav_path = out_dir / f"{rid}.wav"
                if not wav_path.exists():
                    blob = row[audio_col]["bytes"]
                    try:
                        data, sr = sf.read(io.BytesIO(blob), dtype="float32")
                    except Exception as exc:  # noqa: BLE001
                        LOG.debug("decode fail %s: %s", rid, exc)
                        continue
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    sf.write(str(wav_path), data, int(sr))
                yield {"id": f"{dataset}_{rid}", "path": str(wav_path), "category": cat,
                       "dataset": dataset, "label": label_str}
                n += 1
                if limit and n >= limit:
                    return


# ------------------------------------------------------------------------- main

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Output dir for sources_*.jsonl")
    ap.add_argument("--cache", type=Path, default=Path("/mnt/sdc/audio_dataset_tmp/source_wav_cache"))
    ap.add_argument("--datasets", default="audiocaps,musiccaps,audioset,picoaudio,vggsound",
                    help="Comma list of adapters to run. NOTE: 'sls' is intentionally "
                         "omitted -- SLS is used directly as real FOA, not synthesized.")
    # roots
    ap.add_argument("--sdd", type=Path, default=Path("/mnt/sdd/audio_dataset/datasets"))
    ap.add_argument("--sdb", type=Path, default=Path("/mnt/sdb/audio_dataset/datasets"))
    ap.add_argument("--audiocaps-jsonl", type=Path,
                    default=Path("/mnt/sdc/audio_dataset_tmp/audiocaps_train.jsonl"))
    # per-dataset caps (None = all)
    ap.add_argument("--audiocaps-max", type=int, default=None)
    ap.add_argument("--musiccaps-max", type=int, default=None)
    ap.add_argument("--sls-max", type=int, default=None)
    ap.add_argument("--mrsdrama-max", type=int, default=None)
    ap.add_argument("--audioset-max", type=int, default=None)
    ap.add_argument("--fsd50k-max", type=int, default=None)
    ap.add_argument("--fsd50k-include-eval", action="store_true",
                    help="Also index FSD50K eval clips. Default uses dev only.")
    ap.add_argument("--picoaudio-max", type=int, default=None)
    ap.add_argument("--vggsound-max", type=int, default=None)
    ap.add_argument("--vggsound-root", type=Path,
                    default=Path("/mnt/sdc/audio_dataset/datasets/vggsound"),
                    help="Root of the extracted VGGSound dataset")
    ap.add_argument("--vggsound-audio", type=Path,
                    default=None,
                    help="Dir of wavs produced by extract_vggsound.py; defaults to <vggsound-root>/extracted/audio")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sinks = {c: (args.out / f"sources_{c}.jsonl").open("w", encoding="utf-8")
             for c in ("audio", "music", "speech")}
    counts = {"audio": 0, "music": 0, "speech": 0}

    def emit(rows: Iterator[dict]) -> None:
        for r in rows:
            cat = r["category"]
            if cat not in sinks:
                continue
            sinks[cat].write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[cat] += 1
            total = sum(counts.values())
            if total % 5000 == 0:
                LOG.info("indexed %d (audio=%d music=%d speech=%d)", total, *counts.values())

    sel = set(s.strip() for s in args.datasets.split(",") if s.strip())
    if "audiocaps" in sel:
        LOG.info("adapter: audiocaps")
        emit(adapt_audiocaps(args.audiocaps_jsonl, args.cache, args.audiocaps_max,
                             args.sdd / "audiocaps" / "snapshot"))
    if "musiccaps" in sel:
        LOG.info("adapter: musiccaps")
        emit(adapt_musiccaps(args.sdd / "musiccaps" / "audio",
                             args.sdd / "musiccaps" / "snapshot" / "musiccaps-public.csv",
                             args.musiccaps_max))
    if "sls" in sel:
        LOG.info("adapter: spatial_librispeech")
        emit(adapt_sls(args.sdb / "spatial_librispeech" / "ambisonics",
                       args.sdb / "spatial_librispeech" / "sls_prompts.jsonl", args.sls_max))
    if "mrsdrama" in sel:
        LOG.info("adapter: mrsdrama")
        emit(adapt_mrsdrama(args.sdd / "mrsdrama" / "mrsdrama_prompts.jsonl", args.mrsdrama_max))
    if "audioset" in sel:
        LOG.info("adapter: audioset (extracting parquet -> wav cache, may take a while)")
        emit(adapt_audioset(args.sdb / "audioset" / "snapshot", args.cache, args.audioset_max))
    if "fsd50k" in sel:
        LOG.info("adapter: fsd50k")
        emit(adapt_fsd50k(args.sdd / "FSD50k", args.fsd50k_max, args.fsd50k_include_eval))
    if "picoaudio" in sel:
        LOG.info("adapter: picoaudio")
        emit(adapt_picoaudio(args.sdd / "picoaudio" / "snapshot", args.cache, args.picoaudio_max))
    if "vggsound" in sel:
        LOG.info("adapter: vggsound")
        vggsound_audio = args.vggsound_audio or (args.vggsound_root / "extracted" / "audio")
        emit(adapt_vggsound(vggsound_audio,
                            args.vggsound_root / "snapshot" / "vggsound.csv",
                            args.vggsound_max))

    for s in sinks.values():
        s.close()
    LOG.info("DONE. audio=%d music=%d speech=%d -> %s",
             counts["audio"], counts["music"], counts["speech"], args.out)


if __name__ == "__main__":
    main()
