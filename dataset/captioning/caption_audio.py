#!/usr/bin/env python3
"""Lightweight AUDIO-only captioner using Qwen3-Omni-30B-A3B-Captioner.

This is the audio counterpart of caption_sphere360.py (which is the VIDEO/AV captioner).
Qwen3-Omni-30B-A3B-Captioner is purpose-built for fine-grained, low-hallucination audio
captions: it is single-turn, takes ONE audio input, accepts NO text prompt, and outputs
text only. That makes it lighter and more accurate than the Instruct model for plain
audio (speech, environmental sound, music, FOA/binaural spatial audio).

It produces, for each clip, one English caption used as the `text`/`prompt` condition for
the 4ch spatial-audio DiT. The output jsonl is keyed so dataset_4ch.load_caption_map can
attach it during pre-encode (id = file stem with FOA suffix stripped, e.g.
`audiocaps_5_WYZX_4ch.flac` -> `audiocaps_5`).

IMPORTANT: the model takes MONO audio. We downmix channel 0 (FOA W / binaural L) to a 16 kHz
mono wav and feed THAT. The 4ch ambisonic stream is never sent to the model. Clips are
trimmed to --max-seconds (default 30 s, the model's recommended max for detail).

Inputs (either or both):
    --audio-dir DIR        recursively caption *.flac/*.wav under DIR
    --input-jsonl FILE     caption rows with a foa_path / audio_path / path field
Output:
    --out FILE             jsonl of {"id", "caption", "audio", "model"} (resumable)

Environment (DEDICATED venv; Qwen3-Omni needs transformers from source):
    uv venv .venv --python 3.10
    source .venv/bin/activate
    pip install git+https://github.com/huggingface/transformers accelerate qwen-omni-utils soundfile
    pip install -U flash-attn --no-build-isolation       # or pass --attn sdpa
    hf download Qwen/Qwen3-Omni-30B-A3B-Captioner --local-dir ${AMBIT_CKPT_ROOT}/Qwen3-Omni-30B-A3B-Captioner

Run (single instance, sharded across all visible GPUs via device_map=auto):
    python dataset/captioning/caption_audio.py \
        --audio-dir ${AMBIT_CACHE_ROOT}/audiocaps_foa/train \
        --out ${AMBIT_CACHE_ROOT}/audiocaps_foa/audio_captions.jsonl

Scale out (data-parallel: one process per GPU group, disjoint shards):
    CUDA_VISIBLE_DEVICES=0,1 python dataset/captioning/caption_audio.py --audio-dir DIR --out OUT --num-shards 4 --shard 0 &
    CUDA_VISIBLE_DEVICES=2,3 python dataset/captioning/caption_audio.py --audio-dir DIR --out OUT --num-shards 4 --shard 1 &
    ...

Note: AudioCaps-FOA already ships captions in its synthesis manifest (train_manifest.jsonl),
so you do NOT need to re-caption it; point dataset_4ch at that manifest directly. Use this
script for sources WITHOUT text (e.g. Sphere360 audio, or richer captions than transcripts).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__:
    from .qwen_model import build_model
else:
    from qwen_model import build_model

DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
AUDIO_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a")
FOA_SUFFIXES = ("_WYZX_4ch", "_LR00_4ch", "_4ch")


def clip_id_from_path(path: Path, strip_suffix: bool = True) -> str:
    stem = path.stem
    if strip_suffix:
        for suf in FOA_SUFFIXES:
            if stem.endswith(suf):
                return stem[: -len(suf)]
    return stem


def extract_mono(src: Path, dst_wav: Path, sr: int = 16000, max_seconds: float = 30.0) -> None:
    """Downmix channel 0 (FOA W / binaural L) to a trimmed mono wav for the captioner."""
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-t", str(max_seconds), "-i", str(src),
        "-map", "0:a:0", "-filter:a", f"pan=mono|c0=c0,aresample={sr}",
        "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-f", "wav", str(dst_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst_wav.exists() or dst_wav.stat().st_size < 256:
        raise RuntimeError(proc.stderr.strip()[:300] or "mono extraction failed")


def discover_clips(args) -> list[tuple[str, Path]]:
    """Return (clip_id, audio_path) from --audio-dir and/or --input-jsonl."""
    clips: list[tuple[str, Path]] = []
    seen: set[str] = set()

    if args.audio_dir:
        root = Path(args.audio_dir)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS:
                cid = clip_id_from_path(p, args.strip_suffix)
                if cid not in seen:
                    seen.add(cid)
                    clips.append((cid, p))

    if args.input_jsonl:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ap = row.get("foa_path") or row.get("audio_path") or row.get("path")
                if not ap:
                    continue
                p = Path(ap)
                cid = row.get("id") or row.get("clip_id") or clip_id_from_path(p, args.strip_suffix)
                if cid not in seen:
                    seen.add(cid)
                    clips.append((str(cid), p))
    return clips


def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done




def caption_one(model, processor, process_mm_info, mono_wav: Path, max_new_tokens: int) -> str:
    # Captioner takes ONE audio input and NO text prompt (per the model card).
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": str(mono_wav)}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    gen = model.generate(
        **inputs, thinker_return_dict_in_generate=True, return_audio=False,
        # Qwen3-Omni owns a two-stage generate() wrapper.  A plain
        # max_new_tokens kwarg is treated as a shared value and loses to the
        # wrapper's thinker default (1024); route the limit explicitly.
        do_sample=False, thinker_max_new_tokens=max_new_tokens,
    )
    text_ids = gen[0] if isinstance(gen, tuple) else gen
    seq = getattr(text_ids, "sequences", text_ids)
    new_tokens = seq[:, inputs["input_ids"].shape[1]:]
    caption = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    return " ".join(caption.split())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", type=str, default=None, help="Recursively caption audio under this dir.")
    p.add_argument("--input-jsonl", type=str, default=None, help="Caption rows with foa_path/audio_path/path.")
    p.add_argument("--out", type=Path, required=True, help="Output jsonl (resumable).")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--attn", default="flash_attention_2", help="attn_implementation; use 'sdpa' if no flash-attn.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-seconds", type=float, default=30.0, help="Trim audio fed to the captioner (model max ~30s).")
    p.add_argument("--no-strip-suffix", dest="strip_suffix", action="store_false",
                   help="Keep full stem as id (default strips _WYZX_4ch/_LR00_4ch/_4ch to match dataset_4ch).")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Caption at most N clips (debug).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.audio_dir and not args.input_jsonl:
        sys.exit("Provide --audio-dir and/or --input-jsonl")

    out_path = args.out
    if args.num_shards > 1:
        out_path = out_path.with_name(f"{out_path.stem}.shard{args.shard}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clips = discover_clips(args)
    if not clips:
        sys.exit("No audio clips found.")
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    done = load_done(out_path)
    todo = [c for c in clips if c[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    logging.info("shard=%d/%d clips=%d already=%d todo=%d -> %s",
                 args.shard, args.num_shards, len(clips), len(done), len(todo), out_path)
    if not todo:
        logging.info("Nothing to do."); return

    from qwen_omni_utils import process_mm_info
    model, processor = build_model(args.model_path, args.attn)

    ok = fail = 0
    with out_path.open("a", encoding="utf-8") as sink, tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "mono.wav"
        for i, (cid, audio_src) in enumerate(todo, 1):
            try:
                extract_mono(audio_src, tmp_wav, max_seconds=args.max_seconds)
                caption = caption_one(model, processor, process_mm_info, tmp_wav, args.max_new_tokens)
                if not caption:
                    raise RuntimeError("empty caption")
                sink.write(json.dumps({"id": cid, "caption": caption,
                                       "audio": str(audio_src), "model": args.model_path},
                                      ensure_ascii=False) + "\n")
                sink.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001 - log and keep going
                fail += 1
                logging.warning("caption FAILED %s: %s", cid, str(exc)[:200])
            if i % 50 == 0:
                logging.info("  %d/%d (ok=%d fail=%d)", i, len(todo), ok, fail)

    logging.info("DONE shard=%d ok=%d fail=%d -> %s", args.shard, ok, fail, out_path)


if __name__ == "__main__":
    main()
