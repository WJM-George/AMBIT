#!/usr/bin/env python3
"""VIDEO / AV captioner for Sphere360 (Qwen3-Omni-30B-A3B-Instruct), for VT -> spatial-audio.

This is the VIDEO captioner: it can ground the caption in the 360 video (--mode av).
For plain AUDIO-only captioning (no video), use the lighter, purpose-built
`dataset/captioning/caption_audio.py` (Qwen3-Omni-30B-A3B-Captioner) instead.

Sphere360 ships no text. This generates one English caption per clip (the `text`
condition for your audio-generation model), describing the audible sound events /
acoustic scene, grounded in the 360 video (--mode av) or audio-only (--mode audio).

IMPORTANT: Qwen consumes MONO audio, never the 4-channel ambisonic stream. This script
downmixes the omnidirectional W channel (channel 0) to a temporary 16 kHz mono wav and
feeds THAT to the model (plus the muted video in --mode av).

Inputs (produced by split_sphere360_av.py, with the original webm as fallback):
    media/<split>_audio/<id>.flac    # 4-ch FOA  (W channel is downmixed here)
    media/<split>_video/<id>.webm    # video only (used in --mode av)
    media/<split>/<id>.webm          # fallback source if the split folders are absent
Output:
    media/<split>_captions.jsonl     # {"clip_id", "caption", "model", "mode"} per line (resumable)

Environment (use a DEDICATED venv; Qwen3-Omni needs transformers from source):
    uv venv .venv --python 3.10
    source .venv/bin/activate
    pip install git+https://github.com/huggingface/transformers accelerate qwen-omni-utils soundfile
    pip install -U flash-attn --no-build-isolation        # or use --attn sdpa to skip flash-attn
    # weights (~60 GB) auto-download, or pre-fetch:
    hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ${AMBIT_CKPT_ROOT}/Qwen3-Omni-30B-A3B-Instruct

Run (single instance, model sharded across all visible GPUs via device_map=auto):
    python dataset/captioning/caption_sphere360.py --split test --mode av

Scale out (data-parallel: one process per GPU group; each gets a disjoint shard):
    CUDA_VISIBLE_DEVICES=0,1 python dataset/captioning/caption_sphere360.py --split train --num-shards 4 --shard 0 &
    CUDA_VISIBLE_DEVICES=2,3 python dataset/captioning/caption_sphere360.py --split train --num-shards 4 --shard 1 &
    CUDA_VISIBLE_DEVICES=4,5 python dataset/captioning/caption_sphere360.py --split train --num-shards 4 --shard 2 &
    CUDA_VISIBLE_DEVICES=6,7 python dataset/captioning/caption_sphere360.py --split train --num-shards 4 --shard 3 &

Tip: Qwen/Qwen3-Omni-30B-A3B-Captioner (audio-only, thinker-only) is lighter and purpose-built
for audio captions; pass --model-path Qwen/Qwen3-Omni-30B-A3B-Captioner --mode audio.
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

DEFAULT_DATASET_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
DEFAULT_MEDIA_ROOT = DEFAULT_DATASET_ROOT / "datasets" / "sphere360" / "media"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

PROMPT_AUDIO = (
    "You are an expert audio annotator building a text-to-audio dataset. Listen to the clip and "
    "write a description (max 100 words) describing the audible sound events and the acoustic "
    "scene (sources, actions, environment, spatial impression if any). Describe only what is actually "
    "heard; do not guess or mention silence. Output the caption text only, with no prefix."
)
PROMPT_AV = (
    "You are an expert annotator building a video-to-spatial-audio dataset. Watch the muted 360-degree "
    "video and listen to the audio, then write a description (max 100 words) describing the sound "
    "in the scene: what is making sound, the action, and the environment. Describe only what is heard "
    "and consistent with the scene; do not guess. Output the caption text only, with no prefix."
)


def extract_mono_w(src: Path, dst_wav: Path, sr: int = 16000) -> None:
    """Downmix the FOA W channel (channel 0) to a mono wav for the captioner."""
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-map", "0:a:0", "-filter:a", f"pan=mono|c0=c0,aresample={sr}",
        "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-f", "wav", str(dst_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst_wav.exists() or dst_wav.stat().st_size < 256:
        raise RuntimeError(proc.stderr.strip()[:300] or "mono extraction failed")


def discover_clips(media_root: Path, split: str) -> list[tuple[str, Path, Path | None]]:
    """Return (clip_id, audio_source, video_source_or_None) for every downloaded clip."""
    audio_dir = media_root / f"{split}_audio"
    video_dir = media_root / f"{split}_video"
    webm_dir = media_root / split

    clips: list[tuple[str, Path, Path | None]] = []
    if audio_dir.is_dir() and any(audio_dir.glob("*.flac")):
        for flac in sorted(audio_dir.glob("*.flac")):
            cid = flac.stem
            vid = video_dir / f"{cid}.webm"
            clips.append((cid, flac, vid if vid.exists() else (webm_dir / f"{cid}.webm")))
    else:  # fallback: caption straight from the original webm
        for webm in sorted(webm_dir.glob("*.webm")):
            clips.append((webm.stem, webm, webm))
    return clips


def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["clip_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done




def caption_one(model, processor, process_mm_info, mode: str,
                mono_wav: Path, video: Path | None, max_new_tokens: int) -> str:
    content = []
    if mode == "av" and video is not None and video.exists():
        content.append({"type": "video", "video": str(video)})
    content.append({"type": "audio", "audio": str(mono_wav)})
    content.append({"type": "text", "text": PROMPT_AV if mode == "av" else PROMPT_AUDIO})
    conversation = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)

    gen = model.generate(
        **inputs, thinker_return_dict_in_generate=True, return_audio=False,
        use_audio_in_video=False, do_sample=False, max_new_tokens=max_new_tokens,
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
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--mode", choices=["audio", "av"], default="av",
                   help="audio: caption from sound only; av: also condition on the 360 video.")
    p.add_argument("--attn", default="flash_attention_2",
                   help="attn_implementation; use 'sdpa' if flash-attn is not installed.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-shards", type=int, default=1, help="Total parallel processes.")
    p.add_argument("--shard", type=int, default=0, help="This process's shard index [0, num_shards).")
    p.add_argument("--limit", type=int, default=None, help="Caption at most N clips (debug).")
    p.add_argument("--out", type=Path, default=None, help="Output jsonl (default media/<split>_captions.jsonl).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    media_root = args.media_root.resolve()
    out_path = args.out or (media_root / f"{args.split}_captions.jsonl")
    if args.num_shards > 1:
        out_path = out_path.with_name(f"{out_path.stem}.shard{args.shard}{out_path.suffix}")

    clips = discover_clips(media_root, args.split)
    if not clips:
        sys.exit(f"No clips found for split={args.split} under {media_root}. Download/split first.")
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    done = load_done(out_path)
    todo = [c for c in clips if c[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    logging.info("split=%s shard=%d/%d clips=%d already=%d todo=%d -> %s",
                 args.split, args.shard, args.num_shards, len(clips), len(done), len(todo), out_path)
    if not todo:
        logging.info("Nothing to do."); return

    from qwen_omni_utils import process_mm_info
    model, processor = build_model(args.model_path, args.attn)

    ok = fail = 0
    with out_path.open("a", encoding="utf-8") as sink, tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "mono.wav"
        for i, (cid, audio_src, video_src) in enumerate(todo, 1):
            try:
                extract_mono_w(audio_src, tmp_wav)
                caption = caption_one(model, processor, process_mm_info, args.mode,
                                      tmp_wav, video_src, args.max_new_tokens)
                if not caption:
                    raise RuntimeError("empty caption")
                sink.write(json.dumps({"clip_id": cid, "caption": caption,
                                       "model": args.model_path, "mode": args.mode},
                                      ensure_ascii=False) + "\n")
                sink.flush()
                ok += 1
            except Exception as exc:  # noqa: BLE001 - log and keep going
                fail += 1
                logging.warning("caption FAILED %s: %s", cid, str(exc)[:200])
            if i % 50 == 0:
                logging.info("  %d/%d (ok=%d fail=%d)", i, len(todo), ok, fail)

    logging.info("DONE split=%s shard=%d ok=%d fail=%d -> %s", args.split, args.shard, ok, fail, out_path)


if __name__ == "__main__":
    main()
