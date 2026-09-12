#!/usr/bin/env python3
"""Offline VideoMAE-v2 MOTION feature extractor for the V/VT -> spatial-audio branches.

Why offline: VideoMAE-v2 (up to ViT-g, ~1B) is too heavy to run inside the DiT training
loop. Like the Synchformer sync frames, we precompute per-clip features once and let
`VideoMAEv2Conditioner` (a light adapter) consume them. Mirrors the repo's own
`extract_tad_feature.py` (sliding 16-frame windows -> model.forward_features -> [T_feat, C]).

Output: one `.npy` per clip, shape [T_feat, feat_dim] (feat_dim: vit_g=1408, vit_b=768),
named by the clip stem so it matches the FOA latent id (e.g. `<video_id>_<start>.npy`).

Setup (deps are in the `spatial` extra; modern timm works because we call the model
builder directly instead of timm.create_model):
    uv sync --extra train --extra spatial      # installs timm + decord
    # weights (HF OpenGVLab/VideoMAE2), already fetched for ViT-B:
    #   /mnt/sdc/ckpts/videomae/distill/vit_b_k710_dl_from_giant.pth   (feat_dim 768)
    # for max quality (heavier, feat_dim 1408):
    #   hf download OpenGVLab/VideoMAE2 mae-g/vit_g_hybrid_pt_1200e_k710_ft.pth --local-dir /mnt/sdc/ckpts/videomae

Run (sharded across GPUs; ViT-B defaults below):
    uv run python dataset/features/extract_videomae_features.py \
        --video-dir /mnt/sdb/audio_dataset/datasets/sphere360/media/test \
        --out-dir   /mnt/sdc/audio_dataset_tmp/sphere360_videomae/test
    # ViT-g: --model vit_giant_patch14_224 --ckpt-path .../mae-g/vit_g_hybrid_pt_1200e_k710_ft.pth
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_REPO = "/home/tanhe/dataset_storage/VideoMAEv2"
VIDEO_EXTS = (".webm", ".mp4", ".mkv", ".mov", ".avi")
# vit_giant_patch14_224 -> 1408, vit_large -> 1024, vit_base -> 768
FEAT_DIM = {"vit_giant_patch14_224": 1408, "vit_huge_patch16_224": 1280,
            "vit_large_patch16_224": 1024, "vit_base_patch16_224": 768,
            "vit_small_patch16_224": 384}


def build_model(repo: str, model_name: str, ckpt_path: str, device: str):
    """Lazily build a VideoMAE-v2 backbone from the repo and load a K710-finetuned ckpt.

    We call the model builder directly (NOT timm.create_model): modern timm injects a
    `pretrained_cfg` kwarg that VideoMAEv2's (timm-0.4.x-era) VisionTransformer rejects.
    Calling the registered builder function bypasses that, so a modern `timm` works.
    """
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    import models.modeling_finetune as mf  # noqa: F401  (uses timm.layers utils via shim)

    builder = getattr(mf, model_name, None)
    if builder is None:
        raise ValueError(f"Unknown VideoMAE model '{model_name}'. Options: {list(FEAT_DIM)}")
    model = builder(
        pretrained=False, img_size=224, num_classes=710,
        all_frames=16, tubelet_size=2, drop_path_rate=0.0, use_mean_pooling=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for k in ("model", "module"):
        if isinstance(ckpt, dict) and k in ckpt:
            ckpt = ckpt[k]
            break
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[videomae] missing keys: {len(missing)} (ok if only head)")
    model.eval().to(device)
    return model


def transform_clip(frames_np, device):
    """[16,H,W,3] uint8 -> [1,3,16,224,224] float in [0,1] (matches repo extract_tad)."""
    import torch
    import torch.nn.functional as F

    vid = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0  # [3,16,H,W]
    vid = F.interpolate(vid, size=(224, 224), mode="bilinear", align_corners=False)
    return vid.unsqueeze(0).to(device)


def discover_videos(video_dir: str) -> list[Path]:
    root = Path(video_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def discover_from_index(index_jsonl: str, video_key: str, out_key: str, start_key: str, duration_key: str) -> list[dict]:
    rows = []
    with open(index_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video = row.get(video_key)
            if not video:
                continue
            out = row.get(out_key)
            rows.append({
                "video": Path(video),
                "out": Path(out) if out else None,
                "start": float(row.get(start_key, 0.0) or 0.0),
                "duration": float(row.get(duration_key, 0.0) or 0.0),
                "id": row.get("clip_id") or row.get("id") or Path(video).stem,
            })
    return rows


def frame_windows_for_video(vr, fps: float, start_sec: float, duration_sec: float, nf: int, stride: int):
    native_fps = float(vr.get_avg_fps() or fps or 30.0)
    n = len(vr)
    start_frame = max(0, int(round(start_sec * native_fps)))
    if duration_sec and duration_sec > 0:
        end_frame = min(n, int(round((start_sec + duration_sec) * native_fps)))
    else:
        end_frame = n
    if end_frame <= start_frame:
        end_frame = min(n, start_frame + 1)

    # Sample at the requested VideoMAE fps so a fixed 10s window yields stable
    # temporal density regardless of source fps. Windows are then formed over
    # sampled frames and padded by repeating the last frame if needed.
    sample_step = max(1, int(round(native_fps / max(fps, 1e-6))))
    sampled = list(range(start_frame, end_frame, sample_step))
    if not sampled:
        sampled = [min(start_frame, max(0, n - 1))]
    for s in range(0, max(1, len(sampled) - nf + 1), stride):
        ids = sampled[s:s + nf]
        if len(ids) < nf:
            ids.extend([ids[-1]] * (nf - len(ids)))
        yield ids


def ffmpeg_sampled_frames(video_path: Path, fps: float, start_sec: float, duration_sec: float):
    """Decode a selected time window to RGB frames via ffmpeg. Works for AV1 webm."""
    cmd = ["ffmpeg", "-v", "error"]
    if start_sec and start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.4f}"])
    cmd.extend(["-i", str(video_path)])
    if duration_sec and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.4f}"])
    width, height = 224, 224
    cmd.extend([
        "-vf", f"fps={fps},scale={width}:{height}:flags=bilinear,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[:300] or f"ffmpeg decode failed: {video_path}")
    frame_size = width * height * 3
    if len(proc.stdout) < frame_size:
        return np.zeros((1, height, width, 3), dtype=np.uint8)
    usable = (len(proc.stdout) // frame_size) * frame_size
    return np.frombuffer(proc.stdout[:usable], dtype=np.uint8).reshape(-1, height, width, 3)


def frame_windows_from_sampled_frames(frames: np.ndarray, nf: int, stride: int):
    if len(frames) == 0:
        return
    for s in range(0, max(1, len(frames) - nf + 1), stride):
        chunk = frames[s:s + nf]
        if len(chunk) < nf:
            pad = np.repeat(chunk[-1:], nf - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        yield chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--index-jsonl", default=None,
                    help="Optional JSONL index with per-row video_path/start/duration/out path.")
    ap.add_argument("--video-key", default="video_path")
    ap.add_argument("--out-key", default="videomae_feats_path")
    ap.add_argument("--start-key", default="video_start")
    ap.add_argument("--duration-key", default="video_duration")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--model", default="vit_base_patch16_224", choices=list(FEAT_DIM))
    ap.add_argument("--ckpt-path", default="/mnt/sdc/ckpts/videomae/distill/vit_b_k710_dl_from_giant.pth")
    ap.add_argument("--num-frames", type=int, default=16, help="frames per VideoMAE window")
    ap.add_argument("--stride", type=int, default=16, help="hop between windows (16=non-overlap)")
    ap.add_argument("--fps", type=float, default=8.0, help="sample fps before VideoMAE windowing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()
    if not args.video_dir and not args.index_jsonl:
        ap.error("Provide either --video-dir or --index-jsonl")

    import torch
    try:
        from decord import VideoReader, cpu
    except ImportError:
        sys.exit("decord not installed: uv pip install decord")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.index_jsonl:
        items = discover_from_index(
            args.index_jsonl, args.video_key, args.out_key, args.start_key, args.duration_key
        )
    else:
        items = [{"video": v, "out": None, "start": 0.0, "duration": 0.0, "id": v.stem}
                 for v in discover_videos(args.video_dir)]
    items = [v for i, v in enumerate(items) if i % args.num_shards == args.shard]
    print(f"[videomae] {len(items)} videos (shard {args.shard}/{args.num_shards}) -> {out_dir}")
    if not items:
        return

    model = build_model(args.repo, args.model, args.ckpt_path, args.device)
    nf, stride = args.num_frames, args.stride

    ok = fail = skip = 0
    for idx, item in enumerate(items, 1):
        vpath = item["video"]
        out = item["out"] or (out_dir / f"{vpath.stem}.npy")
        if not out.is_absolute():
            out = out_dir / out
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            skip += 1
            continue
        try:
            feats = []
            try:
                vr = VideoReader(str(vpath), ctx=cpu(0))
                for frame_ids in frame_windows_for_video(
                    vr, fps=args.fps, start_sec=item["start"], duration_sec=item["duration"],
                    nf=nf, stride=stride,
                ):
                    frames = vr.get_batch(frame_ids).asnumpy()
                    inp = transform_clip(frames, args.device)
                    with torch.no_grad():
                        f = model.forward_features(inp)  # [1, C] (use_mean_pooling)
                    feats.append(f.squeeze(0).float().cpu().numpy())
            except Exception as dec_exc:  # noqa: BLE001 - AV1/VP9 webm may fail in decord builds
                print(f"[videomae] decord fallback {vpath.name}: {str(dec_exc)[:120]}")
                feats = []
                sampled = ffmpeg_sampled_frames(
                    vpath, fps=args.fps, start_sec=item["start"], duration_sec=item["duration"]
                )
                for frames in frame_windows_from_sampled_frames(sampled, nf=nf, stride=stride):
                    inp = transform_clip(frames, args.device)
                    with torch.no_grad():
                        f = model.forward_features(inp)  # [1, C] (use_mean_pooling)
                    feats.append(f.squeeze(0).float().cpu().numpy())
            if not feats:
                fail += 1
                continue
            np.save(out, np.stack(feats, axis=0).astype(np.float32))  # [T_feat, C]
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[videomae] FAIL {vpath.name}: {str(exc)[:200]}")
        if idx % 50 == 0:
            print(f"  [{idx}/{len(items)}] ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"[videomae] DONE ok={ok} skip={skip} fail={fail} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
