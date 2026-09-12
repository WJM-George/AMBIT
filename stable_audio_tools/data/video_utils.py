"""Video I/O utilities for CLIP and sync-aware video conditioning."""

import math
import os

import torch
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from PIL import Image


def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[0]
    target_duration = int(duration * target_fps)
    if current_duration > target_duration:
        video_tensor = video_tensor[:target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[-1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(repeat_times, 1, 1, 1)), dim=0)
    return video_tensor


def read_video(filepath, seek_time=0.0, duration=-1, target_fps=2):
    """
    Load video frames as float tensor [T, C, H, W] in 0-255 range (CLIP path divides by 255).

    duration: seconds of clip to load; -1 loads full video at target_fps sampling.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        resize_transform = transforms.Resize((224, 224))
        image = Image.open(filepath).convert("RGB")
        frame = transforms.ToTensor()(image).unsqueeze(0) * 255.0
        frame = resize_transform(frame)
        target_frames = int(duration * target_fps) if duration > 0 else 1
        frame = frame.repeat(int(math.ceil(target_frames / frame.shape[0])), 1, 1, 1)[:target_frames]
        return frame

    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    seek_frame = int(seek_time * fps)
    if duration > 0:
        total_frames_to_read = int(target_fps * duration)
        frame_interval = int(math.ceil(fps / target_fps))
        end_frame = min(seek_frame + total_frames_to_read * frame_interval, total_frames)
        frame_ids = list(range(seek_frame, end_frame, frame_interval))
    else:
        frame_interval = int(math.ceil(fps / target_fps))
        frame_ids = list(range(0, total_frames, frame_interval))

    frames = vr.get_batch(frame_ids).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

    if frames.shape[2] != 224 or frames.shape[3] != 224:
        resize_transform = transforms.Resize((224, 224))
        frames = resize_transform(frames)

    if duration > 0:
        video_tensor = adjust_video_duration(frames, duration, target_fps)
    else:
        video_tensor = frames

    return video_tensor


def encode_video_with_synchformer(
    video_path,
    synchformer_ckpt_path,
    seconds_start=0.0,
    seconds_total=10.0,
    device="cuda",
):
    """
    Encode video with Synchformer. Returns tensor for CLIPWithSync conditioner.
    """
    from torchvision.transforms import v2

    from stable_audio_tools.models.synchformer.features_utils import FeaturesUtils

    if not os.path.isfile(synchformer_ckpt_path):
        raise FileNotFoundError(
            f"Synchformer checkpoint not found: {synchformer_ckpt_path}\n"
            "Download synchformer_state_dict.pth from the sync-aware video model release."
        )

    sync_feature_extractor = FeaturesUtils(
        tod_vae_ckpt="vae_path",
        enable_conditions=True,
        bigvgan_vocoder_ckpt="bigvgan_path",
        synchformer_ckpt=synchformer_ckpt_path,
        mode="44k",
    ).eval().to(device)

    sync_video_tensor = read_video(
        video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25
    )
    sync_transform = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    sync_video = sync_transform(sync_video_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        video_sync_frames = sync_feature_extractor.encode_video_with_sync(sync_video)

    return video_sync_frames


def build_video_condition_dict(
    video_path=None,
    synchformer_ckpt_path=None,
    seconds_start=0.0,
    seconds_total=10.0,
    video_fps=5,
    device="cuda",
):
    """
    Build metadata dict for CLIPWithSync conditioner forward (video_tensors + video_sync_frames).
    Pass zero tensors when video_path is None (text-only CFG training).
    """
    duration = seconds_total
    target_frames = int(duration * video_fps)

    if video_path is None or (isinstance(video_path, str) and video_path.strip() == ""):
        video_tensors = torch.zeros(1, target_frames, 3, 224, 224)
        video_sync_frames = torch.zeros(1, 240, 768)
        return {"video_tensors": video_tensors, "video_sync_frames": video_sync_frames}

    video_tensors = read_video(video_path, seek_time=seconds_start, duration=duration, target_fps=video_fps)
    video_tensors = video_tensors.unsqueeze(0)

    if synchformer_ckpt_path:
        video_sync_frames = encode_video_with_synchformer(
            video_path,
            synchformer_ckpt_path,
            seconds_start=seconds_start,
            seconds_total=seconds_total,
            device=device,
        )
    else:
        video_sync_frames = torch.zeros(1, 240, 768)

    return {"video_tensors": video_tensors, "video_sync_frames": video_sync_frames}
