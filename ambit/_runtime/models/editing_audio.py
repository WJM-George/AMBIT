from __future__ import annotations
import copy
import math
from typing import Any, Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ambit._runtime.sceneplan.schema import (
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    validate_model_sceneplan,
)
from ambit._runtime.sceneplan.plan_conditioning import (
    compile_plan_condition,
    make_editing_dit_cfg_unknown_metadata,
)

FORBIDDEN_OLD_PLAN_KEYS = {
    "old_sceneplan",
    "old_plan",
    "source_sceneplan",
    "source_plan",
    "previous_sceneplan",
    "previous_plan",
}


def _reject_old_plan_keys(rows: Sequence[dict[str, Any]], *, where: str) -> None:
    for row in rows:
        keys = {str(key).lower().replace("-", "_") for key in row}
        leaked = keys & FORBIDDEN_OLD_PLAN_KEYS
        if leaked:
            raise RuntimeError(f"old ScenePlan leaked into {where}: {sorted(leaked)}")


def _align_decoded_sceneplan_to_audio_duration(
    sceneplan: dict[str, Any], exact_duration_sec: float
) -> dict[str, Any]:
    """Restore exact waveform duration after codec-v4 frame-grid decoding.

    Codec-v4 deliberately represents time as VAE-frame ids. Its decoded
    duration is therefore the end of the final (possibly partial) latent frame,
    while Editing DiT is trained with the exact source/target waveform sample
    count. Re-anchor only that known boundary and clip temporal fields that land
    on the padded end of the final frame.
    """
    validate_model_sceneplan(sceneplan)
    requested_duration = float(exact_duration_sec)
    if not math.isfinite(requested_duration) or requested_duration <= 0.0:
        raise ValueError("exact Editing audio duration is outside the model envelope")
    samples = int(round(requested_duration * MODEL_SAMPLE_RATE))
    if samples <= 0 or samples > MAX_MODEL_SAMPLES:
        raise ValueError("exact Editing audio duration is outside the model envelope")
    duration = round(samples / MODEL_SAMPLE_RATE, 6)
    expected_frames = math.ceil(samples / VAE_HOP_SAMPLES)
    decoded_frames = int(
        round(float(sceneplan["duration_sec"]) * MODEL_SAMPLE_RATE / VAE_HOP_SAMPLES)
    )
    if decoded_frames != expected_frames:
        raise ValueError("decoded ScenePlan duration frame differs from source audio")
    aligned = copy.deepcopy(sceneplan)
    aligned["duration_sec"] = duration
    for source in aligned["sources"]:
        activity = source["activity"]
        onset = float(activity["onset_sec"])
        offset = min(float(activity["offset_sec"]), duration)
        if not 0.0 <= onset < offset:
            raise ValueError("decoded source activity cannot align to exact audio duration")
        activity["onset_sec"] = onset
        activity["offset_sec"] = offset
        trajectory = source["trajectory"]
        if trajectory["type"] != "keyframed":
            continue
        original = list(trajectory["keyframes"])
        adjusted = []
        for keyframe in original:
            item = copy.deepcopy(keyframe)
            item["time_sec"] = max(onset, min(float(item["time_sec"]), offset))
            if adjusted and item["time_sec"] <= adjusted[-1]["time_sec"]:
                adjusted[-1] = item
            else:
                adjusted.append(item)
        if len(adjusted) < 2:
            source["trajectory"] = {
                "type": "linear",
                "start": copy.deepcopy(original[0]["position"]),
                "end": copy.deepcopy(original[-1]["position"]),
            }
        else:
            trajectory["keyframes"] = adjusted
    validate_model_sceneplan(aligned)
    return aligned


class ScenePlanTransfusionEditingDiTPipeline(nn.Module):
    """Execute explicit new plans with source FOA, without AR or M2D.

    The shared sampling/codec path also serves the full AR pipeline below.
    This class exposes no old-plan or target-audio model input.
    """

    def __init__(self, *, diffusion, audio_autoencoder: nn.Module | None = None):
        super().__init__()
        self.diffusion = diffusion
        self.audio_autoencoder = audio_autoencoder
        self.diffusion.pretransform = None

    @property
    def device(self) -> torch.device:
        return next(self.diffusion.model.parameters()).device

    def _require_audio_autoencoder(self) -> nn.Module:
        if self.audio_autoencoder is None:
            raise RuntimeError("this Editing pipeline was loaded without the frozen FOA VAE")
        return self.audio_autoencoder

    @torch.no_grad()
    def encode_source_foa(
        self,
        source_foa: Tensor,
        *,
        model_num_samples: Sequence[int],
        vae_seeds: int | Sequence[int] = 42,
    ) -> tuple[Tensor, Tensor]:
        """Encode real WYZX/SN3D FOA into the aligned Editing prefix.

        Reparameterization noise is deliberately sampled per row, so that RNG
        input is invariant to rank and batch composition. FP16 quantization
        reproduces the train-time latent storage boundary before values return
        to FP32 model arithmetic.
        """
        vae = self._require_audio_autoencoder()
        if source_foa.ndim != 3 or int(source_foa.shape[1]) != 4:
            raise ValueError("source FOA audio must be [batch,4,samples]")
        batch = int(source_foa.shape[0])
        samples = [int(value) for value in model_num_samples]
        if len(samples) != batch or any(
            (value <= 0 or value > MAX_MODEL_SAMPLES for value in samples)
        ):
            raise ValueError("source FOA sample counts are invalid")
        valid_frames = [math.ceil(value / VAE_HOP_SAMPLES) for value in samples]
        buckets = [432 if frames <= 432 else 648 for frames in valid_frames]
        if len(set(buckets)) != 1:
            raise ValueError("one audio Editing batch must use one 432/648 bucket")
        bucket = buckets[0]
        padded_samples = bucket * VAE_HOP_SAMPLES
        if int(source_foa.shape[-1]) < max(samples):
            raise ValueError("source FOA tensor is shorter than model_num_samples")
        audio = torch.zeros((batch, 4, padded_samples), device=self.device, dtype=torch.float32)
        incoming = source_foa.to(device=self.device, dtype=torch.float32)
        for index, count in enumerate(samples):
            audio[index, :, :count] = incoming[index, :, :count]
        if not torch.isfinite(audio).all():
            raise ValueError("source FOA audio contains non-finite samples")
        with torch.autocast(device_type=self.device.type, enabled=False):
            statistics = vae.encoder(audio.float()).float()
            if tuple(statistics.shape[:2]) != (batch, 128):
                raise RuntimeError("frozen VAE encoder did not produce 128 statistics")
            (mean, scale) = statistics.chunk(2, dim=1)
            if int(mean.shape[-1]) != bucket:
                raise RuntimeError("frozen VAE encoder changed its 1024-sample hop")
            if isinstance(vae_seeds, int):
                if batch != 1:
                    raise ValueError(
                        "batched source FOA encoding requires one stable VAE seed per row"
                    )
                seeds = [int(vae_seeds)]
            else:
                seeds = [int(value) for value in vae_seeds]
            if len(seeds) != batch or any((value < 0 for value in seeds)):
                raise ValueError("one non-negative frozen-VAE seed is required per row")
            rows = []
            stdev = F.softplus(scale) + 0.0001
            for index, seed in enumerate(seeds):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(
                    mean[index].shape, generator=generator, device="cpu", dtype=torch.float32
                ).to(device=self.device, dtype=torch.float32)
                rows.append(mean[index].float() + noise * stdev[index].float())
            latent = torch.stack(rows).to(torch.float16).to(torch.float32)
        mask = torch.zeros((batch, bucket), device=self.device, dtype=torch.bool)
        for index, frames in enumerate(valid_frames):
            mask[index, :frames] = True
        latent = latent * mask[:, None]
        if tuple(latent.shape) != (batch, 64, bucket) or not torch.isfinite(latent).all():
            raise RuntimeError("frozen VAE produced an invalid source latent")
        return (latent, mask)

    @torch.no_grad()
    def decode_foa_latents(
        self, latents: Tensor, *, model_num_samples: Sequence[int]
    ) -> tuple[Tensor, Tensor]:
        """Decode aligned latents and retain exact waveform-length masks."""
        vae = self._require_audio_autoencoder()
        if (
            latents.ndim != 3
            or int(latents.shape[1]) != 64
            or int(latents.shape[2]) not in (432, 648)
        ):
            raise ValueError("FOA latent decoder input must be [B,64,432|648]")
        samples = [int(value) for value in model_num_samples]
        if len(samples) != int(latents.shape[0]) or any(
            (value <= 0 or value > int(latents.shape[2]) * VAE_HOP_SAMPLES for value in samples)
        ):
            raise ValueError("FOA decode sample counts are invalid")
        with torch.autocast(device_type=self.device.type, enabled=False):
            decoded = vae.decode(latents.to(device=self.device, dtype=torch.float32)).float()
            expected = int(latents.shape[2]) * VAE_HOP_SAMPLES
            if tuple(decoded.shape) != (int(latents.shape[0]), 4, expected):
                raise RuntimeError("frozen VAE decoder output geometry changed")
            sample_mask = torch.zeros(
                (len(samples), expected), device=self.device, dtype=torch.bool
            )
            for index, count in enumerate(samples):
                sample_mask[index, :count] = True
            decoded = decoded * sample_mask[:, None]
            if not torch.isfinite(decoded).all():
                raise RuntimeError("frozen VAE decoder produced non-finite FOA")
        return (decoded, sample_mask)

    def _conditioning_metadata(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        new_sceneplans: Sequence[dict[str, Any]],
        model_num_samples: Sequence[int],
    ) -> list[dict[str, Any]]:
        (batch, channels, frames) = source_foa_latent.shape
        if channels != 64 or frames not in (432, 648):
            raise ValueError("Editing pipeline source must be [B,64,432|648]")
        if tuple(source_attention_mask.shape) != (batch, frames):
            raise ValueError("Editing pipeline source mask is not time-aligned")
        if len(new_sceneplans) != batch or len(model_num_samples) != batch:
            raise ValueError("Editing plan/audio geometry batch counts differ")
        if not bool(source_attention_mask.to(torch.bool).any(dim=1).all()):
            raise ValueError("every Editing row needs valid clean source frames")
        tokenizer = self.diffusion.conditioner.conditioners["prompt"].tokenizer
        rows = []
        for index, (plan, samples) in enumerate(zip(new_sceneplans, model_num_samples)):
            valid_frames = int(source_attention_mask[index].sum().item())
            if not math.isclose(
                float(plan["duration_sec"]), float(samples) / 44100.0, rel_tol=0.0, abs_tol=1.1e-06
            ):
                raise ValueError("new ScenePlan duration is not source-time aligned")
            condition = compile_plan_condition(
                plan,
                tokenizer=tokenizer,
                model_num_samples=int(samples),
                latent_frames_valid=valid_frames,
                latent_crop_length=frames,
                caption_max_tokens=512,
            )
            rows.append(
                {
                    **condition,
                    "sample_id": str(plan["sample_id"]),
                    "model_num_samples": int(samples),
                    "source_foa_latent": source_foa_latent[index],
                    "padding_mask": [source_attention_mask[index].to(torch.bool)],
                    "seconds_start": 0.0,
                    "seconds_total": float(samples) / 44100.0,
                }
            )
        _reject_old_plan_keys(rows, where="Editing-DiT conditioning")
        return rows

    @torch.no_grad()
    def sample_edited_latents(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        new_sceneplans: Sequence[dict[str, Any]],
        *,
        model_num_samples: Sequence[int],
        steps: int = 20,
        cfg_scale: float = 1.0,
        seed: int | Sequence[int] = 42,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Execute complete new plans with ``[noise || clean source]`` RF."""
        if int(steps) <= 0 or float(cfg_scale) < 0:
            raise ValueError("Editing RF steps must be positive and CFG nonnegative")
        source = source_foa_latent.to(device=self.device, dtype=torch.float32)
        mask = source_attention_mask.to(device=self.device, dtype=torch.bool)
        metadata = self._conditioning_metadata(
            source, mask, new_sceneplans, list(model_num_samples)
        )
        positive = self.diffusion.conditioner(metadata, self.device)
        positive_inputs = self.diffusion.get_conditioning_inputs(positive)
        negative_inputs = None
        if float(cfg_scale) != 1.0:
            negative_rows = [make_editing_dit_cfg_unknown_metadata(row) for row in metadata]
            if any(
                (
                    negative["source_foa_latent"] is not positive_row["source_foa_latent"]
                    for (negative, positive_row) in zip(negative_rows, metadata)
                )
            ):
                raise RuntimeError("Editing CFG did not retain the exact source")
            negative = self.diffusion.conditioner(negative_rows, self.device)
            negative_inputs = self.diffusion.get_conditioning_inputs(negative)
        if initial_noise is None:
            if isinstance(seed, int):
                if int(source.shape[0]) != 1:
                    raise ValueError(
                        "batched Editing sampling requires one stable noise seed per row"
                    )
                seeds = [int(seed)]
            else:
                seeds = [int(value) for value in seed]
            if len(seeds) != int(source.shape[0]) or any((value < 0 for value in seeds)):
                raise ValueError("one non-negative Editing noise seed is required per row")
            noise_rows = []
            for row_seed in seeds:
                generator = torch.Generator(device="cpu").manual_seed(row_seed)
                noise_rows.append(
                    torch.randn(
                        source.shape[1:], generator=generator, device="cpu", dtype=torch.float32
                    )
                )
            value = torch.stack(noise_rows).to(device=self.device)
        else:
            value = initial_noise.to(device=self.device, dtype=torch.float32).clone()
            if tuple(value.shape) != tuple(source.shape):
                raise ValueError("Editing initial noise shape differs from source")
        value = value * mask[:, None]
        delta = 1.0 / int(steps)
        for index in range(int(steps)):
            timestep = 1.0 - index * delta
            t = torch.full((source.shape[0],), timestep, device=self.device)
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                positive_velocity = self.diffusion.model(
                    value, t, **positive_inputs, cfg_dropout_prob=0.0, padding_mask=mask
                )
                if negative_inputs is None:
                    velocity = positive_velocity
                else:
                    negative_velocity = self.diffusion.model(
                        value, t, **negative_inputs, cfg_dropout_prob=0.0, padding_mask=mask
                    )
                    velocity = negative_velocity + float(cfg_scale) * (
                        positive_velocity - negative_velocity
                    )
            value = (value - delta * velocity.float()) * mask[:, None]
        return value
