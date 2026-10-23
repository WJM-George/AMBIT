from __future__ import annotations
from contextlib import nullcontext
import math
from typing import Any, Sequence
import torch
from torch import Tensor, nn
from .editing_audio import (
    ScenePlanTransfusionEditingDiTPipeline,
    _align_decoded_sceneplan_to_audio_duration,
)

CLAP44_PIPELINE_CONTRACT = "source_foa_44100_instruction_clap44_ar_complete_plan_aligned_dit_v1"


class ScenePlanTransfusionEditingCLAP44Pipeline(ScenePlanTransfusionEditingDiTPipeline):
    def __init__(self, *, diffusion, editing_ar, codec, audio_autoencoder: nn.Module | None = None):
        super().__init__(diffusion=diffusion, audio_autoencoder=audio_autoencoder)
        self.editing_ar = editing_ar
        self.codec = codec
        if editing_ar.shared_transformer is not diffusion.model.model.transformer:
            raise RuntimeError("CLAP44 AR and Editing DiT must share one Transformer")
        if editing_ar.source_semantic_mode not in {"latent_only", "clap44_audio_caption_aux"}:
            raise ValueError("CLAP44 pipeline cannot load an M2D route")
        encoder = editing_ar.source_clap_model
        if editing_ar.source_semantic_mode != "latent_only" and (
            encoder is None or any((p.requires_grad for p in encoder.parameters()))
        ):
            raise ValueError("CLAP44 pipeline requires its frozen source encoder")

    @torch.no_grad()
    def generate_new_sceneplans(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        edit_instructions: Sequence[str],
        *,
        duration_sec: float | Sequence[float],
        max_plan_tokens: int = 512,
    ):
        (batch, channels, frames) = (
            source_foa_latent.shape if source_foa_latent.ndim == 3 else (0, 0, 0)
        )
        if (
            batch < 1
            or channels != 64
            or frames not in (432, 648)
            or (source_attention_mask.shape != (batch, frames))
        ):
            raise ValueError("CLAP44 inference requires [B,64,432|648] and its valid-frame mask")
        durations = (
            [float(duration_sec)] * batch
            if isinstance(duration_sec, (int, float))
            else [float(x) for x in duration_sec]
        )
        if (
            len(durations) != batch
            or len(edit_instructions) != batch
            or any((not math.isfinite(x) or x <= 0 or x * 44100 > frames * 1024 for x in durations))
        ):
            raise ValueError("CLAP44 instructions/durations do not match source audio")
        samples = [round(x * 44100) for x in durations]
        expected = (
            torch.arange(frames, device=source_attention_mask.device)[None]
            < torch.tensor(
                [(x + 1023) // 1024 for x in samples], device=source_attention_mask.device
            )[:, None]
        )
        if not torch.equal(source_attention_mask.bool(), expected):
            raise ValueError("CLAP44 source mask is not the exact valid-frame prefix")
        source = source_foa_latent.to(self.device, dtype=torch.float32)
        mask = source_attention_mask.to(self.device, dtype=torch.bool)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            tokens = self.editing_ar.generate_batch(
                source,
                mask,
                edit_instructions,
                codec=self.codec,
                max_plan_tokens=max_plan_tokens,
                fixed_duration_sec=durations,
            )
        plans = [
            _align_decoded_sceneplan_to_audio_duration(
                self.codec.decode(ids.tolist(), sample_id=f"edited_{i:06d}"), duration
            )
            for (i, (ids, duration)) in enumerate(zip(tokens, durations))
        ]
        if len(plans) != batch:
            raise RuntimeError("CLAP44 AR omitted a complete output plan")
        return (plans, tokens)

    @torch.no_grad()
    def edit_latents(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        edit_instructions: Sequence[str],
        *,
        model_num_samples: Sequence[int],
        max_plan_tokens=512,
        steps=20,
        cfg_scale=1.0,
        seed: int | Sequence[int] = 42,
        initial_noise: Tensor | None = None,
    ):
        samples = [int(x) for x in model_num_samples]
        (plans, tokens) = self.generate_new_sceneplans(
            source_foa_latent,
            source_attention_mask,
            edit_instructions,
            duration_sec=[x / 44100 for x in samples],
            max_plan_tokens=max_plan_tokens,
        )
        output = self.sample_edited_latents(
            source_foa_latent,
            source_attention_mask,
            plans,
            model_num_samples=samples,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            initial_noise=initial_noise,
        )
        return (output, plans, tokens)

    @torch.no_grad()
    def edit_audio(
        self,
        source_foa: Tensor,
        edit_instructions: Sequence[str],
        *,
        model_num_samples: Sequence[int],
        sample_rate=44100,
        vae_seeds: int | Sequence[int] = 42,
        max_plan_tokens=512,
        steps=20,
        cfg_scale=1.0,
        noise_seed: int | Sequence[int] = 42,
        initial_noise: Tensor | None = None,
    ) -> dict[str, Any]:
        if sample_rate != 44100:
            raise ValueError("CLAP44 Editing uses native 44.1 kHz FOA; resampling is not implicit")
        samples = [int(x) for x in model_num_samples]
        (source, mask) = self.encode_source_foa(
            source_foa, model_num_samples=samples, vae_seeds=vae_seeds
        )
        (edited, plans, tokens) = self.edit_latents(
            source,
            mask,
            edit_instructions,
            model_num_samples=samples,
            max_plan_tokens=max_plan_tokens,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=noise_seed,
            initial_noise=initial_noise,
        )
        (audio, audio_mask) = self.decode_foa_latents(edited, model_num_samples=samples)
        return {
            "edited_foa": audio,
            "sample_attention_mask": audio_mask,
            "new_sceneplans": plans,
            "new_sceneplan_token_ids": tokens,
            "source_foa_latent": source,
            "source_attention_mask": mask,
            "edited_foa_latent": edited,
            "pipeline_contract": CLAP44_PIPELINE_CONTRACT,
        }
