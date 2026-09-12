"""End-to-end audio-reference Transfusion Editing inference pipeline."""

from __future__ import annotations

import copy
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.model_sceneplan import (
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    validate_model_sceneplan,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (
    compile_editing_dit_plan_condition,
    make_editing_dit_cfg_unknown_metadata,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (
    EDITING_M2D_CACHE_MERGER,
    EDITING_M2D_SHARD_BUILDER,
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    editing_m2d_cache_implementation_sha256,
    editing_m2d_cache_online_parity_path,
    validate_editing_m2d_temporal_pilot,
)
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (
    EDITING_AR_CONTRACT,
    ScenePlanTransfusionEditingAR,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (
    EDITING_M2D_CLAP_CONTRACT,
    EDITING_M2D_CLAP_SIDE_INPUT,
    canonicalize_editing_m2d_embedding,
    editing_m2d_mode_flags,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (
    FrozenEditingM2DCLAP,
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
    decoded_foa_w_to_m2d_waveform,
    editing_m2d_numeric_runtime_fingerprint,
    expected_editing_m2d_clap_asset_report,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import (
    JOINT_SELECTION_SOURCE_PATHS,
    JOINT_TRAINING_SOURCE_PATHS,
    verify_frozen_qwen_runtime,
)
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import (
    CHECKPOINT_SCHEMA as JOINT_CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION as JOINT_CHECKPOINT_SCHEMA_VERSION,
    RUN_CONTRACT_SCHEMA as JOINT_RUN_CONTRACT_SCHEMA,
    RUN_CONTRACT_SCHEMA_VERSION as JOINT_RUN_CONTRACT_SCHEMA_VERSION,
    RUN_IDENTITY_NAME as JOINT_RUN_IDENTITY_NAME,
    validate_run_identity as validate_joint_run_identity,
)


EDITING_PIPELINE_CONTRACT = (
    "fp32_vae_decode_W_m2d_cpu_l2_fp16_boundary_bf16_ar_editing_dit_v7"
)
DIAGNOSTIC_EXTERNAL_M2D_EMBEDDING_ORIGIN = (
    "diagnostic_external_unbound_source_m2d_embedding_v1"
)
JOINT_SELECTION_SCHEMA = "sceneplan_transfusion_editing_joint_checkpoint_selection"
JOINT_SELECTION_CONTRACT = (
    "full_20k_10k_select_10k_holdout_joint_ar_rf_source_"
    "base_dit_noninferiority_m2d_stratified_free_ar_5k_v5"
)
FROZEN_VAE_CONFIG = Path(__file__).resolve().parents[1] / (
    "configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
FROZEN_VAE_CHECKPOINT = Path(
    "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
FROZEN_VAE_CONFIG_SHA256 = (
    "0f0373c9b32deb3d9ea875a3a0f98a898fa1d3a3aa0d82f6cade6dc2ab97b179"
)
FROZEN_VAE_CHECKPOINT_SHA256 = (
    "0229e48729bb6cf138c277d37c598d659000cf78e0a16f498171f2f1f83e8a87"
)
FORBIDDEN_OLD_PLAN_KEYS = {
    "old_sceneplan",
    "old_plan",
    "source_sceneplan",
    "source_plan",
    "previous_sceneplan",
    "previous_plan",
}
EXPECTED_JOINT_SELECTION_SOURCE_PATHS = set(JOINT_SELECTION_SOURCE_PATHS)
EXPECTED_JOINT_TRAINING_SOURCE_PATHS = set(JOINT_TRAINING_SOURCE_PATHS)


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
    # Model ScenePlans canonically serialize time to six decimals. The integer
    # sample count remains the alignment authority; this representation stays
    # within the validator/compiler tolerance and matches stored plan targets.
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
            item["time_sec"] = max(
                onset, min(float(item["time_sec"]), offset)
            )
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


@dataclass(frozen=True)
class EditingPipelineLoadReport:
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_step: int
    model_config: str
    model_config_sha256: str
    editing_ar_contract: str
    pipeline_contract: str
    shared_transformer_same_object: bool
    old_sceneplan_input: bool
    source_semantic_mode: str = "latent_only"
    source_caption_model_input: bool = False
    frozen_m2d_clap_assets: dict[str, Any] | None = None
    checkpoint_selection: str | None = None
    checkpoint_selection_sha256: str | None = None
    frozen_qwen_runtime: dict[str, Any] | None = None
    frozen_vae_config: str | None = None
    frozen_vae_config_sha256: str | None = None
    frozen_vae_checkpoint: str | None = None
    frozen_vae_checkpoint_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


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
            raise RuntimeError(
                "this Editing pipeline was loaded without the frozen FOA VAE"
            )
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
            value <= 0 or value > MAX_MODEL_SAMPLES for value in samples
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
        audio = torch.zeros(
            (batch, 4, padded_samples),
            device=self.device,
            dtype=torch.float32,
        )
        incoming = source_foa.to(device=self.device, dtype=torch.float32)
        for index, count in enumerate(samples):
            audio[index, :, :count] = incoming[index, :, :count]
        if not torch.isfinite(audio).all():
            raise ValueError("source FOA audio contains non-finite samples")
        with torch.autocast(device_type=self.device.type, enabled=False):
            statistics = vae.encoder(audio.float()).float()
            if tuple(statistics.shape[:2]) != (batch, 128):
                raise RuntimeError("frozen VAE encoder did not produce 128 statistics")
            mean, scale = statistics.chunk(2, dim=1)
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
            if len(seeds) != batch or any(value < 0 for value in seeds):
                raise ValueError("one non-negative frozen-VAE seed is required per row")
            rows = []
            stdev = F.softplus(scale) + 1.0e-4
            for index, seed in enumerate(seeds):
                # CPU Philox input keeps the encoded row invariant to CUDA rank and
                # physical-device assignment. Only the deterministic reparameterized
                # sample is transferred into the frozen encoder's device domain.
                generator = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.randn(
                    mean[index].shape,
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                ).to(device=self.device, dtype=torch.float32)
                rows.append(mean[index].float() + noise * stdev[index].float())
            latent = torch.stack(rows).to(torch.float16).to(torch.float32)
        mask = torch.zeros((batch, bucket), device=self.device, dtype=torch.bool)
        for index, frames in enumerate(valid_frames):
            mask[index, :frames] = True
        latent = latent * mask[:, None]
        if tuple(latent.shape) != (batch, 64, bucket) or not torch.isfinite(
            latent
        ).all():
            raise RuntimeError("frozen VAE produced an invalid source latent")
        return latent, mask

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
            value <= 0 or value > int(latents.shape[2]) * VAE_HOP_SAMPLES
            for value in samples
        ):
            raise ValueError("FOA decode sample counts are invalid")
        with torch.autocast(device_type=self.device.type, enabled=False):
            decoded = vae.decode(
                latents.to(device=self.device, dtype=torch.float32)
            ).float()
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
        return decoded, sample_mask

    def _conditioning_metadata(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        new_sceneplans: Sequence[dict[str, Any]],
        model_num_samples: Sequence[int],
    ) -> list[dict[str, Any]]:
        batch, channels, frames = source_foa_latent.shape
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
        for index, (plan, samples) in enumerate(
            zip(new_sceneplans, model_num_samples)
        ):
            valid_frames = int(source_attention_mask[index].sum().item())
            if not math.isclose(
                float(plan["duration_sec"]),
                float(samples) / 44_100.0,
                rel_tol=0.0,
                abs_tol=1.1e-6,
            ):
                raise ValueError("new ScenePlan duration is not source-time aligned")
            condition = compile_editing_dit_plan_condition(
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
                    "seconds_total": float(samples) / 44_100.0,
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
            negative_rows = [
                make_editing_dit_cfg_unknown_metadata(row) for row in metadata
            ]
            # The exact same source tensor object must survive both branches.
            if any(
                negative["source_foa_latent"] is not positive_row["source_foa_latent"]
                for negative, positive_row in zip(negative_rows, metadata)
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
            if len(seeds) != int(source.shape[0]) or any(value < 0 for value in seeds):
                raise ValueError("one non-negative Editing noise seed is required per row")
            noise_rows = []
            for row_seed in seeds:
                generator = torch.Generator(device="cpu").manual_seed(row_seed)
                noise_rows.append(
                    torch.randn(
                        source.shape[1:],
                        generator=generator,
                        device="cpu",
                        dtype=torch.float32,
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                positive_velocity = self.diffusion.model(
                    value,
                    t,
                    **positive_inputs,
                    cfg_dropout_prob=0.0,
                    padding_mask=mask,
                )
                if negative_inputs is None:
                    velocity = positive_velocity
                else:
                    negative_velocity = self.diffusion.model(
                        value,
                        t,
                        **negative_inputs,
                        cfg_dropout_prob=0.0,
                        padding_mask=mask,
                    )
                    velocity = negative_velocity + float(cfg_scale) * (
                        positive_velocity - negative_velocity
                    )
            value = (value - delta * velocity.float()) * mask[:, None]
        return value


class ScenePlanTransfusionEditingPipeline(ScenePlanTransfusionEditingDiTPipeline):
    """Generate a new plan, then execute it with aligned Editing DiT."""

    def __init__(
        self,
        *,
        diffusion,
        editing_ar: ScenePlanTransfusionEditingAR,
        codec: ModelScenePlanCodecV4,
        audio_autoencoder: nn.Module | None = None,
        source_semantic_encoder: FrozenEditingM2DCLAP | None = None,
    ) -> None:
        super().__init__(diffusion=diffusion, audio_autoencoder=audio_autoencoder)
        self.editing_ar = editing_ar
        self.codec = codec
        self.source_semantic_encoder = source_semantic_encoder
        if (
            self.editing_ar.shared_transformer
            is not self.diffusion.model.model.transformer
        ):
            raise RuntimeError("Editing AR and DiT do not share one Transformer")
        inject_m2d, _ = editing_m2d_mode_flags(
            self.editing_ar.source_semantic_mode
        )
        if inject_m2d != (self.source_semantic_encoder is not None):
            raise RuntimeError(
                "Editing AR semantic mode and frozen M2D runtime disagree"
            )


    def _require_source_semantic_encoder(self) -> FrozenEditingM2DCLAP:
        if self.source_semantic_encoder is None:
            raise RuntimeError(
                "this Editing AR was loaded without its frozen M2D audio encoder"
            )
        return self.source_semantic_encoder

    def _prepare_source_m2d_audio_embedding(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        *,
        durations: Sequence[float],
        source_m2d_audio_embedding: Tensor | None,
        source_m2d_audio_embedding_origin: str | None,
    ) -> Tensor | None:
        """Resolve the source-derived AR side input at its FP16 boundary.

        Production inference always derives this value from the aligned source
        latent.  The retained external-tensor argument is diagnostic-only: it
        is not cryptographically bound to the source latent, so callers must
        acknowledge that provenance explicitly and it is canonicalized before
        entering AR.
        """

        inject_m2d, _ = editing_m2d_mode_flags(
            self.editing_ar.source_semantic_mode
        )
        if not inject_m2d:
            if (
                source_m2d_audio_embedding is not None
                or source_m2d_audio_embedding_origin is not None
            ):
                raise ValueError(
                    "latent-only Editing AR cannot accept an M2D side input"
                )
            return None
        if source_m2d_audio_embedding is None:
            if source_m2d_audio_embedding_origin is not None:
                raise ValueError(
                    "M2D embedding origin was provided without an external tensor"
                )
            semantic_samples = [
                int(round(duration * MODEL_SAMPLE_RATE)) for duration in durations
            ]
            value = self.encode_source_m2d_audio(
                source_foa_latent,
                source_attention_mask,
                model_num_samples=semantic_samples,
            )
            if value.dtype != torch.float16:
                raise RuntimeError(
                    "internally derived M2D embedding missed its FP16 boundary"
                )
            canonical = value
        else:
            if (
                source_m2d_audio_embedding_origin
                != DIAGNOSTIC_EXTERNAL_M2D_EMBEDDING_ORIGIN
            ):
                raise RuntimeError(
                    "external source_m2d_audio_embedding is diagnostic-only and "
                    "not source-bound; pass its explicit diagnostic origin token"
                )
            value = source_m2d_audio_embedding
            canonical = canonicalize_editing_m2d_embedding(value)
        expected = (int(source_foa_latent.shape[0]), 768)
        if tuple(canonical.shape) != expected:
            raise ValueError(
                "Editing AR source M2D embedding must be "
                f"[batch,768], got {tuple(canonical.shape)}"
            )
        return canonical.to(device=self.device)

    @torch.no_grad()
    def encode_source_m2d_audio(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        *,
        model_num_samples: Sequence[int],
    ) -> Tensor:
        """Decode the aligned source latent and encode its full valid W view."""

        vae = self._require_audio_autoencoder()
        encoder = self._require_source_semantic_encoder()
        if (
            source_foa_latent.ndim != 3
            or int(source_foa_latent.shape[1]) != 64
            or int(source_foa_latent.shape[2]) not in (432, 648)
            or tuple(source_attention_mask.shape)
            != (
                int(source_foa_latent.shape[0]),
                int(source_foa_latent.shape[2]),
            )
        ):
            raise ValueError(
                "M2D source must be an aligned [B,64,432|648] latent and mask"
            )
        samples = [int(value) for value in model_num_samples]
        if len(samples) != int(source_foa_latent.shape[0]) or any(
            value <= 0
            or value > int(source_foa_latent.shape[-1]) * VAE_HOP_SAMPLES
            for value in samples
        ):
            raise ValueError("M2D source sample-count batch is invalid")
        source = source_foa_latent.to(device=self.device, dtype=torch.float32)
        mask = source_attention_mask.to(device=self.device, dtype=torch.bool)
        if not bool(torch.isfinite(source).all()):
            raise ValueError("M2D source latent contains non-finite values")
        by_geometry: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, sample_count in enumerate(samples):
            valid_frames = math.ceil(sample_count / VAE_HOP_SAMPLES)
            expected_mask = torch.zeros_like(mask[index])
            expected_mask[:valid_frames] = True
            if not bool(torch.equal(mask[index], expected_mask)):
                raise ValueError(
                    "M2D source mask is not the exact contiguous valid-frame prefix"
                )
            by_geometry[(valid_frames, sample_count)].append(index)

        # Decode and run M2D in exact-length groups.  This keeps the semantic
        # view invariant to padding while avoiding one VAE/M2D launch per row
        # during the 5K real-audio evaluation.
        rows: list[Tensor | None] = [None] * len(samples)
        for (valid_frames, sample_count), positions in by_geometry.items():
            # The semantic cache was produced with a full-FP32 frozen
            # VAE->W->M2D path.  A caller-owned AR autocast region must not
            # silently change that source view before the canonical FP16 edge.
            with torch.autocast(device_type=self.device.type, enabled=False):
                decoded = vae.decode(
                    source[positions, :, :valid_frames].float()
                ).float()
                expected_samples = valid_frames * VAE_HOP_SAMPLES
                if tuple(decoded.shape) != (
                    len(positions),
                    4,
                    expected_samples,
                ) or not bool(torch.isfinite(decoded).all()):
                    raise RuntimeError(
                        "frozen VAE source decode geometry/value changed"
                    )
                waveforms = torch.stack(
                    [
                        decoded_foa_w_to_m2d_waveform(
                            audio.float(), valid_samples=sample_count
                        )
                        for audio in decoded
                    ]
                ).float()
                encoded = encoder.encode_audio(waveforms).float()
            for position, value in zip(positions, encoded.unbind(0)):
                rows[position] = value
        if any(value is None for value in rows):
            raise RuntimeError("M2D source geometry batching lost an input row")
        embedding = canonicalize_editing_m2d_embedding(
            torch.stack([value for value in rows if value is not None])
        ).to(device=self.device)
        if tuple(embedding.shape) != (len(samples), 768) or not bool(
            torch.isfinite(embedding).all()
        ):
            raise RuntimeError("frozen M2D source embedding is invalid")
        return embedding


    @torch.no_grad()
    def generate_new_sceneplans(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        edit_instructions: Sequence[str],
        *,
        duration_sec: float | Sequence[float],
        max_plan_tokens: int = 1024,
        source_m2d_audio_embedding: Tensor | None = None,
        source_m2d_audio_embedding_origin: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[Tensor]]:
        """Run ``source audio + instruction -> complete new ScenePlan``.

        The normal path derives M2D from ``source_foa_latent``. An externally
        supplied embedding is retained only for explicitly acknowledged
        diagnostics and crosses the same canonical FP16 boundary.
        """

        if (
            source_foa_latent.ndim != 3
            or int(source_foa_latent.shape[1]) != 64
            or int(source_foa_latent.shape[2]) not in (432, 648)
            or tuple(source_attention_mask.shape)
            != (int(source_foa_latent.shape[0]), int(source_foa_latent.shape[2]))
        ):
            raise ValueError(
                "Editing AR inference source must use an aligned 432/648-frame "
                "bucket envelope"
            )

        if isinstance(duration_sec, (int, float)):
            durations = [float(duration_sec)] * int(source_foa_latent.shape[0])
        else:
            durations = [float(value) for value in duration_sec]
        if len(durations) != int(source_foa_latent.shape[0]):
            raise ValueError("Editing AR duration batch does not match source audio")
        source_m2d_audio_embedding = self._prepare_source_m2d_audio_embedding(
            source_foa_latent,
            source_attention_mask,
            durations=durations,
            source_m2d_audio_embedding=source_m2d_audio_embedding,
            source_m2d_audio_embedding_origin=source_m2d_audio_embedding_origin,
        )
        # Joint checkpoint selection performs free AR in CUDA bf16. Formal
        # inference uses the same arithmetic path so grammar argmax cannot
        # flip merely because selector and deployment precision differ.
        ar_autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if source_foa_latent.is_cuda
            else nullcontext()
        )
        with ar_autocast:
            token_ids = self.editing_ar.generate_batch(
                source_foa_latent,
                source_attention_mask,
                edit_instructions,
                codec=self.codec,
                max_plan_tokens=int(max_plan_tokens),
                fixed_duration_sec=durations,
                source_m2d_audio_embedding=source_m2d_audio_embedding,
            )
        plans = []
        for index, (ids, duration) in enumerate(zip(token_ids, durations)):
            decoded = self.codec.decode(
                ids.tolist(), sample_id=f"edited_{index:06d}"
            )
            plans.append(
                _align_decoded_sceneplan_to_audio_duration(decoded, duration)
            )
        return plans, token_ids


    @torch.no_grad()
    def edit_latents(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        edit_instructions: Sequence[str],
        *,
        model_num_samples: Sequence[int],
        duration_sec: float | Sequence[float],
        max_plan_tokens: int = 1024,
        steps: int = 20,
        cfg_scale: float = 1.0,
        seed: int | Sequence[int] = 42,
        source_m2d_audio_embedding: Tensor | None = None,
        source_m2d_audio_embedding_origin: str | None = None,
    ) -> tuple[Tensor, list[dict[str, Any]], list[Tensor]]:
        """Run the latent-reference AR -> Editing-DiT route.

        Its external M2D option is diagnostic-only. Production real-audio
        Editing uses :meth:`edit_audio`, which never exposes that option and
        always derives the embedding from source.
        """

        plans, token_ids = self.generate_new_sceneplans(
            source_foa_latent,
            source_attention_mask,
            edit_instructions,
            duration_sec=duration_sec,
            max_plan_tokens=max_plan_tokens,
            source_m2d_audio_embedding=source_m2d_audio_embedding,
            source_m2d_audio_embedding_origin=(
                source_m2d_audio_embedding_origin
            ),
        )
        edited = self.sample_edited_latents(
            source_foa_latent,
            source_attention_mask,
            plans,
            model_num_samples=model_num_samples,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        return edited, plans, token_ids

    @torch.no_grad()
    def edit_audio(
        self,
        source_foa: Tensor,
        edit_instructions: Sequence[str],
        *,
        model_num_samples: Sequence[int],
        vae_seeds: int | Sequence[int] = 42,
        max_plan_tokens: int = 512,
        steps: int = 20,
        cfg_scale: float = 1.0,
        noise_seed: int | Sequence[int] = 42,
        initial_noise: Tensor | None = None,
    ) -> dict[str, Any]:
        """Execute real FOA -> VAE -> AR -> DiT -> VAE -> FOA Editing."""

        samples = [int(value) for value in model_num_samples]
        source_latent, latent_mask = self.encode_source_foa(
            source_foa, model_num_samples=samples, vae_seeds=vae_seeds
        )
        durations = [value / MODEL_SAMPLE_RATE for value in samples]
        plans, token_ids = self.generate_new_sceneplans(
            source_latent,
            latent_mask,
            edit_instructions,
            duration_sec=durations,
            max_plan_tokens=max_plan_tokens,
        )
        edited_latent = self.sample_edited_latents(
            source_latent,
            latent_mask,
            plans,
            model_num_samples=samples,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=noise_seed,
            initial_noise=initial_noise,
        )
        edited_foa, sample_mask = self.decode_foa_latents(
            edited_latent, model_num_samples=samples
        )
        source_codec_foa, _ = self.decode_foa_latents(
            source_latent, model_num_samples=samples
        )
        return {
            "edited_foa": edited_foa,
            "source_codec_foa": source_codec_foa,
            "sample_attention_mask": sample_mask,
            "source_foa_latent": source_latent,
            "source_attention_mask": latent_mask,
            "edited_foa_latent": edited_latent,
            "new_sceneplans": plans,
            "new_sceneplan_token_ids": token_ids,
        }


def _load_ar_specific(
    ar: ScenePlanTransfusionEditingAR, state: dict[str, Tensor]
) -> None:
    incompatible = ar.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
        not (
            key.startswith("editing_dit.")
            or key.startswith("instruction_conditioner.")
        )
        for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "Editing pipeline AR adapter mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )


def _validate_m2d_cache_provenance(source_semantic: dict[str, Any]) -> None:
    cache = dict(source_semantic.get("cache") or {})
    records = [dict(cache.get(split) or {}) for split in ("train", "validation")]
    pilots = [dict(record.get("temporal_pilot") or {}) for record in records]
    expected_cache_assets = expected_editing_m2d_clap_asset_report(
        require_text=True
    )
    expected_implementation = editing_m2d_cache_implementation_sha256()
    expected_numeric_runtime = editing_m2d_numeric_runtime_fingerprint()
    expected_builder = EDITING_M2D_SHARD_BUILDER.resolve(strict=True)
    expected_merger = EDITING_M2D_CACHE_MERGER.resolve(strict=True)
    if not (
        cache.get("contract") == EDITING_M2D_CLAP_CONTRACT
        and cache.get("mode") == "m2d_audio_caption_aux"
        and records[0].get("split") == "train"
        and int(records[0].get("rows", -1)) == 1_000_000
        and records[1].get("split") == "validation"
        and int(records[1].get("rows", -1)) == 20_000
        and all(
            record.get("vae_config_sha256") == EDITING_M2D_VAE_CONFIG_SHA256
            and record.get("vae_checkpoint_sha256")
            == EDITING_M2D_VAE_CHECKPOINT_SHA256
            and record.get("source_audio_view") == M2D_CLAP_SOURCE_AUDIO_VIEW
            and record.get("temporal_policy") == M2D_CLAP_TEMPORAL_POLICY
            and record.get("m2d_assets") == expected_cache_assets
            and record.get("implementation_sha256") == expected_implementation
            and record.get("numeric_runtime_fingerprint")
            == expected_numeric_runtime
            and Path(record.get("shard_builder", "")).resolve()
            == expected_builder
            and record.get("shard_builder_sha256")
            == sha256_file(expected_builder)
            and Path(record.get("merger", "")).resolve() == expected_merger
            and record.get("merger_sha256") == sha256_file(expected_merger)
            for record in records
        )
    ):
        raise RuntimeError("joint Editing M2D cache provenance is stale")
    for record in records:
        parity = dict(record.get("cache_online_parity") or {})
        parity_path = editing_m2d_cache_online_parity_path(record["path"])
        if not (
            Path(parity.get("path", "")).resolve() == parity_path
            and parity.get("sha256") == sha256_file(parity_path.resolve(strict=True))
            and int(parity.get("rows", -1)) == 10
            and parity.get("batch_sizes") == [1, 2, 4]
            and parity.get("fp16_exact") is True
        ):
            raise RuntimeError("joint Editing M2D cache/online parity is stale")
    validated = [
        validate_editing_m2d_temporal_pilot(
            pilot.get("path", ""), expected_sha256=pilot.get("sha256")
        )
        for pilot in pilots
    ]
    if validated[0] != validated[1]:
        raise RuntimeError("joint Editing M2D caches used different temporal pilots")


def _validate_joint_selection(
    path: Path,
    *,
    checkpoint: Path,
    model_config: Path,
    codec: Path,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve(strict=True)
    observed_sha = sha256_file(path)
    if expected_sha256 is not None and observed_sha != str(expected_sha256):
        raise RuntimeError("joint Editing selection SHA256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    sources = dict(value.get("source_sha256") or {})
    source_hashes_valid = set(sources) == EXPECTED_JOINT_SELECTION_SOURCE_PATHS and all(
        sha256_file((Path(__file__).resolve().parents[2] / relative).resolve(strict=True))
        == expected
        for relative, expected in sources.items()
    )
    latest = dict(value.get("latest_route") or {})
    source_semantic = dict(value.get("source_semantic") or {})
    _validate_m2d_cache_provenance(source_semantic)
    training_run = dict(value.get("training_run") or {})
    run_dir = Path(training_run.get("run_dir", "")).resolve(strict=True)
    checkpoint_dir = (run_dir / "checkpoints").resolve(strict=True)
    if checkpoint_dir != run_dir / "checkpoints":
        raise RuntimeError("joint Editing checkpoint directory escaped its run")
    candidate_steps = list(range(5_000, 25_001, 5_000))
    candidates = list(value.get("candidates") or [])
    candidates_valid = len(candidates) == len(candidate_steps)
    for step, record in zip(candidate_steps, candidates):
        try:
            candidate_path = Path(record["checkpoint"]).resolve(strict=True)
            ranking_values = (
                float(record["selection_10k"]["ar"]["clean_ce"]["mean"]),
                float(
                    record["selection_10k"]["ar"]["token_accuracy"]["mean"]
                ),
                float(record["selection_10k"]["rf"]["mean"]),
            )
            candidates_valid = candidates_valid and (
                int(record["step"]) == step
                and candidate_path
                == checkpoint_dir / f"step-{step:08d}.pt"
                and record.get("checkpoint_sha256")
                == sha256_file(candidate_path)
                and int(record.get("checkpoint_metadata", {}).get("global_step", -1))
                == step
                and record.get("checkpoint_metadata", {}).get(
                    "shared_transformer_same_object"
                )
                is True
                and all(math.isfinite(item) for item in ranking_values)
            )
        except (KeyError, TypeError, ValueError, OSError):
            candidates_valid = False
    eligible = [
        record
        for record in candidates
        if record.get("selection_10k", {})
        .get("base_dit_noninferiority", {})
        .get("pass")
        is True
    ]
    ranked = (
        sorted(
            eligible,
            key=lambda record: (
                float(record["selection_10k"]["ar"]["clean_ce"]["mean"]),
                -float(
                    record["selection_10k"]["ar"]["token_accuracy"]["mean"]
                ),
                float(record["selection_10k"]["rf"]["mean"]),
                -int(record["step"]),
            ),
        )
        if candidates_valid
        else []
    )
    selected_step = int(value.get("selected_checkpoint_step", -1))
    selected_record = ranked[0] if ranked else {}
    run_contract_path = Path(training_run.get("run_contract_path", "")).resolve(
        strict=True
    )
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    identity_path = (run_dir / JOINT_RUN_IDENTITY_NAME).resolve(strict=True)
    run_identity = validate_joint_run_identity(
        run_dir, identity_path=identity_path
    )
    final_path = Path(training_run.get("final_path", "")).resolve(strict=True)
    latest_path = Path(training_run.get("latest_path", "")).resolve(strict=True)
    if not (
        run_contract_path == run_dir / "RUN_CONTRACT.json"
        and final_path == run_dir / "FINAL.json"
        and latest_path == run_dir / "checkpoints/LATEST.json"
    ):
        raise RuntimeError("joint Editing selection artifacts escaped their run")
    free_records = Path(value.get("free_ar_records_path", "")).resolve(strict=True)
    if not (
        value.get("schema") == JOINT_SELECTION_SCHEMA
        and int(value.get("schema_version", -1)) == 1
        and value.get("status") == "PASS"
        and value.get("selection_contract") == JOINT_SELECTION_CONTRACT
        and value.get("physical_gpus") == [3, 4, 5, 6, 7]
        and int(value.get("world_size", -1)) == 5
        and value.get("cuda_visible_devices") == "3,4,5,6,7"
        and latest.get("editing_ar_inputs")
        == ["source_foa_latent", "raw_edit_request"]
        and latest.get("editing_ar_target") == "complete_new_sceneplan"
        and latest.get("old_sceneplan_input") is False
        and latest.get("source_caption_model_input") is False
        and latest.get("source_derived_semantic_side_input")
        == EDITING_M2D_CLAP_SIDE_INPUT
        and int(latest.get("editing_dit_frame_channels", -1)) == 384
        and latest.get("shared_transformer_same_object") is True
        and candidates_valid
        and value.get("candidate_steps") == candidate_steps
        and value.get("selection_ranked_steps")
        == [int(record["step"]) for record in ranked]
        and int(selected_record.get("step", -1)) == selected_step
        and Path(value.get("selected_checkpoint", "")).resolve(strict=True)
        == checkpoint
        and value.get("selected_checkpoint_sha256") == sha256_file(checkpoint)
        and selected_step in candidate_steps
        and Path(selected_record.get("checkpoint", "")).resolve(strict=True)
        == checkpoint
        and selected_record.get("checkpoint_sha256")
        == value.get("selected_checkpoint_sha256")
        and value.get("selected_full_20k_teacher")
        == selected_record.get("clean_full_20k")
        and value.get("selected_base_dit_noninferiority_gate", {}).get("pass")
        is True
        and value.get("selected_source_intervention_gate", {}).get("pass") is True
        and source_semantic.get("contract") == EDITING_M2D_CLAP_CONTRACT
        and source_semantic.get("mode") == "m2d_audio_caption_aux"
        and source_semantic.get("source_audio_view")
        == M2D_CLAP_SOURCE_AUDIO_VIEW
        and source_semantic.get("temporal_policy")
        == M2D_CLAP_TEMPORAL_POLICY
        and source_semantic.get("caption_model_input") is False
        and source_semantic.get("old_sceneplan_model_input") is False
        and value.get("selected_free_ar_gate", {}).get("pass") is True
        and int(value.get("validation_index", {}).get("rows", -1)) == 20_000
        and int(value.get("validation_folds", {}).get("selection", {}).get("rows", -1))
        == 10_000
        and int(value.get("validation_folds", {}).get("holdout", {}).get("rows", -1))
        == 10_000
        and Path(value.get("model_config", {}).get("path", "")).resolve()
        == model_config
        and value.get("model_config", {}).get("sha256") == sha256_file(model_config)
        and Path(value.get("codec", {}).get("path", "")).resolve() == codec
        and run_contract.get("schema") == JOINT_RUN_CONTRACT_SCHEMA
        and int(run_contract.get("schema_version", -1))
        == JOINT_RUN_CONTRACT_SCHEMA_VERSION
        and run_contract.get("run_dir") == str(run_dir)
        and run_contract.get("run_id") == run_identity["run_id"]
        and run_contract.get("run_identity")
        == {"path": str(identity_path), "sha256": sha256_file(identity_path)}
        and training_run.get("run_id") == run_identity["run_id"]
        and training_run.get("run_identity_path") == str(identity_path)
        and training_run.get("run_identity_sha256") == sha256_file(identity_path)
        and training_run.get("run_contract_sha256")
        == sha256_file(run_contract_path)
        and training_run.get("final_sha256") == sha256_file(final_path)
        and training_run.get("latest_sha256") == sha256_file(latest_path)
        and value.get("free_ar_records_sha256") == sha256_file(free_records)
        and source_hashes_valid
    ):
        raise RuntimeError("joint Editing checkpoint selection is stale or invalid")
    return value, observed_sha


def _load_frozen_foa_vae(device: torch.device | str) -> tuple[nn.Module, dict[str, str]]:
    config_path = FROZEN_VAE_CONFIG.expanduser().resolve(strict=True)
    checkpoint = FROZEN_VAE_CHECKPOINT.expanduser().resolve(strict=True)
    observed_config_sha = sha256_file(config_path)
    if observed_config_sha != FROZEN_VAE_CONFIG_SHA256:
        raise RuntimeError("frozen FOA VAE configuration SHA256 changed")
    observed_checkpoint_sha = sha256_file(checkpoint)
    if observed_checkpoint_sha != FROZEN_VAE_CHECKPOINT_SHA256:
        raise RuntimeError("frozen FOA VAE checkpoint SHA256 changed")
    vae = create_model_from_config(load_config(config_path))
    copy_state_dict(vae, load_ckpt_state_dict(str(checkpoint)))
    vae.eval().requires_grad_(False).to(device)
    return vae, {
        "config": str(config_path),
        "config_sha256": observed_config_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": observed_checkpoint_sha,
    }


def load_sceneplan_transfusion_editing_pipeline(
    *,
    checkpoint: str | Path,
    model_config: str | Path,
    codec: str | Path,
    device: torch.device | str,
    checkpoint_selection: str | Path | None = None,
    checkpoint_selection_sha256: str | None = None,
    load_audio_autoencoder: bool = True,
    allow_unselected_diagnostic: bool = False,
) -> tuple[ScenePlanTransfusionEditingPipeline, EditingPipelineLoadReport]:
    """Load a selected joint checkpoint without accepting an old-plan route.

    Raw checkpoints are rejected by default. The sole escape hatch is explicit
    and named diagnostic-only so formal evaluation cannot silently bypass
    independent checkpoint promotion.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    model_config_path = Path(model_config).expanduser().resolve(strict=True)
    codec_path = Path(codec).expanduser().resolve(strict=True)
    selection_path = None
    selection_sha = None
    selection_value = None
    if checkpoint_selection is None:
        if checkpoint_selection_sha256 is not None:
            raise RuntimeError("selection SHA256 was supplied without a selection")
        if not bool(allow_unselected_diagnostic):
            raise RuntimeError(
                "formal Editing pipeline loading requires a selected checkpoint"
            )
    else:
        if checkpoint_selection_sha256 is None:
            raise RuntimeError(
                "formal Editing pipeline loading requires a pinned selection SHA256"
            )
        selection_path = Path(checkpoint_selection).expanduser().resolve(strict=True)
        selection_value, selection_sha = _validate_joint_selection(
            selection_path,
            checkpoint=checkpoint_path,
            model_config=model_config_path,
            codec=codec_path,
            expected_sha256=checkpoint_selection_sha256,
        )
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    if (
        payload.get("schema") != JOINT_CHECKPOINT_SCHEMA
        or int(payload.get("schema_version", -1))
        != JOINT_CHECKPOINT_SCHEMA_VERSION
        or payload.get("contract") != EDITING_AR_CONTRACT
        or payload.get("joint_dataset_contract")
        != "source_audio_instruction_to_canonical_new_plan_plus_aligned_rf_v2"
    ):
        raise RuntimeError("checkpoint is not a full latest-route Editing bundle")
    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, dict):
        raise RuntimeError("joint Editing checkpoint run contract is missing")
    payload_run_dir = Path(str(run_contract.get("run_dir", ""))).resolve()
    payload_contract_path = (payload_run_dir / "RUN_CONTRACT.json").resolve(
        strict=True
    )
    if payload_contract_path != payload_run_dir / "RUN_CONTRACT.json":
        raise RuntimeError("joint Editing run contract escaped its run directory")
    payload_identity_path = (
        payload_run_dir / JOINT_RUN_IDENTITY_NAME
    ).resolve(strict=True)
    payload_identity = validate_joint_run_identity(
        payload_run_dir, identity_path=payload_identity_path
    )
    if not (
        run_contract.get("schema") == JOINT_RUN_CONTRACT_SCHEMA
        and int(run_contract.get("schema_version", -1))
        == JOINT_RUN_CONTRACT_SCHEMA_VERSION
        and json.loads(payload_contract_path.read_text(encoding="utf-8"))
        == run_contract
        and run_contract.get("run_id") == payload_identity["run_id"]
        and payload.get("run_dir") == str(payload_run_dir)
        and payload.get("run_id") == payload_identity["run_id"]
        and payload.get("run_contract_path") == str(payload_contract_path)
        and payload.get("run_contract_sha256")
        == sha256_file(payload_contract_path)
    ):
        raise RuntimeError("joint Editing checkpoint lineage changed")
    latest = dict(run_contract.get("latest_route") or {}) if isinstance(run_contract, dict) else {}
    source_semantic = (
        dict(run_contract.get("source_semantic") or {})
        if isinstance(run_contract, dict)
        else {}
    )
    _validate_m2d_cache_provenance(source_semantic)
    if selection_value is not None:
        selected_training = dict(selection_value.get("training_run") or {})
        selected_contract_path = Path(
            selected_training.get("run_contract_path", "")
        ).resolve(strict=True)
        selected_disk_contract = json.loads(
            selected_contract_path.read_text(encoding="utf-8")
        )
        if not (
            run_contract == selected_disk_contract
            and selected_training.get("run_contract_sha256")
            == sha256_file(selected_contract_path)
            and selection_value.get("source_semantic")
            == selected_disk_contract.get("source_semantic")
        ):
            raise RuntimeError(
                "joint Editing selection/checkpoint run contract changed"
            )
    training_sources = (
        dict(run_contract.get("source_sha256") or {})
        if isinstance(run_contract, dict)
        else {}
    )
    qwen_runtime = verify_frozen_qwen_runtime()
    training_source_hashes_valid = (
        set(training_sources) == EXPECTED_JOINT_TRAINING_SOURCE_PATHS
        and all(
            sha256_file(
                (Path(__file__).resolve().parents[2] / relative).resolve(strict=True)
            )
            == expected
            for relative, expected in training_sources.items()
        )
    )
    if not (
        isinstance(run_contract, dict)
        and latest.get("editing_ar_inputs")
        == ["source_foa_latent", "raw_edit_request"]
        and latest.get("editing_ar_target") == "complete_new_sceneplan"
        and latest.get("old_sceneplan_input") is False
        and latest.get("source_caption_model_input") is False
        and latest.get("target_audio_or_latent_ar_input") is False
        and latest.get("source_derived_semantic_side_input")
        == EDITING_M2D_CLAP_SIDE_INPUT
        and latest.get("editing_dit_frame_input")
        == [
            "noisy_target_64",
            "new_sceneplan_256",
            "clean_source_foa_latent_64",
        ]
        and latest.get("ar_and_dit_share_exact_transformer_object") is True
        and int(run_contract.get("world_size", -1)) == 5
        and run_contract.get("physical_gpus") == [3, 4, 5, 6, 7]
        and run_contract.get("cuda_visible_devices") == "3,4,5,6,7"
        and run_contract.get("cuda_device_order") == "PCI_BUS_ID"
        and run_contract.get("frozen_qwen_runtime") == qwen_runtime
        and source_semantic.get("contract") == EDITING_M2D_CLAP_CONTRACT
        and source_semantic.get("mode") == "m2d_audio_caption_aux"
        and source_semantic.get("inject_frozen_m2d_audio") is True
        and source_semantic.get("align_source_caption") is True
        and source_semantic.get("audio_embedding_is_source_derived") is True
        and source_semantic.get("source_audio_view")
        == M2D_CLAP_SOURCE_AUDIO_VIEW
        and source_semantic.get("temporal_policy")
        == M2D_CLAP_TEMPORAL_POLICY
        and source_semantic.get("caption_is_training_label_only") is True
        and source_semantic.get("caption_model_input") is False
        and source_semantic.get("old_sceneplan_model_input") is False
        and source_semantic.get("new_sceneplan_or_target_information_used") is False
        and training_source_hashes_valid
    ):
        raise RuntimeError("checkpoint run contract predates the latest Editing route")
    if (
        run_contract.get("model_config_sha256") != sha256_file(model_config_path)
        or Path(str(run_contract.get("model_config", ""))).resolve()
        != model_config_path
    ):
        raise RuntimeError("Editing checkpoint/model configuration identity changed")
    config = load_config(model_config_path)
    if selection_value is not None and int(payload.get("global_step", -1)) != int(
        selection_value["selected_checkpoint_step"]
    ):
        raise RuntimeError("joint Editing selection/checkpoint step changed")
    diffusion = create_model_from_config(config)
    # Joint bundles intentionally omit the immutable VAE/pretransform. Remove
    # it before strict-loading the complete trainable Editing state.
    diffusion.pretransform = None
    incompatible = diffusion.load_state_dict(
        payload["diffusion_state_dict"], strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Editing pipeline diffusion state mismatch")
    prompt = diffusion.conditioner.conditioners["prompt"]
    plan_codec = ModelScenePlanCodecV4(codec_path)
    if run_contract.get("codec_fingerprint") != plan_codec.fingerprint:
        raise RuntimeError("Editing checkpoint/codec fingerprint changed")
    if (
        selection_value is not None
        and selection_value.get("codec", {}).get("fingerprint")
        != plan_codec.fingerprint
    ):
        raise RuntimeError("joint Editing selection/codec fingerprint changed")
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=diffusion.model.model,
        instruction_conditioner=prompt,
        pad_id=plan_codec.pad_id,
        activation_checkpointing=False,
        source_semantic_mode=str(source_semantic["mode"]),
        source_semantic_dropout=float(source_semantic["source_semantic_dropout"]),
    )
    _load_ar_specific(ar, payload["editing_ar_specific_state_dict"])
    inject_m2d, _ = editing_m2d_mode_flags(str(source_semantic["mode"]))
    source_semantic_encoder = (
        FrozenEditingM2DCLAP(device=device, load_text_encoder=False)
        if inject_m2d
        else None
    )
    audio_autoencoder = None
    vae_report = {
        "config": None,
        "config_sha256": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
    }
    if load_audio_autoencoder:
        audio_autoencoder, vae_report = _load_frozen_foa_vae(device)
    pipeline = ScenePlanTransfusionEditingPipeline(
        diffusion=diffusion,
        editing_ar=ar,
        codec=plan_codec,
        audio_autoencoder=audio_autoencoder,
        source_semantic_encoder=source_semantic_encoder,
    )
    pipeline.eval().requires_grad_(False).to(device)
    report = EditingPipelineLoadReport(
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_step=int(payload["global_step"]),
        model_config=str(model_config_path),
        model_config_sha256=sha256_file(model_config_path),
        editing_ar_contract=EDITING_AR_CONTRACT,
        pipeline_contract=EDITING_PIPELINE_CONTRACT,
        shared_transformer_same_object=(
            pipeline.editing_ar.shared_transformer
            is pipeline.diffusion.model.model.transformer
        ),
        old_sceneplan_input=False,
        source_semantic_mode=str(source_semantic["mode"]),
        source_caption_model_input=False,
        frozen_m2d_clap_assets=(
            None
            if source_semantic_encoder is None
            else source_semantic_encoder.asset_report
        ),
        checkpoint_selection=(None if selection_path is None else str(selection_path)),
        checkpoint_selection_sha256=selection_sha,
        frozen_qwen_runtime=qwen_runtime,
        frozen_vae_config=vae_report["config"],
        frozen_vae_config_sha256=vae_report["config_sha256"],
        frozen_vae_checkpoint=vae_report["checkpoint"],
        frozen_vae_checkpoint_sha256=vae_report["checkpoint_sha256"],
    )
    return pipeline, report


__all__ = [
    "DIAGNOSTIC_EXTERNAL_M2D_EMBEDDING_ORIGIN",
    "EDITING_PIPELINE_CONTRACT",
    "EditingPipelineLoadReport",
    "ScenePlanTransfusionEditingPipeline",
    "load_sceneplan_transfusion_editing_pipeline",
]
