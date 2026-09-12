"""Pinned external M2D-CLAP runtime for Editing-only cache/inference.

The upstream implementation and weights remain outside this repository.  We
verify their immutable identities before importing the unmodified portable
loader.  The runtime is frozen and is never placed in the Editing optimizer.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import version as distribution_version
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchaudio

from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (
    EDITING_M2D_CLAP_EMBED_DIM,
)


M2D_CLAP_EXTERNAL_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/m2d-clap-v0.5.0"
)
M2D_CLAP_PORTABLE_SOURCE = M2D_CLAP_EXTERNAL_ROOT / (
    "m2d-source-3d0c4de/examples/portable_m2d.py"
)
M2D_CLAP_CHECKPOINT = M2D_CLAP_EXTERNAL_ROOT / (
    "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth"
)
M2D_CLAP_LICENSE = M2D_CLAP_EXTERNAL_ROOT / "m2d-source-3d0c4de/LICENSE.pdf"
M2D_CLAP_HF_HOME = M2D_CLAP_EXTERNAL_ROOT / "hf-cache"
M2D_CLAP_BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
M2D_CLAP_BERT_SNAPSHOT = M2D_CLAP_HF_HOME / (
    "hub/models--google-bert--bert-base-uncased/snapshots/"
    + M2D_CLAP_BERT_REVISION
)
M2D_CLAP_UPSTREAM_COMMIT = "3d0c4de9447c404a8d3f9f37e04f53bc902e09b3"
M2D_CLAP_PORTABLE_SHA256 = (
    "b64d94a18b558c0643b1bba578f215629e3dd00656284a8cbf992fbc2f563ee2"
)
M2D_CLAP_CHECKPOINT_SHA256 = (
    "238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1"
)
M2D_CLAP_RELEASE_ZIP_SHA256 = (
    "fd193ae591720df7f1e27ed728ce127e0309b8bd427f0f4b3e5cd17d7ee5e1e1"
)
M2D_CLAP_LICENSE_SHA256 = (
    "0b788bf171752248ae738249a52c934811a39fdd939039941e38065afcf919fc"
)
M2D_CLAP_BERT_FILES = {
    "config.json": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
    "model.safetensors": (
        "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
    ),
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": (
        "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"
    ),
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
M2D_CLAP_INPUT_SAMPLE_RATE = 16_000
M2D_CLAP_FOA_CHANNEL = "W"
M2D_CLAP_NORMALIZATION = "valid_W_per_clip_peak_minus_1db"
M2D_CLAP_SOURCE_AUDIO_VIEW = "frozen_vae_decode_of_source_foa_latent_W"
M2D_CLAP_PATCH_TIME_FRAMES = 16
M2D_CLAP_CHUNK_CONTENT_FRAMES = 992
M2D_CLAP_TEMPORAL_POLICY = (
    "aligned_992mel_full_coverage_chunks_concat_then_audio_semantic_token_v2"
)


# ``torchaudio.functional.resample`` rebuilds the same sinc kernel on every
# call.  The formal cache contains one million rows, so keep one immutable
# kernel per device while preserving the exact same anti-aliased transform at
# cache construction and real-audio inference time.
_M2D_RESAMPLERS: dict[tuple[str, int | None], torchaudio.transforms.Resample] = {}


def _m2d_resampler(device: torch.device) -> torchaudio.transforms.Resample:
    key = (device.type, device.index)
    resampler = _M2D_RESAMPLERS.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(
            orig_freq=44_100,
            new_freq=M2D_CLAP_INPUT_SAMPLE_RATE,
            dtype=torch.float32,
        ).to(device)
        resampler.eval().requires_grad_(False)
        _M2D_RESAMPLERS[key] = resampler
    return resampler


def expected_editing_m2d_clap_asset_report(
    *, require_text: bool
) -> dict[str, Any]:
    """Return the canonical encoder identity without loading model weights."""

    bert_files = (
        {
            name: {
                "path": str((M2D_CLAP_BERT_SNAPSHOT / name).resolve()),
                "sha256": expected_sha,
            }
            for name, expected_sha in M2D_CLAP_BERT_FILES.items()
        }
        if require_text
        else {}
    )
    return {
        "upstream_repository": "https://github.com/nttcslab/m2d",
        "upstream_commit": M2D_CLAP_UPSTREAM_COMMIT,
        "release": "v0.5.0",
        "release_zip_sha256": M2D_CLAP_RELEASE_ZIP_SHA256,
        "portable_source": str(M2D_CLAP_PORTABLE_SOURCE.resolve()),
        "portable_source_sha256": M2D_CLAP_PORTABLE_SHA256,
        "checkpoint": str(M2D_CLAP_CHECKPOINT.resolve()),
        "checkpoint_sha256": M2D_CLAP_CHECKPOINT_SHA256,
        "license": str(M2D_CLAP_LICENSE.resolve()),
        "license_sha256": M2D_CLAP_LICENSE_SHA256,
        "license_scope": "internal_noncommercial_evaluation_only",
        "flat_features": True,
        "audio_embedding_dim": EDITING_M2D_CLAP_EMBED_DIM,
        "text_embedding_dim": EDITING_M2D_CLAP_EMBED_DIM,
        "sample_rate": M2D_CLAP_INPUT_SAMPLE_RATE,
        "foa_channel": M2D_CLAP_FOA_CHANNEL,
        "normalization": M2D_CLAP_NORMALIZATION,
        "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
        "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
        "bert_repository": (
            "google-bert/bert-base-uncased" if require_text else None
        ),
        "bert_revision": M2D_CLAP_BERT_REVISION if require_text else None,
        "bert_files": bert_files,
    }


def editing_m2d_software_runtime_fingerprint() -> dict[str, Any]:
    """Return package/backend settings independent of a particular GPU."""

    packages = {
        name: distribution_version(name)
        for name in ("torch", "torchaudio", "nnAudio", "timm", "transformers")
    }
    return {
        "schema": "editing_m2d_numeric_runtime_fingerprint_v2",
        "packages": packages,
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_fp16_accumulation": bool(
            torch.backends.cuda.matmul.allow_fp16_accumulation
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_benchmark_limit": int(torch.backends.cudnn.benchmark_limit),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cuda_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "cuda_mem_efficient_sdp_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "cuda_math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "cuda_cudnn_sdp_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "torch_allow_tf32_cublas_override": os.environ.get(
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
        ),
    }


def editing_m2d_numeric_runtime_fingerprint(
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Freeze software and GPU facts that can change cached FP16 features."""

    if not torch.cuda.is_available():
        raise RuntimeError("Editing M2D numeric fingerprint requires CUDA")
    resolved = torch.device(
        "cuda", torch.cuda.current_device()
    ) if device is None else torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("Editing M2D numeric fingerprint requires a CUDA device")
    properties = torch.cuda.get_device_properties(resolved)
    return {
        **editing_m2d_software_runtime_fingerprint(),
        "device_name": str(properties.name),
        "device_capability": list(torch.cuda.get_device_capability(resolved)),
    }


def verify_editing_m2d_clap_assets(*, require_text: bool) -> dict[str, Any]:
    """Fail closed unless the exact external evaluation assets are present."""

    source = M2D_CLAP_PORTABLE_SOURCE.resolve(strict=True)
    checkpoint = M2D_CLAP_CHECKPOINT.resolve(strict=True)
    license_path = M2D_CLAP_LICENSE.resolve(strict=True)
    observed = {
        "portable_source": sha256_file(source),
        "checkpoint": sha256_file(checkpoint),
        "license": sha256_file(license_path),
    }
    expected = {
        "portable_source": M2D_CLAP_PORTABLE_SHA256,
        "checkpoint": M2D_CLAP_CHECKPOINT_SHA256,
        "license": M2D_CLAP_LICENSE_SHA256,
    }
    if observed != expected:
        raise RuntimeError("external M2D-CLAP code/weight/license identity changed")
    bert_files = {}
    if require_text:
        reference = M2D_CLAP_HF_HOME / (
            "hub/models--google-bert--bert-base-uncased/refs/main"
        )
        if reference.read_text(encoding="utf-8").strip() != M2D_CLAP_BERT_REVISION:
            raise RuntimeError("M2D BERT tokenizer cache revision changed")
        for name, expected_sha in M2D_CLAP_BERT_FILES.items():
            path = (M2D_CLAP_BERT_SNAPSHOT / name).resolve(strict=True)
            actual = sha256_file(path)
            if actual != expected_sha:
                raise RuntimeError(f"M2D BERT asset changed: {name}")
            bert_files[name] = {"path": str(path), "sha256": actual}
    report = expected_editing_m2d_clap_asset_report(require_text=require_text)
    if report["bert_files"] != bert_files:
        raise RuntimeError("M2D BERT asset report changed")
    return report


def _import_pinned_portable() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stable_audio_tools_external_pinned_m2d_portable",
        M2D_CLAP_PORTABLE_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct pinned M2D portable import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenEditingM2DCLAP(nn.Module):
    """Frozen 768-d M2D-CLAP audio/text encoder used outside training graph."""

    def __init__(
        self, *, device: torch.device | str, load_text_encoder: bool
    ) -> None:
        super().__init__()
        if os.environ.get("M2D_NONCOMMERCIAL_EVALUATION_ACK") != "1":
            raise RuntimeError(
                "M2D-CLAP is licensed for internal non-commercial evaluation; "
                "set M2D_NONCOMMERCIAL_EVALUATION_ACK=1 only when authorized"
            )
        self.asset_report = verify_editing_m2d_clap_assets(
            require_text=bool(load_text_encoder)
        )
        # Keep every incidental Transformers lookup offline.  The text tower
        # below is constructed from the explicit pinned snapshot path rather
        # than relying on environment-dependent Hugging Face cache globals.
        os.environ["HF_HOME"] = str(M2D_CLAP_HF_HOME)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        portable = _import_pinned_portable()
        runtime = portable.PortableM2D(
            str(M2D_CLAP_CHECKPOINT),
            cfg=portable.Config(),
            flat_features=True,
        )
        # This particular pBpTI checkpoint owns AudioSemanticProj.  Its
        # semantic token attends the patch sequence produced by all aligned
        # 992-mel chunks.  A generic CLAP head would average elsewhere and
        # would not implement the frozen 15-second policy below.
        audio_projection = getattr(runtime.backbone, "audio_proj", None)
        if getattr(audio_projection, "dont_average", False) is not True:
            raise RuntimeError(
                "pinned M2D checkpoint lost official all-chunk semantic aggregation"
            )
        runtime.eval().requires_grad_(False).to(device)
        if load_text_encoder:
            text_encoder = portable.BertXEncoder(
                clip_weight=str(M2D_CLAP_BERT_SNAPSHOT)
            )
            weights = torch.load(
                M2D_CLAP_CHECKPOINT,
                map_location="cpu",
                weights_only=False,
            )
            weights = weights["model"] if "model" in weights else weights
            if any("module.ar.runtime." in key for key in weights):
                weights = {
                    key.replace("module.ar.runtime.", ""): value
                    for key, value in weights.items()
                }
            weights = portable.extract_weight(weights, "text_encoder.")
            incompatible = text_encoder.load_state_dict(weights, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("pinned M2D text encoder weights did not match")
            runtime.text_encoder = text_encoder.to(device)
            runtime.text_encoder.eval().requires_grad_(False)
        self.runtime = runtime
        self.load_text_encoder = bool(load_text_encoder)
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("frozen Editing M2D-CLAP retained trainable parameters")

    @property
    def device(self) -> torch.device:
        return next(self.runtime.backbone.parameters()).device

    @torch.inference_mode()
    def encode_audio(self, waveform_16khz: Tensor) -> Tensor:
        if waveform_16khz.ndim != 2 or int(waveform_16khz.shape[-1]) < 400:
            raise ValueError("M2D audio input must be mono [B,N>=400] at 16 kHz")
        # ``inference_mode`` does not disable an autocast region owned by a
        # caller.  Cache construction is FP32 through the frozen mel/backbone/
        # projector, so enforce the same arithmetic here before the shared
        # CPU-L2 -> FP16 model boundary canonicalizes the result.
        with torch.autocast(device_type=self.device.type, enabled=False):
            value = waveform_16khz.to(device=self.device, dtype=torch.float32)
            if not bool(torch.isfinite(value).all()) or float(value.abs().amax()) > 1.0001:
                raise ValueError("M2D audio waveform must be finite and within [-1,1]")
            # The upstream 1001-frame chunker silently drops nine mel frames at
            # every boundary because its patch kernel/stride is 16.  Tile the same
            # frozen backbone in its effective 992-frame receptive width instead:
            # every real mel frame then enters exactly one patch (the final chunk
            # is right-padded), and the original AudioSemanticProj still aggregates
            # all concatenated patch tokens into one global vector. This is global
            # semantic coverage, not a claim of exact temporal coordinates; those
            # remain the job of the aligned 64xT source latent.
            normalized = self.runtime.to_normalized_feature(value).float()
            backbone = self.runtime.backbone
            patch_frames = int(backbone.patch_size()[1])
            unit_frames = int(self.runtime.cfg.input_size[1])
            content_frames = (unit_frames // patch_frames) * patch_frames
            if (
                not bool(self.runtime.cfg.flat_features)
                or patch_frames != M2D_CLAP_PATCH_TIME_FRAMES
                or content_frames != M2D_CLAP_CHUNK_CONTENT_FRAMES
                or getattr(backbone.audio_proj, "dont_average", False) is not True
            ):
                raise RuntimeError("pinned M2D seamless chunk geometry changed")
            patch_embeddings = []
            for start in range(0, int(normalized.shape[-1]), content_frames):
                chunk = normalized[..., start : start + content_frames]
                remainder = int(chunk.shape[-1]) % patch_frames
                if remainder:
                    chunk = F.pad(chunk, (0, patch_frames - remainder))
                encoded = backbone.forward_encoder(chunk.float())
                patch_embeddings.append(encoded[..., 1:, :].float())
            if not patch_embeddings:
                raise RuntimeError("M2D seamless chunker produced no patch tokens")
            embedding = backbone.audio_proj(
                torch.cat(patch_embeddings, dim=-2).float()
            ).float()
            if tuple(embedding.shape) != (
                int(value.shape[0]),
                EDITING_M2D_CLAP_EMBED_DIM,
            ) or not bool(torch.isfinite(embedding).all()):
                raise RuntimeError("pinned M2D-CLAP returned an invalid audio embedding")
            return F.normalize(embedding, dim=-1)

    @torch.inference_mode()
    def encode_text(self, captions: Sequence[str]) -> Tensor:
        if not self.load_text_encoder:
            raise RuntimeError("this frozen M2D runtime omitted its text encoder")
        if not captions or not all(
            isinstance(value, str) and value.strip() for value in captions
        ):
            raise ValueError("M2D source captions must be non-empty strings")
        with torch.autocast(device_type=self.device.type, enabled=False):
            embedding = self.runtime.encode_clap_text(
                list(captions), truncate=True
            ).float()
            if tuple(embedding.shape) != (
                len(captions),
                EDITING_M2D_CLAP_EMBED_DIM,
            ) or not bool(torch.isfinite(embedding).all()):
                raise RuntimeError("pinned M2D-CLAP returned an invalid text embedding")
            return F.normalize(embedding, dim=-1)


def decoded_foa_w_to_m2d_waveform(
    decoded_foa: Tensor, *, valid_samples: int
) -> Tensor:
    """Take valid decoded WYZX W, peak-normalize, and sinc-resample to 16 kHz."""

    count = int(valid_samples)
    if decoded_foa.ndim != 2 or int(decoded_foa.shape[0]) != 4:
        raise ValueError("decoded FOA must be [4,N] in WYZX order")
    if count <= 0 or count > int(decoded_foa.shape[-1]):
        raise ValueError("M2D valid sample count is outside decoded FOA")
    with torch.autocast(device_type=decoded_foa.device.type, enabled=False):
        waveform = decoded_foa[0, :count].float()
        if not bool(torch.isfinite(waveform).all()):
            raise RuntimeError("decoded FOA W contains non-finite values")
        peak = waveform.abs().amax()
        if float(peak) > 1.0e-8:
            waveform = waveform / peak * (10.0 ** (-1.0 / 20.0))
        output_samples = max(
            400, round(count * M2D_CLAP_INPUT_SAMPLE_RATE / 44_100)
        )
        # Linear interpolation aliases energy above the new 8 kHz Nyquist limit
        # into M2D's semantic band.  A cached band-limited sinc kernel prevents
        # that source-only side channel from learning spurious folded frequencies.
        waveform = _m2d_resampler(waveform.device)(waveform.float()).float()
        if int(waveform.shape[-1]) < output_samples:
            waveform = F.pad(
                waveform, (0, output_samples - int(waveform.shape[-1]))
            )
        else:
            waveform = waveform[:output_samples]
        return waveform.clamp(-1.0, 1.0)


__all__ = [
    "FrozenEditingM2DCLAP",
    "M2D_CLAP_BERT_REVISION",
    "M2D_CLAP_CHECKPOINT",
    "M2D_CLAP_CHECKPOINT_SHA256",
    "M2D_CLAP_EXTERNAL_ROOT",
    "M2D_CLAP_INPUT_SAMPLE_RATE",
    "M2D_CLAP_LICENSE_SHA256",
    "M2D_CLAP_PORTABLE_SHA256",
    "M2D_CLAP_SOURCE_AUDIO_VIEW",
    "M2D_CLAP_TEMPORAL_POLICY",
    "M2D_CLAP_UPSTREAM_COMMIT",
    "decoded_foa_w_to_m2d_waveform",
    "editing_m2d_numeric_runtime_fingerprint",
    "editing_m2d_software_runtime_fingerprint",
    "expected_editing_m2d_clap_asset_report",
    "verify_editing_m2d_clap_assets",
]
