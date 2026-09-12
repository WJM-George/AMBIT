"""Reuse the real four Transfusion routes without altering SFT model classes."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from .objectives import ARRollout, sample_ar


@dataclass(frozen=True)
class GenerationObservation:
    sample_id: str
    request: str


@dataclass(frozen=True)
class EditingObservation:
    sample_id: str
    request: str
    source_foa_latent: Tensor
    source_attention_mask: Tensor
    model_num_samples: int
    source_m2d_audio_embedding: Tensor | None = None

    def __post_init__(self):
        source, mask = self.source_foa_latent, self.source_attention_mask
        if source.ndim != 3 or source.shape[:2] != (1, 64) or source.shape[-1] not in (432, 648):
            raise ValueError("Editing observation requires aligned [1,64,432|648] source")
        if mask.shape != (1, source.shape[-1]) or mask.dtype != torch.bool:
            raise ValueError("Editing source mask geometry/type mismatch")
        frames = math.ceil(self.model_num_samples / 1024)
        expected = torch.arange(source.shape[-1], device=mask.device)[None] < frames
        if not 0 < self.model_num_samples <= 661500 or not torch.equal(mask, expected):
            raise ValueError("Editing source duration and contiguous mask disagree")
        if source.requires_grad or not torch.isfinite(source).all():
            raise ValueError("source observations must be detached and finite")


@dataclass(frozen=True)
class RenderCondition:
    plan: Mapping[str, Any]
    positive: list[dict[str, Any]]
    negative: list[dict[str, Any]]
    mask: Tensor
    model_num_samples: int


def _fork_components(ar: nn.Module, diffusion: nn.Module):
    """Clone trainable candidates together, preserving all internal aliases.

    Frozen Qwen is an unregistered object in this repo; share only that frozen
    backbone, while copying its small projection and role embeddings.
    """
    prompt = getattr(ar, "prompt_conditioner", getattr(ar, "instruction_conditioner", None))
    memo = {}
    backbone = getattr(prompt, "model", None)
    if isinstance(backbone, nn.Module):
        if any(p.requires_grad for p in backbone.parameters()):
            raise ValueError("OPSD requires a frozen Qwen backbone")
        memo[id(backbone)] = backbone
    # Pretransform is external to the RL state. Never deep-copy a frozen VAE.
    pretransform = getattr(diffusion, "pretransform", None)
    if pretransform is not None:
        memo[id(pretransform)] = pretransform
    return copy.deepcopy((ar, diffusion), memo)


class TransfusionOPSDAdapter(nn.Module):
    """One literal shared AR–DiT bundle, either Generation or Editing.

    Generation and Editing are distinct instances with separate parameters.
    Teacher text is accepted only by the explicit teacher method; it is never
    stored in or appended to a student observation.
    """

    contract = "sceneplan_transfusion_execution_coupled_opsd_bundle_v2"

    def __init__(self, *, mode: str, ar: nn.Module, diffusion: nn.Module, codec,
                 audio_autoencoder=None, editing_pipeline=None,
                 fork: bool = True, cfg_scale: float | None = None,
                 cfg_rescale_phi: float | None = None):
        super().__init__()
        if mode not in {"generation", "editing"}:
            raise ValueError("mode must be generation or editing")
        if fork:
            ar, diffusion = _fork_components(ar, diffusion)
        self.mode, self.ar, self.diffusion, self.codec = mode, ar, diffusion, codec
        self.__dict__["audio_autoencoder"] = audio_autoencoder
        self.__dict__["source_pipeline"] = editing_pipeline
        self.cfg_scale = float(cfg_scale if cfg_scale is not None else (3 if mode == "generation" else 1))
        self.cfg_rescale_phi = float(cfg_rescale_phi if cfg_rescale_phi is not None else (0.4 if mode == "generation" else 0))
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0 or not 0 <= self.cfg_rescale_phi <= 1:
            raise ValueError("invalid CFG settings")
        if mode == "editing" and self.cfg_rescale_phi != 0:
            raise ValueError("current Editing pipeline has no CFG rescale")
        core = diffusion.model.model
        if ar.shared_transformer is not core.transformer:
            raise ValueError("AR and DiT must own the same Transformer object")
        self.diffusion.pretransform = None
        self.ar.requires_grad_(True)
        self.diffusion.requires_grad_(True)
        prompt = self.prompt_conditioner
        if prompt is not self.diffusion.conditioner.conditioners["prompt"]:
            raise ValueError("AR and DiT must share their prompt projection/role parameters")
        if bool(getattr(prompt, "enable_grad", False)):
            raise ValueError("Qwen enable_grad must remain false")
        backbone = getattr(prompt, "model", None)
        if isinstance(backbone, nn.Module):
            backbone.eval().requires_grad_(False)
        if audio_autoencoder is not None:
            audio_autoencoder.eval().requires_grad_(False)
        self.float().eval()

    @property
    def device(self):
        return next(self.ar.parameters()).device

    def frozen_copy(self):
        """Independent complete renderer snapshot for the explicit fixed-feedback ablation."""
        result = type(self)(mode=self.mode, ar=self.ar, diffusion=self.diffusion,
            codec=self.codec, audio_autoencoder=self.audio_autoencoder,
            editing_pipeline=self.source_pipeline, cfg_scale=self.cfg_scale,
            cfg_rescale_phi=self.cfg_rescale_phi)
        return result.eval().requires_grad_(False)

    @property
    def prompt_conditioner(self):
        return self.ar.prompt_conditioner if self.mode == "generation" else self.ar.instruction_conditioner

    @classmethod
    def from_editing_pipeline(cls, pipeline, **kwargs):
        return cls(mode="editing", ar=pipeline.editing_ar, diffusion=pipeline.diffusion,
                   codec=pipeline.codec, audio_autoencoder=pipeline.audio_autoencoder,
                   editing_pipeline=pipeline, **kwargs)

    @classmethod
    def from_generation_components(cls, *, diffusion, codec, ar_adapter_state,
                                   audio_autoencoder=None, **kwargs):
        """Use an independently loaded EMA P10 diffusion wrapper + AR adapter.

        Unlike the SFT checkpoint, the resulting RL checkpoint includes the
        entire acoustic field and trainable conditioners. Input objects are
        copied before the existing Generation constructor freezes anything.
        """
        from ...models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR
        prompt = diffusion.conditioner.conditioners["prompt"]
        memo = {}
        if isinstance(getattr(prompt, "model", None), nn.Module):
            if any(p.requires_grad for p in prompt.model.parameters()):
                raise ValueError("Qwen must already be frozen")
            memo[id(prompt.model)] = prompt.model
        pretransform = getattr(diffusion, "pretransform", None)
        if pretransform is not None:
            memo[id(pretransform)] = pretransform
        candidate = copy.deepcopy(diffusion, memo)
        ar = ScenePlanTransfusionGenerationAR(
            p10_dit=candidate.model.model,
            prompt_conditioner=candidate.conditioner.conditioners["prompt"],
            pad_id=codec.pad_id, vocab_size=codec.vocab_size,
        )
        ar.load_trainable_state_dict(ar_adapter_state)
        return cls(mode="generation", ar=ar, diffusion=candidate, codec=codec,
                   audio_autoencoder=audio_autoencoder, fork=False, **kwargs)

    def observe_editing(self, *, sample_id, request, source_foa_latent,
                        source_attention_mask, model_num_samples):
        """Obtain the canonical source-only M2D view via the existing pipeline."""
        if self.mode != "editing":
            raise ValueError("source observations belong only to Editing")
        source = source_foa_latent.detach().to(self.device, dtype=torch.float32)
        mask = source_attention_mask.detach().to(self.device, dtype=torch.bool)
        embedding = None
        from ...models.sceneplan_transfusion_editing_m2d_clap import editing_m2d_mode_flags
        inject, _ = editing_m2d_mode_flags(self.ar.source_semantic_mode)
        if inject:
            if self.source_pipeline is None:
                raise ValueError("M2D mode requires the validated source pipeline")
            embedding = self.source_pipeline.encode_source_m2d_audio(
                source, mask, model_num_samples=[model_num_samples]).detach().to(self.device)
        return EditingObservation(sample_id, request, source, mask, model_num_samples, embedding)

    def _encode_requests(self, texts):
        # The SFT Generation method is @no_grad AND explicitly detaches. Repeat
        # its tokenization boundary while retaining projection gradients for RL.
        tokenizer = self.prompt_conditioner.tokenizer
        encoded = tokenizer(texts, add_special_tokens=True, padding=True,
                            truncation=False, return_tensors="pt")
        ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        mask = torch.as_tensor(encoded["attention_mask"], dtype=torch.bool)
        if not mask.any(-1).all() or (mask.sum(-1) > 512).any():
            raise ValueError("student/teacher request must fit the existing 512-token boundary")
        zeros = torch.zeros_like(ids)
        rows = [dict(input_ids=ids[i], attention_mask=mask[i],
                     event_source_ids=zeros[i], speech_source_ids=zeros[i]) for i in range(len(texts))]
        context, keep = self.prompt_conditioner(rows, self.device)
        return self.diffusion.model.model.to_cond_embed(context), keep.bool()

    def _ar_logits(self, observation, token_ids, text, *, return_query=False):
        expected = GenerationObservation if self.mode == "generation" else EditingObservation
        if not isinstance(observation, expected) or not text.strip():
            raise ValueError("observation does not match the active route")
        context, context_mask = self._encode_requests([text])
        return self._forward_ar(observation, token_ids, context, context_mask, return_query=return_query)

    def _forward_ar(self, observation, token_ids, context, context_mask, *, return_query=False):
        tokens = token_ids.to(self.device)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        if self.mode == "generation":
            if return_query:
                raise ValueError("Generation has no source-caption auxiliary")
            return self.ar(tokens, mask, context, context_mask)
        return self.ar(observation.source_foa_latent, observation.source_attention_mask,
                       tokens, mask, context, context_mask,
                       source_m2d_audio_embedding=observation.source_m2d_audio_embedding,
                       return_source_contrastive_query=return_query)

    def student_logits(self, observation, token_ids, *, return_query=False):
        return self._ar_logits(observation, token_ids, observation.request, return_query=return_query)

    @torch.no_grad()
    def teacher_logits(self, observation, token_ids, *, privileged_text: str):
        return self._ar_logits(observation, token_ids, privileged_text).detach()

    def allowed_next_ids(self, observation, prefix):
        if self.mode == "editing":
            from ...data.sceneplan_transfusion_editing_plan import editing_ar_allowed_next_ids
            return editing_ar_allowed_next_ids(self.codec, prefix,
                fixed_duration_sec=observation.model_num_samples / 44100)
        return self.codec.allowed_next_ids(prefix)

    @torch.no_grad()
    def sample_plan(self, observation, *, privileged_text=None, **kwargs) -> ARRollout:
        expected = GenerationObservation if self.mode == "generation" else EditingObservation
        if not isinstance(observation, expected):
            raise ValueError("observation does not match the active route")
        text = observation.request if privileged_text is None else privileged_text
        if not text.strip():
            raise ValueError("rollout text must not be empty")
        # Frozen Qwen/request projections are constant for this rollout. Cache
        # context once; fitting still recomputes projections with gradients.
        context, context_mask = self._encode_requests([text])
        def logits(tokens):
            return self._forward_ar(observation, tokens, context, context_mask)
        return sample_ar(logits, self.codec, device=self.device,
                         allowed_fn=lambda prefix: self.allowed_next_ids(observation, prefix), **kwargs)

    def decode_plan(self, observation, rollout: ARRollout):
        if not rollout.finished:
            raise ValueError("incomplete AR samples cannot be silently repaired for DiT")
        plan = self.codec.decode(rollout.token_ids, sample_id=observation.sample_id)
        if self.mode == "editing":
            from ...models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
            plan = _align_decoded_sceneplan_to_audio_duration(plan, observation.model_num_samples / 44100)
        return plan

    def render_condition(self, observation, plan) -> RenderCondition:
        from ...data.model_sceneplan import make_sceneplan_cfg_unknown_metadata
        if self.mode == "editing":
            from ...models.sceneplan_transfusion_editing_pipeline import ScenePlanTransfusionEditingDiTPipeline
            from ...data.sceneplan_transfusion_editing_dataset import make_editing_dit_cfg_unknown_metadata
            # This lightweight view adds no parameters and uses the current RL
            # conditioner, never the frozen source-observation pipeline's DiT.
            view = ScenePlanTransfusionEditingDiTPipeline(diffusion=self.diffusion)
            rows = view._conditioning_metadata(observation.source_foa_latent,
                observation.source_attention_mask, [plan], [observation.model_num_samples])
            negative = [make_editing_dit_cfg_unknown_metadata(row) for row in rows]
            return RenderCondition(plan, rows, negative, observation.source_attention_mask,
                                   observation.model_num_samples)
        from ...data.sceneplan_p11_single_turn import finalize_sceneplan_for_p10, P11Task
        bundle = finalize_sceneplan_for_p10(self.codec, self.codec.encode(plan),
            tokenizer=self.prompt_conditioner.tokenizer, task=P11Task.GENERATION,
            sample_id=observation.sample_id)
        rows = [dict(bundle.p10_metadata)]
        mask = torch.ones((1, bundle.latent_frames_valid), device=self.device, dtype=torch.bool)
        return RenderCondition(plan, rows, [make_sceneplan_cfg_unknown_metadata(rows[0])],
                               mask, bundle.model_num_samples)

    def velocity_function(self, condition: RenderCondition, *, differentiable: bool):
        """Rebuild small conditioner projections for every fitting closure."""
        context = nullcontext() if differentiable else torch.no_grad()
        with context:
            positive = self.diffusion.get_conditioning_inputs(
                self.diffusion.conditioner(condition.positive, self.device))
            negative = None
            if self.cfg_scale != 1:
                negative = self.diffusion.get_conditioning_inputs(
                    self.diffusion.conditioner(condition.negative, self.device),
                    negative=self.mode == "generation")

        def velocity(z, t):
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            with amp:
                if self.mode == "generation":
                    return self.diffusion.model(z, t, **positive, **(negative or {}),
                        cfg_scale=self.cfg_scale, batch_cfg=True,
                        scale_phi=self.cfg_rescale_phi, apg_scale=0.,
                        cfg_dropout_prob=0., padding_mask=condition.mask).float()
                pos = self.diffusion.model(z, t, **positive, cfg_dropout_prob=0., padding_mask=condition.mask)
                if negative is None:
                    return pos.float()
                neg = self.diffusion.model(z, t, **negative, cfg_dropout_prob=0., padding_mask=condition.mask)
                return (neg + self.cfg_scale * (pos - neg)).float()
        return velocity

    def schedule(self, steps: int, frames: int):
        if steps < 1:
            raise ValueError("steps must be positive")
        if self.mode == "editing":
            return tuple(1 - i / steps for i in range(steps)) + (0.,)
        from ...inference.sampling import build_schedule
        return tuple(build_schedule(steps=steps, sigma_max=1.,
            dist_shift=self.diffusion.sampling_dist_shift, effective_seq_len=None,
            fallback_seq_len=frames, include_endpoint=True, device=self.device).float().tolist())

    def decode_for_reward(self, latent: Tensor, model_num_samples: int) -> Tensor:
        """Frozen VAE weights, differentiable latent input (no inference wrapper)."""
        if self.audio_autoencoder is None:
            raise ValueError("audio rewards require the frozen FOA VAE")
        with torch.autocast(device_type=self.device.type, enabled=False):
            waveform = self.audio_autoencoder.decode(latent.float()).float()
        if waveform.ndim != 3 or waveform.shape[1] != 4 or waveform.shape[-1] < model_num_samples:
            raise ValueError("FOA VAE reward decode geometry changed")
        return waveform[..., :model_num_samples]

    def dependency_parameters(self):
        """Structural route dependencies; acoustic project_in/out are DiT-only."""
        transformer = self.ar.shared_transformer
        excluded = {id(p) for module in (transformer.project_in, transformer.project_out,
                    getattr(transformer, "global_cond_embedder", None)) if module is not None
                    for p in module.parameters()}
        shared = {id(p) for p in transformer.parameters() if id(p) not in excluded}
        shared.update(id(p) for p in self.prompt_conditioner.parameters())
        shared.update(id(p) for p in self.diffusion.model.model.to_cond_embed.parameters())
        dit = {id(p) for p in self.diffusion.parameters() if p.requires_grad}
        all_parameters = [(name, p) for name, p in self.named_parameters() if p.requires_grad]
        return {
            "shared": [(n, p) for n, p in all_parameters if id(p) in shared],
            "ar_private": [(n, p) for n, p in all_parameters if id(p) not in dit],
            "dit_private": [(n, p) for n, p in all_parameters if id(p) in dit and id(p) not in shared],
        }
