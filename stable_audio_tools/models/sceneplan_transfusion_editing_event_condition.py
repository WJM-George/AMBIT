"""Wire route C into Editing Transfusion without changing the A default path.

OPSD launchers, configs and run directories are not imported here. Existing
joint 40k training files are not modified: this module is an opt-in wrapper
around the same shared Transformer, discrete ScenePlan and ``sceneplan_44``.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from stable_audio_tools.models.sceneplan_transfusion_editing_event_head import (
    EditingEventHead,
    EventHeadConfig,
    gather_plan_states,
    inplace_caption_condition,
    plan_readout_positions,
    pool_qwen_event_targets,
)


def caption_roles_from_metadata(metadata: Sequence[dict[str, Any]], length: int, device) -> dict[str, Tensor]:
    aligned = {key: [] for key in ("event_source_ids", "speech_source_ids", "attention_mask")}
    for row in metadata:
        prompt = row["prompt"]
        if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, dict):
            raise ValueError("event-head C requires the tokenized Editing caption")
        raw_mask = torch.as_tensor(prompt["attention_mask"]).bool()
        if bool(raw_mask[length:].any()):
            raise ValueError("Qwen truncated valid caption tokens")
        for key in aligned:
            value = torch.as_tensor(prompt[key], device=device)
            if value.ndim != 1 or value.shape != raw_mask.shape:
                raise ValueError("caption token roles must align with the input mask")
            padded = torch.zeros(length, dtype=value.dtype, device=device)
            padded[: min(length, len(value))] = value[:length]
            aligned[key].append(padded)
    return {key: torch.stack(value) for key, value in aligned.items()}


def replace_editing_prompt_event_spans(
    conditions: dict[str, Any],
    metadata: Sequence[dict[str, Any]],
    event_vectors: Tensor,
    event_mask: Tensor,
    *,
    max_events: int,
) -> dict[str, Any]:
    """Swap only event-description spans inside the already-encoded prompt."""
    if "prompt" not in conditions:
        raise ValueError("Editing conditioner did not return a prompt caption")
    caption, caption_mask = conditions["prompt"][:2]
    roles = caption_roles_from_metadata(metadata, caption.shape[1], caption.device)
    if not torch.equal(roles["attention_mask"].bool(), caption_mask.bool()):
        raise ValueError("Qwen changed token positions; event spans no longer align")
    replaced, mask = inplace_caption_condition(
        caption,
        caption_mask,
        roles["event_source_ids"],
        roles["speech_source_ids"],
        event_vectors,
        event_mask,
        max_events=max_events,
    )
    updated = dict(conditions)
    updated["prompt"] = (replaced, mask)
    return updated


def encode_complete_plans(codec, plans: Sequence[Any], token_ids: Sequence[Any] | None, device):
    if token_ids is None:
        token_ids = [codec.encode(plan)["input_ids"] for plan in plans]
    rows = [torch.as_tensor(row, device=device, dtype=torch.long) for row in token_ids]
    ids = pad_sequence(rows, batch_first=True, padding_value=codec.pad_id)
    lengths = torch.tensor([len(row) for row in rows], device=ids.device)
    mask = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
    return ids, mask


class EventHeadEditingTransfusion(nn.Module):
    """One Editing AR+DiT forward whose DiT prompt uses route C.

    AR still emits the discrete ScenePlan. DiT still sees ``sceneplan_44`` and
    the clean source latent. Qwen still encodes the same caption; only event
    description spans are overwritten by the event head.
    """

    def __init__(self, *, diffusion, ar, codec, event_head: EditingEventHead | None = None):
        super().__init__()
        if ar.shared_transformer is not diffusion.model.model.transformer:
            raise RuntimeError("Editing AR and DiT must share one Transformer object")
        if list(diffusion.cross_attn_cond_ids) != ["prompt"]:
            raise ValueError("event-head C expects the native prompt cross-attention")
        self.diffusion = diffusion
        self.ar = ar
        self.codec = codec
        self.event_head = event_head or EditingEventHead(EventHeadConfig())

    def _ar_states(
        self,
        *,
        source_foa_latent,
        source_attention_mask,
        plan_input_ids,
        plan_attention_mask,
        raw_edit_requests,
        source_m2d_audio_embedding=None,
        source_clap_features=None,
    ):
        context, context_mask = self.ar.encode_edit_instructions(
            raw_edit_requests, device=source_foa_latent.device
        )
        positions = plan_readout_positions(
            self.codec,
            plan_input_ids,
            plan_attention_mask,
            max_events=self.event_head.config.max_events,
            allow_speech=self.event_head.config.speech_tokens,
        )
        captured: list[tuple[Tensor, Tensor]] = []

        def read_states(_module, args):
            hidden = args[0]
            if hidden.shape[:2] != plan_input_ids.shape:
                raise RuntimeError("event-head hook did not receive plan-token states")
            captured.append(gather_plan_states(hidden, positions))

        extras = {}
        if source_m2d_audio_embedding is not None:
            extras["source_m2d_audio_embedding"] = source_m2d_audio_embedding
        if source_clap_features is not None:
            extras["source_clap_features"] = source_clap_features
        handle = self.ar.plan_adapter.output_norm.register_forward_pre_hook(read_states)
        try:
            ar_output = self.ar(
                source_foa_latent,
                source_attention_mask,
                plan_input_ids,
                plan_attention_mask,
                context,
                context_mask,
                return_source_contrastive_query=self.ar.source_semantic_bridge.align_caption,
                **extras,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("expected exactly one plan output_norm call")
        return ar_output, captured[0][0], captured[0][1]

    def teacher_event_targets(self, metadata, device):
        prompts = []
        for row in metadata:
            prompt = row["prompt"]
            if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
                prompt = prompt[0]
            prompts.append(prompt)
        embeddings, mask = self.diffusion.conditioner.conditioners["prompt"](prompts, device)[:2]
        roles = caption_roles_from_metadata(metadata, embeddings.shape[1], embeddings.device)
        if not torch.equal(roles["attention_mask"].bool(), mask.bool()):
            raise ValueError("Qwen changed token positions; teacher spans no longer align")
        return pool_qwen_event_targets(
            embeddings,
            mask,
            roles["event_source_ids"],
            roles["speech_source_ids"],
            max_events=self.event_head.config.max_events,
            allow_speech=self.event_head.config.speech_tokens,
        )

    def forward(
        self,
        *,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        plan_input_ids: Tensor,
        plan_attention_mask: Tensor,
        raw_edit_requests: Sequence[str],
        metadata: list[dict[str, Any]],
        noised_target: Tensor,
        timesteps: Tensor,
        rf_padding_mask: Tensor,
        source_m2d_audio_embedding: Tensor | None = None,
        source_clap_features: dict[str, Any] | None = None,
    ):
        ar_output, states, event_mask = self._ar_states(
            source_foa_latent=source_foa_latent,
            source_attention_mask=source_attention_mask,
            plan_input_ids=plan_input_ids,
            plan_attention_mask=plan_attention_mask,
            raw_edit_requests=raw_edit_requests,
            source_m2d_audio_embedding=source_m2d_audio_embedding,
            source_clap_features=source_clap_features,
        )
        if self.ar.source_semantic_bridge.align_caption:
            ar_logits, source_caption_query = ar_output
        else:
            ar_logits = ar_output
            source_caption_query = source_foa_latent.new_empty((0, 768))
        predicted = self.event_head.adapt(self.event_head(states, event_mask), event_mask)
        conditioning = self.diffusion.conditioner(metadata, source_foa_latent.device)
        caption, caption_mask = conditioning["prompt"][:2]
        roles = caption_roles_from_metadata(metadata, caption.shape[1], caption.device)
        if not torch.equal(roles["attention_mask"].bool(), caption_mask.bool()):
            raise ValueError("Qwen changed token positions; event spans no longer align")
        teacher, teacher_mask = pool_qwen_event_targets(
            caption,
            caption_mask,
            roles["event_source_ids"],
            roles["speech_source_ids"],
            max_events=self.event_head.config.max_events,
            allow_speech=self.event_head.config.speech_tokens,
        )
        if not torch.equal(event_mask, teacher_mask):
            raise ValueError("AR source slots and teacher event spans disagree")
        conditioned_source = conditioning.get("source_foa_latent", [None])[0]
        if (
            not isinstance(conditioned_source, torch.Tensor)
            or tuple(conditioned_source.shape) != tuple(source_foa_latent.shape)
            or not torch.equal(conditioned_source.float(), source_foa_latent.float())
        ):
            raise RuntimeError("AR and DiT clean source tensors diverged")
        conditioning = replace_editing_prompt_event_spans(
            dict(conditioning),
            metadata,
            predicted,
            event_mask,
            max_events=self.event_head.config.max_events,
        )
        conditioning["source_foa_latent"] = [source_foa_latent, None]
        rf_prediction = self.diffusion.model(
            noised_target,
            timesteps,
            **self.diffusion.get_conditioning_inputs(conditioning),
            cfg_dropout_prob=0.0,
            padding_mask=rf_padding_mask,
        )
        return {
            "ar_logits": ar_logits,
            "rf_prediction": rf_prediction,
            "source_caption_query": source_caption_query,
            "event_prediction": self.event_head(states, event_mask),
            "event_teacher": teacher,
            "event_mask": event_mask,
        }
