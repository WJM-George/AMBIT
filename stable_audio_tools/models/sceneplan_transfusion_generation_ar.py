"""Generation AR on the immutable P10-v11 ContinuousTransformer stack.

The acoustic DiT and the discrete ScenePlan decoder own one actual
``ContinuousTransformer`` object.  DiT calls retain the checkpoint-native
bidirectional attention and input/output projections.  AR calls bypass only
the acoustic projections and request causal *self*-attention; cross-attention
continues to see the complete frozen-Qwen request context.
"""

from __future__ import annotations
import os

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from stable_audio_tools.configuration import load_config
from stable_audio_tools.paths import repo_root
from stable_audio_tools.models.conditioners import QwenTextConditioner
from stable_audio_tools.models.dit import DiffusionTransformer
from stable_audio_tools.models.transformer import (
    ContinuousTransformer,
    apply_rotary_pos_emb,
)
from stable_audio_tools.models.utils import load_ckpt_state_dict


P10_V11_MODEL_CONFIG = (
    repo_root()
    / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit"
    / "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_resume_cosine_40k.json"
)
P10_V11_MODEL_CONFIG_SHA256 = (
    "3ebcd2b6b3c9a8b78b9160243f6509fb9eb11a32959b2eeddb86f48bb0b44827"
)
P10_V11_RESOLVED_CONFIG_SHA256 = (
    "cc3e0881b1b5a87ad4dcaa9d4e2f12080d2414630eff17a84942496924236c71"
)
P10_V11_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
P10_V11_CHECKPOINT_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)
P10_EMA_PREFIX = "diffusion_ema.ema_model.model."

GENERATION_AR_CONTRACT = "p10v11_frozen_shared_blocks_generation_ar_v1"
GENERATION_AR_VOCAB_SIZE = 4096
GENERATION_AR_HIDDEN_DIM = 1024
GENERATION_AR_DEPTH = 15
GENERATION_AR_HEADS = 16
GENERATION_AR_MAX_REQUEST_TOKENS = 512


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class P10GenerationARLoadReport:
    contract: str
    checkpoint: str
    checkpoint_sha256: str
    model_config: str
    model_config_sha256: str
    resolved_config_sha256: str
    ema_tensor_count: int
    ema_parameter_count: int
    transformer_depth: int
    transformer_hidden_dim: int
    transformer_heads: int
    prompt_ema_parameter_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_config": self.model_config,
            "model_config_sha256": self.model_config_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "ema_tensor_count": self.ema_tensor_count,
            "ema_parameter_count": self.ema_parameter_count,
            "transformer_depth": self.transformer_depth,
            "transformer_hidden_dim": self.transformer_hidden_dim,
            "transformer_heads": self.transformer_heads,
            "prompt_ema_parameter_names": list(
                self.prompt_ema_parameter_names
            ),
        }


@dataclass
class GenerationARDecodeCache:
    self_keys: list[Tensor]
    self_values: list[Tensor]
    cross_keys: list[Tensor]
    cross_values: list[Tensor]
    context_mask: Tensor
    position: int
    max_plan_tokens: int


class DiscreteScenePlanAdapter(nn.Module):
    """The only trainable component in first-stage Generation AR."""

    def __init__(self, *, vocab_size: int, hidden_dim: int, pad_id: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token_embedding = nn.Embedding(
            int(vocab_size), int(hidden_dim), padding_idx=int(pad_id)
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.plan_head = nn.Linear(int(hidden_dim), int(vocab_size), bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.plan_head.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[int(pad_id)].zero_()

    def embed(self, token_ids: Tensor) -> Tensor:
        return self.token_embedding(token_ids) * math.sqrt(self.hidden_dim)

    def logits(self, hidden_states: Tensor) -> Tensor:
        return self.plan_head(self.output_norm(hidden_states))


class ScenePlanTransfusionGenerationAR(nn.Module):
    """Discrete AR adapter sharing the canonical P10-v11 block object."""

    def __init__(
        self,
        *,
        p10_dit: DiffusionTransformer,
        prompt_conditioner: QwenTextConditioner,
        pad_id: int,
        vocab_size: int = GENERATION_AR_VOCAB_SIZE,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        transformer = getattr(p10_dit, "transformer", None)
        if not isinstance(transformer, ContinuousTransformer):
            raise TypeError("P10 Generation AR requires ContinuousTransformer")
        if (
            int(transformer.depth) != GENERATION_AR_DEPTH
            or int(transformer.dim) != GENERATION_AR_HIDDEN_DIM
            or len(transformer.layers) != GENERATION_AR_DEPTH
        ):
            raise ValueError("P10-v11 transformer depth/width contract changed")
        if any(
            int(layer.self_attn.num_heads) != GENERATION_AR_HEADS
            or not bool(layer.cross_attend)
            for layer in transformer.layers
        ):
            raise ValueError("P10-v11 head/cross-attention contract changed")
        if any(
            layer.global_cond_dim is not None
            or layer.conformer is not None
            or layer.sceneplan_moe is not None
            or layer.self_attn.qk_norm != "none"
            or layer.self_attn.differential
            or layer.self_attn.feat_scale
            or layer.cross_attn.qk_norm != "none"
            or layer.cross_attn.differential
            or layer.cross_attn.feat_scale
            for layer in transformer.layers
        ):
            raise ValueError("P10-v11 cached-decoding block contract changed")
        project_in = getattr(transformer, "project_in", None)
        project_out = getattr(transformer, "project_out", None)
        if (
            not isinstance(project_in, nn.Linear)
            or tuple(project_in.weight.shape) != (1024, 320)
            or not isinstance(project_out, nn.Linear)
            or tuple(project_out.weight.shape) != (64, 1024)
        ):
            raise ValueError("P10-v11 acoustic projection contract changed")

        self.p10_dit = p10_dit.eval().requires_grad_(False)
        self.prompt_conditioner = prompt_conditioner.eval().requires_grad_(False)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.pad_id = int(pad_id)
        self.vocab_size = int(vocab_size)
        self.ar_adapter = DiscreteScenePlanAdapter(
            vocab_size=self.vocab_size,
            hidden_dim=GENERATION_AR_HIDDEN_DIM,
            pad_id=self.pad_id,
        )

    @property
    def shared_transformer(self) -> ContinuousTransformer:
        """The same object used by ``self.p10_dit`` acoustic inference."""

        return self.p10_dit.transformer

    def train(self, mode: bool = True):
        super().train(mode)
        # ``nn.Module.train`` is recursive; restore immutable components.
        self.p10_dit.eval()
        self.prompt_conditioner.eval()
        return self

    @torch.no_grad()
    def encode_requests(
        self, requests: Sequence[str], *, device: torch.device | str
    ) -> tuple[Tensor, Tensor]:
        if not requests or not all(isinstance(text, str) and text for text in requests):
            raise ValueError("Generation AR requests must be non-empty strings")
        tokenizer = self.prompt_conditioner.tokenizer
        encoded = tokenizer(
            list(requests),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(
            encoded["attention_mask"], dtype=torch.bool
        )
        lengths = attention_mask.sum(dim=1)
        if bool((lengths > GENERATION_AR_MAX_REQUEST_TOKENS).any()):
            offenders = [
                (index, int(length))
                for index, length in enumerate(lengths.tolist())
                if int(length) > GENERATION_AR_MAX_REQUEST_TOKENS
            ]
            raise ValueError(
                "Generation request exceeds the frozen 512-token limit: "
                f"{offenders[:8]}"
            )
        zeros = torch.zeros_like(input_ids)
        tokenized = [
            {
                "input_ids": input_ids[index],
                "attention_mask": attention_mask[index],
                "event_source_ids": zeros[index],
                "speech_source_ids": zeros[index],
            }
            for index in range(len(requests))
        ]
        context_768, context_mask = self.prompt_conditioner(tokenized, device)
        context_1024 = self.p10_dit.to_cond_embed(context_768)
        return context_1024.detach(), context_mask.to(torch.bool).detach()

    def forward(
        self,
        plan_input_ids: Tensor,
        plan_attention_mask: Tensor,
        request_context: Tensor,
        request_attention_mask: Tensor,
    ) -> Tensor:
        if plan_input_ids.ndim != 2:
            raise ValueError("plan_input_ids must be [batch, sequence]")
        if tuple(plan_attention_mask.shape) != tuple(plan_input_ids.shape):
            raise ValueError("plan attention mask must align with token ids")
        if (
            request_context.ndim != 3
            or int(request_context.shape[0]) != int(plan_input_ids.shape[0])
            or int(request_context.shape[-1]) != GENERATION_AR_HIDDEN_DIM
            or tuple(request_attention_mask.shape)
            != tuple(request_context.shape[:2])
        ):
            raise ValueError("request context/mask shape does not match AR batch")
        if bool(
            (plan_input_ids < 0).any()
            or (plan_input_ids >= self.vocab_size).any()
        ):
            raise ValueError("plan token id is outside codec-v4 vocabulary")

        hidden = self.ar_adapter.embed(plan_input_ids)
        hidden = self.shared_transformer(
            hidden,
            context=request_context,
            context_mask=request_attention_mask.to(torch.bool),
            padding_mask=plan_attention_mask.to(torch.bool),
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=True,
            use_checkpointing=self.activation_checkpointing and self.training,
        )
        return self.ar_adapter.logits(hidden)

    @staticmethod
    def _split_heads(value: Tensor, heads: int) -> Tensor:
        batch, sequence, width = value.shape
        if width % int(heads):
            raise ValueError("attention width is not divisible by head count")
        return value.view(batch, sequence, int(heads), width // int(heads)).transpose(1, 2)

    @staticmethod
    def _merge_heads(value: Tensor) -> Tensor:
        batch, heads, sequence, width = value.shape
        return value.transpose(1, 2).reshape(batch, sequence, heads * width)

    @torch.no_grad()
    def prepare_decode_cache(
        self,
        request_context: Tensor,
        request_attention_mask: Tensor,
        *,
        max_plan_tokens: int = 1024,
    ) -> GenerationARDecodeCache:
        """Precompute cross K/V and allocate causal self K/V storage."""

        if (
            request_context.ndim != 3
            or int(request_context.shape[-1]) != GENERATION_AR_HIDDEN_DIM
            or tuple(request_attention_mask.shape)
            != tuple(request_context.shape[:2])
        ):
            raise ValueError("request context/mask cannot initialize decode cache")
        maximum = int(max_plan_tokens)
        if maximum <= 1:
            raise ValueError("decode cache requires at least two plan tokens")
        batch = int(request_context.shape[0])
        self_keys: list[Tensor] = []
        self_values: list[Tensor] = []
        cross_keys: list[Tensor] = []
        cross_values: list[Tensor] = []
        for layer in self.shared_transformer.layers:
            self_attention = layer.self_attn
            head_dim = int(self_attention.dim_heads)
            heads = int(self_attention.num_heads)
            shape = (batch, heads, maximum, head_dim)
            self_keys.append(request_context.new_zeros(shape))
            self_values.append(request_context.new_zeros(shape))

            cross_attention = layer.cross_attn
            cross_kv = cross_attention.to_kv(request_context)
            cross_key, cross_value = cross_kv.chunk(2, dim=-1)
            cross_keys.append(
                self._split_heads(cross_key, cross_attention.kv_heads)
            )
            cross_values.append(
                self._split_heads(cross_value, cross_attention.kv_heads)
            )
        return GenerationARDecodeCache(
            self_keys=self_keys,
            self_values=self_values,
            cross_keys=cross_keys,
            cross_values=cross_values,
            context_mask=request_attention_mask.to(torch.bool),
            position=0,
            max_plan_tokens=maximum,
        )

    @torch.no_grad()
    def decode_step(
        self, token_ids: Tensor, cache: GenerationARDecodeCache
    ) -> Tensor:
        """Return next-token logits while updating exact P10 block K/V state."""

        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.ndim != 2 or int(token_ids.shape[1]) != 1:
            raise ValueError("decode_step consumes exactly one token per batch row")
        if int(token_ids.shape[0]) != int(cache.context_mask.shape[0]):
            raise ValueError("decode token batch does not match cache batch")
        if not 0 <= int(cache.position) < int(cache.max_plan_tokens):
            raise RuntimeError("Generation AR decode cache is exhausted")

        hidden = self.ar_adapter.embed(token_ids)
        model_dtype = next(self.shared_transformer.parameters()).dtype
        hidden = hidden.to(model_dtype)
        position = int(cache.position)
        rotary_frequencies = None
        rotary = self.shared_transformer.rotary_pos_emb
        if rotary is not None:
            rotary_frequencies, _ = rotary(
                torch.tensor([position], device=hidden.device)
            )

        for layer_index, layer in enumerate(self.shared_transformer.layers):
            normalized = layer.pre_norm(hidden)
            query, key, value = layer.self_attn.to_qkv(normalized).chunk(3, dim=-1)
            query = self._split_heads(query, layer.self_attn.num_heads)
            key = self._split_heads(key, layer.self_attn.num_heads)
            value = self._split_heads(value, layer.self_attn.num_heads)
            if rotary_frequencies is not None:
                query_dtype = query.dtype
                key_dtype = key.dtype
                query = apply_rotary_pos_emb(
                    query.float(), rotary_frequencies.float()
                ).to(query_dtype)
                key = apply_rotary_pos_emb(
                    key.float(), rotary_frequencies.float()
                ).to(key_dtype)
            cache.self_keys[layer_index][:, :, position : position + 1].copy_(key)
            cache.self_values[layer_index][:, :, position : position + 1].copy_(value)
            attended = layer.self_attn.apply_attn(
                query,
                cache.self_keys[layer_index][:, :, : position + 1],
                cache.self_values[layer_index][:, :, : position + 1],
                causal=False,
            )
            attended = layer.self_attn.to_out(self._merge_heads(attended))
            hidden = hidden + layer.self_attn_scale(attended)

            cross_attention = layer.cross_attn
            cross_query = cross_attention.to_q(layer.cross_attend_norm(hidden))
            cross_query = self._split_heads(
                cross_query, cross_attention.num_heads
            )
            cross_attended = cross_attention.apply_attn(
                cross_query,
                cache.cross_keys[layer_index],
                cache.cross_values[layer_index],
                causal=False,
                padding_mask=cache.context_mask,
                mask_padding_logits=True,
            )
            cross_attended = cross_attention.to_out(
                self._merge_heads(cross_attended)
            )
            hidden = hidden + layer.cross_attn_scale(cross_attended)
            hidden = hidden + layer.ff_scale(layer.ff(layer.ff_norm(hidden)))

        cache.position += 1
        return self.ar_adapter.logits(hidden)[:, 0]

    @torch.no_grad()
    def generate_constrained(
        self,
        requests: Sequence[str],
        codec,
        *,
        device: torch.device | str,
        max_plan_tokens: int = 1024,
    ) -> list[list[int]]:
        """Greedy codec-v4 generation with cached P10-v11 blocks."""

        context, context_mask = self.encode_requests(requests, device=device)
        cache = self.prepare_decode_cache(
            context, context_mask, max_plan_tokens=max_plan_tokens
        )
        prefixes = [[int(codec.bos_id)] for _ in requests]
        finished = [False for _ in requests]
        current = torch.full(
            (len(requests),),
            int(codec.bos_id),
            device=context.device,
            dtype=torch.long,
        )
        for _ in range(int(max_plan_tokens) - 1):
            logits = self.decode_step(current, cache)
            next_values: list[int] = []
            for index, prefix in enumerate(prefixes):
                if finished[index]:
                    next_values.append(int(codec.eos_id))
                    continue
                allowed = sorted(int(value) for value in codec.allowed_next_ids(prefix))
                if not allowed:
                    raise RuntimeError("codec-v4 returned an empty legal next-token set")
                allowed_tensor = torch.tensor(
                    allowed, device=logits.device, dtype=torch.long
                )
                scores = logits[index].index_select(0, allowed_tensor)
                selected = int(allowed_tensor[int(scores.argmax().item())].item())
                prefix.append(selected)
                next_values.append(selected)
                if selected == int(codec.eos_id):
                    finished[index] = True
            if all(finished):
                return prefixes
            current = torch.tensor(
                next_values, device=logits.device, dtype=torch.long
            )
        incomplete = [index for index, value in enumerate(finished) if not value]
        raise RuntimeError(
            f"Generation AR did not emit EOS within {max_plan_tokens} tokens: "
            f"rows={incomplete[:8]}"
        )

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.ar_adapter.state_dict().items()
        }

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        self.ar_adapter.load_state_dict(state, strict=True)


def load_p10v11_generation_ar(
    *,
    pad_id: int,
    checkpoint_path: str | Path = P10_V11_CHECKPOINT,
    model_config_path: str | Path = P10_V11_MODEL_CONFIG,
    verify_sha256: bool = True,
    activation_checkpointing: bool = True,
) -> tuple[ScenePlanTransfusionGenerationAR, P10GenerationARLoadReport]:
    """Strictly restore the EMA DiT stack and EMA P10 prompt bridge."""

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    model_config_file = Path(model_config_path).expanduser().resolve(strict=True)
    config_sha256 = _sha256_file(model_config_file)
    if config_sha256 != P10_V11_MODEL_CONFIG_SHA256:
        raise RuntimeError(
            f"canonical P10-v11 model config changed: {config_sha256}"
        )
    config = load_config(model_config_file)
    resolved_sha256 = _canonical_sha256(config)
    if resolved_sha256 != P10_V11_RESOLVED_CONFIG_SHA256:
        raise RuntimeError(
            f"resolved P10-v11 model config changed: {resolved_sha256}"
        )
    checkpoint_sha256 = (
        _sha256_file(checkpoint) if verify_sha256 else P10_V11_CHECKPOINT_SHA256
    )
    if checkpoint_sha256 != P10_V11_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"canonical P10-v11 checkpoint changed: {checkpoint_sha256}"
        )

    diffusion = dict(config["model"]["diffusion"])
    if (
        diffusion.get("type") != "dit"
        or diffusion.get("diffusion_objective") != "rectified_flow"
    ):
        raise RuntimeError("canonical P10-v11 is no longer rectified-flow DiT")
    dit_config = dict(diffusion["config"])
    with torch.device("meta"):
        p10_dit = DiffusionTransformer(
            diffusion_objective=diffusion["diffusion_objective"],
            **dit_config,
        )

    checkpoint_state, metadata = load_ckpt_state_dict(
        str(checkpoint), return_metadata=True
    )
    ema_state = {
        key[len(P10_EMA_PREFIX) :]: value
        for key, value in checkpoint_state.items()
        if key.startswith(P10_EMA_PREFIX)
    }
    incompatible = p10_dit.load_state_dict(ema_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "P10-v11 EMA DiT restore mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )

    prompt_config = next(
        item["config"]
        for item in config["model"]["conditioning"]["configs"]
        if item["id"] == "prompt" and item["type"] == "qwen_text"
    )
    prompt_conditioner = QwenTextConditioner(
        output_dim=int(config["model"]["conditioning"]["cond_dim"]),
        **prompt_config,
    )
    saved_names = tuple(metadata.get("conditioner_ema_parameter_names") or ())
    prompt_names = tuple(
        name for name in saved_names if name.startswith("conditioners.prompt.")
    )
    prompt_parameters = dict(prompt_conditioner.named_parameters())
    loaded_prompt_names: list[str] = []
    with torch.no_grad():
        for index, saved_name in enumerate(saved_names):
            if not saved_name.startswith("conditioners.prompt."):
                continue
            local_name = saved_name[len("conditioners.prompt.") :]
            parameter = prompt_parameters.get(local_name)
            shadow = checkpoint_state.get(f"conditioner_ema.shadow_{index:05d}")
            if parameter is None or shadow is None or parameter.shape != shadow.shape:
                raise RuntimeError(
                    f"P10-v11 prompt EMA mapping changed at {saved_name!r}"
                )
            parameter.copy_(shadow.to(dtype=parameter.dtype))
            loaded_prompt_names.append(saved_name)
    if tuple(loaded_prompt_names) != prompt_names or len(prompt_names) != 4:
        raise RuntimeError(
            "P10-v11 prompt conditioner EMA parameter contract changed: "
            f"{loaded_prompt_names}"
        )

    model = ScenePlanTransfusionGenerationAR(
        p10_dit=p10_dit,
        prompt_conditioner=prompt_conditioner,
        pad_id=int(pad_id),
        activation_checkpointing=activation_checkpointing,
    )
    report = P10GenerationARLoadReport(
        contract=GENERATION_AR_CONTRACT,
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        model_config=str(model_config_file),
        model_config_sha256=config_sha256,
        resolved_config_sha256=resolved_sha256,
        ema_tensor_count=len(ema_state),
        ema_parameter_count=sum(int(value.numel()) for value in ema_state.values()),
        transformer_depth=GENERATION_AR_DEPTH,
        transformer_hidden_dim=GENERATION_AR_HIDDEN_DIM,
        transformer_heads=GENERATION_AR_HEADS,
        prompt_ema_parameter_names=prompt_names,
    )
    return model, report


__all__ = [
    "GENERATION_AR_CONTRACT",
    "GENERATION_AR_DEPTH",
    "GENERATION_AR_HEADS",
    "GENERATION_AR_HIDDEN_DIM",
    "GENERATION_AR_MAX_REQUEST_TOKENS",
    "GENERATION_AR_VOCAB_SIZE",
    "GenerationARDecodeCache",
    "P10GenerationARLoadReport",
    "P10_V11_CHECKPOINT",
    "P10_V11_CHECKPOINT_SHA256",
    "P10_V11_MODEL_CONFIG",
    "ScenePlanTransfusionGenerationAR",
    "load_p10v11_generation_ar",
]
