"""Canonical P11: Qwen planner, hybrid audio bridge, and atomic edit programs."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import math
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..data.model_sceneplan_codec import load_model_sceneplan_codec
from ..data.model_sceneplan_codec_v3 import (
    KIND_TOKENS,
    ROOM_TOKENS,
    SOURCE_SLOT_TOKENS,
)
from ..data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from ..data.sceneplan_edit_patch import (
    PATCH_MAX_TOKENS,
    PATCH_OUTPUT_CONTRACT,
    ScenePlanEditPatchCodec,
)
from ..data.sceneplan_p11_single_turn import (
    MAX_LATENT_FRAMES,
    P10_CANONICAL_CHECKPOINT,
    P10_CANONICAL_CHECKPOINT_SHA256,
    P10_CANONICAL_CHECKPOINT_STEP,
    P10_CANONICAL_EXECUTOR_FAMILY,
    P10_CANONICAL_MODEL_CONFIG,
    P10_CANONICAL_MODEL_CONFIG_SHA256,
    P10_MAX_LATENT_FRAMES,
    P10_SEMANTIC_CAPTION_COMPILER_VERSION,
    P10_SEMANTIC_CAPTION_CONTRACT,
    P10_SEMANTIC_CAPTION_SURFACE,
    P10_TRANSCRIPT_STATE_AUTHORITY,
    P11_EDITING_CONTRACT,
    P11_EDITING_INPUT_CONTRACT,
    P11_EXECUTION_CONTRACT,
    P11_GAIN_POLICY,
    P11_SUPPORTED_MOTION_TYPES,
    P11Task,
    SCENEPLAN_PLAN_MAX_TOKENS,
    SEMANTIC_CAPTION_MAX_TOKENS,
    ScenePlanExecutionBundle,
    ScenePlanResolver,
    finalize_sceneplan_for_p10,
    normalize_p11_task,
)


P11_DECODE_OUTPUT_CONTRACT = PATCH_OUTPUT_CONTRACT


class _LoRALinear(nn.Module):
    """Trainable low-rank residual around an unregistered frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        sct_stage: str = "shared",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("P11 LoRA rank must be positive")
        if sct_stage not in {"shared", "understanding", "generation"}:
            raise ValueError(f"unsupported P11 LoRA SCT stage {sct_stage!r}")
        base.eval().requires_grad_(False)
        # The pretrained matrix is deliberately not registered in P11. Every
        # rank reloads it from the pinned Qwen directory, keeping checkpoints
        # small enough for frequent optimizer/EMA saves.
        self.__dict__["base"] = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.dropout = nn.Dropout(float(dropout))
        self.scale = float(alpha) / float(rank)
        self.sct_stage = str(sct_stage)
        self.gradient_route = "trainable"

    def move_base(self, device: torch.device) -> None:
        self.base.to(device)

    def forward(self, value: Tensor) -> Tensor:
        frozen = self.base(value)
        dropped = self.dropout(value).float()
        if self.gradient_route == "trainable":
            residual = self.lora_b(self.lora_a(dropped))
        elif self.gradient_route == "input_only":
            # The generation tower consumes the language tower's features, but
            # its continuous loss must not update the language-only adapters.
            # Detaching only the weights keeps the Jacobian to ``value`` and
            # therefore preserves gradients to continuous slot inputs.
            residual = F.linear(
                F.linear(dropped, self.lora_a.weight.detach()),
                self.lora_b.weight.detach(),
            )
        else:
            raise RuntimeError(
                f"invalid P11 LoRA gradient route {self.gradient_route!r}"
            )
        return frozen + residual.to(dtype=frozen.dtype) * self.scale


class _TemporalConnectorBlock(nn.Module):
    def __init__(self, channels: int, *, expansion: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("temporal connector kernel must be a positive odd number")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, channels * expansion)
        self.out_proj = nn.Linear(channels * expansion, channels)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, value: Tensor) -> Tensor:
        residual = self.depthwise(value).transpose(1, 2)
        residual = self.norm(residual)
        residual = F.gelu(self.in_proj(residual), approximate="tanh")
        residual = self.out_proj(residual).transpose(1, 2)
        return value + self.scale * residual


class _TemporalLatentTokenizer(nn.Module):
    """Compress a P11-profile FOA latent into a short Qwen audio prefix."""

    def __init__(self, config: Mapping[str, Any], *, output_dim: int) -> None:
        super().__init__()
        channels = int(config.get("channels", 64))
        if channels != 64:
            raise ValueError("P11 FOA latent bridge must consume 64 channels")
        depth = int(config.get("connector_depth", 3))
        if not 1 <= depth <= 8:
            raise ValueError("P11 connector depth must be within [1,8]")
        self.blocks = nn.ModuleList(
            [
                _TemporalConnectorBlock(
                    channels,
                    expansion=int(config.get("expansion", 4)),
                    kernel_size=int(config.get("kernel_size", 7)),
                )
                for _ in range(depth)
            ]
        )
        self.stride = int(config.get("stride", 8))
        if not 1 <= self.stride <= 32:
            raise ValueError("P11 temporal tokenizer stride must be within [1,32]")
        self.projection = nn.Conv1d(
            channels,
            int(output_dim),
            kernel_size=self.stride,
            stride=self.stride,
        )
        self.norm = nn.LayerNorm(int(output_dim))
        maximum = math.ceil(MAX_LATENT_FRAMES / self.stride)
        self.position = nn.Parameter(torch.zeros(maximum, int(output_dim)))
        nn.init.normal_(self.position, std=0.01)

    def forward(self, value: Tensor) -> Tensor:
        source = torch.as_tensor(value)
        if source.ndim != 2 or int(source.shape[0]) != 64:
            raise ValueError("P11 temporal bridge input must be [64,T]")
        output = source.float().unsqueeze(0)
        for block in self.blocks:
            output = block(output)
        if output.shape[-1] < self.stride:
            output = F.pad(output, (0, self.stride - int(output.shape[-1])))
        output = self.projection(output).transpose(1, 2).squeeze(0)
        output = self.norm(output)
        return output + self.position[: output.shape[0]]


class _SemanticResamplerLayer(nn.Module):
    def __init__(self, dim: int, *, heads: int, expansion: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        update, _ = self.attention(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ff(self.ff_norm(queries))


class _SemanticResampler(nn.Module):
    """Resample global or window-level frozen audio features into query tokens."""

    def __init__(self, config: Mapping[str, Any], *, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(config.get("input_dim", 512))
        self.num_queries = int(config.get("num_queries", 8))
        if self.input_dim <= 0 or not 1 <= self.num_queries <= 32:
            raise ValueError("invalid P11 semantic resampler dimensions")
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(self.input_dim, output_dim)
        self.queries = nn.Parameter(torch.empty(self.num_queries, output_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList(
            [
                _SemanticResamplerLayer(
                    output_dim,
                    heads=int(config.get("heads", 8)),
                    expansion=int(config.get("expansion", 4)),
                )
                for _ in range(int(config.get("depth", 2)))
            ]
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, value: Tensor) -> Tensor:
        source = torch.as_tensor(value).float()
        if source.ndim == 1:
            source = source.unsqueeze(0)
        if source.ndim != 2 or int(source.shape[-1]) != self.input_dim:
            raise ValueError(
                "P11 semantic cache row must be [D] or [windows,D], got "
                f"{tuple(source.shape)}"
            )
        if not bool(torch.isfinite(source).all()):
            raise ValueError("P11 semantic cache row contains non-finite values")
        context = self.input_projection(self.input_norm(source)).unsqueeze(0)
        queries = self.queries.unsqueeze(0)
        for layer in self.layers:
            queries = layer(queries, context)
        return self.output_norm(queries.squeeze(0))


class ScenePlanP11Planner(nn.Module):
    """Qwen-backed G/U planner and deterministic edit-patch planner."""

    ROUTE_ID = "sceneplan_p11"

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.model_config = dict(model_config)
        if str(model_config.get("route_id")) != self.ROUTE_ID:
            raise ValueError(f"P11 requires route_id={self.ROUTE_ID!r}")
        self.sample_rate = int(model_config.get("sample_rate", 44_100))
        self.audio_channels = int(model_config.get("audio_channels", 4))
        self.sample_size = int(model_config.get("sample_size", 442_368))
        if self.sample_rate != 44_100 or self.audio_channels != 4:
            raise ValueError("P11 is aligned to P10's 44.1-kHz four-channel contract")

        model = dict(model_config.get("model") or {})
        if model.get("output_protocol") != PATCH_OUTPUT_CONTRACT:
            raise ValueError(f"P11 output_protocol must be {PATCH_OUTPUT_CONTRACT!r}")
        if model.get("editing_input_contract") != P11_EDITING_INPUT_CONTRACT:
            raise ValueError(
                f"P11 editing_input_contract must be {P11_EDITING_INPUT_CONTRACT!r}"
            )
        # The active route observes Editing from FOA.  A caller-provided plan
        # is only a fallible prior and may be absent; it is never the state to
        # which the decoded patch is authoritatively applied.
        self.audio_aware_editing = (
            str(model_config.get("model_type")) == "sceneplan_p11_audio_aware_v1"
        )
        self.requires_input_sceneplan = not self.audio_aware_editing

        text = dict(model.get("text") or {})
        if text.get("mode") != "qwen_sceneplan_planner":
            raise ValueError("P11 text.mode must be 'qwen_sceneplan_planner'")
        model_path = text.get("model_path")
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("P11 requires a pinned Qwen model_path")
        codec_path = text.get("plan_codec_path")
        if not isinstance(codec_path, str) or not codec_path:
            raise ValueError("P11 requires model.text.plan_codec_path")
        plan_codec = load_model_sceneplan_codec(codec_path)
        if not isinstance(plan_codec, ModelScenePlanCodecV4):
            raise ValueError("canonical P11 requires the frozen 648-frame v4 ScenePlan codec")
        self.plan_codec = plan_codec
        self.patch_codec = ScenePlanEditPatchCodec(plan_codec)
        self.plan_max_tokens = int(text.get("plan_max_tokens", SCENEPLAN_PLAN_MAX_TOKENS))
        self.patch_max_tokens = int(text.get("patch_max_tokens", PATCH_MAX_TOKENS))
        self.plan_text_field_max_tokens = int(text.get("plan_text_field_max_tokens", 192))
        self.max_prompt_tokens = int(text.get("max_length", SEMANTIC_CAPTION_MAX_TOKENS))
        self.sequence_length = int(text.get("sequence_length", 1024))
        self.dense_right_padding = text.get("dense_right_padding") is True
        self.activation_checkpointing = bool(
            text.get("activation_checkpointing", False)
        )
        self.discrete_decode_mode = str(
            text.get("discrete_decode_mode", "prefix_recompute")
        )
        if self.plan_max_tokens != SCENEPLAN_PLAN_MAX_TOKENS:
            raise ValueError("P11 full-plan token ceiling must remain 1024")
        if self.patch_max_tokens != PATCH_MAX_TOKENS:
            raise ValueError(f"P11 patch token ceiling must remain {PATCH_MAX_TOKENS}")
        if self.max_prompt_tokens != SEMANTIC_CAPTION_MAX_TOKENS:
            raise ValueError("P11 Qwen prompt ceiling must remain 512")
        if self.sequence_length != 1024:
            raise ValueError("canonical P11 sequence_length must remain 1024")
        if not self.dense_right_padding:
            raise ValueError("canonical P11 requires dense right-padding execution")
        if self.discrete_decode_mode not in {"cached", "prefix_recompute"}:
            raise ValueError(
                "P11 text.discrete_decode_mode must be cached or prefix_recompute"
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        causal_lm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        backbone = getattr(causal_lm, "model", None)
        if not isinstance(backbone, nn.Module):
            raise TypeError("Qwen CausalLM does not expose its text model at .model")
        backbone.eval().requires_grad_(False)
        hidden_dim = int(getattr(backbone.config, "hidden_size", 0))
        configured_hidden = int(text.get("hidden_size", hidden_dim))
        if hidden_dim <= 0 or configured_hidden != hidden_dim:
            raise ValueError(
                f"P11 Qwen hidden size mismatch: checkpoint={hidden_dim}, config={configured_hidden}"
            )
        del causal_lm
        self.hidden_dim = hidden_dim
        self.__dict__["qwen_backbone"] = backbone
        self._qwen_cpu_fallback_enabled = False

        lora = dict(text.get("lora") or {})
        top_layers = int(lora.get("top_layers", 8))
        layers = getattr(backbone, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not 1 <= top_layers <= len(layers):
            raise ValueError("P11 Qwen LoRA top_layers is invalid")
        sct = dict((model.get("transfusion_cot") or {}).get("sct") or {})
        self.sct_enabled = bool(sct.get("enabled", False))
        self.sct_contract = str(sct.get("contract") or "")
        self.sct_total_layers = len(layers)
        self.sct_split_layer = len(layers)
        self.sct_understanding_lora_top_layers = 0
        self.sct_generation_layers = 0
        if self.sct_enabled:
            if (
                self.sct_contract
                != "audiochat_style_understanding_then_execution_sct_v1"
            ):
                raise ValueError("P11-v4 SCT contract changed")
            self.sct_split_layer = int(sct.get("understanding_layers", -1))
            self.sct_generation_layers = int(sct.get("generation_layers", -1))
            self.sct_understanding_lora_top_layers = int(
                sct.get("understanding_lora_top_layers", -1)
            )
            if (
                self.sct_split_layer + self.sct_generation_layers
                != self.sct_total_layers
            ):
                raise ValueError(
                    "P11-v4 SCT understanding/generation layers do not cover Qwen"
                )
            if not (
                1
                <= self.sct_understanding_lora_top_layers
                <= self.sct_split_layer
            ):
                raise ValueError("P11-v4 SCT understanding LoRA depth is invalid")
            if self.sct_generation_layers != top_layers:
                raise ValueError(
                    "P11-v4 SCT generation depth must equal text.lora.top_layers"
                )
        target_names = set(
            lora.get("target_modules")
            or (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )
        adapters: list[_LoRALinear] = []
        if self.sct_enabled:
            understanding_first_layer = (
                self.sct_split_layer - self.sct_understanding_lora_top_layers
            )
            layer_stages = {
                **{
                    layer_index: "understanding"
                    for layer_index in range(
                        understanding_first_layer, self.sct_split_layer
                    )
                },
                **{
                    layer_index: "generation"
                    for layer_index in range(
                        self.sct_split_layer, self.sct_total_layers
                    )
                },
            }
        else:
            first_layer = len(layers) - top_layers
            layer_stages = {
                layer_index: "shared"
                for layer_index in range(first_layer, len(layers))
            }
        for layer_index, sct_stage in sorted(layer_stages.items()):
            layer = layers[layer_index]
            replacements: list[tuple[nn.Module, str, nn.Linear]] = []
            for module_name, module in layer.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                leaf = module_name.rsplit(".", 1)[-1]
                if leaf not in target_names:
                    continue
                parent = layer
                parts = module_name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                replacements.append((parent, parts[-1], module))
            for parent, name, base in replacements:
                adapter = _LoRALinear(
                    base,
                    rank=int(lora.get("rank", 16)),
                    alpha=float(lora.get("alpha", 32.0)),
                    dropout=float(lora.get("dropout", 0.05)),
                    sct_stage=sct_stage,
                )
                setattr(parent, name, adapter)
                adapters.append(adapter)
        if not adapters:
            raise RuntimeError("P11 LoRA target selection matched no Qwen linear layers")
        self.lora_adapters = nn.ModuleList(adapters)
        self.lora_first_layer = int(min(layer_stages))

        self.plan_embedding = nn.Embedding(self.plan_codec.vocab_size, hidden_dim)
        nn.init.normal_(self.plan_embedding.weight, std=0.02)
        self.output_bias = nn.Parameter(torch.zeros(self.plan_codec.vocab_size))
        self.task_embedding = nn.Embedding(len(P11Task), hidden_dim)
        self.input_plan_boundary = nn.Parameter(torch.empty(2, hidden_dim))
        self.audio_boundary = nn.Parameter(torch.empty(2, hidden_dim))
        self.output_start = nn.Parameter(torch.empty(hidden_dim))
        self.audio_type_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        for value in (
            self.task_embedding.weight,
            self.input_plan_boundary,
            self.audio_boundary,
            self.output_start,
            self.audio_type_embedding,
        ):
            nn.init.normal_(value, std=0.02)
        if bool(text.get("initialize_plan_embeddings_from_qwen", True)):
            self._initialize_plan_embeddings_from_qwen()

        bridge = dict(model.get("audio_bridge") or {})
        if bridge.get("type") != "hybrid_temporal_semantic":
            raise ValueError("P11 audio_bridge.type must be hybrid_temporal_semantic")
        self.temporal_tokenizer = _TemporalLatentTokenizer(
            dict(bridge.get("temporal") or {}), output_dim=hidden_dim
        )
        semantic = dict(bridge.get("semantic") or {})
        if semantic.get("type") != "frozen_feature_resampler":
            raise ValueError("P11 semantic bridge must use frozen_feature_resampler")
        if not isinstance(semantic.get("encoder_revision"), str):
            raise ValueError("P11 semantic bridge requires a pinned encoder revision")
        self.semantic_resampler = _SemanticResampler(semantic, output_dim=hidden_dim)
        self.require_semantic = bool(semantic.get("required", True))

        executor = dict(model.get("executor") or {})
        expected_executor = {
            "type": "external_p10_sceneplan_dit",
            "handoff_contract": P11_EXECUTION_CONTRACT,
            "editing_contract": P11_EDITING_CONTRACT,
            "localized_editing": False,
            "preserves_unedited_waveform": False,
            "capability_contract": "p10_sceneplan_44_capability_v1",
            "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
            "semantic_caption_compiler_version": (
                P10_SEMANTIC_CAPTION_COMPILER_VERSION
            ),
            "semantic_caption_surface": P10_SEMANTIC_CAPTION_SURFACE,
            "transcript_state_authority": P10_TRANSCRIPT_STATE_AUTHORITY,
            "canonical_executor_family": P10_CANONICAL_EXECUTOR_FAMILY,
            "canonical_model_config": P10_CANONICAL_MODEL_CONFIG,
            "canonical_model_config_sha256": P10_CANONICAL_MODEL_CONFIG_SHA256,
            "canonical_checkpoint_step": P10_CANONICAL_CHECKPOINT_STEP,
            "canonical_checkpoint": P10_CANONICAL_CHECKPOINT,
            "canonical_checkpoint_sha256": P10_CANONICAL_CHECKPOINT_SHA256,
            "p10_max_latent_frames": P10_MAX_LATENT_FRAMES,
            "planner_max_latent_frames": MAX_LATENT_FRAMES,
            "planner_motion_types": list(P11_SUPPORTED_MOTION_TYPES),
            "gain_policy": P11_GAIN_POLICY,
            "word_level_timing_supported": False,
        }
        for key, expected in expected_executor.items():
            if executor.get(key) != expected:
                raise ValueError(
                    f"P11 executor.{key}={executor.get(key)!r}, expected {expected!r}"
                )
        self.executor_contract = executor
        # Codec-v4 retains the structurally valid keyframed branch.  The live
        # P11 decoder must not expose it:
        # the current P10/P11 training intersection contains static and linear
        # trajectories only.
        self.blocked_plan_token_ids = {
            self.plan_codec._tid("<motion_keyframed>")
        }

        self._pretransform_cfg = model.get("pretransform")
        self.pretransform = None
        self.downsampling_ratio = 1024
        if self._pretransform_cfg is not None:
            self.downsampling_ratio = int(
                self._pretransform_cfg["config"]["downsampling_ratio"]
            )
        if self.downsampling_ratio != 1024:
            raise ValueError("P11 must reuse P10's 1024-sample VAE hop")
        self.default_plan_duration_sec = float(
            model.get("default_plan_duration_sec", 15.0)
        )
        if self.default_plan_duration_sec != 15.0:
            raise ValueError("canonical P11 default duration must remain 15 seconds")

        plan_mask = torch.zeros(self.plan_codec.vocab_size, dtype=torch.bool)
        plan_mask[: int(self.plan_codec.details["used_vocab_size"])] = True
        patch_mask = torch.zeros_like(plan_mask)
        patch_mask[list(self.patch_codec.used_token_ids)] = True
        patch_mask[list(self.plan_codec.frame_ids)] = True
        patch_mask[
            [self.plan_codec._tid(token) for token in ROOM_TOKENS.values()]
        ] = True
        patch_mask[
            [self.plan_codec._tid(token) for token in SOURCE_SLOT_TOKENS]
        ] = True
        self.register_buffer("plan_vocab_mask", plan_mask, persistent=False)
        self.register_buffer("patch_vocab_mask", patch_mask, persistent=False)

        thought = dict(model.get("scene_thought") or {})
        if thought:
            raise ValueError(
                "the core40 SceneThought route is retired; use model.transfusion_cot"
            )

    def _initialize_plan_embeddings_from_qwen(self) -> None:
        labels = ["unused scene plan token"] * self.plan_codec.vocab_size
        for token, token_id in self.plan_codec.token_to_id.items():
            labels[int(token_id)] = token.replace("<", " ").replace(">", " ").replace("_", " ")
        for piece_id in range(self.plan_codec.text_vocab_size):
            token_id = self.plan_codec.text_offset + piece_id
            piece = self.plan_codec.text_processor.decode([piece_id]).strip()
            labels[token_id] = piece or self.plan_codec.text_processor.id_to_piece(piece_id)
        for token, token_id in self.patch_codec.token_to_id.items():
            labels[int(token_id)] = token.replace("<", " ").replace(">", " ").replace("_", " ")
        embedding = self.qwen_backbone.embed_tokens
        with torch.no_grad():
            for start in range(0, len(labels), 256):
                encoded = self.tokenizer(
                    labels[start : start + 256],
                    padding=True,
                    truncation=True,
                    max_length=16,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                ids = encoded["input_ids"]
                mask = encoded["attention_mask"].to(torch.bool)
                vectors = embedding(ids)
                pooled = (
                    vectors * mask.unsqueeze(-1).to(vectors.dtype)
                ).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                self.plan_embedding.weight[start : start + len(pooled)].copy_(
                    pooled.float()
                )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _ensure_qwen_device(self, device: torch.device) -> torch.dtype:
        if device.type == "cpu" and not self._qwen_cpu_fallback_enabled:
            self._enable_qwen_cpu_fallback()
        embedding = self.qwen_backbone.embed_tokens
        if embedding.weight.device != device:
            self.qwen_backbone.to(device)
        for adapter in self.lora_adapters:
            adapter.move_base(device)
            adapter.train(self.training)
        self.qwen_backbone.train(False)
        # Restore adapter train/eval state after backbone.eval() traverses them.
        for adapter in self.lora_adapters:
            adapter.train(self.training)
        return embedding.weight.dtype

    def configure_qwen_runtime_kernels(self, mode: str) -> dict[str, Any]:
        """Select and report the Qwen3.5 GatedDeltaNet inference kernels.

        The optional FLA kernels are useful for throughput but are not a safe
        basis for scientific A/B reports because their reductions can drift
        across fresh CUDA processes.  ``torch_reference`` keeps the same
        checkpoint and architecture while selecting Transformers' shipped
        torch implementations and eager full attention.
        """

        if mode not in {
            "fast",
            "fast_pinned_warps2",
            "fast_fixed_bv32_w2_s2",
            "torch_reference",
        }:
            raise ValueError(f"unsupported Qwen runtime kernel mode: {mode}")
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35

        fast_pin = None
        if mode in {"fast_pinned_warps2", "fast_fixed_bv32_w2_s2"}:
            from fla.ops.common import chunk_delta_h
            from fla.ops.utils.cache import FLA_CACHE_MODE

            kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
            autotuner = kernel
            while getattr(autotuner, "configs", None) is None:
                autotuner = getattr(autotuner, "fn", None)
                if autotuner is None:
                    raise RuntimeError(
                        "could not locate FLA chunk GatedDeltaRule autotuner"
                    )
            original_configs = list(autotuner.configs)
            if mode == "fast_pinned_warps2":
                pinned_configs = [
                    config
                    for config in original_configs
                    if int(config.num_warps) == 2
                ]
                selection_policy = "all_num_warps_2_autotuned"
            else:
                pinned_configs = [
                    config
                    for config in original_configs
                    if int(config.kwargs.get("BV", -1)) == 32
                    and int(config.num_warps) == 2
                    and int(config.num_stages) == 2
                ]
                selection_policy = "single_bv32_num_warps2_num_stages2"
            if not pinned_configs:
                raise RuntimeError(
                    f"FLA GatedDeltaRule lacks required {selection_policy} config"
                )
            if mode == "fast_fixed_bv32_w2_s2" and len(pinned_configs) != 1:
                raise RuntimeError(
                    "deterministic FLA pin requires exactly one kernel config"
                )
            autotuner.configs = pinned_configs
            cache = getattr(autotuner, "cache", None)
            if cache is not None:
                cache.clear()
            fast_pin = {
                "kernel": (
                    "fla.ops.common.chunk_delta_h."
                    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
                ),
                "fla_cache_mode": FLA_CACHE_MODE.value,
                "original_config_count": len(original_configs),
                "pinned_config_count": len(pinned_configs),
                "selection_policy": selection_policy,
                "allowed_num_warps": [2],
                "allowed_configs": [
                    {
                        "kwargs": dict(config.kwargs),
                        "num_warps": int(config.num_warps),
                        "num_stages": int(config.num_stages),
                    }
                    for config in pinned_configs
                ],
            }

        layers = []
        for index, module in enumerate(self.qwen_backbone.modules()):
            if not isinstance(module, qwen35.Qwen3_5GatedDeltaNet):
                continue
            if mode == "torch_reference":
                module.causal_conv1d_fn = None
                module.causal_conv1d_update = qwen35.torch_causal_conv1d_update
                module.chunk_gated_delta_rule = qwen35.torch_chunk_gated_delta_rule
                module.recurrent_gated_delta_rule = (
                    qwen35.torch_recurrent_gated_delta_rule
                )
            layers.append(index)
        if not layers:
            raise RuntimeError("P11 Qwen backbone has no GatedDeltaNet layers")
        if mode == "torch_reference":
            self.qwen_backbone.config._attn_implementation = "eager"
        return {
            "contract": "p11_qwen35_runtime_kernel_selection_v1",
            "mode": mode,
            "fast_kernel_pin": fast_pin,
            "gated_delta_layers": len(layers),
            "full_attention_implementation": str(
                getattr(self.qwen_backbone.config, "_attn_implementation", "unknown")
            ),
            "causal_conv1d": (
                "transformers_torch_reference"
                if mode == "torch_reference"
                else "optional_fast_runtime"
            ),
            "chunk_gated_delta_rule": (
                "transformers_torch_reference"
                if mode == "torch_reference"
                else "optional_fast_runtime"
            ),
            "recurrent_gated_delta_rule": (
                "transformers_torch_reference"
                if mode == "torch_reference"
                else "optional_fast_runtime"
            ),
        }

    def _enable_qwen_cpu_fallback(self) -> None:
        """Replace optional CUDA-only Qwen kernels for the mandatory CPU smoke."""

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

        self.configure_qwen_runtime_kernels("torch_reference")

        for layer in self.qwen_backbone.layers:
            linear_attention = getattr(layer, "linear_attn", None)
            if linear_attention is None:
                continue
            if not isinstance(linear_attention.norm, Qwen3_5RMSNormGated):
                replacement = Qwen3_5RMSNormGated(
                    linear_attention.head_v_dim,
                    eps=linear_attention.layer_norm_epsilon,
                )
                source_weight = getattr(linear_attention.norm, "weight", None)
                if source_weight is not None:
                    replacement.weight.data.copy_(source_weight.detach().float().cpu())
                replacement.eval().requires_grad_(False)
                linear_attention.norm = replacement
        self._qwen_cpu_fallback_enabled = True

    @staticmethod
    def _ids(value: Mapping[str, Tensor] | Tensor) -> Tensor:
        if isinstance(value, Mapping):
            value = value.get("input_ids")
        result = torch.as_tensor(value, dtype=torch.long).flatten()
        if result.numel() < 2:
            raise ValueError("P11 output/input token sequence is too short")
        return result

    @staticmethod
    def _crop_input_latent(
        value: Tensor,
        valid_mask: Tensor | None,
        *,
        device: torch.device,
    ) -> Tensor:
        source = torch.as_tensor(value)
        if source.ndim != 2 or int(source.shape[0]) != 64:
            raise ValueError("P11 input FOA latent must be [64,T]")
        if valid_mask is None:
            return source.to(device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid_mask, device="cpu", dtype=torch.bool).flatten()
        if mask.numel() != source.shape[1]:
            raise ValueError("P11 input latent and valid mask do not align")
        frames = int(mask.sum())
        if not 1 <= frames <= int(source.shape[1]):
            raise ValueError("P11 input mask retains no valid frames")
        if not torch.equal(mask, torch.arange(mask.numel()) < frames):
            raise ValueError("P11 input mask must be one contiguous prefix")
        return source[:, :frames].to(device=device, dtype=torch.float32)

    def _prompt_ids(self, value: str | Mapping[str, Any]) -> Tensor:
        if isinstance(value, str):
            from ..data.sceneplan_p11_dataset import tokenize_p11_task_prompt

            value = tokenize_p11_task_prompt(value, self.tokenizer)
        ids = torch.as_tensor(value["input_ids"], dtype=torch.long).flatten()
        mask = torch.as_tensor(value["attention_mask"], dtype=torch.bool).flatten()
        if ids.shape != mask.shape or ids.numel() != self.max_prompt_tokens:
            raise ValueError("P11 prompt ids/mask do not match the 512-token contract")
        length = int(mask.sum())
        prefix = torch.arange(mask.numel(), device=mask.device) < length
        if length <= 0 or not torch.equal(mask, prefix):
            raise ValueError("P11 prompt mask must be one non-empty prefix")
        return ids[:length]

    def _context_embeddings(
        self,
        *,
        task: P11Task,
        prompt: str | Mapping[str, Any],
        input_foa: Tensor | None,
        input_valid_mask: Tensor | None,
        input_semantic: Tensor | None,
        input_plan: Mapping[str, Tensor] | Tensor | None,
        device: torch.device,
        qwen_dtype: torch.dtype,
    ) -> tuple[Tensor, dict[str, int]]:
        expected_audio = (
            task in {P11Task.UNDERSTANDING, P11Task.EDITING}
            if self.audio_aware_editing
            else task is P11Task.UNDERSTANDING
        )
        if (input_foa is not None) != expected_audio:
            raise ValueError(f"P11 {task.value} audio-input truth table changed")
        if task is not P11Task.EDITING and input_plan is not None:
            raise ValueError(f"P11 {task.value} cannot carry a current-plan prior")
        if (
            task is P11Task.EDITING
            and self.requires_input_sceneplan
            and input_plan is None
        ):
            raise ValueError("legacy P11 Editing requires the current ScenePlan")
        if input_semantic is not None and not expected_audio:
            raise ValueError("only audio-observation tasks may carry semantic features")
        if expected_audio and self.require_semantic and input_semantic is None:
            raise ValueError("canonical P11 audio observation requires semantic features")

        task_index = list(P11Task).index(task)
        parts = [self.task_embedding.weight[task_index : task_index + 1]]
        prompt_ids = self._prompt_ids(prompt).to(device)
        with torch.no_grad():
            prompt_embeddings = self.qwen_backbone.embed_tokens(prompt_ids)
        parts.append(prompt_embeddings.float())
        metrics = {
            "prompt_tokens": int(prompt_ids.numel()),
            "input_plan_tokens": 0,
            "input_audio_tokens": 0,
            "input_semantic_tokens": 0,
        }
        if input_plan is not None:
            plan_ids = self._ids(input_plan).to(device)
            if int(plan_ids.min()) < 0 or int(plan_ids.max()) >= self.plan_codec.vocab_size:
                raise ValueError("P11 current-plan token is outside the shared vocabulary")
            parts.extend(
                [
                    self.input_plan_boundary[0:1],
                    self.plan_embedding(plan_ids),
                    self.input_plan_boundary[1:2],
                ]
            )
            metrics["input_plan_tokens"] = int(plan_ids.numel())
        if input_foa is not None:
            latent = self._crop_input_latent(input_foa, input_valid_mask, device=device)
            temporal = self.temporal_tokenizer(latent) + self.audio_type_embedding[0]
            semantic = None
            if input_semantic is not None:
                semantic = self.semantic_resampler(
                    torch.as_tensor(input_semantic, device=device)
                ) + self.audio_type_embedding[1]
            audio_parts = [self.audio_boundary[0:1], temporal]
            if semantic is not None:
                audio_parts.append(semantic)
                metrics["input_semantic_tokens"] = int(semantic.shape[0])
            audio_parts.append(self.audio_boundary[1:2])
            parts.extend(audio_parts)
            metrics["input_audio_tokens"] = int(temporal.shape[0])
        return torch.cat([part.to(dtype=qwen_dtype) for part in parts], dim=0), metrics

    def _run_backbone(self, embeddings: Tensor, attention_mask: Tensor, **kwargs):
        if attention_mask.ndim != 2 or attention_mask.shape[0] != embeddings.shape[0]:
            raise ValueError("P11 attention-mask batch does not align with embeddings")
        # Every row is a valid causal prefix followed only by right padding.
        # Outputs and gradients inside that prefix cannot depend on later padded
        # positions, so feeding the mask would only force Qwen3.5/FLA onto its
        # substantially slower ragged Triton path. The dense bucket path is
        # mathematically identical for every hidden state consumed below.
        if (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and kwargs.get("use_cache") is False
        ):
            # Checkpoint the whole selected SCT tower. The route context above
            # mutates Qwen's visible layer count and LoRA gradient routing;
            # capture and restore both inside the closure so backward
            # recomputation is identical after the outer context has exited.
            from torch.utils.checkpoint import checkpoint

            selected_layers = int(self.qwen_backbone.config.num_hidden_layers)
            selected_routes = tuple(
                adapter.gradient_route for adapter in self.lora_adapters
            )

            def checkpointed(value: Tensor) -> Tensor:
                previous_layers = int(
                    self.qwen_backbone.config.num_hidden_layers
                )
                previous_routes = tuple(
                    adapter.gradient_route for adapter in self.lora_adapters
                )
                try:
                    self.qwen_backbone.config.num_hidden_layers = selected_layers
                    for adapter, route in zip(
                        self.lora_adapters, selected_routes
                    ):
                        adapter.gradient_route = route
                    return self.qwen_backbone(
                        inputs_embeds=value,
                        return_dict=True,
                        **kwargs,
                    ).last_hidden_state
                finally:
                    self.qwen_backbone.config.num_hidden_layers = previous_layers
                    for adapter, route in zip(
                        self.lora_adapters, previous_routes
                    ):
                        adapter.gradient_route = route

            hidden = checkpoint(checkpointed, embeddings, use_reentrant=False)
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)
        if (
            kwargs.get("use_cache") is True
            and kwargs.get("past_key_values") is None
            and self.sct_enabled
        ):
            # Qwen3.5's hybrid DynamicCache derives both its list lengths and
            # ``last_linear_layer`` from the config.  SCT temporarily shortens
            # ``num_hidden_layers`` but intentionally leaves the checkpoint's
            # full ``layer_types`` list intact; allowing Transformers to build
            # the cache from that inconsistent view makes
            # ``has_previous_state`` index beyond the shortened state lists.
            # Build an explicit cache from a private, tower-local config while
            # leaving the live backbone config untouched.
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DynamicCache,
            )

            selected_layers = int(self.qwen_backbone.config.num_hidden_layers)
            layer_types = list(self.qwen_backbone.config.layer_types)
            if not 1 <= selected_layers <= len(layer_types):
                raise RuntimeError("P11 SCT selected an invalid Qwen layer count")
            cache_config = copy.copy(self.qwen_backbone.config)
            cache_config.num_hidden_layers = selected_layers
            cache_config.layer_types = layer_types[:selected_layers]
            kwargs["past_key_values"] = Qwen3_5DynamicCache(config=cache_config)
        return self.qwen_backbone(
            inputs_embeds=embeddings,
            return_dict=True,
            **kwargs,
        )

    @contextmanager
    def _sct_backbone_route(self, route: str):
        """Select the true SCT language or continuous execution tower.

        The language route stops after the first ``U`` Qwen layers.  The
        execution route uses all layers, but evaluates the understanding LoRA
        weights as constants so continuous losses cannot update them.  This is
        the local P11 analogue of AudioChat's understanding-then-generation
        Self-Cascaded Transformer, while preserving input gradients needed by
        continuous thought slots.
        """

        if not self.sct_enabled:
            raise RuntimeError("SCT routing is available only for P11-v4")
        if route not in {"understanding", "generation"}:
            raise ValueError(f"unsupported P11-v4 SCT route {route!r}")
        config = self.qwen_backbone.config
        previous_layers = int(config.num_hidden_layers)
        previous_routes = [adapter.gradient_route for adapter in self.lora_adapters]
        try:
            config.num_hidden_layers = int(
                self.sct_split_layer
                if route == "understanding"
                else self.sct_total_layers
            )
            for adapter in self.lora_adapters:
                adapter.gradient_route = (
                    "input_only"
                    if route == "generation"
                    and adapter.sct_stage == "understanding"
                    else "trainable"
                )
            yield
        finally:
            config.num_hidden_layers = previous_layers
            for adapter, previous in zip(self.lora_adapters, previous_routes):
                adapter.gradient_route = previous

    def _run_sct_understanding_backbone(
        self, embeddings: Tensor, attention_mask: Tensor, **kwargs
    ):
        with self._sct_backbone_route("understanding"):
            return self._run_backbone(embeddings, attention_mask, **kwargs)

    def _run_sct_generation_backbone(
        self, embeddings: Tensor, attention_mask: Tensor, **kwargs
    ):
        with self._sct_backbone_route("generation"):
            return self._run_backbone(embeddings, attention_mask, **kwargs)

    def _trainable_graph_anchor(self, reference: Tensor) -> Tensor:
        """Keep the DDP graph identical when a local batch lacks one task."""

        anchor = reference.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad:
                anchor = anchor + parameter.reshape(-1)[0].to(reference.dtype) * 0.0
        return anchor

    def forward_planner(
        self,
        prompts: Sequence[Mapping[str, Any]],
        target_outputs: Sequence[Mapping[str, Tensor] | Tensor],
        *,
        tasks: Sequence[P11Task | str],
        input_foa: Sequence[Optional[Tensor]],
        input_valid_masks: Sequence[Optional[Tensor]],
        input_semantic: Sequence[Optional[Tensor]],
        input_plans: Sequence[Optional[Mapping[str, Tensor] | Tensor]],
        loss_group_weights: Optional[Mapping[int, float]] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not prompts or not (
            len(prompts)
            == len(target_outputs)
            == len(tasks)
            == len(input_foa)
            == len(input_valid_masks)
            == len(input_semantic)
            == len(input_plans)
        ):
            raise ValueError("P11 Planner batch fields must be non-empty and aligned")
        normalized_tasks = [normalize_p11_task(task) for task in tasks]
        device = self.plan_embedding.weight.device
        qwen_dtype = self._ensure_qwen_device(device)
        contexts: list[Tensor] = []
        labels: list[Tensor] = []
        groups: list[Tensor] = []
        context_metrics: list[dict[str, int]] = []
        for prompt, target, task, audio, mask, semantic, input_plan in zip(
            prompts,
            target_outputs,
            normalized_tasks,
            input_foa,
            input_valid_masks,
            input_semantic,
            input_plans,
        ):
            target_ids = self._ids(target).to(device)
            expected_codec = self.patch_codec if task is P11Task.EDITING else self.plan_codec
            if int(target_ids[0]) != expected_codec.bos_id or int(target_ids[-1]) != expected_codec.eos_id:
                raise ValueError(f"P11 {task.value} target uses the wrong output grammar")
            context, one_metrics = self._context_embeddings(
                task=task,
                prompt=prompt,
                input_foa=audio,
                input_valid_mask=mask,
                input_semantic=semantic,
                input_plan=input_plan,
                device=device,
                qwen_dtype=qwen_dtype,
            )
            contexts.append(context)
            labels.append(target_ids)
            context_metrics.append(one_metrics)
            if isinstance(target, Mapping) and target.get("loss_group_ids") is not None:
                one_groups = torch.as_tensor(
                    target["loss_group_ids"], device=device, dtype=torch.long
                ).flatten()
                if one_groups.shape != target_ids.shape:
                    raise ValueError("P11 loss groups do not align with target tokens")
            else:
                one_groups = torch.ones_like(target_ids)
            groups.append(one_groups)

        rows: list[Tensor] = []
        output_starts: list[int] = []
        for context, target_ids in zip(contexts, labels):
            output_starts.append(int(context.shape[0]))
            teacher = self.plan_embedding(target_ids[:-1]).to(dtype=qwen_dtype)
            rows.append(
                torch.cat(
                    [context, self.output_start.view(1, -1).to(qwen_dtype), teacher],
                    dim=0,
                )
            )

        padded = pad_sequence(rows, batch_first=True)
        if padded.shape[1] > self.sequence_length:
            raise ValueError(
                f"P11 sequence needs {padded.shape[1]} tokens > fixed "
                f"{self.sequence_length}-token execution contract"
            )
        if padded.shape[1] < self.sequence_length:
            padded = F.pad(padded, (0, 0, 0, self.sequence_length - padded.shape[1]))
        row_lengths = [int(row.shape[0]) for row in rows]
        lengths = torch.tensor(row_lengths, device=device)
        attention = torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]
        output = self._run_backbone(padded, attention, use_cache=False)
        selected: list[Tensor] = []
        for batch_index, (start, target_ids, task) in enumerate(
            zip(output_starts, labels, normalized_tasks)
        ):
            hidden = output.last_hidden_state[
                batch_index, start : start + target_ids.numel()
            ].float()
            logits = F.linear(hidden, self.plan_embedding.weight.float(), self.output_bias)
            mask = self.patch_vocab_mask if task is P11Task.EDITING else self.plan_vocab_mask
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            selected.append(logits)
        padded_logits = pad_sequence(selected, batch_first=True)
        ignore_index = -100
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=ignore_index)
        padded_groups = pad_sequence(groups, batch_first=True, padding_value=0)
        token_loss = F.cross_entropy(
            padded_logits.transpose(1, 2),
            padded_labels,
            ignore_index=ignore_index,
            reduction="none",
        )
        valid = padded_labels.ne(ignore_index)
        weights = valid.to(token_loss)
        if loss_group_weights:
            weights.zero_()
            for group_id, weight in loss_group_weights.items():
                weights = torch.where(
                    padded_groups.eq(int(group_id)),
                    token_loss.new_tensor(float(weight)),
                    weights,
                )
            weights = weights * valid
        weighted = token_loss * weights
        row_weight = weights.sum(dim=1)
        per_row = weighted.sum(dim=1) / row_weight.clamp_min(1.0)
        # G/U full plans are hundreds of tokens while an E patch is only
        # 3--6 tokens. Token-global averaging would reduce Editing to roughly
        # one percent of the objective despite the balanced G/U/E manifest.
        # Normalize within each row first so each task receives its promised
        # one-third share of every interleaved batch.
        planner_ce = per_row.mean()
        total = planner_ce + self._trainable_graph_anchor(planner_ce)
        return total, {
            "planner_ce": planner_ce.detach(),
            "planner_ce_per_row": per_row.detach(),
            "total_loss": total.detach(),
            "planner_tokens": valid.sum().detach(),
            "planner_tokens_per_row": valid.sum(dim=1).detach(),
            "sequence_tokens": total.new_tensor(sum(row_lengths)),
            "sequence_padding_tokens": total.new_tensor(
                padded.shape[0] * padded.shape[1] - sum(row_lengths)
            ),
            "prompt_tokens": total.new_tensor(sum(v["prompt_tokens"] for v in context_metrics)),
            "input_audio_tokens": total.new_tensor(sum(v["input_audio_tokens"] for v in context_metrics)),
            "input_semantic_tokens": total.new_tensor(sum(v["input_semantic_tokens"] for v in context_metrics)),
            "input_plan_tokens": total.new_tensor(sum(v["input_plan_tokens"] for v in context_metrics)),
        }

    @torch.no_grad()
    def decode_output_tokens(
        self,
        prompt: str | Mapping[str, Any],
        *,
        task: P11Task | str,
        input_foa: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
        input_semantic: Tensor | None = None,
        input_sceneplan: Mapping[str, Any] | None = None,
        duration_sec: float | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        constrained: bool = True,
        sampling_seed: int | None = None,
        discrete_decode_mode: str | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        task = normalize_p11_task(task)
        if (
            task is P11Task.EDITING
            and self.requires_input_sceneplan
            and input_sceneplan is None
        ):
            raise ValueError("P11 Editing requires the current ScenePlan")
        discrete_decode_mode = (
            self.discrete_decode_mode
            if discrete_decode_mode is None
            else str(discrete_decode_mode)
        )
        if discrete_decode_mode not in {"cached", "prefix_recompute"}:
            raise ValueError(
                "P11 discrete_decode_mode must be 'cached' or 'prefix_recompute'"
            )
        input_plan = (
            None
            if input_sceneplan is None
            else self.plan_codec.encode(input_sceneplan, max_tokens=self.plan_max_tokens)
        )
        codec = self.patch_codec if task is P11Task.EDITING else self.plan_codec
        ceiling = self.patch_max_tokens if task is P11Task.EDITING else self.plan_max_tokens
        max_tokens = ceiling if max_tokens is None else int(max_tokens)
        if not 1 <= max_tokens <= ceiling:
            raise ValueError(f"P11 {task.value} max_tokens must be within [1,{ceiling}]")
        device = self.plan_embedding.weight.device
        sampling_generator = None
        if sampling_seed is not None:
            sampling_seed = int(sampling_seed)
            if not 0 <= sampling_seed < 2**63:
                raise ValueError("P11 sampling_seed must be within [0,2**63)")
            sampling_generator = torch.Generator(device=device)
            sampling_generator.manual_seed(sampling_seed)
        qwen_dtype = self._ensure_qwen_device(device)
        context, context_metrics = self._context_embeddings(
            task=task,
            prompt=prompt,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_plan=input_plan,
            device=device,
            qwen_dtype=qwen_dtype,
        )
        embeddings = torch.cat(
            [context, self.output_start.view(1, -1).to(qwen_dtype)], dim=0
        ).unsqueeze(0)
        attention = torch.ones((1, embeddings.shape[1]), device=device, dtype=torch.bool)
        cache = None
        hidden = None
        if discrete_decode_mode == "cached":
            output = self._run_backbone(embeddings, attention, use_cache=True)
            cache = output.past_key_values
            hidden = output.last_hidden_state[0, -1].float()
        generated: list[int] = []
        interventions = 0
        text_safety_interventions = 0
        canonical_text_forcing_interventions = 0
        canonical_text_forcing_token_delta = 0
        source_kind_scores: list[dict[str, Any]] = []
        kind_token_ids = {
            kind: self.plan_codec._tid(token)
            for kind, token in KIND_TOKENS.items()
        }
        source_slot_by_token = {
            self.plan_codec._tid(token): f"source_{slot}"
            for slot, token in enumerate(SOURCE_SLOT_TOKENS)
        }
        kind_marker_id = self.plan_codec._tid("<kind>")
        valid_mask = self.patch_vocab_mask if task is P11Task.EDITING else self.plan_vocab_mask
        valid_ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
        valid_set = set(int(value) for value in valid_ids.tolist())
        text_begin = self.plan_codec.token_to_id.get("<text_begin>")
        text_end = self.plan_codec.token_to_id.get("<text_end>")
        # User-facing durations are decimal seconds, while codec v4 emits one
        # atomic latent-frame id. Snap the request to the nearest executable
        # frame before constraining the grammar. Passing a rounded prompt value
        # directly to the source-envelope ceil rule can incorrectly force the
        # next frame (for example, 10.01 s), overriding an otherwise-correct
        # model prediction.
        fixed_plan_duration_sec = (
            None
            if duration_sec is None or task is P11Task.EDITING
            else self.plan_codec.snap_numeric_to_grid("seconds", duration_sec)
        )
        for _ in range(max_tokens):
            if discrete_decode_mode == "prefix_recompute":
                step_embeddings = embeddings
                if generated:
                    generated_tensor = torch.tensor(
                        [generated], device=device, dtype=torch.long
                    )
                    step_embeddings = torch.cat(
                        [
                            embeddings,
                            self.plan_embedding(generated_tensor).to(qwen_dtype),
                        ],
                        dim=1,
                    )
                if step_embeddings.shape[1] > self.sequence_length:
                    break
                attention = torch.ones(
                    (1, step_embeddings.shape[1]),
                    device=device,
                    dtype=torch.bool,
                )
                output = self._run_backbone(
                    step_embeddings, attention, use_cache=False
                )
                hidden = output.last_hidden_state[0, -1].float()
            if hidden is None:
                raise RuntimeError("P11 discrete decoder did not produce a hidden state")
            if constrained:
                if task is P11Task.EDITING:
                    allowed = self.patch_codec.allowed_next_ids(
                        generated, input_sceneplan=input_sceneplan
                    )
                    # The frozen patch-v1 grammar exposes every non-terminal
                    # frame as a possible SET_ACTIVITY onset, including frames
                    # beyond a shorter current ScenePlan.  Such a choice has no
                    # legal offset on the next step.  Keep the frozen codec and
                    # manifest hashes intact, while making live constrained
                    # decoding prefix-complete by applying the tighter P10
                    # duration bound before selecting the onset token.
                    if (
                        len(generated) == 3
                        and generated[1]
                        == self.patch_codec.token_to_id["<op_set_activity>"]
                    ):
                        duration_frame = self.plan_codec._duration_frame(
                            input_sceneplan["duration_sec"]
                        )
                        allowed &= set(
                            self.plan_codec.frame_ids[:duration_frame]
                        )
                else:
                    allowed = self.plan_codec.allowed_next_ids(
                        generated,
                        min_sources=1,
                        max_sources=4,
                        fixed_duration_sec=fixed_plan_duration_sec,
                    )
                allowed &= valid_set
            else:
                allowed = set(valid_set)
            if task is not P11Task.EDITING:
                allowed -= self.blocked_plan_token_ids
            if not allowed:
                break
            if (
                constrained
                and task is not P11Task.EDITING
                and text_begin is not None
                and text_end is not None
                and text_begin in generated
            ):
                last_begin = len(generated) - 1 - generated[::-1].index(text_begin)
                last_end = (
                    len(generated) - 1 - generated[::-1].index(text_end)
                    if text_end in generated
                    else -1
                )
                if last_begin > last_end and len(generated) - last_begin - 1 >= self.plan_text_field_max_tokens:
                    if allowed != {text_end}:
                        text_safety_interventions += 1
                    allowed = {text_end}
            logits = F.linear(hidden, self.plan_embedding.weight.float(), self.output_bias)
            # Preserve the pre-constraint source-kind likelihoods needed by
            # the shared, input-only reliable-ASR owner rule.  D0 remains a
            # full-ScenePlan autoregressive baseline; these diagnostics only
            # let the evaluator apply the same deterministic lexical boundary
            # as Direct/Flow without consulting the target transcript.
            if (
                task is not P11Task.EDITING
                and generated
                and generated[-1] == kind_marker_id
            ):
                if len(generated) < 2 or generated[-2] not in source_slot_by_token:
                    raise RuntimeError("P11 D0 kind step lost its source owner")
                ordered_kinds = ("music", "sound", "speech")
                kind_logits = torch.stack(
                    [logits[kind_token_ids[kind]] for kind in ordered_kinds]
                )
                probabilities = torch.softmax(kind_logits, dim=0)
                speech_index = ordered_kinds.index("speech")
                acoustic_max = torch.stack(
                    [
                        kind_logits[ordered_kinds.index("music")],
                        kind_logits[ordered_kinds.index("sound")],
                    ]
                ).max()
                source_kind_scores.append(
                    {
                        "source_id": source_slot_by_token[generated[-2]],
                        "music_probability": float(
                            probabilities[ordered_kinds.index("music")]
                        ),
                        "sound_probability": float(
                            probabilities[ordered_kinds.index("sound")]
                        ),
                        "speech_probability": float(probabilities[speech_index]),
                        "speech_margin": float(
                            kind_logits[speech_index] - acoustic_max
                        ),
                    }
                )
            raw = logits.index_select(0, valid_ids)
            if not bool(torch.isfinite(raw).all()):
                raise RuntimeError("P11 produced non-finite planner logits")
            raw_token = int(valid_ids[int(raw.argmax())])
            if constrained and raw_token not in allowed:
                interventions += 1
            choices = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
            candidates = logits.index_select(0, choices)
            if temperature <= 0.0 or choices.numel() == 1:
                selected = int(candidates.argmax())
            else:
                probabilities = torch.softmax(candidates / float(temperature), dim=-1)
                selected = int(
                    torch.multinomial(
                        probabilities,
                        1,
                        generator=sampling_generator,
                    )
                )
            token = int(choices[selected])
            generated.append(token)
            if token == codec.eos_id:
                break
            replayed_canonical_prefix = False
            if (
                constrained
                and task is not P11Task.EDITING
                and text_begin is not None
                and text_end is not None
                and token == text_end
            ):
                # SentencePiece admits multiple valid segmentations for the
                # same decoded text.  Training targets use exactly one codec
                # segmentation, so normalize each completed text field at its
                # boundary and replay the causal cache before generating any
                # subsequent executable fields.  This is online canonical
                # forcing, not end-of-sequence repair: downstream predictions
                # condition on the canonical prefix.
                last_begin = len(generated) - 1 - generated[::-1].index(text_begin)
                piece_ids = [
                    value - self.plan_codec.text_offset
                    for value in generated[last_begin + 1 : -1]
                ]
                decoded_text = self.plan_codec.text_processor.decode(piece_ids)
                canonical_span = self.plan_codec._text(decoded_text)
                previous_span = generated[last_begin:]
                if previous_span != canonical_span:
                    canonical_text_forcing_interventions += 1
                    canonical_text_forcing_token_delta += abs(
                        len(previous_span) - len(canonical_span)
                    )
                    generated[last_begin:] = canonical_span
                    if len(generated) > max_tokens:
                        break
                    if discrete_decode_mode == "cached":
                        generated_tensor = torch.tensor(
                            [generated], device=device, dtype=torch.long
                        )
                        generated_embeddings = self.plan_embedding(
                            generated_tensor
                        ).to(qwen_dtype)
                        replay_embeddings = torch.cat(
                            [embeddings, generated_embeddings], dim=1
                        )
                        attention = torch.ones(
                            (1, replay_embeddings.shape[1]),
                            device=device,
                            dtype=torch.bool,
                        )
                        output = self._run_backbone(
                            replay_embeddings,
                            attention,
                            use_cache=True,
                        )
                        cache = output.past_key_values
                        hidden = output.last_hidden_state[0, -1].float()
                    replayed_canonical_prefix = True
            if replayed_canonical_prefix:
                continue
            if discrete_decode_mode == "prefix_recompute":
                continue
            next_embedding = self.plan_embedding(
                torch.tensor([[token]], device=device)
            ).to(qwen_dtype)
            attention = torch.ones(
                (1, attention.shape[1] + 1), device=device, dtype=torch.bool
            )
            output = self._run_backbone(
                next_embedding,
                attention,
                use_cache=True,
                past_key_values=cache,
            )
            cache = output.past_key_values
            hidden = output.last_hidden_state[0, -1].float()

        result = torch.tensor(generated, device=device, dtype=torch.long)
        terminated = bool(generated and generated[-1] == codec.eos_id)
        canonicalization_changed = False
        if constrained and terminated:
            canonical = codec.canonicalize(result.detach().cpu())["input_ids"].to(device)
            canonicalization_changed = not torch.equal(result, canonical)
            result = canonical
        diagnostics = {
            "output_kind": "edit_patch" if task is P11Task.EDITING else "sceneplan",
            "decode_output_contract": P11_DECODE_OUTPUT_CONTRACT,
            "constrained": bool(constrained),
            "terminated": terminated,
            "generated_tokens": int(result.numel()),
            "canonicalization_changed": bool(canonicalization_changed),
            "grammar_interventions": interventions,
            "grammar_intervention_rate": interventions / len(generated) if generated else 0.0,
            "text_safety_interventions": text_safety_interventions,
            "canonical_text_forcing_interventions": (
                canonical_text_forcing_interventions
            ),
            "canonical_text_forcing_token_delta": (
                canonical_text_forcing_token_delta
            ),
            "planner_arm": "discrete_d0",
            "executor_capability_contract": self.executor_contract[
                "capability_contract"
            ],
            "planner_max_latent_frames": MAX_LATENT_FRAMES,
            "p10_max_latent_frames": P10_MAX_LATENT_FRAMES,
            "planner_motion_types": list(P11_SUPPORTED_MOTION_TYPES),
            "requested_duration_sec": (
                None if duration_sec is None else float(duration_sec)
            ),
            "sampling_seed": sampling_seed,
            "sampling_rng": (
                "local_explicit_generator"
                if sampling_seed is not None
                else "process_global_generator"
            ),
            "fixed_plan_duration_sec": fixed_plan_duration_sec,
            "discrete_decode_mode": discrete_decode_mode,
            "source_kind_scores": source_kind_scores,
            **context_metrics,
        }
        return result, diagnostics

    @torch.no_grad()
    def generate_output(self, *args, **kwargs) -> Tensor:
        generated, diagnostics = self.decode_output_tokens(*args, **kwargs, constrained=True)
        if not diagnostics["terminated"]:
            raise RuntimeError("P11 planner did not terminate within its token ceiling")
        return generated

    def _ensure_pretransform(self, device: torch.device | str | None = None):
        if self.pretransform is None:
            if self._pretransform_cfg is None:
                raise RuntimeError("P11 config has no frozen VAE pretransform")
            from .factory import create_pretransform_from_config

            self.pretransform = create_pretransform_from_config(
                self._pretransform_cfg, self.sample_rate
            )
            self.pretransform.eval().requires_grad_(False)
        if device is not None:
            self.pretransform.to(device)
        return self.pretransform

    def ensure_pretransform(self, device: torch.device | str | None = None):
        return self._ensure_pretransform(device)

    def load_pretransform_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        self._ensure_pretransform("cpu").load_state_dict(state_dict)

    @torch.no_grad()
    def encode_audio(self, audio: Tensor) -> Tensor:
        return self._ensure_pretransform(audio.device).encode(audio)

    def finalize_plan(
        self,
        plan_token_ids: Mapping[str, Tensor] | Tensor | Sequence[int],
        *,
        task: P11Task | str,
        sample_id: str,
        resolver: ScenePlanResolver | None = None,
        edit_patch_token_ids: Tensor | None = None,
        edit_patch: Mapping[str, Any] | None = None,
    ) -> ScenePlanExecutionBundle:
        return finalize_sceneplan_for_p10(
            self.plan_codec,
            plan_token_ids,
            tokenizer=self.tokenizer,
            task=task,
            sample_id=sample_id,
            resolver=resolver,
            edit_patch_token_ids=edit_patch_token_ids,
            edit_patch=edit_patch,
        )

    @torch.no_grad()
    def plan_generation(
        self,
        user_text: str,
        *,
        duration_sec: float | None = None,
        sample_id: str = "p11_generation",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
    ) -> ScenePlanExecutionBundle:
        duration = self.default_plan_duration_sec if duration_sec is None else float(duration_sec)
        tokens = self.generate_output(
            user_text,
            task=P11Task.GENERATION,
            duration_sec=duration,
            temperature=temperature,
        )
        return self.finalize_plan(
            tokens, task=P11Task.GENERATION, sample_id=sample_id, resolver=resolver
        )

    @torch.no_grad()
    def plan_understanding(
        self,
        input_foa: Tensor,
        *,
        input_semantic: Tensor | None = None,
        prompt: str = "Transcribe the input FOA into its complete ScenePlan.",
        sample_id: str = "p11_understanding",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
    ) -> ScenePlanExecutionBundle:
        duration = min(
            MAX_LATENT_FRAMES * self.downsampling_ratio / self.sample_rate,
            float(input_foa.shape[-1]) * self.downsampling_ratio / self.sample_rate,
        )
        tokens = self.generate_output(
            prompt,
            task=P11Task.UNDERSTANDING,
            input_foa=input_foa,
            input_semantic=input_semantic,
            duration_sec=duration,
            temperature=temperature,
        )
        return self.finalize_plan(
            tokens, task=P11Task.UNDERSTANDING, sample_id=sample_id, resolver=resolver
        )

    @torch.no_grad()
    def plan_editing(
        self,
        edit_instruction: str,
        *,
        input_sceneplan: Mapping[str, Any],
        sample_id: str = "p11_editing",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
    ) -> ScenePlanExecutionBundle:
        patch_tokens = self.generate_output(
            edit_instruction,
            task=P11Task.EDITING,
            input_sceneplan=input_sceneplan,
            temperature=temperature,
        )
        patch = self.patch_codec.decode(patch_tokens)
        revised = self.patch_codec.apply(input_sceneplan, patch)
        plan_tokens = self.plan_codec.encode(
            revised, max_tokens=self.plan_max_tokens
        )["input_ids"]
        return self.finalize_plan(
            plan_tokens,
            task=P11Task.EDITING,
            sample_id=sample_id,
            resolver=resolver,
            edit_patch_token_ids=patch_tokens,
            edit_patch=patch,
        )


def create_sceneplan_p11_from_config(
    model_config: Mapping[str, Any],
) -> ScenePlanP11Planner:
    return ScenePlanP11Planner(model_config)


__all__ = [
    "P11_DECODE_OUTPUT_CONTRACT",
    "ScenePlanP11Planner",
    "create_sceneplan_p11_from_config",
]
