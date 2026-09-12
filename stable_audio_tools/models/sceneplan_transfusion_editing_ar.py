"""Audio-reference Editing AR on the Editing-DiT Transformer blocks.

The only Editing-AR information sources are the clean source FOA latent,
audio-derived CLAP44 features, and the raw edit instruction. Legacy semantic
modes remain loadable for historical checkpoint reproducibility. A source/old ScenePlan or
caption is deliberately absent from
every public forward method. Teacher-forced new-ScenePlan tokens are labels,
not conditioning truth: the prefix-LM mask prevents reference frames and
earlier plan positions from seeing future target tokens.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from stable_audio_tools.models.conditioners import QwenTextConditioner
from stable_audio_tools.models.dit import DiffusionTransformer
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (
    EditingARSourceSemanticBridge,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import (
    EditingCLAP44, EditingCLAP44SourceBridge,
)
from stable_audio_tools.models.transformer import ContinuousTransformer
from stable_audio_tools.data.sceneplan_transfusion_editing_plan import (
    editing_ar_allowed_next_ids,
)


EDITING_AR_CONTRACT = "p10v11_shared_blocks_audio_reference_editing_ar_v4"
EDITING_AR_CLAP44_CONTRACT = "p10v11_shared_blocks_audio_reference_editing_ar_clap44_v1"
EDITING_AR_VOCAB_SIZE = 4096
EDITING_AR_HIDDEN_DIM = 1024
EDITING_AR_DEPTH = 15
EDITING_AR_HEADS = 16
EDITING_AR_SOURCE_CHANNELS = 64
EDITING_AR_MAX_SOURCE_FRAMES = 648
EDITING_AR_MAX_INSTRUCTION_TOKENS = 512


def editing_ar_prefix_attention_allowed(
    source_frames: int,
    plan_tokens: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the exact source-prefix/causal-plan self-attention relation.

    Rows are queries and columns are keys.  Source queries see all source
    frames and no plan tokens.  Plan queries see all source frames and only
    their causal plan prefix (including the current input token).
    """

    source_length = int(source_frames)
    plan_length = int(plan_tokens)
    if not 1 <= source_length <= EDITING_AR_MAX_SOURCE_FRAMES:
        raise ValueError("Editing AR source length must be in [1,648]")
    if plan_length <= 0:
        raise ValueError("Editing AR requires at least one plan input token")
    sequence = source_length + plan_length
    positions = torch.arange(sequence, device=device)
    query = positions[:, None]
    key = positions[None, :]
    source_query = query < source_length
    source_key = key < source_length
    plan_query = ~source_query
    plan_key = ~source_key
    causal_plan = plan_key & (key <= query)
    return (source_query & source_key) | (
        plan_query & (source_key | causal_plan)
    )


def editing_ar_prefix_attention_bias(
    batch_size: int,
    source_frames: int,
    plan_tokens: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    key_padding_mask: Tensor | None = None,
) -> Tensor:
    """Build the complete Editing-AR relation and key-padding SDPA bias.

    The Editing prefix relation selects the additive-bias SDPA path even when
    FlashAttention varlen metadata is available.  Consequently padding must be
    folded into this same bias instead of relying on the varlen backend to mask
    keys for us.
    """

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError("Editing AR attention bias requires a positive batch")
    if not dtype.is_floating_point:
        raise ValueError("Editing AR attention bias must use a floating dtype")
    allowed = editing_ar_prefix_attention_allowed(
        source_frames, plan_tokens, device=device
    )
    bias = torch.zeros(allowed.shape, device=device, dtype=dtype)
    bias.masked_fill_(~allowed, float("-inf"))
    bias = bias[None, None].expand(batch, 1, -1, -1)
    if key_padding_mask is not None:
        key_mask = key_padding_mask.to(device=device, dtype=torch.bool)
        expected = (batch, source_length := int(source_frames) + int(plan_tokens))
        if tuple(key_mask.shape) != expected:
            raise ValueError(
                "Editing AR key padding mask must be [batch,source+plan], "
                f"got {tuple(key_mask.shape)} vs {expected}"
            )
        if not bool(key_mask.any(dim=1).all()):
            raise ValueError("every Editing AR row requires at least one valid key")
        bias = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
    return bias


class EditingScenePlanAdapter(nn.Module):
    """Discrete codec-v4 adapter/head for the new ScenePlan target."""

    def __init__(self, *, vocab_size: int, hidden_dim: int, pad_id: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token_embedding = nn.Embedding(
            int(vocab_size), self.hidden_dim, padding_idx=int(pad_id)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.plan_head = nn.Linear(self.hidden_dim, int(vocab_size), bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.plan_head.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[int(pad_id)].zero_()

    def embed(self, token_ids: Tensor) -> Tensor:
        return self.token_embedding(token_ids) * math.sqrt(self.hidden_dim)

    def logits(self, hidden_states: Tensor) -> Tensor:
        return self.plan_head(self.output_norm(hidden_states))


class ScenePlanTransfusionEditingAR(nn.Module):
    """Predict a complete new ScenePlan from reference audio + instruction.

    ``editing_dit.transformer`` is used directly; no Transformer block is
    copied.  The source-audio and discrete-plan adapters are task-specific.
    """

    def __init__(
        self,
        *,
        editing_dit: DiffusionTransformer,
        instruction_conditioner: QwenTextConditioner,
        pad_id: int,
        vocab_size: int = EDITING_AR_VOCAB_SIZE,
        activation_checkpointing: bool = True,
        source_semantic_mode: str = "latent_only",
        source_semantic_dropout: float = 0.0,
        source_clap_model: EditingCLAP44 | None = None,
        clap44_global_features: bool = True,
        clap44_sequence_features: bool = True,
    ) -> None:
        super().__init__()
        transformer = getattr(editing_dit, "transformer", None)
        if not isinstance(transformer, ContinuousTransformer):
            raise TypeError("Editing AR requires Editing DiT ContinuousTransformer")
        if (
            int(transformer.depth) != EDITING_AR_DEPTH
            or int(transformer.dim) != EDITING_AR_HIDDEN_DIM
            or len(transformer.layers) != EDITING_AR_DEPTH
        ):
            raise ValueError("P10-v11 Editing Transformer depth/width changed")
        if any(
            int(layer.self_attn.num_heads) != EDITING_AR_HEADS
            or not bool(layer.cross_attend)
            for layer in transformer.layers
        ):
            raise ValueError("P10-v11 Editing Transformer attention contract changed")
        project_in = getattr(transformer, "project_in", None)
        project_out = getattr(transformer, "project_out", None)
        if (
            not isinstance(project_in, nn.Linear)
            or tuple(project_in.weight.shape) not in {(1024, 320), (1024, 384)}
            or not isinstance(project_out, nn.Linear)
            or tuple(project_out.weight.shape) != (64, 1024)
        ):
            raise ValueError("Editing DiT acoustic projection contract changed")

        self.editing_dit = editing_dit
        self.instruction_conditioner = instruction_conditioner
        # The Qwen backbone is frozen by its conditioner configuration.  Its
        # small projection/role parameters remain trainable so instruction
        # conditioning can adapt during joint AR+RF replay.
        self.instruction_conditioner.eval()
        self.activation_checkpointing = bool(activation_checkpointing)
        self.pad_id = int(pad_id)
        self.vocab_size = int(vocab_size)

        self.source_audio_adapter = nn.Linear(
            EDITING_AR_SOURCE_CHANNELS, EDITING_AR_HIDDEN_DIM, bias=False
        )
        # The first 64 P10/Editing-DiT columns already encode an FOA latent.
        # They are the most faithful initialization for the clean reference
        # audio adapter, while remaining a separate modality-specific head.
        with torch.no_grad():
            self.source_audio_adapter.weight.copy_(project_in.weight[:, :64])
        self.source_audio_type_embedding = nn.Parameter(
            torch.zeros(EDITING_AR_HIDDEN_DIM)
        )
        self.plan_type_embedding = nn.Parameter(torch.zeros(EDITING_AR_HIDDEN_DIM))
        self.plan_adapter = EditingScenePlanAdapter(
            vocab_size=self.vocab_size,
            hidden_dim=EDITING_AR_HIDDEN_DIM,
            pad_id=self.pad_id,
        )
        self.source_clap_model = source_clap_model
        if source_semantic_mode == "clap44_audio_caption_aux":
            if source_clap_model is None:
                raise ValueError("CLAP44 AR requires its own trained audio encoder")
            self.source_clap_model.eval().requires_grad_(False)
            self.source_semantic_bridge = EditingCLAP44SourceBridge(
                EDITING_AR_HIDDEN_DIM, source_clap_model.config,
                use_global=clap44_global_features, use_sequence=clap44_sequence_features,
                audio_feature_dropout=source_semantic_dropout,
            )
        else:
            if source_clap_model is not None:
                raise ValueError("CLAP44 encoder requires the CLAP44 semantic mode")
            self.source_semantic_bridge = EditingARSourceSemanticBridge(
                mode=source_semantic_mode,
                hidden_dim=EDITING_AR_HIDDEN_DIM,
                audio_feature_dropout=source_semantic_dropout,
            )

    @property
    def source_semantic_mode(self) -> str:
        return self.source_semantic_bridge.mode

    @property
    def ar_contract(self) -> str:
        return EDITING_AR_CLAP44_CONTRACT if self.source_semantic_mode == "clap44_audio_caption_aux" else EDITING_AR_CONTRACT

    @property
    def shared_transformer(self) -> ContinuousTransformer:
        """Return the exact block-stack object used by Editing DiT."""

        return self.editing_dit.transformer

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen Qwen instruction encoder must stay deterministic.
        self.instruction_conditioner.eval()
        if getattr(self, "source_clap_model", None) is not None:
            self.source_clap_model.eval()
        return self

    def encode_edit_instructions(
        self, instructions: Sequence[str], *, device: torch.device | str
    ) -> tuple[Tensor, Tensor]:
        if not instructions or not all(
            isinstance(text, str) and text.strip() for text in instructions
        ):
            raise ValueError("Editing AR instructions must be non-empty strings")
        encoded = self.instruction_conditioner.tokenizer(
            list(instructions),
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
        if bool((lengths > EDITING_AR_MAX_INSTRUCTION_TOKENS).any()):
            raise ValueError("Editing instruction exceeds the 512-token limit")
        zeros = torch.zeros_like(input_ids)
        tokenized = [
            {
                "input_ids": input_ids[index],
                "attention_mask": attention_mask[index],
                "event_source_ids": zeros[index],
                "speech_source_ids": zeros[index],
            }
            for index in range(len(instructions))
        ]
        context_768, context_mask = self.instruction_conditioner(
            tokenized, device
        )
        context_1024 = self.editing_dit.to_cond_embed(context_768)
        return context_1024, context_mask.to(torch.bool)

    def forward(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        plan_input_ids: Tensor,
        plan_attention_mask: Tensor,
        instruction_context: Tensor,
        instruction_attention_mask: Tensor,
        source_m2d_audio_embedding: Tensor | None = None,
        source_m2d_audio_keep_mask: Tensor | None = None,
        return_source_contrastive_query: bool = False,
        source_clap_features: dict[str, Any] | None = None,
        source_clap_keep_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if (
            source_foa_latent.ndim != 3
            or int(source_foa_latent.shape[1]) != EDITING_AR_SOURCE_CHANNELS
            or int(source_foa_latent.shape[2]) > EDITING_AR_MAX_SOURCE_FRAMES
        ):
            raise ValueError("source_foa_latent must be [batch,64,T<=648]")
        batch, _, source_frames = source_foa_latent.shape
        if tuple(source_attention_mask.shape) != (batch, source_frames):
            raise ValueError("source attention mask must align with source latent")
        if plan_input_ids.ndim != 2 or int(plan_input_ids.shape[0]) != batch:
            raise ValueError("plan_input_ids must be [batch,plan_tokens]")
        if tuple(plan_attention_mask.shape) != tuple(plan_input_ids.shape):
            raise ValueError("plan attention mask must align with plan ids")
        if (
            instruction_context.ndim != 3
            or tuple(instruction_attention_mask.shape)
            != tuple(instruction_context.shape[:2])
            or int(instruction_context.shape[0]) != batch
            or int(instruction_context.shape[-1]) != EDITING_AR_HIDDEN_DIM
        ):
            raise ValueError("instruction context/mask does not match AR batch")
        if not bool(source_attention_mask.to(torch.bool).any(dim=1).all()):
            raise ValueError("every Editing AR row requires valid reference audio")
        if not bool(plan_attention_mask.to(torch.bool).any(dim=1).all()):
            raise ValueError("every Editing AR row requires a plan prefix")
        if bool(
            (plan_input_ids < 0).any()
            or (plan_input_ids >= self.vocab_size).any()
        ):
            raise ValueError("plan token id is outside codec-v4 vocabulary")
        if not bool(torch.isfinite(source_foa_latent).all()):
            raise ValueError("source FOA latent contains non-finite values")

        source_mask = source_attention_mask.to(
            device=source_foa_latent.device, dtype=torch.bool
        )
        plan_mask = plan_attention_mask.to(
            device=source_foa_latent.device, dtype=torch.bool
        )
        source_values = source_foa_latent.transpose(1, 2)
        source_values = source_values * source_mask.unsqueeze(-1).to(
            source_values.dtype
        )
        source_hidden = self.source_audio_adapter(source_values)
        source_hidden = source_hidden + self.source_audio_type_embedding
        if getattr(self, "source_clap_model", None) is not None:
            if source_m2d_audio_embedding is not None or source_m2d_audio_keep_mask is not None:
                raise ValueError("CLAP44 AR must not consume M2D features")
            features = source_clap_features
            if features is None:
                features = self.source_clap_model.source_features(source_foa_latent, source_mask)
            source_hidden = self.source_semantic_bridge.inject(source_hidden, features, source_clap_keep_mask)
        else:
            if source_clap_features is not None or source_clap_keep_mask is not None:
                raise ValueError("CLAP44 features supplied to a different AR route")
            source_hidden = self.source_semantic_bridge.inject(
                source_hidden, source_m2d_audio_embedding, source_m2d_audio_keep_mask,
            )
        source_contrastive_query = None
        if return_source_contrastive_query:
            source_contrastive_query = (
                self.source_semantic_bridge.contrastive_query(
                    source_hidden, source_mask
                )
            )
        plan_hidden = self.plan_adapter.embed(plan_input_ids)
        plan_hidden = plan_hidden + self.plan_type_embedding
        hidden = torch.cat((source_hidden, plan_hidden), dim=1)
        combined_mask = torch.cat((source_mask, plan_mask), dim=1)
        attention_bias = editing_ar_prefix_attention_bias(
            batch,
            source_frames,
            int(plan_input_ids.shape[1]),
            device=hidden.device,
            dtype=hidden.dtype,
            key_padding_mask=combined_mask,
        )
        hidden = self.shared_transformer(
            hidden,
            context=instruction_context,
            context_mask=instruction_attention_mask.to(torch.bool),
            padding_mask=combined_mask,
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=False,
            self_attention_bias=attention_bias,
            use_checkpointing=self.activation_checkpointing and self.training,
        )
        plan_hidden = hidden[:, source_frames:, :]
        logits = self.plan_adapter.logits(plan_hidden)
        if return_source_contrastive_query:
            assert source_contrastive_query is not None
            return logits, source_contrastive_query
        return logits

    @torch.no_grad()
    def generate_batch(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        instructions: Sequence[str],
        *,
        codec: Any,
        max_plan_tokens: int = 1024,
        fixed_duration_sec: float | Sequence[float | None] | None = None,
        source_m2d_audio_embedding: Tensor | None = None,
    ) -> list[Tensor]:
        """Batch grammar-constrained prefix-recompute decoding.

        Instruction context is encoded once and all unfinished rows share each
        Transformer call.  The implementation deliberately uses the same
        source-prefix attention path as training; it never reconstructs or
        accepts an old ScenePlan.
        """

        if source_foa_latent.ndim != 3:
            raise ValueError("generate_batch source latent must be [batch,64,T]")
        batch = int(source_foa_latent.shape[0])
        if batch <= 0 or len(instructions) != batch:
            raise ValueError("generate_batch instruction count must match audio")
        if tuple(source_attention_mask.shape) != (
            batch,
            int(source_foa_latent.shape[-1]),
        ):
            raise ValueError("generate_batch source mask must align with audio")
        if int(max_plan_tokens) <= 1:
            raise ValueError("max_plan_tokens must exceed one")
        if isinstance(fixed_duration_sec, Sequence) and not isinstance(
            fixed_duration_sec, (str, bytes)
        ):
            durations = list(fixed_duration_sec)
            if len(durations) != batch:
                raise ValueError("fixed-duration count must match audio batch")
        else:
            durations = [fixed_duration_sec for _ in range(batch)]
        context, context_mask = self.encode_edit_instructions(
            instructions, device=source_foa_latent.device
        )
        clap_features = None
        if getattr(self, "source_clap_model", None) is not None:
            if source_m2d_audio_embedding is not None:
                raise ValueError("CLAP44 generation must not consume M2D embeddings")
            # Encode the reference once, not once for every generated token.
            clap_features = self.source_clap_model.source_features(
                source_foa_latent, source_attention_mask.to(torch.bool)
            )
        prefixes = [[int(codec.bos_id)] for _ in range(batch)]
        finished = [False] * batch
        for _ in range(int(max_plan_tokens) - 1):
            maximum_prefix = max(len(prefix) for prefix in prefixes)
            plan_ids = torch.tensor(
                [
                    prefix
                    + [int(self.pad_id)] * (maximum_prefix - len(prefix))
                    for prefix in prefixes
                ],
                device=source_foa_latent.device,
                dtype=torch.long,
            )
            plan_mask = plan_ids.ne(int(self.pad_id))
            semantic_kwargs = (
                {}
                if source_m2d_audio_embedding is None
                else {
                    "source_m2d_audio_embedding": source_m2d_audio_embedding
                }
            )
            if clap_features is not None:
                semantic_kwargs["source_clap_features"] = clap_features
            logits = self(
                source_foa_latent,
                source_attention_mask,
                plan_ids,
                plan_mask,
                context,
                context_mask,
                **semantic_kwargs,
            )
            for index in range(batch):
                if finished[index]:
                    continue
                prefix = prefixes[index]
                allowed = sorted(
                    int(value)
                    for value in editing_ar_allowed_next_ids(
                        codec, prefix, fixed_duration_sec=durations[index]
                    )
                )
                if not allowed:
                    raise RuntimeError("Editing AR grammar produced no next token")
                allowed_tensor = torch.tensor(
                    allowed, device=logits.device, dtype=torch.long
                )
                row_logits = logits[index, len(prefix) - 1]
                selected = int(
                    allowed_tensor[row_logits[allowed_tensor].argmax()].item()
                )
                prefix.append(selected)
                finished[index] = selected == int(codec.eos_id)
            if all(finished):
                return [torch.tensor(prefix, dtype=torch.long) for prefix in prefixes]
        unfinished = [index for index, value in enumerate(finished) if not value]
        raise RuntimeError(
            "Editing AR did not emit EOS within max_plan_tokens for rows "
            f"{unfinished}"
        )

    @torch.no_grad()
    def generate_one(
        self,
        source_foa_latent: Tensor,
        source_attention_mask: Tensor,
        instruction: str,
        *,
        codec: Any,
        max_plan_tokens: int = 1024,
        fixed_duration_sec: float | None = None,
        source_m2d_audio_embedding: Tensor | None = None,
    ) -> Tensor:
        """Grammar-constrained decoding convenience wrapper for one edit."""

        if int(source_foa_latent.shape[0]) != 1:
            raise ValueError("generate_one requires a batch of exactly one")
        return self.generate_batch(
            source_foa_latent,
            source_attention_mask,
            [instruction],
            codec=codec,
            max_plan_tokens=max_plan_tokens,
            fixed_duration_sec=fixed_duration_sec,
            source_m2d_audio_embedding=source_m2d_audio_embedding,
        )[0]


__all__ = [
    "EDITING_AR_CONTRACT",
    "EDITING_AR_CLAP44_CONTRACT",
    "EditingScenePlanAdapter",
    "ScenePlanTransfusionEditingAR",
    "editing_ar_prefix_attention_allowed",
    "editing_ar_prefix_attention_bias",
]
