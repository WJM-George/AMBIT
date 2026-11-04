from torch.nn import functional as F

from torch import nn
from .editing_ar import EditingScenePlanAdapter
from .transcript import TranscriptAdapter
from .transcript_attention import AudioTranscriptAttention, pack_memory


class UnifiedTransfusion(nn.Module):
    """One Transformer and one plan vocabulary/head, with optional audio reference.

    Editing retains its trained audio slot residual. Generation uses the same
    embedding, output norm and plan head without requiring fictitious audio.
    All target-plan metadata belongs exclusively to the RF forward.
    """

    def __init__(self, pipeline):
        super().__init__()
        self.ar = pipeline.editing_ar
        self.diffusion = pipeline.diffusion
        self.codec = pipeline.codec
        self.ar_task = nn.Embedding(2, 1024)
        self.rf_task = nn.Embedding(2, 768)
        nn.init.zeros_(self.ar_task.weight)
        nn.init.zeros_(self.rf_task.weight)
        self.ar.source_clap_model.requires_grad_(False).eval()
        self.ar.instruction_conditioner.model.requires_grad_(False).eval()
        self.ar.source_structure.readout.output_heads.requires_grad_(False)
        self.ar.source_structure.target_heads.requires_grad_(False)
        if self.ar.shared_transformer is not self.diffusion.model.model.transformer:
            raise RuntimeError("Unified model lost its single shared Transformer")

    def instruction_context(self, task, instructions, device):
        (context, context_mask) = self.ar.encode_edit_instructions(instructions, device=device)
        context = context + self.ar_task.weight[task].to(context.dtype)[None, None]
        return (context, context_mask)

    def plan_logits(
        self,
        task,
        ids,
        mask,
        instructions,
        source=None,
        source_mask=None,
        *,
        encoded_context=None,
        source_features=None,
    ):
        (context, context_mask) = (
            self.instruction_context(task, instructions, ids.device)
            if encoded_context is None
            else encoded_context
        )
        if task == 1:
            if source is None or source_mask is None:
                raise ValueError("Editing requires clean source audio")
            return self.ar(
                source,
                source_mask,
                ids,
                mask,
                context,
                context_mask,
                source_clap_features=source_features,
            )
        if task != 0 or source is not None:
            raise ValueError("Generation accepts no source audio")
        hidden = self.ar.plan_adapter.embed(ids) + self.ar.plan_type_embedding
        hidden = self.ar.shared_transformer(
            hidden,
            context=context,
            context_mask=context_mask,
            padding_mask=mask,
            skip_input_projection=True,
            skip_output_projection=True,
            self_attention_causal=True,
        )
        return EditingScenePlanAdapter.logits(self.ar.plan_adapter, hidden)

    def renderer_inputs(self, task, metadata, device):
        if task not in (0, 1):
            raise ValueError("Unknown task")
        if task == 0 and any((bool(row["source_foa_latent"].count_nonzero()) for row in metadata)):
            raise ValueError("Generation RF reference must be absent/zero")
        conditioning = self.diffusion.conditioner(metadata, device)
        inputs = self.diffusion.get_conditioning_inputs(conditioning)
        context = inputs["cross_attn_cond"]
        inputs["cross_attn_cond"] = (
            context + self.rf_task.weight[task].to(context.dtype)[None, None]
        )
        return inputs


class TranscriptUnifiedTransfusion(UnifiedTransfusion):
    def __init__(self, pipeline, settings):
        super().__init__(pipeline)
        self.transcript_adapter = TranscriptAdapter(**settings)
        if list(self.diffusion.cross_attn_cond_ids) != ["prompt"]:
            raise ValueError("Lexical residual requires the native single prompt context")

    def renderer_inputs(self, task, metadata, device):
        inputs = super().renderer_inputs(task, metadata, device)
        inputs["cross_attn_cond"] = self.transcript_adapter(
            inputs["cross_attn_cond"],
            metadata,
            self.diffusion.conditioner.conditioners["prompt"].tokenizer,
        )
        return inputs


class AudioTranscriptUnifiedTransfusion(TranscriptUnifiedTransfusion):
    def __init__(self, pipeline, settings, audio_settings):
        super().__init__(pipeline, settings)
        hidden = self.ar.shared_transformer.dim
        if self.diffusion.model.model.patch_size != 1:
            raise ValueError("Audio transcript memory requires native one-frame patches")
        self.audio_transcript_attention = AudioTranscriptAttention(
            hidden_dim=hidden, memory_dim=settings["width"], **audio_settings
        )

    def renderer_inputs(self, task, metadata, device):
        inputs = UnifiedTransfusion.renderer_inputs(self, task, metadata, device)
        context = inputs["cross_attn_cond"]
        tokenizer = self.diffusion.conditioner.conditioners["prompt"].tokenizer
        adapter = self.transcript_adapter
        (sequences, slots, rows, columns, destinations, weights) = adapter.prepare(
            metadata, tokenizer, context.shape[1]
        )
        if not sequences:
            return inputs
        hidden = adapter.encode(sequences, slots, context.device)
        inputs["cross_attn_cond"] = adapter.pool(
            context, hidden, rows, columns, destinations, weights
        )
        memory = pack_memory(
            hidden, sequences, slots, rows, destinations, tokens=context.shape[1], metadata=metadata
        )
        inputs["audio_transcript_condition"] = (self.audio_transcript_attention, memory)
        return inputs


class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.qkv = nn.Linear(1024, 3072)
        self.out = nn.Linear(1024, 1024)
        self.ff_norm = nn.LayerNorm(1024)
        self.up = nn.Linear(1024, 4096)
        self.down = nn.Linear(4096, 1024)
        for layer in (self.out, self.down):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        (b, t, _) = x.shape
        (q, k, v) = self.qkv(self.norm(x)).reshape(b, t, 3, 8, 128).permute(2, 0, 3, 1, 4).unbind(0)
        x = x + self.out(
            F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            .transpose(1, 2)
            .reshape(b, t, 1024)
        )
        return x + self.down(F.gelu(self.up(self.ff_norm(x))))


class ExpandedPlanHead(nn.Module):
    def __init__(self, base, blocks):
        super().__init__()
        self.base = base
        self.blocks = nn.ModuleList([CausalBlock() for _ in range(blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.base(x)
