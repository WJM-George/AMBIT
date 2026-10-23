import copy
import torch
from torch import nn
from ambit._runtime.models.heads import MAX_KEYFRAMES
from ambit._runtime.sceneplan.schema import MAX_DURATION_SEC
from ambit._runtime.models.editing_ar import EditingScenePlanAdapter, ScenePlanTransfusionEditingAR

ACTIONS = ("EMPTY", "KEEP", "REMOVE", "RELOCATE", "STATIC_TO_LINEAR", "LINEAR_TO_STATIC", "ADD")


def predict_fields(hidden, heads):
    out = {name: head(hidden).float() for (name, head) in heads.items()}
    out["activity_sec"] = out["activity_sec"].sigmoid() * MAX_DURATION_SEC
    out["keyframe_time_sec"] = out["keyframe_time_sec"].sigmoid() * MAX_DURATION_SEC
    out["gain_db"] = out["gain_db"].squeeze(-1).sigmoid() * 36 - 24
    out["direction_xyz"] = out["direction_xyz"].reshape(*hidden.shape[:2], MAX_KEYFRAMES, 3)
    return out


class SourceStructure(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = readout
        width = readout.queries.shape[-1]
        self.new_query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.slot_identity = nn.Embedding(5, width)
        self.instruction_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
        layer = nn.TransformerDecoderLayer(
            width, 6, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.edit_decoder = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(width))
        self.edit_target = nn.Linear(width, 1)
        self.operation = nn.Linear(width, len(ACTIONS))
        self.edit_delta = nn.Linear(width, width)
        nn.init.zeros_(self.edit_delta.weight)
        nn.init.zeros_(self.edit_delta.bias)
        self.target_heads = copy.deepcopy(readout.output_heads)
        self.operation_embedding = nn.Parameter(torch.randn(len(ACTIONS), width) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(width) * 0.02)
        self.memory_norm = nn.LayerNorm(width)

    def forward(self, sequence, sequence_mask, instruction_context, instruction_mask):
        if (
            sequence.shape[:2] != sequence_mask.shape
            or instruction_context.shape[:2] != instruction_mask.shape
        ):
            raise ValueError("Structured AR feature masks do not align")
        projected = self.readout.input_projection(sequence)
        source_hidden = self.readout.decoder(
            self.readout.queries[None].expand(len(sequence), -1, -1),
            projected,
            memory_key_padding_mask=~sequence_mask.bool(),
        )
        source = predict_fields(source_hidden, self.readout.output_heads)
        slots = torch.cat(
            (source_hidden, self.new_query[None].expand(len(sequence), -1, -1)), dim=1
        )
        query = slots + self.slot_identity.weight[None]
        context = self.instruction_projection(instruction_context)
        bound = self.edit_decoder(query, context, memory_key_padding_mask=~instruction_mask.bool())
        target_logits = self.edit_target(bound).squeeze(-1)
        operation_logits = self.operation(bound)
        edited_hidden = slots + self.edit_delta(bound)
        target = predict_fields(edited_hidden, self.target_heads)
        memory = self.memory_norm(
            edited_hidden
            + self.slot_identity.weight[None]
            + operation_logits.softmax(-1) @ self.operation_embedding
            + target_logits.softmax(-1)[..., None] * self.target_embedding
        )
        return {
            "source": source,
            "target": target,
            "target_logits": target_logits,
            "operation_logits": operation_logits,
            "memory": memory,
        }


class SlotResidual(nn.Module):
    def __init__(self, hidden=1024, width=192):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.query_projection = nn.Linear(hidden, width)
        self.attention = nn.MultiheadAttention(width, 6, dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(width, hidden)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden, memory):
        query = self.query_projection(self.query_norm(hidden))
        value = self.attention(query, memory, memory, need_weights=False)[0]
        return hidden + self.gate.tanh() * self.output_projection(value)


class SlotPlanAdapter(EditingScenePlanAdapter):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.hidden_dim = original.hidden_dim
        self.token_embedding = original.token_embedding
        self.output_norm = original.output_norm
        self.plan_head = original.plan_head
        self.slot_residual = SlotResidual(self.hidden_dim)
        self.slot_memory = None

    def logits(self, hidden_states):
        if self.slot_memory is None:
            raise RuntimeError("Structured decoder needs predicted audio/request slot memory")
        return super().logits(self.slot_residual(hidden_states, self.slot_memory))


def ar_forward(
    self,
    source_foa_latent,
    source_attention_mask,
    plan_input_ids,
    plan_attention_mask,
    instruction_context,
    instruction_attention_mask,
    source_clap_features=None,
    *,
    return_structure=False,
):
    features = source_clap_features
    if features is None:
        features = self.source_clap_model.source_features(source_foa_latent, source_attention_mask)
    cached = features.get("_structured_audio_request_cache")
    if (
        not torch.is_grad_enabled()
        and True
        and (cached is not None)
        and (cached[0] is instruction_context)
        and (cached[1] is instruction_attention_mask)
    ):
        outputs = cached[2]
    else:
        outputs = self.source_structure(
            features["sequence"],
            features["sequence_mask"],
            instruction_context,
            instruction_attention_mask,
        )
        if not torch.is_grad_enabled() and True:
            features["_structured_audio_request_cache"] = (
                instruction_context,
                instruction_attention_mask,
                outputs,
            )
    self.plan_adapter.slot_memory = outputs["memory"]
    try:
        logits = ScenePlanTransfusionEditingAR.forward(
            self,
            source_foa_latent,
            source_attention_mask,
            plan_input_ids,
            plan_attention_mask,
            instruction_context,
            instruction_attention_mask,
            source_clap_features=features,
        )
    finally:
        self.plan_adapter.slot_memory = None
    return (logits, outputs) if return_structure else logits
