"""Unpooled byte memory -> audio hidden states, with source-activity masks.

The existing byte encoder and token residual are reused. No forced alignment,
word timestamps, target waveform or target latent enters this condition path.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TranscriptMemory:
    values: torch.Tensor  # [B, bytes, encoder_width]
    readable: torch.Tensor  # [B, frames, bytes], True = allowed


def pack_memory(hidden, sequences, slots, locations, destinations, *, tokens, metadata):
    """Pack full transcripts per scene; never pool or silently truncate bytes."""
    batch = len(metadata)
    owners = [None] * len(sequences)
    for sequence, destination in zip(locations, destinations):
        owner = destination // tokens
        if owners[sequence] not in (None, owner):
            raise ValueError("One transcript cannot belong to two scenes")
        owners[sequence] = owner
    if any(owner is None for owner in owners):
        raise ValueError("A transcript lost its native token coverage")
    frames = [int(row["latent_crop_length"]) for row in metadata]
    if len(set(frames)) != 1:
        raise ValueError("Transcript memory requires one audio bucket per batch")
    length = max(
        sum(len(s) for s, owner in zip(sequences, owners) if owner == b) for b in range(batch)
    )
    packed, masks = [], []
    for b, row in enumerate(metadata):
        parts, allowed = [], []
        controls = row["sceneplan_44"]
        valid = torch.arange(frames[b], device=hidden.device) < int(row["latent_frames_valid"])
        for i, (sequence, slot, owner) in enumerate(zip(sequences, slots, owners)):
            if owner != b:
                continue
            parts.append(hidden[i, : len(sequence)])
            # Independent structured dropout must not leak GT source intervals.
            # With activity unknown, text remains readable at all valid frames.
            if controls.get("cfg_unknown", False):
                active = valid
            else:
                events = torch.as_tensor(
                    controls["source_event_frame_ids"][slot], device=hidden.device
                )
                if events.shape != valid.shape:
                    raise ValueError("Source activity and audio frames disagree")
                active = events.eq(slot + 1) & valid
            allowed.append(active[:, None].expand(-1, len(sequence)))
        values = torch.cat(parts) if parts else hidden.new_zeros((0, hidden.shape[-1]))
        mask = (
            torch.cat(allowed, dim=1)
            if allowed
            else torch.zeros(frames[b], 0, device=hidden.device, dtype=torch.bool)
        )
        packed.append(F.pad(values, (0, 0, 0, length - values.shape[0])))
        masks.append(F.pad(mask, (0, length - mask.shape[1]), value=False))
    return TranscriptMemory(torch.stack(packed), torch.stack(masks))


class AudioTranscriptAttention(nn.Module):
    def __init__(self, hidden_dim=1024, memory_dim=256, width=256, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.heads, self.width = heads, width
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.to_q = nn.Linear(hidden_dim, width, bias=False)
        self.to_k = nn.Linear(memory_dim, width, bias=False)
        self.to_v = nn.Linear(memory_dim, width, bias=False)
        self.out = nn.Linear(width, hidden_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, audio, memory, padding_mask=None):
        b, frames, _ = audio.shape
        if memory.values.shape[0] != b or memory.readable.shape != (
            b,
            frames,
            memory.values.shape[1],
        ):
            raise ValueError("Transcript attention must align to the unprefixed audio sequence")
        readable = memory.readable
        if padding_mask is not None:
            if padding_mask.shape != (b, frames):
                raise ValueError("Transcript padding mask shape mismatch")
            readable = readable & padding_mask.bool()[:, :, None]
        active = readable.any(dim=-1)
        # One zero-valued key is readable ONLY on all-masked queries. This
        # avoids NaNs on inactive/padded/no-speech rows without diluting speech.
        allowed = torch.cat((readable, ~active[:, :, None]), dim=-1)
        values = self.memory_norm(memory.values)
        q = self.to_q(self.query_norm(audio))
        k = F.pad(self.to_k(values), (0, 0, 0, 1))
        v = F.pad(self.to_v(values), (0, 0, 0, 1))

        def split(x):
            return x.reshape(b, -1, self.heads, self.width // self.heads).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split(q), split(k), split(v), attn_mask=allowed[:, None], dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(b, frames, self.width)
        residual = self.out(attended).to(audio.dtype)
        return audio + residual * active[:, :, None].to(audio.dtype)
