"""Exact-text residual at existing Qwen token positions; no audio alignment.

UTF-8 bytes preserve every character without an English-only vocabulary. Each
speech source is encoded independently, then pooled over the tokenizer's exact
character offsets. Zero output projection preserves the native attention
normalizer, sequence length, source roles and initial renderer function.
"""

from functools import lru_cache
import math

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, width * 4)
        self.contract = nn.Linear(width * 4, width)
        self.scale = nn.Parameter(torch.full((width,), 1e-3))

    def forward(self, value, mask):
        hidden = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        hidden = self.contract(torch.nn.functional.gelu(self.expand(self.norm(hidden))))
        return (value + hidden * self.scale) * mask[..., None]


class TranscriptAdapter(nn.Module):
    def __init__(self, output_dim=768, width=256, blocks=4, max_bytes=4096):
        super().__init__()
        self.width, self.max_bytes = width, max_bytes
        self.byte_embedding = nn.Embedding(257, width, padding_idx=0)
        self.source_embedding = nn.Embedding(4, width)
        self.blocks = nn.ModuleList([ConvBlock(width) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, output_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    @staticmethod
    @lru_cache(maxsize=4096)
    def token_offsets(tokenizer, text, max_length):
        encoded = tokenizer(
            text,
            truncation=False,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        if len(encoded["input_ids"]) != max_length:
            raise ValueError("Adapter refuses truncated transcript conditioning")
        return encoded["input_ids"], encoded["offset_mapping"]

    def prepare(self, metadata, tokenizer, tokens):
        from ..sceneplan.schema import compile_model_semantic_caption_v2

        sequences, slots, locations, byte_positions, destinations, weights = [], [], [], [], [], []
        for batch_index, row in enumerate(metadata):
            prompt = row["prompt"]
            # Check before accessing a plan: unconditional inference and caption
            # dropout must have no lexical information, even if GT metadata exists.
            if prompt.get("cfg_unknown", False):
                continue
            caption = compile_model_semantic_caption_v2(row["model_sceneplan"])
            if caption["text"] != row["prompt_text"]:
                raise ValueError("Adapter and semantic prompt refer to different plans")
            if not caption["speech_regions"]:
                continue
            ids, offsets = self.token_offsets(tokenizer, caption["text"], len(prompt["input_ids"]))
            if not torch.equal(torch.as_tensor(ids), prompt["input_ids"].detach().cpu().long()):
                raise ValueError("Adapter token positions differ from the native Qwen input")
            roles = prompt["speech_source_ids"].detach().cpu().tolist()
            valid = prompt["attention_mask"].detach().cpu().tolist()
            for region in caption["speech_regions"]:
                start, end, slot = region["start"], region["end"], region["source_slot"]
                text = caption["text"][start:end]
                raw = text.encode("utf-8")
                if not raw or len(raw) > self.max_bytes:
                    raise ValueError("Empty or oversized transcript: never silently truncate")
                index = len(sequences)
                sequences.append([b + 1 for b in raw])
                slots.append(slot)
                byte_offsets = [0]
                for character in text:
                    byte_offsets.append(byte_offsets[-1] + len(character.encode("utf-8")))
                covered = set()
                for token_index, ((a, b), role, is_valid) in enumerate(zip(offsets, roles, valid)):
                    left, right = max(a, start), min(b, end)
                    if not is_valid or left >= right:
                        continue
                    if token_index >= tokens or role != slot + 1:
                        raise ValueError("Transcript token lost its source role or was truncated")
                    lo, hi = byte_offsets[left - start], byte_offsets[right - start]
                    for byte in range(lo, hi):
                        locations.append(index)
                        byte_positions.append(byte)
                        destinations.append(batch_index * tokens + token_index)
                        weights.append(1.0 / (hi - lo))
                        covered.add(byte)
                if len(covered) != len(raw):
                    raise ValueError("Tokenizer offsets do not cover the complete transcript")
        return sequences, slots, locations, byte_positions, destinations, weights

    def forward(self, context, metadata, tokenizer):
        batch, tokens, _ = context.shape
        sequences, slots, rows, columns, destinations, weights = self.prepare(
            metadata, tokenizer, tokens
        )
        if not sequences:
            return context
        hidden = self.encode(sequences, slots, context.device)
        return self.pool(context, hidden, rows, columns, destinations, weights)

    def encode(self, sequences, slots, device):
        length = max(map(len, sequences))
        ids = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
        # One transfer avoids a GPU synchronization for every character.
        packed = [s + [0] * (length - len(s)) for s in sequences]
        ids.copy_(torch.tensor(packed, dtype=torch.long, device=device))
        mask = ids.ne(0)
        position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequency = torch.exp(
            torch.arange(0, self.width, 2, device=device).float()
            * (-math.log(10000.0) / self.width)
        )
        sinusoid = torch.stack(
            (torch.sin(position * frequency), torch.cos(position * frequency)), -1
        ).flatten(-2)
        hidden = self.byte_embedding(ids) + sinusoid[None]
        hidden = (
            hidden + self.source_embedding(torch.tensor(slots, device=device))[:, None]
        ) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.norm(hidden)

    def pool(self, context, hidden, rows, columns, destinations, weights):
        batch, tokens, _ = context.shape
        device = context.device
        selected = hidden[torch.tensor(rows, device=device), torch.tensor(columns, device=device)]
        selected = selected * torch.tensor(weights, device=device, dtype=selected.dtype)[:, None]
        pooled = torch.zeros(batch * tokens, self.width, device=device, dtype=selected.dtype)
        pooled.index_add_(0, torch.tensor(destinations, device=device), selected)
        residual = self.out(pooled).reshape(batch, tokens, -1)
        return context + residual.to(context.dtype)
