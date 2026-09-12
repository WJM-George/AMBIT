"""Frozen native Qwen text features; no pretrained audio/CLAP teacher."""
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


class FrozenCLAP44TextFeatures(nn.Module):
    def __init__(self, model_path: str, *, hidden_dim: int = 1024, max_tokens: int = 1024, batch_size: int = 32):
        super().__init__()
        from stable_audio_tools.models.conditioners import QwenTextConditioner
        self.model_path = str(Path(model_path).resolve(strict=True))
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("text feature batch size must be positive")
        self.conditioner = QwenTextConditioner(
            output_dim=hidden_dim, hidden_dim=hidden_dim, model_path=self.model_path,
            max_length=max_tokens, enable_grad=False, project_out=False,
            fail_on_truncation=True, padding_mode="zero",
        ).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        self.conditioner.eval()
        return self

    @torch.no_grad()
    def forward(self, texts: Sequence[str], device: torch.device | str):
        if not texts or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError("CLAP44 text supervision must be nonempty")
        pooled = []
        for start in range(0, len(texts), self.batch_size):
            tokens, mask = self.conditioner(list(texts[start:start + self.batch_size]), device)
            mask = mask.to(device=tokens.device, dtype=torch.bool)
            if not bool(mask.any(1).all()):
                raise ValueError("empty CLAP44 text tokenization")
            positions = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
            last = positions.masked_fill(~mask, -1).max(1).values
            # The last valid causal state has seen the complete description.
            pooled.append(tokens[torch.arange(tokens.shape[0], device=tokens.device), last].float())
        return torch.cat(pooled, 0)
