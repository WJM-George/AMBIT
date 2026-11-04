import torch
from torch import nn
from ..sceneplan.schema import MAX_SOURCES, MAX_DURATION_SEC

MAX_KEYFRAMES = 8


class AudioEventReadout(nn.Module):
    """Learned source queries see only the unpooled CLAP audio sequence."""

    def __init__(self, *, audio_width=384, width=192, heads=6, layers=2, text_dim=1024):
        super().__init__()
        self.input_projection = nn.Linear(audio_width, width)
        self.queries = nn.Parameter(torch.randn(MAX_SOURCES, width) * 0.02)
        layer = nn.TransformerDecoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))
        sizes = {
            "kind_logits": 4,
            "motion_logits": 3,
            "frame_count_logits": MAX_KEYFRAMES,
            "content": text_dim,
            "activity_sec": 2,
            "gain_db": 1,
            "keyframe_time_sec": MAX_KEYFRAMES,
            "direction_xyz": 3 * MAX_KEYFRAMES,
            "log_distance": MAX_KEYFRAMES,
        }
        self.output_heads = nn.ModuleDict(
            {key: nn.Linear(width, size) for (key, size) in sizes.items()}
        )

    def forward(self, sequence, sequence_mask):
        if (
            sequence.ndim != 3
            or sequence_mask.shape != sequence.shape[:2]
            or sequence_mask.dtype != torch.bool
        ):
            raise ValueError("Event readout requires masked audio sequence features")
        if not bool(sequence_mask.any(-1).all()) or not bool(torch.isfinite(sequence).all()):
            raise ValueError("Missing or invalid audio sequence")
        memory = self.input_projection(sequence)
        hidden = self.decoder(
            self.queries[None].expand(sequence.shape[0], -1, -1),
            memory,
            memory_key_padding_mask=~sequence_mask,
        )
        out = {key: head(hidden).float() for (key, head) in self.output_heads.items()}
        out["activity_sec"] = out["activity_sec"].sigmoid() * MAX_DURATION_SEC
        out["keyframe_time_sec"] = out["keyframe_time_sec"].sigmoid() * MAX_DURATION_SEC
        out["gain_db"] = out["gain_db"].squeeze(-1).sigmoid() * 36 - 24
        out["direction_xyz"] = out["direction_xyz"].reshape(
            sequence.shape[0], MAX_SOURCES, MAX_KEYFRAMES, 3
        )
        return out
