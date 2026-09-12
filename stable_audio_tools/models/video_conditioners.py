"""Video conditioning modules.

Two complementary streams for (V|VT) -> spatial audio:
  * CLIPConditioner / CLIPWithSync... — SEMANTIC stream (frame-level CLIP, language-aligned).
  * VideoMAEv2Conditioner             — MOTION stream (precomputed VideoMAE-v2 spatiotemporal
    features). Optional; coexists with CLIP (list both in cross_attention_cond_ids). This
    mirrors InternVideo-style "CLIP semantic + MAE motion" dual encoding.

Note: each conditioner returns (tokens[B,N,D], mask[B,N]); the DiT concatenates tokens AND
masks over the sequence dim, so a conditioner's mask width MUST equal its token count.
"""

import typing as tp

import einops
import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from .temporal_self_attention import SA_Transformer
from .conditioners import Conditioner


class CLIPConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
        self,
        output_dim: int,
        clip_model_name: str = "clip-vit-base-patch32",
        video_fps: int = 5,
        out_features: int = 128,
        enable_grad: bool = False,
        in_features: int = 5000,
        project_out: bool = False,
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        sa_depth = 4
        num_heads = 16
        dim_head = 64
        hidden_scale = 4
        duration = 10

        self.clip_model_name = clip_model_name

        if self.clip_model_name == "clip-vit-base-patch32":
            out_features = 128
            temporal_dim = 768

            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)

            in_features = 50 * video_fps * duration

            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.proj = nn.Linear(in_features=in_features, out_features=out_features)

            self.in_features = in_features
            self.out_features = out_features

            self.Temp_transformer = SA_Transformer(
                temporal_dim, sa_depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.0
            )
            self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration * video_fps, temporal_dim))

            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            self.preprocess_CLIP = transforms.Compose([transforms.Normalize(mean=clip_mean, std=clip_std)])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def forward(
        self, Video_tensors: tp.List[torch.Tensor], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)

        if isinstance(Video_tensors[0], dict):
            Video_tensors = [item["video_tensors"] for item in Video_tensors]

        original_videos = torch.cat(Video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))
        Video_tensors = einops.rearrange(original_videos, "b t c h w -> (b t) c h w")

        video_cond_pixel_values = self.process_video_with_custom_preprocessing(
            video_tensor=Video_tensors.to(device)
        ).to(device)

        if self.clip_model_name == "clip-vit-base-patch32":
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            video_hidden = einops.rearrange(
                video_hidden, "(b t) q h -> (b q) t h", b=batch_size, t=time_length
            )
            video_hidden += self.Temp_pos_embedding
            video_hidden = self.Temp_transformer(video_hidden)
            video_hidden = einops.rearrange(
                video_hidden, "(b q) t h -> b (t q) h", b=batch_size, t=time_length
            )

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)

        # Mask must match token count (cross-attn concatenates masks over the seq dim).
        return video_hidden, torch.ones(video_hidden.shape[0], video_hidden.shape[1], device=device)


class CLIPWithSyncWithEmptyFeatureConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
        self,
        output_dim: int,
        clip_model_name: str = "clip-vit-base-patch32",
        video_fps: int = 5,
        out_features: int = 128,
        enable_grad: bool = False,
        in_features: int = 5000,
        project_out: bool = False,
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        sa_depth = 4
        num_heads = 16
        dim_head = 64
        hidden_scale = 4
        duration = 10

        self.clip_model_name = clip_model_name

        if self.clip_model_name == "clip-vit-base-patch32":
            in_features = 50 * video_fps * duration
            temporal_dim = 768

            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)

            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            self.proj_sync = nn.Linear(in_features=240, out_features=out_features)
            self.sync_weight = nn.Parameter(torch.tensor(0.0))

            self.in_features = in_features
            self.out_features = out_features

            self.Temp_transformer = SA_Transformer(
                temporal_dim, sa_depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.0
            )
            self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration * video_fps, temporal_dim))

            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            self.preprocess_CLIP = transforms.Compose([transforms.Normalize(mean=clip_mean, std=clip_std)])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def forward(
        self, Video_list: tp.List[tp.Union[torch.Tensor, tp.Dict]], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(Video_list[0], dict):
            Video_tensors = [item["video_tensors"] for item in Video_list]
            video_sync_frames = torch.cat([item["video_sync_frames"] for item in Video_list], dim=0).to(device)
        else:
            Video_tensors = Video_list
            batch_size = len(Video_tensors)
            video_sync_frames = torch.zeros(batch_size, 240, 768).to(device)

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)

        original_videos = torch.cat(Video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))

        Video_tensors = einops.rearrange(original_videos, "b t c h w -> (b t) c h w")
        video_cond_pixel_values = self.process_video_with_custom_preprocessing(
            video_tensor=Video_tensors.to(device)
        ).to(device)

        if self.clip_model_name == "clip-vit-base-patch32":
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            video_hidden = einops.rearrange(
                video_hidden, "(b t) q h -> (b q) t h", b=batch_size, t=time_length
            )
            video_hidden += self.Temp_pos_embedding
            video_hidden = self.Temp_transformer(video_hidden)
            video_hidden = einops.rearrange(
                video_hidden, "(b q) t h -> b (t q) h", b=batch_size, t=time_length
            )

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        video_sync_frames = self.proj_sync(video_sync_frames.view(-1, 240))
        video_sync_frames = video_sync_frames.view(batch_size, self.out_features, -1)

        video_hidden = video_hidden + self.sync_weight * video_sync_frames
        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)

        # Mask must match token count (cross-attn concatenates masks over the seq dim).
        return video_hidden, torch.ones(video_hidden.shape[0], video_hidden.shape[1], device=device)


class VideoMAEv2Conditioner(Conditioner):
    """MOTION stream: adapt PRECOMPUTED VideoMAE-v2 features into cross-attention tokens.

    Complements CLIPConditioner (semantic). List both in ``cross_attention_cond_ids`` to get
    the InternVideo-style "CLIP semantic + MAE motion" dual conditioning.

    The 1B-scale VideoMAE-v2 backbone is NOT run here (same offline pattern as Synchformer
    sync frames): features are precomputed by ``dataset/features/extract_videomae_features.py`` and
    passed in per-sample as a tensor ``[T_feat, feat_dim]`` (or a dict with key
    ``videomae_feats`` / ``video_motion_feats``). Missing video -> learned empty token (CFG).

    Args:
        output_dim: cross-attn token dim (set by the conditioning ``cond_dim``).
        feat_dim: VideoMAE feature width (vit_g=1408, vit_l=1024, vit_b=768).
        max_tokens: pad/crop the per-clip feature sequence to this many motion tokens.
        sa_depth/num_heads/dim_head/hidden_scale: small temporal transformer over motion tokens.
    """

    def __init__(
        self,
        output_dim: int,
        feat_dim: int = 1408,
        max_tokens: int = 64,
        sa_depth: int = 2,
        num_heads: int = 8,
        dim_head: int = 64,
        hidden_scale: int = 4,
        project_out: bool = False,
    ):
        super().__init__(dim=output_dim, output_dim=output_dim, project_out=project_out)
        self.feat_dim = feat_dim
        self.max_tokens = max_tokens
        self.in_proj = nn.Linear(feat_dim, output_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, output_dim) * 0.02)
        self.temporal = SA_Transformer(output_dim, sa_depth, num_heads, dim_head, output_dim * hidden_scale, 0.0)
        self.empty_feat = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.constant_(self.empty_feat, 0)

    def _to_feat(self, item, device):
        if item is None:
            return None
        if isinstance(item, dict):
            item = item.get("videomae_feats", item.get("video_motion_feats", None))
            if item is None:
                return None
        if not torch.is_tensor(item):
            item = torch.as_tensor(item)
        item = item.to(device).float()
        if item.ndim == 1:
            item = item.unsqueeze(0)
        if item.ndim != 2 or item.shape[-1] != self.feat_dim:
            return None
        return item  # [T_feat, feat_dim]

    def forward(self, inputs: tp.List[tp.Any], device: tp.Union[torch.device, str]):
        B, N = len(inputs), self.max_tokens
        feats = torch.zeros(B, N, self.feat_dim, device=device)
        mask = torch.zeros(B, N, device=device)
        present = torch.zeros(B, dtype=torch.bool, device=device)

        for i, item in enumerate(inputs):
            t = self._to_feat(item, device)
            if t is None or t.numel() == 0 or bool(torch.all(t == 0)):
                continue
            present[i] = True
            k = min(t.shape[0], N)
            feats[i, :k] = t[:k]
            mask[i, :k] = 1.0

        x = self.in_proj(feats) + self.pos_embedding[:, :N]
        x = self.temporal(x)

        # Samples with no video -> single learned empty token (CFG / missing modality).
        empty = self.empty_feat.expand(B, N, -1)
        x = torch.where(present.view(B, 1, 1), x, empty)
        miss = ~present
        if miss.any():
            mask[miss, 0] = 1.0

        return x, mask


class AlignedVideoMAEv2Conditioner(Conditioner):
    """Adapt precomputed VideoMAE-v2 features into time-aligned DiT input channels.

    This is for V2A when video should enter the main DiT token stream through
    self-attention rather than cross-attention. The offline VideoMAE sequence is
    first processed by a small temporal transformer, then explicitly expanded to
    the VAE latent time axis. For Sphere360 dynamic10 this maps 5 VideoMAE tokens
    onto 431 latent steps, so each latent step receives the corresponding video
    condition channel via input_concat.
    """

    def __init__(
        self,
        output_dim: int,
        feat_dim: int = 768,
        max_tokens: int = 64,
        target_length: int = 431,
        sa_depth: int = 2,
        num_heads: int = 8,
        dim_head: int = 64,
        hidden_scale: int = 4,
        align_mode: str = "repeat",
        project_out: bool = False,
    ):
        super().__init__(dim=output_dim, output_dim=output_dim, project_out=project_out)
        if align_mode not in {"repeat", "nearest", "linear"}:
            raise ValueError(f"Unknown align_mode '{align_mode}', expected repeat/nearest/linear")

        self.feat_dim = feat_dim
        self.max_tokens = max_tokens
        self.target_length = target_length
        self.align_mode = align_mode

        self.in_proj = nn.Linear(feat_dim, output_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, output_dim) * 0.02)
        self.temporal = SA_Transformer(output_dim, sa_depth, num_heads, dim_head, output_dim * hidden_scale, 0.0)
        self.empty_feat = nn.Parameter(torch.zeros(1, output_dim, 1))
        nn.init.constant_(self.empty_feat, 0)

    def _unwrap(self, item):
        latent_length = None
        if isinstance(item, dict):
            latent_length = item.get("latent_length", item.get("target_length", None))
            item = item.get("videomae_feats", item.get("video_motion_feats", item.get("video_motion", None)))
        return item, latent_length

    def _to_feat(self, item, device):
        item, latent_length = self._unwrap(item)
        if item is None:
            return None, latent_length
        if not torch.is_tensor(item):
            item = torch.as_tensor(item)
        item = item.to(device).float()
        if item.ndim == 1:
            item = item.unsqueeze(0)
        if item.ndim != 2 or item.shape[-1] != self.feat_dim:
            return None, latent_length
        return item, latent_length

    def _expand_to_latent_axis(self, x: torch.Tensor, mask: torch.Tensor, target_lengths: tp.List[int]):
        B, T, C = x.shape
        max_target = max(int(t) for t in target_lengths)
        aligned = x.new_zeros(B, C, max_target)
        aligned_mask = mask.new_zeros(B, max_target)

        for i, target in enumerate(target_lengths):
            target = int(target)
            valid_tokens = int(mask[i].sum().item())
            if target <= 0:
                continue
            if valid_tokens <= 0:
                aligned[i, :, :target] = self.empty_feat[0].expand(C, target)
                continue

            seq = x[i, :valid_tokens].transpose(0, 1).unsqueeze(0)  # [1, C, T_video]
            if self.align_mode in {"repeat", "nearest"}:
                # Nearest expansion is an explicit frame/token copy onto the VAE latent axis.
                idx = torch.div(
                    torch.arange(target, device=x.device) * valid_tokens,
                    target,
                    rounding_mode="floor",
                ).clamp(max=valid_tokens - 1)
                expanded = seq[0, :, idx]
            else:
                expanded = torch.nn.functional.interpolate(seq, size=target, mode="linear", align_corners=False)[0]

            aligned[i, :, :target] = expanded
            aligned_mask[i, :target] = 1.0

        return aligned, aligned_mask

    def forward(self, inputs: tp.List[tp.Any], device: tp.Union[torch.device, str]):
        B, N = len(inputs), self.max_tokens
        feats = torch.zeros(B, N, self.feat_dim, device=device)
        mask = torch.zeros(B, N, device=device)
        present = torch.zeros(B, dtype=torch.bool, device=device)
        target_lengths = []

        for i, item in enumerate(inputs):
            t, latent_length = self._to_feat(item, device)
            target_lengths.append(int(latent_length or self.target_length))
            if t is None or t.numel() == 0 or bool(torch.all(t == 0)):
                continue

            present[i] = True
            k = min(t.shape[0], N)
            feats[i, :k] = t[:k]
            mask[i, :k] = 1.0

        x = self.in_proj(feats) + self.pos_embedding[:, :N]
        x = self.temporal(x)

        empty_tokens = self.empty_feat.transpose(1, 2).expand(B, N, -1)
        x = torch.where(present.view(B, 1, 1), x, empty_tokens)
        miss = ~present
        if miss.any():
            mask[miss, 0] = 1.0

        # input_concat conditioning expects [B, channels, latent_steps].
        return self._expand_to_latent_axis(x, mask, target_lengths)


class SpatialFormatConditioner(Conditioner):
    """Maps spatial output format strings to embed tokens (foa / binaural / bin)."""

    FORMAT_TO_ID = {"foa": 0, "binaural": 1, "bin": 1, "stereo": 1, "2ch": 1, "4ch": 0}

    def __init__(self, output_dim: int):
        super().__init__(output_dim, output_dim)
        # Several string aliases map to the same semantic class; embedding rows
        # should follow the ids, not the number of aliases.
        self.embed = nn.Embedding(max(self.FORMAT_TO_ID.values()) + 1, output_dim)

    def forward(self, formats: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        ids = []
        for f in formats:
            key = str(f).lower().strip()
            if key not in self.FORMAT_TO_ID:
                raise ValueError(f"Unknown spatial format '{f}', expected one of {list(self.FORMAT_TO_ID)}")
            ids.append(self.FORMAT_TO_ID[key])
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        emb = self.embed(ids_t).unsqueeze(1)
        mask = torch.ones(emb.shape[0], 1, device=device)
        return emb, mask
