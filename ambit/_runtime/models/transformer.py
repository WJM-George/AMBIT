from .utils import compile
from functools import reduce
import logging
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from typing import Callable, Literal, Optional

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    flex_attention_available = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    flex_attention_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print("flash_attn not installed, disabling Flash Attention")
    flash_attn_kvpacked_func = None
    flash_attn_func = None
try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
except ImportError as e:
    print(e)
    print("flash_attn varlen/bert_padding not available, disabling varlen attention")
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    index_first_axis = None


def precompute_varlen_metadata(padding_mask: torch.Tensor):
    """
    Precompute varlen attention metadata once to avoid recomputation in every attention layer.

    Args:
        padding_mask: Boolean tensor of shape (batch, seq_len) where True = valid

    Returns:
        Dict with cu_seqlens, max_seqlen, indices, batch_size, seq_len for use in attention
    """
    if padding_mask is None or unpad_input is None:
        return None
    (batch_size, seq_len) = padding_mask.shape
    seqlens = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    return {
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "indices": indices,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def _left_pad_to_match(emb, target_len):
    """Left-pad or right-trim emb along seq dim to match target_len.

    Used for local conditioning embeddings that need to align with x
    without affecting prepended tokens (memory tokens, global cond, etc.).
    """
    emb_len = emb.shape[-2]
    if emb_len < target_len:
        return F.pad(emb, (0, 0, target_len - emb_len, 0), value=0.0)
    elif emb_len > target_len:
        return emb[:, -target_len:, :]
    return emb


if flex_attention_available:
    try:
        torch._dynamo.config.cache_size_limit = 5000
        flex_attention_compiled = torch.compile(
            flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
        )
    except Exception as e:
        logging.debug(f"Could not compile flex_attention, using uncompiled version: {e}")
        flex_attention_compiled = flex_attention
else:
    flex_attention_compiled = None
_SLIDING_WINDOW_BLOCK_MASK_CACHE = {}


def _get_sliding_window_block_mask(seq_q, seq_k, w_left, w_right, device):
    key = (seq_q, seq_k, int(w_left), int(w_right), str(device))
    bm = _SLIDING_WINDOW_BLOCK_MASK_CACHE.get(key)
    if bm is None:
        (wl, wr) = (int(w_left), int(w_right))

        def _band_mod(b, h, q_idx, kv_idx):
            delta = kv_idx - q_idx
            return (delta >= -wl) & (delta <= wr)

        bm = create_block_mask(_band_mod, B=None, H=None, Q_LEN=seq_q, KV_LEN=seq_k, device=device)
        _SLIDING_WINDOW_BLOCK_MASK_CACHE[key] = bm
    return bm


def _sliding_window_additive_mask(seq_q, seq_k, w_left, w_right, device, dtype):
    """Build a (seq_q, seq_k) additive mask for masked SDPA fallback.
    0 inside the band [i - w_left, i + w_right], -inf outside.
    """
    ii = torch.arange(seq_q, device=device)
    jj = torch.arange(seq_k, device=device)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
    mask = torch.zeros((seq_q, seq_k), dtype=dtype, device=device)
    return mask.masked_fill(~in_band, float("-inf"))


_SLIDING_WINDOW_CHUNK_SIZE = 1024


def _sliding_window_chunked_halo_sdpa(
    q, k, v, w_left, w_right, chunk_size=_SLIDING_WINDOW_CHUNK_SIZE
):
    (B, H, N, D) = q.shape
    outs = []
    for q_start in range(0, N, chunk_size):
        q_end = min(q_start + chunk_size, N)
        k_start = max(0, q_start - int(w_left))
        k_end = min(N, q_end + int(w_right))
        q_c = q[..., q_start:q_end, :]
        k_c = k[..., k_start:k_end, :]
        v_c = v[..., k_start:k_end, :]
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        delta = k_idx[None, :] - q_idx[:, None]
        in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
        mask = torch.zeros(delta.shape, dtype=q.dtype, device=q.device).masked_fill(
            ~in_band, float("-inf")
        )
        outs.append(F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=mask, is_causal=False))
    return torch.cat(outs, dim=-2)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** (-0.5)
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        (seq_len, device) = (x.shape[1], x.device)
        assert seq_len <= self.max_seq_len, (
            f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"
        )
        if pos is None:
            pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min=0)
        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0, "dimension must be divisible by 2"
        self.scale = nn.Parameter(torch.ones(1) * dim ** (-0.5))
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** (-freq_seq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        (seq_len, device) = (x.shape[1], x.device)
        if pos is None:
            pos = torch.arange(seq_len, device=device)
        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]
        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq)
        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor
        if not use_xpos:
            self.register_buffer("scale", None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(self, t):
        device = self.inv_freq.device
        t = t.to(torch.float32)
        t = t / self.interpolation_factor
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if self.scale is None:
            return (freqs, 1.0)
        power = (torch.arange(t.shape[0], device=device) - t.shape[0] // 2) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)
        return (freqs, scale)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    (x1, x2) = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    out_dtype = t.dtype
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    (rot_dim, seq_len) = (freqs.shape[-1], t.shape[-2])
    (freqs, t) = (freqs.to(dtype), t.to(dtype))
    freqs = freqs[-seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    (t, t_unrotated) = (t[..., :rot_dim], t[..., rot_dim:])
    t = t * freqs.cos() * scale + rotate_half(t) * freqs.sin() * scale
    (t, t_unrotated) = (t.to(out_dtype), t_unrotated.to(out_dtype))
    return torch.cat((t, t_unrotated), dim=-1)


class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-05):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))
        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        else:
            output = F.layer_norm(
                x.float(),
                x.shape[-1:],
                weight=self.gamma.float(),
                bias=self.beta.float(),
                eps=self.eps,
            )
            return output.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, fix_scale=False, force_fp32=False, eps=1e-05):
        super().__init__()
        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.rms_norm(x, x.shape[-1:], weight=self.gamma, eps=self.eps)
        else:
            output = F.rms_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), eps=self.eps)
            return output.to(x.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_val=1e-05):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))

    def forward(self, x):
        return x * self.scale


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation: Callable, use_conv=False, conv_kernel_size=3):
        super().__init__()
        self.act = activation
        self.proj = (
            nn.Linear(dim_in, dim_out * 2)
            if not use_conv
            else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding=conv_kernel_size // 2)
        )
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, "b n d -> b d n")
            x = self.proj(x)
            x = rearrange(x, "b d n -> b n d")
        else:
            x = self.proj(x)
        (x, gate) = x.chunk(2, dim=-1)
        return x * self.act(gate)


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(3.14159265359 * x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        no_bias=False,
        glu=True,
        use_conv=False,
        conv_kernel_size=3,
        zero_init_output=True,
        sinusoidal=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        activation = nn.SiLU() if not sinusoidal else Sin()
        dim_out = dim if dim_out is None else dim_out
        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias=not no_bias)
                if not use_conv
                else nn.Conv1d(
                    dim,
                    inner_dim,
                    conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    bias=not no_bias,
                ),
                Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
                activation,
            )
        linear_out = (
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
            if not use_conv
            else nn.Conv1d(
                inner_dim,
                dim_out,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
                bias=not no_bias,
            )
        )
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)
        self.ff = nn.Sequential(
            linear_in,
            Rearrange("b d n -> b n d") if use_conv else nn.Identity(),
            linear_out,
            Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
        )

    def forward(self, x, varlen_metadata=None):
        if varlen_metadata is not None and index_first_axis is not None and (pad_input is not None):
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            indices = varlen_metadata["indices"]
            dim = x.shape[-1]
            x_packed = index_first_axis(x.reshape(-1, dim), indices)
            x_packed = self.ff(x_packed.unsqueeze(0)).squeeze(0)
            return pad_input(x_packed, indices, batch_size, seq_len)
        else:
            return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        dim_context=None,
        causal=False,
        zero_init_output=True,
        qk_norm_eps=1e-06,
        qk_norm: Literal["l2", "ln", "rms", "dyt", "none"] = "none",
        differential=False,
        feat_scale=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.differential = differential
        dim_kv = dim_context if dim_context is not None else dim
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads
        if dim_context is not None:
            if differential:
                self.to_q = nn.Linear(dim, dim * 2, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
            else:
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        elif differential:
            self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
        if qk_norm not in ["l2", "ln", "rms", "dyt", "none"]:
            raise ValueError(
                f'qk_norm must be one of ["l2", "ln", "rms" ,"dyt", "none"], got {qk_norm}'
            )
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif self.qk_norm == "dyt":
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)
        self.feat_scale = feat_scale
        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))
        self.causal = causal
        self._attention_bias_activated = False

    @compile
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return (q, k)

    def apply_attn(
        self,
        q,
        k,
        v,
        causal=None,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
        mask_padding_logits=False,
        attention_bias=None,
    ):
        if self.num_heads != self.kv_heads:
            heads_per_kv_head = self.num_heads // self.kv_heads
            (k, v) = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))
        if attention_bias is not None:
            expected_tail = (q.shape[-2], k.shape[-2])
            if (
                attention_bias.ndim != 4
                or int(attention_bias.shape[0]) != int(q.shape[0])
                or int(attention_bias.shape[1]) not in {1, int(q.shape[1])}
                or (tuple(attention_bias.shape[-2:]) != expected_tail)
            ):
                raise ValueError(
                    f"attention_bias must match [batch,1|heads,query,key] with batch={q.shape[0]}, heads={q.shape[1]}, query/key={expected_tail}; got {tuple(attention_bias.shape)}"
                )
            attn_mask = attention_bias.to(device=q.device, dtype=q.dtype)
            if mask_padding_logits and padding_mask is not None:
                key_mask = padding_mask.to(device=q.device, dtype=torch.bool)
                if key_mask.ndim != 2 or key_mask.shape != (q.shape[0], k.shape[-2]):
                    raise ValueError(
                        f"attention-bias key mask must be [batch,key_length], got {tuple(key_mask.shape)}"
                    )
                attn_mask = attn_mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            if causal:
                causal_mask = torch.ones(
                    q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool
                ).tril(diagonal=k.shape[-2] - q.shape[-2])
                attn_mask = attn_mask.masked_fill(~causal_mask[None, None], float("-inf"))
            bias_is_only_gradient_path = attention_bias.requires_grad and (
                not (q.requires_grad or k.requires_grad or v.requires_grad)
            )
            if bias_is_only_gradient_path and q.is_cuda:
                if sdpa_kernel is not None and SDPBackend is not None:
                    with sdpa_kernel([SDPBackend.MATH]):
                        biased_out = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask, is_causal=False
                        )
                else:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False, enable_math=True
                    ):
                        biased_out = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask, is_causal=False
                        )
            else:
                biased_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False
                )
            if not self._attention_bias_activated:
                has_nonzero_bias = bool(torch.count_nonzero(attention_bias.detach()).item())
                if has_nonzero_bias:
                    self._attention_bias_activated = True
                else:
                    if mask_padding_logits and padding_mask is not None:
                        key_mask = padding_mask.to(device=q.device, dtype=torch.bool)
                        baseline_mask = key_mask[:, None, None, :].expand(-1, 1, q.shape[-2], -1)
                        if causal:
                            causal_mask = torch.ones(
                                q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool
                            ).tril(diagonal=k.shape[-2] - q.shape[-2])
                            baseline_mask = baseline_mask & causal_mask[None, None, :, :]
                        baseline_out = F.scaled_dot_product_attention(
                            q, k, v, attn_mask=baseline_mask, is_causal=False
                        )
                    else:
                        baseline_out = F.scaled_dot_product_attention(
                            q, k, v, is_causal=causal if causal is not None else False
                        )
                    return baseline_out.detach() + (biased_out - biased_out.detach())
            return biased_out
        if mask_padding_logits and padding_mask is not None:
            key_mask = padding_mask.to(device=q.device, dtype=torch.bool)
            if key_mask.ndim != 2 or key_mask.shape != (q.shape[0], k.shape[-2]):
                raise ValueError(
                    f"cross-attention mask must be [batch, context_length], got {tuple(key_mask.shape)} for keys {tuple(k.shape)}"
                )
            attn_mask = key_mask[:, None, None, :].expand(-1, 1, q.shape[-2], -1)
            if causal:
                causal_mask = torch.ones(
                    q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool
                ).tril(diagonal=k.shape[-2] - q.shape[-2])
                attn_mask = attn_mask & causal_mask[None, None, :, :]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        flash_attn_available = flash_attn_func is not None and q.is_cuda
        flash_attn_varlen_available = (
            q.is_cuda and flash_attn_varlen_func is not None and (index_first_axis is not None)
        )
        if causal and (
            flex_attention_block_mask is not None or flex_attention_score_mod is not None
        ):
            flex_attention_block_mask = None
            flex_attention_score_mod = None
        if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            out = flex_attention_compiled(
                q, k, v, block_mask=flex_attention_block_mask, score_mod=flex_attention_score_mod
            )
        elif flash_attn_available and varlen_metadata is not None and flash_attn_varlen_available:
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            cu_seqlens = varlen_metadata["cu_seqlens"]
            max_seqlen = varlen_metadata["max_seqlen"]
            indices = varlen_metadata["indices"]
            fa_dtype_in = q.dtype
            (q, k, v) = map(lambda t: rearrange(t, "b h n d -> b n h d"), (q, k, v))
            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                (q, k, v) = map(lambda t: t.to(torch.float16), (q, k, v))
            (num_heads, head_dim) = (q.shape[2], q.shape[3])
            q_unpad = index_first_axis(q.reshape(-1, num_heads, head_dim), indices)
            k_unpad = index_first_axis(k.reshape(-1, num_heads, head_dim), indices)
            v_unpad = index_first_axis(v.reshape(-1, num_heads, head_dim), indices)
            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=causal if causal is not None else False,
                window_size=flash_attn_sliding_window
                if flash_attn_sliding_window is not None
                else (-1, -1),
            )
            out = pad_input(out_unpad, indices, batch_size, seq_len)
            out = rearrange(out.to(fa_dtype_in), "b n h d -> b h n d")
        elif flash_attn_available:
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            fa_dtype_in = q.dtype
            (q, k, v) = map(lambda t: rearrange(t, "b h n d -> b n h d"), (q, k, v))
            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                (q, k, v) = map(lambda t: t.to(torch.float16), (q, k, v))
            out = flash_attn_func(
                q,
                k,
                v,
                causal=causal,
                window_size=flash_attn_sliding_window
                if flash_attn_sliding_window is not None
                else [-1, -1],
            )
            out = rearrange(out.to(fa_dtype_in), "b n h d -> b h n d")
        else:
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            if flash_attn_sliding_window is not None:
                (seq_q, seq_k) = (q.shape[2], k.shape[2])
                (wl, wr) = flash_attn_sliding_window
                handled = False
                if flex_attention_available and flex_attention_compiled is not None:
                    try:
                        bm = _get_sliding_window_block_mask(seq_q, seq_k, wl, wr, q.device)
                        out = flex_attention_compiled(q, k, v, block_mask=bm)
                        handled = True
                    except Exception as _flex_err:
                        logging.debug(
                            f"flex_attention failed, trying chunked-halo SDPA: {_flex_err}"
                        )
                if not handled:
                    try:
                        out = _sliding_window_chunked_halo_sdpa(q, k, v, wl, wr)
                        handled = True
                    except Exception as _chunk_err:
                        logging.debug(
                            f"chunked-halo SDPA failed, falling back to full masked SDPA: {_chunk_err}"
                        )
                if not handled:
                    add_mask = _sliding_window_additive_mask(
                        seq_q, seq_k, wl, wr, q.device, q.dtype
                    )
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=add_mask, is_causal=False
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=causal if causal is not None else False
                )
        return out

    def forward(
        self,
        x,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal=None,
        flex_attention_block_mask=None,
        flex_attention_score_mod=None,
        flash_attn_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
        attention_bias=None,
    ):
        (h, kv_h, has_context) = (self.num_heads, self.kv_heads, context is not None)
        kv_input = context if has_context else x
        if hasattr(self, "to_q"):
            if self.differential:
                (q, q_diff) = self.to_q(x).chunk(2, dim=-1)
                (q, q_diff) = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, q_diff))
                q = torch.stack([q, q_diff], dim=1)
                kv = self.to_kv(kv_input)
                (k, k_diff, v) = kv.chunk(3, dim=-1)
                (k, k_diff, v) = map(
                    lambda t: rearrange(t, "b n (h d) -> b h n d", h=kv_h), (k, k_diff, v)
                )
                k = torch.stack([k, k_diff], dim=1)
            else:
                q = self.to_q(x)
                q = rearrange(q, "b n (h d) -> b h n d", h=h)
                kv = self.to_kv(kv_input)
                (k, v) = kv.chunk(2, dim=-1)
                (k, v) = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=kv_h), (k, v))
        elif self.differential:
            (q, k, v, q_diff, k_diff) = self.to_qkv(x).chunk(5, dim=-1)
            (q, k, v, q_diff, k_diff) = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v, q_diff, k_diff)
            )
            q = torch.stack([q, q_diff], dim=1)
            k = torch.stack([k, k_diff], dim=1)
        else:
            (q, k, v) = self.to_qkv(x).chunk(3, dim=-1)
            (q, k, v) = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            (q, k) = self.apply_qk_layernorm(q, k)
        if rotary_pos_emb is not None:
            (freqs, _) = rotary_pos_emb
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
            q_freqs = freqs
            if rotary_pos_emb_k is not None:
                (k_freqs, _) = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs
                if q.shape[-2] >= k.shape[-2]:
                    ratio = q.shape[-2] / k.shape[-2]
                    (q_freqs, k_freqs) = (freqs, ratio * freqs)
                else:
                    ratio = k.shape[-2] / q.shape[-2]
                    (q_freqs, k_freqs) = (ratio * freqs, freqs)
            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        (n, _device) = (q.shape[-2], q.device)
        causal = self.causal if causal is None else causal
        if n == 1 and causal:
            causal = False
        exact_padding_logits = has_context or (
            padding_mask is not None
            and varlen_metadata is None
            and (flex_attention_block_mask is None)
            and (flex_attention_score_mod is None)
            and (flash_attn_sliding_window is None)
        )
        if self.differential:
            (q, q_diff) = q.unbind(dim=1)
            (k, k_diff) = k.unbind(dim=1)
            out = self.apply_attn(
                q,
                k,
                v,
                causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
                mask_padding_logits=exact_padding_logits,
                attention_bias=attention_bias,
            )
            out_diff = self.apply_attn(
                q_diff,
                k_diff,
                v,
                causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
                mask_padding_logits=exact_padding_logits,
                attention_bias=attention_bias,
            )
            out = out - out_diff
        else:
            out = self.apply_attn(
                q,
                k,
                v,
                causal=causal,
                flex_attention_block_mask=flex_attention_block_mask,
                flex_attention_score_mod=flex_attention_score_mod,
                flash_attn_sliding_window=flash_attn_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
                mask_padding_logits=exact_padding_logits,
                attention_bias=attention_bias,
            )
        out = rearrange(out, " b h n d -> b n (h d)")
        out = self.to_out(out)
        if self.feat_scale:
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(out.dtype)
                out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(
                    dim=-2, keepdim=True
                ).clamp(min=1)
                out_hf = out - out_dc
                out = out + (self.lambda_dc * out_dc + self.lambda_hf * out_hf) * mask
            else:
                out_dc = out.mean(dim=-2, keepdim=True)
                out_hf = out - out_dc
                out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf
        return out


class ConformerModule(nn.Module):
    def __init__(self, dim, norm_kwargs={}):
        super().__init__()
        self.dim = dim
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs)
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.pointwise_conv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.glu(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.depthwise_conv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.pointwise_conv_2(x)
        x = rearrange(x, "b d n -> b n d")
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        cross_attend=False,
        dim_context=None,
        global_cond_dim=None,
        local_add_cond_dim=None,
        modular_local_cond_configs=None,
        causal=False,
        zero_init_branch_outputs=True,
        conformer=False,
        layer_ix=-1,
        add_rope=False,
        layer_scale=False,
        norm_type="layer_norm",
        attn_kwargs={},
        ff_kwargs={},
        norm_kwargs={},
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads, dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
        if layer_scale and zero_init_branch_outputs:
            print(
                "zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False"
            )
            zero_init_branch_outputs = False
        if norm_type not in ["layer_norm", "rms_norm", "dyt"]:
            raise ValueError(
                f'norm_type must be one of ["layer_norm", "rms_norm", "dyt"], got {norm_type}'
            )
        norm_layer_map = {"layer_norm": LayerNorm, "rms_norm": RMSNorm, "dyt": DynamicTanh}
        norm_layer = norm_layer_map[norm_type]
        self.pre_norm = norm_layer(dim, **norm_kwargs)
        self.add_rope = add_rope
        self.self_attn = Attention(
            dim,
            dim_heads=self.dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )
        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = norm_layer(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads=self.dim_heads,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.layer_ix = layer_ix
        self.conformer = None
        if conformer:
            self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
            self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()
        self.global_cond_dim = global_cond_dim
        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6 * dim) / dim**0.5)
        self.local_add_cond_dim = local_add_cond_dim
        if local_add_cond_dim is not None:
            self.to_local_embed = nn.Sequential(
                nn.Linear(local_add_cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
            nn.init.zeros_(self.to_local_embed[-1].weight)
            nn.init.zeros_(self.to_local_embed[-1].bias)
        else:
            self.to_local_embed = None
        self.modular_local_cond_configs = modular_local_cond_configs or []
        self.modular_local_embeds = nn.ModuleDict()
        for config in self.modular_local_cond_configs:
            cond_id = config["id"]
            cond_dim = config["dim"]
            proj = nn.Sequential(nn.Linear(cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.modular_local_embeds[cond_id] = proj
        self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None

    def _sceneplan_cross_attention_bias(self, base_bias, event_mask, speech_mask, *, dtype):
        return base_bias

    def _feed_forward(self, hidden_states, *, varlen_metadata, context, context_mask, padding_mask):
        return (self.ff(hidden_states, varlen_metadata=varlen_metadata), None)

    def _apply_local_conditioning(self, x, local_add_cond, modular_local_cond):
        """Apply local additive and modular local conditioning to x."""
        if local_add_cond is not None and self.to_local_embed is not None:
            local_emb = self.to_local_embed(local_add_cond)
            x = x + _left_pad_to_match(local_emb, x.shape[-2])
        if modular_local_cond is not None and len(self.modular_local_embeds) > 0:
            modular_sum = None
            for cond_id, proj in self.modular_local_embeds.items():
                if cond_id in modular_local_cond:
                    local_emb = proj(modular_local_cond[cond_id])
                    local_emb = _left_pad_to_match(local_emb, x.shape[-2])
                    modular_sum = local_emb if modular_sum is None else modular_sum + local_emb
            if modular_sum is not None:
                x = x + modular_sum
        return x

    @compile
    def forward(
        self,
        x,
        context=None,
        context_mask=None,
        global_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        rotary_pos_emb=None,
        cross_attn_rotary_pos_emb=None,
        self_attention_block_mask=None,
        self_attention_score_mod=None,
        cross_attention_block_mask=None,
        cross_attention_score_mod=None,
        self_attention_bias=None,
        cross_attention_bias=None,
        self_attention_flash_sliding_window=None,
        cross_attention_flash_sliding_window=None,
        padding_mask=None,
        varlen_metadata=None,
        self_attention_causal: Optional[bool] = None,
    ):
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])
        cross_attention_bias = self._sceneplan_cross_attention_bias(
            cross_attention_bias, None, None, dtype=x.dtype
        )
        cross_attention_kwargs = {
            "context": context,
            "flex_attention_block_mask": cross_attention_block_mask,
            "flex_attention_score_mod": cross_attention_score_mod,
            "flash_attn_sliding_window": cross_attention_flash_sliding_window,
            "padding_mask": context_mask,
            "attention_bias": cross_attention_bias,
        }
        if cross_attn_rotary_pos_emb is not None:
            cross_attention_kwargs.update(
                rotary_pos_emb=rotary_pos_emb, rotary_pos_emb_k=cross_attn_rotary_pos_emb
            )
        if (
            self.global_cond_dim is not None
            and self.global_cond_dim > 0
            and (global_cond is not None)
        ):
            (scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff) = (
                (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)
            )
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(
                x,
                rotary_pos_emb=rotary_pos_emb,
                causal=self_attention_causal,
                flex_attention_block_mask=self_attention_block_mask,
                flex_attention_score_mod=self_attention_score_mod,
                flash_attn_sliding_window=self_attention_flash_sliding_window,
                padding_mask=padding_mask,
                varlen_metadata=varlen_metadata,
                attention_bias=self_attention_bias,
            )
            x = x * torch.sigmoid(1 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual
            if context is not None and self.cross_attend:
                x = x + self.cross_attn_scale(
                    self.cross_attn(self.cross_attend_norm(x), **cross_attention_kwargs)
                )
            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))
            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            (x, _) = self._feed_forward(
                x,
                varlen_metadata=varlen_metadata,
                context=context,
                context_mask=context_mask,
                padding_mask=padding_mask,
            )
            x = x * torch.sigmoid(1 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual
        else:
            x = x + self.self_attn_scale(
                self.self_attn(
                    self.pre_norm(x),
                    rotary_pos_emb=rotary_pos_emb,
                    causal=self_attention_causal,
                    flex_attention_block_mask=self_attention_block_mask,
                    flex_attention_score_mod=self_attention_score_mod,
                    flash_attn_sliding_window=self_attention_flash_sliding_window,
                    padding_mask=padding_mask,
                    varlen_metadata=varlen_metadata,
                    attention_bias=self_attention_bias,
                )
            )
            if context is not None and self.cross_attend:
                x = x + self.cross_attn_scale(
                    self.cross_attn(self.cross_attend_norm(x), **cross_attention_kwargs)
                )
            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))
            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)
            (ff_output, _) = self._feed_forward(
                self.ff_norm(x),
                varlen_metadata=varlen_metadata,
                context=context,
                context_mask=context_mask,
                padding_mask=padding_mask,
            )
            x = x + self.ff_scale(ff_output)
        return x


class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in=None,
        dim_out=None,
        dim_heads=64,
        cross_attend=False,
        cond_token_dim=None,
        final_cross_attn_ix=-1,
        global_cond_dim=None,
        local_add_cond_dim=None,
        modular_local_cond_configs=None,
        causal=False,
        rotary_pos_emb=True,
        cross_attn_rotary_pos_emb=False,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        num_memory_tokens=0,
        sliding_window=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])
        self.project_in = (
            nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        )
        self.project_out = (
            nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()
        )
        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None
        if cross_attn_rotary_pos_emb:
            self.cross_attn_rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.cross_attn_rotary_pos_emb = None
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(
                dim, abs_pos_emb_max_length + self.num_memory_tokens
            )
        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim * 6)
            )
        self.final_cross_attn_ix = final_cross_attn_ix
        self.sliding_window = sliding_window
        for config in (None, None):
            if config not in (None, {}, {"enabled": False}):
                raise ValueError("Only the dense AMBIT mainline is included in this release")
        for i in range(depth):
            should_cross_attend = cross_attend and (
                self.final_cross_attn_ix == -1 or i <= self.final_cross_attn_ix
            )
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=should_cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    local_add_cond_dim=local_add_cond_dim,
                    modular_local_cond_configs=modular_local_cond_configs,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs,
                )
            )

    def forward(
        self,
        x,
        context=None,
        context_mask: Optional[torch.Tensor] = None,
        prepend_embeds=None,
        prepend_mask: Optional[torch.Tensor] = None,
        global_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        return_info=False,
        exit_layer_ix=None,
        padding_mask: Optional[torch.Tensor] = None,
        skip_input_projection: bool = False,
        skip_output_projection: bool = False,
        self_attention_causal: Optional[bool] = None,
        self_attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        (batch, seq, device) = (*x.shape[:2], x.device)
        if exit_layer_ix is not None and False:
            raise ValueError("chunk-MoE diagnostics do not support early exit")
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        info = {"hidden_states": []}
        if skip_input_projection:
            if int(x.shape[-1]) != int(self.dim):
                raise ValueError(
                    f"skipping ContinuousTransformer input projection requires hidden width {self.dim}, got {x.shape[-1]}"
                )
        else:
            x = self.project_in(x)
        if prepend_embeds is not None:
            (prepend_length, prepend_dim) = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], "prepend dimension must match sequence dimension"
            if prepend_mask is not None and prepend_mask.shape != (batch, prepend_length):
                raise ValueError(
                    f"prepend_mask must match [batch, prepend_length], got {tuple(prepend_mask.shape)}"
                )
            x = torch.cat((prepend_embeds, x), dim=-2)
        elif prepend_mask is not None:
            raise ValueError("prepend_mask was provided without prepend_embeds")
        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(batch, -1, -1)
            x = torch.cat((memory_tokens, x), dim=1)
        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None
        if self.cross_attn_rotary_pos_emb is not None and context is not None:
            cross_attn_rotary_pos_emb = self.cross_attn_rotary_pos_emb.forward_from_seq_len(
                context.shape[1]
            )
        else:
            cross_attn_rotary_pos_emb = None
        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)
        if self_attention_bias is not None:
            expected_sequence = int(x.shape[1])
            if (
                self_attention_bias.ndim != 4
                or int(self_attention_bias.shape[0]) != batch
                or int(self_attention_bias.shape[1])
                not in {1, int(self.layers[0].self_attn.num_heads)}
                or (tuple(self_attention_bias.shape[-2:]) != (expected_sequence, expected_sequence))
                or (not self_attention_bias.is_floating_point())
            ):
                raise ValueError(
                    f"self_attention_bias must be floating point [batch,1|heads,sequence,sequence] aligned after prefixes; got {tuple(self_attention_bias.shape)} for sequence {expected_sequence}"
                )
        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)
        extended_padding_mask = None
        varlen_metadata = None
        if padding_mask is not None or prepend_mask is not None:
            content_mask = (
                padding_mask.to(device=device, dtype=torch.bool)
                if padding_mask is not None
                else torch.ones(batch, seq, device=device, dtype=torch.bool)
            )
            if content_mask.shape != (batch, seq):
                raise ValueError(
                    f"padding_mask must match the un-prepended input [batch, seq], got {tuple(content_mask.shape)} vs {(batch, seq)}"
                )
            prefix_masks = []
            if self.num_memory_tokens > 0:
                prefix_masks.append(
                    torch.ones(batch, self.num_memory_tokens, device=device, dtype=torch.bool)
                )
            if prepend_embeds is not None:
                prefix_masks.append(
                    prepend_mask.to(device=device, dtype=torch.bool)
                    if prepend_mask is not None
                    else torch.ones(batch, prepend_embeds.shape[1], device=device, dtype=torch.bool)
                )
            extended_padding_mask = torch.cat([*prefix_masks, content_mask], dim=-1)
            if flash_attn_varlen_func is not None and index_first_axis is not None:
                varlen_metadata = precompute_varlen_metadata(extended_padding_mask)
        for layer_ix, layer in enumerate(self.layers):
            layer_kwargs = {
                "context": context,
                "context_mask": context_mask,
                "rotary_pos_emb": rotary_pos_emb,
                "cross_attn_rotary_pos_emb": cross_attn_rotary_pos_emb,
                "global_cond": global_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "self_attention_flash_sliding_window": self.sliding_window,
                "padding_mask": extended_padding_mask,
                "varlen_metadata": varlen_metadata,
                "self_attention_causal": self_attention_causal,
                "self_attention_bias": self_attention_bias,
            }
            layer_result = layer(x, **layer_kwargs, **kwargs)
            x = layer_result
            if return_info:
                info["hidden_states"].append(x)
            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens :, :]
                if return_info:
                    return (x, info)
                return x
        x = x[:, self.num_memory_tokens :, :]
        if not skip_output_projection:
            x = self.project_out(x)
        if return_info:
            return (x, info)
        return x
