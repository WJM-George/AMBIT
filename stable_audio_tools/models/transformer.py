from functools import reduce, partial
from packaging import version
import logging
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.amp import autocast
from torch.nn.utils.parametrizations import weight_norm
from typing import Callable, Literal, Optional
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # PyTorch < 2.5 compatibility.
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
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
except ImportError as e:
    print(e)
    print('flash_attn varlen/bert_padding not available, disabling varlen attention')
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

    batch_size, seq_len = padding_mask.shape

    # Compute cumulative sequence lengths (same for all of q, k, v)
    seqlens = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()

    # Compute indices for gathering valid tokens
    # indices maps from packed position -> original (batch, seq) position
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()

    return {
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "indices": indices,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

from .utils import compile
from .dit_moe import CPEMoEDeltaFeedForward


def _left_pad_to_match(emb, target_len):
    """Left-pad or right-trim emb along seq dim to match target_len.

    Used for local conditioning embeddings that need to align with x
    without affecting prepended tokens (memory tokens, global cond, etc.).
    """
    emb_len = emb.shape[-2]
    if emb_len < target_len:
        return F.pad(emb, (0, 0, target_len - emb_len, 0), value=0.)
    elif emb_len > target_len:
        return emb[:, -target_len:, :]
    return emb

if flex_attention_available:
    try:
        torch._dynamo.config.cache_size_limit = 5000
        flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    except Exception as e:
        logging.debug(f"Could not compile flex_attention, using uncompiled version: {e}")
        flex_attention_compiled = flex_attention
else:
    flex_attention_compiled = None


# Cache band block_masks for sliding-window attention fallback (flex_attention path).
# Keyed by (seq_q, seq_k, w_left, w_right, device). create_block_mask is expensive
# but the result is reused across all transformer layers and forward passes.
_SLIDING_WINDOW_BLOCK_MASK_CACHE = {}

def _get_sliding_window_block_mask(seq_q, seq_k, w_left, w_right, device):
    key = (seq_q, seq_k, int(w_left), int(w_right), str(device))
    bm = _SLIDING_WINDOW_BLOCK_MASK_CACHE.get(key)
    if bm is None:
        wl, wr = int(w_left), int(w_right)
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
    return mask.masked_fill(~in_band, float('-inf'))


# Chunked-halo SDPA fallback. Math-equivalent to masked SDPA with a band
# mask, but processes queries in non-overlapping chunks with a (w_left,
# w_right) halo of keys/values on each side — every query stays inside its
# chunk's softmax. Avoids materializing the O(N^2) mask.
#
# At realistic SAME-L decoder shapes (N=69632, W=17, packed sequence is
# latent_length * (stride+1)): ~34x faster than full masked SDPA, and
# ~140x less peak mask memory (~1 MB per chunk vs 9.7 GB for one N x N mask).
# Chunk size is a tunable; 1024 is a good default at typical pretransform
# decoder shapes. Larger chunks waste more compute on out-of-band tiles;
# smaller chunks suffer from launch overhead.
_SLIDING_WINDOW_CHUNK_SIZE = 1024

def _sliding_window_chunked_halo_sdpa(q, k, v, w_left, w_right, chunk_size=_SLIDING_WINDOW_CHUNK_SIZE):
    B, H, N, D = q.shape
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
        mask = torch.zeros(delta.shape, dtype=q.dtype, device=q.device).masked_fill(~in_band, float('-inf'))
        outs.append(F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=mask, is_causal=False))
    return torch.cat(outs, dim=-2)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    # Preserve autocast context during recomputation to avoid dtype mismatches
    if "context_fn" not in kwargs:
        from torch.amp import autocast
        import functools
        # Get current autocast state
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype('cuda')
            def get_contexts():
                return (
                    autocast('cuda', dtype=dtype),
                    autocast('cuda', dtype=dtype),
                )
            kwargs["context_fn"] = get_contexts
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast("cuda", enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast("cuda", enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

# norms
class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class RunningInstanceNorm(nn.Module):
    def __init__(self, dim, momentum = 0.99, eps = 1e-4, saturate = True, trainable_gain = True):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1,1,dim))
        self.register_buffer("running_std", torch.ones(1,1,dim))
        self.saturate = saturate
        self.eps = eps
        self.momentum = momentum
        self.dim = dim
        self.trainable_gain = trainable_gain
        if self.trainable_gain:
            self.gain = nn.Parameter(torch.ones(1))
    
    def _update_stats(self, x):
        self.running_mean = self.running_mean * self.momentum + x.detach().mean(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)
        self.running_std  = (self.running_std * self.momentum + x.detach().std(dim = [0,1]).view(1, 1, self.dim) * (1 - self.momentum)).clip(min = self.eps)

    def forward(self, x):
        if self.training:
            self._update_stats(x)
        x = (x - self.running_mean) / self.running_std
        if self.saturate:
            x = torch.asinh(x)
        if self.trainable_gain:
            x = x * self.gain
        return x
        
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
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
            output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
            return output.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, fix_scale=False, force_fp32=False, eps=1e-5):
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
    def __init__(self, dim, init_val = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
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
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
        sinusoidal = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU() if not sinusoidal else Sin()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    #@compile
    def forward(self, x, varlen_metadata=None):
        if varlen_metadata is not None and index_first_axis is not None and pad_input is not None:
            # Pack valid tokens for efficient FFN computation (skip padding tokens)
            # Padding positions become zeros after unpack, which is fine since FFN output
            # is added to residual, preserving values at padding positions
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            indices = varlen_metadata["indices"]
            dim = x.shape[-1]

            # Pack to (N_valid, D)
            x_packed = index_first_axis(x.reshape(-1, dim), indices)

            # FFN on packed representation with pseudo-batch dim
            x_packed = self.ff(x_packed.unsqueeze(0)).squeeze(0)

            # Unpack back to (B, T, D)
            return pad_input(x_packed, indices, batch_size, seq_len)
        else:
            return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm_eps = 1e-6,
        qk_norm: Literal['l2', 'ln', 'rms', 'dyt', 'none'] = 'none',
        differential = False,
        feat_scale = False
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
        else:
            if differential:
                self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
            else:
                self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'rms', 'dyt','none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "rms" ,"dyt", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif self.qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

        self.causal = causal
        # A zero-initialized regional score adapter must preserve the frozen
        # acoustic prior bit-for-bit on its first optimization step while still
        # receiving the true attention gradient.  Once a non-zero score is
        # observed, only the ordinary biased SDPA path is used.
        self._attention_bias_activated = False
        
    @compile
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return q, k


    def apply_attn(self, q, k, v, causal = None, flex_attention_block_mask = None, flex_attention_score_mod = None, flash_attn_sliding_window = None, padding_mask = None, varlen_metadata = None, mask_padding_logits = False, attention_bias = None):

        if self.num_heads != self.kv_heads:
             # Repeat interleave kv_heads to match q_heads for grouped query attention
             heads_per_kv_head = self.num_heads // self.kv_heads
             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        # A frame-to-caption regional bias must behave identically regardless
        # of which optional attention backend is installed. Exact SDPA is the
        # only common additive-score path here, so a supplied bias explicitly
        # selects it instead of being silently ignored by Flash/Flex kernels.
        if attention_bias is not None:
            expected_tail = (q.shape[-2], k.shape[-2])
            if (
                attention_bias.ndim != 4
                or int(attention_bias.shape[0]) != int(q.shape[0])
                or int(attention_bias.shape[1]) not in {1, int(q.shape[1])}
                or tuple(attention_bias.shape[-2:]) != expected_tail
            ):
                raise ValueError(
                    "attention_bias must match [batch,1|heads,query,key] "
                    f"with batch={q.shape[0]}, heads={q.shape[1]}, "
                    f"query/key={expected_tail}; got {tuple(attention_bias.shape)}"
                )
            attn_mask = attention_bias.to(device=q.device, dtype=q.dtype)
            if mask_padding_logits and padding_mask is not None:
                key_mask = padding_mask.to(device=q.device, dtype=torch.bool)
                if key_mask.ndim != 2 or key_mask.shape != (
                    q.shape[0],
                    k.shape[-2],
                ):
                    raise ValueError(
                        "attention-bias key mask must be [batch,key_length], "
                        f"got {tuple(key_mask.shape)}"
                    )
                attn_mask = attn_mask.masked_fill(
                    ~key_mask[:, None, None, :], float("-inf")
                )
            if causal:
                causal_mask = torch.ones(
                    q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool
                ).tril(diagonal=k.shape[-2] - q.shape[-2])
                attn_mask = attn_mask.masked_fill(
                    ~causal_mask[None, None], float("-inf")
                )
            bias_is_only_gradient_path = attention_bias.requires_grad and not (
                q.requires_grad or k.requires_grad or v.requires_grad
            )
            if bias_is_only_gradient_path and q.is_cuda:
                # CUDA fused SDPA in torch 2.7 can produce a misaligned LSE
                # buffer when q/k/v are frozen and the additive score mask is
                # the only differentiable input.  The math backend has the
                # required mask gradient and avoids that kernel-specific
                # failure.  Inference and all ordinary attention remain fused.
                if sdpa_kernel is not None and SDPBackend is not None:
                    with sdpa_kernel([SDPBackend.MATH]):
                        biased_out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=attn_mask,
                            is_causal=False,
                        )
                else:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_mem_efficient=False,
                        enable_math=True,
                    ):
                        biased_out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=attn_mask,
                            is_causal=False,
                        )
            else:
                biased_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False
                )
            if not self._attention_bias_activated:
                has_nonzero_bias = bool(
                    torch.count_nonzero(attention_bias.detach()).item()
                )
                if has_nonzero_bias:
                    self._attention_bias_activated = True
                else:
                    if mask_padding_logits and padding_mask is not None:
                        key_mask = padding_mask.to(
                            device=q.device, dtype=torch.bool
                        )
                        baseline_mask = key_mask[:, None, None, :].expand(
                            -1, 1, q.shape[-2], -1
                        )
                        if causal:
                            causal_mask = torch.ones(
                                q.shape[-2],
                                k.shape[-2],
                                device=q.device,
                                dtype=torch.bool,
                            ).tril(diagonal=k.shape[-2] - q.shape[-2])
                            baseline_mask = (
                                baseline_mask
                                & causal_mask[None, None, :, :]
                            )
                        baseline_out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            attn_mask=baseline_mask,
                            is_causal=False,
                        )
                    else:
                        baseline_out = F.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            is_causal=causal if causal is not None else False,
                        )
                    # Forward is exactly the native Dense result; backward is
                    # the true derivative of the additive score-bias branch.
                    return baseline_out.detach() + (
                        biased_out - biased_out.detach()
                    )
            return biased_out

        # Cross-attention needs an exact key-padding mask. Merely zeroing V (the
        # self-attention fallback below) is not equivalent: padded zero keys still
        # receive softmax probability and dilute every valid conditioning token.
        # PyTorch SDPA keeps this path fused where supported and accepts the
        # broadcast [B,1,Q,K] boolean mask directly.
        if mask_padding_logits and padding_mask is not None:
            key_mask = padding_mask.to(device=q.device, dtype=torch.bool)
            if key_mask.ndim != 2 or key_mask.shape != (q.shape[0], k.shape[-2]):
                raise ValueError(
                    "cross-attention mask must be [batch, context_length], got "
                    f"{tuple(key_mask.shape)} for keys {tuple(k.shape)}"
                )
            attn_mask = key_mask[:, None, None, :].expand(-1, 1, q.shape[-2], -1)
            if causal:
                causal_mask = torch.ones(
                    q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool
                ).tril(diagonal=k.shape[-2] - q.shape[-2])
                attn_mask = attn_mask & causal_mask[None, None, :, :]
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False
            )

        # flash-attn may be importable in a CUDA build even when this forward
        # is intentionally running on CPU (contract tests, config probes, or
        # CPU inference).  Its kernels are CUDA-only, so availability must be
        # a property of both the installation and the current tensor device.
        flash_attn_available = flash_attn_func is not None and q.is_cuda
        flash_attn_varlen_available = (
            q.is_cuda
            and flash_attn_varlen_func is not None
            and index_first_axis is not None
        )

        if causal and (flex_attention_block_mask is not None or flex_attention_score_mod is not None):
            flex_attention_block_mask = None
            flex_attention_score_mod = None

        if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
            # Flex attention path - use V-zeroing for padding mask
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            out = flex_attention_compiled(q,k,v,
                block_mask = flex_attention_block_mask,
                score_mod = flex_attention_score_mod)
        elif flash_attn_available and varlen_metadata is not None and flash_attn_varlen_available:
            # Flash attention with varlen using precomputed metadata (fast path)
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            cu_seqlens = varlen_metadata["cu_seqlens"]
            max_seqlen = varlen_metadata["max_seqlen"]
            indices = varlen_metadata["indices"]

            fa_dtype_in = q.dtype
            # Rearrange to (B, T, H, D) for flash_attn
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))

            # Pack q, k, v using precomputed indices (much faster than calling unpad_input 3x)
            num_heads, head_dim = q.shape[2], q.shape[3]
            q_unpad = index_first_axis(q.reshape(-1, num_heads, head_dim), indices)
            k_unpad = index_first_axis(k.reshape(-1, num_heads, head_dim), indices)
            v_unpad = index_first_axis(v.reshape(-1, num_heads, head_dim), indices)

            out_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                causal=causal if causal is not None else False,
                window_size=flash_attn_sliding_window if flash_attn_sliding_window is not None else (-1, -1),
            )

            # Pad output back to original shape
            out = pad_input(out_unpad, indices, batch_size, seq_len)
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        elif flash_attn_available:
            # Standard flash attention (no padding mask, or varlen imports not available)
            # Apply V-zeroing fallback if padding_mask provided but we couldn't use varlen
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))

            out = flash_attn_func(q, k, v, causal = causal, window_size=flash_attn_sliding_window if (flash_attn_sliding_window is not None) else [-1,-1])

            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        else:
            # No flash-attn available. Sliding-window fallback cascade:
            #   Tier 2: flex_attention with band block_mask (best when torch.compile works)
            #   Tier 3: chunked-halo masked SDPA           (math-equivalent, ~30x faster than tier 4)
            #   Tier 4: full masked SDPA (N x N mask)      (last resort; high memory)
            # For the no-sliding-window case, fall through to plain SDPA full attention.
            # All apply V-zeroing for padding masks (cheap and equivalent to masking
            # those positions out of attention output).
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            if flash_attn_sliding_window is not None:
                seq_q, seq_k = q.shape[2], k.shape[2]
                wl, wr = flash_attn_sliding_window
                handled = False
                if flex_attention_available and flex_attention_compiled is not None:
                    try:
                        bm = _get_sliding_window_block_mask(seq_q, seq_k, wl, wr, q.device)
                        out = flex_attention_compiled(q, k, v, block_mask=bm)
                        handled = True
                    except Exception as _flex_err:
                        logging.debug(f"flex_attention failed, trying chunked-halo SDPA: {_flex_err}")
                if not handled:
                    try:
                        out = _sliding_window_chunked_halo_sdpa(q, k, v, wl, wr)
                        handled = True
                    except Exception as _chunk_err:
                        logging.debug(f"chunked-halo SDPA failed, falling back to full masked SDPA: {_chunk_err}")
                if not handled:
                    add_mask = _sliding_window_additive_mask(seq_q, seq_k, wl, wr, q.device, q.dtype)
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask, is_causal=False)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal if causal is not None else False)
        return out


    #@compile
    def forward(
        self,
        x,
        context = None,
        rotary_pos_emb = None,
        rotary_pos_emb_k = None,
        causal = None,
        flex_attention_block_mask = None,
        flex_attention_score_mod = None,
        flash_attn_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
        attention_bias = None,
        external_kv_lora_down = None,
        external_kv_lora_up = None,
        external_kv_lora_scale = 1.0,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            if self.differential:
                q, q_diff = self.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, q_diff))
                q = torch.stack([q, q_diff], dim = 1)
                kv = self.to_kv(kv_input)
                if external_kv_lora_down is not None or external_kv_lora_up is not None:
                    raise ValueError(
                        "external K/V LoRA currently supports ordinary "
                        "cross-attention only"
                    )
                k, k_diff, v = kv.chunk(3, dim=-1)
                k, k_diff, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, k_diff, v))
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
                kv = self.to_kv(kv_input)
                if external_kv_lora_down is not None or external_kv_lora_up is not None:
                    if not has_context or not all(
                        value is not None
                        for value in (
                            external_kv_lora_down,
                            external_kv_lora_up,
                        )
                    ):
                        raise ValueError(
                            "external K/V LoRA requires both weights on "
                            "cross-attention"
                        )
                    if (
                        external_kv_lora_down.ndim != 2
                        or external_kv_lora_up.ndim != 2
                        or tuple(external_kv_lora_down.shape)[0]
                        != int(kv_input.shape[-1])
                        or int(external_kv_lora_down.shape[1]) <= 0
                        or tuple(external_kv_lora_up.shape)
                        != (
                            int(external_kv_lora_down.shape[1]),
                            int(kv.shape[-1]),
                        )
                    ):
                        raise ValueError(
                            "external K/V LoRA weights must be "
                            "[context_dim,rank] and [rank,to_kv_out]"
                        )
                    if (
                        isinstance(external_kv_lora_scale, bool)
                        or not isinstance(external_kv_lora_scale, (int, float))
                        or not math.isfinite(float(external_kv_lora_scale))
                        or float(external_kv_lora_scale) <= 0.0
                    ):
                        raise ValueError(
                            "external K/V LoRA scale must be finite and positive"
                        )
                    down = external_kv_lora_down.to(
                        device=kv_input.device, dtype=kv_input.dtype
                    )
                    up = external_kv_lora_up.to(
                        device=kv_input.device, dtype=kv_input.dtype
                    )
                    low_rank = torch.einsum("btd,dr->btr", kv_input, down)
                    kv = kv + torch.einsum(
                        "btr,ro->bto", low_rank, up
                    ) * float(external_kv_lora_scale)
                k, v = kv.chunk(2, dim=-1)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            if self.differential:
                q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
                q, k, v, q_diff, k_diff  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, q_diff, k_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
        
            q_freqs = freqs

            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs

                if q.shape[-2] >= k.shape[-2]:
                    ratio = q.shape[-2] / k.shape[-2]
                    q_freqs, k_freqs = freqs, ratio * freqs
                else:
                    ratio = k.shape[-2] / q.shape[-2]
                    q_freqs, k_freqs = ratio * freqs, freqs

            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        
        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        # Cross-attention always needs exact key masking. Self-attention uses
        # FlashAttention varlen when available; ordinary full attention falls
        # back to exact masked SDPA when varlen support is unavailable.
        exact_padding_logits = has_context or (
            padding_mask is not None
            and varlen_metadata is None
            and flex_attention_block_mask is None
            and flex_attention_score_mod is None
            and flash_attn_sliding_window is None
        )

        if self.differential:
            q, q_diff = q.unbind(dim = 1)
            k, k_diff = k.unbind(dim = 1)
            out = self.apply_attn(q, k, v,  causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata, mask_padding_logits = exact_padding_logits, attention_bias = attention_bias)
            out_diff = self.apply_attn(q_diff, k_diff, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata, mask_padding_logits = exact_padding_logits, attention_bias = attention_bias)
            out = out - out_diff
        else:
            out = self.apply_attn(q, k, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata, mask_padding_logits = exact_padding_logits, attention_bias = attention_bias)
        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if self.feat_scale:
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(out.dtype)  # (b, n, 1)
                out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1)
                out_hf = out - out_dc
                out = out + (self.lambda_dc * out_dc + self.lambda_hf * out_hf) * mask
            else:
                out_dc = out.mean(dim=-2, keepdim=True)
                out_hf = out - out_dc
                out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    #@compile
    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            local_add_cond_dim = None,
            modular_local_cond_configs = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            add_rope = False,
            layer_scale = False,
            norm_type = 'layer_norm',
            sceneplan_soft_block_attention = None,
            sceneplan_chunk_moe = None,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads,dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
       
        if layer_scale and zero_init_branch_outputs:
            print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
            zero_init_branch_outputs = False
        
        if norm_type not in ['layer_norm', 'rms_norm', 'dyt']:
            raise ValueError(f'norm_type must be one of ["layer_norm", "rms_norm", "dyt"], got {norm_type}')

        norm_layer_map = {
            'layer_norm': LayerNorm,
            'rms_norm': RMSNorm,
            'dyt': DynamicTanh
        }
        norm_layer = norm_layer_map[norm_type]

        self.pre_norm = norm_layer(dim,**norm_kwargs)
        self.add_rope = add_rope

        self.self_attn = Attention(
            dim,
            dim_heads = self.dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = norm_layer(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads = self.dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        soft_block_config = dict(sceneplan_soft_block_attention or {})
        self.sceneplan_soft_block_max_bias = None
        self.sceneplan_event_bias_gate = None
        self.sceneplan_speech_bias_gate = None
        if soft_block_config:
            allowed = {"max_bias"}
            unknown = set(soft_block_config) - allowed
            if unknown:
                raise ValueError(
                    "unknown per-layer ScenePlan soft-block settings: "
                    f"{sorted(unknown)}"
                )
            maximum = float(soft_block_config.get("max_bias", 4.0))
            if not math.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("ScenePlan soft-block max_bias must be positive")
            if not cross_attend:
                raise ValueError(
                    "ScenePlan soft-block attention requires a cross-attention block"
                )
            self.sceneplan_soft_block_max_bias = maximum
            self.sceneplan_event_bias_gate = nn.Parameter(torch.zeros(()))
            self.sceneplan_speech_bias_gate = nn.Parameter(torch.zeros(()))

        moe_config = dict(sceneplan_chunk_moe or {})
        self.sceneplan_moe = None
        self.sceneplan_moe_sparse_scale = 0.0
        if moe_config:
            sparse_scale = float(moe_config.pop("sparse_scale", 0.5))
            if not math.isfinite(sparse_scale) or sparse_scale <= 0.0:
                raise ValueError("ScenePlan chunk-MoE sparse_scale must be positive")
            self.sceneplan_moe_sparse_scale = sparse_scale
            self.sceneplan_moe = CPEMoEDeltaFeedForward(dim, **moe_config)

        self.layer_ix = layer_ix

        self.conformer = None
        if conformer:
            self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
            self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

        self.local_add_cond_dim = local_add_cond_dim

        if local_add_cond_dim is not None:
            self.to_local_embed = nn.Sequential(
                nn.Linear(local_add_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

            nn.init.zeros_(self.to_local_embed[-1].weight)
            nn.init.zeros_(self.to_local_embed[-1].bias)

        else:
            self.to_local_embed = None

        # Modular local conditioning - independent projections per conditioning ID
        self.modular_local_cond_configs = modular_local_cond_configs or []
        self.modular_local_embeds = nn.ModuleDict()

        for config in self.modular_local_cond_configs:
            cond_id = config["id"]
            cond_dim = config["dim"]
            proj = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )
            # Zero-init output layer so new conditioning doesn't affect model initially
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.modular_local_embeds[cond_id] = proj

        self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None

    def _sceneplan_cross_attention_bias(
        self,
        base_bias,
        event_mask,
        speech_mask,
        *,
        dtype,
    ):
        if self.sceneplan_soft_block_max_bias is None:
            return base_bias
        if event_mask is None or speech_mask is None:
            raise ValueError(
                "enabled ScenePlan soft-block attention requires event and speech masks"
            )
        event_gate = (
            self.sceneplan_soft_block_max_bias
            * torch.tanh(self.sceneplan_event_bias_gate.float())
        ).to(dtype=dtype)
        speech_gate = (
            self.sceneplan_soft_block_max_bias
            * torch.tanh(self.sceneplan_speech_bias_gate.float())
        ).to(dtype=dtype)
        source_bias = (
            event_gate * event_mask.to(dtype=dtype)
            + speech_gate * speech_mask.to(dtype=dtype)
        )
        return (
            source_bias
            if base_bias is None
            else base_bias.to(dtype=dtype) + source_bias
        )

    def _feed_forward(
        self,
        hidden_states,
        *,
        varlen_metadata,
        context,
        context_mask,
        moe_audio_mask,
        moe_time_condition,
        moe_context_source_ids,
        moe_frame_source_ids,
        padding_mask,
    ):
        shared = self.ff(hidden_states, varlen_metadata=varlen_metadata)
        if self.sceneplan_moe is None:
            return shared, None
        required = {
            "context": context,
            "context_mask": context_mask,
            "moe_audio_mask": moe_audio_mask,
            "moe_time_condition": moe_time_condition,
            "moe_context_source_ids": moe_context_source_ids,
            "moe_frame_source_ids": moe_frame_source_ids,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "enabled ScenePlan chunk-MoE is missing inputs: " f"{missing}"
            )
        routed = self.sceneplan_moe(
            hidden_states,
            audio_mask=moe_audio_mask,
            time_condition=moe_time_condition,
            context=context,
            context_mask=context_mask,
            context_source_ids=moe_context_source_ids,
            frame_source_ids=moe_frame_source_ids,
            valid_mask=padding_mask,
        )
        return (
            shared + self.sceneplan_moe_sparse_scale * routed.hidden_states,
            routed,
        )

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
        context = None,
        context_mask = None,
        global_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        rotary_pos_emb = None,
        cross_attn_rotary_pos_emb = None,
        self_attention_block_mask = None,
        self_attention_score_mod = None,
        cross_attention_block_mask = None,
        cross_attention_score_mod = None,
        self_attention_bias = None,
        cross_attention_bias = None,
        cross_attention_event_mask = None,
        cross_attention_speech_mask = None,
        self_attention_flash_sliding_window = None,
        cross_attention_flash_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
        moe_audio_mask = None,
        moe_time_condition = None,
        moe_context_source_ids = None,
        moe_frame_source_ids = None,
        return_moe_info = False,
        external_cross_attn_kv_lora_down = None,
        external_cross_attn_kv_lora_up = None,
        external_cross_attn_kv_lora_scale = 1.0,
        self_attention_causal: Optional[bool] = None,
    ):
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        cross_attention_bias = self._sceneplan_cross_attention_bias(
            cross_attention_bias,
            cross_attention_event_mask,
            cross_attention_speech_mask,
            dtype=x.dtype,
        )
        cross_attention_kwargs = {
            "context": context,
            "flex_attention_block_mask": cross_attention_block_mask,
            "flex_attention_score_mod": cross_attention_score_mod,
            "flash_attn_sliding_window": cross_attention_flash_sliding_window,
            "padding_mask": context_mask,
            "attention_bias": cross_attention_bias,
            "external_kv_lora_down": external_cross_attn_kv_lora_down,
            "external_kv_lora_up": external_cross_attn_kv_lora_up,
            "external_kv_lora_scale": external_cross_attn_kv_lora_scale,
        }
        if cross_attn_rotary_pos_emb is not None:
            cross_attention_kwargs.update(
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_k=cross_attn_rotary_pos_emb,
            )

        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, rotary_pos_emb = rotary_pos_emb, causal = self_attention_causal, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata, attention_bias = self_attention_bias)
            x = x * torch.sigmoid(1 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual

            if context is not None and self.cross_attend:
                x = x + self.cross_attn_scale(
                    self.cross_attn(
                        self.cross_attend_norm(x), **cross_attention_kwargs
                    )
                )

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x, moe_output = self._feed_forward(
                x,
                varlen_metadata=varlen_metadata,
                context=context,
                context_mask=context_mask,
                moe_audio_mask=moe_audio_mask,
                moe_time_condition=moe_time_condition,
                moe_context_source_ids=moe_context_source_ids,
                moe_frame_source_ids=moe_frame_source_ids,
                padding_mask=padding_mask,
            )
            x = x * torch.sigmoid(1 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual

        else:
            x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb = rotary_pos_emb, causal = self_attention_causal, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata, attention_bias = self_attention_bias))

            if context is not None and self.cross_attend:
                x = x + self.cross_attn_scale(
                    self.cross_attn(
                        self.cross_attend_norm(x), **cross_attention_kwargs
                    )
                )

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            ff_output, moe_output = self._feed_forward(
                self.ff_norm(x),
                varlen_metadata=varlen_metadata,
                context=context,
                context_mask=context_mask,
                moe_audio_mask=moe_audio_mask,
                moe_time_condition=moe_time_condition,
                moe_context_source_ids=moe_context_source_ids,
                moe_frame_source_ids=moe_frame_source_ids,
                padding_mask=padding_mask,
            )
            x = x + self.ff_scale(ff_output)
            

        if return_moe_info:
            if moe_output is None:
                raise ValueError("return_moe_info selected a block without chunk-MoE")
            routing = moe_output.routing
            conflict_values = routing["conflict_gate"].detach().float()
            conflict_mean = (
                conflict_values.mean()
                if conflict_values.numel()
                else x.new_zeros((), dtype=torch.float32)
            )
            compact_stats = torch.cat(
                (
                    routing["expert_dispatch"].detach().float(),
                    torch.stack(
                        (
                            # These two scalars remain attached because the
                            # outer transformer applies router-v3 auxiliary
                            # losses after aggregating all active MoE layers.
                            routing["router_entropy"].float(),
                            conflict_mean,
                            routing["router_max_probability"].detach().float(),
                            routing["top1_weight_mean"].detach().float(),
                            routing[
                                "conflict_gate_saturation_fraction"
                            ].detach().float(),
                            routing["conflict_logit_l2"].float(),
                            routing[
                                "prior_evidence_top1_agreement"
                            ].detach().float(),
                            routing["chunk_length"].new_tensor(
                                routing["chunk_length"].numel(),
                                dtype=torch.float32,
                            ),
                        )
                    ),
                )
            )
            return x, moe_output.auxiliary_loss, compact_stats

        return x
        
class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
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
        sceneplan_soft_block_attention=None,
        sceneplan_chunk_moe=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

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
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6)
            )

        self.final_cross_attn_ix = final_cross_attn_ix

        self.sliding_window = sliding_window

        soft_block_config = dict(sceneplan_soft_block_attention or {})
        soft_block_enabled = bool(soft_block_config.pop("enabled", False))
        soft_block_start = int(soft_block_config.pop("layer_start", 0))
        soft_block_count = int(
            soft_block_config.pop("layer_count", depth - soft_block_start)
        )
        # Source-count validation belongs to the mask builder; blocks only own
        # their two scalar gates.
        soft_block_config.pop("max_sources", None)
        if soft_block_enabled and (
            soft_block_start < 0
            or soft_block_count <= 0
            or soft_block_start + soft_block_count > depth
        ):
            raise ValueError("ScenePlan soft-block layer range must fit Transformer depth")
        if not soft_block_enabled and soft_block_config:
            raise ValueError(
                "disabled ScenePlan soft-block attention has unused settings: "
                f"{sorted(soft_block_config)}"
            )
        self.sceneplan_soft_block_enabled = soft_block_enabled

        moe_config = dict(sceneplan_chunk_moe or {})
        moe_enabled = bool(moe_config.pop("enabled", False))
        moe_start = int(moe_config.pop("layer_start", max(0, depth - 4)))
        moe_count = int(moe_config.pop("layer_count", depth - moe_start))
        self.sceneplan_chunk_moe_load_balance_weight = float(
            moe_config.pop("load_balance_loss_weight", 0.01)
        )
        if not math.isfinite(self.sceneplan_chunk_moe_load_balance_weight) or (
            self.sceneplan_chunk_moe_load_balance_weight < 0.0
        ):
            raise ValueError("chunk-MoE load-balance weight must be finite and non-negative")
        self.sceneplan_chunk_moe_router_entropy_weight = float(
            moe_config.pop("router_entropy_loss_weight", 0.0)
        )
        if not math.isfinite(
            self.sceneplan_chunk_moe_router_entropy_weight
        ) or self.sceneplan_chunk_moe_router_entropy_weight < 0.0:
            raise ValueError(
                "chunk-MoE router-entropy weight must be finite and non-negative"
            )
        entropy_target = moe_config.pop("router_entropy_target", None)
        self.sceneplan_chunk_moe_router_entropy_target = (
            None if entropy_target is None else float(entropy_target)
        )
        self.sceneplan_chunk_moe_conflict_logit_l2_weight = float(
            moe_config.pop("conflict_logit_l2_loss_weight", 0.0)
        )
        if not math.isfinite(
            self.sceneplan_chunk_moe_conflict_logit_l2_weight
        ) or self.sceneplan_chunk_moe_conflict_logit_l2_weight < 0.0:
            raise ValueError(
                "chunk-MoE conflict-logit L2 weight must be finite and non-negative"
            )
        if moe_enabled and (
            moe_start < 0
            or moe_count <= 0
            or moe_start + moe_count > depth
        ):
            raise ValueError("ScenePlan chunk-MoE layer range must fit Transformer depth")
        if not moe_enabled and moe_config:
            raise ValueError(
                "disabled ScenePlan chunk-MoE has unused settings: "
                f"{sorted(moe_config)}"
            )
        self.sceneplan_chunk_moe_enabled = moe_enabled
        self.sceneplan_chunk_moe_num_experts = int(
            moe_config.get("num_experts", 4) if moe_enabled else 0
        )
        if (
            self.sceneplan_chunk_moe_router_entropy_target is not None
            and not (
                moe_enabled
                and 0.0
                < self.sceneplan_chunk_moe_router_entropy_target
                < math.log(float(self.sceneplan_chunk_moe_num_experts))
            )
        ):
            raise ValueError(
                "chunk-MoE router entropy target must lie strictly between "
                "zero and log(num_experts)"
            )

        for i in range(depth):
            should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
            layer_soft_block = None
            if soft_block_enabled and soft_block_start <= i < soft_block_start + soft_block_count:
                layer_soft_block = dict(soft_block_config)
            layer_moe = None
            if moe_enabled and moe_start <= i < moe_start + moe_count:
                layer_moe = dict(moe_config)
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = should_cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    local_add_cond_dim = local_add_cond_dim,
                    modular_local_cond_configs = modular_local_cond_configs,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    sceneplan_soft_block_attention=layer_soft_block,
                    sceneplan_chunk_moe=layer_moe,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        context = None,
        context_mask: Optional[torch.Tensor] = None,
        prepend_embeds = None,
        prepend_mask: Optional[torch.Tensor] = None,
        global_cond = None,
        local_add_cond = None,
        modular_local_cond = None,
        external_cross_attn_kv_lora_down = None,
        external_cross_attn_kv_lora_up = None,
        external_cross_attn_kv_lora_scale = 1.0,
        external_cross_attn_kv_lora_start_index = 0,
        external_layer_replacements = None,
        external_layer_replacement_start_index = 0,
        return_info = False,
        return_moe_info = False,
        use_checkpointing = True,
        exit_layer_ix = None,
        padding_mask: Optional[torch.Tensor] = None,
        moe_time_condition = None,
        moe_context_source_ids = None,
        moe_frame_source_ids = None,
        skip_input_projection: bool = False,
        skip_output_projection: bool = False,
        self_attention_causal: Optional[bool] = None,
        self_attention_bias: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        if return_info and return_moe_info:
            raise ValueError("return_info and return_moe_info are mutually exclusive")
        if exit_layer_ix is not None and return_moe_info:
            raise ValueError("chunk-MoE diagnostics do not support early exit")
        if return_moe_info and not self.sceneplan_chunk_moe_enabled:
            raise ValueError("return_moe_info requires an enabled ScenePlan chunk-MoE")

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        if skip_input_projection:
            if int(x.shape[-1]) != int(self.dim):
                raise ValueError(
                    "skipping ContinuousTransformer input projection requires "
                    f"hidden width {self.dim}, got {x.shape[-1]}"
                )
        else:
            x = self.project_in(x)

        replacement_layer_count = 0
        if external_layer_replacements is not None:
            if not isinstance(external_layer_replacements, nn.ModuleList):
                raise ValueError(
                    "external_layer_replacements must be a registered ModuleList"
                )
            replacement_layer_count = len(external_layer_replacements)
            if (
                replacement_layer_count <= 0
                or isinstance(external_layer_replacement_start_index, bool)
                or not isinstance(external_layer_replacement_start_index, int)
                or external_layer_replacement_start_index < 0
                or external_layer_replacement_start_index
                + replacement_layer_count
                > self.depth
            ):
                raise ValueError(
                    "external native layer replacement range must lie inside "
                    "the Transformer depth: "
                    f"start={external_layer_replacement_start_index}, "
                    f"count={replacement_layer_count}, depth={self.depth}"
                )
            for offset, replacement in enumerate(external_layer_replacements):
                native = self.layers[
                    external_layer_replacement_start_index + offset
                ]
                native_signature = {
                    name: tuple(parameter.shape)
                    for name, parameter in native.named_parameters()
                }
                replacement_signature = {
                    name: tuple(parameter.shape)
                    for name, parameter in replacement.named_parameters()
                }
                if (
                    type(replacement) is not type(native)
                    or replacement_signature != native_signature
                ):
                    raise ValueError(
                        "external native layer replacement architecture does "
                        "not match Transformer block "
                        f"{external_layer_replacement_start_index + offset}"
                    )

        kv_lora_layer_count = 0
        kv_lora_values = (
            external_cross_attn_kv_lora_down,
            external_cross_attn_kv_lora_up,
        )
        if any(value is not None for value in kv_lora_values):
            if not all(value is not None for value in kv_lora_values):
                raise ValueError(
                    "external cross-attention K/V LoRA requires down and up "
                    "weights together"
                )
            if (
                external_cross_attn_kv_lora_down.ndim != 3
                or external_cross_attn_kv_lora_up.ndim != 3
            ):
                raise ValueError(
                    "external cross-attention K/V LoRA weights must be "
                    "[layers,context_dim,rank] and [layers,rank,to_kv_out]"
                )
            kv_lora_layer_count = int(
                external_cross_attn_kv_lora_down.shape[0]
            )
            if (
                kv_lora_layer_count <= 0
                or int(external_cross_attn_kv_lora_up.shape[0])
                != kv_lora_layer_count
                or int(external_cross_attn_kv_lora_down.shape[2]) <= 0
                or int(external_cross_attn_kv_lora_up.shape[1])
                != int(external_cross_attn_kv_lora_down.shape[2])
                or context is None
                or int(external_cross_attn_kv_lora_down.shape[1])
                != int(context.shape[-1])
                or isinstance(external_cross_attn_kv_lora_start_index, bool)
                or not isinstance(external_cross_attn_kv_lora_start_index, int)
                or external_cross_attn_kv_lora_start_index < 0
                or external_cross_attn_kv_lora_start_index + kv_lora_layer_count
                > self.depth
                or isinstance(external_cross_attn_kv_lora_scale, bool)
                or not isinstance(
                    external_cross_attn_kv_lora_scale, (int, float)
                )
                or not math.isfinite(float(external_cross_attn_kv_lora_scale))
                or float(external_cross_attn_kv_lora_scale) <= 0.0
            ):
                raise ValueError(
                    "external cross-attention K/V LoRA shape/range must match "
                    "the selected Transformer context and depth"
                )
            for offset in range(kv_lora_layer_count):
                layer = self.layers[
                    external_cross_attn_kv_lora_start_index + offset
                ]
                cross_attention = getattr(layer, "cross_attn", None)
                to_kv = getattr(cross_attention, "to_kv", None)
                if (
                    not isinstance(to_kv, nn.Linear)
                    or int(to_kv.in_features) != int(context.shape[-1])
                    or int(external_cross_attn_kv_lora_up.shape[2])
                    != int(to_kv.out_features)
                ):
                    raise ValueError(
                        "external cross-attention K/V LoRA selected a layer "
                        "without a matching native to_kv projection"
                    )

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'
            if prepend_mask is not None and prepend_mask.shape != (
                batch,
                prepend_length,
            ):
                raise ValueError(
                    "prepend_mask must match [batch, prepend_length], got "
                    f"{tuple(prepend_mask.shape)}"
                )

            x = torch.cat((prepend_embeds, x), dim = -2)
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
            cross_attn_rotary_pos_emb = (
                self.cross_attn_rotary_pos_emb.forward_from_seq_len(
                    context.shape[1]
                )
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
                or int(self_attention_bias.shape[1]) not in {1, int(self.layers[0].self_attn.num_heads)}
                or tuple(self_attention_bias.shape[-2:])
                != (expected_sequence, expected_sequence)
                or not self_attention_bias.is_floating_point()
            ):
                raise ValueError(
                    "self_attention_bias must be floating point "
                    "[batch,1|heads,sequence,sequence] aligned after prefixes; "
                    f"got {tuple(self_attention_bias.shape)} for sequence "
                    f"{expected_sequence}"
                )

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        # Extend padding mask for prepended tokens if provided
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
                    "padding_mask must match the un-prepended input [batch, seq], "
                    f"got {tuple(content_mask.shape)} vs {(batch, seq)}"
                )

            prefix_masks = []
            if self.num_memory_tokens > 0:
                prefix_masks.append(
                    torch.ones(
                        batch,
                        self.num_memory_tokens,
                        device=device,
                        dtype=torch.bool,
                    )
                )
            if prepend_embeds is not None:
                prefix_masks.append(
                    prepend_mask.to(device=device, dtype=torch.bool)
                    if prepend_mask is not None
                    else torch.ones(
                        batch,
                        prepend_embeds.shape[1],
                        device=device,
                        dtype=torch.bool,
                    )
                )
            extended_padding_mask = torch.cat(
                [*prefix_masks, content_mask],
                dim=-1,
            )

            # Precompute varlen metadata once for all layers (major performance optimization)
            # Only compute if varlen attention is actually available
            if flash_attn_varlen_func is not None and index_first_axis is not None:
                varlen_metadata = precompute_varlen_metadata(extended_padding_mask)

        moe_audio_mask = None
        padded_moe_frame_source_ids = None
        if self.sceneplan_chunk_moe_enabled:
            required = {
                "context": context,
                "context_mask": context_mask,
                "moe_time_condition": moe_time_condition,
                "moe_context_source_ids": moe_context_source_ids,
                "moe_frame_source_ids": moe_frame_source_ids,
            }
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "enabled ScenePlan chunk-MoE is missing Transformer inputs: "
                    f"{missing}"
                )
            content_mask = (
                padding_mask.to(device=device, dtype=torch.bool)
                if padding_mask is not None
                else torch.ones(batch, seq, device=device, dtype=torch.bool)
            )
            if content_mask.shape != (batch, seq):
                raise ValueError("chunk-MoE content mask must align with audio frames")
            prefix_length = int(x.shape[1]) - int(seq)
            moe_audio_mask = F.pad(
                content_mask, (prefix_length, 0), value=False
            )
            if (
                moe_frame_source_ids.ndim != 3
                or int(moe_frame_source_ids.shape[0]) != batch
                or int(moe_frame_source_ids.shape[-1]) != seq
            ):
                raise ValueError(
                    "moe_frame_source_ids must be [B,max_sources,audio_frames]"
                )
            padded_moe_frame_source_ids = F.pad(
                moe_frame_source_ids.to(device=device, dtype=torch.long),
                (prefix_length, 0),
                value=0,
            )
            if extended_padding_mask is None:
                extended_padding_mask = torch.ones(
                    batch, x.shape[1], device=device, dtype=torch.bool
                )

        moe_auxiliary_losses = []
        moe_compact_stats = []

        # Iterate over the transformer layers
        for layer_ix, layer in enumerate(self.layers):

            if (
                external_layer_replacements is not None
                and external_layer_replacement_start_index
                <= layer_ix
                < external_layer_replacement_start_index
                + replacement_layer_count
            ):
                layer = external_layer_replacements[
                    layer_ix - external_layer_replacement_start_index
                ]


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
                "moe_audio_mask": moe_audio_mask,
                "moe_time_condition": moe_time_condition,
                "moe_context_source_ids": moe_context_source_ids,
                "moe_frame_source_ids": padded_moe_frame_source_ids,
                "self_attention_causal": self_attention_causal,
                "self_attention_bias": self_attention_bias,
            }

            layer_returns_moe = return_moe_info and (
                getattr(layer, "sceneplan_moe", None) is not None
            )
            layer_kwargs["return_moe_info"] = layer_returns_moe


            if (
                external_cross_attn_kv_lora_down is not None
                and external_cross_attn_kv_lora_start_index
                <= layer_ix
                < external_cross_attn_kv_lora_start_index
                + kv_lora_layer_count
            ):
                kv_lora_index = (
                    layer_ix - external_cross_attn_kv_lora_start_index
                )
                layer_kwargs.update(
                    external_cross_attn_kv_lora_down=(
                        external_cross_attn_kv_lora_down[kv_lora_index]
                    ),
                    external_cross_attn_kv_lora_up=(
                        external_cross_attn_kv_lora_up[kv_lora_index]
                    ),
                    external_cross_attn_kv_lora_scale=(
                        external_cross_attn_kv_lora_scale
                    ),
                )

            if use_checkpointing:
                layer_result = checkpoint(layer, x, **layer_kwargs, **kwargs)
            else:
                layer_result = layer(x, **layer_kwargs, **kwargs)
            if layer_returns_moe:
                x, layer_auxiliary, layer_stats = layer_result
                moe_auxiliary_losses.append(layer_auxiliary)
                moe_compact_stats.append(layer_stats)
            else:
                x = layer_result

            if return_info:
                info["hidden_states"].append(x)

            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens:, :]

                if return_info:
                    return x, info
                
                return x

        x = x[:, self.num_memory_tokens:, :]

        if not skip_output_projection:
            x = self.project_out(x)

        if return_info:
            return x, info

        if return_moe_info:
            if not moe_auxiliary_losses:
                raise RuntimeError("enabled chunk-MoE produced no layer diagnostics")
            raw_balance = torch.stack(moe_auxiliary_losses).mean()
            stats = torch.stack(moe_compact_stats)
            experts = self.sceneplan_chunk_moe_num_experts
            router_entropy = stats[:, experts].mean()
            router_entropy_loss = router_entropy
            if self.sceneplan_chunk_moe_router_entropy_target is not None:
                router_entropy_loss = (
                    router_entropy
                    - self.sceneplan_chunk_moe_router_entropy_target
                ).square()
            conflict_logit_l2 = stats[:, experts + 5].mean()
            moe_info = {
                "auxiliary_loss": (
                    raw_balance * self.sceneplan_chunk_moe_load_balance_weight
                    + router_entropy_loss
                    * self.sceneplan_chunk_moe_router_entropy_weight
                    + conflict_logit_l2
                    * self.sceneplan_chunk_moe_conflict_logit_l2_weight
                ),
                "raw_load_balance_loss": raw_balance.detach(),
                "expert_dispatch": stats[:, :experts].mean(dim=0),
                "router_entropy": router_entropy.detach(),
                "router_entropy_loss": router_entropy_loss.detach(),
                "conflict_gate_mean": stats[:, experts + 1].mean(),
                "router_max_probability": stats[:, experts + 2].mean(),
                "top1_weight_mean": stats[:, experts + 3].mean(),
                "conflict_gate_saturation_fraction": stats[
                    :, experts + 4
                ].mean(),
                "conflict_logit_l2": conflict_logit_l2.detach(),
                "prior_evidence_top1_agreement": stats[:, experts + 6].mean(),
                "chunk_count": stats[:, experts + 7].mean(),
                "routed_chunk_evaluations": stats[:, experts + 7].sum(),
                "active_layers": stats.new_tensor(len(moe_auxiliary_losses)),
            }
            return x, moe_info
        
        return x
