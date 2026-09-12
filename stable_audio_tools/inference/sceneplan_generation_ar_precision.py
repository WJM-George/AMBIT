"""Explicit FP32 AR math while preserving BF16 frozen request encoding.

Use with freshly restored FP32 P10 weights. Autocast scopes are attached only
to the AR wrapper's methods; native P10 calls retain their caller's precision.
"""
from contextvars import ContextVar
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def configure_float32_ar(model):
    if getattr(model,'_generation_ar_float32_configured',False): raise ValueError('FP32 AR already configured')
    if any(p.dtype!=torch.float32 for p in model.p10_dit.parameters()):
        raise ValueError('restore P10 weights directly in FP32 before configuring FP32 AR')
    if any(p.dtype!=torch.float32 for p in model.ar_adapter.parameters()):
        raise ValueError('AR adapter must retain FP32 parameters')
    active=ContextVar(f'ar_float32_math_{id(model)}',default=False)
    for layer in model.shared_transformer.layers:
        original_attention=layer.self_attn.apply_attn
        def precise_attention(q,k,v,causal=None,*,_original=original_attention,**kwargs):
            if not active.get():return _original(q,k,v,causal=causal,**kwargs)
            # The installed native FlashAttention path silently casts FP32
            # Q/K/V to FP16. Use actual FP32 SDPA only inside the AR scope.
            if q.dtype!=torch.float32 or k.dtype!=torch.float32 or v.dtype!=torch.float32:
                raise ValueError('FP32 AR attention received a lower-precision tensor')
            if q.shape[1]!=k.shape[1]:raise ValueError('canonical AR attention requires equal query/key heads')
            for key in ('attention_bias','flex_attention_block_mask','flex_attention_score_mod','flash_attn_sliding_window'):
                if kwargs.get(key) is not None:raise ValueError('FP32 AR self attention does not support '+key)
            padding=kwargs.get('padding_mask')
            if padding is not None:v=v*padding[:,None,:,None].to(v.dtype)
            with sdpa_kernel([SDPBackend.MATH]):
                return F.scaled_dot_product_attention(q,k,v,is_causal=bool(causal))
        layer.self_attn.apply_attn=precise_attention
    original_encode=model.encode_requests
    def encode(requests,*,device):
        kind=torch.device(device).type
        with torch.autocast(kind,dtype=torch.bfloat16,enabled=kind=='cuda'):
            context,mask=original_encode(requests,device=device)
        return context.float(),mask
    model.encode_requests=encode
    for name in ('forward','prepare_decode_cache','decode_step'):
        original=getattr(model,name)
        def precise(*args,_original=original,**kwargs):
            kind=next(model.ar_adapter.parameters()).device.type
            token=active.set(True)
            try:
                with torch.autocast(kind,enabled=False):return _original(*args,**kwargs)
            finally:active.reset(token)
        setattr(model,name,precise)
    model._generation_ar_float32_configured=True
    return model
