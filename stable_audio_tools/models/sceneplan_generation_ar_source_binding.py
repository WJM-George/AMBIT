"""AR-only source attention bias over the unchanged shared P10 blocks.

Prototype for a separately gated experiment. No base parameter is added or
modified. Source identities come only from explicit request spans and source
slot tokens already present in the autoregressive prefix.
"""
from __future__ import annotations
from contextvars import ContextVar
from typing import Sequence
import torch

from stable_audio_tools.models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR
from stable_audio_tools.data.sceneplan_generation_ar_request_sources import request_source_spans, source_ids_for_offsets


def source_attention_bias(query_sources, context_sources, *, strength: float, dtype):
    """Penalize other-source keys, retaining global keys and global queries."""
    if query_sources.ndim!=2 or context_sources.ndim!=2 or query_sources.shape[0]!=context_sources.shape[0]:
        raise ValueError('source identity tensors must be [batch,sequence]')
    if not 0<=float(strength)<=10:
        raise ValueError('source attention strength must be in [0,10]')
    if float(strength)==0:
        return None
    query=query_sources[:,:,None]
    context=context_sources[:,None,:]
    other=(query>0)&(context>0)&(query!=context)
    # An all-zero additive mask can choose a different CUDA attention kernel
    # than the native padding mask. Preserve the native path when no source
    # actually needs downweighting, including every single-source request.
    if not bool(other.any()):
        return None
    return other[:,None].to(dtype=dtype)*(-float(strength))


def prefix_source_ids(token_ids, slot_token_ids):
    """Current source from already consumed codec slot tokens; 0 before slots."""
    if token_ids.ndim!=2:
        raise ValueError('prefix token ids must be [batch,sequence]')
    slots=torch.as_tensor(slot_token_ids,device=token_ids.device)
    numbers=torch.arange(1,len(slot_token_ids)+1,device=token_ids.device)
    found=((token_ids[:,:,None]==slots)*numbers).amax(dim=-1)
    # Generation codec slots are strictly ordered and contiguous.
    return found.cummax(dim=1).values


class SourceBoundGenerationAR(ScenePlanTransfusionGenerationAR):
    """A new AR mode with explicit source IDs in training and scoped cache bias."""

    def __init__(self, base: ScenePlanTransfusionGenerationAR, codec, *, strength: float=2.):
        if not 0<=float(strength)<=10:raise ValueError('invalid source attention strength')
        super().__init__(p10_dit=base.p10_dit,prompt_conditioner=base.prompt_conditioner,
                         pad_id=base.pad_id,vocab_size=base.vocab_size,
                         activation_checkpointing=base.activation_checkpointing)
        self.ar_adapter=base.ar_adapter
        self.binding_strength=float(strength)
        self.slot_token_ids=tuple(codec._tid(f'<source_slot_{i}>') for i in range(4))
        self._source_bias=ContextVar(f'ar_source_bias_{id(self)}',default=None)
        self._decode_request_sources=ContextVar(f'ar_decode_source_ids_{id(self)}',default=None)
        for layer in self.shared_transformer.layers:
            attention=layer.cross_attn
            if getattr(attention,'_generation_ar_source_binding_installed',False):
                raise ValueError('source binding has already been installed on these shared blocks')
            original=attention.apply_attn
            def scoped(q,k,v,*args,_original=original,**kwargs):
                bias=self._source_bias.get()
                if bias is not None:
                    if bias.shape[0]!=q.shape[0] or bias.shape[-2:]!=(q.shape[-2],k.shape[-2]):
                        raise ValueError('AR source bias does not match current cross attention')
                    existing=kwargs.get('attention_bias')
                    kwargs['attention_bias']=bias if existing is None else existing+bias
                return _original(q,k,v,*args,**kwargs)
            attention.apply_attn=scoped
            attention._generation_ar_source_binding_installed=True

    def request_source_ids(self, requests: Sequence[str], *, device):
        tokenizer=self.prompt_conditioner.tokenizer
        encoded=tokenizer(list(requests),add_special_tokens=True,padding=True,truncation=False,
                          return_offsets_mapping=True,return_tensors='pt')
        values=[source_ids_for_offsets(offsets.tolist(),request_source_spans(text))
                for text,offsets in zip(requests,encoded['offset_mapping'])]
        return torch.tensor(values,device=device,dtype=torch.long)

    def forward(self,plan_input_ids,plan_attention_mask,request_context,request_attention_mask,request_source_ids=None):
        if request_source_ids is None or self.binding_strength==0:
            return super().forward(plan_input_ids,plan_attention_mask,request_context,request_attention_mask)
        if self.activation_checkpointing and self.training:
            # A checkpointed backward recomputation outlives this context scope.
            raise ValueError('source binding requires activation_checkpointing=False until a recomputation-safe path is implemented')
        if tuple(request_source_ids.shape)!=tuple(request_attention_mask.shape):
            raise ValueError('request source IDs must align with request attention mask')
        bias=source_attention_bias(prefix_source_ids(plan_input_ids,self.slot_token_ids),request_source_ids,
                                   strength=self.binding_strength,dtype=next(self.shared_transformer.parameters()).dtype)
        token=self._source_bias.set(bias)
        try:return super().forward(plan_input_ids,plan_attention_mask,request_context,request_attention_mask)
        finally:self._source_bias.reset(token)

    @torch.no_grad()
    def generate_constrained(self,requests,codec,*,device,max_plan_tokens=1024):
        sources=self.request_source_ids(requests,device=device)
        token=self._decode_request_sources.set(sources)
        try:return super().generate_constrained(requests,codec,device=device,max_plan_tokens=max_plan_tokens)
        finally:self._decode_request_sources.reset(token)

    @torch.no_grad()
    def prepare_decode_cache(self,request_context,request_attention_mask,*,max_plan_tokens=1024):
        cache=super().prepare_decode_cache(request_context,request_attention_mask,max_plan_tokens=max_plan_tokens)
        sources=self._decode_request_sources.get()
        if sources is not None and tuple(sources.shape)!=tuple(request_attention_mask.shape):
            raise ValueError('decode source IDs do not align with request mask')
        cache.ar_binding_sources=sources
        cache.ar_current_source=torch.zeros(request_context.shape[0],1,device=request_context.device,dtype=torch.long)
        return cache

    @torch.no_grad()
    def decode_step(self,token_ids,cache):
        if cache.ar_binding_sources is None or self.binding_strength==0:
            return super().decode_step(token_ids,cache)
        current=prefix_source_ids(token_ids.reshape(-1,1),self.slot_token_ids)
        cache.ar_current_source=torch.maximum(cache.ar_current_source,current)
        bias=source_attention_bias(cache.ar_current_source,cache.ar_binding_sources,strength=self.binding_strength,
                                   dtype=next(self.shared_transformer.parameters()).dtype)
        token=self._source_bias.set(bias)
        try:return super().decode_step(token_ids,cache)
        finally:self._source_bias.reset(token)
