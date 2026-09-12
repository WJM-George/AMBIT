"""Optional count constraints derived exclusively from explicit request labels."""
from stable_audio_tools.data.sceneplan_generation_ar_request_sources import request_source_spans


class _DeclaredCountCodec:
    def __init__(self,codec,count):
        self.codec=codec
        self.count=count

    def __getattr__(self,name):
        return getattr(self.codec,name)

    def allowed_next_ids(self,prefix,**kwargs):
        return self.codec.allowed_next_ids(prefix,**{**kwargs,'min_sources':self.count,'max_sources':self.count})


def generate_with_declared_source_count(model,requests,codec,*,device,max_plan_tokens=512):
    """Batch by explicit count, decode, then restore the caller's input order.

    This is an inference policy, not learned count accuracy. Unrecognized or
    ambiguous numbering falls back to the model's original count prediction.
    Return policy metadata so evaluations cannot silently conflate the modes.
    """
    if not requests:
        raise ValueError('generation requests must be nonempty')
    counts=[len(request_source_spans(text)) or None for text in requests]
    groups={}
    for index,count in enumerate(counts):
        groups.setdefault(count,[]).append(index)
    outputs=[None]*len(requests)
    for count,indices in groups.items():
        active_codec=codec if count is None else _DeclaredCountCodec(codec,count)
        generated=model.generate_constrained([requests[i] for i in indices],active_codec,
                                             device=device,max_plan_tokens=max_plan_tokens)
        if len(generated)!=len(indices):raise RuntimeError('generation returned incomplete batch coverage')
        for index,tokens in zip(indices,generated):outputs[index]=tokens
    if any(tokens is None for tokens in outputs):raise RuntimeError('constrained generation lost a request')
    return {'token_ids':outputs,'declared_source_counts':counts,'count_constraint_applied':[count is not None for count in counts],
            'policy':'explicit_request_source_count_v1','count_source':'request text only; no reference plans or target counts'}
