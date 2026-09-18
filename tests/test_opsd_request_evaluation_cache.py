from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.request_evaluation_cache import request_evaluation_cache


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.calls=0
    def student_logits(self,obs,tokens):
        self.calls+=1
        return tokens*self.weight
    def native_plan(self,obs):
        self.calls+=1
        return dict(sources=[dict(description='unaltered')]),torch.tensor([1,2,3])


def learner():
    count=[0]
    def decode(z,*,model_num_samples):
        count[0]+=1;return z+1,model_num_samples
    return SimpleNamespace(adapter=Adapter().eval(),reference=Adapter().eval(),
        pipeline=SimpleNamespace(decode_foa_latents=decode),decode_count=count)


def test_repeated_reference_prefix_and_decode_reused_only_in_scope():
    x=learner();obs=object();tokens=torch.tensor([[1,2,3]]);z=torch.zeros(1,4,10)
    methods=(x.reference.native_plan,x.adapter.student_logits,x.pipeline.decode_foa_latents)
    with torch.no_grad(),request_evaluation_cache(x) as report:
        p,_=x.reference.native_plan(obs);p['sources'][0]['description']='mutated caller'
        assert x.reference.native_plan(obs)[0]['sources'][0]['description']=='unaltered'
        assert torch.equal(x.adapter.student_logits(obs,tokens),x.adapter.student_logits(obs,tokens.clone()))
        assert torch.equal(x.pipeline.decode_foa_latents(z,model_num_samples=[10])[0],
                           x.pipeline.decode_foa_latents(z.detach(),model_num_samples=[10])[0])
    assert report['hits']==dict(reference_native_plan=1,student_logits=1,FOA_decode=1)
    assert (x.reference.native_plan,x.adapter.student_logits,x.pipeline.decode_foa_latents)==methods
    assert 'student_logits' not in x.adapter.__dict__
    x.adapter.student_logits(obs,tokens)
    assert x.adapter.calls==2


def test_gradients_training_and_changed_inputs_bypass_cached_outputs():
    x=learner();obs=object();tokens=torch.tensor([[1.,2.]])
    with request_evaluation_cache(x):
        with torch.no_grad():x.adapter.student_logits(obs,tokens)
        result=x.adapter.student_logits(obs,tokens);result.sum().backward()
        assert x.adapter.weight.grad==3
        with torch.no_grad():
            x.adapter.student_logits(obs,tokens+1)
            x.adapter.train();x.adapter.student_logits(obs,tokens)
    assert x.adapter.calls==4


def test_restore_methods_after_error_and_reject_unbounded_storage():
    x=learner();original=x.reference.native_plan
    with pytest.raises(RuntimeError),torch.no_grad(),request_evaluation_cache(x,maximum_bytes=0) as report:
        for _ in range(2):x.reference.native_plan(object())
        raise RuntimeError('cancel request')
    assert not report['hits'] and report['bytes']==0
    assert x.reference.native_plan==original
