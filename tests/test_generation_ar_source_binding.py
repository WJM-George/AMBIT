"""Routing uses consumed source IDs and preserves ordinary global attention."""
import torch
from stable_audio_tools.models.sceneplan_generation_ar_source_binding import source_attention_bias, prefix_source_ids


def test_global_and_same_source_keys_remain_visible():
    query=torch.tensor([[0,1,2]])
    keys=torch.tensor([[0,1,1,2]])
    bias=source_attention_bias(query,keys,strength=2.,dtype=torch.float32)
    torch.testing.assert_close(bias[0,0],torch.tensor([[0.,0.,0.,0.],[0.,0.,0.,-2.],[0.,-2.,-2.,0.]]))
    assert source_attention_bias(query,keys,strength=0.,dtype=torch.float32) is None


def test_only_already_consumed_slots_determine_query_source():
    ids=torch.tensor([[8,9,10,77,11,88,12,13]])
    result=prefix_source_ids(ids,[77,88,99,100])
    assert result.tolist()==[[0,0,0,1,1,2,2,2]]
    changed=ids.clone();changed[0,5]=99
    assert torch.equal(prefix_source_ids(changed,[77,88,99,100])[:,:5],result[:,:5])


def test_no_other_source_preserves_native_attention_dispatch():
    assert source_attention_bias(torch.tensor([[0,1,1]]),torch.tensor([[0,1,1,0]]),strength=2.,dtype=torch.float32) is None
    assert source_attention_bias(torch.tensor([[0,0]]),torch.tensor([[0,1,2,3]]),strength=2.,dtype=torch.float32) is None


def test_other_source_penalty_changes_attention_but_preserves_gradients():
    q=torch.zeros(1,1,1,2,requires_grad=True)
    k=torch.tensor([[[[1.,0.],[0.,1.]]]])
    v=torch.tensor([[[[1.,0.],[0.,1.]]]])
    bias=source_attention_bias(torch.tensor([[1]]),torch.tensor([[1,2]]),strength=2.,dtype=torch.float32)
    output=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias)
    assert output[0,0,0,0]>.85 and output[0,0,0,1]<.15
    output[0,0,0,0].backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0
