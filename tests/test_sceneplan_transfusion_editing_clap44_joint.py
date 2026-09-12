from dataclasses import asdict
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from stable_audio_tools.models.sceneplan_transfusion_editing_ar import EDITING_AR_CLAP44_CONTRACT
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44SourceBridge
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import ScenePlanTransfusionEditingCLAP44Pipeline
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import JOINT44_RUN_SCHEMA, atomic_json, ensure_run_identity, load_joint_checkpoint, save_joint_checkpoint
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_joint import source_feature_distillation, shuffle_source_features
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _capture_rank_rng_state


def test_joint_forward_shares_gradients_and_keeps_new_plan_out_of_ar():
    import copy
    import importlib.util
    from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_joint import CLAP44JointEditingModule,optimizer_groups
    path=Path(__file__).with_name("test_sceneplan_transfusion_editing_clap44_integration.py")
    spec=importlib.util.spec_from_file_location("clap44_ar_fixtures",path)
    fixtures=importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    ar=fixtures.ar_harness()
    class Stack(nn.Module):
        def __init__(self): super().__init__(); self.layers=nn.ModuleList([nn.Linear(1024,1024)])
        def forward(self,hidden,**kwargs): return self.layers[0](hidden+hidden[:,:12].mean(1,keepdim=True))
    ar.editing_dit.transformer=Stack()
    class RF(nn.Module):
        def __init__(self):
            super().__init__(); self.model=ar.editing_dit; self.input=nn.Linear(64,1024); self.output=nn.Linear(1024,64)
        def forward(self,noised,times,*,source,gain,**kwargs):
            return self.output(self.model.transformer(self.input((noised+source).transpose(1,2)))).transpose(1,2)+gain[:,None,None]
    class Conditioner(nn.Module):
        def forward(self,metadata,device):
            return {"source_foa_latent":[torch.stack([row["source_foa_latent"] for row in metadata]),None],"gain":torch.tensor([row["model_sceneplan"]["sources"][0]["gain_db"] for row in metadata],device=device)}
    class Diffusion(nn.Module):
        def __init__(self): super().__init__(); self.model=RF(); self.conditioner=Conditioner()
        def get_conditioning_inputs(self,conditioning): return {"source":conditioning["source_foa_latent"][0],"gain":conditioning["gain"]}
    ar.encode_edit_instructions=lambda texts,device: (torch.zeros(len(texts),1,1024),torch.ones(len(texts),1,dtype=torch.bool))
    module=CLAP44JointEditingModule(diffusion=Diffusion(),ar=ar).train()
    groups,counts=optimizer_groups(module,ar_lr=.01,shared_lr=.001,dit_lr=.001)
    assert sum(counts.values())==sum(p.numel() for p in module.parameters() if p.requires_grad)
    x=torch.randn(2,64,12); mask=torch.ones(2,12,dtype=torch.bool)
    metadata=[{"source_foa_latent":row,"model_sceneplan":{"sources":[{"gain_db":0.}]}} for row in x]
    kwargs=dict(source_foa_latent=x,source_attention_mask=mask,plan_input_ids=torch.tensor([[1,3],[1,4]]),plan_attention_mask=torch.ones(2,2,dtype=torch.bool),raw_edit_requests=["move","move"],metadata=metadata,noised_target=torch.randn_like(x),timesteps=torch.ones(2)*.5,rf_padding_mask=mask)
    logits,rf,query,teacher=module(**kwargs)
    changed=copy.deepcopy(metadata)
    for row in changed: row["model_sceneplan"]["sources"][0]["gain_db"]=.5
    other_logits,other_rf,_,_=module(**{**kwargs,"metadata":changed})
    torch.testing.assert_close(logits,other_logits,rtol=0,atol=0)
    torch.testing.assert_close(other_rf-rf,torch.full_like(rf,.5))
    shared=module.ar.shared_transformer.layers[0].weight
    ar_loss=logits.square().mean(); rf_loss=rf.square().mean()
    ga=torch.autograd.grad(ar_loss,shared,retain_graph=True)[0]
    gr=torch.autograd.grad(rf_loss,shared,retain_graph=True)[0]
    auxiliary=source_feature_distillation(query,teacher,semantic_dim=16)[0].mean()
    (ar_loss+rf_loss+auxiliary).backward()
    torch.testing.assert_close(shared.grad,ga+gr,atol=2e-6,rtol=2e-6)
    assert ga.abs().sum()>0 and gr.abs().sum()>0
    assert all(p.grad is None for p in ar.source_clap_model.parameters())
    with pytest.raises(RuntimeError,match="source tensors diverged"):
        module(**{**kwargs,"metadata":list(reversed(metadata))})


def _weighted_ddp_worker(rank,init_file,output):
    import torch.distributed as dist
    from contextlib import nullcontext
    from torch.nn.parallel import DistributedDataParallel
    from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_joint import normalized_joint_loss
    torch.set_num_threads(1); torch.manual_seed(5)
    dist.init_process_group("gloo",init_method="file://"+init_file,rank=rank,world_size=2)
    wrapped=DistributedDataParallel(nn.Linear(3,3,bias=False))
    xs=torch.tensor([[2.,3.,1.],[5.,7.,2.],[3.,4.,1.],[7.,8.,3.]])
    denominators=xs.sum(0).double()
    for micro in range(2):
        with wrapped.no_sync() if micro==0 else nullcontext():
            value=wrapped(xs[2*rank+micro])
            loss=normalized_joint_loss(*value,denominators,world_size=2,lambda_ar=.1,lambda_rf=1.,lambda_auxiliary=.05)
            loss.backward()
    observed=wrapped.module.weight.grad.clone()
    dist.destroy_process_group()
    torch.manual_seed(5); reference=nn.Linear(3,3,bias=False)
    totals=reference(xs).sum(0)
    normalized_joint_loss(*totals,denominators,world_size=1,lambda_ar=.1,lambda_rf=1.,lambda_auxiliary=.05).backward()
    torch.testing.assert_close(observed,reference.weight.grad,atol=1e-7,rtol=1e-6)
    Path(output,f"rank-{rank}.ok").write_text("global variable-length objective matches")


def test_global_joint_loss_and_accumulated_ddp_gradients(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_weighted_ddp_worker,args=(str(tmp_path/"gloo"),str(tmp_path)),nprocs=2,join=True)
    assert len(list(tmp_path.glob("rank-*.ok")))==2


def test_source_distillation_is_headwise_and_stops_teacher_gradient():
    query = torch.randn(3,24,requires_grad=True); teacher = torch.randn(3,24,requires_grad=True)
    loss,_ = source_feature_distillation(query,teacher,semantic_dim=16)
    scaled,_ = source_feature_distillation(query,torch.cat((teacher[:,:16]*20,teacher[:,16:]*.01),-1),semantic_dim=16)
    torch.testing.assert_close(loss,scaled)
    loss.mean().backward()
    assert query.grad.abs().sum() > 0 and teacher.grad is None
    matched,_ = source_feature_distillation(teacher,teacher,semantic_dim=16)
    assert matched.abs().max() < 1e-6


def test_feature_permutation_moves_global_sequence_and_mask_together():
    features = {"global":torch.arange(6).reshape(3,2),"sequence":torch.arange(12).reshape(3,2,2),"sequence_mask":torch.tensor([[1,1],[1,0],[0,1]],dtype=torch.bool),"stride":4}
    order = torch.tensor([2,0,1]); result = shuffle_source_features(features,order)
    for key in ("global","sequence","sequence_mask"):
        assert torch.equal(result[key],features[key][order])
    assert result["stride"] == 4


def test_disabled_feature_projections_are_absent_from_trainable_set():
    bridge = EditingCLAP44SourceBridge(20,CLAP44Config(),use_sequence=False)
    assert not any(x.requires_grad for x in bridge.sequence_projection.parameters())
    assert all(x.requires_grad for x in bridge.global_projection.parameters())


def test_joint_checkpoint_binds_variant_identity_and_rng(tmp_path):
    identity = ensure_run_identity(tmp_path)
    contract = {"schema":JOINT44_RUN_SCHEMA,"run_dir":str(tmp_path),"run_id":identity["run_id"],"repo_root":str(tmp_path),"ar_contract":EDITING_AR_CLAP44_CONTRACT,"variant":"global_and_sequence","m2d_used":False,"independent_test_used":False,"world_size":1,"schedule":{"max_steps":2},"source_sha256":{}}
    atomic_json(tmp_path/"RUN_CONTRACT.json",contract)
    module = nn.Module(); module.diffusion = nn.Linear(3,3); module.ar = nn.Module(); module.ar.adapter = nn.Linear(3,3)
    module.ar.source_clap_model = nn.Linear(3,3).requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad]); scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    sum(p.square().sum() for p in module.parameters() if p.requires_grad).backward()
    optimizer.step(); scheduler.step()
    step = tmp_path/"checkpoints/step-00000001.pt"
    rng = _capture_rank_rng_state(rank=0,device=None)
    rng["torch_cuda_rng_state"] = torch.zeros(8,dtype=torch.uint8)  # serialization fixture; no GPU is accessed
    save_joint_checkpoint(step,module=module,optimizer=optimizer,scheduler=scheduler,step=1,epoch=0,next_batch=4,contract=contract,rng_states=[rng])
    payload,record = load_joint_checkpoint(step,expected_contract=contract,verify_sources=False,require_latest=True)
    assert payload["global_step"] == 1 and payload["quality_gate_passed"] is False
    assert not any(key.startswith("source_clap_model.") for key in payload["editing_ar_specific_state_dict"])
    with pytest.raises(RuntimeError,match="resume contract"):
        load_joint_checkpoint(step,expected_contract={**contract,"variant":"global_only"},verify_sources=False)
    record["step"] = 2; atomic_json(step.parent/"LATEST.json",record)
    with pytest.raises(RuntimeError,match="latest"):
        load_joint_checkpoint(step,verify_sources=False,require_latest=True)


def test_native_pipeline_has_no_external_semantic_or_old_plan_inputs():
    for method in ("generate_new_sceneplans","edit_latents","edit_audio"):
        names = inspect.signature(getattr(ScenePlanTransfusionEditingCLAP44Pipeline,method)).parameters
        assert not any(any(word in name for word in ("old_sceneplan","source_caption","m2d","clap_features","target_audio")) for name in names)


def test_native_pipeline_routes_audio_and_generated_complete_plan():
    class FakeVAE(nn.Module):
        def encoder(self,audio):
            mean = audio.reshape(len(audio),4,-1,1024).mean(-1).repeat(1,16,1)
            return torch.cat((mean,torch.full_like(mean,-12)),1)
        def decode(self,latent): return latent[:,:4].repeat_interleave(1024,-1)
    class FakeAR(nn.Module):
        source_semantic_mode = "latent_only"; source_clap_model = None
        def __init__(self,transformer): super().__init__(); self.shared_transformer = transformer
        def generate_batch(self,source,mask,instructions,**kwargs):
            assert instructions == ["move the sound left"] and int(mask.sum()) == 44
            return [torch.tensor([1,7,2])]
    class Codec:
        def decode(self,ids,*,sample_id):
            assert ids == [1,7,2]
            return {"sample_id":sample_id,"duration_sec":44*1024/44100,"room":{"type":"moderate"},"sources":[{"source_id":"source_0","kind":"sound","description":"a bell","activity":{"onset_sec":.1,"offset_sec":.8},"gain_db":0.,"trajectory":{"type":"static","position":{"azimuth_deg":60.,"elevation_deg":0.,"distance_m":2.}}}]}
    class Pipeline(ScenePlanTransfusionEditingCLAP44Pipeline):
        def sample_edited_latents(self,source,mask,plans,**kwargs):
            assert plans[0]["sources"][0]["trajectory"]["position"]["azimuth_deg"] == 60.
            self.executed_source = source.clone()
            return (source+1)*mask[:,None]
    diffusion = nn.Module(); diffusion.model = nn.Module(); diffusion.model.model = nn.Module()
    diffusion.model.model.transformer = nn.Linear(2,2)
    pipeline = Pipeline(diffusion=diffusion,editing_ar=FakeAR(diffusion.model.model.transformer),codec=Codec(),audio_autoencoder=FakeVAE()).eval()
    audio = torch.randn(1,4,44100)
    result = pipeline.edit_audio(audio,["move the sound left"],model_num_samples=[44100])
    assert result["edited_foa"].shape == (1,4,432*1024)
    assert int(result["sample_attention_mask"].sum()) == 44100
    assert not result["sample_attention_mask"][:,44100:].any()
    assert not result["edited_foa"][:,:,44100:].any()
    assert torch.equal(result["source_foa_latent"],pipeline.executed_source)
    assert not torch.equal(result["source_codec_foa"],result["edited_foa"])
    with pytest.raises(ValueError,match="44.1"):
        pipeline.edit_audio(audio,["move"],model_num_samples=[44100],sample_rate=48000)
    with pytest.raises(ValueError,match="exact valid-frame"):
        pipeline.generate_new_sceneplans(result["source_foa_latent"],torch.ones(1,432,dtype=torch.bool),["move"],duration_sec=1.)
