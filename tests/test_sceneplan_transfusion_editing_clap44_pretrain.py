"""CPU/Gloo equivalence and recovery tests for the real CLAP44 training step."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import load_clap44_checkpoint
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import (
    load_training_checkpoint, optimizer_and_scheduler, resolve_training_resume, save_checkpoint, sha256, training_step,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_clap44 import rng_state, restore_rng


TRAINING = {"learning_rate":.001,"max_steps":6,"warmup_steps":1,"scene_weight":.25,"binding_weight":.1}


class Text(nn.Module):
    @torch.no_grad()
    def forward(self,texts,device):
        return torch.stack([torch.randn(12,generator=torch.Generator().manual_seed(
            int(hashlib.sha256(text.encode()).hexdigest()[:15],16))) for text in texts]).to(device)


def model(dropout=0.):
    return EditingCLAP44(CLAP44Config(width=32,layers=1,heads=4,semantic_dim=16,scene_dim=8,text_dim=12,dropout=dropout))


def batch(rank=None):
    rows = list(range(8)) if rank is None else list(range(rank*4,rank*4+4))
    labels = [{"pair_id":f"pair-{i//2}","role":"source" if i%2==0 else "target",
        "semantic_key":f"content-{i//2}","scene_key":f"scene-{i}",
        "semantic_text":f"sound content {i//2}","scene_text":f"sound content {i//2} at position {i}",
        "asset_ids":[f"asset-{i//2}"],"content_ids":[f"kind-{i//2}"]} for i in rows]
    owners = [i for i in (4,4,7) if i in rows]
    return {"latent":torch.randn(8,64,20,generator=torch.Generator().manual_seed(5))[rows],
        "mask":torch.ones(len(rows),20,dtype=torch.bool),"labels":labels,
        "negative_scene_texts":[f"incorrect bound position {i}-{j}" for j,i in enumerate((4,4,7)) if i in rows],
        "negative_owners":torch.tensor([rows.index(i) for i in owners],dtype=torch.long)}


@pytest.fixture
def own_run(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_rng_state",lambda:torch.zeros(8,dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda,"set_rng_state",lambda value:None)
    source = tmp_path/"bound.py"
    source.write_text("fixture = 1\n")
    torch.manual_seed(4)
    core = model(dropout=.2)
    optimizer,scheduler = optimizer_and_scheduler(core,TRAINING)
    contract = {"schema":"editing_clap44_gpu_preflight_v1","config":{"model":asdict(core.config),"training":TRAINING},
        "world_size":1,"source_sha256":{str(source):sha256(source)}}
    (tmp_path/"TRAIN_CONTRACT.json").write_text(json.dumps(contract)+"\n")
    def advance(step):
        return training_step(core,Text(),optimizer,scheduler,batch(),step=step,training=TRAINING,device="cpu")
    def save(step):
        return save_checkpoint(tmp_path/f"step-{step:06d}.pt",core,optimizer,scheduler,step,0,step,contract,[rng_state()])
    return tmp_path,contract,core,optimizer,scheduler,advance,save


def test_resume_reproduces_dropout_optimizer_scheduler_and_rng(own_run):
    root,contract,core,optimizer,scheduler,advance,save = own_run
    for step in range(2): advance(step)
    saved = save(2)
    expected_losses = [advance(step) for step in range(2,4)]
    expected = deepcopy(core.state_dict())
    expected_optimizer = deepcopy(optimizer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    restored = model(dropout=.2)
    opt,schedule = optimizer_and_scheduler(restored,TRAINING)
    payload,manifest = load_training_checkpoint(saved["checkpoint"],contract=contract)
    assert manifest==saved
    restored.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    schedule.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng_states"][0])
    actual_losses = [training_step(restored,Text(),opt,schedule,batch(),step=step,training=TRAINING,device="cpu") for step in range(2,4)]
    for left,right in zip(expected_losses,actual_losses):
        for key in left: torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
    for key,tensor in expected.items(): torch.testing.assert_close(tensor,restored.state_dict()[key],rtol=0,atol=0)
    for key,state in expected_optimizer["state"].items():
        for name,value in state.items(): torch.testing.assert_close(value,opt.state_dict()["state"][key][name],rtol=0,atol=0)
    assert schedule.state_dict()==scheduler.state_dict()
    assert torch.equal(torch.get_rng_state(),expected_rng)
    # A profiling checkpoint is intentionally rejected as a deployable encoder.
    with pytest.raises(ValueError,match="foreign or unapproved"):
        load_clap44_checkpoint(saved["checkpoint"],expected_sha256=saved["sha256"])


@pytest.mark.parametrize("failure",["missing_manifest","missing_latest","partial_latest","stale_latest"])
def test_checkpoint_publication_recovers_without_rewriting_candidates(own_run,failure):
    root,contract,_,_,_,advance,save = own_run
    advance(0)
    first = save(1)
    advance(1)
    second = save(2)
    path = Path(second["checkpoint"])
    before = path.stat().st_mtime_ns
    if failure=="missing_manifest": path.with_suffix(".manifest.json").unlink()
    elif failure=="missing_latest": (root/"LATEST.json").unlink()
    elif failure=="partial_latest": (root/"LATEST.json").write_text("{")
    elif failure=="stale_latest": (root/"LATEST.json").write_text(json.dumps(first))
    assert resolve_training_resume(root)==second
    assert path.stat().st_mtime_ns==before and sha256(first["checkpoint"])==first["sha256"]
    assert load_training_checkpoint(path,contract=contract)[1]==second
    with pytest.raises(RuntimeError,match="never overwrite"):
        save(2)
    with pytest.raises(RuntimeError,match="never overwrite"):
        save(1)


@pytest.mark.parametrize("failure",["weights","optimizer","scheduler","rng","source"])
def test_checkpoint_recovery_refuses_corruption_and_stale_state(own_run,failure):
    root,contract,_,_,_,advance,save = own_run
    advance(0)
    saved = save(1)
    path = Path(saved["checkpoint"])
    if failure=="source":
        (root/"bound.py").write_text("fixture = 2\n")
    else:
        payload = torch.load(path,weights_only=True)
        if failure=="weights": next(iter(payload["model"].values())).add_(.1)
        elif failure=="optimizer": next(iter(payload["optimizer"]["state"].values()))["step"].zero_()
        elif failure=="scheduler": payload["scheduler"]["last_epoch"]=0
        elif failure=="rng": payload["rng_states"]=[]
        torch.save(payload,path)
    original = path.with_suffix(".manifest.json").read_bytes()
    with pytest.raises(RuntimeError): resolve_training_resume(root)
    assert path.with_suffix(".manifest.json").read_bytes()==original


def test_partial_temporary_snapshots_are_not_adopted(own_run):
    root,_,_,_,_,_,_ = own_run
    partial = root/"step-000001.pt.tmp.123"
    partial.write_bytes(b"interrupted bytes")
    assert resolve_training_resume(root) is None
    assert partial.read_bytes()==b"interrupted bytes"
    with pytest.raises(RuntimeError,match="finalized candidate"):
        load_training_checkpoint(partial,contract={})


def test_epoch_data_iterators_do_not_consume_model_rng():
    values = [torch.ones(2),torch.zeros(2)]
    loader_generator = torch.Generator()
    loader = DataLoader(values,batch_size=1,generator=loader_generator)
    torch.manual_seed(9)
    initial = torch.get_rng_state().clone()
    for epoch in range(2):
        loader_generator.manual_seed(42+900001+epoch*5)
        list(loader)
        assert torch.equal(torch.get_rng_state(),initial)


def _ddp_worker(rank,init_path,output,dtype_name):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+init_path,rank=rank,world_size=2)
    torch.manual_seed(8)
    dtype = getattr(torch,dtype_name)
    core = model().to(dtype)
    wrapped = DistributedDataParallel(core)
    opt,schedule = optimizer_and_scheduler(wrapped,TRAINING)
    # Mature binding includes three local negatives on rank 1 and none on rank 0.
    training_step(wrapped,Text(),opt,schedule,batch(rank),step=3,training=TRAINING,device="cpu")
    gradients = {key:value.grad.clone() for key,value in core.named_parameters()}
    dist.destroy_process_group()
    torch.manual_seed(8)
    reference = model().to(dtype)
    ref_opt,ref_schedule = optimizer_and_scheduler(reference,TRAINING)
    training_step(reference,Text(),ref_opt,ref_schedule,batch(),step=3,training=TRAINING,device="cpu")
    differences = {}
    for key,value in reference.named_parameters():
        torch.testing.assert_close(gradients[key],value.grad,atol=3e-6,rtol=3e-5)
        delta = (core.state_dict()[key]-value).abs()
        if delta.max()>3e-6:
            index = int(delta.flatten().argmax())
            differences[key] = {"max_update_difference":float(delta.max()),"flat_index":index,
                "distributed_gradient":float(gradients[key].flatten()[index]),
                "global_gradient":float(value.grad.flatten()[index])}
        # The mathematical gradient comparison includes production FP32.
        # Double precision also tests Adam updates: a near-zero attention key
        # bias gradient can otherwise be amplified by Adam's epsilon despite
        # the corresponding softmax being invariant to that bias.
        if dtype==torch.float64:
            torch.testing.assert_close(core.state_dict()[key],value,atol=3e-6,rtol=3e-5)
    Path(output,f"rank-{rank}-numerics.json").write_text(json.dumps(differences,indent=2))
    Path(output,f"rank-{rank}.ok").write_text("global contrastive and unequal binding gradients agree\n")


@pytest.mark.parametrize("dtype_name",["float32","float64"])
def test_distributed_real_training_step_matches_global_gradient(tmp_path,dtype_name):
    torch.multiprocessing.spawn(_ddp_worker,args=(str(tmp_path/"init"),str(tmp_path),dtype_name),nprocs=2,join=True)
    assert len(list(tmp_path.glob("rank-*.ok")))==2
