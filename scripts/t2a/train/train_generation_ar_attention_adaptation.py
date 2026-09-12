#!/usr/bin/env python3
"""Bounded AR-only attention adaptation with exact global-token accumulation."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SNAPSHOT=Path('/mnt/sdc/ckpts/transfusion_sceneplan/generation_ar/source_snapshots/generation_ar_attention_adaptation_20260905_v2')
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SNAPSHOT))
from generation_ar_sampling import ShuffledGlobalBatchSampler
import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import GenerationARSQLiteDataset,LengthBucketDistributedSampler,collate_generation_ar
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar

CACHE=Path('/dev/shm/generation_ar_manifests_20260905')
PARENT=Path('/mnt/sdc/ckpts/transfusion_sceneplan/generation_ar/sampling_repair_1ep_20260905_v1/training')
CHECKPOINT=PARENT/'checkpoints/step_00008334.pt'
CHECKPOINT_SHA='c2bdeef77f8c50ee2ffe9090ade6177cad8fce18dcf843c137ebc8f55644ceb7'
CODEC=Path('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
ACCEPTANCE=Path('/home/tanhe/dataset_storage/reports/generation_ar_goal_acceptance_20260905.md')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n');temp.replace(path)


def tensor_digest(named):
    h=hashlib.sha256()
    for name,value in named:
        h.update(name.encode());h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def grouped_batches(loader,accumulation):
    pending=[]
    for batch in loader:
        pending.append(batch)
        if len(pending)==accumulation:yield pending;pending=[]
    if pending:yield pending


def save_checkpoint(path,model,optimizer,scheduler,step,contract):
    value={'global_step':step,'epoch':0,'batch_in_epoch':step,'run_contract':contract,
           'ar_adapter':model.trainable_state_dict(),'ar_lora':model.lora_state_dict(),
           'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'rng_state_torch_rank0':torch.get_rng_state(),'rng_state_cuda_rank0':torch.cuda.get_rng_state()}
    temp=path.with_name(path.name+'.tmp');torch.save(value,temp);temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--binding-strength',type=float,choices=(0.,1.,2.),required=True)
    parser.add_argument('--micro-batch-size',type=int,choices=(16,32),default=32)
    parser.add_argument('--steps',type=int,default=2084)
    parser.add_argument('--max-wall-seconds',type=int,default=5400)
    parser.add_argument('--gate',action='store_true');parser.add_argument('--gate-proof',type=Path)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2' and os.environ.get('WORLD_SIZE')=='3'
    assert 1<=args.steps<=2084 and 0<args.max_wall_seconds<=5400
    rank,local_rank=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',device_id=device)
    torch.manual_seed(42+rank);np.random.seed(42+rank)
    run=args.run_dir.resolve()
    if rank==0:run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
    dist.barrier()
    parent=json.loads((PARENT/'RUN_CONTRACT.json').read_text())
    sampler_path=Path(__file__).with_name('generation_ar_sampling.py')
    source_paths=[Path(__file__),sampler_path,SNAPSHOT/'SOURCE_SNAPSHOT_MANIFEST.json',
                  SNAPSHOT/'stable_audio_tools/models/sceneplan_generation_ar_lora.py',
                  SNAPSHOT/'stable_audio_tools/models/sceneplan_generation_ar_source_binding.py',
                  SNAPSHOT/'stable_audio_tools/inference/sceneplan_generation_ar_precision.py']
    source_hashes={str(p):sha(p) for p in source_paths}
    if rank==0:
        assert sha(CHECKPOINT)==CHECKPOINT_SHA
        for split,digest in parent['manifest_sha256'].items():assert sha(CACHE/(split+'.sqlite'))==digest
        if not args.gate:
            if args.gate_proof is None:raise ValueError('formal adaptation requires successful distributed gate')
            proof=json.loads(args.gate_proof.read_text())
            assert proof['status']=='PASS' and proof['source_sha256']==source_hashes
            assert proof['binding_strength']==args.binding_strength and proof['micro_batch_size']==args.micro_batch_size
    dist.barrier()
    codec=ModelScenePlanCodecV4(CODEC)
    base,report=load_p10v11_generation_ar(pad_id=codec.pad_id,verify_sha256=rank==0,activation_checkpointing=False)
    assert report.as_dict()==parent['p10_load'] and codec.fingerprint==parent['codec_fingerprint']
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);assert state['run_contract']==parent
    base.load_trainable_state_dict(state['ar_adapter']);del state
    model=AdaptedGenerationAR(base,codec,rank=8,alpha=8.,binding_strength=args.binding_strength);del base
    model.p10_dit.to(device=device,dtype=torch.float32);model.prompt_conditioner.to(device=device,dtype=torch.bfloat16)
    model.ar_adapter.to(device=device,dtype=torch.float32);model.ar_lora.to(device=device,dtype=torch.float32)
    configure_float32_ar(model);model.train()
    parameter_contract=model.adaptation_contract()
    columns=np.load(CACHE/'train_columns.npz');lengths,counts=columns['lengths'],columns['source_counts']
    if args.gate:
        # Stress the maximum available lengths in each count group, not only short rows.
        ordinals=np.concatenate([sorted(np.flatnonzero(counts==n),key=lambda i:(-int(lengths[i]),int(i)))[:48] for n in range(1,5)])
        dataset=GenerationARSQLiteDataset(CACHE/'train.sqlite',split='train',row_ordinals=ordinals)
        base_sampler=LengthBucketDistributedSampler(lengths[ordinals],num_replicas=3,rank=rank,batch_size=64)
        steps,warmup=3,0
    else:
        ordinals=None;dataset=GenerationARSQLiteDataset(CACHE/'train.sqlite',split='train')
        base_sampler=LengthBucketDistributedSampler(lengths,num_replicas=3,rank=rank,batch_size=64)
        steps,warmup=args.steps,100
    sampler=ShuffledGlobalBatchSampler(base_sampler);sampler.set_epoch(11)
    accumulation=64//args.micro_batch_size
    loader=DataLoader(dataset,batch_size=args.micro_batch_size,sampler=sampler,num_workers=2 if args.gate else 8,
                       collate_fn=functools.partial(collate_generation_ar,pad_id=codec.pad_id),pin_memory=True,
                       drop_last=False,persistent_workers=True)
    optimizer=torch.optim.AdamW([{'params':model.ar_adapter.parameters(),'lr':1e-5},
                                {'params':model.ar_lora.parameters(),'lr':1e-4}],weight_decay=.01)
    helper_path=SNAPSHOT/'scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py'
    spec=importlib.util.spec_from_file_location('adaptation_frozen_helper',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:helper._cosine_floor_multiplier(step,schedule_steps=steps,warmup_steps=warmup,floor_ratio=.1))
    contract={'schema':'generation_ar_attention_adaptation_v1','mode':'distributed_correctness_gate' if args.gate else 'bounded_quarter_epoch',
              'initialize_checkpoint':str(CHECKPOINT),'initialize_checkpoint_sha256':CHECKPOINT_SHA,'parent_contract_sha256':sha(PARENT/'RUN_CONTRACT.json'),
              'source_snapshot':str(SNAPSHOT),'source_sha256':source_hashes,'manifest_sha256':parent['manifest_sha256'],
              'acceptance':str(ACCEPTANCE),'acceptance_sha256':sha(ACCEPTANCE),'test_used':False,'official_selection_modified':False,
              'hypothesis':'The frozen audio attention blocks lack sufficient AR-specific transformation capacity; train small AR-only residuals to improve source-content and numeric binding.',
              'parameter_contract':parameter_contract,'codec_fingerprint':codec.fingerprint,'p10_load':report.as_dict(),
              'precision':'FP32 AR including math SDPA; BF16 request encoder; TF32 disabled; native P10 caller path unchanged',
              'loss':'global nonpadding token CE; no field reweighting','optimizer':'fresh AdamW','adapter_lr':1e-5,'lora_lr':1e-4,'weight_decay':.01,
              'steps':steps,'warmup_steps':warmup,'schedule':'cosine floor .1','world_size':3,'seed':42,'sampler_epoch':11,
              'micro_batch_size':args.micro_batch_size,'accumulation':accumulation,'global_batch_size':192,
              'sampling':'first planned global batches of the repaired full-epoch permutation, no repeats; unvisited rows are intentionally outside this bounded stage',
              'planned_presentations':576 if args.gate else steps*192,'max_wall_seconds':args.max_wall_seconds,
              'gate_ordinals':ordinals.tolist() if ordinals is not None else [],'gate_proof_sha256':sha(args.gate_proof) if args.gate_proof else None,
              'success_rule':'versus the matched R2 inference policy, mean class-3/4 core-semantic recall or medium spatiotemporal joint rate improves >=5 percentage points; no class semantic or joint regression >2 points; every count class >=95% with explicit-count policy; final acceptance unchanged'}
    if rank==0:
        if (run/'RUN_CONTRACT.json').exists():assert json.loads((run/'RUN_CONTRACT.json').read_text())==contract
        else:atomic(run/'RUN_CONTRACT.json',contract)
    dist.barrier();start_step=0
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False);assert state['run_contract']==contract
        model.load_trainable_state_dict(state['ar_adapter']);model.load_lora_state_dict(state['ar_lora'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_step=int(state['global_step']);del state
        assert 0<=start_step<steps
    frozen_before={name:tensor_digest(module.named_parameters()) for name,module in [('p10',model.p10_dit),('conditioner',model.prompt_conditioner),('qwen',model.prompt_conditioner.model)]} if args.gate else None
    adapter_before=tensor_digest(model.ar_adapter.named_parameters()) if args.gate else None
    # DDP broadcasts rank-0 LoRA initialization before gradients are accumulated.
    wrapped=DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False)
    lora_before=tensor_digest(model.ar_lora.named_parameters()) if args.gate else None
    trainable=[p for p in model.parameters() if p.requires_grad]
    started=time.monotonic();global_step=start_step;local_tokens=local_rows=0;local_loss=0.;probe=None
    optimizer.zero_grad(set_to_none=True)
    for repeat in range(steps if args.gate else 1):
        for batch_index,raw_group in enumerate(grouped_batches(loader,accumulation)):
            if not args.gate and batch_index<start_step:continue
            local_denominator=sum(int((raw['plan_labels']!=-100).sum()) for raw in raw_group)
            denominator=torch.tensor(float(local_denominator),device=device,dtype=torch.float64);dist.all_reduce(denominator)
            for micro,raw in enumerate(raw_group):
                batch=helper._move_batch(raw,device)
                with torch.no_grad():
                    context,mask=model.encode_requests(batch['raw_user_requests'],device=device)
                    roles=model.request_source_ids(batch['raw_user_requests'],device=device)
                sync=wrapped.no_sync() if micro<len(raw_group)-1 else nullcontext()
                with sync:
                    logits=wrapped(batch['plan_input_ids'],batch['plan_attention_mask'],context,mask,roles)
                    loss_sum=F.cross_entropy(logits.reshape(-1,4096),batch['plan_labels'].reshape(-1),ignore_index=-100,reduction='sum')
                    (loss_sum*(3./denominator.item())).backward()
                local_loss+=float(loss_sum.detach());local_tokens+=int((batch['plan_labels']!=-100).sum());local_rows+=len(batch['raw_user_requests'])
                if args.gate:probe=(batch['plan_input_ids'],batch['plan_attention_mask'],context,mask,roles)
                del logits,loss_sum
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite adaptation gradient')
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
            if global_step%25==0 or global_step==steps or args.gate:
                stats=torch.tensor([local_loss,local_tokens,local_rows,torch.cuda.max_memory_allocated(device)],device=device,dtype=torch.float64)
                dist.all_reduce(stats[:3]);dist.all_reduce(stats[3:],op=dist.ReduceOp.MAX)
                if rank==0:
                    status={'status':'RUNNING','step':global_step,'steps':steps,'elapsed_s':time.monotonic()-started,
                            'global_rows_since_start':int(stats[2]),'global_tokens_per_s':float(stats[1])/max(1.,time.monotonic()-started),
                            'mean_loss_since_start':float(stats[0]/stats[1]),'learning_rates':scheduler.get_last_lr(),'peak_gpu_bytes_max_rank':int(stats[3])}
                    atomic(run/'STATUS.json',status)
                    with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
                    print(json.dumps(status),flush=True)
            if global_step in (max(1,steps//2),steps):
                dist.barrier()
                if rank==0:save_checkpoint(run/'checkpoints'/f'step_{global_step:08d}.pt',model,optimizer,scheduler,global_step,contract)
                dist.barrier()
            elapsed=torch.tensor(time.monotonic()-started,device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            if elapsed.item()>args.max_wall_seconds and global_step<steps:
                if rank==0:
                    save_checkpoint(run/'checkpoints'/f'budget_stop_{global_step:08d}.pt',model,optimizer,scheduler,global_step,contract)
                    atomic(run/'STATUS.json',{'status':'BUDGET_STOP','step':global_step,'elapsed_s':elapsed.item()})
                dist.barrier();dist.destroy_process_group();return 3
            if global_step==steps:break
        if global_step==steps:break
    assert global_step==steps
    if args.gate:
        for name,module in [('p10',model.p10_dit),('conditioner',model.prompt_conditioner),('qwen',model.prompt_conditioner.model)]:assert tensor_digest(module.named_parameters())==frozen_before[name]
        assert tensor_digest(model.ar_adapter.named_parameters())!=adapter_before and tensor_digest(model.ar_lora.named_parameters())!=lora_before
        model.eval()
        with torch.no_grad():before_reload=model(*probe)
        saved=torch.load(run/'checkpoints'/f'step_{steps:08d}.pt',map_location='cpu',weights_only=False)
        model.load_trainable_state_dict(saved['ar_adapter']);model.load_lora_state_dict(saved['ar_lora'])
        with torch.no_grad():after_reload=model(*probe)
        assert torch.equal(before_reload,after_reload)
        if rank==0:atomic(run/'GATE.json',{'status':'PASS','source_sha256':source_hashes,'binding_strength':args.binding_strength,'micro_batch_size':args.micro_batch_size,
                                         'frozen_p10_qwen_conditioner_unchanged':True,'ar_adapter_and_lora_updated':True,'checkpoint_reload_exact':True,'steps':steps})
    if rank==0:atomic(run/'STATUS.json',{'status':'COMPLETE','step':steps,'elapsed_s':time.monotonic()-started,'goal_complete':False})
    dataset.close();dist.barrier();dist.destroy_process_group();return 0


if __name__=='__main__':raise SystemExit(main())
