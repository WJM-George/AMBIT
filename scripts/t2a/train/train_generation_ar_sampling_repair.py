#!/usr/bin/env python3
"""A bounded, full-data sampler intervention on the existing frozen P10 AR.

This is a new experiment contract. Initialization from an earlier checkpoint
is not represented as resuming the old experiment or its official selection.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SNAPSHOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/source_snapshots/snapshot")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SNAPSHOT))
from generation_ar_sampling import ShuffledGlobalBatchSampler

import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (
    GenerationARSQLiteDataset, LengthBucketDistributedSampler, collate_generation_ar,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar

CACHE = Path(os.environ.get("AMBIT_CACHE_ROOT", "cache")) / "generation_ar_manifests"
ACCEPTANCE = Path("reports/generation_ar_goal_acceptance.md")
PARENT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/run")
CODEC = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def frozen_trainer():
    spec = importlib.util.spec_from_file_location('original_frozen_ar_trainer', SNAPSHOT / 'scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_digest(named):
    h = hashlib.sha256()
    for name, tensor in named:
        h.update(name.encode())
        h.update(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--initialize-step', type=int, choices=(66672,70839,83340), required=True)
    parser.add_argument('--gate', action='store_true', help='Only three tiny training updates and save/reload checks; not a training sweep.')
    parser.add_argument('--gate-proof', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2' or os.environ.get('WORLD_SIZE') != '3':
        raise RuntimeError('This experiment requires precisely GPU 0-2 in three-rank DDP')
    if not 0 < args.learning_rate <= 3e-5 or args.max_wall_seconds <= 0:
        raise ValueError('invalid bounded repair budget')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda',local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    run = args.run_dir.resolve()
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        (run / 'checkpoints').mkdir(exist_ok=True)
    dist.barrier()
    helper = frozen_trainer()
    parent_contract = json.loads((PARENT / 'RUN_CONTRACT.json').read_text())
    selection = json.loads((PARENT / 'CHECKPOINT_SELECTION.json').read_text())
    candidate = next(c for c in selection['candidates'] if c['step'] == args.initialize_step)
    preflight = [None]
    if rank == 0:
        assert sha(candidate['checkpoint']) == candidate['checkpoint_sha256']
        for name,digest in parent_contract['source_sha256'].items():
            assert sha(SNAPSHOT / name) == digest, name
        cache_record = json.loads((CACHE / 'CACHE.json').read_text())
        for split in ('train','validation'):
            assert sha(CACHE / f'{split}.sqlite') == parent_contract[f'{split}_manifest_sha256']
            assert cache_record['manifests'][split]['sha256'] == parent_contract[f'{split}_manifest_sha256']
        assert sha(CACHE / 'train_columns.npz') == cache_record['columns_sha256']
        if not args.gate:
            if args.gate_proof is None:
                raise RuntimeError('full experiment requires passing gate proof')
            gate = json.loads(args.gate_proof.read_text())
            assert gate['status'] == 'PASS' and gate['initialize_checkpoint_sha256'] == candidate['checkpoint_sha256']
            assert gate['trainer_sha256'] == sha(Path(__file__))
            assert gate['sampler_repair_sha256'] == sha(REPO / 'scripts/t2a/train/generation_ar_sampling.py')
        preflight[0] = {'cache': cache_record, 'candidate': {k:candidate[k] for k in ('step','checkpoint','checkpoint_sha256')}}
    dist.broadcast_object_list(preflight,src=0,device=device)
    codec = ModelScenePlanCodecV4(CODEC)
    model, report = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=rank==0, activation_checkpointing=False)
    assert report.as_dict() == parent_contract['p10_load'] and codec.fingerprint == parent_contract['codec_fingerprint']
    initial = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False)
    assert initial['run_contract'] == parent_contract and initial['global_step'] == args.initialize_step
    model.load_trainable_state_dict(initial['ar_adapter'])
    del initial
    model.p10_dit.to(device=device,dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device,dtype=torch.bfloat16)
    model.ar_adapter.to(device=device,dtype=torch.float32)
    model.train()
    parameter_contract = helper._parameter_contract(model)
    columns = np.load(CACHE / 'train_columns.npz')
    lengths, source_counts = columns['lengths'], columns['source_counts']
    if args.gate:
        # Fixed 48 rows per count across three ranks, including difficult four-source requests.
        ordinals = np.concatenate([np.flatnonzero(source_counts==count)[:48] for count in range(1,5)])
        dataset = GenerationARSQLiteDataset(CACHE / 'train.sqlite',split='train',row_ordinals=ordinals)
        base = LengthBucketDistributedSampler(lengths[ordinals],num_replicas=3,rank=rank,batch_size=64)
        steps, warmup = 3, 0
    else:
        ordinals = None
        dataset = GenerationARSQLiteDataset(CACHE / 'train.sqlite',split='train')
        base = LengthBucketDistributedSampler(lengths,num_replicas=3,rank=rank,batch_size=64)
        steps, warmup = 8334, 100
    sampler = ShuffledGlobalBatchSampler(base)
    sampler.set_epoch(10)
    loader = DataLoader(dataset,batch_size=64,sampler=sampler,num_workers=2 if args.gate else 8,
                        collate_fn=functools.partial(collate_generation_ar,pad_id=codec.pad_id),
                        pin_memory=True,drop_last=False,persistent_workers=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: helper._cosine_floor_multiplier(step,schedule_steps=steps,warmup_steps=warmup,floor_ratio=.1))
    source_files = [Path(__file__),REPO / 'scripts/t2a/train/generation_ar_sampling.py',
                    SNAPSHOT / 'scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py']
    contract = {'schema':'generation_ar_sampling_repair_v1','mode':'tiny_correctness_gate' if args.gate else 'full_data_intervention',
                'initialize_from':preflight[0]['candidate'],'official_parent_selection_overridden':False,
                'initialization_note':'Existing completed-run candidate initializes a new experiment; it is not declared a new official selected model.',
                'original_source_snapshot':str(SNAPSHOT),'source_sha256':{str(p):sha(p) for p in source_files},
                'manifest_sha256':{s:parent_contract[f'{s}_manifest_sha256'] for s in ('train','validation')},
                'train_cache':str(CACHE / 'train.sqlite'),'columns_sha256':preflight[0]['cache']['columns_sha256'],
                'acceptance_v1':str(ACCEPTANCE),'acceptance_sha256':sha(ACCEPTANCE),
                'hypothesis':'Permuting complete global batches removes long source-count-conditioned runs while retaining padding efficiency and exact coverage.',
                'loss':'unchanged global non-padding token mean CE; no count/field reweighting in R1',
                'sampler':'legacy batch membership, independent shared permutation of global batches; unchanged uneven tail',
                'seed':42,'sampler_epoch':10,'world_size':3,'cuda_visible_devices':'0,1,2','batch_size_per_rank':64,
                'rows_per_epoch':len(dataset),'dropped_rows':0,'duplicated_rows':0,'steps':steps,'warmup_steps':warmup,
                'learning_rate':args.learning_rate,'weight_decay':.01,'schedule':'cosine floor .1',
                'optimizer_state':'fresh AdamW; no inherited momentum','max_wall_seconds':args.max_wall_seconds,
                'codec_fingerprint':codec.fingerprint,'p10_load':report.as_dict(),'parameter_contract':parameter_contract,
                'gate_proof_sha256':sha(args.gate_proof) if args.gate_proof is not None else None,
                'gate_ordinals':ordinals.tolist() if ordinals is not None else []}
    contract_path = run / 'RUN_CONTRACT.json'
    if rank==0:
        if contract_path.exists():
            assert json.loads(contract_path.read_text())==contract
        else:
            atomic(contract_path,contract)
    dist.barrier()
    start_step = 0
    if args.resume:
        resumed = torch.load(args.resume,map_location='cpu',weights_only=False)
        assert resumed['run_contract']==contract
        model.load_trainable_state_dict(resumed['ar_adapter'])
        optimizer.load_state_dict(resumed['optimizer']);scheduler.load_state_dict(resumed['scheduler'])
        start_step = int(resumed['global_step'])
        assert 0<=start_step<steps
        del resumed
    frozen_before = tensor_digest(model.p10_dit.named_parameters()) if args.gate else None
    qwen_before = tensor_digest(model.prompt_conditioner.model.named_parameters()) if args.gate else None
    condition_before = tensor_digest(model.prompt_conditioner.named_parameters()) if args.gate else None
    adapter_before = tensor_digest(model.ar_adapter.named_parameters()) if args.gate else None
    wrapped = DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    local_tokens = 0; local_loss_sum = 0.; local_rows = 0
    global_step = start_step
    fixed_probe = None
    repeats = steps if args.gate else 1
    for repeat in range(repeats):
        for batch_index, raw in enumerate(loader):
            if not args.gate and batch_index < start_step:
                continue
            batch = helper._move_batch(raw,device)
            with torch.no_grad():
                context,mask = model.encode_requests(batch['raw_user_requests'],device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits = wrapped(batch['plan_input_ids'],batch['plan_attention_mask'],context,mask)
                loss_sum = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),batch['plan_labels'].reshape(-1),ignore_index=-100,reduction='sum')
            tokens = int((batch['plan_labels']!=-100).sum().item())
            global_tokens = torch.tensor([tokens],device=device,dtype=torch.float64)
            dist.all_reduce(global_tokens)
            loss = loss_sum * (3. / global_tokens.item())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(norm):
                raise RuntimeError('nonfinite repair gradient')
            optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            global_step += 1
            local_tokens += tokens;local_loss_sum += float(loss_sum.detach());local_rows += len(batch['raw_user_requests'])
            if args.gate:
                fixed_probe=(batch['plan_input_ids'],batch['plan_attention_mask'],context,mask)
            if global_step % 100==0 or global_step==steps:
                stats=torch.tensor([local_loss_sum,local_tokens,local_rows],device=device,dtype=torch.float64)
                dist.all_reduce(stats)
                if rank==0:
                    status={'status':'RUNNING','step':global_step,'steps':steps,'elapsed_s':time.monotonic()-started,
                            'mean_loss_since_start':stats[0].item()/stats[1].item(),'global_rows_since_start':int(stats[2].item()),
                            'global_tokens_per_s':stats[1].item()/max(1.,time.monotonic()-started),'learning_rate':scheduler.get_last_lr()[0]}
                    atomic(run/'STATUS.json',status)
                    with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
                    print(json.dumps(status),flush=True)
            if global_step in (4167,steps):
                dist.barrier()
                if rank==0:
                    helper._save_checkpoint(run/'checkpoints'/f'step_{global_step:08d}.pt',model=model,optimizer=optimizer,scheduler=scheduler,
                                            global_step=global_step,epoch=0,batch_in_epoch=global_step,run_contract=contract)
                dist.barrier()
            elapsed = torch.tensor([time.monotonic()-started],device=device)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            if elapsed.item()>args.max_wall_seconds and global_step<steps:
                if rank==0:
                    helper._save_checkpoint(run/'checkpoints'/f'budget_stop_{global_step:08d}.pt',model=model,optimizer=optimizer,scheduler=scheduler,
                                            global_step=global_step,epoch=0,batch_in_epoch=global_step,run_contract=contract)
                    atomic(run/'STATUS.json',{'status':'BUDGET_STOP','step':global_step,'elapsed_s':elapsed.item()})
                dist.barrier();dist.destroy_process_group();return 3
            if global_step==steps:break
        if global_step==steps:break
    assert global_step==steps
    if args.gate:
        assert tensor_digest(model.p10_dit.named_parameters())==frozen_before
        assert tensor_digest(model.prompt_conditioner.model.named_parameters())==qwen_before
        assert tensor_digest(model.prompt_conditioner.named_parameters())==condition_before
        assert tensor_digest(model.ar_adapter.named_parameters())!=adapter_before
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            before_reload=model(*fixed_probe)
        saved=torch.load(run/'checkpoints'/f'step_{steps:08d}.pt',map_location='cpu',weights_only=False)
        model.load_trainable_state_dict(saved['ar_adapter'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            after_reload=model(*fixed_probe)
        torch.testing.assert_close(before_reload,after_reload,rtol=0,atol=0)
        if rank==0:
            atomic(run/'GATE.json',{'status':'PASS','initialize_checkpoint_sha256':candidate['checkpoint_sha256'],
                                   'trainer_sha256':sha(Path(__file__)),'sampler_repair_sha256':sha(REPO/'scripts/t2a/train/generation_ar_sampling.py'),
                                   'frozen_p10_unchanged':True,'frozen_qwen_unchanged':True,'frozen_conditioning_unchanged':True,
                                   'ar_weights_updated':True,'checkpoint_reload_logits_exact':True,'steps':steps})
    if rank==0:
        atomic(run/'FINAL.json',{'status':'COMPLETE','steps':steps,'elapsed_s':time.monotonic()-started,'run_contract_sha256':sha(contract_path)})
        atomic(run/'STATUS.json',{'status':'COMPLETE','step':steps,'elapsed_s':time.monotonic()-started})
    dataset.close()
    dist.barrier();dist.destroy_process_group()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
