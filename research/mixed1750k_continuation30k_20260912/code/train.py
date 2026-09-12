"""A persistent 50k→80k DiT continuation, with complete recoverable milestones."""
import argparse
from datetime import timedelta
import math
import os
from pathlib import Path
import sys
import time
from common import *
sys.path[:0]=[str(MAIN),str(OLD_RUN)]
import numpy as np
import torch

def lr_at(step,plan):
    warmup=plan['lr_schedule']['warmup_steps'];peak=plan['optimizer']['lr'];floor=5e-6
    if step<warmup:return peak*(0.1+0.9*(step+1)/warmup)
    fraction=(step-warmup)/(30000-warmup-1)
    return floor+(peak-floor)*0.5*(1+math.cos(math.pi*fraction))

def train(technical=False,resume=None):
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from throughput_runtime_v1 import load_runtime
    from generation_runtime import parameter_digest,EditingForward
    from editing_conditioner_only_state_v1 import all_parameter_state,copy_all_parameters,buffer_digest
    from probe_unfrozen import cpu_tree,rng
    from train_generation import restore_rng
    from mixed_inputs import MixedStream,weighted_rf_loss

    run=ROOT/'training';plan=read(run/'PLAN.json')
    assert read(ROOT/'DATA_READY.json')['counts']=={'train':1750000,'validation':35000,'test':8750}
    assert plan['schema']=='editing_mixed1750k_training_plan_v1' and plan['physical_gpus']==GPUS
    assert sha(run/'PLAN.json')==read(run/'PREPARED.json')['plan_sha256']
    assert plan['checkpoint_optimizer_steps']==[55000,60000,65000,70000,75000,80000]
    if not technical:
        review=read(run/'TECHNICAL_REVIEW.json')
        assert review['status']=='PASS_TWO_BUCKET_DDP_UPDATES_FROM_50K' and review['plan_sha256']==sha(run/'PLAN.json')
    checkpoint=read(resume.with_suffix('.json')) if resume else plan['initial_checkpoint']
    assert sha(checkpoint['path'])==checkpoint['sha256']
    saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=True,mmap=True)
    start=int(saved['optimizer_steps']);local_start=start-50000
    assert 50000<=start<80000
    if resume:
        assert not technical and saved['schema']=='editing_mixed1750k_checkpoint_v1'
        assert saved['plan_sha256']==sha(run/'PLAN.json')
    else:assert start==50000 and saved['recovery_schema']=='editing_continuous50k_recovery_v1'
    rank,device,diffusion,vae,pipeline,runtime=load_runtime(Path(plan['runtime_reference']),
        qwen_mode=plan['Qwen_mode'],numa=True,deterministic=False)
    io_group=dist.new_group(backend='gloo',timeout=timedelta(seconds=1800))
    initial=parameter_digest(diffusion);names=list(dict(diffusion.named_parameters()))
    qwen=diffusion.conditioner.conditioners['prompt'].model
    frozen={'qwen':parameter_digest(qwen),'vae':parameter_digest(vae),'model_buffers':buffer_digest(diffusion.model)}
    assert len(names)==201 and sum(p.numel() for p in diffusion.parameters())==319314336
    assert initial==saved['initial_parameter_sha256'] and names==saved['optimizer_parameter_names']
    assert saved['frozen_parameters']==frozen and saved['physical_world_size']==3
    copy_all_parameters(diffusion,saved['state']);active_start=parameter_digest(diffusion)
    assert active_start==saved['final_parameter_sha256']
    diffusion.train();qwen.eval();vae.eval()
    modules=[m for m in diffusion.modules() if hasattr(m,'activation_checkpointing')]
    assert len(modules)==1;modules[0].activation_checkpointing=False
    forward=EditingForward(diffusion)
    ddp=DDP(forward,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True)
    parameters=list(diffusion.parameters());optimizer=torch.optim.AdamW(parameters,**plan['optimizer'])
    optimizer.load_state_dict(saved['optimizer'])
    assert len(optimizer.state)==201 and all(float(v['step'])==start for v in optimizer.state.values())
    restore_rng(saved['rng_by_rank'][rank],device)
    chains=dict(saved['input_chains_by_rank'][rank]) if resume else {'data':'0'*64,'noise':'0'*64}
    del saved
    stop=2 if technical else 30000
    attempt_token=[str(time.time_ns()) if rank==0 else None]
    dist.broadcast_object_list(attempt_token,src=0,group=io_group)
    label=('TECHNICAL' if technical else f'FROM{start:06d}')+'_'+attempt_token[0]
    attempt=run/'attempts'/label
    if rank==0:attempt.mkdir(parents=True,exist_ok=False)
    dist.barrier(group=io_group)
    write(attempt/f'INITIAL_RANK_{rank}.json',dict(started_at=now(),rank=rank,pid=os.getpid(),physical_gpu=rank+2,
        initial_parameter_sha256=initial,loaded_parameter_sha256=active_start,optimizer_steps=start,technical_only=technical,runtime=runtime))
    exposures=np.load(run/'EXPOSURES.npy',mmap_mode='r');seen=np.load(run/'SEEN_COUNTS.npy',mmap_mode='r')
    stream=MixedStream(run,diffusion,pipeline,rank,device,local_start,stop,chains,(runtime['NUMA_binding'] or {}).get('numa_node'))
    log=(attempt/f'TRACE_RANK_{rank}.jsonl').open('x');gradients=set();began=time.monotonic();rows_seen=0

    def snapshot(step):
        global_step=50000+step;optimizer.zero_grad(set_to_none=True);active=parameter_digest(diffusion)
        assert len(gradients)==201 and parameter_digest(qwen)==frozen['qwen'] and parameter_digest(vae)==frozen['vae']
        assert buffer_digest(diffusion.model)==frozen['model_buffers']
        gathered=[None]*3
        dist.all_gather_object(gathered,{'parameters':active,'rng':rng(device),'chains':dict(stream.chains)},group=io_group)
        assert len({r['parameters'] for r in gathered})==1
        if rank==0:
            directory=run/'checkpoints'/f'STEP{global_step:06d}'
            if directory.exists():
                assert not (directory/'CHECKPOINT.json').exists(),'Refuse to overwrite a completed checkpoint'
                directory.rename(directory.with_name(directory.name+f'.interrupted_{time.time_ns()}'))
            directory.mkdir(parents=True,exist_ok=False)
            path=directory/f'step{global_step:06d}.ckpt';temporary=path.with_suffix('.tmp')
            model_state=all_parameter_state(diffusion);adam=cpu_tree(optimizer.state_dict())
            assert all(torch.isfinite(v).all() for v in model_state.values())
            for item in adam['state'].values():
                assert float(item['step'])==global_step
                assert all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for v in item.values())
            payload=dict(schema='editing_mixed1750k_checkpoint_v1',optimizer_steps=global_step,new_optimizer_steps=step,
                next_update_index=step,configured_train_pairs=1750000,new_run_seen_unique_pairs=int(seen[step]),
                pair_exposures=plan['historical_pair_exposures']+int(exposures[step]),new_pair_exposures=int(exposures[step]),
                state=model_state,optimizer=adam,optimizer_parameter_names=names,rng_by_rank=[r['rng'] for r in gathered],
                input_chains_by_rank=[r['chains'] for r in gathered],initial_parameter_sha256=initial,final_parameter_sha256=active,
                frozen_parameters=frozen,physical_world_size=3,plan_sha256=sha(run/'PLAN.json'),
                generation_initialization=plan['generation_initialization'],initial_checkpoint=plan['initial_checkpoint'],
                train_index=plan['dataset'],technical_only=False,quality_gate_passed=False)
            with temporary.open('xb') as handle:torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
            temporary.replace(path)
            manifest=dict(path=str(path),sha256=sha(path),optimizer_steps=global_step,new_optimizer_steps=step,
                final_parameter_sha256=active,configured_train_pairs=1750000,optimizer_saved=True,rng_all3ranks_saved=True,
                new_pair_exposures=int(exposures[step]),technical_only=False,quality_gate_passed=False)
            write(path.with_suffix('.json'),manifest);write(directory/'CHECKPOINT.json',manifest);write(run/'CURRENT_RECOVERY.json',manifest)
            del payload,model_state,adam
        dist.barrier(group=io_group)

    try:
        for step in range(local_start,stop):
            begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=lr_at(step,plan)
            batch=stream.next(step);noised,t,rows,mask,target=stream.device_batch(batch)
            torch.manual_seed(plan['forward_seed_offset']+step*3+rank)
            with torch.autocast('cuda',dtype=torch.bfloat16):predicted=ddp(noised,t,rows,mask)
            loss=weighted_rf_loss(predicted,target,mask,batch);assert bool(torch.isfinite(loss));loss.backward()
            assert all(p.grad is None for p in qwen.parameters()) and all(p.grad is None for p in vae.parameters())
            gradients.update(n for n,p in diffusion.named_parameters() if p.grad is not None)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize(device);stream.commit(step,batch)
            record=dict(optimizer_steps=50001+step,new_steps=step+1,rf_loss=float(loss.detach()),lr=optimizer.param_groups[0]['lr'],
                gradient_norm=float(norm),batch=batch['audit'],input_chains=dict(stream.chains),conditioner=forward.input_checks[-1],
                elapsed_seconds=time.monotonic()-begin,new_pair_exposures=int(exposures[step+1]),technical_only=technical)
            log.write(canonical(record)+'\n');log.flush();forward.input_checks.clear();rows_seen+=len(rows)
            del batch,noised,t,rows,mask,target,predicted,loss
            if not technical and 50001+step in plan['checkpoint_optimizer_steps']:snapshot(step+1)
            if rank==0 and ((step+1)%100==0 or step==local_start):
                write(run/'STATE.json',dict(stage='technical_check' if technical else 'training',observed_at=now(),optimizer_steps=50001+step,
                    new_steps=step+1,target_optimizer_steps=80000,rf_loss=record['rf_loss'],lr=record['lr'],physical_gpus=GPUS))
                print(canonical({'step':50001+step,'loss':record['rf_loss'],'seconds':record['elapsed_seconds']}),flush=True)
    finally:
        log.close();stream.close()
    active=parameter_digest(diffusion);assert active!=active_start and len(gradients)==201
    assert parameter_digest(qwen)==frozen['qwen'] and parameter_digest(vae)==frozen['vae'] and buffer_digest(diffusion.model)==frozen['model_buffers']
    summary=dict(rank=rank,pid=os.getpid(),physical_gpu=rank+2,final_parameter_sha256=active,all201_parameters_received_gradients=True,
        actual_updates=stop-local_start,rows_seen_on_rank=rows_seen,technical_only=technical,elapsed_seconds=time.monotonic()-began)
    write(attempt/f'FINAL_RANK_{rank}.json',summary);gathered=[None]*3;dist.all_gather_object(gathered,summary,group=io_group)
    assert len({r['final_parameter_sha256'] for r in gathered})==1
    if rank==0:
        if technical:write(run/'TECHNICAL_REVIEW.json',dict(status='PASS_TWO_BUCKET_DDP_UPDATES_FROM_50K',completed_at=now(),
            plan_sha256=sha(run/'PLAN.json'),rank_results=gathered,original_50k_unchanged=True,technical_weights_discarded=True,production_starts_again_from_50k=True))
        else:
            assert stop==30000
            write(run/'RESULT.json',dict(status='COMPLETE_30K_CONTINUATION_SIX_CHECKPOINTS_NEED_VALIDATION',completed_at=now(),
                optimizer_steps=80000,checkpoint_steps=plan['checkpoint_optimizer_steps'],train_rows=1750000,rank_results=gathered))
    dist.barrier(group=io_group);dist.destroy_process_group(io_group);dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--technical',action='store_true');parser.add_argument('--resume',type=Path);args=parser.parse_args()
    train(args.technical,args.resume)
