"""Matched full-split inference and established independent editing metrics."""
import argparse
from datetime import timedelta
import os
import sys
import time
from common import *
from evaluation_common import *
sys.path[:0]=[str(MAIN),str(OLD_RUN)]

def ensure_plan(split,step):
    index,specs=cases(split);ref=checkpoint(step);run=ROOT/split/f'STEP{step:06d}'
    run.mkdir(parents=True,exist_ok=True)
    if not (run/'PLAN.json').exists():
        plan=dict(schema='editing_mixed1750k_fullsplit_evaluation_v1',created_at=now(),split=split,
            rows=len(specs),checkpoint_steps=step,checkpoint=ref,index=index,physical_gpus=GPUS,
            sampler=SAMPLER,seed=42,Qwen_mode='fast',torch_deterministic_algorithms=False,
            case_manifest_sha256=digest(specs),selection_protocol_sha256=sha(ROOT/'SELECTION_PROTOCOL.json'),
            listening_ordinals=([s['pair_ordinal'] for s in specs] if split.startswith('test_') else listening_cases(specs)),runtime_reference=read(ROOT/'training/PLAN.json')['runtime_reference'],
            runtime_inputs=['source_FOA','complete_NEW_ScenePlan'],test_used_for_selection=False)
        write(run/'PLAN.json',plan)

def evaluate(split,step,context=None):
    import torch
    from torch import distributed as dist
    from throughput_runtime_v1 import load_runtime
    from generation_runtime import parameter_digest,tensor_sha
    from editing_conditioner_only_state_v1 import copy_all_parameters,buffer_digest
    from p10_editing_sampling import sample_with_profile
    from mixture_dataset import MixtureDataset
    from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio
    from truth_cache import TruthResolver

    if split.startswith('test_'):
        selection=read(ROOT/'SELECTION.json')
        assert selection['selection_split']=='validation'
        assert selection['selected_checkpoint_steps']==step,'Only validation-selected weights may enter test evaluation'
    protocol=read(ROOT/'SELECTION_PROTOCOL.json');assert protocol['steps']==STEPS
    index,specs=cases(split);ref=checkpoint(step);run=ROOT/split/f'STEP{step:06d}'
    rank=int(os.environ['RANK'])
    if rank==0 and context is None:ensure_plan(split,step)
    # Runtime constructs the NCCL process group after binding physical UUIDs.
    training_plan=read(ROOT/'training/PLAN.json')
    if context is None:
        rank,device,diffusion,vae,pipeline,runtime=load_runtime(Path(training_plan['runtime_reference']),qwen_mode='fast',numa=True,deterministic=False)
        group=dist.new_group(backend='gloo',timeout=timedelta(hours=24));dist.barrier(group=group)
    else:
        rank,device,diffusion,vae,pipeline,runtime=context['runtime']
        group=context['group']
    plan=read(run/'PLAN.json');plan['plan_sha256']=sha(run/'PLAN.json')
    assert plan['case_manifest_sha256']==digest(specs) and plan['index']==index and plan['checkpoint']==ref
    assert plan['selection_protocol_sha256']==sha(ROOT/'SELECTION_PROTOCOL.json')
    initial=parameter_digest(diffusion) if context is None else context['initial_parameter_sha256']
    saved=torch.load(ref['path'],map_location='cpu',weights_only=True,mmap=True)
    assert saved['optimizer_steps']==step and not saved['technical_only']
    if step==50000:
        assert saved['configured_train_pairs']==1000000 and saved['recovery_schema']=='editing_continuous50k_recovery_v1'
    else:
        assert saved['schema']=='editing_mixed1750k_checkpoint_v1' and saved['configured_train_pairs']==1750000
        assert saved['plan_sha256']==sha(ROOT/'training/PLAN.json')
        verified=read(ROOT/f'training/checkpoints/STEP{step:06d}/VERIFIED.json');assert verified['checkpoint']==ref
    assert initial==saved['initial_parameter_sha256']
    frozen={'qwen':parameter_digest(diffusion.conditioner.conditioners['prompt'].model),'vae':parameter_digest(vae),'model_buffers':buffer_digest(diffusion.model)}
    assert frozen==saved['frozen_parameters']
    copy_all_parameters(diffusion,saved['state']);diffusion.eval()
    active=parameter_digest(diffusion);assert active==saved['final_parameter_sha256']==ref['final_parameter_sha256'];del saved
    local=specs[rank::3]
    dataset=MixtureDataset(index,(diffusion.conditioner.conditioners['prompt'].tokenizer,512),sample_cases=local)
    resolver=TruthResolver(index)
    scorer=(audio.IndependentEditingContentEvaluator(device=device,device_index=int(device.index)) if context is None else context['scorer'])
    output=run/'evaluation';output.mkdir(parents=True,exist_ok=True)
    entries=[];started=time.monotonic();attempt=time.time_ns()
    write(output/f'ATTEMPT_{rank}_{attempt}.json',dict(pid=os.getpid(),rank=rank,physical_gpu=rank+2,runtime=runtime,started_at=now(),plan_sha256=plan['plan_sha256']))
    args=argparse.Namespace(seed=42,max_plan_tokens=512,ode_steps=20,cfg_scale=1.,save_all_audio=False)
    try:
        with torch.no_grad():
            for offset,spec in enumerate(local):
                ordinal=spec['pair_ordinal'];bucket=spec['latent_bucket_frames'];n=spec['model_num_samples']
                path=output/'cases'/f'rank-{rank}'/f'{ordinal:07d}.json'
                if path.exists():load_case(path,spec,plan)
                else:
                    target,metadata=dataset[offset];truth=resolver.row(ordinal)
                    assert metadata['pair_ordinal']==ordinal and metadata['pair_id']==spec['pair_id']
                    assert metadata['model_sceneplan']==truth['offline_new_sceneplan']
                    source,mask=pipeline.encode_source_foa(audio._pad_audio([truth['source_foa']],bucket),
                        model_num_samples=[n],vae_seeds=[audio._stable_seed(42,spec['pair_id'],'source-vae')])
                    noise=audio._initial_noise([spec],bucket,42)
                    # The complete NEW plan and source are the only conditions.
                    latent=sample_with_profile(pipeline,source,mask,[metadata['model_sceneplan']],
                        model_num_samples=[n],initial_noise=noise.clone(),profile=SAMPLER)
                    wave,sample_mask=pipeline.decode_foa_latents(latent,model_num_samples=[n])
                    source_codec,_=pipeline.decode_foa_latents(source,model_num_samples=[n])
                    sampled=dict(edited_foa=wave,source_codec_foa=source_codec,source_foa_latent=source,
                        source_attention_mask=mask,sample_attention_mask=sample_mask,edited_foa_latent=latent,new_sceneplans=[metadata['model_sceneplan']])
                    row=audio._process_batch(pipeline=pipeline,content_evaluator=scorer,codec=None,
                        samples=[(target,metadata,{})],truths=[truth],bucket=bucket,device=device,args=args,
                        output_dir=output,listening_ordinals=set(plan['listening_ordinals']),sampled_gt_result=sampled)[0]
                    row.update(evaluation_split=split,training_example_seen=False,cohort=spec['cohort'],native_pair_ordinal=spec['native_pair_ordinal'],
                        sampler_settings=SAMPLER,source_latent_sha256=tensor_sha(source),initial_noise_sha256=tensor_sha(noise),
                        output_latent_sha256=tensor_sha(latent),source_foa_sha256=spec['source_foa_sha256'],target_foa_sha256=spec['target_foa_sha256'],
                        new_sceneplan_sha256=spec['new_sceneplan_sha256'],active_parameter_sha256=active,initial_parameter_sha256=initial,
                        checkpoint_steps=step,evaluation_plan_sha256=plan['plan_sha256'],test_used_for_checkpoint_selection=False,quality_gate_passed=False)
                    audio._validate_batch_records([row],[spec],bucket=bucket);check_row(row,spec,plan)
                    write(path,dict(row=row,row_sha256=digest(row)))
                entries.append(dict(path=str(path),sha256=sha(path)))
                if (offset+1)%50==0 or offset+1==len(local):
                    state=dict(stage='inference_and_independent_metrics',split=split,checkpoint_steps=step,rank=rank,physical_gpu=rank+2,
                        completed_rows=offset+1,total_rows=len(local),elapsed_seconds=time.monotonic()-started,observed_at=now())
                    write(run/f'STATE_RANK{rank}.json',state);print(canonical(state),flush=True)
    except BaseException as error:
        write(run/f'FAILURE_RANK{rank}_{attempt}.json',dict(error=repr(error),completed_rows=len(entries),at=now(),partial_metrics_not_eligible_for_selection=True));raise
    finally:
        resolver.close()
        dataset.close()
    assert parameter_digest(diffusion)==active
    assert parameter_digest(vae)==frozen['vae'] and parameter_digest(diffusion.conditioner.conditioners['prompt'].model)==frozen['qwen']
    write(output/f'rank-{rank}.json',dict(rank=rank,physical_gpu=rank+2,records=entries,plan_sha256=plan['plan_sha256'],completed_at=now()))
    if context is not None:return
    dist.barrier(group=group)
    if rank==0:
        allr=[read(output/f'rank-{r}.json') for r in range(3)]
        assert sum(len(r['records']) for r in allr)==len(specs)
        write(run/'INFERENCE_COMPLETE.json',dict(status='ALL_ROWS_SCORED_NEEDS_INDEPENDENT_CPU_REVIEW',split=split,rows=len(specs),checkpoint_steps=step,plan_sha256=plan['plan_sha256'],completed_at=now()))
    dist.barrier(group=group);dist.destroy_process_group(group);dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=SCOPES,required=True);p.add_argument('--step',type=int,choices=STEPS,required=True);a=p.parse_args();evaluate(a.split,a.step)
