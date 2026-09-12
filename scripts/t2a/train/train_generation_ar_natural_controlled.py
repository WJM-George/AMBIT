#!/usr/bin/env python3
"""Matched-target natural-language training, one GPU per independent arm."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1','2')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'checkpoints').mkdir(exist_ok=True)
    lock=(args.output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(args.snapshot))
    import torch
    from torch.nn import functional as F
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=json.loads(args.data.read_text());parents=data['parents'];schedule=data['schedule'];experiment=data.get('experiment',{})
    objective=None
    if args.objective_config:
        objective=json.loads(args.objective_config.read_text())
        assert objective['schema']=='generation_ar_count_auxiliary_v1'
        assert objective['variant'] in ('count_aux','count_header_aug') and objective['weight']==.1
        assert args.arm=='natural_mix' and objective['training_sha256']==sha(args.data)
        experiment={**experiment,**objective['experiment_overrides']}
    batch_size=len(schedule[0]);assert schedule and all(len(x)==batch_size for x in schedule) and batch_size%4==0
    started=time.monotonic();budget=900 if args.gate else experiment.get('wall_cap_seconds',2700)
    def check_budget():
        if time.monotonic()-started>budget:raise TimeoutError('Training wall budget exceeded; preserve checkpoint and diagnose before changing the contract')
    atomic(args.output/'STATUS.json',{'status':'LOADING','pid':os.getpid(),'arm':args.arm})
    if args.gate:
        # Real longest targets in all counts; include both request routes.
        batch=[]
        for n in range(1,5):
            per_count=batch_size//4;choices=[]
            for route,num in [('plan_to_request',per_count//2),('request_to_plan',per_count-per_count//2)]:
                choices.extend(sorted((i for i,p in enumerate(parents) if p['source_count']==n and p['route']==route),key=lambda i:-len(parents[i]['target_token_ids']))[:num])
            assert len(choices)==per_count
            batch.extend({'index':i,'mode':'request_first' if parents[i]['route']=='request_to_plan' else 'five_view','view':max(range(len(parents[i]['natural_requests'])),key=lambda v:len(parents[i]['natural_requests'][v]))} for i in choices)
        schedule=[batch]*3
    codec=ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    base,p10=load_p10v11_generation_ar(pad_id=codec.pad_id,verify_sha256=True,activation_checkpointing=False)
    assert p10.as_dict()==state['run_contract']['p10_load'] and codec.fingerprint==state['run_contract']['codec_fingerprint']
    base.load_trainable_state_dict(state['ar_adapter']);model=AdaptedGenerationAR(base,codec,rank=8,alpha=8.,binding_strength=0.);del base
    model.load_lora_state_dict(state['ar_lora']);del state
    model.p10_dit.to(device=device,dtype=torch.float32);model.prompt_conditioner.to(device=device,dtype=torch.bfloat16)
    model.ar_adapter.to(device=device,dtype=torch.float32);model.ar_lora.to(device=device,dtype=torch.float32)
    configure_float32_ar(model);model.eval()
    contract={'schema':'generation_ar_natural_controlled_v2','experiment_id':experiment.get('id','legacy_N1'),'arm':args.arm,'mode':'startup_gate' if args.gate else 'controlled_pilot',
              'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),'data_sha256':sha(args.data),'script_sha256':sha(Path(__file__)),
              'snapshot':str(args.snapshot),'snapshot_manifest_sha256':sha(args.snapshot/'SOURCE_SNAPSHOT_MANIFEST.json'),
              'steps':len(schedule),'batch_size':batch_size,'adapter_lr':5e-5,'lora_lr':1e-4,'warmup_steps':0 if args.gate else 30,
              'optimizer':'fresh AdamW; weight_decay .01; global nonpadding token CE; gradient clip 1; cosine LR floor .1',
              'request_encoder':'Frozen Qwen BF16; precompute only model.encode_requests outputs; no trainable projection is cached; trimmed trailing padding; FP32 cache.',
              'model_input':'Raw English only, plus causal target prefix for training. No count hints, source IDs or requirement metadata.',
              'binding_strength':0.,'count_forcing':False,'precision':'FP32 AR/math SDPA; BF16 frozen request encoder; TF32 disabled',
              'schedule':experiment.get('schedule_description','Identical target sequence and target-token denominators in both arms. Natural mix presentations: 20% precise replay, 40% five views, 40% request-first.'),
              'p10_load':p10.as_dict(),'codec_fingerprint':codec.fingerprint,'wall_cap_seconds':budget,'test_used':False,
              'success_rule':experiment.get('success_rule','Independent 16-scene natural validation joint improves >=10 percentage points over matched precise control; no per-count precise count/core/constraint regression >2 points on 1024-scene validation. Pilot evidence only; full data promotion requires broader independent natural validation.'),
              'limitations':experiment.get('limitations','Only 32 unique request-first training scenarios; oversampling is not diversity. Matched updates/targets are not identical request-encoder FLOPs; elapsed time is recorded.')}
    if objective:
        contract['count_auxiliary']={'config':objective,'config_sha256':sha(args.objective_config),
            'loss':'Global nonpadding full-sequence CE + 0.1 * per-example count CE on a separate six-token prefix forward; full 4096-way vocabulary.',
            'inference':'Unchanged raw request plus codec grammar. This objective adds no count hint or inference-time parser.'}
    if not args.gate:
        gate=json.loads(args.gate_proof.read_text());assert gate['status']=='PASS' and gate['script_sha256']==contract['script_sha256'] and gate['data_sha256']==contract['data_sha256']
        assert gate['checkpoint_sha256']==contract['checkpoint_sha256'];contract['gate_proof_sha256']=sha(args.gate_proof)
        if objective:assert gate['objective_config_sha256']==contract['count_auxiliary']['config_sha256']
        review=json.loads(args.parent_review.read_text());assert review['status']=='PASS';contract['parent_review_sha256']=sha(args.parent_review)
        if experiment:assert review['data_sha256']==contract['data_sha256']
    path=args.output/'CONTRACT.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:atomic(path,contract)
    def digest(module):
        h=hashlib.sha256()
        for name,value in module.named_parameters():h.update(name.encode());h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    frozen_before={name:digest(m) for name,m in [('p10',model.p10_dit),('conditioner',model.prompt_conditioner)]}
    def text_for(item):
        p=parents[item['index']]
        precise=p['precise_requests_by_view'][item['view']] if 'precise_requests_by_view' in p else p['precise_request']
        return precise if args.arm=='precise_control' or item['mode']=='precise' else p['natural_requests'][item['view']]
    texts=sorted({text_for(item) for batch in schedule for item in batch});cache={}
    for offset in range(0,len(texts),32):
        check_budget();part=texts[offset:offset+32]
        with torch.no_grad():context,mask=model.encode_requests(part,device=device)
        for i,text in enumerate(part):
            positions=mask[i].nonzero().flatten();length=int(positions[-1])+1 if len(positions) else 0;assert length>0
            cache[text]=(context[i,:length].cpu(),mask[i,:length].cpu())
        if offset==0 or offset%512==0:atomic(args.output/'STATUS.json',{'status':'CACHING_FROZEN_REQUEST_CONTEXT','pid':os.getpid(),'requests_done':min(offset+32,len(texts)),'requests':len(texts),'elapsed_s':time.monotonic()-started})
    cache_elapsed=time.monotonic()-started
    auxiliary_candidates={}
    if objective:
        from stable_audio_tools.data.sceneplan_generation_ar_count_supervision import count_prefix_candidates,choose_count_prefix
        metadata_path=Path(objective['request_first_metadata']);assert sha(metadata_path)==objective['request_first_metadata_sha256']
        metadata={r['id']:r for r in json.loads(metadata_path.read_text())['records']}
        for parent_index,p in enumerate(parents):
            if p['route']!='request_to_plan':continue
            record=metadata[p['id']]
            for view,tokens in enumerate(p['target_token_ids_by_view']):
                assert record['targets'][view]['tokens']==tokens and record['views'][view]['request']==p['natural_requests'][view]
                auxiliary_candidates[parent_index,view]=count_prefix_candidates(codec,tokens,p['natural_requests'][view],record['requirements'][view])
        del metadata
    def collate(items):
        ids=[parents[x['index']]['target_token_ids_by_view'][x['view']] if 'target_token_ids_by_view' in parents[x['index']] else parents[x['index']]['target_token_ids'] for x in items];contexts=[cache[text_for(x)] for x in items]
        width=max(map(len,ids));length=max(c.shape[0] for c,m in contexts);batch=len(items)
        tokens=torch.full((batch,width),codec.pad_id,dtype=torch.long,device=device)
        ctx=torch.zeros(batch,length,contexts[0][0].shape[-1],device=device);cmask=torch.zeros(batch,length,dtype=torch.bool,device=device)
        for i,(values,(c,m)) in enumerate(zip(ids,contexts)):
            tokens[i,:len(values)]=torch.tensor(values,device=device);ctx[i,:c.shape[0]]=c.to(device);cmask[i,:m.shape[0]]=m.to(device)
        inputs=tokens[:,:-1];labels=tokens[:,1:].clone();labels[labels==codec.pad_id]=-100
        return inputs,inputs!=codec.pad_id,labels,ctx,cmask
    probe=collate(schedule[0]);context,context_mask=probe[-2:]
    acoustic=torch.randn(4,9,320,device=device,dtype=torch.bfloat16);acoustic_mask=torch.ones(4,9,device=device,dtype=torch.bool)
    def native_probe():
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model.shared_transformer(acoustic,context=context[:4],context_mask=context_mask[:4],padding_mask=acoustic_mask,use_checkpointing=False).clone()
    native_before=native_probe()
    trainable=[p for p in model.parameters() if p.requires_grad];assert all(n.startswith(('ar_adapter.','ar_lora.')) for n,p in model.named_parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([{'params':model.ar_adapter.parameters(),'lr':5e-5},{'params':model.ar_lora.parameters(),'lr':1e-4}],weight_decay=.01)
    warmup=contract['warmup_steps'];steps=len(schedule)
    def multiplier(step):
        if warmup and step<warmup:return max(1,step)/warmup
        progress=min(1.,max(0.,(step-warmup)/max(1,steps-warmup)))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier);start_step=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False);assert saved['run_contract']==contract
        model.load_trainable_state_dict(saved['ar_adapter']);model.load_lora_state_dict(saved['ar_lora']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);start_step=saved['global_step']
        torch.set_rng_state(saved['rng_torch']);torch.cuda.set_rng_state(saved['rng_cuda']);del saved
    def checkpoint(step,name=None):
        path=args.output/'checkpoints'/(name or f'step_{step:08d}.pt');temp=path.with_suffix('.tmp')
        torch.save({'global_step':step,'run_contract':contract,'ar_adapter':model.trainable_state_dict(),'ar_lora':model.lora_state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},temp);temp.replace(path);return path
    losses=[];tokens_seen=0;training_started=time.monotonic()
    for index in range(start_step,steps):
        try:check_budget()
        except TimeoutError:checkpoint(index,'budget_stop.pt');raise
        model.train();inputs,mask,labels,ctx,cmask=collate(schedule[index]);optimizer.zero_grad(set_to_none=True)
        logits=model(inputs,mask,ctx,cmask);token_loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
        count_loss=None;loss=token_loss
        if objective:
            prefixes=inputs[:,:6].clone()
            if objective['variant']=='count_header_aug':
                for row,item in enumerate(schedule[index]):
                    if item['mode']=='request_first':
                        candidates=auxiliary_candidates[item['index'],item['view']]
                        chosen=choose_count_prefix(candidates,step=index,row=row,parent_index=item['index'],view=item['view'])
                        prefixes[row]=torch.tensor(chosen,device=device)
            count_logits=model(prefixes,torch.ones_like(prefixes,dtype=torch.bool),ctx,cmask)[:,-1]
            count_loss=F.cross_entropy(count_logits,labels[:,5]);loss=token_loss+objective['weight']*count_loss
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
        optimizer.step();scheduler.step();losses.append(float(loss));tokens_seen+=int((labels!=-100).sum());step=index+1
        if step==1 or step%25==0 or step==steps:
            value={'status':'TRAINING','pid':os.getpid(),'arm':args.arm,'step':step,'steps':steps,'loss':float(loss),'recent_mean_loss':sum(losses[-25:])/len(losses[-25:]),'gradient_norm':float(norm),'elapsed_s':time.monotonic()-started,'training_s':time.monotonic()-training_started,'tokens_per_s':tokens_seen/max(time.monotonic()-training_started,1e-6),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            if objective:value.update(token_ce=float(token_loss),count_auxiliary_ce=float(count_loss),objective=objective['variant'])
            with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
            atomic(args.output/'STATUS.json',value);print(json.dumps(value),flush=True)
        del logits,loss,token_loss
        if objective:del count_logits,count_loss
        if step%100==0 or step==steps:checkpoint(step)
    model.eval();inputs,mask,labels,ctx,cmask=probe
    with torch.no_grad():before=model(inputs,mask,ctx,cmask).clone()
    final=args.output/'checkpoints'/f'step_{steps:08d}.pt';saved=torch.load(final,map_location='cpu',weights_only=False)
    model.load_trainable_state_dict(saved['ar_adapter']);model.load_lora_state_dict(saved['ar_lora']);del saved
    with torch.no_grad():after=model(inputs,mask,ctx,cmask)
    frozen_after={name:digest(m) for name,m in [('p10',model.p10_dit),('conditioner',model.prompt_conditioner)]}
    checks={'frozen_parameters_unchanged':frozen_before==frozen_after,'native_p10_pre_post_exact':torch.equal(native_before,native_probe()),'checkpoint_reload_logits_exact':torch.equal(before,after)}
    result={'status':'PASS' if all(checks.values()) else 'FAIL','arm':args.arm,'checks':checks,'steps':steps,'elapsed_s':time.monotonic()-started,'context_cache_including_load_s':cache_elapsed,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'script_sha256':contract['script_sha256'],'data_sha256':contract['data_sha256'],'checkpoint_sha256':contract['checkpoint_sha256'],'final_checkpoint':str(final),'training_complete':True,'acceptance_complete':False,'goal_complete':False}
    if objective:result['objective_config_sha256']=contract['count_auxiliary']['config_sha256']
    atomic(args.output/('GATE.json' if args.gate else 'TRAINING_RESULT.json'),result);atomic(args.output/'STATUS.json',result)
    if not all(checks.values()):raise RuntimeError('Frozen-model preservation/reload checks failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','checkpoint','data','snapshot'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('precise_control','natural_mix'),required=True);parser.add_argument('--gate',action='store_true')
    parser.add_argument('--gate-proof',type=Path);parser.add_argument('--parent-review',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--objective-config',type=Path)
    args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True,exist_ok=True);atomic(args.output/'STATUS.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
        raise
