#!/usr/bin/env python3
"""Bounded four-example natural-request training feasibility gate, not an eval."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20), b''): h.update(part)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=json.loads(args.pairs.read_text())['records'];assert len(records)==4
    assert all(r['pair']['split']=='train' and r['pair']['semantic_pairing_review']=='CODEX_AUDITED_WITH_RECORDED_ANNOTATION_REPAIRS' for r in records)
    sys.path.insert(0,str(args.snapshot))
    import torch
    from torch.nn import functional as F
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    from stable_audio_tools.inference.sceneplan_generation_ar_vectorized import generate_constrained_vectorized
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    codec=ModelScenePlanCodecV4('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    started=time.monotonic()
    atomic(args.output/'STATUS.json',{'status':'LOADING','pid':os.getpid()})
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    base,p10=load_p10v11_generation_ar(pad_id=codec.pad_id,verify_sha256=True,activation_checkpointing=False)
    assert p10.as_dict()==state['run_contract']['p10_load'] and codec.fingerprint==state['run_contract']['codec_fingerprint']
    base.load_trainable_state_dict(state['ar_adapter']);model=AdaptedGenerationAR(base,codec,rank=8,alpha=8.,binding_strength=0.);del base
    model.load_lora_state_dict(state['ar_lora']);del state
    model.p10_dit.to(device=device,dtype=torch.float32);model.prompt_conditioner.to(device=device,dtype=torch.bfloat16)
    model.ar_adapter.to(device=device,dtype=torch.float32);model.ar_lora.to(device=device,dtype=torch.float32)
    configure_float32_ar(model);model.eval()
    contract={'schema':'generation_ar_four_natural_training_fit_gate_v1','purpose':'Training feasibility only; no validation/test or generalization claim',
              'checkpoint_sha256':sha(args.checkpoint),'pairs_sha256':sha(args.pairs),'script_sha256':sha(Path(__file__)),
              'snapshot_manifest_sha256':sha(args.snapshot/'SOURCE_SNAPSHOT_MANIFEST.json'),
              'steps':256,'batch':4,'unique_train_requests':4,'presentations':1024,'adapter_lr':1e-4,'lora_lr':3e-4,
              'optimizer':'fresh AdamW, weight_decay .01, gradient clipping 1, constant learning rates',
              'loss':'nonpadding token CE; exactness measures memorization only and is not request-acceptance scoring',
              'model_inputs':'raw request plus causal teacher-forcing prefix during training; raw request only at generation',
              'count_forcing':False,'source_binding_bias':0,'max_plan_tokens':512,'gpu_scope':[0],'wall_cap_s':600,
              'precision':'FP32 AR/math-SDPA, BF16 frozen request encoder, TF32 off',
              'p10_load':p10.as_dict(),'codec_fingerprint':codec.fingerprint,
              'success_rule':'final training NLL <= .02; all four generated counts correct; at least three exact trained token sequences; frozen parameter hashes unchanged; native P10 pre/post exact; compact checkpoint reload logits exact',
              'checkpoint_reuse':'Never use this overfit checkpoint as a general model initialization or final delivery.'}
    atomic(args.output/'CONTRACT.json',contract)
    def digest(module):
        h=hashlib.sha256()
        for name,value in module.named_parameters():
            h.update(name.encode());h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    frozen_before={name:digest(module) for name,module in [('p10',model.p10_dit),('conditioner',model.prompt_conditioner)]}
    requests=[r['request'] for r in records];token_lists=[r['pair']['target_token_ids'] for r in records]
    width=max(map(len,token_lists));tokens=torch.full((4,width),codec.pad_id,device=device,dtype=torch.long)
    for i,values in enumerate(token_lists):tokens[i,:len(values)]=torch.tensor(values,device=device)
    inputs=tokens[:,:-1];mask=inputs!=codec.pad_id;labels=tokens[:,1:].clone();labels[labels==codec.pad_id]=-100
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        context,context_mask=model.encode_requests(requests,device=device)
    acoustic=torch.randn(4,9,320,device=device,dtype=torch.bfloat16);acoustic_mask=torch.ones(4,9,device=device,dtype=torch.bool)
    def native_probe():
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model.shared_transformer(acoustic,context=context,context_mask=context_mask,padding_mask=acoustic_mask,use_checkpointing=False).clone()
    native_before=native_probe()
    spec=importlib.util.spec_from_file_location('raw_failure_isolation',args.raw_entry);raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
    def evaluate(step):
        model.eval()
        def generate(part):
            if time.monotonic()-started>600:raise TimeoutError('four-example gate budget exceeded')
            return generate_constrained_vectorized(model,part,codec,device=device,max_plan_tokens=512)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=raw.isolate_generation_failures(generate,requests)
            logits=model(inputs,mask,context,context_mask)
            nll=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100))
        rows=[]
        for r,expected,(ids,error) in zip(records,token_lists,output):
            plan=None
            if ids is not None:
                try:plan=codec.decode(ids,sample_id=r['id'])
                except Exception as exc:error=repr(exc)
            rows.append({'id':r['id'],'request':r['request'],'prediction':plan,'tokens':ids,'error':error,
                         'count_correct':bool(plan) and len(plan['sources'])==len(r['pair']['target_sceneplan']['sources']),
                         'exact_supervised_tokens_auxiliary':ids==expected})
        result={'step':step,'train_nll':nll,'count_correct':sum(r['count_correct'] for r in rows),
                'exact_trained_token_sequences_auxiliary':sum(r['exact_supervised_tokens_auxiliary'] for r in rows),'rows':rows}
        atomic(args.output/f'GENERATIONS_{step:04d}.json',result);return result,logits.detach().clone()
    before,_=evaluate(0)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert all(name.startswith(('ar_adapter.','ar_lora.')) for name,p in model.named_parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([{'params':model.ar_adapter.parameters(),'lr':1e-4},{'params':model.ar_lora.parameters(),'lr':3e-4}],weight_decay=.01)
    with (args.output/'metrics.jsonl').open('a') as log:
        for step in range(1,257):
            if time.monotonic()-started>600:raise TimeoutError('four-example gate budget exceeded')
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(inputs,mask,context,context_mask)
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),ignore_index=-100)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()
            if step==1 or step%32==0:
                value={'step':step,'loss':float(loss),'gradient_norm':float(norm),'elapsed_s':time.monotonic()-started}
                log.write(json.dumps(value)+'\n');log.flush();atomic(args.output/'STATUS.json',{'status':'TRAINING',**value})
    after,expected_logits=evaluate(256)
    frozen_after={name:digest(module) for name,module in [('p10',model.p10_dit),('conditioner',model.prompt_conditioner)]}
    checkpoint=args.output/'OVERFIT_ONLY.pt';torch.save({'run_contract':contract,'global_step':256,'ar_adapter':model.trainable_state_dict(),'ar_lora':model.lora_state_dict()},checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_trainable_state_dict(saved['ar_adapter']);model.load_lora_state_dict(saved['ar_lora']);model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):reloaded=model(inputs,mask,context,context_mask)
    checks={'nll_le_002':after['train_nll']<=.02,'all_four_count_correct':after['count_correct']==4,
            'three_exact_trained_sequences':after['exact_trained_token_sequences_auxiliary']>=3,
            'frozen_parameters_unchanged':frozen_before==frozen_after,
            'native_p10_pre_post_exact':torch.equal(native_before,native_probe()),'checkpoint_reload_logits_exact':torch.equal(expected_logits,reloaded)}
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'before_nll':before['train_nll'],'after_nll':after['train_nll'],
            'before_count_correct':before['count_correct'],'after_count_correct':after['count_correct'],
            'exact_trained_sequences_auxiliary':after['exact_trained_token_sequences_auxiliary'],'frozen_before':frozen_before,'frozen_after':frozen_after,
            'elapsed_s':time.monotonic()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'generalization_evidence':False,'goal_complete':False}
    atomic(args.output/'GATE.json',result);atomic(args.output/'STATUS.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','checkpoint','pairs','snapshot','raw-entry'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True,exist_ok=True);atomic(args.output/'STATUS.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
        raise
