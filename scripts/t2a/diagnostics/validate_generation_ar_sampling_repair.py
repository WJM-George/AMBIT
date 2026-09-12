#!/usr/bin/env python3
"""After R1 finishes, compare initialization/midpoint/final on validation only.

Reuses the completed D0 decoding implementation and its fixed balanced panel.
The original selection and all test artifacts remain separate experiments.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/stable-audio-tools-workspace")
D0=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/transfusion_sceneplan/generation_ar/validation_diagnosis_20260905_v1")
R1=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/transfusion_sceneplan/generation_ar/sampling_repair_1ep_20260905_v1")
PYTHON=os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/stable-audio-tools-workspace/.venv/bin/python3"
sys.path.insert(0,str(REPO/'scripts/t2a/diagnostics'))
import run_generation_ar_validation_comparison as baseline


def configs():
    original=json.loads((D0/'CONTRACT.json').read_text())
    initial=next(c for c in original['candidates'] if c['step']==70839)
    return [('initialize_70839',initial,Path(original['training_contract_path'])),
            *[(f'repair_{step}',{'step':step,'checkpoint':str(R1/'training/checkpoints'/f'step_{step:08d}.pt'),
                                'checkpoint_sha256':baseline.sha(R1/'training/checkpoints'/f'step_{step:08d}.pt')},
               R1/'training/RUN_CONTRACT.json') for step in (4167,8334)]]


def prepare(index):
    label,candidate,training_path=configs()[index]
    training=json.loads(training_path.read_text())
    contract=copy.deepcopy(json.loads((D0/'CONTRACT.json').read_text()))
    contract.update({'schema':'generation_ar_sampling_repair_validation_v1','purpose':'R1_DIAGNOSTIC_VALIDATION_ONLY',
                     'candidates':[candidate],'training_contract_path':str(training_path),
                     'training_contract_sha256':baseline.sha(training_path),
                     'physical_gpu_by_step':{str(candidate['step']):index},'experiment_label':label,
                     'selection_note':'R1 experiment comparison. Full teacher-forcing plus unchanged D0 true-AR panel; no production selection or test.'})
    contract['source_sha256'][str(Path(__file__).resolve())]=baseline.sha(Path(__file__))
    for path,digest in training.get('source_sha256',{}).items():
        if Path(path).is_absolute():contract['source_sha256'][path]=digest
    root=R1/'validation'/label
    root.mkdir(parents=True,exist_ok=True)
    path=root/'CONTRACT.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:baseline.atomic(path,contract)
    return root,contract


def full_teacher(root,contract):
    candidate=contract['candidates'][0]
    target=root/'TEACHER_FULL_VALIDATION.json'
    identity={'checkpoint_sha256':candidate['checkpoint_sha256'],'validation_sha256':baseline.MANIFEST_SHA,
              'codec_source_snapshot':str(baseline.SNAPSHOT),'contract_sha256':baseline.sha(root/'CONTRACT.json')}
    if target.exists():
        prior=json.loads(target.read_text())
        assert prior['identity']==identity and prior['metrics']['sequences']==32000
        return
    frozen=baseline.load_frozen()
    torch=frozen.torch
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.manual_seed(42)
    frozen.dist.init_process_group('nccl',init_method='file://'+str(root/f'process_group_{os.getpid()}'),rank=0,world_size=1)
    codec=frozen.ModelScenePlanCodecV4(frozen.CODEC_PATH)
    model,p10=frozen.load_p10v11_generation_ar(pad_id=codec.pad_id,verify_sha256=True,activation_checkpointing=False)
    state=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False)
    training=json.loads(Path(contract['training_contract_path']).read_text())
    assert baseline.sha(candidate['checkpoint'])==candidate['checkpoint_sha256']
    assert state['run_contract']==training and state['global_step']==candidate['step']
    assert p10.as_dict()==training['p10_load'] and codec.fingerprint==training['codec_fingerprint']
    model.load_trainable_state_dict(state['ar_adapter']);del state
    model.p10_dit.to(device=device,dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device,dtype=torch.bfloat16)
    model.ar_adapter.to(device=device,dtype=torch.float32);model.eval()
    manifest=Path('/dev/shm/generation_ar_manifests_20260905/validation.sqlite')
    if not manifest.exists():manifest=baseline.MANIFEST
    assert baseline.sha(manifest)==baseline.MANIFEST_SHA
    dataset=frozen.GenerationARSQLiteDataset(manifest,split='validation')
    started=time.monotonic()
    metrics=frozen._teacher_forced_metrics(model,dataset,codec,device=device,batch_size=64,num_workers=4)
    assert metrics['sequences']==32000 and metrics['tokens']==5573342
    baseline.atomic(target,{'identity':identity,'metrics':metrics,'elapsed_s':time.monotonic()-started})
    print(json.dumps({'event':'full_teacher_complete','label':contract['experiment_label'],'loss':metrics['loss'],'count':metrics['source_count']['accuracy']}),flush=True)
    dataset.close();del model,dataset
    frozen.dist.destroy_process_group();gc.collect();torch.cuda.empty_cache()


def worker(index):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(index)
    root,contract=prepare(index)
    lock=(root/'WORKER_LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in contract['source_sha256'].items():assert baseline.sha(path)==digest,path
    full_teacher(root,contract)
    baseline.worker(root,contract['candidates'][0]['step'])


def summarize():
    values=[]
    for label,candidate,_ in configs():
        root=R1/'validation'/label
        values.append({'label':label,'teacher_full_validation':json.loads((root/'TEACHER_FULL_VALIDATION.json').read_text()),
                       'ar_panel':json.loads((root/f"step_{candidate['step']:08d}/SUMMARY.json").read_text())})
    initial=values[0]['ar_panel']['by_source_count']
    base_min=min(x['source_count_accuracy'] for x in initial.values())
    for value in values[1:]:
        current=value['ar_panel']['by_source_count']
        minimum=min(x['source_count_accuracy'] for x in current.values())
        checks={'count_min_ge_90_or_gain_ge_10pp':minimum>=.9 or minimum>=base_min+.1,
                'no_class_count_drop_gt_2pp':all(current[k]['source_count_accuracy']>=initial[k]['source_count_accuracy']-.02 for k in initial),
                'no_class_lexical_proxy_drop_gt_002':all(current[k]['lexical_description_token_f1_proxy']>=initial[k]['lexical_description_token_f1_proxy']-.02 for k in initial),
                'no_class_spatiotemporal_pass_drop_gt_2pp':all(current[k]['scene_spatiotemporal_pass_rate']['medium']>=initial[k]['scene_spatiotemporal_pass_rate']['medium']-.02 for k in initial)}
        value['predeclared_R1_success_checks']=checks
        value['R1_experiment_success']=all(checks.values())
    baseline.atomic(R1/'validation/COMPARISON.json',{'status':'COMPLETE','goal_complete':False,'results':values,
                     'limits':'AR remains a fixed 1024-row validation panel. Semantic core-content acceptance not yet established. R1 uses a fresh optimizer and stage schedule, so this is not a matched causal sampler-only ablation.'})


def supervise():
    root=R1/'validation';root.mkdir(exist_ok=True)
    lock=(root/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline.atomic(root/'STATUS.json',{'status':'WAITING_FOR_R1_TRAINING','supervisor_pid':os.getpid(),'gpu_scope':[0,1,2]})
    while True:
        status=json.loads((R1/'STATUS.json').read_text())
        if status['status']=='TRAINING_COMPLETE_AWAITING_VALIDATION':break
        if status['status'] in ('FAILED','INTERRUPTED'):
            baseline.atomic(root/'STATUS.json',{'status':'NEEDS_ATTENTION','reason':'R1 training did not complete','training_status':status})
            return 2
        time.sleep(60)
    active=subprocess.check_output(['nvidia-smi','-i','0,1,2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:
        baseline.atomic(root/'STATUS.json',{'status':'NEEDS_ATTENTION','reason':'GPU ownership changed before validation','active_pids':active})
        return 2
    children=[]
    for index in range(3):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        log=(root/f'worker_{index}.log').open('a')
        command=[PYTHON,'-u',str(Path(__file__).resolve()),'--worker-index',str(index)]
        child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children.append((child,log))
    baseline.atomic(root/'STATUS.json',{'status':'RUNNING','worker_pids':[child.pid for child,_ in children],'started_unix':time.time(),'gpu_scope':[0,1,2]})
    codes=[]
    for child,log in children:codes.append(child.wait());log.close()
    if not any(codes):summarize()
    baseline.atomic(root/'STATUS.json',{'status':'COMPLETE' if not any(codes) else 'FAILED','exit_codes':codes,'finished_unix':time.time(),'goal_complete':False})
    return int(any(codes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-index',type=int,choices=(0,1,2))
    args=parser.parse_args()
    if args.worker_index is None:raise SystemExit(supervise())
    worker(args.worker_index)
