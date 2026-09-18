#!/usr/bin/env python3
"""R3 initialization/midpoint/final true-AR validation with the same policy."""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import zlib

R3=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/run")
R1=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/parent/training")
SNAPSHOT=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/source_snapshots/snapshot")
D0=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/diagnosis")
REPO=Path(__file__).resolve().parents[3]
PYTHON=REPO/'.venv/bin/python3'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n');temp.replace(path)


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare():
    root=R3/'validation';root.mkdir(exist_ok=True)
    candidates=[]
    for label,run,step,lora in [('initialize_8334',R1,8334,False),('adapt_1042',R3/'training',1042,True),('adapt_2084',R3/'training',2084,True)]:
        checkpoint=run/'checkpoints'/f'step_{step:08d}.pt';contract=run/'RUN_CONTRACT.json'
        candidates.append({'label':label,'step':step,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                           'training_contract':str(contract),'training_contract_sha256':sha(contract),'trained_lora':lora})
    source=R3/'evaluation_source'
    contract={'schema':'generation_ar_attention_adaptation_validation_v1','purpose':'VALIDATION_DIAGNOSTIC_ONLY','test_used':False,
              'candidates':candidates,'snapshot':str(SNAPSHOT),'snapshot_manifest_sha256':sha(SNAPSHOT/'SOURCE_SNAPSHOT_MANIFEST.json'),
              'source_sha256':{str(p):sha(p) for p in source.iterdir() if p.is_file()},
              'panel':str(D0/'panel.sqlite'),'panel_sha256':sha(D0/'panel.sqlite'),'parent_panel_contract_sha256':sha(D0/'CONTRACT.json'),
              'rows':1024,'per_source_count':256,'batch_size':32,'max_plan_tokens':512,'seed':42,'gpu_scope':[0,1,2],
              'decode_policy':'explicit request count, source penalty 2, native no-bias dispatch when no other source, vectorized greedy',
              'precision':'FP32 AR including self-attention math SDPA; BF16 request encoder; TF32 disabled',
              'numerical_gate':{'cached_full_max_abs':.001,'legal_greedy_disagreements':0,'native_p10_pre_post_exact':True},
              'wall_cap_s_per_worker':1800,'selection_rule':'Compare matched-policy outputs; semantics must be joined before complete R3 success assessment.',
              'success_rule':json.loads((R3/'PROTOCOL.json').read_text())['success_rule']}
    p=root/'CONTRACT.json'
    if p.exists():assert json.loads(p.read_text())==contract
    else:atomic(p,contract)
    return contract


def cache_gate(model,codec,torch,device):
    import numpy as np
    db=sqlite3.connect('file:' + str(Path(os.environ.get("AMBIT_CACHE_ROOT", "cache")) / "generation_ar_manifests" / "train.sqlite") + '?mode=ro&immutable=1',uri=True)
    ordinal,request,blob=db.execute('SELECT ordinal,raw_user_request,target_token_ids_u16le FROM rows WHERE source_count=2 ORDER BY target_token_count,ordinal LIMIT 1').fetchone();db.close()
    ids=torch.tensor(np.frombuffer(blob,dtype='<u2').astype(np.int64),device=device)[None,:-1];mask=torch.ones_like(ids,dtype=torch.bool)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        context,context_mask=model.encode_requests([request],device=device);roles=model.request_source_ids([request],device=device)
        acoustic=torch.randn(1,9,320,device=device,dtype=torch.bfloat16);acoustic_mask=torch.ones(1,9,device=device,dtype=torch.bool)
        before=model.shared_transformer(acoustic,context=context,context_mask=context_mask,padding_mask=acoustic_mask,use_checkpointing=False)
        full=model(ids,mask,context,context_mask,roles)
        cache=model.prepare_decode_cache(context,context_mask,max_plan_tokens=ids.shape[1]+1);cache.ar_binding_sources=roles
        cached=torch.stack([model.decode_step(ids[:,i],cache) for i in range(ids.shape[1])],dim=1)
        difference=float((full-cached).abs().max());disagreements=0;prefix=ids[0].tolist()
        for i in range(len(prefix)):
            legal=torch.tensor(sorted(codec.allowed_next_ids(prefix[:i+1],min_sources=2,max_sources=2)),device=device)
            disagreements+=int(full[0,i,legal].argmax()!=cached[0,i,legal].argmax())
        after=model.shared_transformer(acoustic,context=context,context_mask=context_mask,padding_mask=acoustic_mask,use_checkpointing=False)
    return {'train_ordinal':ordinal,'cached_steps':len(prefix),'cached_full_max_abs':difference,
            'legal_greedy_disagreements':disagreements,'native_p10_pre_post_exact':torch.equal(before,after)}


def worker(index):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(index) and index in (0,1,2)
    root=R3/'validation';contract=json.loads((root/'CONTRACT.json').read_text());candidate=contract['candidates'][index]
    run=root/candidate['label'];run.mkdir(exist_ok=True)
    lock=(run/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,digest in contract['source_sha256'].items():assert sha(path)==digest,path
    assert sha(SNAPSHOT/'SOURCE_SNAPSHOT_MANIFEST.json')==contract['snapshot_manifest_sha256']
    assert sha(contract['panel'])==contract['panel_sha256']
    for key in ('checkpoint','training_contract'):assert sha(candidate[key])==candidate[key+'_sha256']
    sys.path.insert(0,str(SNAPSHOT))
    import torch
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    from stable_audio_tools.inference.sceneplan_generation_ar_vectorized import generate_constrained_vectorized
    from stable_audio_tools.inference.sceneplan_generation_ar_constraints import generate_with_declared_source_count
    frozen=module_at('adaptation_frozen_evaluation',SNAPSHOT/'scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py')
    fields=module_at('adaptation_fidelity',R3/'evaluation_source/generation_ar_fidelity.py')
    ScenePlanTransfusionGenerationAR.generate_constrained=generate_constrained_vectorized
    torch.set_num_threads(4);torch.manual_seed(contract['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.set_device(device);codec=frozen.ModelScenePlanCodecV4(frozen.CODEC_PATH)
    state=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False);training=json.loads(Path(candidate['training_contract']).read_text())
    assert state['run_contract']==training and state['global_step']==candidate['step']
    base,p10=frozen.load_p10v11_generation_ar(pad_id=codec.pad_id,verify_sha256=True,activation_checkpointing=False)
    assert p10.as_dict()==training['p10_load'] and codec.fingerprint==training['codec_fingerprint']
    base.load_trainable_state_dict(state['ar_adapter']);model=AdaptedGenerationAR(base,codec,rank=8,alpha=8.,binding_strength=2.);del base
    if candidate['trained_lora']:model.load_lora_state_dict(state['ar_lora'])
    else:assert all(torch.count_nonzero(branch.up)==0 for branch in model.ar_lora.values())
    del state
    model.p10_dit.to(device=device,dtype=torch.float32);model.prompt_conditioner.to(device=device,dtype=torch.bfloat16)
    model.ar_adapter.to(device=device,dtype=torch.float32);model.ar_lora.to(device=device,dtype=torch.float32)
    configure_float32_ar(model);model.eval();model.adaptation_contract()
    gate=cache_gate(model,codec,torch,device);atomic(run/'GPU_GATE.json',gate)
    limits=contract['numerical_gate'];assert gate['cached_full_max_abs']<=limits['cached_full_max_abs']
    assert gate['legal_greedy_disagreements']==0 and gate['native_p10_pre_post_exact']
    class Policy:
        def generate_constrained(self,requests,codec,**kwargs):
            return generate_with_declared_source_count(model,requests,codec,**kwargs)['token_ids']
    source=sqlite3.connect(f'file:{contract["panel"]}?mode=ro&immutable=1',uri=True);source.row_factory=sqlite3.Row
    rows=[dict(r) for r in source.execute('SELECT r.*,p.panel_index FROM rows r JOIN panel_order p USING(ordinal) ORDER BY p.panel_index')];source.close()
    db=sqlite3.connect(run/'predictions.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(panel_index INTEGER PRIMARY KEY,ordinal INTEGER NOT NULL,sample_id TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL)')
    identity={'contract_sha256':sha(root/'CONTRACT.json'),'checkpoint_sha256':candidate['checkpoint_sha256']}
    existing=dict(db.execute('SELECT key,value FROM metadata'))
    if existing:assert existing==identity
    else:db.executemany('INSERT INTO metadata VALUES (?,?)',identity.items());db.commit()
    done={r[0] for r in db.execute('SELECT panel_index FROM results')};pending=[r for r in rows if r['panel_index'] not in done];started=time.monotonic()
    for offset in range(0,len(pending),contract['batch_size']):
        if time.monotonic()-started>contract['wall_cap_s_per_worker']:raise TimeoutError('R3 validation worker budget exceeded')
        batch=pending[offset:offset+contract['batch_size']]
        outputs=frozen._generate_with_fallback(Policy(),batch,codec,device=device,max_plan_tokens=contract['max_plan_tokens'])
        for row,(tokens,error,elapsed) in zip(batch,outputs):
            raw=zlib.decompress(row['target_sceneplan_zlib']);assert hashlib.sha256(raw).hexdigest()==row['target_sceneplan_sha256'];target=json.loads(raw)
            prediction=None;status='generation_error' if tokens is None else 'ok'
            if tokens is not None:
                try:prediction=codec.decode(tokens,sample_id=row['sample_id'])
                except Exception as exc:status,error='parse_error',repr(exc)
            value={'panel_index':row['panel_index'],'ordinal':row['ordinal'],'sample_id':row['sample_id'],'source_count':row['source_count'],
                   'template_id':row['template_id'],'request':row['raw_user_request'],'target':target,'prediction':prediction,'tokens':tokens,'status':status,'error':error,
                   'generation_sec':elapsed,'target_sceneplan_sha256':row['target_sceneplan_sha256'],'fidelity':fields.compare_fields(target,prediction),
                   'legacy_metrics':frozen.score_parsed_generation(target,prediction) if prediction else {}}
            db.execute('INSERT INTO results VALUES (?,?,?,?,?)',(row['panel_index'],row['ordinal'],row['sample_id'],status,json.dumps(value,ensure_ascii=False)))
        db.commit();atomic(run/'STATUS.json',{'status':'RUNNING','rows_done':len(done)+offset+len(batch),'rows':len(rows),'elapsed_s':time.monotonic()-started})
    records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results ORDER BY panel_index')]
    assert [r['panel_index'] for r in records]==list(range(contract['rows']))
    def summarize(part):
        value=fields.summarize_fields([r['fidelity'] for r in part]);value['lexical_description_token_f1_proxy']=sum(r['legacy_metrics'].get('persistent_semantic_token_f1',0.) for r in part)/len(part);return value
    summary={'status':'COMPLETE','candidate':candidate,'contract_sha256':identity['contract_sha256'],'status_counts':dict(Counter(r['status'] for r in records)),
             'elapsed_s':time.monotonic()-started,'by_source_count':{str(n):summarize([r for r in records if r['source_count']==n]) for n in range(1,5)},
             'semantic_acceptance_pending':True,'goal_complete':False}
    atomic(run/'SUMMARY.json',summary);db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close();atomic(run/'STATUS.json',{'status':'COMPLETE','summary':str(run/'SUMMARY.json')})
    print(json.dumps({'status':'COMPLETE','candidate':candidate['label']}),flush=True)


def supervise():
    root=R3/'validation';root.mkdir(exist_ok=True);lock=(root/'SUPERVISOR_LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(root/'STATUS.json',{'status':'WAITING_FOR_TRAINING','supervisor_pid':os.getpid(),'gpu_scope':[0,1,2]})
    while True:
        status=json.loads((R3/'STATUS.json').read_text())['status']
        if status=='TRAINING_COMPLETE_AWAITING_VALIDATION':break
        if status in ('TRAINING_FAILED','BUDGET_STOP','NEEDS_ATTENTION'):
            atomic(root/'STATUS.json',{'status':'NEEDS_ATTENTION','training_status':status});return 2
        time.sleep(60)
    prepare();deadline=time.monotonic()+60
    while True:
        active=subprocess.check_output(['nvidia-smi','-i','0,1,2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not active:break
        if time.monotonic()>=deadline:
            atomic(root/'STATUS.json',{'status':'NEEDS_ATTENTION','reason':'GPU ownership after training','active_pids':active});return 2
        time.sleep(5)
    children=[]
    for index in range(3):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        log=(root/f'worker_{index}.log').open('a');child=subprocess.Popen([str(PYTHON),'-u',str(Path(__file__).resolve()),'--worker-index',str(index)],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);children.append((child,log))
    atomic(root/'STATUS.json',{'status':'RUNNING','worker_pids':[c.pid for c,l in children],'started_unix':time.time(),'gpu_scope':[0,1,2]})
    codes=[]
    for child,log in children:codes.append(child.wait());log.close()
    if not any(codes):
        values=[json.loads(p.read_text()) for p in sorted(root.glob('*/SUMMARY.json'))]
        assert len(values)==3;atomic(root/'COMPARISON.json',{'status':'COMPLETE_FIELDS_SEMANTICS_PENDING','results':values,'goal_complete':False})
    atomic(root/'STATUS.json',{'status':'COMPLETE_FIELDS_SEMANTICS_PENDING' if not any(codes) else 'FAILED','exit_codes':codes,'finished_unix':time.time()});return int(any(codes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker-index',type=int,choices=(0,1,2));args=parser.parse_args()
    if args.worker_index is None:raise SystemExit(supervise())
    try:worker(args.worker_index)
    except BaseException as exc:
        contract=json.loads((R3/'validation/CONTRACT.json').read_text());run=R3/'validation'/contract['candidates'][args.worker_index]['label'];run.mkdir(exist_ok=True)
        atomic(run/'STATUS.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}'});raise
