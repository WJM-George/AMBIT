#!/usr/bin/env python3
"""Apply the frozen calibrated semantic judge to separately identified pairs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1,2'
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=args.judge/'CONTRACT.json';contract=json.loads(parent.read_text())
    proof=json.loads((args.judge/'CALIBRATION_REPORT.json').read_text())
    assert proof['status']=='PASS' and proof['contract_sha256']==sha(parent)
    verified_since=json.loads((args.judge/'LAUNCH.json').read_text())['started_unix']
    asset_stats={}
    for path,digest in contract['model_asset_sha256'].items():
        stat=Path(path).stat()
        # Reuse the completed full-hash proof only if the file demonstrably
        # predates that verification. Any subsequently changed file is rehashed.
        changed=max(stat.st_mtime,stat.st_ctime)>=verified_since
        if changed:assert sha(path)==digest,path
        asset_stats[path]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,'rehash_required':changed}
    for path,digest in contract['dependency_sha256'].items():assert sha(path)==digest,path
    assert sha(contract['prompt'])==contract['prompt_sha256']
    pair_data=json.loads(args.pairs.read_text());pairs=pair_data['pairs']
    assert len({r['id'] for r in pairs})==len(pairs)
    identity={'schema':'generation_ar_frozen_semantic_pairs_v1','judge_contract_sha256':sha(parent),
              'calibration_proof_sha256':sha(args.judge/'CALIBRATION_REPORT.json'),
              'pairs':str(args.pairs.resolve()),'pairs_sha256':sha(args.pairs),
              'labels_sha256':sha(args.labels) if args.labels else None,'script_sha256':sha(Path(__file__)),
              'asset_verification':'completed calibration full hashes, ctime/mtime guard; rehash changed files','asset_stats':asset_stats,
              'gpu_scope':[1,2],'wall_cap_s':args.max_wall_seconds,'test_used':False}
    if (args.output/'CONTRACT.json').exists():assert json.loads((args.output/'CONTRACT.json').read_text())==identity
    else:atomic(args.output/'CONTRACT.json',identity)
    sys.path.insert(0,contract['dependencies'])
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    torch.set_num_threads(8);torch.manual_seed(42)
    atomic(args.output/'STATUS.json',{'status':'LOADING','pid':os.getpid()})
    tokenizer=AutoTokenizer.from_pretrained(contract['model'],local_files_only=True);tokenizer.padding_side='left'
    model=Qwen3_5ForConditionalGeneration.from_pretrained(contract['model'],local_files_only=True,dtype=torch.bfloat16,
                device_map='auto',max_memory={0:'42GiB',1:'42GiB'},attn_implementation='sdpa')
    assert set(model.hf_device_map.values()).issubset({0,1,'cuda:0','cuda:1'})
    model.eval();device=model.get_input_embeddings().weight.device
    prompt=Path(contract['prompt']).read_text().strip();started=time.monotonic()
    db=sqlite3.connect(args.output/'results.sqlite');db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    complete={r[0] for r in db.execute('SELECT id FROM results')};pending=[r for r in pairs if r['id'] not in complete]
    for offset in range(0,len(pending),contract['batch_size']):
        if time.monotonic()-started>identity['wall_cap_s']:raise TimeoutError('semantic pair wall budget exceeded')
        batch=pending[offset:offset+contract['batch_size']]
        texts=[tokenizer.apply_chat_template([{'role':'system','content':prompt},
                {'role':'user','content':json.dumps({k:r[k] for k in ['kind','reference','candidate']},ensure_ascii=False)}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
        inputs=tokenizer(texts,padding=True,truncation=False,return_tensors='pt').to(device)
        assert inputs['input_ids'].shape[1]<=contract['max_input_tokens']
        with torch.inference_mode():
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float()
            labels=logits[:,[contract['labels']['FAIL'],contract['labels']['PASS']]]
            probabilities=labels.softmax(-1)[:,1].cpu().tolist()
            masses=(labels.logsumexp(-1)-logits.logsumexp(-1)).exp().cpu().tolist()
        for pair,p,mass in zip(batch,probabilities,masses):
            result={**pair,'prediction':'PASS' if p>=contract['threshold'] else 'FAIL','pass_probability_restricted':p,'label_probability_mass':mass}
            db.execute('INSERT INTO results VALUES (?,?)',(pair['id'],json.dumps(result,ensure_ascii=False)))
        db.commit();atomic(args.output/'STATUS.json',{'status':'RUNNING','pairs_done':len(complete)+offset+len(batch),'pairs':len(pairs)})
    results={r[0]:json.loads(r[1]) for r in db.execute('SELECT id,payload FROM results')};db.close()
    assert set(results)=={r['id'] for r in pairs}
    summary={'status':'COMPLETE','pairs':len(results),'contract_sha256':sha(args.output/'CONTRACT.json'),'elapsed_scoring_s':time.monotonic()-started,
             'automatic_judge_only':True,'acceptance_established':False}
    if args.labels:
        annotation=json.loads(args.labels.read_text());assert annotation['pairs_sha256']==sha(args.pairs)
        binary=[r for r in annotation['rows'] if r['label'] in ('PASS','FAIL')]
        summary['blinded_codex_agreement']={'pairs':len(binary),'agreements':sum(results[r['id']]['prediction']==r['label'] for r in binary),
                'disagreements':[{'annotation':r,'judge':results[r['id']]} for r in binary if results[r['id']]['prediction']!=r['label']],
                'review_cases':[r for r in annotation['rows'] if r['label']=='REVIEW'], 'assessor_limit':annotation['assessor']}
    atomic(args.output/'SUMMARY.json',summary);atomic(args.output/'STATUS.json',{'status':'COMPLETE','summary':str(args.output/'SUMMARY.json')})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge',type=Path,required=True);parser.add_argument('--pairs',type=Path,required=True)
    parser.add_argument('--labels',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-wall-seconds',type=int,default=1800);args=parser.parse_args()
    try:run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True,exist_ok=True);atomic(args.output/'STATUS.json',{'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
        raise
