#!/usr/bin/env python3
"""N1 pilot: shared targets, five English views, request-first seeds and replay."""
import os
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import sys
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_transfusion_generation import render_generation_request


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    from transformers import AutoTokenizer
    import numpy as np
    spec=importlib.util.spec_from_file_location('five_views',Path(__file__).with_name('prepare_generation_ar_natural_pilot.py'));views=importlib.util.module_from_spec(spec);spec.loader.exec_module(views)
    tokenizer=AutoTokenizer.from_pretrained(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B",local_files_only=True)
    codec=ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    quality=json.loads((args.n0/'QUALITY_GATE.json').read_text());assert quality['status']=='PASS_CURATED_SMALL_BATCH'
    db=sqlite3.connect('file:'+str(args.train_db)+'?mode=ro&immutable=1',uri=True)
    high=db.execute('SELECT MAX(ordinal) FROM rows').fetchone()[0]
    candidates=random.Random(142).sample(range(high+1),20000);parents=[];counts=Counter();rejects=Counter()
    for ordinal in candidates:
        if len(parents)==800:break
        sid,request,blob,tokens,number=db.execute('SELECT sample_id,raw_user_request,target_sceneplan_zlib,target_token_ids_u16le,source_count FROM rows WHERE ordinal=?',(ordinal,)).fetchone()
        if counts[number]>=200:continue
        p=json.loads(zlib.decompress(blob))
        if views.seed_caption_risks(p):rejects['conservative_caption_ambiguity']+=1;continue
        if views.re_non_english_script(json.dumps(p)):rejects['non_english_script']+=1;continue
        rendered=[views.five_view(p,i) for i in range(5)]
        texts=[request]+[r['request'] for r in rendered]
        lengths=[len(tokenizer(t)['input_ids']) for t in texts]
        if max(lengths)>512:rejects['any_view_exceeds_512_encoder_tokens']+=1;continue
        # Do not trust a serialized target length if it disagrees with the codec.
        ids=np.frombuffer(tokens,dtype='<u2').astype(int).tolist()
        assert codec.decode(ids,sample_id=sid)==p
        parents.append({'id':sid,'family_id':'old_train/'+sid,'route':'plan_to_request','source_count':number,'target_sceneplan':p,'target_token_ids':ids,'precise_request':request,'natural_requests':[r['request'] for r in rendered],'request_token_lengths':lengths,'source_ordinal':ordinal})
        counts[number]+=1
    db.close();assert len(parents)==800 and all(counts[n]==200 for n in range(1,5))
    b=[]
    for pair in json.loads((args.n0/'train_pairs.json').read_text())['pairs']:
        if pair['route']!='request_to_plan':continue
        precise,_=render_generation_request(pair['target_sceneplan'],split='train')
        texts=[precise,pair['request']];lengths=[len(tokenizer(t)['input_ids']) for t in texts];assert max(lengths)<=512
        b.append({'id':pair['id'],'family_id':pair['family_id'],'route':'request_to_plan','source_count':len(pair['target_sceneplan']['sources']),'target_sceneplan':pair['target_sceneplan'],'target_token_ids':pair['target_token_ids'],'precise_request':precise,'natural_requests':[pair['request']],'request_token_lengths':lengths,'reviewed_pair':pair['id']})
    assert len(b)==32
    validation=json.loads((args.n0/'validation_pairs.json').read_text())['pairs']
    panel=sqlite3.connect('file:'+str(args.panel)+'?mode=ro&immutable=1',uri=True)
    precise_val=[]
    for sid,request,blob in panel.execute('SELECT sample_id,raw_user_request,target_sceneplan_zlib FROM rows ORDER BY ordinal'):
        p=json.loads(zlib.decompress(blob));req=views.five_view(p,0)['requirements']
        req['output_order']=[s['key'] for s in req['sources']];req['order_evidence']=request
        for s in req['sources']:
            s['evidence']=request
            for c in s['constraints']:c['evidence']=request
        for c in req['scene']:c['evidence']=request
        precise_val.append({'id':sid,'family_id':'old_validation/'+sid,'split':'validation','route':'precise_validation','request':request,'requirements':req,'target_sceneplan':p})
    panel.close();assert len(precise_val)==1024
    families={p['family_id'] for p in parents+b};assert not families & {p['family_id'] for p in validation+precise_val}
    # Same deterministic presentations and target token arrays in both arms.
    rng=random.Random(20260905);groups={(route,n):[i for i,p in enumerate(parents+b) if p['route']==route and p['source_count']==n] for route in ('plan_to_request','request_to_plan') for n in range(1,5)}
    schedule=[]
    for step in range(600):
        batch=[]
        for n in range(1,5):
            for mode in ('precise','five_view','five_view','request_first','request_first'):
                route='request_to_plan' if mode=='request_first' else 'plan_to_request'
                index=rng.choice(groups[route,n]);view=rng.randrange(5) if mode=='five_view' else 0
                batch.append({'index':index,'mode':mode,'view':view})
        rng.shuffle(batch);schedule.append(batch)
    args.output.mkdir(parents=True,exist_ok=False)
    artifacts={'training.json':{'parents':parents+b,'schedule':schedule},'validation_pairs.json':{'pairs':validation+precise_val},'validation_raw_requests.json':{'requests':[{k:p[k] for k in ('id','request')} for p in validation+precise_val]},'natural_validation_raw_requests.json':{'requests':[{k:p[k] for k in ('id','request')} for p in validation]}}
    for name,value in artifacts.items():(args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    # A reproducible stratified review panel; promotion waits for this audit.
    sampled=[p for n in range(1,5) for p in [x for x in parents if x['source_count']==n][:8]]
    (args.output/'PARENT_REVIEW_SAMPLE.json').write_text(json.dumps({'parents':sampled},ensure_ascii=False,indent=2)+'\n')
    report={'status':'PREPARED_ADDITIONAL_PARENT_REVIEW_PENDING','old_training_parents':800,'source_counts':dict(counts),'five_views_per_parent':5,'request_first_unique_train_requests':32,'request_first_unique_validation_requests':16,'precise_validation_scenes':1024,'steps':600,'batch_size':20,'presentations':12000,'presentations_by_mode':dict(Counter(x['mode'] for step in schedule for x in step)),'max_request_tokens':max(n for p in parents+b for n in p['request_token_lengths']),'screen_rejections':dict(rejects),'test_used':False,'family_overlap':False,'teacher_augmented_unique_diversity_limit':'Request-first has 32 authored scenarios oversampled to 40% of presentations; this pilot does not claim 40% unique request-first coverage in a rebuilt full corpus.','n0_gate_sha256':sha(args.n0/'QUALITY_GATE.json'),'source_sha256':sha(Path(__file__)),'artifact_sha256':{p.name:sha(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/'PREPARATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','old_training_parents','request_first_unique_train_requests','max_request_tokens','presentations_by_mode')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('n0','train-db','panel','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
