#!/usr/bin/env python3
"""N2: 25/25/50 presentations, diverse request-first families, raw-only panels."""
import os
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))


def local_module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_transfusion_generation import render_generation_request
    from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import evaluate_natural_request
    grammar=local_module('build_generation_ar_compositional_requests')
    rewrite=local_module('prepare_generation_ar_natural_pilot')
    annotate=local_module('curate_generation_ar_natural_pilot')
    codec=ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    tokenizer=AutoTokenizer.from_pretrained(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B",local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(stage,**kw):
        value={'status':stage,'elapsed_s':time.monotonic()-started,**kw};dump(args.output/'STATUS.json',value);print(json.dumps(value),flush=True)
    status('BUILDING_REQUEST_FIRST_FAMILIES')
    b,attempts=grammar.make_pairs(252,400,codec,split='train')
    bv,val_attempts=grammar.make_pairs(8252,64,codec,split='validation')
    assert not {p['family_id'] for p in b}&{p['family_id'] for p in bv}
    for record in b+bv:
        record['completion_provenance_by_view']=[annotate.provenance(req) for req in record['requirements']]
    dump(args.output/'request_first_train_pairs.json',{'records':b})
    dump(args.output/'request_first_validation_pairs.json',{'records':bv})
    status('PACKING_OLD_AND_REQUEST_FIRST_TRAINING',request_first_train_families=len(b),request_first_validation_families=len(bv))
    old=[p for p in json.loads((args.n1/'training.json').read_text())['parents'] if p['route']=='plan_to_request'];assert len(old)==800
    parents=[]
    for p in old:
        p=deepcopy(p);plans=[];token_arrays=[];precise=[]
        for view in range(5):
            plan=deepcopy(p['target_sceneplan'])
            if view in (2,4):plan['sources'].reverse()
            for i,source in enumerate(plan['sources']):source['source_id']=f'source_{i}'
            ids=codec.encode(plan)['input_ids'].tolist();assert codec.decode(ids,sample_id=plan['sample_id'])==plan
            plans.append(plan);token_arrays.append(ids);precise.append(render_generation_request(plan,split='train')[0])
        assert len(set(map(len,token_arrays)))==1
        p.update(target_token_ids_by_view=token_arrays,precise_requests_by_view=precise)
        parents.append(p)
    for r in b:
        parents.append({'id':r['id'],'family_id':r['family_id'],'route':'request_to_plan','source_count':r['source_count'],
                        'target_sceneplan':r['targets'][0]['plan'],'target_token_ids':r['targets'][0]['tokens'],
                        'target_token_ids_by_view':[t['tokens'] for t in r['targets']],
                        'precise_request':render_generation_request(r['targets'][0]['plan'],split='train')[0],
                        'precise_requests_by_view':[render_generation_request(t['plan'],split='train')[0] for t in r['targets']],
                        'natural_requests':[v['request'] for v in r['views']]})
    for p in parents:
        lengths=[len(ids) for ids in tokenizer(p['precise_requests_by_view']+p['natural_requests'])['input_ids']]
        if max(lengths)>512:raise ValueError(f"Request exceeds encoder limit: {p['id']} {max(lengths)}")
        p['request_token_lengths']=lengths
    # Balanced shuffled pools avoid the length-sorted class cycles found in D0.
    groups={(mode,n):[i for i,p in enumerate(parents) if p['source_count']==n and (mode=='precise' or p['route']==('plan_to_request' if mode=='five_view' else 'request_to_plan'))] for mode in ('precise','five_view','request_first') for n in range(1,5)}
    pools={};rng=random.Random(202609052)
    def draw(mode,n):
        key=(mode,n)
        if not pools.get(key):pools[key]=groups[key].copy();rng.shuffle(pools[key])
        return pools[key].pop()
    schedule=[]
    for step in range(2000):
        batch=[]
        for n in range(1,5):
            for mode in ('precise','precise','five_view','five_view','request_first','request_first','request_first','request_first'):
                batch.append({'index':draw(mode,n),'mode':mode,'view':rng.randrange(5)})
        rng.shuffle(batch);schedule.append(batch)
    modes=Counter(x['mode'] for batch in schedule for x in batch)
    assert modes=={'precise':16000,'five_view':16000,'request_first':32000}
    token_modes=Counter()
    for batch in schedule:
        for x in batch:token_modes[x['mode']]+=len(parents[x['index']]['target_token_ids_by_view'][x['view']])-1
    experiment={'id':'N2_compositional_252550','steps':2000,'batch_size':32,'wall_cap_seconds':5400,
                'schedule_description':'Same presentations, per-view complete-source ordering and target token arrays in both arms; natural mix 25% precise / 25% existing-plan rewrite / 50% request-first.',
                'success_rule':'Primary independent N0 natural joint >=10 pp above precise control; generated held-out-family natural joint >=10 pp above control; no per-count precise count/core/applicable-constraint regression >2 pp versus control or R3. Pilot only; no final acceptance or full-corpus promotion from a grammar panel alone.',
                'limitations':'Finite authored event bank and five compositional English styles. Semantic scenario families are held out but grammar and entity vocabulary are shared. Request-first families increase from 32 to 1600. Data diversity, proportions and mention-order supervision form one intervention bundle; individual effects are not identified. Elapsed time and target-token shares are reported.'}
    dump(args.output/'training.json',{'parents':parents,'schedule':schedule,'experiment':experiment})
    original_val=json.loads((args.n1/'validation_pairs.json').read_text())['pairs'];precise=[p for p in original_val if p['route']=='precise_validation'];external=[p for p in original_val if p['route']!='precise_validation']
    assert len(precise)==1024 and len(external)==16
    selected=[]
    for n in range(1,5):selected.extend([p for p in precise if len(p['requirements']['sources'])==n][:64])
    # New small 512-row mixed panel has exactly 128/128/256 rows. Old precise
    # guardrails (1024) and 16 independently authored natural rows remain separate.
    mixed_precise=[];a_val=[]
    for n in range(1,5):
        rows=[p for p in selected if len(p['requirements']['sources'])==n]
        for p in rows[:32]:
            p=deepcopy(p);p['panels']=['mixed_precise','precise_retention'];mixed_precise.append(p)
        for i,p in enumerate(rows[32:]):
            r=rewrite.five_view(p['target_sceneplan'],i%5);r['split']='validation';r['family_id']=p['family_id'];r['id']='N2_rewrite_validation/'+p['id'];r['route']='plan_to_request_validation';r['panels']=['mixed_plan_rewrite']
            a_val.append(r)
    natural_val=[]
    for i,r in enumerate(bv):
        view=i%5;plan=r['targets'][view]['plan'];req=r['requirements'][view];binding=r['source_bindings_by_view'][view]
        labels={(q['key'],s['source_id']):binding[q['key']]==s['source_id'] for q in req['sources'] for s in plan['sources']}
        assert evaluate_natural_request(r['views'][view]['request'],req,plan,labels,completion_reasonable=True)['acceptance_joint']
        natural_val.append({'id':r['id'],'family_id':r['family_id'],'split':'validation','route':'request_to_plan_compositional_validation','request':r['views'][view]['request'],'requirements':req,'target_sceneplan':plan,'source_bindings':binding,'panels':['mixed_request_first'],'completion_provenance':r['completion_provenance_by_view'][view]})
    for p in precise:p['panels']=['precise_retention']+(['mixed_precise'] if p['id'] in {q['id'] for q in mixed_precise} else [])
    for p in external:p['panels']=['independent_N0_natural']
    val=precise+external+a_val+natural_val
    assert len({p['id'] for p in val})==len(val)==1424
    assert not {p['family_id'] for p in parents}&{p['family_id'] for p in val}
    assert not {r for p in parents for r in p['natural_requests']+p['precise_requests_by_view']}&{p['request'] for p in val}
    dump(args.output/'validation_pairs.json',{'pairs':val});dump(args.output/'validation_raw_requests.json',{'requests':[{k:p[k] for k in ('id','request')} for p in val]})
    dump(args.output/'mixed_validation_ids.json',{'precise':[p['id'] for p in mixed_precise],'plan_to_request':[p['id'] for p in a_val],'request_to_plan':[p['id'] for p in natural_val]})
    sample=[p for n in range(1,5) for p in [r for r in b if r['source_count']==n][:8]]
    dump(args.output/'PAIR_REVIEW_SAMPLE.json',{'records':sample})
    report={'status':'DETERMINISTIC_CHECKS_PASS_MANUAL_REVIEW_PENDING','old_train_families':800,'request_first_train_families':1600,'request_first_validation_families':256,'views_per_train_parent':5,'train_presentations':64000,'presentations_by_mode':dict(modes),'target_tokens_by_mode':dict(token_modes),'max_request_tokens':max(max(p['request_token_lengths']) for p in parents),'validation_rows':1424,'mixed_validation_rows':512,'mixed_validation_counts':{'precise':128,'plan_to_request':128,'request_to_plan':256},'train_val_family_overlap':False,'request_first_generation_attempts':attempts,'request_first_validation_attempts':val_attempts,'test_used':False,'source_sha256':sha(Path(__file__)),'artifact_sha256':{p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='STATUS.json'}}
    dump(args.output/'PREPARATION.json',report);status(report['status'],training_sha256=report['artifact_sha256']['training.json'],validation_rows=len(val))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--n1',type=Path,required=True);p.add_argument('--output',type=Path,required=True);prepare(p.parse_args())
