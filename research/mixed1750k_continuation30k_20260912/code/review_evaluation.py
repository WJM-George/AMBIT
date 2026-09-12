"""Separate CPU process verifies complete coverage, records and fixed noise."""
import argparse
from collections import defaultdict
import csv
import sys
from common import *
from evaluation_common import *
sys.path[:0]=[str(MAIN),str(OLD_RUN)]

def review(split,step):
    import torch
    torch.set_num_threads(2)
    from generation_runtime import tensor_sha
    from review_normal5000_v2 import summarize,KEYS,verify_audio
    from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio
    run=ROOT/split/f'STEP{step:06d}';completed=read(run/'INFERENCE_COMPLETE.json')
    index,specs=cases(split);plan=read(run/'PLAN.json');plan['plan_sha256']=sha(run/'PLAN.json')
    assert completed['rows']==len(specs) and completed['plan_sha256']==plan['plan_sha256']
    assert plan['index']==index and plan['case_manifest_sha256']==digest(specs)
    rows={};inventory=[];manifest=[]
    for rank in range(3):
        record=read(run/f'evaluation/rank-{rank}.json');local=specs[rank::3]
        assert record['rank']==rank and record['physical_gpu']==rank+2 and record['plan_sha256']==plan['plan_sha256']
        assert len(record['records'])==len(local)
        for spec,ref in zip(local,record['records']):
            ordinal=spec['pair_ordinal'];path=run/f'evaluation/cases/rank-{rank}/{ordinal:07d}.json'
            assert str(path)==ref['path'] and sha(path)==ref['sha256']
            row=load_case(path,spec,plan)
            audio._validate_batch_records([row],[spec],bucket=spec['latent_bucket_frames'])
            assert row['initial_noise_sha256']==tensor_sha(audio._initial_noise([spec],spec['latent_bucket_frames'],42))
            if row['edited_foa_path']:inventory.append(verify_audio((row['edited_foa_path'],row['edited_foa_sha256'],spec['model_num_samples'],True)))
            assert ordinal not in rows;rows[ordinal]=row;manifest.append(ref)
    assert set(rows)=={s['pair_ordinal'] for s in specs}
    keys=sorted(set(KEYS)|{k for r in rows.values() for k,v in r['metrics'].items() if isinstance(v,(float,int)) or v is None})
    summary,_=summarize({str(step):rows},keys,candidate=str(step))
    all_metrics,_=summarize({str(step):rows},keys,candidate=str(step),identity=True)
    summary.update({'all_finite/'+k:v for k,v in all_metrics.items()})
    # Keep original/new strata visible rather than burying them in total means.
    for cohort in ['original','addition250k','spatial_multi500k']:
        subset={o:r for o,r in rows.items() if r['cohort']==cohort}
        if not subset:continue
        s,_=summarize({str(step):subset},keys,candidate=str(step))
        summary.update({cohort+'/'+k:v for k,v in s.items()})
    write(run/'SUMMARY.json',summary)
    result=dict(status='PASS_FULL_SPLIT_INDEPENDENT_CPU_REVIEW',split=split,rows=len(rows),checkpoint_steps=step,
        plan_sha256=plan['plan_sha256'],row_manifest_sha256=digest(manifest),summary_sha256=sha(run/'SUMMARY.json'),
        independently_recomputed_noise_rows=len(rows),verified_listening_audio=inventory,all_rows_present_and_bound=True,
        existing_metric_definitions_preserved=True,human_rated=False,quality_gate_passed=False,reviewed_at=now(),
        reviewer_sha256=sha(__file__),test_used_for_selection=False,
        evaluation_index=index,cohorts=sorted({r['cohort'] for r in specs}))
    write(run/'RESULT.json',result)
    print(canonical({k:result[k] for k in ['status','rows','split','checkpoint_steps']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=SCOPES,required=True);p.add_argument('--step',type=int,choices=STEPS,required=True);a=p.parse_args();review(a.split,a.step)
