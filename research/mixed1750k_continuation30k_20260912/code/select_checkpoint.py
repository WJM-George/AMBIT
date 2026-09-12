"""Select once on complete, matched validation evidence; test is never read."""
from collections import defaultdict
import csv
import math
import statistics
from common import *
from evaluation_common import *

def finite(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)

def matched_metric(base,candidate,ordinals,metric,protocol):
    available=[o for o in ordinals if finite(base[o]['metrics'].get(metric))]
    if len(available)<protocol['minimum_paired_rows']:return None
    paired=[o for o in available if finite(candidate[o]['metrics'].get(metric))]
    retention=len(paired)/len(available)
    if len(paired)<protocol['minimum_paired_rows'] or retention<protocol['minimum_baseline_metric_retention']:
        return {'coverage_failed':True,'baseline_rows':len(available),'paired_rows':len(paired),'retention':retention}
    b=[base[o]['metrics'][metric] for o in paired];c=[candidate[o]['metrics'][metric] for o in paired]
    delta=[v-u for u,v in zip(b,c)];mean=statistics.mean(delta)
    se=statistics.stdev(delta)/math.sqrt(len(delta)) if len(delta)>1 else 0
    return dict(coverage_failed=False,baseline_rows=len(available),paired_rows=len(paired),retention=retention,
        baseline_mean=statistics.mean(b),candidate_mean=statistics.mean(c),baseline_std=statistics.stdev(b),
        paired_mean_delta=mean,paired_99pct_descriptive_interval=[mean-2.5758293035489004*se,mean+2.5758293035489004*se])

def score_candidate(base,candidate,protocol):
    groups=defaultdict(list);oldgroups=defaultdict(list)
    assert set(base)==set(candidate)
    for o,r in base.items():
        groups[r['operation']].append(o)
        if r['cohort']=='original':oldgroups[r['operation']].append(o)
    dimensions={};details={};guards={};eligible=True
    for dimension,spec in protocol['dimensions'].items():
        operation_scores=[];details[dimension]={}
        for operation in spec['operations']:
            metric_scores=[]
            for metric,direction in spec['metrics'].items():
                direction=spec.get('operation_metric_directions',{}).get(operation,{}).get(metric,direction)
                result=matched_metric(base,candidate,groups[operation],metric,protocol)
                details[dimension][operation+'/'+metric]=result
                if result is None:continue
                if result['coverage_failed']:eligible=False;continue
                scale=max(result['baseline_std'],abs(result['baseline_mean'])*.01,1e-6)
                metric_scores.append(max(-3.,min(3.,direction*result['paired_mean_delta']/scale)))
            if metric_scores:operation_scores.append(statistics.mean(metric_scores))
        if not operation_scores:eligible=False;dimensions[dimension]=None
        else:dimensions[dimension]=statistics.mean(operation_scores)
    for operation,ordinals in oldgroups.items():
        for metric,guard in protocol['original_operation_guards'].items():
            if operation not in guard.get('operations',oldgroups):continue
            direction=guard.get('operation_directions',{}).get(operation,guard['direction'])
            result=matched_metric(base,candidate,ordinals,metric,protocol)
            if result is None:continue
            if result['coverage_failed']:passed=False
            else:
                tolerance=max(guard['absolute_tolerance'],abs(result['baseline_mean'])*guard['relative_tolerance'])
                worsening=-direction*result['paired_mean_delta'];passed=worsening<=tolerance
                result.update(allowed_worsening=tolerance,observed_worsening=worsening)
            guards[operation+'/'+metric]=dict(**result,passed=passed)
            eligible &= passed
    score=sum(protocol['dimension_weights'][k]*v for k,v in dimensions.items()) if all(v is not None for v in dimensions.values()) else None
    return dict(eligible=bool(eligible),score=score,dimension_scores=dimensions,metric_details=details,original_operation_guards=guards,
        failed_guards=[k for k,v in guards.items() if not v['passed']])

def decide(scores):
    eligible=[(step,r['score']) for step,r in scores.items() if step!=50000 and r['eligible'] and r['score'] is not None and r['score']>0]
    return sorted(eligible,key=lambda x:(-x[1],x[0]))[0][0] if eligible else 50000

def load_reviewed(step,specs):
    run=ROOT/f'validation/STEP{step:06d}';report=read(run/'RESULT.json');plan=read(run/'PLAN.json');plan['plan_sha256']=sha(run/'PLAN.json')
    assert report['status']=='PASS_FULL_SPLIT_INDEPENDENT_CPU_REVIEW' and report['rows']==len(specs)
    assert report['checkpoint_steps']==step and report['split']=='validation' and report['plan_sha256']==plan['plan_sha256']
    assert report['summary_sha256']==sha(run/'SUMMARY.json') and plan['selection_protocol_sha256']==sha(ROOT/'SELECTION_PROTOCOL.json')
    rows={};manifest=[]
    for rank in range(3):
        rr=read(run/f'evaluation/rank-{rank}.json')['records'];local=specs[rank::3];assert len(rr)==len(local)
        for spec,ref in zip(local,rr):
            path=run/f'evaluation/cases/rank-{rank}/{spec["pair_ordinal"]:07d}.json'
            assert str(path)==ref['path'] and sha(path)==ref['sha256']
            row=load_case(path,spec,plan)
            keep=['pair_id','pair_ordinal','operation','cohort','source_foa_sha256','target_foa_sha256','new_sceneplan_sha256','source_latent_sha256','initial_noise_sha256','metrics']
            rows[spec['pair_ordinal']]={k:row[k] for k in keep};manifest.append(ref)
    assert report['row_manifest_sha256']==digest(manifest) and set(rows)=={s['pair_ordinal'] for s in specs}
    return rows,dict(path=str(run/'RESULT.json'),sha256=sha(run/'RESULT.json'))

def main():
    protocol=read(ROOT/'SELECTION_PROTOCOL.json');assert protocol['steps']==STEPS and protocol['rows']==2000
    _,specs=cases('validation');base,ref=load_reviewed(50000,specs);references={'50000':ref};scores={50000:score_candidate(base,base,protocol)}
    match=['pair_id','pair_ordinal','operation','cohort','source_foa_sha256','target_foa_sha256','new_sceneplan_sha256','source_latent_sha256','initial_noise_sha256']
    for step in STEPS[1:]:
        rows,ref=load_reviewed(step,specs);references[str(step)]=ref
        for o in base:
            for k in match:assert rows[o][k]==base[o][k],(step,o,k)
        scores[step]=score_candidate(base,rows,protocol);del rows
    selected=decide(scores)
    result=dict(status='COMPLETE_VALIDATION_ONLY_CHECKPOINT_SELECTION',selected_checkpoint_steps=selected,
        new_training_steps=selected-50000,selection_split='validation',selection_rows=len(specs),
        protocol_sha256=sha(ROOT/'SELECTION_PROTOCOL.json'),reviews=references,scores=scores,
        selected_checkpoint=checkpoint(selected),test_used_for_selection=False,quality_gate_passed=False,human_rated=False,selected_at=now())
    path=ROOT/'SELECTION.json'
    if path.exists():
        previous=read(path)
        assert previous['selected_checkpoint_steps']==selected and previous['protocol_sha256']==result['protocol_sha256'] and previous['reviews']==references
        return
    write(path,result)
    write(ROOT/'BEST_CHECKPOINT.json',dict(checkpoint=result['selected_checkpoint'],selection=str(path),selection_sha256=sha(path),validation_only=True,existing_50k_retained=True))
    with (ROOT/'VALIDATION_CHECKPOINT_COMPARISON.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['total_step','continued_step','eligible','selected','overall_score','audio','content','spatial','timing','preservation','failed_old_operation_guards'])
        for step,r in scores.items():writer.writerow([step,step-50000,r['eligible'],step==selected,r['score'],*[r['dimension_scores'][k] for k in protocol['dimension_weights']],';'.join(r['failed_guards'])])
    print(canonical({'selected_checkpoint_steps':selected,'selection_split':'validation','test_used':False}),flush=True)

if __name__=='__main__':main()
