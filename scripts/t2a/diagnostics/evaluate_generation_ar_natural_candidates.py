#!/usr/bin/env python3
"""Freeze/score raw-only candidate results with one coupled source assignment."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import evaluate_natural_request, source_constraint_pass
from stable_audio_tools.data.sceneplan_generation_ar_exact_proof import prove_exact_satisfaction


def digest(value): return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()!=text:raise ValueError('refusing to overwrite a different frozen artifact '+str(path))
    path.write_text(text)


def db_rows(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro&immutable=1',uri=True)
    rows={i:json.loads(p) for i,p in db.execute('SELECT id,payload FROM results')};db.close();return rows


def prepare(args):
    gold=json.loads(args.validation.read_text())['pairs'];byid={r['id']:r for r in gold};assert len(byid)==len(gold)
    candidates=[];bindings=[];pairs={};completion=[];exact_proofs=[]
    for value in args.candidate:
        name,path=value.split('=',1);path=Path(path).resolve()
        assert json.loads((path.parent/'STATUS.json').read_text())['status']=='COMPLETE'
        predictions=db_rows(path);assert set(predictions)==set(byid)
        for sid,row in predictions.items():
            expected=byid[sid];assert row['request']==expected['request']
            assert row['model_input_sha256']==hashlib.sha256(expected['request'].encode()).hexdigest()
            prediction=row['prediction'];sources=prediction['sources'] if prediction else []
            exact_proof = (prove_exact_satisfaction(expected['request'], expected['requirements'], prediction)
                if getattr(args, 'prove_exact_satisfied', False) else None)
            if exact_proof is not None:
                exact_proofs.append({'candidate': name, 'id': sid, 'prediction_sha256': digest(prediction),
                    'requirements_sha256': digest(expected['requirements']), **exact_proof})
            if expected['route']!='precise_validation':
                completion.append({'candidate':name,'id':sid,'request':expected['request'],'prediction':prediction,'prediction_sha256':digest(prediction),'reasonable':False if prediction is None else None,'review_reason':'No generated plan' if prediction is None else 'PENDING_CODEX_REVIEW_OF_FREE_COMPLETION'})
            refs=expected['requirements']['sources'];ordered=expected['requirements'].get('output_order') is not None
            for index,ref in enumerate(refs):
                possible=sources[index:index+1] if ordered else sources
                for source in possible:
                    if ref['kind']!=source['kind']:continue
                    field='speaker_description' if ref['kind']=='speech' else 'description'
                    pair={'kind':ref['kind'],'reference':ref['core'],'candidate':source[field]}
                    bound={'candidate':name,'id':sid,'key':ref['key'],'source_id':source['source_id']}
                    if ' '.join(pair['reference'].split())==' '.join(pair['candidate'].split()):bound['exact_core_text']=True
                    elif exact_proof is not None:bound['irrelevant_to_proven_complete_assignment']=True
                    else:
                        pid=digest(pair);pairs[pid]={'id':pid,**pair};bound['pair_id']=pid
                    bindings.append(bound)
        candidates.append({'name':name,'prediction_db':str(path),'prediction_sha256':sha(path),'contract_sha256':sha(path.parent/'CONTRACT.json'),'rows':len(predictions)})
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'pairs.json',{'pairs':[pairs[k] for k in sorted(pairs)],'test_used':False})
    write(args.output/'bindings.json',{'candidates':candidates,'bindings':bindings,'validation':str(args.validation.resolve()),'validation_sha256':sha(args.validation),'source_sha256':sha(Path(__file__)),'exact_satisfaction_proofs':exact_proofs,'scope':'All same-kind semantic possibilities for unresolved unordered requests; a sufficient complete coupled assignment may bypass irrelevant semantic edges. Fixed order for explicitly ordered precise requests; one shared assignment for every field.'})
    write(args.output/'completion_review_inputs.json',{'rows':completion,'reviewer':'PENDING; not independent human review','prediction_hash_required':True})
    print(json.dumps({'candidates':len(candidates),'semantic_pairs':len(pairs),'completion_reviews':len(completion),'proven_exact_satisfied_scenes':len(exact_proofs)}))


def metrics(rows):
    rates={};n=len(rows)
    def rate(key,values): rates[key]={'correct':sum(values),'total':len(values),'rate':sum(values)/len(values) if values else None}
    for name,field in [('count','count_correct'),('request_joint_before_completion_review','request_constraints_joint'),('joint','acceptance_joint')]:rate(name,[r['scored'][field] for r in rows])
    rate('valid',[r['scored']['valid'] for r in rows]);rate('all_requested_fields_joint',[r['fields_joint'] for r in rows])
    rate('joint_upper_bound',[r['scored']['acceptance_joint'] or (r['scored']['request_constraints_joint'] and r['scored']['completion_reasonable'] is None) for r in rows])
    # Scene-level intervals only. Individual source scores within a scene are
    # correlated and must not be presented as independent Bernoulli trials.
    for name in ('count','joint','joint_upper_bound','valid','all_requested_fields_joint'):
        k,m=rates[name]['correct'],rates[name]['total']
        if m:
            p=k/m;z=1.95996398454;d=1+z*z/m;mid=(p+z*z/(2*m))/d;half=z*math.sqrt(p*(1-p)/m+z*z/(4*m*m))/d
            rates[name]['wilson_95_scene_interval']=[max(0.,mid-half),min(1.,mid+half)]
    correct=sum(sum(s['core_semantics'] is True for s in r['scored']['sources']) for r in rows)
    requested=sum(r['scored']['requested_count'] for r in rows);predicted=sum(r['scored']['predicted_count'] for r in rows)
    rates['core_recall']={'correct':correct,'total':requested,'rate':correct/requested if requested else 0.}
    rates['core_precision']={'correct':correct,'total':predicted,'rate':correct/predicted if predicted else 0.}
    for name in ('motion','motion_with_direction_constraints','onset','offset','event_duration','start','end','numeric_start','numeric_end','numeric_onset','numeric_offset','relations','speech_words'):
        rate(name,[v for r in rows for v in r['field_checks'][name]])
    return {'scenes':n,'metrics':rates,'missing_sources':sum(r['scored']['missing'] for r in rows),'extra_sources':sum(r['scored']['extra'] for r in rows),'pending_semantics':sum(r['scored']['semantic_pending'] for r in rows),'pending_completion_review':sum(r['scored']['completion_reasonable'] is None for r in rows),'pending_completion_that_can_change_joint':sum(r['scored']['request_constraints_joint'] and r['scored']['completion_reasonable'] is None for r in rows)}


def n2_decision(by_panel):
    """Frozen N2 comparisons; independent and generated language stay separate."""
    required=('precise_control','natural_mix','r3_baseline')
    if not all(c in by_panel for c in required):return {'status':'PENDING_REQUIRED_CANDIDATES','full_data_promotion':False}
    checks=[]
    for panel in ('independent_N0_natural','mixed_request_first'):
        gain=0.;low=0.;high=0.;pending=0
        for n in map(str,range(1,5)):
            a=by_panel['natural_mix'][panel][n];b=by_panel['precise_control'][panel][n]
            gain+=a['metrics']['joint']['rate']-b['metrics']['joint']['rate']
            low+=a['metrics']['joint']['rate']-b['metrics']['joint_upper_bound']['rate']
            high+=a['metrics']['joint_upper_bound']['rate']-b['metrics']['joint']['rate']
            pending+=a['pending_semantics']+b['pending_semantics']+a['pending_completion_that_can_change_joint']+b['pending_completion_that_can_change_joint']
        checks.append({'panel':panel,'joint_macro_gain':gain/4,'gain_lower_bound_pending_reviews':low/4,'gain_upper_bound_pending_reviews':high/4,'pending_actionable_reviews':pending,'pass':pending==0 and gain/4>=.1-1e-9,'certain_fail':high/4<.1-1e-9})
    guards=[]
    for comparison in ('precise_control','r3_baseline'):
        for n in map(str,range(1,5)):
            a=by_panel['natural_mix']['precise_retention'][n]['metrics'];b=by_panel[comparison]['precise_retention'][n]['metrics']
            for metric in ('count','core_recall','core_precision','motion','onset','offset','start','end','speech_words','all_requested_fields_joint'):
                if a[metric]['rate'] is None or b[metric]['rate'] is None:continue
                delta=a[metric]['rate']-b[metric]['rate'];guards.append({'comparison':comparison,'count':int(n),'metric':metric,'natural_minus_comparison':delta,'pass':delta>=-.02-1e-9})
    certain_fail=any(c['certain_fail'] for c in checks) or any(not g['pass'] for g in guards)
    pending=any(c['pending_actionable_reviews'] for c in checks)
    return {'status':'FAIL_PILOT_RULE' if certain_fail else 'PENDING_ACTIONABLE_REVIEWS' if pending else 'PASS_PILOT_ONLY' if all(c['pass'] for c in checks) else 'FAIL_PILOT_RULE','natural_panel_comparisons':checks,'precise_guardrails':guards,'full_data_promotion':False,'reason_for_no_full_promotion':'Finite shared grammar and event vocabulary plus only 16 independently authored natural requests. Broader independent natural validation is required.'}


def n3_decision(by_panel):
    """Keep a count-component signal separate from complete request execution."""
    required=('n2_natural_mix','r3_baseline','count_aux','count_header_aug')
    if not all(c in by_panel for c in required):return {'status':'PENDING_REQUIRED_CANDIDATES','full_data_promotion':False}
    baseline=by_panel['n2_natural_mix'];arms={}
    for candidate in ('count_aux','count_header_aug'):
        current=by_panel[candidate];guards=[]
        for panel in ('mixed_request_first','precise_retention'):
            for n in map(str,range(1,5)):
                delta=current[panel][n]['metrics']['count']['rate']-baseline[panel][n]['metrics']['count']['rate']
                guards.append({'panel':panel,'count':int(n),'delta':delta,'pass':delta>=-.02-1e-9})
        gain=sum(current['mixed_request_first'][str(n)]['metrics']['count']['rate']-baseline['mixed_request_first'][str(n)]['metrics']['count']['rate'] for n in (2,3,4))/3
        natural_correct=lambda p:sum(p['independent_N0_natural'][str(n)]['metrics']['count']['correct'] for n in range(1,5))
        independent_delta=natural_correct(current)-natural_correct(baseline)
        count_pass=gain>=.1-1e-9 and independent_delta>=0 and all(g['pass'] for g in guards)
        request_rule=n2_decision({'natural_mix':current,'precise_control':baseline,'r3_baseline':by_panel['r3_baseline']})
        arms[candidate]={'count_hypothesis_status':'PASS_COUNT_COMPONENT_ONLY' if count_pass else 'FAIL_COUNT_HYPOTHESIS',
            'complex_count_macro_gain':gain,'independent_natural_count_correct_delta':independent_delta,
            'count_retention_guards':guards,'overall_request_pilot_rule':request_rule,
            'model_promoted':False,'full_data_promotion':False}
    return {'status':'COUNT_COMPONENT_AND_REQUEST_RULES_REPORTED_SEPARATELY','arms':arms,'full_data_promotion':False,
            'limits':'Same finite N2 data. Count-only improvement cannot establish ordinary English, full precise retention, or end-to-end audio acceptance.'}


def score(args):
    root=args.output;binding=json.loads((root/'bindings.json').read_text());validation=Path(binding['validation']);assert sha(validation)==binding['validation_sha256']
    refs={r['id']:r for r in json.loads(validation.read_text())['pairs']}
    judged=db_rows(root/'scoring/results.sqlite')
    assert json.loads((root/'scoring/STATUS.json').read_text())['status']=='COMPLETE'
    assert json.loads((root/'scoring/CONTRACT.json').read_text())['pairs_sha256']==sha(root/'pairs.json')
    reviews=json.loads(args.completion_reviews.read_text()) if args.completion_reviews else {'rows':[]}
    completed={(r['candidate'],r['id']):r for r in reviews['rows']};labels=defaultdict(dict)
    exact_proofs={(r['candidate'],r['id']):r for r in binding.get('exact_satisfaction_proofs',[])}
    for b in binding['bindings']:
        if b.get('irrelevant_to_proven_complete_assignment'):
            assert (b['candidate'],b['id']) in exact_proofs
            continue  # Unknown, not a fabricated negative semantic label.
        if b.get('exact_core_text'):label=True
        else:
            j=judged[b['pair_id']];assert digest({k:j[k] for k in ('kind','reference','candidate')})==b['pair_id'];label=j['prediction']=='PASS'
        labels[b['candidate'],b['id']][b['key'],b['source_id']]=label
    groups=defaultdict(list);panel_groups=defaultdict(list);scene_results=[]
    for c in binding['candidates']:
        name=c['name'];assert sha(c['prediction_db'])==c['prediction_sha256']
        for sid,row in db_rows(c['prediction_db']).items():
            ref=refs[sid];prediction=row['prediction'];requirements=ref['requirements'];precise=ref['route']=='precise_validation'
            review=completed.get((name,sid))
            if review:assert review['prediction_sha256']==digest(prediction) and isinstance(review['reasonable'],bool)
            reasonable=True if precise else review['reasonable'] if review else False if prediction is None else None
            scored=evaluate_natural_request(ref['request'],requirements,prediction,labels[name,sid],completion_reasonable=reasonable)
            if (name,sid) in exact_proofs:
                proof=exact_proofs[name,sid]
                assert proof['prediction_sha256']==digest(prediction) and proof['requirements_sha256']==digest(requirements)
                assert prove_exact_satisfaction(ref['request'],requirements,prediction) is not None
                assert scored['request_constraints_joint'] and not scored['semantic_pending']
            sources={s['source_id']:s for s in prediction['sources']} if prediction else {}
            field_checks={k:[] for k in ('motion','motion_with_direction_constraints','onset','offset','event_duration','start','end','numeric_start','numeric_end','numeric_onset','numeric_offset','relations','speech_words')}
            for s in scored['sources']:
                hyp=sources.get(s['prediction_source_id']);checks=s['constraints'];kind=s['kind']
                for name_,opset in [('motion',{'motion','angular_direction','distance_change'}),('speech_words',{'transcript'})]:
                    applicable=[r['pass'] for r in checks if r['constraint']['op'] in opset]
                    if applicable:field_checks[name_].append(all(applicable))
                moving=any(r['constraint']['op'] in ('angular_direction','distance_change') or (r['constraint']['op']=='motion' and r['constraint']['value']=='linear') for r in checks)
                if moving:
                    direction_checks=[r['pass'] for r in checks if r['constraint']['op'] in ('motion','angular_direction','distance_change') or (r['constraint']['op'] in ('sector','direct','compass') and r['constraint'].get('point') in ('start','end','both','path'))]
                    field_checks['motion_with_direction_constraints'].append(all(direction_checks))
                for name_,field,ops in [('onset','onset_sec',{'full_scene','starts_scene'}),('offset','offset_sec',{'full_scene','ends_scene'}),('event_duration','event_duration_sec',set())]:
                    applicable=[r['pass'] for r in checks if r['constraint'].get('field')==field or r['constraint']['op'] in ops]
                    if applicable:field_checks[name_].append(all(applicable))
                    numeric=[r['pass'] for r in checks if r['constraint']['op']=='numeric' and r['constraint'].get('field')==field]
                    if numeric and name_ in ('onset','offset'):field_checks['numeric_'+name_].append(all(numeric))
                for endpoint in ('start','end'):
                    values=[]
                    for r in checks:
                        c_=deepcopy(r['constraint']);op=c_['op']
                        if op=='numeric' and c_['field'].startswith(endpoint+'.'):values.append(r['pass'])
                        elif op in ('sector','direct','distance_range','compass') and c_.get('point') in ('both',endpoint):
                            c_['point']=endpoint;values.append(bool(kind and source_constraint_pass(hyp,c_,scene_duration=prediction['duration_sec'])))
                    if values:field_checks[endpoint].append(all(values))
                    numeric=[r['pass'] for r in checks if r['constraint']['op']=='numeric' and r['constraint'].get('field','').startswith(endpoint+'.')]
                    if numeric:field_checks['numeric_'+endpoint].append(all(numeric))
            field_checks['relations']=[r['pass'] for r in scored['relations']]
            fields_joint=scored['count_correct'] and all(s['kind'] and all(x['pass'] for x in s['constraints']) for s in scored['sources']) and all(x['pass'] for x in scored['relations']+scored['scene_constraints'])
            result={'candidate':c['name'],'id':sid,'family_id':ref.get('family_id',sid),'panels':ref.get('panels',['precise_retention' if precise else 'independent_N0_natural']),'subset':'precise' if precise else 'natural','scored':scored,'fields_joint':bool(fields_joint),'field_checks':field_checks}
            groups[c['name'],result['subset'],scored['requested_count']].append(result);scene_results.append(result)
            for panel in result['panels']:panel_groups[c['name'],panel,scored['requested_count']].append(result)
    report={'schema':'generation_ar_request_satisfaction_diagnostic_v2','test_used':False,'goal_complete':False,'judge':'Frozen calibrated Qwen3.5-27B; synonyms allowed, no independent human assessment.','source_matching':binding['scope'],'by_candidate':{},'by_panel':{},'semantic_pair_file_sha256':sha(root/'pairs.json'),'completion_reviews_sha256':sha(args.completion_reviews) if args.completion_reviews else None,'metric_notes':{'numeric_fields':'Only explicit request numbers enter numeric endpoint/onset/offset denominators; hidden witness numbers are never compared.','motion_with_direction_constraints':'Checks requested linear/radial/angular motion and qualitative start/end/path sectors together. Exact numerical endpoint compliance is reported separately.','confidence_intervals':'Wilson intervals use scenes within each source-count/panel. Do not combine overlapping panels as independent samples.'}}
    for c in binding['candidates']:
        report['by_candidate'][c['name']]={subset:{str(n):metrics(groups[c['name'],subset,n]) for n in range(1,5)} for subset in ('natural','precise')}
        panels=sorted({panel for candidate,panel,n in panel_groups if candidate==c['name']})
        report['by_panel'][c['name']]={panel:{str(n):metrics(panel_groups[c['name'],panel,n]) for n in range(1,5)} for panel in panels}
    # Report exact predefined rates, including pending reviews; no automatic
    # full-dataset promotion from four natural examples per source count.
    if args.experiment_contract:
        contract=json.loads(args.experiment_contract.read_text());assert contract['schema'] in ('generation_ar_N2_pretraining_experiment_v1','generation_ar_N3_pretraining_experiment_v1')
        assert contract['validation_pairs_sha256']==binding['validation_sha256']
        report['experiment_contract_sha256']=sha(args.experiment_contract)
        report['pilot_decision']=(n2_decision if contract['schema']=='generation_ar_N2_pretraining_experiment_v1' else n3_decision)(report['by_panel'])
    elif {'precise_control','natural_mix'}.issubset(report['by_candidate']):
        by=report['by_candidate'];gains=[];guardrails=[]
        for n in map(str,range(1,5)):
            gains.append(by['natural_mix']['natural'][n]['metrics']['joint']['rate']-by['precise_control']['natural'][n]['metrics']['joint']['rate'])
            for metric in ('count','core_recall','core_precision','all_requested_fields_joint'):
                delta=by['natural_mix']['precise'][n]['metrics'][metric]['rate']-by['precise_control']['precise'][n]['metrics'][metric]['rate']
                guardrails.append({'count':int(n),'metric':metric,'natural_minus_control':delta,'pass':delta>=-.02-1e-9})
        pending=sum(r['scored']['semantic_pending'] or r['scored']['completion_reasonable'] is None for r in scene_results)
        report['pilot_decision']={'status':'PENDING_REVIEWS' if pending else 'PASS_PILOT_ONLY' if sum(gains)/4>=.1-1e-9 and all(r['pass'] for r in guardrails) else 'FAIL_PILOT_RULE','natural_joint_macro_gain':sum(gains)/4,'precise_guardrails':guardrails,'full_data_promotion':False,'reason_for_no_full_promotion':'Only four independent natural requests per count and 32 unique request-first training scenarios; broader validation/data coverage required.'}
    tag='.'+args.report_tag if args.report_tag else ''
    write(root/f'SCENE_SCORES{tag}.json',{'rows':scene_results})
    write(root/f'REQUEST_SATISFACTION_REPORT{tag}.json',report)
    print(json.dumps({'report':str(root/f'REQUEST_SATISFACTION_REPORT{tag}.json'),'decision':report.get('pilot_decision')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','score'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validation',type=Path);parser.add_argument('--candidate',action='append');parser.add_argument('--completion-reviews',type=Path)
    parser.add_argument('--experiment-contract',type=Path);parser.add_argument('--report-tag')
    parser.add_argument('--prove-exact-satisfied',action='store_true',help='Skip only semantic edges irrelevant to a proven fully satisfied coupled assignment; unresolved synonyms are judged normally.')
    args=parser.parse_args();prepare(args) if args.mode=='prepare' else score(args)
