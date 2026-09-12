#!/usr/bin/env python3
"""Audited N0 request annotations and recorded offline teacher-plan repairs.

These finite annotations were authored from the English requests, before N1.
They are evaluation metadata, never additional inputs to the AR model. Teacher
plans are witnesses; their freely chosen numbers are not acceptance answers.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import SCHEMA, evaluate_natural_request

# key: requested spatial/motion constraints. Blank means unconstrained, not static.
# Relations use a op b; temporal/spatial direction is defined in the evaluator.
SPECS = {
 'train/1/0': ('dog:left', '', 'outdoor'),
 'train/1/1': ('rain:behind,event10', '', ''),
 'train/1/2': ('car:rl', '', ''),
 'train/1/3': ('piano:direct_front,static', '', ''),
 'train/1/4': ('woman:right', '', 'under12'),
 'train/1/5': ('bell:far,above,front', '', ''),
 'train/1/6': ('helicopter:behind,approaching', '', ''),
 'train/1/7': ('kettle:near', '', ''),
 'train/2/0': ('rain:;dog:left', 'dog starts_after rain', ''),
 'train/2/1': ('cello:front,static;bell:lr', 'cello overlaps bell', ''),
 'train/2/2': ('footsteps:right,starts_scene;window:left', 'window starts_after footsteps;window overlaps footsteps', ''),
 'train/2/3': ('man:direct_front;door:behind', 'door starts_after_end man', ''),
 'train/2/4': ('waves:;gull:', 'gull higher_than waves;waves nearer_than gull', ''),
 'train/2/5': ('fan:left,static;bell:right', 'fan overlaps bell', ''),
 'train/2/6': ('guitar:near;clock:', 'guitar nearer_than clock', 'dry'),
 'train/2/7': ('train:rl;wind_chime:behind,static', 'train during wind_chime', ''),
 'train/3/0': ('rain:behind;dog:left;car:rl', 'car starts_after rain;car starts_after dog', ''),
 'train/3/1': ('bass:front;shaker:right;flute:left', 'flute starts_after bass;flute starts_after shaker', ''),
 'train/3/2': ('woman:front,starts_scene;applause:left;bell:right', 'applause starts_after_end woman;bell starts_after_end woman;applause overlaps bell', ''),
 'train/3/3': ('fountain:near;birds:front;bicycle:lr', 'fountain nearer_than birds', ''),
 'train/3/4': ('motor:left;steam:right;clanks:behind', 'all_overlap', ''),
 'train/3/5': ('cat:right;footsteps:start_behind,toward_front,linear;keys:left', 'footsteps starts_after cat;keys starts_after cat;footsteps overlaps cat;keys overlaps cat', ''),
 'train/3/6': ('fire:direct_front,static;leaves:behind,static;owl:far,above,static', '', ''),
 'train/3/7': ('thunder:far,front;rain:left;gate:right', 'gate starts_after thunder', 'duration10'),
 'train/4/0': ('fountain:front;birds:above,left;wind_chimes:right;dog:behind', 'dog starts_after fountain;dog starts_after birds;dog starts_after wind_chimes', ''),
 'train/4/1': ('piano:left,static;violin:right,static;drums:front;shaker:behind', 'all_overlap', ''),
 'train/4/2': ('man:front;door:right;footsteps:lr;bell:behind', 'door starts_after_end man;footsteps starts_after_end man;bell starts_after_end man', ''),
 'train/4/3': ('rain:behind;dog:left;bell:right;car:rl', 'car starts_after bell', ''),
 'train/4/4': ('waves:front;gull:above;boat_motor:lr;pebbles:near,behind', '', 'outdoor'),
 'train/4/5': ('clock:left,full_scene;fan:right,full_scene;kettle:front,onset6;door:behind,onset6', '', 'duration12'),
 'train/4/6': ('guitar:near,left;cello:right;flute:;snaps:direct_behind', 'guitar nearer_than cello;flute higher_than guitar;all_overlap', ''),
 'train/4/7': ('wind:behind,starts_scene;cow:far,front,starts_scene;footsteps:rl;dog:right', 'footsteps starts_after wind;footsteps starts_after cow;dog starts_after footsteps;dog overlaps footsteps', ''),
 'validation/1/0': ('owl:above,left', '', ''),
 'validation/1/1': ('motorcycle:lr,path_front', '', ''),
 'validation/1/2': ('male_voice:behind', '', ''),
 'validation/1/3': ('clarinet:front,event8', '', ''),
 'validation/2/0': ('tap:near,right;dog:far,left', 'tap nearer_than dog;dog starts_after tap;dog overlaps tap', ''),
 'validation/2/1': ('tram:rl;busker:behind,static', 'tram overlaps busker', ''),
 'validation/2/2': ('woman:left;creak:right', 'creak starts_after_end woman', ''),
 'validation/2/3': ('bee:above,static;bag:below,static', '', ''),
 'validation/3/0': ('ac:behind;keys:right;suitcase:rl', 'keys starts_after ac;suitcase starts_after keys;suitcase during ac', ''),
 'validation/3/1': ('harp:front;violin:left;tambourine:right', 'harp nearer_than violin;tambourine starts_after harp;tambourine starts_after violin', ''),
 'validation/3/2': ('man:right;engine:far,front;chime:behind', 'man during engine;chime starts_after_end man', ''),
 'validation/3/3': ('stream:near,left;wind:above;woodpecker:right', 'stream nearer_than woodpecker', ''),
 'validation/4/0': ('fan:behind;radio:left;drill:right;footsteps:front,approaching', 'drill starts_after fan;drill starts_after radio;footsteps starts_after drill;footsteps overlaps drill', ''),
 'validation/4/1': ('woman:front;cello:left;piano:right;gong:behind,ends_scene', 'woman overlaps cello;woman overlaps piano;gong starts_after_end woman', ''),
 'validation/4/2': ('water:right;frog:near,left;crickets:behind;duck:front', 'frog nearer_than crickets;duck starts_after frog;duck overlaps frog', ''),
 'validation/4/3': ('skateboard:lr;dog:behind;fountain:front,full_scene;bicycle_bell:left', '', 'duration9'),
}

# This is a small authored pilot, not an English parser. Source content was
# reviewed against each request; speech words are extracted only when quoted.
CORE_OVERRIDES = {
 ('train/2/0','rain'): 'Steady background rain',
 ('train/4/4','waves'): 'Ocean waves',
 ('train/4/0','birds'): 'Bird sounds',
 ('train/4/7','cow'): 'A cow calling',
 ('validation/4/3','dog'): 'Dog sounds',
}


def source_constraints(codes, evidence):
    out = []
    def add(op, **kw): out.append({'op':op, **kw, 'evidence':evidence, 'origin':'relative'})
    for code in filter(None, codes.split(',')):
        if code in ('left','right','front','behind','above','below'): add('sector', point='both', value=code)
        elif code in ('static','linear'): add('motion', value=code)
        elif code in ('near','far'): add('distance_range', point='both', min=.1 if code=='near' else 3., max=2.5 if code=='near' else 50.)
        elif code in ('lr','rl'):
            add('motion',value='linear');add('sector',point='start',value='left' if code=='lr' else 'right');add('sector',point='end',value='right' if code=='lr' else 'left')
        elif code.startswith('direct_'): add('direct',point='both',value=code.removeprefix('direct_'))
        elif code.startswith('path_'): add('sector',point='path',value=code.removeprefix('path_'))
        elif code.startswith('start_'): add('sector',point='start',value=code.removeprefix('start_'))
        elif code.startswith('toward_'): add('angular_direction',value=code.removeprefix('toward_'))
        elif code == 'approaching': add('motion',value='linear');add('distance_change',value='approaching')
        elif code in ('full_scene','starts_scene','ends_scene'): add(code)
        elif code.startswith(('event','onset')):
            field='event_duration_sec' if code.startswith('event') else 'onset_sec'
            add('numeric',field=field,value=float(code[5:]));out[-1]['origin']='explicit' if field=='event_duration_sec' else 'entailed'
        else: raise ValueError(code)
    return out


def provenance(requirements):
    """Field-level dependence, without treating a relative bound as equality."""
    rows = []
    fields = ('onset_sec','offset_sec','start.azimuth_deg','start.elevation_deg','start.distance_m','end.azimuth_deg','end.elevation_deg','end.distance_m','motion','core_text','transcript','gain_db')
    for s in requirements['sources']:
        deps = {f:[] for f in fields}
        deps['core_text'].append({'evidence':s['evidence'],'origin':'explicit','op':'core_semantics'})
        for c in s['constraints']:
            op=c['op']; affected=[]
            if op=='numeric': affected=['onset_sec','offset_sec'] if c['field']=='event_duration_sec' else [c['field']]
            elif op=='transcript': affected=['transcript']
            elif op=='motion': affected=['motion']
            elif op=='full_scene': affected=['onset_sec','offset_sec']
            elif op=='starts_scene': affected=['onset_sec']
            elif op=='ends_scene': affected=['offset_sec']
            else:
                ends=('start','end') if c.get('point','both') in ('both','path') else (c['point'],)
                coords=('distance_m',) if op in ('distance_range','distance_change') else ('azimuth_deg','elevation_deg') if op in ('direct','angular_direction') else ('elevation_deg',) if c['value'] in ('above','below') else ('azimuth_deg',)
                affected=[f'{p}.{k}' for p in ends for k in coords]
            for f in affected: deps[f].append(c)
        for c in requirements['relations']:
            if s['key'] not in (c['a'],c['b']): continue
            op=c['op']; is_a=s['key']==c['a']
            if op in ('nearer_than','left_of','right_of','higher_than'):
                ends=('start','end') if c.get('point','both')=='both' else (c['point'],)
                coords=('distance_m',) if op=='nearer_than' else ('elevation_deg',) if op=='higher_than' else ('azimuth_deg','distance_m')
                affected=[f'{p}.{k}' for p in ends for k in coords]
            elif op in ('starts_after','starts_with'): affected=['onset_sec']
            elif op in ('ends_before','ends_with'): affected=['offset_sec']
            elif op=='starts_after_end': affected=['onset_sec' if is_a else 'offset_sec']
            else: affected=['onset_sec','offset_sec']
            for f in affected: deps[f].append(c)
        for f, constraints in deps.items():
            if f=='transcript' and s['kind']!='speech': continue
            rows.append({'source':s['key'],'field':f,'origin':'canonical_schema_policy' if f=='gain_db' else 'free_completion' if not constraints else 'request_constrained','constraints':constraints})
    for f in ('duration_sec','room'):
        cs=[c for c in requirements['scene'] if (c['op']=='room')==(f=='room')]
        # Event duration and end-of-scene requirements can indirectly bound duration.
        indirect=[c for s in requirements['sources'] for c in s['constraints'] if f=='duration_sec' and (c['op'] in ('full_scene','ends_scene') or c.get('field')=='event_duration_sec')]
        rows.append({'source':None,'field':f,'origin':'request_constrained' if cs or indirect else 'free_completion','constraints':cs+indirect})
    return rows


def curate(root, output):
    spec=importlib.util.spec_from_file_location('teacher_conversion',Path(__file__).with_name('plan_generation_ar_natural_seeds.py'));teacher=importlib.util.module_from_spec(spec);spec.loader.exec_module(teacher)
    import stable_audio_tools.data.sceneplan_generation_ar_natural_constraints as evaluator
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    codec=ModelScenePlanCodecV4('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    seeds={s['id']:s for s in json.loads((root/'request_first_seeds.json').read_text())['seeds']}
    raw=json.loads((root/'TEACHER_QA_ROWS.json').read_text())['rows'];pairs=[];receipts=[]
    assert len(raw)==len(SPECS)==48
    for record in raw:
        seed=deepcopy(seeds[record['id']]);name=record['id'].split('v2/')[-1];value=deepcopy(record['teacher']);original=deepcopy(value);repairs=[]
        if name in ('train/4/2','validation/3/2'):
            seed['request']=seed['request'].replace('behind him','behind the listener')
            repairs.append({'type':'authored_request_clarification','before':record['request'],'after':seed['request'],'reason':'Listener-relative task scope: remove ambiguous speaker-body reference before N1 freeze; preserve family and original raw baseline.'})
        request=seed['request'];declared,relations,scene=SPECS[name];bykey={s['key']:s for s in value['sources']}
        assert set(bykey)=={entry.split(':')[0] for entry in declared.split(';')}
        for entry in declared.split(';'):
            key,codes=entry.split(':');s=bykey[key];s['evidence']=request;s['constraints']=source_constraints(codes,request)
            if (name,key) in CORE_OVERRIDES: s['core']=CORE_OVERRIDES[name,key]
            if s['kind']=='speech':
                phrases=re.findall(r'"([^"]+)"',request);assert len(phrases)==1 and phrases[0]==s['transcript']
                s['constraints'].append({'op':'transcript','value':phrases[0],'evidence':phrases[0],'origin':'explicit'})
        value['scene_constraints']=[]
        if scene in ('dry','outdoor'): value['scene_constraints'].append({'op':'room','value':scene,'evidence':request,'origin':'explicit'})
        elif scene.startswith('duration'): value['scene_constraints'].append({'op':'numeric','field':'duration_sec','value':float(scene[8:]),'evidence':request,'origin':'explicit'})
        elif scene=='under12': value['scene_constraints'].append({'op':'duration_range','min':0.,'max':12.,'max_inclusive':False,'evidence':request,'origin':'explicit'})
        value['relations']=[]
        for clause in filter(None,relations.split(';')):
            expanded=[(a,'overlaps',b) for a,b in itertools.combinations(bykey,2)] if clause=='all_overlap' else [clause.split()]
            for a,op,b in expanded: value['relations'].append({'op':op,'a':a,'b':b,'point':'both','evidence':request,'origin':'relative'})
        flip={'train/2/5':['fan','bell'],'train/4/1':['piano','violin'],'train/4/6':['guitar','cello','flute'],'validation/2/2':['woman','creak']}.get(name,[])
        for key in flip: bykey[key]['start'][0]*=-1
        if name=='validation/4/2': bykey['crickets']['start'][0]=180.
        if name=='validation/4/1': bykey['gong']['activity']=[11.,12.]
        # Preserve short speech margins already chosen by the teacher; never turn
        # their 0.1 s planning margin into a minimum requested delay.
        changed_plans=[]
        for before,after in zip(original['sources'],value['sources']):
            changed={k:{'before':before.get(k),'after':after.get(k)} for k in ('start','end','activity','motion','description','speaker_description','transcript','kind') if before.get(k)!=after.get(k)}
            if changed: changed_plans.append({'key':after['key'],'fields':changed})
        if changed_plans: repairs.append({'type':'teacher_witness_repair','changes':changed_plans,'reason':'Respect independently annotated direction or the explicit final-event request.'})
        pair=teacher.convert(seed,value,codec,evaluator)
        pair['completion_provenance']=provenance(pair['requirements'])
        pair['semantic_pairing_review']='CODEX_AUDITED_WITH_RECORDED_ANNOTATION_REPAIRS'
        pair['english_review']='CODEX_REVIEWED_ENGLISH_REQUEST_AND_ALL_TEXT_FIELDS'
        pair['review_scope']='All 48 N0 pairs, not independent human review; retained teacher core wording was checked against raw requests.'
        labels={(r['key'],s['source_id']):pair['source_bindings'][r['key']]==s['source_id'] for r in pair['requirements']['sources'] for s in pair['target_sceneplan']['sources']}
        pair['deterministic_quality']=evaluate_natural_request(request,pair['requirements'],pair['target_sceneplan'],labels,completion_reasonable=True)
        assert pair['deterministic_quality']['acceptance_joint']
        receipts.append({'id':seed['id'],'original_automatic_status':record['status'],'original_requirements':{'sources':[{k:s[k] for k in ('key','core','evidence','constraints')} for s in original['sources']],'scene':original['scene_constraints'],'relations':original['relations']},'reviewed_requirements':pair['requirements'],'repairs':repairs,'annotation_change_reason':'Independent request-satisfaction annotation; remove unrequested exact coordinates, hidden defaults, fabricated evidence, and planning margins. Add omitted requested relations. Full request is the audit evidence span; source key/core disambiguates its subject.'})
        pairs.append(pair)
    assert len({p['request'] for p in pairs})==48
    assert not ({p['family_id'] for p in pairs if p['split']=='train'} & {p['family_id'] for p in pairs if p['split']=='validation'})
    output.mkdir(parents=True,exist_ok=True)
    artifacts={'request_first_pairs.json':{'pairs':pairs,'test_used':False},'REVIEW.json':{'status':'PASS_RETAINED_PAIRS_AFTER_RECORDED_REPAIRS','reviewer':'Codex, not an independent human','reviewed_rows':48,'train':32,'validation':16,'known_retained_pair_failures':0,'original_automatic_pass_rate_is_not_curated_quality':True,'receipts':receipts}}
    for filename,payload in artifacts.items():
        path=output/filename;text=json.dumps(payload,ensure_ascii=False,indent=2)+'\n'
        if path.exists() and path.read_text()!=text: raise ValueError('refusing changed frozen artifact '+str(path))
        path.write_text(text)
    print(json.dumps({'status':'PASS','rows':48,'plan_repair_rows':sum(any(x['type']=='teacher_witness_repair' for x in r['repairs']) for r in receipts),'request_clarifications':sum(any(x['type']=='authored_request_clarification' for x in r['repairs']) for r in receipts),'output':str(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();curate(args.root,args.output)
