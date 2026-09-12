#!/usr/bin/env python3
"""Request-first English compositions, five views, then a legal witness plan.

The finite event inventory and grammar are authored data, not a free-language
understanding system. They are used offline only. No compiler, supplied count,
or intent annotation is an input to Generation AR at evaluation/inference.
"""
from __future__ import annotations
import os
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import SCHEMA,evaluate_natural_request

# Audible entity/action, without hidden spatial or temporal facts.
SOUNDS = [
 ('dog','a dog barking','A dog barking.','the dog'),
 ('cat','a cat meowing','A cat meowing.','the cat'),
 ('owl','an owl hooting','An owl hooting.','the owl'),
 ('cow','a cow mooing','A cow mooing.','the cow'),
 ('frog','a frog croaking','A frog croaking.','the frog'),
 ('duck','a duck calling','A duck calling.','the duck'),
 ('gull','a gull calling','A gull calling.','the gull'),
 ('birds','birds singing','Birds singing.','the birds'),
 ('woodpecker','a woodpecker pecking','A woodpecker pecking.','the woodpecker'),
 ('crickets','crickets chirping','Crickets chirping.','the crickets'),
 ('bee','a bee buzzing','A bee buzzing.','the bee'),
 ('rain','rain falling steadily','Steady falling rain.','the rain'),
 ('thunder','thunder rumbling','Thunder rumbling.','the thunder'),
 ('wind','wind rustling through leaves','Wind rustling through leaves.','the wind'),
 ('waves','ocean waves breaking','Ocean waves breaking.','the waves'),
 ('stream','a stream flowing','A stream flowing.','the stream'),
 ('water','water flowing','Flowing water.','the water'),
 ('fountain','a fountain trickling','A fountain trickling.','the fountain'),
 ('fire','a fire crackling','A fire crackling.','the fire'),
 ('tap','a tap dripping','A tap dripping.','the tap'),
 ('kettle','a kettle whistling','A kettle whistling.','the kettle'),
 ('fan','an electric fan humming','An electric fan humming.','the fan'),
 ('ac','an air conditioner humming','An air conditioner humming.','the air conditioner'),
 ('clock','a clock ticking','A clock ticking.','the clock'),
 ('keys','keys jingling','Keys jingling.','the keys'),
 ('paperbag','a paper bag rustling','A paper bag rustling.','the paper bag'),
 ('window','a window rattling','A window rattling.','the window'),
 ('door','a door slamming','A door slamming.','the door'),
 ('woodstep','a wooden step creaking','A wooden step creaking.','the wooden step'),
 ('gate','a gate creaking','A gate creaking.','the gate'),
 ('footsteps','footsteps on gravel','Footsteps crunching on gravel.','the footsteps'),
 ('suitcase','a suitcase rolling','A suitcase rolling.','the suitcase'),
 ('pebbles','pebbles crunching','Pebbles crunching.','the pebbles'),
 ('drill','an electric drill running','An electric drill running.','the drill'),
 ('engine','an engine rumbling','An engine rumbling.','the engine'),
 ('car','a car engine running','A car engine running.','the car'),
 ('motorcycle','a motorcycle engine running','A motorcycle engine running.','the motorcycle'),
 ('train','a train clattering','A train clattering on tracks.','the train'),
 ('tram','a tram rattling','A tram rattling.','the tram'),
 ('helicopter','a helicopter rotor beating','A helicopter rotor beating.','the helicopter'),
 ('boat','a boat motor running','A boat motor running.','the boat'),
 ('skateboard','a skateboard rolling','A skateboard rolling.','the skateboard'),
 ('bikebell','a bicycle bell ringing','A bicycle bell ringing.','the bicycle bell'),
 ('churchbell','a church bell ringing','A church bell ringing.','the church bell'),
 ('chime','a chime ringing','A chime ringing.','the chime'),
 ('gong','a gong sounding','A gong sounding.','the gong'),
 ('applause','hands clapping','Hands clapping.','the clapping'),
 ('snaps','fingers snapping','Fingers snapping.','the finger snaps'),
 ('steam','steam hissing','Steam hissing.','the steam'),
 ('metal','metal clanking','Metal clanking.','the metal clanks'),
]
MUSIC = [(name,f'a solo {name} melody',f'A solo {name} melody.',f'the {name}') for name in
         ('piano','violin','cello','guitar','flute','clarinet','harp','accordion','trumpet','saxophone','trombone','harmonica','banjo','mandolin','xylophone','marimba','bassoon','oboe','tuba','synthesizer')]
MUSIC += [('radio','a radio playing instrumental music','Instrumental music playing on a radio.','the radio'),
          ('tambourine','a rhythmic tambourine part','A rhythmic tambourine part.','the tambourine'),
          ('shaker','a rhythmic shaker part','A rhythmic shaker part.','the shaker'),
          ('drums','a drum rhythm','A drum rhythm.','the drums')]

# Conservatively exclude the independent N0 validation entity combinations,
# even when generated numbers/positions would differ. No paraphrase leakage.
N0_VALIDATION_ENTITY_SETS = [
 ('owl',),('motorcycle',),('male_voice',),('clarinet',),('tap','dog'),('tram','accordion'),
 ('female_voice','woodstep'),('bee','paperbag'),('ac','keys','suitcase'),('harp','violin','tambourine'),
 ('male_voice','engine','chime'),('stream','wind','woodpecker'),('fan','radio','drill','footsteps'),
 ('female_voice','cello','piano','gong'),('water','frog','crickets','duck'),('skateboard','dog','fountain','bikebell')]
FORBIDDEN={tuple(sorted(x)) for x in N0_VALIDATION_ENTITY_SETS}
LOCATIONS=('left','right','front','behind','above','below','lr','rl','approach','recede','free','exact')


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def family_descriptor(intent):
    """Group permutations/paraphrases before splitting; ignore witness defaults."""
    req=requirements(intent,'')
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ('evidence','origin')}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    req=clean(req)
    for source in req['sources']:source['constraints'].sort(key=fingerprint)
    req['sources'].sort(key=lambda source:source['key'])
    for relation in req['relations']:
        if relation['op']=='overlaps':relation['a'],relation['b']=sorted((relation['a'],relation['b']))
    req['relations'].sort(key=fingerprint);req['scene'].sort(key=fingerprint)
    return req


def event_bank():
    return [dict(key=k,phrase=p,description=d,alias=a,kind=kind) for kind,values in [('sound',SOUNDS),('music',MUSIC)] for k,p,d,a in values]


def sample_intent(rng,n):
    events=deepcopy(rng.sample(event_bank(),n))
    if rng.random()<.25:
        gender=rng.choice(('man','woman'));mood=rng.choice(('calm','gentle','cheerful'));words=rng.choice(('The train arrives soon.','The doors are open.','Please come this way.','We can begin now.','The room is ready.','I will return soon.','Please wait a moment.','The lights are on.'))
        events[-1]={'key':'male_voice' if gender=='man' else 'female_voice','phrase':f'a {mood} {gender} saying "{words}"','description':f'A {mood} adult {"male" if gender=="man" else "female"} voice.','alias':f'the {gender}','kind':'speech','transcript':words}
    if tuple(sorted(e['key'] for e in events)) in FORBIDDEN:return None
    keys={e['key'] for e in events}
    # Avoid a generic label being read as the same event as a specific label.
    # Separate-source language is possible, but this pilot uses unambiguous sets.
    if 'water' in keys and keys & {'stream','fountain','tap','waves','rain'}:return None
    if 'engine' in keys and keys & {'car','motorcycle','boat','helicopter'}:return None
    if 'birds' in keys and keys & {'owl','duck','gull','woodpecker'}:return None
    temporal=rng.choice(('independent','overlap','sequential','later_overlap')) if n>1 else 'independent'
    duration=rng.choice((8.,10.,12.,14.))
    if temporal=='sequential' and n>=3:duration=14.
    if n==1 and events[0]['kind']=='speech':duration=6.
    explicit_duration=rng.random()<.3;room=rng.choice(('dry','moderate','outdoor')) if rng.random()<.25 else None
    for e in events:
        e['location']=rng.choice(LOCATIONS);e['stationary']=e['location'] not in ('lr','rl','approach','recede') and rng.random()<.35
        if e['location']=='exact':e['requested_position']=[rng.randrange(-170,171,10),rng.choice((-20,0,20)),rng.choice((1.,2.,3.,5.,8.))]
        if temporal=='independent' and rng.random()<.25:
            start=round(rng.uniform(0.,2.),1);span=3. if e['kind']=='speech' else round(rng.uniform(2.,4.),1)
            e['requested_interval']=[start,start+span]
    nearer=n>1 and rng.random()<.2
    if nearer:
        # Keep this relational distance case orthogonal to radial movement.
        for e in events[:2]:e['location']=rng.choice(('left','right','front','behind'));e.pop('requested_position',None)
    # duration_candidate is a planning choice unless explicitly mentioned.
    return {'events':events,'temporal':temporal,'explicit_duration':duration if explicit_duration else None,'planning_duration_candidate':duration,'room':room,'nearer_first':nearer}


def render(intent,view):
    events=intent['events'];order=list(range(len(events)))
    if view in (2,4):order.reverse()
    clauses=[];spans={};offset=0
    header=['Please create a sound scene.','Here is the audio scene I would like.','Could you make this scene for me?','I am looking for the following sound scene.','Let me hear this scene.'][view]
    if intent['explicit_duration'] is not None:header+=f" Make the clip {intent['explicit_duration']:g} seconds long."
    if intent['room'] is not None:header+=' '+{'dry':'Use a dry room.','moderate':'Use a moderately reverberant room.','outdoor':'Set it outdoors.'}[intent['room']]
    text=header+' '
    for i in order:
        e=events[i];loc=e['location'];p=e['phrase']
        locations={'left':['on my left','to the left of the listener','off to my left','on the left-hand side','somewhere to my left'],
                   'right':['on my right','to the right of the listener','off to my right','on the right-hand side','somewhere to my right'],
                   'front':['in front of me','ahead of the listener','somewhere ahead','up front','out in front'],
                   'behind':['behind me','behind the listener','somewhere behind','to my rear','at the back'],
                   'above':['above me','above the listener','somewhere overhead','up above','at a higher elevation than me'],
                   'below':['below me','below the listener','somewhere below','down below','at a lower elevation than me'],
                   'free':['','','','','']}
        if loc in locations:
            lead=['Include','I would like','Let me hear','Add','Please include'][view]
            clause=f"{lead} {p} {locations[loc][view]}.".replace(' .','.')
        elif loc in ('lr','rl'):
            start,end=('left','right') if loc=='lr' else ('right','left')
            clause=[f'Make the sound of {p} move across the front from {start} to {end}.',f'Let the sound of {p} travel across the front, from {start} to {end}.',f'I want the sound of {p} to cross in front of me from my {start} to my {end}.',f'Include the sound of {p}, starting on the {start} and moving across the front to the {end}.',f'Have the sound of {p} pass in front of me from {start} to {end}.'][view]
        elif loc=='approach':clause=[f'Have the sound of {p} approach from ahead.',f'Let the sound of {p} get closer while staying in front of me.',f'I want the sound of {p} to come toward me from the front.',f'Include the sound of {p}, approaching the listener from ahead.',f'Make the sound of {p} move closer from in front of me.'][view]
        elif loc=='recede':clause=[f'Have the sound of {p} move farther away behind me.',f'Let the sound of {p} recede while remaining behind the listener.',f'I want the sound of {p} to move away to my rear.',f'Include the sound of {p}, getting farther away behind me.',f'Make the sound of {p} recede behind the listener.'][view]
        else:
            az,el,d=e['requested_position'];unit='meter' if d==1 else 'meters';clause=f'Include {p} at azimuth {az:g} degrees, elevation {el:g} degrees and distance {d:g} {unit}.'
        if e['stationary']:clause+=' Keep this sound in a fixed position.'
        if 'requested_interval' in e:
            a,b=e['requested_interval'];clause+=f' Let it be audible from {a:g} to {b:g} seconds.'
        spans[e['key']]=[len(text),len(text)+len(clause)];text+=clause+' ';clauses.append(clause)
    mode=intent['temporal'];relation_text=[]
    if mode=='overlap':relation_text.append(['Let all these sounds overlap in time.','Make sure I hear them all together for a while.','There should be a period when they are all audible.','Give all the sounds some shared playing time.','Have all these sounds play together for a while.'][view])
    elif mode=='sequential':
        for a,b in zip(events,events[1:]):relation_text.append([f"Start the sound of {b['alias']} after the sound of {a['alias']} stops.",f"Only once the sound of {a['alias']} has finished should the sound of {b['alias']} begin.",f"Let the sound of {a['alias']} finish before the sound of {b['alias']} comes in.",f"Wait for the sound of {a['alias']} to end, then start the sound of {b['alias']}.",f"Have the sound of {b['alias']} enter after the end of the sound of {a['alias']}."][view])
    elif mode=='later_overlap':
        a,b=events[:2];relation_text.append([f"Bring in the sound of {b['alias']} after the sound of {a['alias']} has started, while both continue.",f"Let the sound of {b['alias']} join the sound of {a['alias']} a little later, with some overlap.",f"Start the sound of {a['alias']} earlier than the sound of {b['alias']}, and let them overlap.",f"Have the sound of {b['alias']} begin later than the sound of {a['alias']} while both remain audible for a while.",f"Let the sound of {a['alias']} start first; the sound of {b['alias']} should enter later before the first sound ends."][view])
    if intent['nearer_first']:
        a,b=events[:2];relation_text.append(f"Keep {a['alias']} closer to me than {b['alias']}.")
    text+=' '.join(relation_text)+(' ' if relation_text else '')+'Use only these sounds.'
    return {'request':text,'mention_order':order,'source_character_spans':spans}


def requirements(intent,request):
    req={'schema':SCHEMA,'output_order':None,'sources':[],'relations':[],'scene':[]}
    def c(op,**kw):return {'op':op,**kw,'evidence':request,'origin':'explicit' if op in ('numeric','transcript','room') else 'relative'}
    if intent['explicit_duration'] is not None:req['scene'].append(c('numeric',field='duration_sec',value=intent['explicit_duration']))
    if intent['room'] is not None:req['scene'].append(c('room',value=intent['room']))
    for e in intent['events']:
        cs=[];loc=e['location']
        if loc in ('left','right','front','behind','above','below'):cs.append(c('sector',point='both',value=loc))
        elif loc in ('lr','rl'):
            cs.extend([c('motion',value='linear'),c('sector',point='start',value='left' if loc=='lr' else 'right'),c('sector',point='end',value='right' if loc=='lr' else 'left')])
            cs.append(c('sector',point='path',value='front'))
        elif loc in ('approach','recede'):cs.extend([c('motion',value='linear'),c('sector',point='both',value='front' if loc=='approach' else 'behind'),c('distance_change',value='approaching' if loc=='approach' else 'receding')])
        elif loc=='exact':
            for endpoint in ('start','end'):
                for field,value in zip(('azimuth_deg','elevation_deg','distance_m'),e['requested_position']):cs.append(c('numeric',field=endpoint+'.'+field,value=value))
        if e['stationary']:cs.append(c('motion',value='static'))
        if 'requested_interval' in e:
            for field,value in zip(('onset_sec','offset_sec'),e['requested_interval']):cs.append(c('numeric',field=field,value=value))
        if e['kind']=='speech':cs.append(c('transcript',value=e['transcript']))
        req['sources'].append({'key':e['key'],'kind':e['kind'],'core':e['description'],'evidence':request,'constraints':cs})
    events=intent['events'];mode=intent['temporal']
    if mode=='overlap':req['relations'].extend(c('overlaps',a=a['key'],b=b['key']) for a,b in itertools.combinations(events,2))
    elif mode=='sequential':req['relations'].extend(c('starts_after_end',a=b['key'],b=a['key']) for a,b in zip(events,events[1:]))
    elif mode=='later_overlap':req['relations'].extend([c('starts_after',a=events[1]['key'],b=events[0]['key']),c('overlaps',a=events[1]['key'],b=events[0]['key'])])
    if intent['nearer_first']:req['relations'].append(c('nearer_than',a=events[0]['key'],b=events[1]['key'],point='both'))
    return req


def plan_after_requests(intent,rng,sid):
    duration=intent['explicit_duration'] or intent['planning_duration_candidate'];events=intent['events'];n=len(events);sources=[]
    def pos(a,e,d):return {'azimuth_deg':a,'elevation_deg':e,'distance_m':d}
    for i,e in enumerate(events):
        loc=e['location'];az=rng.uniform(-175,175);el=rng.uniform(-10,10);distance=rng.uniform(1.,6.)
        if loc=='left':az=rng.uniform(35,145)
        elif loc=='right':az=rng.uniform(-145,-35)
        elif loc=='front':az=rng.uniform(-55,55)
        elif loc=='behind':az=rng.choice((-1,1))*rng.uniform(130,175)
        elif loc=='above':el=rng.uniform(25,55)
        elif loc=='below':el=rng.uniform(-55,-25)
        if intent['nearer_first'] and i<2:distance=rng.uniform(1.,2.) if i==0 else rng.uniform(5.,8.)
        trajectory={'type':'static','position':pos(az,el,distance)}
        if loc=='exact':trajectory['position']=pos(*e['requested_position'])
        elif loc in ('lr','rl'):
            a,b=rng.uniform(45,80),-rng.uniform(45,80)
            if loc=='rl':a,b=b,a
            trajectory={'type':'linear','start':pos(a,0,distance),'end':pos(b,0,distance)}
        elif loc in ('approach','recede'):
            az=rng.uniform(-20,20) if loc=='approach' else rng.choice((-1,1))*rng.uniform(145,175)
            a,b=(rng.uniform(7,10),rng.uniform(1,2)) if loc=='approach' else (rng.uniform(1,2),rng.uniform(7,10))
            trajectory={'type':'linear','start':pos(az,0,a),'end':pos(az,0,b)}
        mode=intent['temporal']
        if mode=='sequential':width=(duration-.2*(n-1))/n;on=i*(width+.2);off=on+width
        elif mode=='overlap':on=rng.uniform(0,2);off=rng.uniform(5,duration)
        elif mode=='later_overlap':on=0. if i==0 else 2. if i==1 else rng.uniform(0,2);off=duration
        else:on=rng.uniform(0,min(2,duration-3));off=rng.uniform(on+2.5,duration)
        if e['kind']=='speech' and mode not in ('sequential',):
            if mode=='overlap':on=2.;off=5.
            elif mode=='later_overlap' and i in (0,1):on=0. if i==0 else 2.;off=3. if i==0 else 5.
            else:off=min(duration,on+3.)
        elif e['kind']=='speech':off=min(off,on+3.)
        if 'requested_interval' in e:on,off=e['requested_interval']
        source={'source_id':f'source_{i}','kind':e['kind'],'gain_db':0.,'activity':{'onset_sec':on,'offset_sec':off},'trajectory':trajectory}
        if e['kind']=='speech':source.update(speaker_description=e['description'],transcript=e['transcript'])
        else:source['description']=e['description']
        sources.append(source)
    return {'sample_id':sid,'duration_sec':duration,'room':{'type':intent['room'] or 'outdoor'},'sources':sources}


def make_pairs(seed,count_per_group,codec,split=None):
    rng=random.Random(seed);records=[];families=set();attempts=0
    for n in range(1,5):
        retained=0
        while retained<count_per_group:
            attempts+=1;intent=sample_intent(rng,n)
            if intent is None:continue
            public={k:v for k,v in intent.items() if k!='planning_duration_candidate'};family=fingerprint(family_descriptor(intent))
            assigned_split='validation' if int(family[:8],16)%5==0 else 'train'
            if split is not None and assigned_split!=split:continue
            if family in families:continue
            # Freeze every raw view before the planner is called or any plan exists.
            views=[render(intent,i) for i in range(5)];request_hashes=[fingerprint(v['request']) for v in views]
            sid='compositional_request_first_v1/'+family[:20]
            witness=codec.project_plan(plan_after_requests(intent,random.Random(int(family[:16],16)^seed),sid))
            targets=[];bindings=[];reqs=[]
            for view in views:
                plan=deepcopy(witness);plan['sources']=[deepcopy(witness['sources'][i]) for i in view['mention_order']];bound={}
                for index,(source,old_index) in enumerate(zip(plan['sources'],view['mention_order'])):source['source_id']=f'source_{index}';bound[intent['events'][old_index]['key']]=source['source_id']
                req=requirements(intent,view['request']);labels={(r['key'],s['source_id']):bound[r['key']]==s['source_id'] for r in req['sources'] for s in plan['sources']}
                result=evaluate_natural_request(view['request'],req,plan,labels,completion_reasonable=True)
                if not result['acceptance_joint']:raise ValueError(json.dumps({'intent':intent,'request':view['request'],'plan':plan,'checks':result}))
                tokens=codec.encode(plan)['input_ids'].tolist();assert codec.decode(tokens,sample_id=sid)==plan
                targets.append({'plan':plan,'tokens':tokens});bindings.append(bound);reqs.append(req)
            assert len({len(t['tokens']) for t in targets})==1
            families.add(family);retained+=1;records.append({'id':sid,'family_id':family,'split':assigned_split,'route':'request_to_plan','source_count':n,'intent':public,'views':views,'requirements':reqs,'targets':targets,'source_bindings_by_view':bindings,'provenance':{'request_first':True,'all_request_hashes_before_planning':request_hashes,'planner':'offline constrained witness sampler, never AR inference','free_fields':'Any target coordinate/time/room not fixed by the stored request constraints; numeric witness values are not hidden evaluation answers.','output_order':'Each complete source follows first mention in that raw view; evaluation permits other valid source orders.','split_rule':'Canonical request-constraint family hash; 20% validation for pilot. No test data generated or consumed.'}})
    return records,attempts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--per-count',type=int,default=8);parser.add_argument('--seed',type=int,default=252);args=parser.parse_args()
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    codec=ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4");records,attempts=make_pairs(args.seed,args.per_count,codec)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'pairs.json').write_text(json.dumps({'records':records,'test_used':False,'quality':'DETERMINISTIC_CHECKS_PASS_MANUAL_ENGLISH_SEMANTIC_REVIEW_PENDING'},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'records':len(records),'pairs':len(records)*5,'attempts':attempts,'output':str(args.output)}))
