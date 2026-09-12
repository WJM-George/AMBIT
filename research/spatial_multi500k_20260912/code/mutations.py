"""Explicit single and compound spatial edits with source identity preserved."""
from common import *
import copy
import random
from catalog import compatible_choices, angle
from scripts.t2a.data import build_sceneplan_transfusion_editing_pairs as native
from stable_audio_tools.data.sceneplan_transfusion_editing import (
    _target_recipe, render_members, sha256_json, validate_model_sceneplan,
    validate_render_recipe_binding)

DIGEST_FIELDS = ['pair_id','split','operation','instruction_sha256','old_sceneplan_sha256',
    'new_sceneplan_sha256','source_render_recipe_sha256','target_render_recipe_sha256',
    'source_render_result_sha256','source_members_sha256','target_members_sha256',
    'source_latent_ref','source_latent_tensor_sha256','target_latent_ref','pair_gain_policy']


class Ineligible(ValueError):
    pass


def position(source):
    t=source['trajectory']
    return t['position'] if t['type']=='static' else t['start']


def descriptor(s):
    detail=s.get('speaker_description') if s['kind']=='speech' else s.get('description')
    detail=detail or s.get('sound_description') or s.get('music_description') or s['kind']
    p=position(s);a=s['activity']
    return (f'the {s["kind"]} source described as "{detail}", initially at azimuth '
        f'{p["azimuth_deg"]:+.4f} degrees, active from {a["onset_sec"]:.6f} to {a["offset_sec"]:.6f} seconds')


def spatial_target(s, rng, direction=None, anchor=True):
    origin=position(s)
    choices=[direction] if direction else list(COMPASS)
    rng.shuffle(choices)
    for key in choices:
        for _ in range(32):
            az=COMPASS[key]+(0 if anchor else rng.uniform(-12,12))
            az=round((az+180)%360-180,6)
            if angle(origin['azimuth_deg'],az)>=30:
                return {**copy.deepcopy(origin),'azimuth_deg':az},key
    raise Ineligible('No target in the requested direction differs by at least 30 degrees')


def mutate(base, spec, variant=0):
    old=unpack(base['old_sceneplan_zlib']);new=copy.deepcopy(old)
    task=spec['task'];ordinal=spec['ordinal'];split=spec['split']
    pair_id='speditspmulti500kv1_'+split+'_'+digest([split,ordinal])[:24]
    new['sample_id']=pair_id+'_target'
    src={s['source_id']:s for s in old['sources']};dest={s['source_id']:s for s in new['sources']}
    choices=compatible_choices(old);rng=random.Random(seed(pair_id,base['source_sample_id'],variant))
    anchor=bool(spec['anchor']);selected=[];actions=[]
    def choose(ids,n):
        if len(ids)<n:raise Ineligible('Source trajectory prerequisites not met')
        return rng.sample(ids,n)
    if task.startswith('diagonal_'):
        candidates=choices['static'][:];rng.shuffle(candidates)
        for sid in candidates:
            try:target,direction=spatial_target(src[sid],rng,spec['direction'],anchor)
            except Ineligible:continue
            selected=[sid];break
        if not selected:raise Ineligible('No static source can satisfy the requested diagonal')
    elif task in ['dual_relocation','dual_start_motion']:
        selected=choose(choices['static'],2)
    elif task=='dual_stop_motion':selected=choose(choices['linear'],2)
    elif task=='mixed_motion_toggle':selected=choose(choices['static'],1)+choose(choices['linear'],1)
    elif task=='position_swap':
        if not choices['swaps']:raise Ineligible('No separated static pair')
        selected=rng.choice(choices['swaps'])
    elif task=='three_position_cycle':
        if not choices['cycles']:raise Ineligible('No separated static triple')
        selected=list(rng.choice(choices['cycles']))
        if rng.randrange(2):selected.reverse()
    else:raise ValueError(task)
    for i,sid in enumerate(selected):
        s=src[sid];before=copy.deepcopy(s['trajectory']);direction=None
        if task in ['position_swap','three_position_cycle']:
            other=selected[(i+1)%len(selected)]
            p=copy.deepcopy(src[other]['trajectory']['position'])
            after=dict(type='static',position=p);operation='stationary_spatial_relocation'
            text=f'Move {descriptor(s)} to the original position of {descriptor(src[other])}.'
        else:
            start_motion=task in ['diagonal_start_motion','dual_start_motion'] or (task=='mixed_motion_toggle' and before['type']=='static')
            stop_motion=task=='dual_stop_motion' or (task=='mixed_motion_toggle' and before['type']=='linear')
            if task.startswith('diagonal_'):
                p=copy.deepcopy(target);direction=spec['direction']
            else:p,direction=spatial_target(s,rng,None,anchor)
            if start_motion:
                after=dict(type='linear',start=copy.deepcopy(before['position']),end=p)
                operation='static_to_linear'
                text=(f'Make {descriptor(s)} move linearly during its active interval from its original '
                      f'position to {direction.replace("_"," ")} at azimuth {p["azimuth_deg"]:+.6f} degrees.')
            else:
                after=dict(type='static',position=p)
                operation='linear_to_static' if stop_motion else 'stationary_spatial_relocation'
                verb='Stop the motion of' if stop_motion else 'Move'
                text=(f'{verb} {descriptor(s)} and keep it stationary at {direction.replace("_"," ")}, '
                      f'azimuth {p["azimuth_deg"]:+.6f} degrees, throughout its active interval.')
            text+=' Keep its distance and elevation unchanged.'
        dest[sid]['trajectory']=after
        actions.append(dict(source_id=sid,operation=operation,before=before,after=after,
            target_direction=direction,instruction=text))
    instruction=' '.join(a['instruction'] for a in actions)+' Preserve all sound content, source gains, timing, and every other source. Positive azimuth is left; zero is front.'
    recipe=unpack(base['source_render_recipe_zlib'])
    target_recipe=_target_recipe(recipe,new,recipe['sources'])
    members=list(render_members(new,target_recipe))
    row={k:base[k] for k in native.PAIR_COLUMNS.split(',')}
    shard,within=divmod(ordinal,SHARD_SIZE)
    latent=ROOT/'materialized/latents'/split/f'latents-{split}-{shard:05d}.safetensors'
    row.update(pair_ordinal=ordinal,pair_id=pair_id,split=split,work_shard=shard,row_in_shard=within,
        target_sample_id=new['sample_id'],donor_provenance_json=None,
        operation=OPERATIONS[task],operation_family='spatial_edit' if len(actions)==1 else 'compound_spatial_edit',
        raw_edit_request=instruction,instruction_template_id=f'spatial_multi500k_v2/{task}',
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
        target_count=len(new['sources']),target_domain=base['source_domain'],
        new_sceneplan_zlib=pack(new),new_sceneplan_sha256=sha256_json(new),
        target_render_recipe_zlib=pack(target_recipe),target_render_recipe_sha256=sha256_json(target_recipe),
        target_members_zlib=pack(members),target_members_sha256=sha256_json(members),
        edited_source_ids_json=canonical(sorted(selected)),unchanged_source_ids_json=canonical(sorted(set(src)-set(selected))),
        target_latent_path=str(latent),target_latent_key=new['sample_id'],target_latent_ref=f'{latent}#{new["sample_id"]}',
        target_latent_tensor_sha256=None,target_latent_shard_sha256=None,target_foa_path=None,target_foa_sha256=None,
        target_render_result_sha256=None,pair_gain_policy=native.PAIR_GAIN_POLICY,materialization_status='planned')
    row['pair_record_sha256']=sha256_json({k:row[k] for k in DIGEST_FIELDS})
    signature=digest(dict(source_audio_sha256=row['source_foa_sha256'],
        target_plan={k:v for k,v in new.items() if k!='sample_id'}))
    audit=dict(spec=spec,variant=variant,source_pair_ordinal=base['pair_ordinal'],
        actions=actions,signature=signature,actual_edited_sources=len(actions))
    validate(row,audit)
    return row,audit


def validate(row,audit):
    old=unpack(row['old_sceneplan_zlib']);new=unpack(row['new_sceneplan_zlib'])
    validate_model_sceneplan(new);validate_render_recipe_binding(new,unpack(row['target_render_recipe_zlib']))
    a={s['source_id']:s for s in old['sources']};b={s['source_id']:s for s in new['sources']}
    assert set(a)==set(b) and old['room']==new['room'] and old['duration_sec']==new['duration_sec']
    changed={sid for sid in a if a[sid]!=b[sid]}
    assert changed==set(json.loads(row['edited_source_ids_json']))=={v['source_id'] for v in audit['actions']}
    assert len(changed)==(1 if audit['spec']['task'].startswith('diagonal_') else 3 if audit['spec']['task']=='three_position_cycle' else 2)
    for sid in a:
        assert {k:v for k,v in a[sid].items() if k!='trajectory'}=={k:v for k,v in b[sid].items() if k!='trajectory'}
        if sid not in changed:assert a[sid]==b[sid]
    for act in audit['actions']:
        assert act['before']==a[act['source_id']]['trajectory'] and act['after']==b[act['source_id']]['trajectory']
        op=act['operation'];before=act['before'];after=act['after']
        assert before['type']==('linear' if op=='linear_to_static' else 'static')
        assert after['type']==('linear' if op=='static_to_linear' else 'static')
        if after['type']=='linear':assert after['start']==before['position'] and after['start']!=after['end']
    if audit['spec']['task'].startswith('diagonal_'):
        t=audit['actions'][0]['after'];p=t.get('end',t.get('position'))
        assert angle(p['azimuth_deg'],DIAGONALS[audit['spec']['direction']])<=12.000001
    assert row['pair_record_sha256']==sha256_json({k:row[k] for k in DIGEST_FIELDS})
