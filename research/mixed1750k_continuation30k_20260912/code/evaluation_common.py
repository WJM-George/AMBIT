"""Frozen cases, component identities, and the established DiT sampler."""
from common import *
from math import isfinite

FIELDS='pair_ordinal,pair_id,operation,source_count,target_count,source_domain,target_domain,latent_bucket_frames,model_num_samples,latent_frames_valid,new_sceneplan_sha256,old_sceneplan_sha256,source_foa_sha256,target_foa_sha256,pair_record_sha256'

def checkpoint(step):
    assert step in STEPS
    directory=MAIN/'CONTINUOUS50K_V2/checkpoints/STEP050000' if step==50000 else ROOT/f'training/checkpoints/STEP{step:06d}'
    ref=read(directory/'CHECKPOINT.json');assert ref['optimizer_steps']==step and sha(ref['path'])==ref['sha256']
    return ref

def all_cases(binding):
    result=[]
    for c in binding['components']:
        con=db(c['index_path'])
        for row in con.execute('SELECT '+FIELDS+' FROM pairs ORDER BY pair_ordinal'):
            row=dict(row);local=row['pair_ordinal'];row.update(native_pair_ordinal=local,pair_ordinal=c['offset']+local,cohort=c['name'])
            result.append(row)
        con.close()
    assert len(result)==binding['rows'] and [r['pair_ordinal'] for r in result]==list(range(binding['rows']))
    assert len({r['pair_id'] for r in result})==len(result)
    return result

def cases(scope):
    assert scope in SCOPES
    spec=read(ROOT/'CASESETS.json')[scope]
    assert spec['case_manifest_sha256']==digest(spec['cases'])
    assert len(spec['cases'])==(2000 if scope=='validation' else {'test_original':5000,'test_addition250k':1250,'test_spatial_multi500k':2500}[scope])
    return spec['index'],spec['cases']

def listening_cases(rows):
    groups={}
    for row in rows:groups.setdefault((row['cohort'],row['operation'],row['latent_bucket_frames']),[]).append(row)
    return sorted(r['pair_ordinal'] for values in groups.values() for r in sorted(values,key=lambda x:digest(['listen',x['pair_id']]))[:2])

def check_row(row,spec,plan):
    assert row['status']=='ok' and row['evaluation_plan_sha256']==plan['plan_sha256']
    assert row['checkpoint_steps']==plan['checkpoint_steps'] and row['evaluation_split']==plan['split']
    assert row['active_parameter_sha256']==plan['checkpoint']['final_parameter_sha256']
    for k in FIELDS.split(','):
        if k not in ['old_sceneplan_sha256','pair_record_sha256']:assert row[k]==spec[k],k
    assert row['cohort']==spec['cohort'] and row['native_pair_ordinal']==spec['native_pair_ordinal']
    assert row['model_input_contract']['old_sceneplan'] is False and not row['training_example_seen']
    assert digest(row['generated_sceneplan'])==spec['new_sceneplan_sha256']
    assert row['sampler_settings']==SAMPLER
    assert bool(row['edited_foa_path'])==(spec['pair_ordinal'] in plan['listening_ordinals'])
    for k in ['source_latent_sha256','initial_noise_sha256','output_latent_sha256']:assert len(row[k])==64
    assert all(v is None or not isinstance(v,(float,int)) or isfinite(v) for v in row['metrics'].values())

def load_case(path,spec,plan):
    envelope=read(path);assert digest(envelope['row'])==envelope['row_sha256'];check_row(envelope['row'],spec,plan)
    return envelope['row']
