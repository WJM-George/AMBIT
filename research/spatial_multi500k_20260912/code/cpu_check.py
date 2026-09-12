"""Review all pair metadata independently; replay representative source PCM in memory."""
from common import *
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import traceback


def inspect_pair(row,audit):
    old=unpack(row['old_sceneplan_zlib']);new=unpack(row['new_sceneplan_zlib'])
    a={s['source_id']:s for s in old['sources']};b={s['source_id']:s for s in new['sources']}
    assert set(a)==set(b) and len(a)==row['source_count']==row['target_count']
    assert {k:v for k,v in old.items() if k not in ['sample_id','sources']}=={k:v for k,v in new.items() if k not in ['sample_id','sources']}
    edited={sid for sid in a if a[sid]!=b[sid]};task=audit['spec']['task']
    assert edited==set(json.loads(row['edited_source_ids_json']))
    assert set(a)-edited==set(json.loads(row['unchanged_source_ids_json']))
    assert len(edited)==(1 if task.startswith('diagonal_') else 3 if task=='three_position_cycle' else 2)
    for sid in a:
        assert {k:v for k,v in a[sid].items() if k!='trajectory'}=={k:v for k,v in b[sid].items() if k!='trajectory'}
    if task in ['position_swap','three_position_cycle']:
        old_positions={sid:a[sid]['trajectory']['position'] for sid in edited}
        new_positions={sid:b[sid]['trajectory']['position'] for sid in edited}
        assert sorted(map(canonical,old_positions.values()))==sorted(map(canonical,new_positions.values()))
        assert all(new_positions[sid]!=old_positions[sid] for sid in edited)
    before=[a[sid]['trajectory']['type'] for sid in sorted(edited)]
    after=[b[sid]['trajectory']['type'] for sid in sorted(edited)]
    if task in ['diagonal_start_motion','dual_start_motion']:assert set(before)=={'static'} and set(after)=={'linear'}
    if task=='dual_stop_motion':assert set(before)=={'linear'} and set(after)=={'static'}
    if task=='mixed_motion_toggle':assert sorted(before)==sorted(after)==['linear','static'] and all(x!=y for x,y in zip(before,after))
    if task.startswith('diagonal_'):
        t=b[next(iter(edited))]['trajectory'];p=t['end'] if t['type']=='linear' else t['position']
        delta=abs((p['azimuth_deg']-DIAGONALS[audit['spec']['direction']]+180)%360-180)
        assert delta<=12.000001
    assert hashlib.sha256(row['raw_edit_request'].encode()).hexdigest()==row['instruction_sha256']
    return task


def inspect_all():
    fixtures=[];results={}
    for split in COUNTS:
        con=db(ROOT/'pair_index'/f'{split}.sqlite');marker=read(ROOT/'pair_index'/f'{split}.json')
        assert sha(ROOT/'pair_index'/f'{split}.sqlite')==marker['sha256']
        counts=Counter();tasks=Counter();selected=set();n=0
        for r in con.execute('SELECT p.*,a.audit_json FROM pairs p JOIN edit_actions a USING(pair_ordinal) ORDER BY pair_ordinal'):
            row=dict(r);audit=json.loads(row.pop('audit_json'));task=inspect_pair(row,audit)
            group=f'{row["source_count"]}|{row["source_domain"]}|{row["latent_bucket_frames"]}'
            counts[group]+=1;tasks[task]+=1;n+=1
            key=(task,row['source_count'],row['latent_bucket_frames'])
            if split=='train' and key not in selected:
                selected.add(key);fixtures.append((row,audit,True))
        assert n==COUNTS[split] and dict(counts)==marker['joint_strata'] and dict(tasks)==marker['tasks']
        # SQL equality against the original, on immutable source identifiers.
        con.close();con=sqlite3.connect(f'file:{ROOT/"pair_index"/f"{split}.sqlite"}?mode=ro&immutable=1',uri=True)
        con.execute('ATTACH DATABASE ? AS original',(f'file:{BASE/"training_index"/f"{split}.sqlite"}?mode=ro&immutable=1',))
        fields=['source_ordinal','source_foa_sha256','old_sceneplan_sha256','source_render_recipe_sha256',
            'source_render_result_sha256','source_members_sha256','source_latent_ref','source_latent_tensor_sha256','source_latent_shard_sha256']
        equal=' AND '.join(f'p.{k} IS o.{k}' for k in fields)
        matched=con.execute(f'SELECT count(*) FROM pairs p JOIN original.pairs o USING(source_sample_id) WHERE {equal}').fetchone()[0]
        assert matched==n
        assert con.execute('SELECT max(n) FROM (SELECT count(*) n FROM pairs GROUP BY source_sample_id)').fetchone()[0]<=2
        results[split]=dict(rows=n,task_counts=dict(tasks),joint_strata=dict(counts),source_records_equal_to_original=matched)
        con.close()
    write(ROOT/'reviews/ALL_PLANS.json',dict(status='PASS_ALL_PAIR_DELTAS_QUOTAS_AND_ORIGINAL_SOURCE_BINDINGS',
        splits=results,fixtures=len(fixtures),checked_at=now()))
    return fixtures


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from render import reservations,render_one,init_cpu
    assert read(ROOT/'DATA_PLAN_COMPLETE.json')['counts']==COUNTS
    fixtures=inspect_all();reservations();replays=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=init_cpu) as pool:
        for row,audit,result in pool.map(render_one,fixtures,chunksize=1):
            inspect_pair(row,audit)
            assert result['source_parity_verified'] and result['pair_gain_qc']['target_master_delta_db']>=-1.0001
            replays.append(dict(pair_id=row['pair_id'],task=audit['spec']['task'],source_count=row['source_count'],
                bucket=row['latent_bucket_frames'],source_pcm_sha256=result['source_parity_sha256'],
                target_pcm_sha256=result['target_foa_sha256'],master_delta_db=result['pair_gain_qc']['target_master_delta_db']))
    forbidden=[str(p) for suffix in ['*.wav','*.flac','*.ogg','*.mp3'] for p in ROOT.rglob(suffix)]
    assert not forbidden,forbidden[:10]
    write(ROOT/'reviews/CPU_RENDER_REVIEW.json',dict(status='PASS_METADATA_AND_BYTE_EXACT_SOURCE_RENDER_CANARIES',
        fixtures=len(replays),replays=replays,retained_audio_files=0,checked_at=now()))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/'reviews/CPU_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),observed_at=now()))
        raise
