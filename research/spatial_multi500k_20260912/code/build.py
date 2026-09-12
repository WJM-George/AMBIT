"""Plan exact joint-stratum quotas; two distinct edits maximum per old scene."""
from common import *
from mutations import mutate,Ineligible,native
from collections import Counter,defaultdict
import fcntl

ORDER=['three_position_cycle','position_swap','dual_stop_motion','mixed_motion_toggle',
       'dual_start_motion','dual_relocation','diagonal_relocation','diagonal_start_motion']


def create_index(path):
    temp=native._create_final(':memory:')
    tables=list(temp.execute("SELECT sql FROM sqlite_master WHERE type='table'"))
    indexes=list(temp.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"));temp.close()
    out=sqlite3.connect(path);out.execute('PRAGMA journal_mode=OFF');out.execute('PRAGMA synchronous=OFF')
    for (sql,) in tables:
        out.execute(sql.replace('source_ordinal INTEGER NOT NULL UNIQUE','source_ordinal INTEGER NOT NULL')
            .replace('source_sample_id TEXT NOT NULL UNIQUE','source_sample_id TEXT NOT NULL'))
    for (sql,) in indexes:out.execute(sql)
    out.execute('CREATE TABLE edit_actions(pair_ordinal INTEGER PRIMARY KEY,audit_json TEXT NOT NULL,signature TEXT NOT NULL UNIQUE)')
    out.execute('CREATE INDEX pairs_source_sample ON pairs(source_sample_id)')
    return out


def metadata(out,split,review):
    source=db(BASE/'training_index'/f'{split}.sqlite')
    meta=dict(source.execute('SELECT key,value FROM metadata'));source.close()
    ops=dict(out.execute('SELECT operation,count(*) FROM pairs GROUP BY operation'))
    families=dict(out.execute('SELECT operation_family,count(*) FROM pairs GROUP BY operation_family'))
    meta.update(schema='sceneplan_transfusion_editing_pair_index',schema_version='1',state='planned_targets_not_materialized',
        mode='spatial_multi_source_expansion_v2',split=split,rows=str(COUNTS[split]),
        editing_pair_contract='spatial_only_compound_edit_actions_v2',
        editing_instruction_contract='unambiguous_per_source_raw_instructions_v2',
        editing_ar_input_contract='source_foa_latent_plus_raw_edit_request_v2',
        operation_counts_json=canonical(ops),operation_family_counts_json=canonical(families),
        source_index_path=str(BASE/'training_index'/f'{split}.sqlite'),source_index_sha256=review['source_index_sha256'],
        source_strata_counts_json=canonical(review['joint_quotas']),target_root=str(ROOT),
        augmentation_revision=ROOT.name,selection_policy='original_joint_strata_exact_quotas_max_two_distinct_edits_per_source',
        work_shard_size=str(SHARD_SIZE),compound_actions_table='edit_actions',
        loader_capability_required='compound_spatial_edit_actions_v2',
        builder_path=str(Path(__file__)),builder_sha256=sha(__file__),mutation_module_sha256=sha(Path(__file__).with_name('mutations.py')))
    out.executemany('INSERT INTO metadata VALUES(?,?)',sorted(meta.items()));out.commit()


def build(split):
    marker=ROOT/'pair_index'/f'{split}.json';path=marker.with_suffix('.sqlite')
    if marker.exists():assert sha(path)==read(marker)['sha256'];return
    review=read(ROOT/'catalog'/f'{split}.json');cat=db(ROOT/'catalog'/f'{split}.sqlite')
    assert sha(ROOT/'catalog'/f'{split}.sqlite')==review['sha256']
    remaining=dict(review['joint_quotas']);task_quotas=tasks_for(split);allocations={}
    for task in ORDER:
        caps={k:min(v,review['eligibility'].get(task+'|'+k,0)*2) for k,v in remaining.items()}
        allocation_task=allocation(remaining,task_quotas[task],caps)
        allocations[task]=allocation_task
        for k,v in allocation_task.items():remaining[k]-=v
    assert sum(remaining.values())==0
    pools={};cursors=defaultdict(int)
    for task,groups in allocations.items():
        for group,n in groups.items():
            if n:pools[task,group]=[r[0] for r in cat.execute('SELECT ordinal FROM eligibility WHERE task=? AND stratum=? ORDER BY rank,ordinal',(task,group))]
    jobs=[];diagonal_index=0
    for task in ORDER:
        for group,n in sorted(allocations[task].items()):
            for i in range(n):
                direction=None
                if task.startswith('diagonal_'):
                    direction=list(DIAGONALS)[diagonal_index%4];diagonal_index+=1
                jobs.append(dict(task=task,stratum=group,direction=direction,anchor=bool(i%2==0),slot=i))
    jobs.sort(key=lambda s:seed(split,s,'output_order'))
    for ordinal,job in enumerate(jobs):job['ordinal']=ordinal
    # Reserve scarce multi-source candidates first; output ordinals stay shuffled.
    jobs.sort(key=lambda s:(ORDER.index(s['task']),s['ordinal']))
    tmp=path.with_suffix(f'.tmp.{os.getpid()}');tmp.unlink(missing_ok=True);out=create_index(tmp)
    source=db(BASE/'training_index'/f'{split}.sqlite');usage=Counter();signatures=set();counts=Counter();dirs=Counter()
    columns=native.PAIR_COLUMNS.split(',');pending=[];actions=[]
    for processed,job in enumerate(jobs,1):
        ordinal=job['ordinal']
        spec=dict(job,ordinal=ordinal,split=split);key=job['task'],job['stratum'];pool=pools[key]
        found=None
        for attempt in range(len(pool)*2):
            candidate=pool[cursors[key]%len(pool)];cursors[key]+=1
            if usage[candidate]>=2:continue
            base=dict(source.execute('SELECT * FROM pairs WHERE pair_ordinal=?',(candidate,)).fetchone())
            try:row,audit=mutate(base,spec)
            except Ineligible:continue
            if audit['signature'] in signatures:continue
            found=(row,audit);usage[candidate]+=1;signatures.add(audit['signature']);break
        if found is None:raise RuntimeError(f'No eligible distinct source for {spec}')
        row,audit=found;pending.append(tuple(row[k] for k in columns));actions.append((ordinal,canonical(audit),audit['signature']))
        counts[job['stratum']]+=1
        if job['direction']:dirs[job['direction']]+=1
        if len(pending)>=512 or processed==len(jobs):
            out.executemany(f'INSERT INTO pairs({native.PAIR_COLUMNS}) VALUES({",".join("?" for _ in columns)})',pending)
            out.executemany('INSERT INTO edit_actions VALUES(?,?,?)',actions);out.commit();pending.clear();actions.clear()
        if processed%10000==0 or processed==len(jobs):state('planning_pairs',split=split,done=processed,total=len(jobs))
    assert counts==Counter(review['joint_quotas']) and max(usage.values())<=2
    assert max(dirs.values())-min(dirs.values())<=1
    metadata(out,split,review)
    assert out.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    assert out.execute('SELECT count(*),count(distinct pair_id) FROM pairs').fetchone()==(COUNTS[split],COUNTS[split])
    out.close();source.close();cat.close();tmp.replace(path)
    write(marker,dict(status='PASS_EXACT_QUOTAS_COMPOUND_PAIR_PLANS',split=split,rows=COUNTS[split],
        tasks=task_quotas,joint_strata=dict(counts),diagonal_targets=dict(dirs),unique_original_scenes=len(usage),
        max_new_edits_per_scene=max(usage.values()),sha256=sha(path),created_at=now()))


if __name__=='__main__':
    lock=(ROOT/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for split in ['validation','test','train']:build(split)
    write(ROOT/'DATA_PLAN_COMPLETE.json',dict(status='PASS_ALL_SPATIAL_MULTI_SOURCE_PLANS',counts=COUNTS,created_at=now()))
