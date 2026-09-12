"""CPU integration on real components; final-data checks run before DDP."""
from common import *
import argparse
import base64
from collections import defaultdict,Counter
import copy
import pickle
import time


def tiny_original(name,split,directory,n=16):
    source=component_path(name,split);old=db(source);out=directory/name/f'{split}.sqlite';out.parent.mkdir(parents=True)
    con=sqlite3.connect(out)
    for table in ['pairs','metadata','target_shards']:
        con.execute(old.execute('SELECT sql FROM sqlite_master WHERE name=?',(table,)).fetchone()[0])
    columns=[r[1] for r in old.execute('PRAGMA table_info(pairs)')]
    values=old.execute('SELECT * FROM pairs ORDER BY pair_ordinal LIMIT ?',(n,)).fetchall()
    con.executemany('INSERT INTO pairs VALUES('+','.join('?' for _ in columns)+')',[tuple(r) for r in values])
    metadata=dict(old.execute('SELECT key,value FROM metadata'));metadata['rows']=str(n)
    con.executemany('INSERT INTO metadata VALUES(?,?)',metadata.items())
    for shard in {r['work_shard'] for r in values}:
        con.execute('INSERT INTO target_shards VALUES(?,?,?,?,?,?)',tuple(old.execute('SELECT * FROM target_shards WHERE work_shard=?',(shard,)).fetchone()))
    con.commit();con.close();old.close()
    marker=read(str(source)+'.frozen.json');marker.update(index_path=str(out),index_sha256=sha(out),rows=n)
    write(str(out)+'.frozen.json',marker)
    return dict(name=name,**marker)


def tiny_spatial(split,directory,n):
    from scripts.t2a.data.finalize_sceneplan_transfusion_editing_index import finalize
    old=db(SPATIAL/'pair_index'/f'{split}.sqlite');planned=directory/f'spatial_{split}_planned.sqlite'
    con=sqlite3.connect(planned)
    for table in ['pairs','metadata']:
        con.execute(old.execute('SELECT sql FROM sqlite_master WHERE name=?',(table,)).fetchone()[0])
    columns=[r[1] for r in old.execute('PRAGMA table_info(pairs)')]
    rows=[]
    for candidate in old.execute('SELECT * FROM pairs WHERE pair_ordinal<? ORDER BY pair_ordinal',(n,)):
        path=SPATIAL/'calibrated_rows'/split/f'work-{candidate["work_shard"]:05d}'/f'{candidate["pair_ordinal"]:07d}.json.zlib'
        record=unpack(path.read_bytes());assert record['candidate_pair_record_sha256']==candidate['pair_record_sha256']
        row={k:base64.b64decode(v['zlib_base64']) if isinstance(v,dict) and set(v)=={'zlib_base64'} else v for k,v in record['row'].items()}
        rows.append(tuple(row[k] for k in columns))
    assert len(rows)==n
    con.executemany('INSERT INTO pairs VALUES('+','.join('?' for _ in columns)+')',rows)
    metadata=dict(old.execute('SELECT key,value FROM metadata'));metadata['rows']=str(n)
    con.executemany('INSERT INTO metadata VALUES(?,?)',metadata.items());con.commit();con.close();old.close()
    out=directory/'spatial_multi500k'/f'{split}.sqlite';out.parent.mkdir(parents=True,exist_ok=True)
    marker=finalize(planned,out,replace=False)
    return dict(name='spatial_multi500k',**marker)


def selection_checks():
    from select_checkpoint import score_candidate,decide
    protocol=read(PREVIOUS/'SELECTION_PROTOCOL.json');base={};o=0
    keys=set(protocol['original_operation_guards'])|{k for v in protocol['dimensions'].values() for k in v['metrics']}
    for op in protocol['dimensions']['audio']['operations']:
        for j in range(40):base[o]=dict(operation=op,cohort='original',metrics={k:1+j*.001 for k in keys});o+=1
    direction={k:d for v in protocol['dimensions'].values() for k,d in v['metrics'].items()}
    direction.update({k:d['direction'] for k,d in protocol['original_operation_guards'].items()})
    good=copy.deepcopy(base);bad=copy.deepcopy(base)
    for rows in [good,bad]:
        for row in rows.values():
            for key in row['metrics']:
                sign=-1 if row['operation']=='event_removal' and key=='independent_clap_output_edit_text_cosine' else direction[key]
                row['metrics'][key]+=.01*sign
    for row in bad.values():row['metrics']['unchanged_demix_si_sdr_delta_vs_copy_db']-=2
    scores={50000:score_candidate(base,base,protocol),55000:score_candidate(base,good,protocol),80000:score_candidate(base,bad,protocol)}
    assert decide(scores)==55000 and not scores[80000]['eligible']
    assert decide({50000:scores[50000],80000:scores[80000]})==50000
    absent=copy.deepcopy(good)
    for row in absent.values():row['metrics']['audio_codec_foa_nmse']=None
    assert not score_candidate(base,absent,protocol)['eligible']
    return dict(preservation_regression_rejected=True,missing_metrics_rejected=True,no_improvement_retains50k=True)


def checks(final_components=False):
    import torch
    import numpy as np
    from transformers import AutoTokenizer
    from mixture_dataset import MixtureDataset
    from prepare_training import schedule
    import importlib.util
    train_spec=importlib.util.spec_from_file_location('mixed1750k_training',ROOT/'code/train.py')
    train_module=importlib.util.module_from_spec(train_spec);train_spec.loader.exec_module(train_module)
    lr_at=train_module.lr_at
    torch.set_num_threads(1)
    if final_components:
        binding=read(ROOT/'training/PLAN.json')['dataset'];verify_binding(binding)
        selected=[]
        for c in binding['components']:
            con=db(c['index_path'])
            for o in con.execute('SELECT min(pair_ordinal) FROM pairs GROUP BY operation,source_count,latent_bucket_frames'):
                row=con.execute('SELECT pair_id FROM pairs WHERE pair_ordinal=?',(o[0],)).fetchone()
                selected.append(dict(pair_ordinal=c['offset']+o[0],pair_id=row[0],cohort=c['name']))
            con.close()
        marker='FINAL_COMPONENT_REVIEW.json';fixture=None
    else:
        fixture=ROOT/'reviews'/f'cpu_integration_{time.time_ns()}';fixture.mkdir()
        comp=[tiny_original('original','train',fixture),tiny_original('addition250k','train',fixture),tiny_spatial('train',fixture,512)]
        offset=0
        for c in comp:c['offset']=offset;offset+=c['rows']
        binding=dict(schema='three_immutable_editing_components_v1',split='train',rows=offset,components=comp);selected=None
        marker='WORKFLOW_CPU_REVIEW.json'
    tokenizer=AutoTokenizer.from_pretrained('/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B',local_files_only=True)
    dataset=MixtureDataset(binding,(tokenizer,512),sample_cases=selected)
    buckets=defaultdict(list);operations=Counter();cohorts=Counter();identities=[]
    for i in range(len(dataset)):
        target,meta=dataset[i]
        assert target.shape==meta['source_foa_latent'].shape==(64,648)
        assert torch.isfinite(target).all() and torch.isfinite(meta['source_foa_latent']).all()
        assert 'old_sceneplan' not in meta and 'model_sceneplan' in meta
        buckets[meta['latent_bucket_frames']].append(i);operations[meta['operation']]+=1;cohorts[meta['dataset_component']]+=1
        identities.append(dict(pair_id=meta['pair_id'],operation=meta['operation'],component=meta['dataset_component']))
    assert set(cohorts)==set(COMPONENT_COUNTS) and set(buckets)=={432,648}
    dataset.close()
    if not final_components:
        assert len(operations)>=10
        cloned=pickle.loads(pickle.dumps(dataset));assert cloned[0][1]['pair_id']==identities[0]['pair_id'];cloned.close()
        from balanced_inputs_v1 import BalancedInputs
        from throughput_cpu_loader_v1 import OrderedCPUStream
        inputs=BalancedInputs.__new__(BalancedInputs);inputs.dataset=dataset
        inputs.order=np.array([[buckets[432][:4]]*5,[buckets[648][:4]]*5],dtype=np.int64)
        inputs.times=np.load(PREVIOUS/'training/TIMES.npy');inputs.schedule={'noise_seed_offset':912175000}
        stream=OrderedCPUStream(inputs,[np.array([0]),np.array([5])],workers=4)
        try:
            for bucket in [432,648]:
                batch=stream.next();assert batch['target'].shape==(4,64,bucket)
                assert not any({'old_sceneplan','target_foa_latent'} & set(row) for row in batch['rows'])
        finally:stream.close();dataset.close()
        pools={('fixture',b):ids for b,ids in buckets.items()}
        frozen=schedule(pools,steps=40);assert frozen['SEEN_COUNTS'][-1]==len(dataset)
        for s,n in enumerate(frozen['GROUP_COUNTS']):
            indices=frozen['ORDER'].reshape(-1,4)[frozen['GROUP_ORDER'][s,:,:n].reshape(-1)].reshape(-1)
            assert all(o in set(buckets[int(frozen['BUCKETS'][s])]) for o in indices)
        lrplan={'lr_schedule':{'warmup_steps':500},'optimizer':{'lr':2e-5}}
        assert 0<lr_at(0,lrplan)<lr_at(499,lrplan)==2e-5 and abs(lr_at(29999,lrplan)-5e-6)<1e-12
        decisions=selection_checks()
        # Build a real held-out spatial fixture, then restore and hash-check
        # source, edited GT, and unchanged-source stems without touching train.
        val=tiny_spatial('validation',fixture,128);val['offset']=0
        from truth_cache import restore_one,TruthResolver
        import truth_cache
        old_root=truth_cache.ROOT;truth_cache.ROOT=fixture/'truth_test'
        connection=db(val['index_path']);chosen=[r[0] for r in connection.execute('SELECT min(pair_ordinal) FROM pairs GROUP BY operation')];connection.close()
        restored=[]
        try:
            for ordinal in chosen:restored.append(restore_one((val,ordinal)))
            resolver=TruthResolver(dict(split='validation',rows=128,components=[val]))
            try:
                for ordinal in chosen:
                    truth=resolver.row(ordinal);assert torch.isfinite(truth['source_foa']).all() and torch.isfinite(truth['target_foa']).all()
                    assert set(truth['source_stem_foa'])==set(truth['unchanged_source_ids'])
            finally:resolver.close()
        finally:truth_cache.ROOT=old_root
        write(fixture/'FIXTURE_BINDING.json',binding)
    else:decisions={};restored=[]
    assert not torch.cuda.is_initialized()
    write(ROOT/'reviews'/marker,dict(status='PASS_FINAL_COMPONENT_READER_CHECK' if final_components else 'PASS_REAL_COMPONENTS_MIXTURE_PREFETCH_AND_GT_RESTORE',
        checked_rows=len(identities),component_counts=dict(cohorts),operation_counts=dict(operations),both_buckets=True,
        fixture=str(fixture) if fixture else None,truth_restore_rows=len(restored),decisions=decisions,
        old_plan_not_in_model_inputs=True,no_training_audio_generated=True,production_DDP_check_pending=True,
        GPU_initialized=False,completed_at=now()))
    print(read(ROOT/'reviews'/marker),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final-components',action='store_true');args=parser.parse_args();checks(args.final_components)
