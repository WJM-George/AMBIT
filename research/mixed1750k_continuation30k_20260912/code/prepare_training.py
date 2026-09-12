"""Freeze 30k updates sampled uniformly over the three immutable components."""
from common import *
from collections import Counter,defaultdict
import numpy as np


def schedule(component_pools,steps=30000,seed=202609121750):
    rng=np.random.default_rng(seed);batches={432:216,648:144}
    pools={b:np.asarray([o for (c,k),rows in component_pools.items() if k==b for o in rows],dtype=np.int64) for b in batches}
    assert all(len(v) for v in pools.values())
    probability=np.asarray([len(pools[b])/batches[b] for b in batches]);probability/=probability.sum()
    buckets=rng.choice(list(batches),size=steps,p=probability).astype(np.int16);buckets[:2]=[432,648]
    decks={b:rng.permutation(v) for b,v in pools.items()};cursor={b:0 for b in pools}
    total=sum(len(v) for v in pools.values());seen=np.zeros(total,dtype=np.bool_);seen_count=0
    group_order=np.zeros((steps,3,18),dtype=np.int64);groups=[];four_rows=[];exposures=[0];unique=[0]
    for step,b in enumerate(buckets):
        b=int(b);n=batches[b]//12;groups.append(n);selected=[]
        for flat in range(n*3):
            chosen=[]
            for _ in range(4):
                if cursor[b]==len(decks[b]):decks[b]=rng.permutation(pools[b]);cursor[b]=0
                chosen.append(int(decks[b][cursor[b]]));cursor[b]+=1
            group_order[step,flat//n,flat%n]=len(four_rows);four_rows.append(chosen);selected.extend(chosen)
        ids=np.unique(selected);seen_count+=int((~seen[ids]).sum());seen[ids]=True
        exposures.append(exposures[-1]+batches[b]);unique.append(seen_count)
    while len(four_rows)%5:four_rows.append(four_rows[-1])
    return dict(ORDER=np.asarray(four_rows,dtype=np.int64).reshape(-1,5,4),GROUP_ORDER=group_order,
        GROUP_COUNTS=np.asarray(groups,dtype=np.int16),BUCKETS=buckets,
        EXPOSURES=np.asarray(exposures,dtype=np.int64),SEEN_COUNTS=np.asarray(unique,dtype=np.int64))


def main():
    ready=read(SPATIAL/'DATA_READY.json')
    assert ready['status']=='PASS_500K_SPATIAL_MULTI_TRAIN_LATENTS_AND_SPLIT_EXTENSIONS'
    assert ready['counts']==COMPONENT_COUNTS['spatial_multi500k']
    run=ROOT/'training';run.mkdir(exist_ok=True)
    if (run/'PREPARED.json').exists():
        marker=read(run/'PREPARED.json');assert sha(run/'PLAN.json')==marker['plan_sha256'];return
    binding=components('train');pools=defaultdict(list);operations=Counter()
    for c in binding['components']:
        con=db(c['index_path'])
        for ordinal,bucket,operation in con.execute('SELECT pair_ordinal,latent_bucket_frames,operation FROM pairs ORDER BY pair_ordinal'):
            pools[c['name'],bucket].append(c['offset']+ordinal);operations[operation]+=1
        con.close()
    arrays=schedule(pools);assert arrays['SEEN_COUNTS'][-1]==COUNTS['train']
    old=read(PREVIOUS/'training/PLAN.json')
    arrays['TIMES']=np.load(PREVIOUS/'training/TIMES.npy',allow_pickle=False)
    assert arrays['TIMES'].shape==(500,5,4)
    refs={}
    for name,value in arrays.items():
        path=run/(name+'.npy');np.save(path,value,allow_pickle=False);refs[path.name]=dict(path=str(path),sha256=sha(path))
    # Count actual exposures, excluding unused padding in the last RF unit.
    exposure_counts=Counter()
    order=arrays['ORDER'].reshape(-1,4)
    for step,n in enumerate(arrays['GROUP_COUNTS']):
        ids=order[arrays['GROUP_ORDER'][step,:,:n].reshape(-1)].reshape(-1)
        for c in binding['components']:
            exposure_counts[c['name']]+=int(((ids>=c['offset'])&(ids<c['offset']+c['rows'])).sum())
    for c in binding['components']:
        assert abs(exposure_counts[c['name']]/sum(exposure_counts.values())-c['rows']/COUNTS['train'])<.005
    initial=old['initial_checkpoint'];assert initial['optimizer_steps']==50000 and sha(initial['path'])==initial['sha256']
    plan={k:old[k] for k in ['runtime_reference','generation_initialization','optimizer','gradient_clip_norm',
        'Qwen_mode','NUMA_binding','torch_deterministic_algorithms','activation_checkpointing',
        'batch_per_gpu_by_bucket','CPU_worker_processes_per_rank','historical_pair_exposures']}
    plan.update(schema='editing_mixed1750k_training_plan_v1',created_at=now(),physical_gpus=GPUS,
        start_optimizer_steps=50000,stop_optimizer_steps=80000,checkpoint_optimizer_steps=STEPS[1:],
        dataset=binding,dataset_rows=COUNTS['train'],validation_rows=COUNTS['validation'],test_rows=COUNTS['test'],
        initial_checkpoint=initial,lr_schedule=dict(warmup_steps=500,peak=2e-5,decay='cosine to 5e-6 at +30k'),
        noise_seed_offset=912175000,forward_seed_offset=1912175000,input_arrays=refs,
        sampling_target={c['name']:c['rows']/COUNTS['train'] for c in binding['components']},
        sampling_actual_counts=dict(exposure_counts),sampling_rule='Uniform per pair; bucket step probabilities compensate unequal batch sizes',
        additional_pair_exposures=int(arrays['EXPOSURES'][-1]),new_run_unique_pairs_seen=int(arrays['SEEN_COUNTS'][-1]),
        selection_uses_validation_only=True,warm_start='Exact original50k model and Adam/RNG state; new sampling and 30k LR schedule')
    assert plan['optimizer']['lr']==2e-5
    write(run/'PLAN.json',plan)
    write(ROOT/'DATA_READY.json',dict(status='PASS_THREE_FROZEN_COMPONENTS_FOR_TRAINING',counts=COUNTS,
        train=binding,spatial_data_ready_sha256=sha(SPATIAL/'DATA_READY.json'),components_preserved=True,prepared_at=now()))
    write(run/'PREPARED.json',dict(status='PASS_FROZEN_UNIFORM_1750K_30K_SCHEDULE',plan_sha256=sha(run/'PLAN.json'),
        counts=dict(exposure_counts),all1750000_pairs_exposed=True,checkpoint_offsets=[5000,10000,15000,20000,25000,30000],prepared_at=now()))


if __name__=='__main__':main()
