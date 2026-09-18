"""Retain original four-rank RF validation noises while using eight workers."""
from collections import defaultdict
import hashlib
import time
import types

import torch
import torch.distributed as dist


def canonical_batches(entries,worker,world):
    if world not in (4,8) or not 0 <= worker < world:
        raise ValueError('Require four or eight validation workers.')
    # Every (logical_rank,index,ids) matches the completed four-GPU baseline.
    result=[]
    for logical_rank in range(4):
        buckets=defaultdict(list)
        for ordinal,bucket,_ in entries:
            if ordinal%4==logical_rank:buckets[bucket].append(ordinal)
        batches=[ids[start:start+32] for _,ids in sorted(buckets.items()) for start in range(0,len(ids),32)]
        for index,ids in enumerate(batches):
            if logical_rank+4*(index%(world//4))==worker:
                result.append((logical_rank,index,ids))
    return result


@torch.no_grad()
def validate_native(learner,model,*,max_batches=0,label='full'):
    from scripts.t2a.experiments.ar_structured_v1 import data,model as native_model
    from scripts.t2a.rl.train_editing_opsd_stream import write
    rank,world,device=learner.rank,learner.world,learner.device
    started=time.monotonic();dataset=learner.validation
    entries=dataset._db().execute('SELECT pair_ordinal,latent_bucket_frames,target_domain FROM pairs ORDER BY pair_ordinal').fetchall()
    if len(dataset)!=20000 or [r[0] for r in entries]!=list(range(20000)):
        raise ValueError('Require the complete original20k validation.')
    batches=canonical_batches(entries,rank,world)
    if max_batches:batches=batches[:max_batches]
    speech={i:domain in ('speech_only','speech_mixed') for i,_,domain in entries}
    totals=None;denominators=torch.zeros(3,dtype=torch.float64,device=device)
    seen=[];speech_rows=0
    was_training=model.training;model.eval()
    forward=types.MethodType(native_model.model_forward,model)
    try:
        for logical_rank,index,ids in batches:
            learner.progress('NATIVE_VALIDATION_LOSS',completed_local_rows=len(seen),total_rows=20000,
                             step=learner.step,canonical_world=4)
            batch=data.collate([dataset[i] for i in ids],pad_id=learner.adapter.codec.pad_id,joint=True)
            ar,target,metadata,mask=learner.native._move_joint_batch(batch,device)
            den=denominators.new_tensor([(ar['plan_labels']!=-100).sum(),mask.sum()*64,len(ids)])
            loss,sums,names,outputs=learner.native.batch_loss(forward,learner.teacher,batch,
                learner.cfg,device,500000+index,logical_rank,4,den)
            if not torch.isfinite(loss) or not torch.isfinite(sums).all():
                raise RuntimeError('Nonfinite canonical native validation.')
            if totals is None:totals=torch.zeros_like(sums)
            totals+=sums;denominators+=den;seen.extend(ids);speech_rows+=sum(speech[i] for i in ids)
            del batch,ar,target,metadata,mask,loss,sums,outputs
    finally:model.train(was_training)
    if totals is None or len(seen)!=len(set(seen)):
        raise ValueError('Empty or duplicated validation shard.')
    count=torch.tensor(speech_rows,dtype=torch.int64,device=device)
    dist.all_reduce(totals);dist.all_reduce(denominators);dist.all_reduce(count)
    if not max_batches and (int(denominators[2])!=20000 or int(count)!=11918):
        raise ValueError('Incomplete native validation population.')
    normalized=torch.cat((totals[:3]/denominators,totals[3:]/denominators[2]))
    receipt=dict(step=learner.step,rows=int(denominators[2]),speech_rows=int(count),local_rows=len(seen),
        unique_local_rows=len(set(seen)),rank=rank,world=world,canonical_noise_world=4,
        ordered_ordinal_sha256=hashlib.sha256(','.join(map(str,seen)).encode()).hexdigest(),
        metrics=dict(zip(('AR_CE','RF_MSE','structured_loss',*names),normalized.cpu().tolist())),
        seconds=time.monotonic()-started,probe=bool(max_batches),
        scope='Original four-rank batches and RF noises dispatched over eight workers; teacher-forced diagnostics.')
    directory=learner.out/'native_validation';directory.mkdir(exist_ok=True)
    write(directory/f'{label}_step{learner.step:06d}_rank{rank}.json',receipt)
    if rank==0:write(directory/f'{label}_step{learner.step:06d}.json',receipt)
    return receipt
