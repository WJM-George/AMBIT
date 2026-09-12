"""CPU verification of all six complete model/Adam/RNG recovery artifacts."""
from common import *

def parameter_hash(state):
    import torch
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def verify():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    run=ROOT/'training';plan=read(run/'PLAN.json');review=read(run/'TECHNICAL_REVIEW.json')
    assert review['plan_sha256']==sha(run/'PLAN.json') and review['status']=='PASS_TWO_BUCKET_DDP_UPDATES_FROM_50K'
    base=plan['initial_checkpoint'];assert sha(base['path'])==base['sha256']
    original=torch.load(base['path'],map_location='cpu',weights_only=True,mmap=True)
    names=original['optimizer_parameter_names'];shapes={k:tuple(v.shape) for k,v in original['state'].items()}
    assert len(shapes)==201 and sum(v.numel() for v in original['state'].values())==319314336
    frozen=original['frozen_parameters'];initial=original['initial_parameter_sha256'];del original
    exposures=np.load(run/'EXPOSURES.npy',mmap_mode='r');unique=np.load(run/'SEEN_COUNTS.npy',mmap_mode='r')
    results=[]
    for step in [55000,60000,65000,70000,75000,80000]:
        directory=run/'checkpoints'/f'STEP{step:06d}';ref=read(directory/'CHECKPOINT.json')
        assert ref==read(Path(ref['path']).with_suffix('.json')) and ref['optimizer_steps']==step
        assert sha(ref['path'])==ref['sha256']
        saved=torch.load(ref['path'],map_location='cpu',weights_only=True,mmap=True)
        assert saved['schema']=='editing_mixed1750k_checkpoint_v1' and not saved['technical_only']
        assert saved['optimizer_steps']==step and saved['new_optimizer_steps']==saved['next_update_index']==step-50000
        assert saved['configured_train_pairs']==1750000 and saved['physical_world_size']==3
        assert saved['optimizer_parameter_names']==names and list(saved['state'])==names
        assert saved['plan_sha256']==sha(run/'PLAN.json') and saved['train_index']==plan['dataset']
        assert saved['frozen_parameters']==frozen and saved['initial_parameter_sha256']==initial
        assert saved['initial_checkpoint']==base and saved['generation_initialization']==plan['generation_initialization']
        assert all(tuple(v.shape)==shapes[k] and bool(torch.isfinite(v).all()) for k,v in saved['state'].items())
        assert parameter_hash(saved['state'])==saved['final_parameter_sha256']==ref['final_parameter_sha256']
        optimizer=saved['optimizer'];assert len(optimizer['state'])==201 and len(optimizer['param_groups'])==1
        assert optimizer['param_groups'][0]['params']==list(range(201))
        for i,item in optimizer['state'].items():
            assert float(item['step'])==step
            for key in ['exp_avg','exp_avg_sq']:
                assert tuple(item[key].shape)==shapes[names[i]] and torch.isfinite(item[key]).all()
        assert len(saved['rng_by_rank'])==len(saved['input_chains_by_rank'])==3
        assert all(set(x)=={'data','noise'} and all(len(h)==64 for h in x.values()) for x in saved['input_chains_by_rank'])
        assert saved['new_pair_exposures']==int(exposures[step-50000]) and saved['new_run_seen_unique_pairs']==int(unique[step-50000])
        result=dict(status='PASS_COMPLETE_CHECKPOINT_STATE_PROVENANCE_AND_RECOVERY',checkpoint=ref,
            trainable_parameters=319314336,trainable_tensors=201,all_optimizer_steps=step,
            model_adam_rng_cursor_present=True,fresh_process_bitwise_resume_claimed=False,verified_at=now())
        write(directory/'VERIFIED.json',result);results.append(result);del saved
    write(run/'CHECKPOINTS_VERIFIED.json',dict(status='PASS_ALL_SIX_CONTINUATION_CHECKPOINTS',results=results,verified_at=now()))

if __name__=='__main__':verify()
