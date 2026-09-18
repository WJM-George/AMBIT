"""Discarded real-model probes before starting the branch-budget experiment."""
import copy
import dataclasses
import hashlib
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from stable_audio_tools.training.transfusion_opsd import branch_request_supervision as branch


def rng_state(device):
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(device)


def restore_rng(state, device):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3], device)


def digest(value):
    h=hashlib.sha256()
    def update(x):
        h.update(type(x).__name__.encode())
        if torch.is_tensor(x):
            h.update(str((tuple(x.shape),x.dtype)).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif dataclasses.is_dataclass(x):
            for field in dataclasses.fields(x):update(field.name);update(getattr(x,field.name))
        elif isinstance(x,dict):
            for key in sorted(x,key=str):update(key);update(x[key])
        elif isinstance(x,(list,tuple)):
            for v in x:update(v)
        elif x is None or isinstance(x,(str,int,float,bool)):
            h.update(repr(x).encode())
        else:
            raise TypeError('Unsupported request probe value: '+repr(type(x)))
    update(value)
    return h.hexdigest()


def check_gradients(learner, ordinal):
    from scripts.t2a.experiments.ar_structured_v1 import data,losses
    row=learner.requests.row(ordinal)
    batch=data.collate([learner.paired[ordinal]],pad_id=learner.adapter.codec.pad_id,joint=True)
    ar,target,metadata,mask=learner.native._move_joint_batch(batch,learner.device)
    den=torch.stack(((ar['plan_labels']!=-100).sum(),mask.sum()*target.shape[1],
                     mask.new_tensor(1,dtype=torch.long))).double()
    names=['ar.plan_adapter.plan_head.weight','ar.editing_dit.postprocess_conv.weight',
           'ar.editing_dit.transformer.layers.0.pre_norm.gamma']
    parameters=[learner.trainable[n] for n in names]
    records=[]
    learner.adapter.train()
    for gt_ar,gt_rf in [(True,True),(True,False),(False,True)]:
        route=dict(execution_AR=not gt_ar,execution_RF=not gt_rf,
                   execution_joint=not(gt_ar or gt_rf),execution_RF_mass=float(not gt_rf))
        cfg=copy.deepcopy(learner.cfg)
        cfg['seed']=branch.request_seed(cfg,learner.step,row['pair_id'])
        cfg['loss_weights'].update(AR=float(gt_ar),RF=float(gt_rf),
                                  **{k:0. for k in losses.DEFAULT_LOSS_WEIGHTS})
        state=rng_state(learner.device)
        learner.optimizer.zero_grad(set_to_none=True)
        reference,sums,_,_=learner.native.batch_loss(learner.adapter,learner.teacher,batch,
            cfg,learner.device,0,0,1,den)
        reference=reference*.5
        reference_value=float(reference.detach())
        reference_grads=torch.autograd.grad(reference,parameters,allow_unused=True)
        reference_grads=[None if g is None else g.detach().clone() for g in reference_grads]
        del reference,sums
        restore_rng(state,learner.device)
        actual,report=branch.request_loss(learner.native,learner.adapter,learner.teacher,batch,
            learner.cfg,learner.device,row=row,step=learner.step,scale=.5,weight=1.,route=route)
        torch.testing.assert_close(actual.detach(),actual.new_tensor(reference_value),rtol=1e-6,atol=1e-6)
        actual_value=float(actual.detach())
        # This backward contains only fallback, before any 512-row paired loss.
        actual.backward()
        expected=set(branch.gradient_parameters(report))
        norms={}
        for name,param,ref in zip(names,parameters,reference_grads):
            grad=param.grad
            norms[name]=None if grad is None else float(grad.detach().float().norm())
            if name in expected:
                if grad is None or not bool(torch.isfinite(grad).all()) or norms[name]<=0:
                    raise RuntimeError('An isolated GT branch missed its required gradient: '+name)
            elif grad is not None and bool((grad!=0).any()):
                raise RuntimeError('An unselected GT branch received a gradient: '+name)
            if grad is None:
                if ref is not None and bool((ref!=0).any()):raise RuntimeError('Native gradient was lost.')
            else:
                torch.testing.assert_close(grad,ref if ref is not None else torch.zeros_like(grad),rtol=.003,atol=1e-6)
        records.append(dict(GT_AR=gt_ar,GT_RF=gt_rf,ordinal=ordinal,operation=row['operation'],
            native_loss=reference_value,branch_loss=actual_value,gradient_norms=norms,
            native_formula_and_gradient_match=True,coverage=branch.covered_request(route,report)))
        del actual,reference_grads
        learner.optimizer.zero_grad(set_to_none=True)
    learner.adapter.eval()
    return records


def profile_collection(learner, ordinals):
    trials=[]
    initial_rng=rng_state(learner.device)
    initial_costs=copy.deepcopy(learner.costs)
    expected=None
    for enabled in (False,True):
        restore_rng(initial_rng,learner.device)
        learner.costs=copy.deepcopy(initial_costs)
        learner.request_cache_enabled=enabled
        learner.forward_cache.clear()
        learner.progress('PROFILE_REQUEST_REUSE',enabled=enabled)
        records=[];elapsed=0.
        torch.cuda.empty_cache();dist.barrier();torch.cuda.synchronize(learner.device)
        for ordinal in ordinals:
            start=time.perf_counter()
            item=learner.collect(ordinal)
            torch.cuda.synchronize(learner.device)
            elapsed+=time.perf_counter()-start
            records.append(dict(ordinal=ordinal,
                fingerprint=digest({k:v for k,v in item.items() if k!='evaluation_cache'}),
                cache=item.get('evaluation_cache')))
            del item
        fingerprints=[r['fingerprint'] for r in records]
        if expected is None:expected=fingerprints
        match=expected==fingerprints
        gathered=learner.gather(dict(rank=learner.rank,seconds=elapsed,exact_outputs=match,requests=records))
        trials.append(dict(enabled=enabled,seconds=max(r['seconds'] for r in gathered),ranks=gathered,
                           exact_outputs=all(r['exact_outputs'] for r in gathered)))
        if not trials[-1]['exact_outputs']:
            raise RuntimeError('Request reuse changed a native plan, execution, quality gate or teacher target.')
    learner.costs=initial_costs
    # Do not enable a cache that makes the measured slowest rank slower.
    selected=trials[1]['seconds']<trials[0]['seconds']*.98
    learner.request_cache_enabled=selected
    return dict(selected=selected,trials=trials,
                collection_speedup=trials[0]['seconds']/trials[1]['seconds'],
                scope='same16 requests, unchanged weights/noises/kernels/gates, all output tensors exact')


def validate_runtime(learner):
    from scripts.t2a.rl.train_editing_opsd_throughput import peek,base
    state=rng_state(learner.device)
    costs=copy.deepcopy(learner.costs)
    streams=(learner.request_stream.state_dict(),learner.pair_stream.state_dict())
    before=base.tensor_digest(learner.trainable)
    training=learner.adapter.training
    ordinals=peek(learner.request_stream,learner.q['request_rows_per_rank'])
    try:
        learner.progress('ISOLATED_BRANCH_GRADIENT_ACCEPTANCE')
        gradients=check_gradients(learner,ordinals[0])
        collection=profile_collection(learner,ordinals)
    finally:
        learner.optimizer.zero_grad(set_to_none=True)
        learner.forward_cache.clear()
        restore_rng(state,learner.device)
        learner.costs=costs
        learner.adapter.train(training)
    if before!=base.tensor_digest(learner.trainable) or learner.optimizer.state:
        raise RuntimeError('A discarded branch probe changed weights or Adam.')
    if streams!=(learner.request_stream.state_dict(),learner.pair_stream.state_dict()):
        raise RuntimeError('A discarded branch probe consumed training rows.')
    report=dict(phase='PASS',rank=learner.rank,step=learner.step,branch_gradients=gradients,
        performance=collection,model_sha256=before,weights_Adam_RNG_and_samplers_unchanged=True,
        isolated_fallback_only=True,at_unix=time.time())
    base.write(learner.performance_dir/f'BRANCH_RUNTIME_rank{learner.rank}.json',report)
    torch.cuda.empty_cache()
    return report
