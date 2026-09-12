"""One physical GPU lease: measure one/two workers, replay latents, then produce."""
from common import *
import argparse
import fcntl
import signal
import subprocess
import time
import traceback


def children_run(commands,root):
    root.mkdir(parents=True,exist_ok=True);processes=[];begin=time.monotonic()
    for slot,command in enumerate(commands):
        log=root/f'slot{slot}.log'
        with log.open('xb') as f:
            p=subprocess.Popen(command,cwd=ROOT/'code',env=os.environ.copy(),stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((p,log))
    try:
        while any(p.poll() is None for p,_ in processes):
            bad=[(p.returncode,str(log)) for p,log in processes if p.poll() not in [None,0]]
            if bad:raise RuntimeError(str(bad))
            time.sleep(1)
        assert all(p.returncode==0 for p,_ in processes),[(p.returncode,str(log)) for p,log in processes]
    finally:
        for p,_ in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,_ in processes:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    return time.monotonic()-begin


def benchmark(physical,slots):
    root=ROOT/'reviews/gpu_benchmark'/f'GPU{physical}'/f'workers{slots}_{time.time_ns()}'
    root.mkdir(parents=True);commands=[]
    selected=[('train',(physical-2)*4+i) for i in range(4)]
    for slot in range(slots):
        plan=root/f'jobs_slot{slot}.json';write(plan,selected[slot::slots])
        commands.append([PYTHON,str(ROOT/'code/worker.py'),'--physical-gpu',str(physical),
            '--render-workers',str(24//slots),'--slot',str(slot),'--bench-list',str(plan),'--output-root',str(root)])
    elapsed=children_run(commands,root/'logs')
    states=[read(p) for p in (root/'materialized/workers').glob('*.json')]
    assert len(states)==slots and all(s['stage']=='COMPLETE' for s in states)
    rows=sum(s['completed_rows'] for s in states);assert rows==4*SHARD_SIZE
    measured=max(s['finished_monotonic'] for s in states)-min(s['model_ready_monotonic'] for s in states)
    result=dict(root=str(root),workers=slots,rows=rows,total_seconds=elapsed,after_model_ready_seconds=measured,
        pairs_per_hour=rows/measured*3600,states=states)
    write(root/'RESULT.json',result);return result


def numerical_compare(one,two):
    import numpy as np
    from safetensors import safe_open
    import pyarrow.parquet as pq
    a=Path(one['root']);b=Path(two['root']);maximum_nmse=0.;maximum_abs=0.;n=0
    for p in sorted((a/'materialized/manifests/train').glob('*.parquet')):
        other=b/'materialized/manifests/train'/p.name
        rows=pq.read_table(p).to_pylist();rr=pq.read_table(other).to_pylist()
        assert len(rows)==len(rr)
        with safe_open(rows[0]['target_latent_ref'].rsplit('#',1)[0],framework='np') as x, safe_open(rr[0]['target_latent_ref'].rsplit('#',1)[0],framework='np') as y:
            for r,s in zip(rows,rr):
                assert r['pair_id']==s['pair_id'] and r['target_foa_sha256']==s['target_foa_sha256']
                assert r['vae_encode_seed']==s['vae_encode_seed']
                u=x.get_tensor(r['target_sample_id']).astype(np.float64);v=y.get_tensor(s['target_sample_id']).astype(np.float64)
                nmse=float(np.sum((u-v)**2)/max(np.sum(u*u),1e-30));delta=float(np.max(np.abs(u-v)))
                assert np.isfinite(u).all() and np.isfinite(v).all()
                assert nmse<=1e-6 and np.allclose(u,v,atol=.002,rtol=.002),(r['pair_id'],nmse,delta)
                maximum_nmse=max(maximum_nmse,nmse);maximum_abs=max(maximum_abs,delta);n+=1
    assert n==4*SHARD_SIZE
    return dict(status='PASS_ONE_VS_TWO_WORKER_PCM_AND_LATENT_REPLAY',rows=n,max_latent_nmse=maximum_nmse,
        max_abs_latent_difference=maximum_abs,PCM_sha256='exact',atol=.002,rtol=.002,nmse_limit=1e-6)


def checked_decision(path,runtime):
    decision=read(path)
    assert decision['status']=='PASS_MEASURED_WORKER_SELECTION'
    if decision['runtime_plan_sha256']!=sha(ROOT/'RUNTIME_PLAN.json'):
        # A reservation-only repair may retain the original, unmodified numeric
        # and throughput measurements. The new plan binds their exact bytes and
        # records the compatibility review; never relabel them as a new run.
        evidence=runtime.get('reused_gpu_benchmarks',{}).get(str(path))
        assert evidence and sha(path)==evidence['benchmark_sha256'],str(path)
        assert decision['runtime_plan_sha256']==evidence['measured_runtime_sha256']
    return decision


def main():
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}: stop owned GPU workers')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--physical-gpu',type=int,required=True);a=p.parse_args();gpu=a.physical_gpu
    assert gpu in GPUS
    sys.path.insert(0,str(OLD_RUN));from editing_allocated_gpu_runtime_v1 import gpu_topology,verify_launcher_leases
    topology=gpu_topology();assert topology['physical_indices']==[gpu];verify_launcher_leases(topology)
    assert read(ROOT/'reviews/CPU_RENDER_REVIEW.json')['status']=='PASS_METADATA_AND_BYTE_EXACT_SOURCE_RENDER_CANARIES'
    runtime=read(ROOT/'RUNTIME_PLAN.json')
    for path,value in runtime['source_sha256'].items():assert sha(path)==value,path
    dest=ROOT/'reviews'/f'GPU{gpu}_BENCHMARK.json'
    if dest.exists():
        decision=checked_decision(dest,runtime)
    else:
        one=benchmark(gpu,1);two=None;numerical=None;failure=None;slots=1
        try:
            two=benchmark(gpu,2);numerical=numerical_compare(one,two)
            if two['pairs_per_hour']>=1.05*one['pairs_per_hour']:slots=2
        except Exception as e:failure=dict(error=repr(e),traceback=traceback.format_exc())
        decision=dict(status='PASS_MEASURED_WORKER_SELECTION',physical_gpu=gpu,chosen_workers=slots,
            one_worker=one,two_workers=two,numerical_review=numerical,two_worker_fallback_reason=failure,
            performance_rule='Use two only if numeric replay passes and measured throughput improves by at least 5 percent',
            runtime_plan_sha256=sha(ROOT/'RUNTIME_PLAN.json'),observed_at=now())
        write(dest,decision)
    # Production cannot overlap another GPU's benchmark of frozen fixture rows.
    while not all((ROOT/'reviews'/f'GPU{i}_BENCHMARK.json').exists() for i in GPUS):time.sleep(5)
    for i in GPUS:
        checked_decision(ROOT/'reviews'/f'GPU{i}_BENCHMARK.json',runtime)
    slots=decision['chosen_workers'];commands=[]
    for slot in range(slots):
        commands.append([PYTHON,str(ROOT/'code/worker.py'),'--physical-gpu',str(gpu),
            '--render-workers',str(24//slots),'--slot',str(slot)])
    root=ROOT/'logs'/f'production_GPU{gpu}_{time.time_ns()}'
    children_run(commands,root)
    write(ROOT/'materialized/workers'/f'GPU{gpu}_COMPLETE.json',dict(status='COMPLETE_PRODUCTION_WORKERS',physical_gpu=gpu,completed_at=now()))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/'reviews'/f'GPU_FAILURE_{os.getpid()}.json',dict(error=repr(e),traceback=traceback.format_exc(),observed_at=now()))
        raise
