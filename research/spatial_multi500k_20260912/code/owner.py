"""Resume the authorized latent-only data generation only after AR and Swan."""
from common import *
import fcntl
import subprocess
import time
import traceback


def main():
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runtime=read(ROOT/'RUNTIME_PLAN.json')
    for path,value in runtime['source_sha256'].items():assert sha(path)==value,path
    assert read(runtime['after'])['status']=='COMPLETE_OFFICIAL_ENTRY_ORIGINAL_TEST5000'
    assert read(runtime['AR_paper_after'])['status']=='PASS_AR_PAPER_TABLES_PUBLISHED'
    if (ROOT/'DATA_READY.json').exists():return
    while not (ROOT/'reviews/CPU_RENDER_REVIEW.json').exists():
        if (ROOT/'reviews/CPU_FAILURE.json').exists():raise RuntimeError('CPU review failed; see reviews/CPU_FAILURE.json')
        state('waiting_for_CPU_review');time.sleep(10)
    assert read(ROOT/'reviews/CPU_RENDER_REVIEW.json')['status']=='PASS_METADATA_AND_BYTE_EXACT_SOURCE_RENDER_CANARIES'
    for split in COUNTS:
        assert sha(ROOT/'pair_index'/f'{split}.sqlite')==runtime['pair_indices'][split]['sha256']
    # Reopen every existing completed shard once on resume, before workers skip it.
    from worker import verify_manifest
    completed=sorted((ROOT/'materialized/done').glob('*/*.json'))
    verified={split:0 for split in COUNTS};receipts={}
    state('verifying_completed_shards',shards=len(completed))
    for p in completed:
        done=read(p);receipt=verify_manifest(done['manifest'])
        assert receipt['manifest_sha256']==done['manifest_sha256']
        assert receipt['rows']==done['rows'] and receipt['latent_sha256']==done['latent_sha256']
        assert done['status']=='PASS_LATENT_ONLY_SHARD'
        assert done['split']==p.parent.name and done['work_shard']==int(p.stem)
        verified[done['split']]+=receipt['rows'];receipts[str(p)]=sha(p)
    write(ROOT/'reviews'/f'RESUME_VERIFICATION_{time.time_ns()}.json',dict(
        status='PASS_ALL_COMPLETED_LATENT_SHARDS_ON_RESUME',counts=verified,
        shards=len(completed),done_sha256=receipts,checked_at=now()))
    children=[];env=cpu_env()
    for physical in GPUS:
        command=[PYTHON,str(LAUNCHER),'--gpus',str(physical),'--',PYTHON,str(ROOT/'code/gpu.py'),'--physical-gpu',str(physical)]
        log=ROOT/'logs'/f'GPU{physical}_{time.time_ns()}.log'
        with log.open('xb') as f:p=subprocess.Popen(command,env=env,cwd=ROOT/'code',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        children.append((p,log,physical))
    state('GPU_benchmark_then_latent_materialization',children=[dict(pid=p.pid,physical_gpu=g,log=str(log)) for p,log,g in children])
    while any(p.poll() is None for p,_,_ in children):
        failures=[(g,p.returncode,str(log)) for p,log,g in children if p.poll() not in [None,0]]
        if failures:
            for p,_,_ in children:
                if p.poll() is None:p.terminate()
            for p,_,_ in children:p.wait()
            raise RuntimeError(str(failures))
        time.sleep(5)
    assert all(p.returncode==0 for p,_,_ in children)
    for split in COUNTS:
        count=sum(read(p)['rows'] for p in (ROOT/'materialized/done'/split).glob('*.json'))
        assert count==COUNTS[split],(split,count)
    write(ROOT/'MATERIALIZATION_COMPLETE.json',dict(status='COMPLETE_LATENT_MATERIALIZATION',counts=COUNTS,completed_at=now()))
    state('final_inventory_and_training_reader_verification')
    subprocess.run([PYTHON,str(ROOT/'code/finalize.py')],env=env,cwd=ROOT/'code',check=True)
    state('COMPLETE',result=str(ROOT/'DATA_READY.json'))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/'OWNER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),observed_at=now(),pid=os.getpid()))
        raise
