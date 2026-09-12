"""A persistent VAE worker with a bounded CPU render pool and one prefetched shard."""
from common import *
from render import render_one,init_cpu,encode
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import argparse
import copy
import fcntl
import multiprocessing as mp
import time
import traceback


def jobs():
    return [(split,shard) for split in ['validation','test','train']
            for shard in range((COUNTS[split]+SHARD_SIZE-1)//SHARD_SIZE)]


def load_jobs(split,shard):
    con=db(ROOT/'pair_index'/f'{split}.sqlite')
    rows=[dict(r) for r in con.execute('SELECT * FROM pairs WHERE work_shard=? ORDER BY row_in_shard',(shard,))]
    out=[]
    for r in rows:
        a=json.loads(con.execute('SELECT audit_json FROM edit_actions WHERE pair_ordinal=?',(r['pair_ordinal'],)).fetchone()[0])
        out.append((r,a,False))
    con.close();assert len(out)==min(SHARD_SIZE,COUNTS[split]-SHARD_SIZE*shard)
    return out


def render_shard(pool,job):
    tick=time.monotonic();results=list(pool.map(render_one,load_jobs(*job),chunksize=1))
    assert len(results)==len({r[0]['pair_id'] for r in results})
    return results,time.monotonic()-tick


def redirected(rows,results,root):
    from mutations import DIGEST_FIELDS,sha256_json
    out=[];rr=[]
    for r,s in zip(rows,results):
        r=dict(r);s=dict(s)
        p=root/'materialized/latents'/r['split']/f'latents-{r["split"]}-{r["work_shard"]:05d}.safetensors'
        r.update(target_latent_path=str(p),target_latent_ref=f'{p}#{r["target_latent_key"]}')
        r['pair_record_sha256']=sha256_json({k:r[k] for k in DIGEST_FIELDS});s['pair_record_sha256']=r['pair_record_sha256']
        out.append(r);rr.append(s)
    return out,rr


def verify_manifest(path):
    import pyarrow.parquet as pq
    from safetensors import safe_open
    import torch
    from scripts.t2a.data.materialize_sceneplan_transfusion_editing_targets import _tensor_sha256
    rows=pq.read_table(path).to_pylist();assert rows
    latent=rows[0]['target_latent_ref'].rsplit('#',1)[0]
    assert sha(latent)==rows[0]['target_latent_shard_sha256']
    with safe_open(latent,framework='pt',device='cpu') as f:
        assert set(f.keys())=={r['target_sample_id'] for r in rows}
        for r in rows:
            assert r['target_foa_path'] is None
            x=f.get_tensor(r['target_sample_id'])
            assert x.dtype==torch.float16 and tuple(x.shape)==(64,r['latent_frames_valid']) and torch.isfinite(x).all()
            assert _tensor_sha256(x)==r['target_latent_tensor_sha256']
    return dict(rows=len(rows),manifest=str(path),manifest_sha256=sha(path),latent_path=latent,
        latent_sha256=rows[0]['target_latent_shard_sha256'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--render-workers',type=int,required=True);p.add_argument('--slot',type=int,default=0)
    p.add_argument('--bench-list');p.add_argument('--output-root');a=p.parse_args()
    assert a.physical_gpu in GPUS and 1<=a.render_workers<=24
    sys.path.insert(0,str(OLD_RUN))
    from editing_allocated_gpu_runtime_v1 import gpu_topology,verify_launcher_leases
    topology=gpu_topology();assert topology['physical_indices']==[a.physical_gpu];verify_launcher_leases(topology)
    import torch
    assert not torch.cuda.is_initialized();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    from scripts.t2a.data.materialize_sceneplan_transfusion_editing_targets import load_vae
    model=load_vae(device)
    root=Path(a.output_root) if a.output_root else ROOT
    benchmark=bool(a.bench_list)
    work=[tuple(x) for x in read(a.bench_list)] if benchmark else jobs()
    label=f'gpu{a.physical_gpu}_slot{a.slot}_{os.getpid()}'
    logroot=root/'materialized/workers';logroot.mkdir(parents=True,exist_ok=True)
    candidates=iter(work);completed=0;started=time.monotonic();claimed=[]
    def claim():
        for split,shard in candidates:
            done=root/'materialized/done'/split/f'{shard:05d}.json'
            if done.exists():continue
            lockpath=root/'materialized/locks'/split/f'{shard:05d}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
            lock=lockpath.open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            if done.exists():lock.close();continue
            claimed.append(lock);return (split,shard),lock,done
        return None
    try:
        with ProcessPoolExecutor(max_workers=a.render_workers,mp_context=mp.get_context('spawn'),initializer=init_cpu) as pool,ThreadPoolExecutor(max_workers=1) as feeder:
            current=claim();future=feeder.submit(render_shard,pool,current[0]) if current else None
            while current is not None:
                values,render_seconds=future.result();rowset=[v[0] for v in values];results=[v[2] for v in values]
                next_job=claim();next_future=feeder.submit(render_shard,pool,next_job[0]) if next_job else None
                write(logroot/(label+'.json'),dict(stage='encoding',physical_gpu=a.physical_gpu,pid=os.getpid(),
                    job=current[0],completed_rows=completed,render_workers=a.render_workers,observed_at=now()))
                if benchmark:rowset,results=redirected(rowset,results,root)
                tick=time.monotonic();manifest=encode(rowset,results,root,device,model);encode_seconds=time.monotonic()-tick
                receipt=verify_manifest(manifest)
                receipt.update(status='PASS_LATENT_ONLY_SHARD',split=current[0][0],work_shard=current[0][1],
                    physical_gpu=a.physical_gpu,pid=os.getpid(),render_seconds=render_seconds,encode_seconds=encode_seconds,
                    target_audio_sha256=[r['target_foa_sha256'] for r in results],completed_at=now())
                write(current[2],receipt);completed+=len(rowset)
                current[1].close();claimed.remove(current[1]);current,future=next_job,next_future
                print(canonical(dict(stage='shard_complete',worker=label,completed_rows=completed,
                    render_seconds=render_seconds,encode_seconds=encode_seconds)),flush=True)
                del values,rowset,results
        write(logroot/(label+'.json'),dict(stage='COMPLETE',physical_gpu=a.physical_gpu,pid=os.getpid(),
            completed_rows=completed,model_ready_monotonic=started,finished_monotonic=time.monotonic(),
            elapsed_seconds=time.monotonic()-started,render_workers=a.render_workers,observed_at=now()))
    finally:
        for lock in claimed:lock.close()


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(ROOT/f'materialized/workers/FAILURE_{os.getpid()}.json',dict(error=repr(e),traceback=traceback.format_exc(),observed_at=now()))
        raise
