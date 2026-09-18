"""Run eight-GPU training segments, with eight-GPU validation between them."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases,read,write,stop_owned
from scripts.t2a.rl.launch_editing_opsd_selective import sha
from scripts.t2a.rl.launch_editing_opsd_fresh2000 import environment,verify_inputs


def training_command(run,checkpoint,until,*,resize=False):
    command=[sys.executable,'-m','torch.distributed.run','--nnodes=1',
        '--nproc_per_node=8','--master_port=29661',str(ROOT/'scripts/t2a/rl/train_editing_opsd_eight_gpu.py'),
        '--config',str(run/'config.json'),'--resume',str(checkpoint),'--limit-updates',str(until),
        '--evaluate-at-updates','250','750','1250','1750']
    if resize:command.append('--resize')
    return command


def evaluation_command(run,checkpoint,step,*,preflight=False):
    command=[sys.executable,'-m','torch.distributed.run','--nnodes=1',
        '--nproc_per_node=8','--master_port=29662',str(ROOT/'scripts/t2a/rl/evaluate_editing_opsd_eight_gpu.py'),
        '--config',str(run/('preflight.json' if preflight else 'config.json')),
        '--checkpoint',str(checkpoint),'--checkpoint-sha256',sha(checkpoint),'--step',str(step),
        '--planner','old' if preflight else 'new','--executor','old' if preflight else 'new',
        '--output',str(run/'preflight_evaluation' if preflight else run/'evaluations'/f'candidate{step}'),
        '--fresh-matched-panel','--no-save-audio','--full-native-validation']
    return command


def audit_preflight(run):
    import numpy as np
    from scripts.t2a.rl.report_editing_opsd_eight_gpu import load_evaluation
    _,_,baseline=load_evaluation(run,'original40k')
    q=read(run/'preflight.json');directory=run/'preflight_evaluation'
    complete=read(directory/'COMPLETE.json')
    if complete['phase']!='COMPLETE' or complete['world_size']!=8:
        raise ValueError('Missing eight-worker evaluation completion.')
    rows=[r for rank in range(8) for r in read(directory/f'eval_step000000_rank{rank}.json')['rows']]
    expected={(i,s) for i in q['validation_ordinals'] for s in q['evaluation_seeds']}
    if len(rows)!=len(expected) or {(r['ordinal'],r['seed']) for r in rows}!=expected:
        raise ValueError('Preflight output coverage changed.')
    for row in rows:
        old=baseline[row['ordinal'],row['seed']]
        if row['audio'] is not None or row['plan']!=old['plan'] or row['scalar']!=old['scalar']:
            raise ValueError('Eight-worker generation/scalars differ from measured four-worker baseline.')
        with np.load(row['features'],allow_pickle=False) as a,np.load(old['features'],allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
                raise ValueError('Eight-worker per-output features differ from four-worker baseline.')
    reference=read(Path(read(run/'PROTOCOL.json')['baseline_reference']['directory'])/'native_validation/full_step000000.json')
    current=read(directory/'native_validation/full_step000000.json')
    if current['rows']!=20000 or current['speech_rows']!=11918:
        raise ValueError('Native validation coverage changed.')
    if any(not math.isclose(current['metrics'][k],v,rel_tol=1e-8,abs_tol=1e-9) for k,v in reference['metrics'].items()):
        raise ValueError('Canonical native validation does not reproduce baseline diagnostics.')
    write(run/'EVALUATION_ACCEPTANCE.json',dict(phase='PASS',world_size=8,matched_outputs=len(rows),
        plans_scalars_and_all_feature_arrays_bitwise_equal=True,full_native_rows=20000,
        native_RF_noises_match_original_four_rank_batches=True,native_metrics=current['metrics'],
        native_seconds=current['seconds'],preflight_seconds=complete['elapsed_seconds'],at_unix=time.time()))


def audit_startup(run):
    import torch
    from scripts.t2a.rl.train_editing_opsd_eight_gpu import validate_state
    p=read(run/'PROTOCOL.json');q=dict(read(run/'config.json'),config_path=str(run/'config.json'))
    out=Path(q['output']);step=p['transition']['step'];until=step+4
    state=torch.load(out/'resume_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    validate_state(q,state)
    if state['step']!=until:
        raise ValueError('Four actual expanded updates were required for startup.')
    original=torch.load(p['transition']['path'],map_location='cpu',weights_only=False,mmap=True)
    migrated=torch.load(out/'migration_start.pt',map_location='cpu',weights_only=False,mmap=True)
    validate_state(q,migrated)
    if migrated['step']!=step or migrated['model_sha256']!=original['model_sha256']:
        raise ValueError('Protected migration state changed the original model.')
    if migrated['optimizer']['param_groups']!=original['optimizer']['param_groups']:
        raise ValueError('Migration changed Adam parameter groups.')
    for key,old in original['optimizer']['state'].items():
        new=migrated['optimizer']['state'][key]
        if set(old)!=set(new) or any(not torch.equal(old[k],new[k]) for k in old):
            raise ValueError('Migration changed an Adam step or moment tensor.')
    del original,migrated
    timings={};requests=[];paired=[];first=[]
    for rank in range(8):
        proof=read(out/f'startup_checks/FIRST_UPDATE_{step+1:06d}_rank{rank}.json')
        if not proof['samples_match_saved_cursors']:
            raise ValueError('First expanded update skipped or repeated samples.')
        expected=read(run/'EXPECTED_FIRST_UPDATE.json')['ranks'][str(rank)]
        if {k:proof[k] for k in ('step','request_ordinals','paired_ordinals')}!=expected:
            raise ValueError('GPU migration differs from independent saved-frontier calculation.')
        first.append(dict(rank=rank,first_update_matches=True))
        requests+=proof['request_ordinals'];paired+=proof['paired_ordinals']
        rows=[json.loads(line) for line in (out/f'UPDATES_rank{rank}.jsonl').read_text().splitlines()]
        rows=list({row['step']:row for row in rows if step<row['step']<=until}.values())
        if len(rows)!=4 or any(row['extra_objectives']['selected_rows']!=1 for row in rows):
            raise ValueError('Expanded updates changed the global decoded auxiliary budget.')
        for row in rows:timings[row['step']]=max(timings.get(row['step'],0),row['performance']['step_seconds'])
    if len(requests)!=16 or len(set(requests))!=16 or len(paired)!=512 or len(set(paired))!=512:
        raise ValueError('Global batch coverage changed.')
    mean=statistics.mean(timings.values());old=p['four_gpu_timing']['last50_mean_seconds']
    write(run/'STARTUP_ACCEPTANCE.json',dict(phase='PASS',step=until,transition_step=step,
        world_size=8,trainable_tensors=len(state['model']),complete_Adam_preserved=True,
        migration_Adam_moments_bitwise_equal=True,
        global_request_batch=16,global_paired_batch=512,global_decoded_rows=8,ranks=first,
        seconds_by_update=timings,mean_seconds=mean,four_gpu_last50_mean_seconds=old,
        observed_speedup=old/mean,scope='Four measured updates; not a long-run speed or quality guarantee.',
        model_sha256=state['model_sha256'],at_unix=time.time()))


class Paused(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--recover',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();run=args.run_dir.resolve();p=verify_inputs(run);q=read(run/'config.json')
    out=Path(q['output']);transition=p['transition'];checkpoint=Path(transition['path'])
    if args.dry_run:
        print(json.dumps(dict(startup=training_command(run,checkpoint,transition['step']+4,resize=True),
            evaluation_candidates=list(range(250,2001,250)),all_gpus=list(range(8)))));return
    if (run/'STATUS.json').exists():
        if not args.recover:raise ValueError('Use explicit --recover for an existing run.')
        previous=read(run/'STATUS.json');cmdline=Path(f'/proc/{previous["pid"]}/cmdline')
        if cmdline.exists() and str(Path(__file__).resolve()).encode() in cmdline.read_bytes():
            raise ValueError('Previous driver is still active.')
        archive=run/'attempts';archive.mkdir(exist_ok=True)
        write(archive/f'STATUS_{time.time_ns()}.json',previous)
    if sha(checkpoint)!=transition['sha256']:
        raise ValueError('Protected migration checkpoint changed.')
    completed=read(run/'EVALUATION_PROGRESS.json')['completed'] if (run/'EVALUATION_PROGRESS.json').exists() else []
    children={};monitor={};logs=[];leases=[];failures=[]
    status=dict(pid=os.getpid(),phase='PREPARING',started_unix=time.time(),physical_gpus=list(range(8)),
        origin='fresh original40k',transition_step=transition['step'],target=2000,training_step=transition['step'],
        completed_evaluations=completed,allocated_GPUs=0,training_output=str(out))

    def run_worker(name,command,*,training=False):
        verify_inputs(run)
        if (run/'PAUSE.json').exists():raise Paused('Pause requested before next stage.')
        log=(run/(name+'.log')).open('a');logs.append(log)
        worker=subprocess.Popen(command,cwd=ROOT,env=environment(range(8)),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            pass_fds=tuple(h.fileno() for h in leases))
        children['worker']=worker
        status.update(phase=name,worker_pid=worker.pid,command=command,allocated_GPUs=8)
        write(run/'STATUS.json',status)
        while worker.poll() is None:
            verify_inputs(run)
            if (run/'PAUSE.json').exists():
                if training:
                    (out/'STOP').touch()
                else:
                    stop_owned(children);raise Paused('Paused evaluation; completed case records remain reusable.')
            f=out/'STATUS_rank0.json'
            if training and f.exists():
                value=read(f)
                if value.get('step') is not None:status['training_step']=value['step']
                status.update(training_phase=value['phase'],training_observed_unix=value['time'])
            status.update(observed_unix=time.time());write(run/'STATUS.json',status)
            time.sleep(10)
        code=worker.returncode;del children['worker']
        status.update(worker_pid=None,last_exit_code=code);write(run/'STATUS.json',status)
        if (run/'PAUSE.json').exists():raise Paused('Training paused after its current optimizer boundary.')
        return code

    try:
        leases=acquire_leases(p['gpu_pool'])
        log=(run/'GPU_UTILIZATION.csv').open('a');logs.append(log)
        monitor['gpu']=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits','--loop-ms=10000'],stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        if not (run/'EVALUATION_ACCEPTANCE.json').exists():
            code=run_worker('EVALUATION_PREFLIGHT',evaluation_command(run,Path(p['original_zero']['path']),0,preflight=True))
            if code:raise RuntimeError(f'Evaluation preflight failed: {code}')
            audit_preflight(run)
        step=transition['step'];resize=True
        if (out/'resume_latest.pt').exists():
            checkpoint=out/'resume_latest.pt';step=read(out/'RESUME.json')['step'];resize=False
        if not (run/'STARTUP_ACCEPTANCE.json').exists():
            if step<transition['step']+4:
                code=run_worker('EIGHT_GPU_STARTUP',training_command(run,checkpoint,transition['step']+4,resize=resize),training=True)
                if code:raise RuntimeError(f'Eight-GPU startup failed: {code}')
            audit_startup(run)
            checkpoint=out/'resume_latest.pt';step=transition['step']+4;resize=False
        while True:
            pending=([s for s in range(250,2001,250) if s not in completed] if step>=2000 else
                     ([step] if step%250==0 and step>0 and step not in completed else []))
            for evaluation_step in pending:
                candidate=out/'evaluation_candidates'/f'step-{evaluation_step:08d}.pt'
                error=None
                for attempt in range(2):
                    code=run_worker(f'EVAL_{evaluation_step:06d}',evaluation_command(run,candidate,evaluation_step))
                    if not code:
                        try:
                            from scripts.t2a.rl.report_editing_opsd_eight_gpu import promote
                            promote(run,f'candidate{evaluation_step}')
                            error=None;break
                        except Exception as exc:error=repr(exc)
                    else:error=f'worker exited {code}'
                if error:
                    failures.append(dict(step=evaluation_step,error=error))
                    write(run/'EVALUATION_FAILURES.json',dict(failures=failures))
                else:
                    completed.append(evaluation_step)
                    write(run/'EVALUATION_PROGRESS.json',dict(completed=completed,at_unix=time.time()))
                status.update(completed_evaluations=list(completed),evaluation_failures=failures)
            if step>=2000:break
            until=min((step//250+1)*250,2000)
            code=run_worker(f'TRAIN_TO_{until:06d}',training_command(run,checkpoint,until),training=True)
            if code:raise RuntimeError(f'Training to{until} failed: {code}')
            step=read(out/'RESUME.json')['step'];checkpoint=out/'resume_latest.pt'
            if step!=until:raise RuntimeError(f'Training stopped unexpectedly at{step}, before{until}.')
        status.update(phase='COMPLETE' if len(completed)==8 else 'TRAINING_COMPLETE_EVALUATION_NEEDS_RECOVERY',
            training_step=2000,completed_unix=time.time(),allocated_GPUs=0,
            top5=str(run/'selection/top_checkpoints/INDEX.json'),report=str(run/'TABLE.md'))
        write(run/'STATUS.json',status)
    except Paused as exc:
        stop_owned(children)
        status.update(phase='PAUSED',reason=str(exc),allocated_GPUs=0,observed_unix=time.time())
        write(run/'STATUS.json',status)
    except BaseException as exc:
        stop_owned(children)
        status.update(phase='FAILED',error=repr(exc),allocated_GPUs=0,observed_unix=time.time())
        write(run/'STATUS.json',status)
        raise
    finally:
        stop_owned(monitor)
        for h in leases+logs:h.close()


if __name__=='__main__':
    main()
