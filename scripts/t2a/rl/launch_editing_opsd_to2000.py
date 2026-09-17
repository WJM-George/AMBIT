"""Own the500→2000 continuation and independent, no-WAV500x2 validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases, read, write, stop_owned
from scripts.t2a.rl.launch_editing_opsd_selective import sha


def verify_inputs(run):
    p = read(run/'PROTOCOL.json')
    for field in ('sources', 'configurations'):
        for path, expected in p[field].items():
            if sha(path) != expected:
                raise ValueError('Pinned input changed: '+path)
    return p


def environment(gpus):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')


def training_command(run, resume, until):
    return [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
        '--nproc_per_node=4', '--master_port=29651', str(ROOT/'scripts/t2a/rl/train_editing_opsd_to2000.py'),
        '--config', str(run/'config.json'), '--resume', str(resume), '--limit-updates', str(until)]


def audit_startup(run):
    import torch
    from scripts.t2a.rl.train_editing_opsd_to2000 import validate_state
    q = read(run/'config.json'); out = Path(q['output'])
    first = []
    expected = read(run/'EXPECTED_FIRST_UPDATE.json')
    for rank in range(4):
        row = read(out/f'continuation_checks/FIRST_UPDATE_000501_rank{rank}.json')
        if ({k:row[k] for k in ('step','request_ordinals','paired_ordinals')} != expected['ranks'][str(rank)]
                or row['samples_match_saved_cursors'] is not True):
            raise ValueError('First continuation update does not match saved streams.')
        first.append(row)
    saved = torch.load(out/'resume_latest.pt', map_location='cpu', weights_only=False, mmap=True)
    validate_state(q, saved, {sha(run/'config.json')})
    if saved['step'] != 502:
        raise ValueError('Startup must finish exactly two real resumed updates.')
    write(run/'STARTUP_ACCEPTANCE.json', dict(phase='PASS', step=502, first_update=first,
        model_sha256=saved['model_sha256'], optimizer_steps=sorted({int(v['step']) for v in saved['optimizer']['state'].values()}),
        full_training_recipe_unchanged=True, validation_requests=500, seeds=2))


def evaluation_specs(run):
    p = read(run/'PROTOCOL.json')
    out = Path(read(run/'config.json')['output'])
    result = [dict(name='original40k', step=500, checkpoint=out/'checkpoints/step-00000500.pt', old=True),
              dict(name='candidate100', step=100, checkpoint=Path(p['protected100']['path']), old=False)]
    result += [dict(name=f'candidate{step}', step=step,
                    checkpoint=out/'checkpoints'/f'step-{step:08d}.pt', old=False)
               for step in (500,1000,1500,2000)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args=parser.parse_args(); run=args.run_dir.resolve()
    p=verify_inputs(run); q=read(run/'config.json')
    first_command=training_command(run, p['protected500']['path'], 502)
    if args.dry_run:
        print(json.dumps(dict(startup=first_command, continuation_limit=2000,
            train_gpus=[4,5,6,7], eval_preferred_gpus=[0,1,2,3], evaluation_jobs=[j['name'] for j in evaluation_specs(run)])))
        return
    if (run/'STATUS.json').exists():
        raise ValueError('Existing run requires an explicit recovery plan; do not overwrite it.')
    for name in ('protected100','protected500'):
        if sha(p[name]['path'])!=p[name]['sha256']:
            raise ValueError('Protected checkpoint changed: '+name)
    train_leases=[]; eval_leases=[]; logs=[]; children={}; monitor={}
    phase='STARTUP_TO502'; training_done=False; evaluation_job=None; completed=[]; last_claim=0.
    status=dict(pid=os.getpid(), phase='PREPARING', started_unix=time.time(),
        start_step=500, maximum_updates=2000, training_gpus=[4,5,6,7], evaluation_gpus=[],
        training_output=q['output'], configuration=str(run/'config.json'), completed_evaluations=[])

    def launch(name, command, gpus, leases):
        verify_inputs(run)
        log=(run/(name+'.log')).open('a');logs.append(log)
        process=subprocess.Popen(command, cwd=ROOT, env=environment(gpus), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=tuple(handle.fileno() for handle in leases))
        return process

    def optional_claim(pool):
        try:
            return acquire_leases(pool)
        except BlockingIOError:
            return []
        except RuntimeError as exc:
            if 'still occupied' not in str(exc):
                raise
            return []

    try:
        train_leases=acquire_leases(p['training_gpu_pool'])
        log=(run/'GPU_UTILIZATION.csv').open('a');logs.append(log)
        monitor['gpu']=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits','--loop-ms=10000'],stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        children['training']=launch('STARTUP_TO502',first_command,[4,5,6,7],train_leases)
        status.update(phase=phase, training_pid=children['training'].pid, training_command=first_command)
        write(run/'STATUS.json',status)
        while True:
            if (run/'USER_HOLD.json').exists() or (Path(q['output'])/'STOP').exists():
                raise RuntimeError('User paused the continuation; preserve recovery and stop owned children.')
            verify_inputs(run)
            worker=children.get('training')
            if worker is not None:
                code=worker.poll()
                if code not in (None,0):
                    raise RuntimeError(f'{phase} failed with exit code {code}')
                if code==0:
                    del children['training']
                    if phase=='STARTUP_TO502':
                        audit_startup(run)
                        command=training_command(run,Path(q['output'])/'resume_latest.pt',2000)
                        children['training']=launch('TRAIN_TO2000',command,[4,5,6,7],train_leases)
                        phase='TRAIN_TO2000'
                        status.update(phase=phase,training_pid=children['training'].pid,training_command=command,
                                      startup_verified=True)
                    else:
                        resume=read(Path(q['output'])/'RESUME.json')
                        if resume['step']!=2000:
                            raise RuntimeError('Training exited before2000.')
                        training_done=True
                        for handle in train_leases:handle.close()
                        train_leases=[]
                        status.update(training_complete=True,training_completed_unix=time.time(),training_pid=None)
            evaluator=children.get('evaluation')
            if evaluator is not None:
                code=evaluator.poll()
                if code not in (None,0):
                    raise RuntimeError(f'{evaluation_job["name"]} evaluation failed: {code}')
                if code==0:
                    name=evaluation_job['name']
                    if name!='original40k':
                        from scripts.t2a.rl.report_editing_opsd_to2000 import promote
                        promote(run,name)
                    completed.append(name)
                    write(run/'EVALUATION_PROGRESS.json',dict(completed=completed,at_unix=time.time()))
                    del children['evaluation'];evaluation_job=None
                    for handle in eval_leases:handle.close()
                    eval_leases=[]
                    status.update(evaluation_gpus=[],evaluation_pid=None,evaluation_job=None)
            if phase=='TRAIN_TO2000' and 'evaluation' not in children and time.time()-last_claim>=30:
                pending=next((item for item in evaluation_specs(run) if item['name'] not in completed),None)
                if pending and pending['checkpoint'].is_file():
                    last_claim=time.time()
                    pool=p['evaluation_gpu_pool'];eval_leases=optional_claim(pool)
                    if not eval_leases and training_done:
                        pool=p['training_gpu_pool'];eval_leases=optional_claim(pool)
                    if eval_leases:
                        checkpoint=dict(path=str(pending['checkpoint']),sha256=sha(pending['checkpoint']),step=pending['step'])
                        side='old' if pending['old'] else 'new';gpus=[x['index'] for x in pool]
                        command=[sys.executable,'-m','torch.distributed.run','--nnodes=1',
                            '--nproc_per_node=4','--master_port=29652',str(ROOT/'scripts/t2a/rl/evaluate_editing_opsd_branch_cross.py'),
                            '--config',str(run/'config.json'),'--checkpoint',checkpoint['path'],'--checkpoint-sha256',checkpoint['sha256'],
                            '--step',str(pending['step']),'--planner',side,'--executor',side,'--output',str(run/'evaluations'/pending['name']),
                            '--fresh-matched-panel','--no-save-audio','--full-native-validation']
                        children['evaluation']=launch('EVAL_'+pending['name'],command,gpus,eval_leases)
                        evaluation_job=pending
                        status.update(evaluation_gpus=gpus,evaluation_pid=children['evaluation'].pid,
                                      evaluation_job=pending['name'],evaluation_checkpoint=checkpoint)
                    else:
                        status['evaluation_waiting_for']='Existing GPU0-3 jobs to release their coordination locks; no preemption.'
            current=Path(q['output'])/'STATUS_rank0.json'
            if current.exists():
                observation=read(current)
                status.update(training_step=observation.get('step'),training_phase=observation.get('phase'),
                              training_observed_unix=observation.get('time'),performance=observation.get('performance'))
            status.update(observed_unix=time.time(),completed_evaluations=list(completed),
                allocated_GPUs=(0 if training_done else 4)+(4 if 'evaluation' in children else 0),
                phase='FINAL_EVALUATIONS' if training_done else phase)
            write(run/'STATUS.json',status)
            if training_done and len(completed)==6:
                status.update(phase='COMPLETE',completed_unix=time.time(),allocated_GPUs=0,
                    report=str(run/'TABLE.md'),top3=str(run/'selection/top_checkpoints/INDEX.json'))
                write(run/'STATUS.json',status)
                break
            time.sleep(10)
    except BaseException as exc:
        stop_owned(children)
        status.update(phase='FAILED',error=repr(exc),observed_unix=time.time(),allocated_GPUs=0)
        write(run/'STATUS.json',status)
        raise
    finally:
        stop_owned(monitor)
        for handle in train_leases+eval_leases+logs:handle.close()


if __name__=='__main__':
    main()
