"""Run the independently identified repaired eight-GPU experiment through2000."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases,read,write,stop_owned
from scripts.t2a.rl.launch_editing_opsd_fresh2000 import environment,verify_inputs
from scripts.t2a.rl.launch_editing_opsd_eight_gpu import evaluation_command
from scripts.t2a.rl.train_editing_opsd_repaired_fresh import validate_configuration,validate_state
from stable_audio_tools.training.transfusion_opsd.request_paired_supervision import (
    OPERATIONS, execution_supervision, covered_request, feedback_coverage,
)
from stable_audio_tools.training.transfusion_opsd.branch_request_supervision import active as branch_recipe


def training_command(run,checkpoint,until):
    command=[sys.executable,'-B','-m','torch.distributed.run','--nnodes=1',
        '--nproc_per_node=8','--master_port=29671',str(ROOT/'scripts/t2a/rl/train_editing_opsd_repaired_fresh.py'),
        '--config',str(run/'config.json'),'--limit-updates',str(until),
        '--evaluate-at-updates','250','750','1250','1750']
    if checkpoint is not None:command+=['--resume',str(checkpoint)]
    return command


def audit_startup(run):
    import torch
    q=dict(read(run/'config.json'),config_path=str(run/'config.json'));out=Path(q['output'])
    state=torch.load(out/'resume_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    zero=torch.load(out/'evaluation_candidates/step-00000000.pt',map_location='cpu',weights_only=False,mmap=True)
    validate_state(q,zero);validate_state(q,state)
    if state['step']!=2 or zero['optimizer']['state']:
        raise ValueError('Startup must contain two real updates from a new optimizer.')
    requests=[];pairs=[];removals=[];gradient_probes=[];performance=[]
    all_requests=[];corrections=[];gradient_operations=set()
    all_operations=bool(q.get('request_paired_correction'))
    per_branch=branch_recipe(q)
    branch_runtime=[]
    for rank in range(8):
        if per_branch:
            proof=read(out/f'startup_checks/BRANCH_RUNTIME_rank{rank}.json')
            if (proof['phase']!='PASS' or not proof['weights_Adam_RNG_and_samplers_unchanged']
                    or not proof['isolated_fallback_only']
                    or {(r['GT_AR'],r['GT_RF']) for r in proof['branch_gradients']} != {(True,True),(True,False),(False,True)}
                    or not all(r['native_formula_and_gradient_match'] for r in proof['branch_gradients'])):
                raise ValueError('Missing isolated real AR-only/DiT-only/joint fallback acceptance.')
            branch_runtime.append(proof)
        prepared=read(out/f'startup_checks/PREPARED_step000000_rank{rank}.json')
        if q.get('initialization_checkpoint') is None and not prepared['original40k_weights_verified']:
            raise ValueError('Fresh original40k weights were not verified.')
        if prepared['optimizer_state_tensors']!=0 or not prepared['samplers_unchanged'] or not prepared['RNG_restored']:
            raise ValueError('Startup probes consumed Adam/RNG/samplers.')
        first=read(out/f'startup_checks/FIRST_UPDATE_000001_rank{rank}.json')
        if not first['samples_match_saved_cursors']:
            raise ValueError('First update changed sample order.')
        requests+=first['request_ordinals'];pairs+=first['paired_ordinals']
        updates=[json.loads(line) for line in (out/f'UPDATES_rank{rank}.jsonl').read_text().splitlines()]
        updates=list({u['step']:u for u in updates if u['step'] in (1,2)}.values())
        if len(updates)!=2:
            raise ValueError('Missing a real startup update.')
        for update in updates:
            if len(update['paired_ordinals'])!=64 or update['extra_objectives']['selected_rows']!=1:
                raise ValueError('Startup changed native paired/decoded budgets.')
            performance.append(dict(rank=rank,step=update['step'],**update['performance']))
            for row in update['request_updates']:
                all_requests.append(row)
                correction=row['paired_removal_correction']
                if all_operations:
                    paired=row['paired_request_correction']
                    route=execution_supervision(row,q['spatial_recipe'])
                    if (covered_request(route,paired)!=row['request_supervision']
                            or bool(paired['enabled'])==route['execution_joint']):
                        raise ValueError('Actual request did not follow the complete fallback policy.')
                    if paired['enabled']:
                        corrections.append(dict(rank=rank,step=update['step'],ordinal=row['ordinal'],correction=paired))
                        if paired.get('gradient_probe'):
                            gradient_operations.add(row['requested_operation'])
                            gradient_probes.append(paired['gradient_probe'])
                if row['requested_operation']=='event_removal':
                    if not correction['enabled'] or correction['execution_teacher'] or row['reference_velocity_enabled']:
                        raise ValueError('Removal repair did not reach the actual training path.')
                    removals.append(dict(rank=rank,step=update['step'],ordinal=row['ordinal'],correction=correction))
                    if not all_operations and correction.get('gradient_probe'):gradient_probes.append(correction['gradient_probe'])
                elif correction['enabled'] or not row['reference_velocity_enabled']:
                    raise ValueError('Repair changed the non-removal path.')
    if len(set(requests))!=16 or len(set(pairs))!=512 or not removals or not gradient_probes:
        raise ValueError('Incomplete startup coverage or missing real removal gradients.')
    coverage=feedback_coverage(all_requests,require_complete=all_operations)
    if all_operations:
        if (len(all_requests)!=32 or set(coverage)!=set(OPERATIONS)
                or gradient_operations!=set(OPERATIONS)
                or any(v['covered_requests']!=v['requests'] or v['uncovered_requests']
                       or not v['paired_request_corrections'] for v in coverage.values())):
            raise ValueError('Startup must verify all five operations, coverage and AR/DiT/shared gradients.')
    write(run/'STARTUP_ACCEPTANCE.json',dict(phase='PASS',step=2,world_size=8,
        original40k_fresh_weights=q.get('initialization_checkpoint') is None,
        fresh_Adam_at_zero=True,complete_Adam_tensors=len(state['optimizer']['state']),
        global_request_batch=16,global_paired_batch=512,paired_catalog_rows=1000000,
        global_decoded_rows=8,removal_requests=len(removals),paired_removal_corrections=len(removals),
        exclusive_per_branch_supervision=per_branch,branch_runtime=branch_runtime,
        objective_budget=q.get('request_paired_correction'),
        all_operation_fallback=all_operations,requests=len(all_requests),
        paired_request_corrections=len(corrections),feedback_coverage=coverage,
        paired_corrections=corrections,gradient_operations=sorted(gradient_operations),
        removals=removals,gradient_probes=gradient_probes,performance=performance,
        model_sha256=state['model_sha256'],at_unix=time.time()))


class Paused(Exception):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--recover',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();run=args.run_dir.resolve();p=verify_inputs(run);q=read(run/'config.json')
    validate_configuration(q);out=Path(q['output'])
    if read(Path(p['prerequisite_test'])/'STATUS.json')['phase']!='COMPLETE':
        raise ValueError('Complete250-step evaluation before restarting training.')
    tables=read(Path(p['prerequisite_test'])/'TWO_TABLES.json')
    if [s['rows'] for s in tables['subsets']]!=[1000,833]:
        raise ValueError('Both requested evaluation tables must exist.')
    from scripts.t2a.rl.report_editing_opsd_eight_gpu import load_evaluation,promote
    load_evaluation(run,'original40k')
    if args.dry_run:
        print(json.dumps(dict(startup=training_command(run,None,2),target=2000,
            evaluation_candidates=list(range(250,2001,250)),policy=q['top_checkpoint_policy'])));return
    if (run/'STATUS.json').exists():
        if not args.recover:raise ValueError('Existing run requires explicit --recover.')
        previous=read(run/'STATUS.json');cmd=Path(f'/proc/{previous["pid"]}/cmdline')
        if cmd.exists() and str(Path(__file__).resolve()).encode() in cmd.read_bytes():
            raise ValueError('Previous driver is still active.')
        archive=run/'attempts';archive.mkdir(exist_ok=True);write(archive/f'STATUS_{time.time_ns()}.json',previous)
    completed=read(run/'EVALUATION_PROGRESS.json')['completed'] if (run/'EVALUATION_PROGRESS.json').exists() else []
    children={};monitor={};logs=[];leases=[]
    status=dict(pid=os.getpid(),phase='PREPARING',started_unix=time.time(),physical_gpus=list(range(8)),
        origin='original40k' if q.get('initialization_checkpoint') is None else 'step250 weights, fresh optimizer',
        target=2000,training_step=0,completed_evaluations=completed,allocated_GPUs=0,training_output=str(out))
    def worker(name,command,*,training):
        verify_inputs(run)
        if (run/'PAUSE.json').exists():raise Paused('Pause requested before next stage.')
        log=(run/(name+'.log')).open('a');logs.append(log)
        child=subprocess.Popen(command,cwd=ROOT,env=dict(environment(range(8)),PYTHONDONTWRITEBYTECODE='1',
            OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=tuple(h.fileno() for h in leases))
        children['worker']=child;status.update(phase=name,worker_pid=child.pid,command=command,allocated_GPUs=8)
        write(run/'STATUS.json',status)
        while child.poll() is None:
            verify_inputs(run)
            if (run/'PAUSE.json').exists():
                if training:(out/'STOP').touch()
                else:stop_owned(children);raise Paused('Evaluation paused; completed case records retained.')
            if training and (out/'STATUS_rank0.json').exists():
                s=read(out/'STATUS_rank0.json');status.update(training_phase=s['phase'],training_observed_unix=s['time'])
                if s.get('step') is not None:status['training_step']=s['step']
            status['observed_unix']=time.time();write(run/'STATUS.json',status);time.sleep(10)
        code=child.returncode;del children['worker'];status.update(worker_pid=None,last_exit_code=code)
        write(run/'STATUS.json',status)
        if (run/'PAUSE.json').exists():raise Paused('Training saved and paused at an optimizer boundary.')
        if code:raise RuntimeError(f'{name} exited{code}; see its log.')
    try:
        leases=acquire_leases(p['gpu_pool'])
        log=(run/'GPU_UTILIZATION.csv').open('a');logs.append(log)
        monitor['gpu']=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits','--loop-ms=10000'],stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        checkpoint=out/'resume_latest.pt' if (out/'resume_latest.pt').exists() else None
        step=read(out/'RESUME.json')['step'] if checkpoint else 0
        if not (run/'STARTUP_ACCEPTANCE.json').exists():
            if step<2:worker('REPAIRED_STARTUP',training_command(run,checkpoint,2),training=True)
            audit_startup(run);checkpoint=out/'resume_latest.pt';step=2
        while True:
            if step and step%250==0 and step not in completed:
                candidate=out/'evaluation_candidates'/f'step-{step:08d}.pt'
                worker(f'EVAL_{step:06d}',evaluation_command(run,candidate,step),training=False)
                promote(run,f'candidate{step}');completed.append(step)
                write(run/'EVALUATION_PROGRESS.json',dict(completed=completed,at_unix=time.time()))
                status['completed_evaluations']=list(completed)
            if step>=2000:break
            until=min((step//250+1)*250,2000)
            worker(f'TRAIN_TO_{until:06d}',training_command(run,checkpoint,until),training=True)
            step=read(out/'RESUME.json')['step'];checkpoint=out/'resume_latest.pt'
            if step!=until:raise RuntimeError('Training did not reach the declared update boundary.')
        status.update(phase='COMPLETE',training_step=2000,allocated_GPUs=0,completed_unix=time.time())
        write(run/'STATUS.json',status)
    except Paused as exc:
        stop_owned(children);status.update(phase='PAUSED',reason=str(exc),allocated_GPUs=0)
        write(run/'STATUS.json',status)
    except BaseException as exc:
        stop_owned(children);status.update(phase='FAILED',error=repr(exc),allocated_GPUs=0,observed_unix=time.time())
        write(run/'STATUS.json',status);raise
    finally:
        stop_owned(monitor)
        for handle in leases+logs:handle.close()


if __name__=='__main__':main()
