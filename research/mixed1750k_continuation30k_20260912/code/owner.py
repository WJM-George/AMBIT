"""Wait for the existing data queue; then run training, selection and all tests."""
from common import *
import fcntl
import resource
import signal
import subprocess
import time
import traceback


def state(stage,**details):write(ROOT/'OWNER_STATE.json',dict(stage=stage,pid=os.getpid(),observed_at=now(),**details))

def run(stage,command):
    logfile=ROOT/'logs'/f'{stage}_{time.time_ns()}.log'
    with logfile.open('xb') as out:
        child=subprocess.Popen(command,cwd=ROOT/'code',env=env_cpu(),stdin=subprocess.DEVNULL,
            stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    state(stage,child_pid=child.pid,log=str(logfile),command=command)
    try:code=child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    write(ROOT/'receipts'/f'{stage}_{time.time_ns()}.json',dict(stage=stage,exit_code=code,log=str(logfile),completed_at=now()))
    if code:raise RuntimeError(f'{stage} exited {code}: {logfile}')

def cpu(script,*args):return [PYTHON,str(ROOT/'code'/script),*args]
def gpu(script,*args):
    return [PYTHON,str(OLD_RUN/'editing_allocated_gpu_runtime_v1.py'),'--gpus','2,3,4','--',
        PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',str(ROOT/'code'/script),*args]

def main():
    lock=(ROOT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    assert read(ROOT/'POLICY.json')['checkpoint_steps']==STEPS[1:]
    ready=read(ROOT/'WORKFLOW_READY.json')
    assert ready['status']=='PASS_VERIFIED_WORKFLOW_ARMED_FOR_DATA_COMPLETION'
    for path,expected in ready['source_sha256'].items():assert sha(path)==expected,path
    waiting=None
    while not ((SPATIAL/'DATA_READY.json').exists() and (PIPE/'RESULT.json').exists()):
        failure=(SPATIAL/'OWNER_FAILURE.json').exists() or (PIPE/'FAILURE.json').exists()
        label='waiting_for_data_recovery' if failure else 'waiting_for_all_data_and_existing_queue_completion'
        if label!=waiting:state(label,depends_on=[str(SPATIAL/'DATA_READY.json'),str(PIPE/'RESULT.json')]);waiting=label
        time.sleep(15)
    assert read(SPATIAL/'DATA_READY.json')['status']=='PASS_500K_SPATIAL_MULTI_TRAIN_LATENTS_AND_SPLIT_EXTENSIONS'
    assert read(PIPE/'RESULT.json')['status']=='COMPLETE_AUTHORIZED_EDITING_PIPELINE'
    # Recheck all sealed code after a potentially long wait before GPU use.
    for path,expected in ready['source_sha256'].items():assert sha(path)==expected,path
    if not (ROOT/'training/PREPARED.json').exists():run('prepare_training',cpu('prepare_training.py'))
    if not (ROOT/'EVALUATION_PREPARED.json').exists():run('freeze_evaluation_cohorts',cpu('prepare_evaluation.py'))
    if not (ROOT/'reviews/FINAL_COMPONENT_REVIEW.json').exists():run('verify_final_training_components',cpu('workflow_check.py','--final-components'))
    if not (ROOT/'training/TECHNICAL_REVIEW.json').exists():run('two_update_DDP_check',gpu('train.py','--technical'))
    if not (ROOT/'training/RESULT.json').exists():
        saved=[]
        for step in STEPS[1:]:
            path=ROOT/f'training/checkpoints/STEP{step:06d}/CHECKPOINT.json'
            if path.exists():
                ref=read(path);assert ref['optimizer_steps']==step and sha(ref['path'])==ref['sha256'];saved.append(ref)
        if saved and saved[-1]['optimizer_steps']==80000:
            run('verify_final_recovery',cpu('verify_checkpoints.py'))
            write(ROOT/'training/RESULT.json',dict(status='COMPLETE_30K_RECOVERED_FROM_VERIFIED_FINAL_CHECKPOINT',
                optimizer_steps=80000,checkpoint_steps=STEPS[1:],train_rows=1750000,completed_at=now()))
        else:
            args=['--resume',saved[-1]['path']] if saved else []
            run('training_30k',gpu('train.py',*args))
    if not (ROOT/'training/CHECKPOINTS_VERIFIED.json').exists():run('verify_six_checkpoints',cpu('verify_checkpoints.py'))
    if not (ROOT/'truth_reviews/validation.json').exists():run('prepare_validation_truth',cpu('truth_cache.py','--scope','validation'))
    for step in STEPS:
        directory=ROOT/f'validation/STEP{step:06d}'
        if not (directory/'RESULT.json').exists():
            if not (directory/'INFERENCE_COMPLETE.json').exists():run(f'validation_{step}',gpu('evaluate.py','--split','validation','--step',str(step)))
            run(f'review_validation_{step}',cpu('review_evaluation.py','--split','validation','--step',str(step)))
    if not (ROOT/'SELECTION.json').exists():run('select_checkpoint',cpu('select_checkpoint.py'))
    chosen=read(ROOT/'SELECTION.json')['selected_checkpoint_steps']
    for scope in SCOPES[1:]:
        directory=ROOT/scope/f'STEP{chosen:06d}'
        if not (ROOT/'truth_reviews'/f'{scope}.json').exists():run(f'prepare_truth_{scope}',cpu('truth_cache.py','--scope',scope))
        if not (directory/'RESULT.json').exists():
            if not (directory/'INFERENCE_COMPLETE.json').exists():run(scope,gpu('evaluate.py','--split',scope,'--step',str(chosen)))
            run('review_'+scope,cpu('review_evaluation.py','--split',scope,'--step',str(chosen)))
    run('deliver_three_test_reports',cpu('deliver_report.py'))
    state('COMPLETE_TRAINING_VALIDATION_SELECTION_AND_THREE_TESTSETS',selected_checkpoint_steps=chosen,result=str(ROOT/'RESULT.json'))


if __name__=='__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Owner interrupted: {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:main()
    except BaseException as error:
        write(ROOT/'OWNER_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),pid=os.getpid(),observed_at=now()));raise
