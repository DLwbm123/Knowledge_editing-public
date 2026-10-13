"""Finite dependency controller, phase budgets and terminal CPU report; no scheduler."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

RUN=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def used():
    d=read(RUN/'RESOURCE_LEDGER.json');return d['gpu_seconds_used']+sum(time.time()-s['started_epoch'] for s in d['gpu_sessions'] if s.get('ended_epoch') is None)
def start(action,gpu,part=None):
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/worker.py',run_name='__main__')\n")
    uuids={5:'GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca',6:'GPU-ffc224df-661c-5aff-e221-ec361bb4e5d6'}
    env=dict(os.environ,ACTION=action,GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuids[gpu])
    if part is not None:env['PARTITION']=str(part)
    p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',name],env=env,stdout=(RUN/'logs'/f'{action}_{gpu}.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    s=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],epoch=time.time(),action=action,gpu=gpu,partition=part,entry=name,argv=['/data/bmw/envs/v0/bin/python','-u',name])
    write(RUN/'private'/f'START_{action}_{gpu}.json',s);print('START',json.dumps(s),flush=True);return p,s
def wait(jobs):
    while any(p.poll() is None for p,_ in jobs):
        m=read(RUN/'RUN_MANIFEST.json')
        stop=(RUN/'STOP').exists() or time.time()>=m['deadline_epoch'] or used()>=m['GPU_seconds_limit']
        if stop:
            write(RUN/'private/BUDGET_STOP.json',dict(epoch=time.time(),gpu_seconds_including_live=used()))
            for p,s in jobs:
                if p.poll() is None and Path(f'/proc/{p.pid}/stat').read_text().split()[21]==s['start_ticks']:os.killpg(p.pid,signal.SIGTERM)
            for p,_ in jobs:
                try:p.wait(timeout=60)
                except subprocess.TimeoutExpired:pass
            raise TimeoutError('Original finite cap; preserve accepted partial state')
        time.sleep(15)
    assert all(p.returncode==0 for p,_ in jobs),'Worker failure; no automatic retry'
def admit(stage,estimate):
    m=read(RUN/'RUN_MANIFEST.json');remaining=m['GPU_seconds_limit']-used()
    d=dict(stage=stage,epoch=time.time(),used_gpu_seconds=used(),remaining_gpu_seconds=remaining,estimated_required_gpu_seconds=estimate,wall_remaining_seconds=m['deadline_epoch']-time.time(),admitted=remaining>=estimate and m['deadline_epoch']-time.time()>=estimate/2+1200)
    write(RUN/'private'/f'PHASE_ADMISSION_{stage}.json',d);return d['admitted']
def guard():
    m=read(RUN/'RUN_MANIFEST.json')
    if (RUN/'STOP').exists() or time.time()>=m['deadline_epoch'] or used()>=m['GPU_seconds_limit']:
        raise TimeoutError('Original clock and cumulative caps; no reset')


def available(gpu):
    uuid=read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert row[0]==uuid
    if int(row[1])<40000:return False,'free memory below40000MiB admission'
    if any(s['gpu_uuid']==uuid and s.get('ended_epoch') is None for s in read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions']):return False,'own active lease'
    return True,'PASS'


def wait_capacity(gpu):
    while True:
        guard();ok,reason=available(gpu)
        write(RUN/'private'/f'CAPACITY_GPU{gpu}.json',dict(epoch=time.time(),gpu=gpu,admitted=ok,reason=reason))
        if ok:return
        write(RUN/'public/PROGRESS.json',dict(status='WAITING_GPU_CAPACITY',gpu=gpu,reason=reason,epoch=time.time()))
        time.sleep(30)


def cpu_tool(name,python='/data/bmw/envs/v0/bin/python'):
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env=dict(os.environ,TASK_FILE=str(RUN/'private/tools'/name),QWEN_EXECUTION_FILE=str(RUN/'private/tools/qwen_scorer.py'),CUDA_VISIBLE_DEVICES='')
    subprocess.run([python,'-u',entry],env=env,stdout=(RUN/'logs'/name.replace('.py','.log')).open('ab'),stderr=subprocess.STDOUT,check=True)

def main():
    lock=(RUN/'private/CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(RUN/'private/CPU_ADMISSION.json')['status']=='PASS'
    assert read(RUN/'private/SCORING_IMPLEMENTATION_ADMISSION.json')['status']=='PASS'
    if not (RUN/'private/GPU_MECHANICAL.json').exists():
        wait_capacity(6);wait([start('mechanical',6)])
        for order in (31,35):
            d=RUN/'private/edits'/f'e{order:03d}';assert read(d/'COMPLETE.json')['init_once_per_route']
            for arm in read(RUN/'PLAN_CONFIG.json')['arms']:
                assert read(d/arm/'ONE_STEP_CHECK.json')['status']=='PASS'
                assert read(d/arm/'STRUCTURE_CHECK.json')['status']=='PASS'
        write(RUN/'private/GPU_MECHANICAL.json',dict(status='PASS',orders=[31,35],formal_first8_branches_reused=True,epoch=time.time()))
    if not (RUN/'private/GENERATION_COMPLETE.json').exists():
        jobs=[]
        for gpu,part in [(6,0),(5,1)]:
            if not (RUN/f'private/P1_PART_{part}_COMPLETE.json').exists():
                wait_capacity(gpu);jobs.append(start('p1',gpu,part))
        wait(jobs)
        assert all((RUN/f'private/P1_PART_{p}_COMPLETE.json').exists() for p in (0,1))
        write(RUN/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',completed_phases=['P0','P1'],full146=False,H8=8,branches=32,epoch=time.time()))
        write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='GENERATION_COMPLETE',Judge_and_metrics_pending=True,epoch=time.time()))
    cpu_tool('qwen_scorer.py')
    assert read(RUN/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
    if not (RUN/'private/REPORT_COMPLETE.json').exists():cpu_tool('reporting.py')
    write(RUN/'public/PROGRESS.json',dict(status='REPORT_COMPLETE_REVIEW_PUBLICATION_PENDING',epoch=time.time(),no_automatic_next_experiment=True))

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(RUN/'private/CONTROLLER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),pid=os.getpid(),automatic_retry=False));raise
