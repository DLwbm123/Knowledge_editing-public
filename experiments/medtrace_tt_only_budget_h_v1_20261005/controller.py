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
def start(action,gpu,part=None,phase=None):
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/worker.py',run_name='__main__')\n")
    uuids={int(k):v for k,v in read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'].items()}
    env=dict(os.environ,ACTION=action,GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=str(gpu),M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuids[gpu])
    if part is not None:env['PARTITION']=str(part)
    if phase is not None:env['PHASE']=phase
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
        assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
    if not (RUN/'private/GENERATION_COMPLETE.json').exists():
        for phase in ('primary','secondary'):
            jobs=[]
            for gpu,part in [(6,0),(5,1),(4,2),(3,3)]:
                if not (RUN/f'private/{phase}_PART_{part}_COMPLETE.json').exists():
                    wait_capacity(gpu);jobs.append(start('p1',gpu,part,phase))
            wait(jobs)
            assert all((RUN/f'private/{phase}_PART_{p}_COMPLETE.json').exists() for p in range(4))
        write(RUN/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',completed_phases=['P0','P1','P2'],full146=False,H8=8,branches=240,warmups=96,seed_slots=3,epoch=time.time()))
        write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='GENERATION_COMPLETE',Judge_and_metrics_pending=True,epoch=time.time()))
    cpu_tool('qwen_scorer.py')
    assert read(RUN/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
    if not (RUN/'private/REPORT_COMPLETE.json').exists():cpu_tool('reporting.py')
    write(RUN/'public/PROGRESS.json',dict(status='REPORT_COMPLETE_REVIEW_PUBLICATION_PENDING',epoch=time.time(),no_automatic_next_experiment=True))

def register_partial():
    from protocol import STRUCTURES,arms_for
    plan=read(RUN/'PLAN_CONFIG.json');items=[]
    for order in plan['H8_orders']:
        d=RUN/'private/edits'/f'e{order:03d}'
        for slot in range(3):
            for structure in STRUCTURES:
                arms=arms_for(structure,slot)
                complete=(d/f'{structure}_s{slot}_COMPLETE.json').exists()
                items.append(dict(order=order,seed_slot=slot,structure=structure,paired_block_complete=complete,warmup_complete=(d/f'{structure}_s{slot}_WARMUP/W0.pt').exists(),continuations_complete=sum((d/a/'TRAINING.json').exists() for a in arms),expected_continuations=len(arms)))
    write(RUN/'private/PARTIAL_BLOCK_LEDGER.json',dict(expected_edits=8,expected_warmups=96,expected_continuations=240,blocks=items,score_status='NOT_COMPLETE_DO_NOT_DROP_MISSING_BLOCKS',clock_reset=False))
    write(RUN/'public/PARTIAL_COMPLETION.json',dict(status='INCOMPLETE',expected_warmups=96,expected_continuations=240,completed_warmups=sum(x['warmup_complete'] for x in items),completed_continuations=sum(x['continuations_complete'] for x in items),paired_blocks_complete=sum(x['paired_block_complete'] for x in items),incomplete_blocks=sum(not x['paired_block_complete'] for x in items),unscored_not_zero_accuracy=True))

if __name__=='__main__':
    try:main()
    except Exception as e:
        register_partial()
        write(RUN/'private/CONTROLLER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),pid=os.getpid(),automatic_retry=False));raise
