"""Finite dependency controller, phase budgets and terminal CPU report; no scheduler."""
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
    env=dict(os.environ,ACTION=action,GPU=str(gpu))
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
def finish(completed,reason):
    write(RUN/'private/GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',completed_phases=completed,reason=reason,epoch=time.time(),full146='P2' in completed))
    write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='GENERATION_COMPLETE',completed_phases=completed,epoch=time.time(),Judge_and_metrics_pending=True,public_delivery_pending=True))
    write(RUN/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_JUDGE_PENDING',completed_phases=completed,full146='P2' in completed))
    while not (RUN/'private/judge_common/SCORER_DONE.json').exists():
        if (RUN/'private/judge_common/SCORER_FAILURE.json').exists() or (RUN/'private/judge_common/SCORER_STOP.json').exists():
            write(RUN/'private/CLOSEOUT_PENDING.json',dict(reason='Judge terminal stop; retained partial/missing evidence',epoch=time.time()));return
        if (RUN/'STOP').exists() or time.time()>=read(RUN/'RUN_MANIFEST.json')['deadline_epoch']:return
        time.sleep(30)
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/reporting.py',run_name='__main__')\n")
    subprocess.run(['/data/bmw/envs/v0/bin/python',name],env=os.environ,stdout=(RUN/'logs/reporting.log').open('ab'),stderr=subprocess.STDOUT,check=True)
def main():
    assert read(RUN/'private/CPU_ADMISSION.json')['status']=='PASS'
    assert not (RUN/'private/CONTROLLER_COMPLETE.json').exists()
    write(RUN/'public/PROGRESS.json',dict(status='ACTUAL_GPU_MECHANICAL_RUNNING',H8=8,branches=32,GPU_allowlist=[4,5]))
    wait([start('mechanical',4,0),start('mechanical',5,1)])
    for order in (31,35):
        d=RUN/'private/edits'/f'e{order:03d}'
        assert read(d/'COMPLETE.json')['init_once']
        for arm in ('A_NO_H','B_H1','C_H025','D_EXTRA_FIT'):assert read(d/arm/'ONE_STEP_CHECK.json')['status']=='PASS'
    write(RUN/'private/GPU_MECHANICAL.json',dict(status='PASS',orders=[31,35],formal_first_eight_branches_reused=True,epoch=time.time()))
    wait([start('p1',4,0),start('p1',5,1)])
    write(RUN/'private/P1_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',H8=8,branches=32,epoch=time.time()))
    # Estimate without selecting an arm or looking at new accuracy. Historical runtime
    # and the largest actually observed H8 common-initialization+NO_H continuation
    # provide a conservative reconstruction envelope; bank/P3 reserves are separate.
    costs=[]
    for order in read(RUN/'PLAN_CONFIG.json')['H8_orders']:
        d=RUN/'private/edits'/f'e{order:03d}';costs.append(read(d/'INITIALIZATION.json')['seconds']+read(d/'A_NO_H/TRAINING.json')['seconds'])
    estimate=138*max(costs)*1.35+3*3600
    if not admit('P2',estimate):finish(['P0','P1'],'P2_NOT_ADMITTED_FINITE_BUDGET');return
    wait([start('shared',4,0),start('shared',5,1)])
    wait([start('bank',4)])
    if not admit('P3',2*3600):finish(['P0','P1','P2'],'P3_NOT_ADMITTED_FINITE_BUDGET');return
    wait([start('oracle',4)])
    finish(['P0','P1','P2','P3'],'FROZEN_GENERATION_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(RUN/'private/CONTROLLER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),pid=os.getpid(),automatic_retry=False));raise
