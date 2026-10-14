"""Single finite Fisher -> two M13 consumers, with no retry or full146 launch."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import runtime146 as worker

c,RUN=worker.c,worker.RUN


def launch(action,gpu,owner):
    hardware=c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs']
    uuid,free=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert uuid==hardware[str(gpu)] and int(free)>=45*1024
    with (RUN/'private'/f'LAUNCH_ONCE_{action}_{owner}').open('x'):pass
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['NATIVE_FOLLOWUP_TOOLS'])\nrunpy.run_path(os.environ['NATIVE_FOLLOWUP_TOOLS']+'/native_followup.py',run_name='__main__')\n")
    env=dict(os.environ,ACTION=action,GPU=str(gpu),OWNER=str(owner),PARTITION=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu))
    process=subprocess.Popen([env['TRAIN_PYTHON'],'-u',name],env=env,
        stdout=(RUN/'logs'/f'{action}_{owner}.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    proc=Path('/proc')/str(process.pid)
    c.write(RUN/'private'/f'START_{action}_{owner}.json',dict(pid=process.pid,
        start_ticks=proc.joinpath('stat').read_text().split(') ',1)[1].split()[19],gpu=gpu,owner=owner,
        argv=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode(),entry=name,
        gpu_uuid=uuid,source=c.read(RUN/'private/NATIVE_FOLLOWUP_SOURCE1.json')['commit'],epoch=time.time()))
    return process


def wait(processes):
    for process in processes:
        while True:
            try:code=process.wait(timeout=10);break
            except subprocess.TimeoutExpired:c.budget()
        assert code==0,f'Actual worker exit {code}; no automatic retry'


def main():
    worker.configure()
    from native_followup import require_admission
    require_admission()
    with (RUN/'private/NATIVE_FOLLOWUP_CONTROLLER_ONCE1').open('x'):pass
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    c.write(RUN/'private/NATIVE_FOLLOWUP_PROGRESS1.json',dict(stage='FISHER_RUNNING',epoch=time.time()))
    wait([launch('native_fisher',0,0)])
    assert c.read(RUN/'private/NATIVE_FISHER_COMPLETE1.json')['status']=='COMPLETE'
    c.write(RUN/'private/NATIVE_FOLLOWUP_PROGRESS1.json',dict(stage='TWO_M13_RUNNING',epoch=time.time()))
    wait([launch('native_m13',0,1),launch('native_m13',1,2)])
    for owner in (1,2):assert c.read(RUN/'private'/f'NATIVE_M13_COMPLETE_{owner}.json')['steps']==160
    counts=[c.read(v) for v in (RUN/'private').glob('COUNTS_*.json')]
    total={k:sum(v.get(k,0) for v in counts) for k in counts[0]}
    for key,cap in c.read(RUN/'PLAN_CONFIG.json')['maximum_counts'].items():
        if key!='Judge_attempts':assert total.get(key,0)<=cap
    assert total['updates']==640 and total['backwards']==3248 and total['operator_backwards']==320
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    c.write(RUN/'private/NATIVE_FOLLOWUP_PROGRESS1.json',dict(stage='TWO_M13_COMPLETE_FMIN_IMPLEMENTATION_PENDING',
        counts=total,full146_started=False,performance_evaluation_complete=False,epoch=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/'private/FAILURE_native_followup_controller1.json',
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
    finally:c.write(RUN/'private/NATIVE_FOLLOWUP_EXIT1.json',dict(exit_code=code,epoch=time.time()))
