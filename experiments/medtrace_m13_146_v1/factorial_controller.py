"""Dependency-ordered bounded factorial: train, freeze, single evaluation, score, report."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import factorial_protocol as f
from interim20_controller import read,write
RUN=Path(os.environ['RUN_ROOT']);TOOLS=RUN/'private/tools'

def launch(file,action,part,gpu):
    fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(TOOLS/file))+',run_name="__main__")\n')
    env=dict(os.environ,ACTION=action,PARTITION=str(part),GPU=str(gpu),CUDA_VISIBLE_DEVICES='' if gpu<0 else str(gpu))
    log=(RUN/'logs'/f'{action}_{part}.log').open('ab')
    child=subprocess.Popen([sys.executable,'-u',entry],env=env,stdout=log,stderr=subprocess.STDOUT);log.close()
    write(RUN/'private'/f'START_{action}_{part}.json',dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],entry=entry,epoch=time.time()))
    return child

def stage(action):
    jobs=[launch('factorial_runtime.py',action,p,gpu) for p,gpu in enumerate([0,1])]
    codes=[j.wait() for j in jobs];assert codes==[0,0],f'{action} failed: {codes}'

def merge(phase):
    parent=Path(read(RUN/'private/FACTORIAL_ADMISSION1.json')['parent']);own=read(RUN/'RESOURCE_LEDGER.json');prior=read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in own['gpu_sessions'])
    field='gpu_seconds_used' if phase=='GPU' else 'Judge_attempts';key='factorial146_'+phase+'_merged'
    with (parent/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(parent/'RESOURCE_LEDGER.json')
        if key in ledger:assert ledger[key]['run']==str(RUN)
        previous=ledger.get(key,{}).get('increment',0);delta=own[field]-prior[field];assert delta>=previous
        ledger[field]+=delta-previous
        if phase=='GPU' and key not in ledger:ledger['gpu_sessions']+=own['gpu_sessions']
        ledger[key]=dict(run=str(RUN),increment=delta,epoch=time.time());write(parent/'RESOURCE_LEDGER.json',ledger)
        write(RUN/'private'/f'FACTORIAL_COST_TRANSFER_{phase}1.json',ledger[key])

def main():
    f.admit(RUN);assert read(RUN/'private/FACTORIAL_CPU_CHECK1.json')['status']=='PASS'
    with (RUN/'private/FACTORIAL_CONTROLLER_ONCE1').open('x'):pass
    stage('factorial_train')
    from m13_only_continue import complete
    for arm in ('M1','M3'):
        for owner in range(1,147):complete(RUN,arm,owner)
    write(RUN/'private/FACTORIAL_FROZEN1.json',dict(candidates=292,steps=160,methods=['M1','M3'],epoch=time.time()))
    stage('m13_146_single')
    counts={k:0 for k in f.CAPS}
    for path in (RUN/'private').glob('COUNTS_*.json'):
        for k,v in read(path).items():
            if k in counts:counts[k]+=v
    assert all(counts[k]<=cap for k,cap in f.CAPS.items())
    assert counts['updates']==46720 and counts['backwards']==93448 and counts['operator_backwards']==23360 and counts['generation_attempts']==5912
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==f.CONSUMERS
    write(RUN/'private/FACTORIAL_GENERATION_COMPLETE1.json',dict(status='COMPLETE',counts=counts,consumers=f.CONSUMERS,epoch=time.time()));merge('GPU')
    root=RUN/'private'/f.QUEUE
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        f.project_budget(RUN)
        assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        # Final DONE upload happens after the last scoring RPC; one status call closes the barrier.
        env=dict(os.environ,QUEUE_REQUEST=json.dumps(dict(action='status')))
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
        Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(TOOLS/'factorial_queue.py'))+',run_name="__main__")\n')
        subprocess.run([sys.executable,entry],env=env,check=True,stdout=subprocess.DEVNULL)
        Path(entry).unlink();time.sleep(20)
    merge('Judge');assert launch('factorial_report.py','factorial_report',0,-1).wait()==0
    assert (RUN/'private/FACTORIAL_REPORT_COMPLETE1.json').exists()
    write(RUN/'private/FACTORIAL_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',epoch=time.time()))

if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;write(RUN/'private/FAILURE_factorial_controller1.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        for phase in ('GPU','Judge'):
            try:merge(phase)
            except BaseException as e:write(RUN/'private'/f'FACTORIAL_COST_TRANSFER_ERROR_{phase}.json',dict(error=repr(e)))
        write(RUN/'private/FACTORIAL_CONTROLLER_EXIT1.json',dict(exit_code=code,epoch=time.time()))
