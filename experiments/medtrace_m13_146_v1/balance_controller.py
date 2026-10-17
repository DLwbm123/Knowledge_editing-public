"""Dependency-ordered bounded balance: train, freeze, single evaluation, score, report."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import balance_protocol as f
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
    jobs=[launch('balance_runtime.py',action,p,gpu) for p,gpu in enumerate([0,1])]
    codes=[j.wait() for j in jobs];assert codes==[0,0],f'{action} failed: {codes}'

def merge(phase):
    parent=Path(read(RUN/'private/BALANCE_ADMISSION1.json')['parent']);own=read(RUN/'RESOURCE_LEDGER.json');prior=read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in own['gpu_sessions'])
    field='gpu_seconds_used' if phase=='GPU' else 'Judge_attempts';key=f.COST_PREFIX+'_'+phase+'_merged'
    with (parent/'RESOURCE_LEDGER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(parent/'RESOURCE_LEDGER.json')
        if key in ledger:assert ledger[key]['run']==str(RUN)
        previous=ledger.get(key,{}).get('increment',0);delta=own[field]-prior[field];assert delta>=previous
        ledger[field]+=delta-previous
        if phase=='GPU' and key not in ledger:ledger['gpu_sessions']+=own['gpu_sessions']
        ledger[key]=dict(run=str(RUN),increment=delta,epoch=time.time());write(parent/'RESOURCE_LEDGER.json',ledger)
        write(RUN/'private'/f'BALANCE_COST_TRANSFER_{phase}1.json',ledger[key])

def main():
    f.admit(RUN);assert read(RUN/'private/BALANCE_CPU_CHECK1.json')['status']=='PASS'
    with (RUN/'private/BALANCE_CONTROLLER_ONCE1').open('x'):pass
    stage('balance_train')
    from m13_only_continue import complete
    for arm in (f.CANDIDATE,):
        for owner in range(1,147):complete(RUN,arm,owner)
    write(RUN/'private/BALANCE_FROZEN1.json',dict(candidates=146,steps=160,methods=[f.CANDIDATE],epoch=time.time()))
    stage('m13_146_single')
    counts={k:0 for k in f.CAPS}
    for path in (RUN/'private').glob('COUNTS_*.json'):
        for k,v in read(path).items():
            if k in counts:counts[k]+=v
    assert all(counts[k]<=cap for k,cap in f.CAPS.items())
    assert counts['updates']==23360 and counts['backwards']==f.CAPS['backwards'] and counts['operator_backwards']==f.CAPS['operator_backwards'] and counts['generation_attempts']==2956
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==f.CONSUMERS
    write(RUN/'private/BALANCE_GENERATION_COMPLETE1.json',dict(status='COMPLETE',counts=counts,consumers=f.CONSUMERS,epoch=time.time()));merge('GPU')
    retire_candidates()
    finish()

def retire_candidates():
    import contract
    from common import digest
    plan=read(RUN/'private/BALANCE_NEW_PLAN1.json');expected={contract.consumer_id(v) for v in plan}
    actual=set();scope=digest(read(RUN/'private/SCOPE_LOCK.json'));source=read(RUN/'private/GPU_SOURCE_VERSION.json')
    for path in (RUN/'private/outputs'/f'{f.CANDIDATE}_single_p1').glob('*/*.json'):
        d=read(path);assert d['method']==f.CANDIDATE and d['lock']==scope and d['execution']==source
        assert d['consumer_id'] in expected and d['consumer_id'] not in actual;actual.add(d['consumer_id'])
    assert actual==expected and len(actual)==2956
    paths=[RUN/'private/candidates'/f.CANDIDATE/f'{owner}.pt' for owner in range(1,147)]
    assert all(p.is_file() and not p.is_symlink() and p.resolve().parent==(RUN/'private/candidates'/f.CANDIDATE).resolve() for p in paths)
    receipt=dict(status='CONSUMERS_COMPLETE_DELETION_PENDING',last_consumer='single generation and durable bindings',count=146,bytes=sum(p.stat().st_size for p in paths),paths=[str(p) for p in paths],no_retained_copy=True,rebuild_requires_training=True)
    write(RUN/'private/BALANCE_DELETION1.json',receipt)
    for path in paths:path.unlink()
    receipt.update(status='DELETED',epoch=time.time());write(RUN/'private/BALANCE_DELETION1.json',receipt)


def finish():
    root=RUN/'private'/f.QUEUE
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        f.project_budget(RUN)
        assert not list((root/'workers').glob('*/SCORER_FAILURE.json'))
        # Final DONE upload happens after the last scoring RPC; one status call closes the barrier.
        env=dict(os.environ,QUEUE_REQUEST=json.dumps(dict(action='status')))
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
        Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(TOOLS/'balance_queue.py'))+',run_name="__main__")\n')
        subprocess.run([sys.executable,entry],env=env,check=True,stdout=subprocess.DEVNULL)
        Path(entry).unlink();time.sleep(20)
    merge('Judge');assert launch('balance_report.py','balance_report',0,-1).wait()==0
    assert (RUN/'private/BALANCE_REPORT_COMPLETE1.json').exists()
    write(RUN/'private/BALANCE_COMPLETE1.json',dict(status='COMPLETE_PUBLICATION_PENDING',epoch=time.time()))

if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;write(RUN/'private/FAILURE_balance_controller1.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:
        for phase in ('GPU','Judge'):
            try:merge(phase)
            except BaseException as e:write(RUN/'private'/f'BALANCE_COST_TRANSFER_ERROR_{phase}.json',dict(error=repr(e)))
        write(RUN/'private/BALANCE_CONTROLLER_EXIT1.json',dict(exit_code=code,epoch=time.time()))
