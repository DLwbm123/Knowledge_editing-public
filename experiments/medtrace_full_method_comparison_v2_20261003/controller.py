"""Dependency-ordered background generation; no scientific retries or extra arms."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

RUN=Path(os.environ['RUN_ROOT'])


def read(p):
    return json.loads(Path(p).read_text())


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)


def start(action,gpu,partition=None):
    entry=Path(tempfile.mktemp(prefix='e.',suffix='.py'))
    entry.write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/source')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/source/worker.py',run_name='__main__')\n")
    env=dict(os.environ,ACTION=action,GPU=str(gpu))
    if partition is not None:env['PARTITION']=str(partition)
    path=RUN/'logs'/f'{action}_{gpu}.log';f=path.open('ab')
    job=subprocess.Popen(['/data/bmw/envs/v0/bin/python','-u',str(entry)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state=dict(pid=job.pid,entry=str(entry),action=action,gpu=gpu,partition=partition,start_epoch=time.time(),start_ticks=Path(f'/proc/{job.pid}/stat').read_text().split()[21],argv=['/data/bmw/envs/v0/bin/python','-u',str(entry)])
    write(RUN/'private'/f'START_{action}_{gpu}.json',state)
    print('START',action,gpu,job.pid,flush=True)
    return job,state


def wait(jobs):
    failure_reported=False
    while any(p.poll() is None for p,_ in jobs):
        manifest=read(RUN/'RUN_MANIFEST.json');ledger=read(RUN/'RESOURCE_LEDGER.json')
        used=ledger['gpu_seconds_used']+sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if s.get('ended_epoch') is None)
        stop=(RUN/'STOP').exists() or time.time()>=manifest['deadline_epoch'] or used>=manifest['GPU_seconds_limit']
        if stop:
            write(RUN/'private/BUDGET_STOP.json',dict(epoch=time.time(),gpu_seconds_including_live=used))
            for p,s in jobs:
                if p.poll() is None and Path(f'/proc/{p.pid}/stat').read_text().split()[21]==s['start_ticks']:
                    os.killpg(p.pid,signal.SIGTERM)
            raise TimeoutError('Persisted stop/deadline/budget; own groups stopped only')
        failed=[(p,s) for p,s in jobs if p.poll() not in (None,0)]
        if failed and not failure_reported:
            # Healthy independent work finishes while the same watchdog keeps its budget active.
            write(RUN/'public/PROGRESS.json',dict(status='FAILED_PARTITION_OTHER_HEALTHY_ALLOWED_TO_FINISH',failed=[s for _,s in failed]))
            failure_reported=True
        time.sleep(15)
    for p,s in jobs:
        if p.returncode!=0:raise RuntimeError('Worker failed '+str(p.returncode))


def main():
    assert read(RUN/'private/CPU_ADMISSION.json')['status']=='PASS'
    assert not (RUN/'private/CONTROLLER_COMPLETE.json').exists()
    write(RUN/'public/PROGRESS.json',dict(status='GPU_MECHANICAL_STARTING',N=146,H_covered=8,GPU_allowlist=[4,5]))
    jobs=[start('mechanical',4)];wait(jobs)
    assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
    write(RUN/'public/PROGRESS.json',dict(status='SINGLE_RUNNING',N=146,H_covered=8,GPU_allowlist=[4,5]))
    jobs=[start('single',4,0),start('single',5,1)];wait(jobs)
    jobs=[start('sequential',4)];wait(jobs)
    assert read(RUN/'private/GENERATION_COMPLETE.json')['status']=='GENERATED_NOT_SCORED'
    write(RUN/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_JUDGE_AND_PUBLICATION_PENDING',N=146))
    write(RUN/'private/CONTROLLER_COMPLETE.json',dict(status='GENERATION_COMPLETE',epoch=time.time(),remaining=['baseline binding admission','common new Judge epoch','full metrics','public delivery']))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(RUN/'private/CONTROLLER_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),epoch=time.time(),pid=os.getpid()));raise
