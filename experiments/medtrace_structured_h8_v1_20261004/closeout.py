"""Finite CPU dependency waiter; report only after complete accepted scoring."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

RUN=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    with (RUN/'private/CLOSEOUT.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (RUN/'private/judge_common/SCORER_DONE.json').exists():
            if (RUN/'STOP').exists() or time.time()>=read(RUN/'RUN_MANIFEST.json')['deadline_epoch']:return
            if any(p.exists() for p in [RUN/'private/CONTROLLER_FAILURE.json',RUN/'private/judge_common/SCORER_FAILURE.json']):
                write(RUN/'private/CLOSEOUT_WAIT_FAILURE.json',dict(status='REPAIR_REQUIRED',epoch=time.time()));return
            time.sleep(30)
        assert read(RUN/'private/SCORING_IMPLEMENTATION_ADMISSION.json')['status']=='PASS'
        assert all(x.get('ended_epoch') for x in read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
        if (RUN/'private/REPORT_COMPLETE.json').exists():return
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py');os.close(fd)
        Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/reporting.py',run_name='__main__')\n")
        subprocess.run(['/data/bmw/envs/v0/bin/python','-u',entry],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4'),stdout=(RUN/'logs/reporting.log').open('ab'),stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(RUN/'private/CLOSEOUT_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),epoch=time.time()));raise
