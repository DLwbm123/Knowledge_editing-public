"""Independent time and storage guard for one local-gradient diagnostic."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

root=Path(os.environ['DIAG_ROOT'])
hard_end=float(os.environ['DIAG_HARD_END'])

def write(name, data):
    path=root/name; temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def allocated():
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total

def main():
    assert time.time()<hard_end-60
    with (root/'GUARD_ONCE').open('x'):pass
    with (root/'worker.log').open('a') as log:
        child=subprocess.Popen([sys.executable,os.environ['NEUTRAL_ENTRY']],env=dict(os.environ,ENTRY_MODE='worker'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while child.poll() is None:
            used=allocated();s=os.statvfs(root);free=s.f_bavail*s.f_frsize
            write('GUARD_HEALTH.json',{'time':time.time(),'bytes':used,'free':free,'hard_end':hard_end})
            if used>=900_000_000 or free<=8*1024**3 or time.time()>=hard_end-30:
                raise RuntimeError('resource or deadline stop')
            time.sleep(5)
    except BaseException:
        write('GUARD_FAILURE.json',{'error':traceback.format_exc(),'time':time.time()})
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait();write('GUARD_EXIT.json',{'code':code,'time':time.time()})
    return code

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        write('GUARD_FAILURE.json',{'error':traceback.format_exc(),'time':time.time()})
        write('GUARD_EXIT.json',{'code':1,'time':time.time()});raise
