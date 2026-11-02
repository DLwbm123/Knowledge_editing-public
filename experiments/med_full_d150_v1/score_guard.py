"""Independent local guard for frozen-payload score reuse; no new judge requests."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

root=Path(os.environ['LOCAL_SCORE_ROOT'])
deadline=float(os.environ['EXPERIMENT_DEADLINE'])


def write(name,value):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)


def allocated():
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total


def main():
    with (root/'GUARD_ONCE').open('x'):pass
    def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
    with (root/'worker.log').open('x') as log:
        child=subprocess.Popen([os.environ['LOCAL_PYTHON'],os.environ['NEUTRAL_SCORE_WORKER']],env=os.environ.copy(),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit)
    try:
        while child.poll() is None:
            used=allocated();fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize
            write('GUARD_HEALTH.json',{'time':time.time(),'bytes':used,'hard_end':deadline,'free':free})
            if used>=2_000_000_000 or free<8*1024**3 or time.time()>=deadline-60:
                raise RuntimeError('local resource or original deadline')
            time.sleep(5)
    except BaseException:
        write('GUARD_FAILURE.private.json',{'traceback':traceback.format_exc()})
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait();write('GUARD_EXIT.json',{'code':code,'time':time.time()})
    if code:raise SystemExit(code)


if __name__=='__main__':main()
