"""Hard resource and wall-clock guard for an additional GPU stage."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT']);STAGE=os.environ['STAGE_KIND']
assert STAGE in ('BASE','EVAL')


def read(name):return json.loads((ROOT/name).read_text())
def write(name,value):
    p=ROOT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def size(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total


def main():
    lock=read(STAGE+'_LOCK.json');clock=read('CLOCK.json');gpu=lock['physical_GPU']
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==lock['expected_GPU_UUID'] and int(row[1])>=lock['minimum_free_MiB']
    assert read(STAGE+'_CPU_ADMISSION.json')['status']=='PASS' and time.time()<clock['hard_end']-620
    with (ROOT/(STAGE+'_GUARD_ONCE')).open('x'):pass
    write(STAGE+'_GUARD_HEALTH.json',dict(time=time.time(),new_bytes=size(ROOT),hard_end=clock['hard_end']))
    with (ROOT/(STAGE.lower()+'.log')).open('a') as log:
        child=subprocess.Popen([sys.executable,os.environ['NEUTRAL_ENTRY']],env=dict(os.environ,ENTRY_MODE='worker'),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19]
    write(STAGE+'_WORKER_IDENTITY.json',dict(pid=child.pid,start_ticks=ticks,time=time.time(),GPU=gpu))
    try:
        while child.poll() is None:
            used=size(ROOT)+size(Path(os.environ['NEUTRAL_ENTRY']).parent)
            free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
            write(STAGE+'_GUARD_HEALTH.json',dict(time=time.time(),new_bytes=used,free_bytes=free,hard_end=clock['hard_end']))
            assert used<47_000_000_000 and free>8*1024**3 and time.time()<clock['hard_end']-45
            time.sleep(5)
    except BaseException:
        write(STAGE+'_STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait();write(STAGE+'_GUARD_EXIT.json',dict(time=time.time(),code=code,original_clock=clock))
    raise SystemExit(code)


if __name__=='__main__':
    try:main()
    except Exception:
        write(STAGE+'_GUARD_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        if not (ROOT/(STAGE+'_GUARD_EXIT.json')).exists():write(STAGE+'_GUARD_EXIT.json',dict(time=time.time(),code=1))
        raise
