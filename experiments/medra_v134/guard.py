"""Hard resource guard for the finite new run; no recurring chat monitor."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT'])


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
    lock,clock=read('PROTOCOL_LOCK.json'),read('CLOCK.json')
    gpu=lock['selected_physical_GPU']
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.free',
                                  '--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert int(row[0])==gpu and row[1].strip()==lock['expected_GPU_UUID']
    assert int(row[2])>=lock['minimum_free_MiB']
    assert read('CPU_ADMISSION.json')['status']=='PASS'
    if (ROOT/'GUARD_ONCE').exists():
        request=read('RECOVERY_REQUEST.json')
        assert request['old_worker_exited'] and request['old_guard_exited'] and request['clock']==clock
        assert request['input_binding']==lock['input_binding']
        with (ROOT/f"GUARD_RECOVERY_{request['sequence']}").open('x'):pass
    else:
        with (ROOT/'GUARD_ONCE').open('x'):pass
    assert time.time()<clock['hard_end']-620
    write('GUARD_HEALTH.json',dict(time=time.time(),new_bytes=size(ROOT),hard_end=clock['hard_end']))
    with (ROOT/'worker.log').open('a') as log:
        child=subprocess.Popen([sys.executable,os.environ['NEUTRAL_ENTRY']],env=dict(os.environ,ENTRY_MODE='worker'),
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19]
    write('WORKER_IDENTITY.json',dict(pid=child.pid,start_ticks=ticks,time=time.time(),GPU=gpu))
    code=1
    try:
        while child.poll() is None:
            used=size(ROOT)+size(Path(os.environ['NEUTRAL_ENTRY']).parent)
            free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
            write('GUARD_HEALTH.json',dict(time=time.time(),new_bytes=used,free_bytes=free,hard_end=clock['hard_end']))
            assert used<47_000_000_000 and free>8*1024**3 and time.time()<clock['hard_end']-45
            assert not (ROOT/'SCORE_FAILURE.json').exists()
            time.sleep(5)
    except BaseException:
        write('STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait()
    if code==0:
        while not (ROOT/'FINAL_REPORT_COMPLETE.json').exists():
            if time.time()>=clock['hard_end']-30:
                write('STOP.json',dict(time=time.time(),error='Final scoring original-clock deadline'));code=124;break
            time.sleep(5)
    write('GUARD_EXIT.json',dict(code=code,time=time.time(),original_clock=clock))
    raise SystemExit(code)


if __name__=='__main__':
    try:main()
    except Exception:
        write('GUARD_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        if not (ROOT/'GUARD_EXIT.json').exists():write('GUARD_EXIT.json',dict(time=time.time(),code=1))
        raise
