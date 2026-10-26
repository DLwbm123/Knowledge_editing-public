"""Temporarily time-share one existing GPU, then always restore its worker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def same(pid, ticks):
    p = Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split()[21] == ticks


def main():
    cfg = json.loads(Path(os.environ['AUDIT_DISPATCH_CONFIG']).read_text())
    out = Path(cfg['out'])
    assert same(cfg['helper_pid'],cfg['helper_ticks'])
    assert os.getpgid(cfg['helper_pid']) == cfg['primary_pid']
    child, started, gpu_started = None,time.time(),None
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('Diagnostic stop')))
    os.kill(cfg['helper_pid'],signal.SIGSTOP)
    write(out/'HELPER_PAUSED.json',dict(pid=cfg['helper_pid'],start_ticks=cfg['helper_ticks'],time=started))
    try:
        free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'],text=True).strip())
        assert free >= 36*1024, 'Insufficient diagnostic headroom; existing worker will resume'
        with (out/'worker.log').open('x') as log:
            gpu_started = time.time()
            child = subprocess.Popen([sys.executable,'-u',cfg['entry']],stdout=log,
                      stderr=subprocess.STDOUT,start_new_session=True)
        write(out/'START.json',dict(time=time.time(),pid=child.pid,
              start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],GPU=0,
              shared_physical_GPU=True,maximum_total_physical_GPUs=3,free_MiB_before=free))
        child.wait(timeout=1800)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        cancel = out.parent/'CAP3_CANCEL.json'
        if same(cfg['helper_pid'],cfg['helper_ticks']) and not cancel.exists():
            os.kill(cfg['helper_pid'],signal.SIGCONT)
            write(out/'HELPER_RESUMED.json',dict(pid=cfg['helper_pid'],time=time.time()))
        write(out/'END.json',dict(time=time.time(),seconds=time.time()-started,
              diagnostic_GPU_process_seconds=time.time()-gpu_started if child else 0,
              child_exit=child.returncode if child else None))
    assert child is not None and child.returncode == 0


if __name__ == '__main__':
    main()
