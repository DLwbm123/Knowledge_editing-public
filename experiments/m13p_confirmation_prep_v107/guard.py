"""One worker, inheriting the original clock, total space and GPU ledger."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
ROOT=Path(os.environ['V107_ROOT'])
cfg=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
assert cfg['gpu_time_limit_enabled'] is False


def write(name,data):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)


def main():
    with (ROOT/'GUARD_ONCE').open('x'):pass
    uuid,free=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    assert uuid==os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID'] and int(free)>=45*1024
    child=None;started=None
    try:
        started=time.time()
        write('GUARD_HEALTH.json',dict(time=started,gpu_seconds=cfg['prior_gpu_seconds']))
        with (ROOT/'worker.log').open('x') as log:
            child=subprocess.Popen(['/data/bmw/envs/v0/bin/python',os.environ.get('NEUTRAL_ENTRY','/data/bmw/.q.20261024.107/q.py')],
                env=dict(os.environ,ACTION_107='worker'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('WORKER_IDENTITY.json',dict(pid=child.pid,start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],start=started))
        while child.poll() is None:
            now=time.time();elapsed=now-started
            size=sum(p.stat().st_size for folder in cfg['guarded_roots'] for p in Path(folder).rglob('*') if p.is_file() and not p.is_symlink())
            own=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
            write('GUARD_HEALTH.json',dict(time=now,gpu_seconds=cfg['prior_gpu_seconds']+elapsed,phase_gpu_seconds=elapsed,bytes=size,phase_bytes=own))
            if now>=cfg['hard_end']-15 or elapsed>=86400-15 or size>=cfg['storage_limit_bytes']-100000000 or own>=cfg['phase_storage_limit_bytes']:
                write('STOP.json',dict(time=now,reason='WALL_OR_STORAGE_LIMIT_GPU_CUMULATIVE_CAP_DISABLED'));break
            time.sleep(5)
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            elapsed=time.time()-started
            write('GUARD_EXIT.json',dict(time=time.time(),code=child.returncode,phase_gpu_seconds=elapsed,
                prior_gpu_seconds=cfg['prior_gpu_seconds'],total_gpu_seconds=cfg['prior_gpu_seconds']+elapsed))


if __name__=='__main__':main()
