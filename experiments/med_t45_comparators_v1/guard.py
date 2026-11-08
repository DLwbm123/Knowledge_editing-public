"""Independent shared clock/storage guard; isolate method-specific failures."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['COMPARE_ROOT'])


def write(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)


def used(root):
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total


def main():
    cfg=json.loads((ROOT/'CONFIG.private.json').read_text());clock=json.loads((ROOT/'CLOCK.json').read_text())
    with (ROOT/'GUARD_ONCE').open('x'):pass
    children={}
    try:
        for method,gpu in cfg['gpus'].items():
            row=subprocess.check_output(['nvidia-smi','-i',gpu['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
            assert row[0].strip()==gpu['uuid'] and int(row[1])>=60000
            env=dict(os.environ,COMPARE_METHOD=method,CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_EXPECTED_GPU_UUID=gpu['uuid'],
                M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu['index'],
                CURRENT_SOURCE=str(ROOT/'tools/worker.py'))
            with (ROOT/(method+'.log')).open('x') as log:
                children[method]=subprocess.Popen([sys.executable,str(Path(cfg['neutral'])/'w.py')],env=env,cwd='/data/bmw',
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3)))
        write('IDENTITIES.private.json',dict(time=time.time(),pids={k:p.pid for k,p in children.items()}))
        while any(p.poll() is None for p in children.values()):
            fs=os.statvfs(ROOT);size=used(ROOT)+used(Path(cfg['neutral']))
            write('HEALTH.json',dict(time=time.time(),bytes=size,free_bytes=fs.f_bavail*fs.f_frsize,hard_end=clock['hard_end']))
            assert size+2*1024**3<47_000_000_000 and fs.f_bavail*fs.f_frsize>8*1024**3
            assert time.time()<clock['hard_end']-1200
            time.sleep(5)
        write('ALL_EXIT.json',dict(time=time.time(),codes={k:p.wait() for k,p in children.items()}))
    except BaseException:
        write('STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        for p in children.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children.values():
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise


if __name__=='__main__':main()
