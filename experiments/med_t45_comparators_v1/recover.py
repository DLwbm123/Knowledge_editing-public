"""One zero-edit dependency recovery, independently bounded by the original clock."""
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


def main():
    cfg=json.loads((ROOT/'CONFIG.private.json').read_text());clock=json.loads((ROOT/'CLOCK.json').read_text())
    job=ROOT/'LORA';archive=job/'engineering_recovery1';archive.mkdir()
    failure=json.loads((job/'FAILURE.json').read_text())
    assert "No module named 'peft'" in failure['error']
    assert not list(job.glob('T*/training/*.json')) and not list(job.glob('T*/outputs/*.json'))
    assert not (job/'METHOD_CONFIG.json').exists()
    for name in ['FAILURE.json','EXIT.json','WORKER_ONCE','runtime']:
        p=job/name
        if p.exists():p.rename(archive/name)
    gpu=cfg['gpus']['LORA']
    row=subprocess.check_output(['nvidia-smi','-i',gpu['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
    assert row[0].strip()==gpu['uuid'] and int(row[1])>=60000
    env=dict(os.environ,COMPARE_METHOD='LORA',COMPARE_HEALTH_FILE='LORA_RECOVERY_HEALTH.json',CURRENT_SOURCE=str(ROOT/'tools/worker_recovery1.py'),
        CUDA_VISIBLE_DEVICES=gpu['index'],M3BENCH_FORMAL_EXPECTED_GPU_UUID=gpu['uuid'],M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=gpu['index'],
        M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=gpu['index'])
    child=None
    try:
        with (ROOT/'LORA_RECOVERY.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(cfg['neutral'])/'x.py')],env=env,cwd='/data/bmw',stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3)))
        write('LORA_RECOVERY_START.private.json',dict(pid=child.pid,time=time.time(),original_clock=clock,completed_edits_before_recovery=0))
        while child.poll() is None:
            size=0
            for root in [ROOT,Path(cfg['neutral'])]:
                for p in root.rglob('*'):
                    try:
                        if p.is_file() and not p.is_symlink():size+=p.stat().st_blocks*512
                    except FileNotFoundError:pass
            s=os.statvfs(ROOT)
            write('LORA_RECOVERY_HEALTH.json',dict(time=time.time(),bytes=size,free_bytes=s.f_bavail*s.f_frsize,hard_end=clock['hard_end']))
            assert size+2*1024**3<47_000_000_000 and s.f_bavail*s.f_frsize>8*1024**3 and time.time()<clock['hard_end']-1200
            time.sleep(5)
        write('LORA_RECOVERY_EXIT.json',dict(code=child.wait(),time=time.time(),clock=clock))
    except BaseException:
        write('LORA_RECOVERY_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()))
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise


if __name__=='__main__':main()
