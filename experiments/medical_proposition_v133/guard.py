"""GPU0 admission and a shared24h finite guard independent of chat/SSH."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['EXPERIMENT_ROOT'])


def read(p):return json.loads(Path(p).read_text())
def write(name,v):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def size(p):return sum(v.stat().st_blocks*512 for v in p.rglob('*') if v.is_file() and not v.is_symlink())


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    row=[v.strip() for v in gpu.splitlines()[0].split(',')]
    lock=read(ROOT/'PROTOCOL_LOCK.json')
    assert row[0]=='0' and row[1]==os.environ['M3BENCH_FORMAL_EXPECTED_GPU_UUID']
    assert int(row[2])>=lock['minimum_free_MiB'],('GPU0 insufficient memory',row)
    dependency=read(ROOT/'MEMIT_DEPENDENCY_TERMINAL.json')
    assert dependency['generation_terminal'] and dependency['scoring_terminal']
    assert read(ROOT/'DATA_ADMISSION.json')['status']=='PASS'
    assert read(ROOT/'CPU_ADMISSION.json')['status']=='PASS'
    recovering=(ROOT/'GUARD_ONCE').exists()
    if recovering:
        request=read(ROOT/'RECOVERY_REQUEST.json');clock=read(ROOT/'CLOCK.json')
        assert request['clock']==clock and request['input_binding']==lock['input_binding']
        assert request['old_worker_exited'] and request['old_guard_exited']
        with (ROOT/('RECOVERY_GUARD_'+str(request['sequence']))).open('x'):pass
        start,end=clock['original_start'],clock['hard_end']
        assert time.time()<end-620 and not (ROOT/'STOP.json').exists()
    else:
        with (ROOT/'GUARD_ONCE').open('x'):pass
        start=time.time();end=start+86400
        write('CLOCK.json',dict(original_start=start,hard_end=end,retries_share_clock=True))
    other_bytes=int(dependency['prior_managed_bytes'])
    budget=lock['budgets']
    initial=dict(time=time.time(),new_bytes=size(ROOT),prior_managed_new_bytes=other_bytes)
    write('GUARD_HEALTH.json',initial)
    env=dict(os.environ,ENTRY_MODE='worker')
    with (ROOT/'worker.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,os.environ['NEUTRAL_ENTRY']],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19]
    write('WORKER_IDENTITY.json',dict(pid=child.pid,start_ticks=ticks,time=start,start=start))
    try:
        while child.poll() is None:
            used=size(ROOT)+size(Path(os.environ['NEUTRAL_ENTRY']).parent);free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
            write('GUARD_HEALTH.json',dict(time=time.time(),new_bytes=used,prior_managed_new_bytes=other_bytes,
                global_managed_new_bytes=used+other_bytes,free_bytes=free,hard_end=end,elapsed=time.time()-start))
            assert used<budget['new_bytes_limit']-100_000_000
            assert used+other_bytes<47_000_000_000
            assert free>budget['minimum_free_bytes']
            assert time.time()<end-45
            time.sleep(5)
    except BaseException:
        write('STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    code=child.wait()
    # The experiment clock continues through final scoring, even after GPU exit.
    if code==0:
        while not (ROOT/'FINAL_REPORT_COMPLETE.json').exists():
            if time.time()>=end-30:
                write('STOP.json',dict(time=time.time(),error='Final scoring shared wall budget'));code=124;break
            used=size(ROOT)+size(Path(os.environ['NEUTRAL_ENTRY']).parent)
            if used+other_bytes>=47_000_000_000:
                write('STOP.json',dict(time=time.time(),error='Storage pre-stop'));code=125;break
            write('GUARD_HEALTH.json',dict(time=time.time(),new_bytes=used,hard_end=end,phase='FINAL_SCORING'))
            time.sleep(5)
    write('GUARD_EXIT.json',dict(code=code,time=time.time(),elapsed=time.time()-start))
    raise SystemExit(code)

if __name__=='__main__':main()
