"""Enforce the same experiment clock and local storage cap around isolated scoring."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def main():
    cfg=json.loads(Path(os.environ['SCORING_CONFIG']).read_text());local=Path(cfg['local'])
    local.mkdir(parents=True,exist_ok=True)
    def write(name,v):
        p=local/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
    p=subprocess.run(['ssh','pro5000','/data/bmw/envs/v0/bin/python','-'],input='from pathlib import Path\nprint((Path('+repr(cfg['remote'])+")/'CLOCK.json').read_text())",capture_output=True,text=True,timeout=30,check=True)
    clock=json.loads(p.stdout);write('CLOCK_LOCAL.json',clock)
    assert time.time()<clock['hard_end']-30
    with (local/'worker.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,os.environ['LOCAL_NEUTRAL_ENTRY']],env=dict(os.environ,LOCAL_ENTRY_MODE='score'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write('LOCAL_SCORER_IDENTITY.json',dict(pid=child.pid,time=time.time(),argv=[sys.executable,os.environ['LOCAL_NEUTRAL_ENTRY']]))
    try:
        while child.poll() is None:
            used=sum(p.stat().st_blocks*512 for p in local.rglob('*') if p.is_file() and not p.is_symlink())
            backup=Path(cfg['backup']);used+=backup.stat().st_blocks*512 if backup.exists() else 0
            # Reserve1GiB for this experiment's small source/config/report files.
            used+=1024**3
            free=os.statvfs(local).f_bavail*os.statvfs(local).f_frsize
            write('LOCAL_GUARD_HEALTH.json',dict(time=time.time(),new_bytes_upper_bound=used,free_bytes=free,hard_end=clock['hard_end']))
            assert used<47_000_000_000 and free>8*1024**3
            assert time.time()<clock['hard_end']-30
            time.sleep(5)
    except BaseException:
        write('LOCAL_GUARD_STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    write('LOCAL_GUARD_EXIT.json',dict(code=child.wait(),time=time.time(),original_clock=clock))
    raise SystemExit(child.returncode)

if __name__=='__main__':main()
