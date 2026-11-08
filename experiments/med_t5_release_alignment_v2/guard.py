"""One deadline/storage guard for the corrected cohort and four fixed methods."""
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


class Adopted:
    """Observe an existing own-method driver without reparenting/restarting it."""
    def __init__(self,pid,start_ticks,method):self.pid=pid;self.start_ticks=str(start_ticks);self.method=method
    def poll(self):
        p=Path('/proc')/str(self.pid)/'stat'
        if p.exists():
            text=p.read_text();fields=text[text.rfind(')')+2:].split()
            if fields[19]==self.start_ticks and fields[0]!='Z':return None
        return 0 if (ROOT/self.method/'T5/GENERATION_COMPLETE.json').exists() else 1
    def wait(self,timeout=None):
        start=time.time()
        while self.poll() is None:
            if timeout is not None and time.time()-start>=timeout:raise subprocess.TimeoutExpired('adopted driver',timeout)
            time.sleep(.2)
        return self.poll()


def write(n,v):
    p=ROOT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)


def main():
    cfg=json.loads((ROOT/'CONFIG.private.json').read_text());clock=json.loads((ROOT/'CLOCK.json').read_text())
    marker='GUARD_RECOVERY1_ONCE' if cfg.get('engineering_resume') else 'GUARD_ONCE'
    with (ROOT/marker).open('x'):pass
    children={m:Adopted(v['pid'],v['start_ticks'],m) for m,v in cfg.get('adopted',{}).items()}
    try:
        pending=[m for m in ['MDU','LORA','BALANCEDIT','GRACE'] if m not in children]
        while pending or any(p.poll() is None for p in children.values()):
            occupied={cfg['gpus'][m]['index'] for m,p in children.items() if p.poll() is None}
            for method in list(pending):
                gpu=cfg['gpus'][method]
                if gpu['index'] in occupied:continue
                row=subprocess.check_output(['nvidia-smi','-i',gpu['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
                assert row[0].strip()==gpu['uuid']
                if int(row[1])<60000:continue
                worker=cfg['workers'][method]
                with (ROOT/(method+'.log')).open('x') as log:
                    children[method]=subprocess.Popen([sys.executable,worker['entry']],env=dict(os.environ,**worker['env']),cwd='/data/bmw',
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3)))
                pending.remove(method);occupied.add(gpu['index'])
                write('IDENTITIES.private.json',dict(pids={m:p.pid for m,p in children.items()},pending=pending,time=time.time()))
            size=0
            for base in [ROOT,Path(cfg['neutral'])]:
                for p in base.rglob('*'):
                    try:
                        if p.is_file() and not p.is_symlink():size+=p.stat().st_blocks*512
                    except FileNotFoundError:pass
            s=os.statvfs(ROOT);write('HEALTH.json',dict(time=time.time(),bytes=size,free_bytes=s.f_bavail*s.f_frsize,hard_end=clock['hard_end'],pending=pending))
            assert size+2*1024**3<47_000_000_000 and s.f_bavail*s.f_frsize>8*1024**3 and time.time()<clock['hard_end']-1200
            time.sleep(5)
        write('ALL_EXIT.json',dict(codes={m:p.wait() for m,p in children.items()},time=time.time()))
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
