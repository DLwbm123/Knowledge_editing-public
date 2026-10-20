"""Independent finite-run guard with registered CPU/GPU deadlines."""
import os,json,time,signal,shutil,traceback
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def identity(target):
    try:
        text=(Path('/proc')/str(target['pid'])/'stat').read_text();tail=text.rsplit(')',1)[1].split()
        return tail[19]==str(target['start_ticks']) and os.getpgid(target['pid'])==target['pid'] and os.getsid(target['pid'])==target['pid']
    except (FileNotFoundError,ProcessLookupError):return False

def signal_owned(target,number):
    if not identity(target):return False
    os.killpg(target['pid'],number);return True

class Bytes:
    def __init__(self):self.cache={};self.total=0
    def measure(self,root,external):
        current={}
        for folder,_,names in os.walk(root):
            for name in names:
                path=Path(folder)/name
                try:
                    if not path.is_symlink():current[str(path)]=path.stat().st_size
                except FileNotFoundError:pass
        for path in external:
            path=Path(path)
            if path.exists() and not path.is_symlink():current[str(path)]=path.stat().st_size
        delta=sum(size-self.cache.get(path,0) for path,size in current.items())-sum(size for path,size in self.cache.items() if path not in current)
        self.total+=delta;self.cache=current
        assert self.total==sum(current.values())
        return self.total

def main(root):
    with (root/'RESOURCE_GUARD_ONCE').open('x'):pass
    limits=read(root/'RESOURCE_LIMITS.json');measure=Bytes();last_scan=0.;last_health=0.;trigger=None;sent_term=False;sent_kill=False
    while True:
        now=time.time();targets=read(root/'RESOURCE_TARGETS.json')['targets'];live=[t for t in targets if identity(t)]
        if not live and ((root/'RESOURCE_SCOPE_CLOSED.json').exists() or now>=limits['hard_end']-60):
            write(root/'RESOURCE_GUARD_EXIT.json',dict(status='ALL_REGISTERED_PROCESSES_EXITED',time=now,original_start=limits['original_start'],trigger=trigger));return
        if now-last_scan>=30 or not last_scan:
            size=measure.measure(root,limits['external_owned_paths']);last_scan=now
            if size>=limits['soft_stop_bytes'] or shutil.disk_usage(root).free<limits['free_reserve_bytes']:
                trigger=trigger or dict(reason='OWNED_STORAGE_OR_FREE_RESERVE',time=now,observed_owned_bytes=size)
        if (root/'STOP').exists():
            trigger=trigger or dict(reason='EXTERNAL_STOP',time=now)
        for target in live:
            if now>=target.get('deadline',limits['hard_end'])-(30 if target.get('phase')=='cpu' else 300):
                trigger=trigger or dict(reason='REGISTERED_PROCESS_WINDOW',time=now,pid=target['pid'])
        if now>=limits['hard_end']-limits['save_margin_seconds']:
            trigger=trigger or dict(reason='ORIGINAL_24H_CLOCK',time=now)
        if trigger:
            stop=root/'STOP'
            if not stop.exists():stop.write_text(json.dumps(trigger)+'\n')
            write(root/'RESOURCE_STOP_REQUEST.json',dict(trigger=trigger,live_targets=live,hard_end=limits['hard_end'],current_owned_bytes=measure.total))
            if not sent_term and now>=trigger['time']+30:
                signals=[dict(pid=t['pid'],signalled=signal_owned(t,signal.SIGTERM)) for t in live];write(root/'RESOURCE_SIGTERM.json',dict(time=now,targets=signals));sent_term=True
            if not sent_kill and (now>=trigger['time']+60 or now>=limits['hard_end']-60):
                signals=[dict(pid=t['pid'],signalled=signal_owned(t,signal.SIGKILL)) for t in live];write(root/'RESOURCE_SIGKILL.json',dict(time=now,targets=signals));sent_kill=True
        if now-last_health>=30:
            write(root/'RESOURCE_GUARD_HEALTH.json',dict(time=now,status='STOPPING' if trigger else 'HEALTHY',original_start=limits['original_start'],hard_end=limits['hard_end'],remaining_wall_seconds=limits['hard_end']-now,owned_bytes=measure.total,soft_stop_bytes=limits['soft_stop_bytes'],hard_storage_bytes=limits['hard_storage_bytes'],live_targets=live));last_health=now
        time.sleep(2)

if __name__=='__main__':
    root=Path(os.environ['V25_ROOT'])
    try:main(root)
    except BaseException:
        write(root/'RESOURCE_GUARD_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
