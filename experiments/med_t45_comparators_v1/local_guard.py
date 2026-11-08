"""Bound all local comparator evidence/scorers by one storage limit and clock."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

cfg=json.loads(Path(os.environ['COMPARE_LOCAL_CONFIG']).read_text())
control=Path(cfg['control']);private=Path(cfg['private_root']);pid=int(os.environ['COMPARE_CONTROLLER_PID'])
clock=json.loads((control/'clock.json').read_text())


def write(name,value):
    p=control/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)


while True:
    table={}
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines():
        pieces=line.strip().split(None,2)
        if len(pieces)==3:table[int(pieces[0])]=(int(pieces[1]),pieces[2])
    if pid not in table or str(control/'q.py') not in table[pid][1]:break
    descendants={pid}
    while True:
        grown=descendants|{child for child,(parent,_) in table.items() if parent in descendants}
        if grown==descendants:break
        descendants=grown
    size=0
    for root in [control,private]:
        for p in root.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink():size+=p.stat().st_blocks*512
            except FileNotFoundError:pass
    s=os.statvfs(control)
    write('RESOURCE_HEALTH.json',dict(time=time.time(),bytes=size,free_bytes=s.f_bavail*s.f_frsize,hard_end=clock['hard_end']))
    if size+128*1024**2>=2_000_000_000 or s.f_bavail*s.f_frsize<=8*1024**3 or time.time()>=clock['hard_end']-45:
        write('RESOURCE_STOP.json',dict(time=time.time(),bytes=size,reason='local storage/free-space/original clock limit'))
        for child in sorted(descendants,reverse=True):
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        break
    time.sleep(5)
write('RESOURCE_GUARD_EXIT.json',dict(time=time.time()))
