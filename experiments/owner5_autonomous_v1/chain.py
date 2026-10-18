"""Run the two frozen diagnostics once, sequentially, preserving failures and costs."""
import json, os, subprocess, sys, time
from pathlib import Path
root=Path(os.environ['CHAIN_ROOT']); start=time.time()
def write(name,value):
    target=root/name;temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
with (root/'CHAIN_ONCE').open('x'):pass
phases=[]
for phase in ('reserve','trajectory'):
    if time.time()>=float(os.environ['DEADLINE_EPOCH']):
        write('CHAIN_COMPLETE.json',dict(status='DEADLINE',phases=phases,seconds=time.time()-start));break
    directory=root/phase
    env=dict(os.environ,OUTPUT_RUN=str(directory))
    with (directory/'tools'/('run.py')).open('rb') as source,(directory/'run.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u','-'],stdin=source,stdout=log,stderr=subprocess.STDOUT,env=env)
        ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]
        write('CURRENT.json',dict(phase=phase,pid=child.pid,start_ticks=ticks,start=time.time()))
        phase_start=time.time();code=child.wait()
    phases.append(dict(phase=phase,exit_code=code,seconds=time.time()-phase_start))
    write('CHAIN_PROGRESS.json',dict(phases=phases,seconds=time.time()-start))
    if code:
        write('CHAIN_COMPLETE.json',dict(status='FAILED',phases=phases,seconds=time.time()-start));break
else:write('CHAIN_COMPLETE.json',dict(status='COMPLETE',phases=phases,seconds=time.time()-start,automatic_successor=False))
