"""Run two pre-registered independent diagnostics on physical GPUs zero and one."""
import json, os, subprocess, sys, time
from pathlib import Path
root=Path(os.environ['CHAIN_ROOT']);start=time.time()
def write(name,value):
    target=root/name;temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
with (root/'PARALLEL_ONCE').open('x'):pass
children=[]
for gpu,phase in enumerate(('reserve','trajectory')):
    assert time.time()<float(os.environ['DEADLINE_EPOCH'])
    free=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert int(free[1])>=46080
    directory=root/phase;env=dict(os.environ,OUTPUT_RUN=str(directory),CUDA_VISIBLE_DEVICES=str(gpu))
    with (directory/'tools/run.py').open('rb') as source,(directory/'run.log').open('ab') as log:
        child=subprocess.Popen([sys.executable,'-u','-'],stdin=source,stdout=log,stderr=subprocess.STDOUT,env=env)
    record=dict(phase=phase,pid=child.pid,start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],start=time.time(),gpu=gpu,uuid=free[0].strip(),free_MiB=int(free[1]))
    children.append((child,record));write('START_'+phase+'.json',record)
write('PARALLEL_START.json',dict(children=[v for _,v in children],start=start))
phases=[]
for child,record in children:
    code=child.wait();phases.append(dict(**record,exit_code=code,seconds=time.time()-record['start']))
    write('CHAIN_PROGRESS.json',dict(phases=phases,seconds=time.time()-start))
write('CHAIN_COMPLETE.json',dict(status='COMPLETE' if all(v['exit_code']==0 for v in phases) else 'FAILED',phases=phases,seconds=time.time()-start,automatic_successor=False))
