"""Wait for predecessor ownership release, then run the finite independent pilot."""
import os,time,json,fcntl,subprocess,shutil,traceback
from pathlib import Path
from resources import ROOT,read,write
from storage import Store

def live(pid,root,entry=None):
 try:
  p=Path('/proc')/str(pid)
  return p.joinpath('stat').read_text().split()[2]!='Z' and p.joinpath('cwd').resolve()==root.resolve() and (entry is None or entry in p.joinpath('cmdline').read_bytes().decode().split('\0'))
 except (FileNotFoundError,PermissionError):return False

def released(old):
 if not (old/'E4_ACTIVATION.json').exists() or not (old/'SCORER_DONE').exists():return False
 if any(j['status']!='COMPLETE' for j in read(old/'QUEUE.json')):return False
 if any(not s.get('ended_epoch') for s in read(old/'RESOURCE_LEDGER.json')['gpu_sessions']):return False
 for n in ['ORCHESTRATOR_PID.json','ACTIVE_PROCESSES.json']:
  if (old/n).exists():
   x=read(old/n)
   if any(live(i['pid'],old) for i in (x if isinstance(x,list) else [x])):return False
 return True

def activate(old):
 # Snapshot counters and terminal payloads only after old ownership is released.
 assert released(old)
 d=read(old/'RESOURCE_LEDGER.json');assert all(a['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for a in d['judge_attempts'])
 write(ROOT/'PREDECESSOR_FINAL_LEDGER.json',d)
 d.update(historical_gpu_seconds=d['lifetime_gpu_seconds'],current_gpu_seconds=0,gpu_sessions=[],historical_judge_attempts=d['judge_submission_attempt_items'],current_judge_attempts=0,predecessor_attempt_limit=d['judge_submission_attempt_items_limit'],judge_submission_attempt_items_limit=None,physical_requests_limit=None)
 write(ROOT/'RESOURCE_LEDGER.json',d)
 for n in ['JUDGE_MISSING_LOCK.json']:
  shutil.copy2(old/'private/judge'/n,ROOT/'private/judge'/n)
 for p in (old/'private/judge/scores').glob('*.json'):shutil.copy2(p,ROOT/'private/judge/scores'/p.name)
 write(ROOT/'LEASE_ACQUIRED.json',dict(epoch=time.time(),predecessor_released=True,devices=[5,6,7],phase='P1_DEV8'))
 write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='P1_DEV8',epoch=time.time()))

def main():
 f=(ROOT/'CONTROLLER.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(ROOT/'ORCHESTRATOR_PID.json',dict(pid=os.getpid(),entry='/tmp/s8.py',epoch=time.time()))
 old=Path(read(ROOT/'PREDECESSOR.json')['root'])
 while not (ROOT/'LEASE_ACQUIRED.json').exists():
  if (ROOT/'STOP').exists() or (old/'STOP').exists():write(ROOT/'RUN_STATUS.json',dict(status='USER_STOPPED'));return
  if released(old):activate(old);break
  write(ROOT/'RUN_STATUS.json',dict(status='READY_WAITING_FOR_LEASE',gpu_started=False,phase='P1_DEV8',epoch=time.time(),reason='E3/E4 ownership reserved'));time.sleep(30)
 active=read(ROOT/'ACTIVE_PROCESSES.json') if (ROOT/'ACTIVE_PROCESSES.json').exists() else []
 while True:
  if (ROOT/'STOP').exists() or (old/'STOP').exists():
   (ROOT/'STOP').touch();write(ROOT/'RUN_STATUS.json',dict(status='USER_STOPPED'));return
  if (ROOT/'JUDGE_FAILURE.json').exists():
   (ROOT/'ADMISSION_STOP').touch();write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',reason='JUDGE_FAILURE_FINISH_CURRENT_UNIT'));return
  q=read(ROOT/'QUEUE.json')
  for a in active[:]:
   if live(a['pid'],ROOT,'/tmp/s7.py'):continue
   active.remove(a)
   unfinished=[j for j in q if j.get('pid')==a['pid'] and j['status']=='RUNNING']
   if unfinished:
    receipt=read(ROOT/'jobs'/f'WORKER_{a["index"]}.json')
    if receipt.get('pid')==a['pid'] and 'GPU lease limit' in receipt.get('error',''):
     with (ROOT/'QUEUE.lock').open('a') as f:
      fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json')
      for j in q:
       if j.get('pid')==a['pid'] and j['status']=='RUNNING':j.update(status='PENDING',resume_gpu=a['index'])
      write(ROOT/'QUEUE.json',q)
    else:
     (ROOT/'ADMISSION_STOP').touch();write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',reason='WORKER_FAILURE',pid=a['pid']));return
  write(ROOT/'ACTIVE_PROCESSES.json',active)
  if all(j['status']=='COMPLETE' for j in q) and not active:
   write(ROOT/'RUN_STATUS.json',dict(status='GPU_QUEUE_FINISHED',phase='P1_DEV8',epoch=time.time(),next='CHECK lambda selection and full DEV24 admission after audit'));return
  if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('Less than 20 GiB physical free space')
  for g in read(ROOT/'GPU_BINDINGS.json')['devices']:
   if any(a['index']==g['index'] for a in active):continue
   if not any(j['status']=='PENDING' and j.get('resume_gpu',g['index'])==g['index'] and all((ROOT/x).exists() for x in j['requires']) for j in q):continue
   free=int(subprocess.check_output(['nvidia-smi','-i',str(g['index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if free<30000:continue
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],PINNED_GPU_UUID=g['uuid'],PHYSICAL_GPU=str(g['index']),PYTHONUNBUFFERED='1',LEASE_SECONDS='7200',TMPDIR='/data/bmw/tmp',HF_HOME='/data/bmw/cache/huggingface',TORCH_HOME='/data/bmw/cache/torch',XDG_CACHE_HOME='/data/bmw/cache',CUDA_CACHE_PATH='/data/bmw/cache/cuda')
   with (ROOT/'logs'/f'worker{g["index"]}.log').open('ab') as log:p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','/tmp/s7.py'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   active.append(dict(pid=p.pid,index=g['index'],uuid=g['uuid']));write(ROOT/'ACTIVE_PROCESSES.json',active)
  time.sleep(15)
if __name__=='__main__':
 try:main()
 except Exception:
  (ROOT/'ADMISSION_STOP').touch();write(ROOT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc()));write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',reason='CONTROLLER_FAILURE'));raise
