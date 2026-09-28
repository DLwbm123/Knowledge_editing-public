"""Finite dependency queue, process identity checks, leases and original deadlines."""
import os,time,json,signal,subprocess,fcntl,resource,traceback
from datetime import datetime
from pathlib import Path
from resources import ROOT,read,write,lock,limits_waived
from report import report,coverage

def alive(pid):
 try:
  p=Path('/proc')/str(pid);return p.exists() and p.joinpath('cwd').resolve()==ROOT.resolve() and '/tmp/d1.py' in p.joinpath('cmdline').read_bytes().decode().split('\0') and p.joinpath('stat').read_text().split()[2]!='Z'
 except (FileNotFoundError,PermissionError):return False

def close_dead(pid):
 with lock() as d:
  for s in d['gpu_sessions']:
   if s['pid']==pid and not s.get('ended_epoch'):
    elapsed=max(0,time.time()-s['started_epoch']);s.update(ended_epoch=s['started_epoch']+elapsed,resident_seconds=elapsed,closure='controller observed dead; conservative charge through observation time');d['current_gpu_seconds']+=elapsed

def stop(pid):
 if alive(pid):os.killpg(pid,signal.SIGTERM)
 time.sleep(.2)
 if alive(pid):os.killpg(pid,signal.SIGKILL)
 close_dead(pid)

def log_limit():
 if read(ROOT/'STORAGE_POLICY.json').get('quotas_enabled',True):resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2))

def launch(g):
 # Only this phase's entries and the three locked physical devices are eligible.
 # Shared GPUs are authorized when the existing free-memory margin is satisfied.
 free=subprocess.check_output(['nvidia-smi','-i',str(g['index']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 if int(free.strip())<30000:return None
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=g['uuid'],PINNED_GPU_UUID=g['uuid'],PHYSICAL_GPU=str(g['index']),LEASE_SECONDS='7200',PYTHONUNBUFFERED='1',TMPDIR='/data/bmw/tmp',HF_HOME='/data/bmw/cache/huggingface',TORCH_HOME='/data/bmw/cache/torch',XDG_CACHE_HOME='/data/bmw/cache',CUDA_CACHE_PATH='/data/bmw/cache/cuda')
 with (ROOT/'logs'/f'worker{g["index"]}.log').open('ab') as f:p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','/tmp/d1.py'],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=log_limit)
 return dict(pid=p.pid,index=g['index'],uuid=g['uuid'],started_epoch=time.time(),entry='/tmp/d1.py')

def build_blocks():
 blocks=[]
 for seed in [20260927,20260928,20260929]:
  for panel,orders in [('REG24',list(range(25,49))),('DEV24',list(range(1,25)))]:
   jobs=[]
   for o in orders:
    if seed==20260927 and o in [1,25]:continue
    specs=[('CP',['M0','M1','M2','M3'] if seed==20260927 else ['M0','M1','M3']),('TK',['M4','M5'])]+([('LR',['M6'])] if seed==20260927 else [])
    for kind,arms in specs:jobs.append(dict(id=f's{seed}-{kind}-{o}',mode='train',kind=kind,methods=arms,seed=seed,order=o,requires=[f'adapters/s{seed}/M0/e{o:03d}.pt'] if kind=='TK' else [],status='PENDING'))
   for m in ['M0','M1','M3','M4','M5']:
    jobs.append(dict(id=f's{seed}-{panel}-{m}-bank',mode='sequential',seed=seed,method=m,orders=orders,prefixes=[12,24],requires=[f'adapters/s{seed}/{m}/e{o:03d}.pt' for o in orders],status='PENDING'))
   blocks.append(dict(id=f's{seed}-{panel}',jobs=jobs))
 for suffix in ['_STRUCT','_R8']:
  jobs=[]
  for o in read(ROOT/'EXPERIMENT_LOCK.json')['stage_diagnostics_orders']:
   for kind,m in [('CP','M1'),('TK','M4')]:jobs.append(dict(id=f'optional-{m}{suffix}-{o}',mode='train',kind=kind,methods=[m+suffix],seed=20260927,order=o,requires=[f'checkpoints/W0/s20260927/{kind}/e{o:03d}.pt'],status='PENDING'))
  blocks.append(dict(id='paired_DEV4'+suffix,jobs=jobs))
 return blocks

def main():
 singleton=(ROOT/'CONTROLLER.lock').open('a');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(ROOT/'ORCHESTRATOR_PID.json',dict(pid=os.getpid(),entry='/tmp/d3.py',started_epoch=time.time()))
 manifest=read(ROOT/'EXPERIMENT_LOCK.json');deadline=datetime.fromisoformat(manifest['deadline_at']).timestamp();trainstop=datetime.fromisoformat(manifest['no_new_training_after']).timestamp();genstop=datetime.fromisoformat(manifest['no_large_generation_after']).timestamp()
 if limits_waived():deadline=trainstop=genstop=float('inf')
 blocks=build_blocks();ap=ROOT/'QUEUE_AMENDMENT.json'
 if not ap.exists():write(ap,dict(blocks=[dict(id=b['id'],status='NOT_STARTED',units=sum(len(j.get('methods',[])) for j in b['jobs']),jobs=len(b['jobs'])) for b in blocks],main_evaluated_units_target=720,additional_seed_M0_teachers=96,optional='After core seeds: fixed diagnostic DEV4 paired structured continuation, then zero-function-change rank8; all later optional families deferred',selection='Resource forecasts only; no method outcome based selection'))
 receipts=read(ROOT/'ACTIVE_PROCESSES.json') if (ROOT/'ACTIVE_PROCESSES.json').exists() else read(ROOT/'PILOT_PROCESSES.json')
 active=[x for x in receipts if alive(x['pid'])];pilot=not (ROOT/'PILOT_CLOSURE.json').exists();last_report=0;reason=None
 while time.time()<deadline:
  now=time.time();q=read(ROOT/'QUEUE.json')
  if (ROOT/'STOP').exists() or (ROOT/'JUDGE_FAILURE.json').exists():reason='STOP_OR_JUDGE_FAILURE';break
  for x in active[:]:
   if not alive(x['pid']):
    close_dead(x['pid']);active.remove(x)
    unfinished=[j for j in q if j['status']=='RUNNING' and j.get('pid')==x['pid']]
    if unfinished:
     receipt=read(ROOT/'jobs'/f'WORKER_{x["index"]}.json')
     if receipt.get('pid')==x['pid'] and 'GPU lease limit' in receipt.get('error',''):
      write(ROOT/'jobs'/f'LEASE_{x["pid"]}.json',receipt)
      with (ROOT/'QUEUE.lock').open('a') as f:
       fcntl.flock(f,fcntl.LOCK_EX);fresh=read(ROOT/'QUEUE.json')
       for j in fresh:
        if j.get('pid')==x['pid'] and j['status']=='RUNNING':j.update(status='PENDING',resume_gpu=x['index']);j.setdefault('attempts',[]).append(dict(pid=x['pid'],reason='lease exhausted',epoch=now))
       write(ROOT/'QUEUE.json',fresh)
      continue
     reason='WORKER_FAILURE';write(ROOT/'BLOCKER.json',dict(reason=reason,jobs=[j['id'] for j in unfinished],pid=x['pid']));break
   else:
    d=read(ROOT/'RESOURCE_LEDGER.json');sessions=[s for s in d['gpu_sessions'] if s['pid']==x['pid'] and not s.get('ended_epoch')]
    if sessions and now-sessions[-1]['started_epoch']>sessions[-1]['reserved_seconds']+30:stop(x['pid']);reason='LEASE_EXCEEDED';break
  if reason:break
  d=read(ROOT/'RESOURCE_LEDGER.json');used=d['current_gpu_seconds']+sum(now-s['started_epoch'] for s in d['gpu_sessions'] if not s.get('ended_epoch'))
  if not limits_waived() and used>=manifest['gpu_seconds_limit']:reason='GPU_BUDGET';break
  if now-last_report>120:
   from storage import Store
   disk=Store(ROOT).disk_audit();write(ROOT/'STORAGE_AUDIT.json',disk);report();last_report=now
  pending=[j for j in q if j['status']=='PENDING'];running=[j for j in q if j['status']=='RUNNING']
  if not pending and not running and not active:
   c=coverage()
   if c['IN_FLIGHT'] or c['UNSUBMITTED']:
    if (ROOT/'SCORER_DONE').exists():reason='JUDGE_BUDGET_OR_SCORER_CLOSED';break
    write(ROOT/'RUN_STATUS.json',dict(status='SCORING_GATE',epoch=now));time.sleep(15);continue
   if pilot:
    if c['MISSING']:reason='PILOT_SCORING_MISSING';break
    write(ROOT/'PILOT_CLOSURE.json',dict(status='PASS',epoch=now,coverage=c,training_save_reload_generation_scoring=True));pilot=False
   amendments=read(ap)
   for a in amendments['blocks']:
    if a['status']=='ADMITTED':a['status']='COMPLETE'
   write(ap,amendments)
   nextblock=next((b for b in blocks if next(x for x in amendments['blocks'] if x['id']==b['id'])['status']=='NOT_STARTED'),None)
   if not nextblock:break
   if disk['soft_reached']:reason='STORAGE_SOFT_WATERMARK';break
   if (ROOT/'REPAIR_HOLD.json').exists():reason='REPAIR_CANARY_REVIEW_REQUIRED';break
   # Conservative pilot-derived per-unit forecast, with a 25% resource reserve.
   completed_units=sum(len(j.get('methods',[])) for j in q if j['status']=='COMPLETE');new_attempts=d['judge_submission_attempt_items']-d['historical_judge_attempts'];unit_seconds=max(60,used/max(1,completed_units));unit_judge=max(1,new_attempts/max(1,completed_units));units=sum(len(j.get('methods',[])) for j in nextblock['jobs']);forecast_gpu=unit_seconds*units*1.3;forecast_judge=unit_judge*units*1.5
   forecast=dict(block=nextblock['id'],units=units,GPU_seconds=forecast_gpu,Judge_items=forecast_judge,observed_units=completed_units,epoch=now)
   write(ROOT/'BUDGET_FORECAST.json',forecast)
   if (not limits_waived() and (now+forecast_gpu/3>=trainstop or used+forecast_gpu>.75*manifest['gpu_seconds_limit'])) or forecast_judge>.75*(d['judge_submission_attempt_items_limit']-d['judge_submission_attempt_items']):reason='FORECAST_RESERVE_GATE';break
   with (ROOT/'QUEUE.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json');q.extend(nextblock['jobs']);write(ROOT/'QUEUE.json',q)
   a=next(x for x in amendments['blocks'] if x['id']==nextblock['id']);a.update(status='ADMITTED',forecast=forecast);write(ap,amendments);pending=nextblock['jobs'];write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase=nextblock['id'],epoch=now))
  if pending and not active and not any(all((ROOT/p).exists() for p in j.get('requires',[])) for j in pending):reason='DEPENDENCY_BLOCKED';break
  if now>=genstop:reason='GENERATION_STOP';break
  if now>=trainstop and any(j['mode']=='train' for j in pending):reason='TRAINING_STOP';break
  for g in read(ROOT/'GPU_BINDINGS.json')['devices']:
   if any(x['index']==g['index'] for x in active):continue
   if not any(j.get('resume_gpu',g['index'])==g['index'] and all((ROOT/p).exists() for p in j.get('requires',[])) for j in pending):continue
   p=launch(g)
   if p:active.append(p);write(ROOT/'ACTIVE_PROCESSES.json',active)
  time.sleep(15)
 if time.time()>=deadline:reason='HARD_DEADLINE'
 if reason:
  if reason=='WORKER_FAILURE':
   from storage import atomic_control
   atomic_control(ROOT/'ADMISSION_STOP',b'Worker failure; preserve healthy current units, stop further admission')
  else:
   for x in active:stop(x['pid'])
 write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED' if reason else 'GPU_QUEUE_FINISHED',reason=reason,epoch=time.time()))
 report(final=not reason)
 if not reason:
  from storage import Store
  store=Store(ROOT)
  with store.lock() as d:rolling=[r for r,a in d['artifacts'].items() if r.startswith('checkpoints/slot') and a['status']=='READY' and not a['readers'] and not a['consumers']]
  for rel in rolling:
   with store.lock() as d:d['artifacts'][rel]['pin']=False
   store.delete(rel)
  models=sum(p.stat().st_size for p in (ROOT/'adapters').rglob('*.pt'));policy=read(ROOT/'STORAGE_POLICY.json');assert not policy.get('quotas_enabled',True) or models<=policy['final_model_bytes']
  total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink());assert not policy.get('quotas_enabled',True) or total<=policy['final_total_bytes'],'FINAL_STORAGE_LIMIT'
  write(ROOT/'RETAINED_CHECKPOINT_MANIFEST.json',[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size) for p in (ROOT/'adapters').rglob('*.pt')])
  write(ROOT/'public/RETAINED_CHECKPOINT_MANIFEST.json',dict(count=len(list((ROOT/'adapters').rglob('*.pt'))),model_bytes=models,phase_artifact_bytes=total,reusable_environment_separately_reserved_bytes=policy['environment_reserved_bytes'],weights_public=False))
  write(ROOT/'public/CLEANUP_RECEIPT.json',dict(rolling_removed=len(rolling),model_bytes=models,final_adapter_retention='All paired main adapters, including negative results'))
if __name__=='__main__':
 try:main()
 except Exception:
  from storage import atomic_control
  atomic_control(ROOT/'ADMISSION_STOP',b'Controller failure; finish current unit and stop admission')
  # A reporting failure must stop admission without killing a healthy atomic job.
  with (ROOT/'QUEUE.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);q=read(ROOT/'QUEUE.json')
   for j in q:
    if j['status']=='PENDING':j['status']='HOLD_CONTROLLER_ERROR'
   write(ROOT/'QUEUE.json',q)
  write(ROOT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),epoch=time.time(),workers_preserved=True));write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',reason='CONTROLLER_EXCEPTION_SAFE_DRAIN'));raise
