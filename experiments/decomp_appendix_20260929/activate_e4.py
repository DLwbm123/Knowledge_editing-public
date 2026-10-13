"""One-shot activation after the existing owner releases all GPU/scoring work."""
import os,sys,json,time,shutil,subprocess,fcntl
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(r))
from resources import read,write,lock

def main():
 q=read(r/'QUEUE.json')
 if any(j.get('block')=='E4_REG24' for j in q):print('ALREADY_ADMITTED');return
 if not all(j['status']=='COMPLETE' for j in q):print('WAIT');return
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if any(x in (p/'cmdline').read_bytes().decode().split('\0') for x in ['/tmp/e1.py','/tmp/e3.py']):print('WAIT');return
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 if not (r/'SCORER_DONE').exists():print('WAIT');return
 assert not (r/'STOP').exists() and not (r/'JUDGE_FAILURE.json').exists()
 assert os.statvfs(r).f_bavail*os.statvfs(r).f_frsize>20*1024**3
 c=read(r/'public/SCORING_COVERAGE.json');assert not c['IN_FLIGHT'] and not c['UNSUBMITTED']
 audit=r/'fix/e3_completed_before_e4';audit.mkdir(parents=True,exist_ok=True)
 for n in ['RESOURCE_LEDGER.json','QUEUE.json','QUEUE_AMENDMENT.json','INCREMENTAL_MATRIX_LOCK.json','RUN_STATUS.json','ACTIVE_IMPLEMENTATION.json']:
  assert not (audit/n).exists();shutil.copy2(r/n,audit/n)
 shutil.copytree(r/'public',audit/'public')
 for n in ['private/curves','private/diagnostics']:
  if (r/n).exists():shutil.copytree(r/n,audit/n)
 with lock() as d:
  assert not any(not s.get('ended_epoch') for s in d['gpu_sessions'])
  for x in d['reservations']:
   if x['block']=='E3_REG24':x.update(status='CLOSED',actual_new_attempt_items=d['judge_submission_attempt_items']-x['baseline_attempt_items'],closed_epoch=time.time())
   if x['block']=='E4_REG24':x.update(status='RESERVED',baseline_attempt_items=d['judge_submission_attempt_items'],activated_epoch=time.time())
  assert d['judge_submission_attempt_items_limit']-d['judge_submission_attempt_items']>=2880
 stage=r/'staged_e4'
 for n in ['e4.py','worker.py','bindings.py','controller.py','report.py']:
  tmp=r/(n+'.e4.tmp');shutil.copyfile(stage/n,tmp);os.replace(tmp,r/n)
 from e4 import jobs
 with (r/'QUEUE.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);write(r/'QUEUE.json',q+jobs())
 a=read(r/'QUEUE_AMENDMENT.json');a['blocks'].append(dict(id='E4_REG24',status='ADMITTED',units=96,jobs=52,Judge_items_reserved=2880,epoch=time.time()));write(r/'QUEUE_AMENDMENT.json',a)
 d=read(r/'INCREMENTAL_MATRIX_LOCK.json');d.update(active='E4_REG24',E3_status='EXECUTION_FINISHED_AUDIT_DELIVERY_PENDING',E4=read(r/'E4_PROTOCOL.json'));write(r/'INCREMENTAL_MATRIX_LOCK.json',d)
 (r/'SCORER_DONE').rename(audit/'SCORER_DONE')
 write(r/'ACTIVE_PROCESSES.json',[]);write(r/'RUN_STATUS.json',dict(status='RUNNING',phase='E4_CANARY',epoch=time.time()))
 import hashlib
 d=read(r/'ACTIVE_IMPLEMENTATION.json');d.update(sha=read(r/'E4_SOURCE_DELIVERY_RECEIPT.json')['sha'],pending_publication=False);d['files'].update({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['e4.py','worker.py','bindings.py','controller.py','report.py']});write(r/'ACTIVE_IMPLEMENTATION.json',d)
 env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',TMPDIR='/data/bmw/tmp')
 with (r/'logs/controller.log').open('ab') as f:p=subprocess.Popen(['/data/bmw/envs/v0/bin/python','/tmp/e3.py'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 write(r/'E4_ACTIVATION.json',dict(pid=p.pid,epoch=time.time(),prior_E3_snapshot=str(audit.relative_to(r))))
 print('STARTED')
if __name__=='__main__':main()
