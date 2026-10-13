"""Atomic future GPU reservations plus cumulative historical accounting."""
import contextlib,fcntl,json,os,time
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
@contextlib.contextmanager
def lock():
 with (ROOT/'RESOURCE_LEDGER.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);d=read(ROOT/'RESOURCE_LEDGER.json');yield d
  d['new_gpu_seconds']=d['gpu_seconds_used']-d['historical_gpu_seconds']
  d['new_judge_attempt_items']=d['judge_submission_attempt_items']-d['historical_judge_attempt_items']
  d['remaining_gpu_seconds']=d['gpu_seconds_limit']-d['gpu_seconds_used']
  d['remaining_judge_attempt_items']=d['judge_submission_attempt_items_limit']-d['judge_submission_attempt_items']
  write(ROOT/'RESOURCE_LEDGER.json',d)
def reserve(jid,gpu,seconds):
 with lock() as d:
  assert not any(x['id']==jid for x in d['reservations']), 'Duplicate reservation'
  active=sum(x['seconds'] for x in d['reservations'] if x['status'] in ['RESERVED','ACTIVE'])
  if d['gpu_seconds_used']+active+seconds>d['gpu_seconds_limit']:raise TimeoutError('Atomic future GPU budget unavailable')
  d['reservations'].append(dict(id=jid,gpu_uuid=gpu,seconds=seconds,status='RESERVED'))
def close(jid):
 with lock() as d:
  r=next(x for x in d['reservations'] if x['id']==jid)
  if r['status']=='CLOSED':return
  sessions=[x for x in d['gpu_sessions'] if x['purpose']==jid and x.get('ended_epoch') is None]
  for s in sessions:
   now=time.time();s.update(ended_epoch=now,resident_seconds=now-s['started_epoch']);d['gpu_seconds_used']+=s['resident_seconds']
  r['status']='CLOSED'
def install(rt):
 original=rt.check_budget
 def check_budget(*,training=False,p0=False):
  original(training=training,p0=p0)
  d=read(ROOT/'RESOURCE_LEDGER.json');jid=os.environ['JOB_ID'];r=next(x for x in d['reservations'] if x['id']==jid)
  if r['status']=='ACTIVE' and time.time()-r['started_epoch']>=r['seconds']:raise TimeoutError('Reserved job residency exhausted')
  if (ROOT/'JUDGE_FAILURE.json').exists():raise RuntimeError('Judge failure; preserve state')
 @contextlib.contextmanager
 def session(purpose):
  check_budget()
  with lock() as d:
   r=next(x for x in d['reservations'] if x['id']==purpose)
   assert r['status']=='RESERVED' and r['gpu_uuid']==rt.GPU
   assert not any(s['gpu_uuid']==rt.GPU and s.get('ended_epoch') is None for s in d['gpu_sessions'])
   r.update(status='ACTIVE',started_epoch=time.time(),pid=os.getpid())
   d['gpu_sessions'].append(dict(pid=os.getpid(),purpose=purpose,gpu_uuid=rt.GPU,started_epoch=r['started_epoch']))
  try:yield
  finally:close(purpose)
 rt.check_budget=check_budget;rt.gpu_session=session;rt.write=write
