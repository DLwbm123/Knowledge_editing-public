"""Current-phase GPU seconds, lifetime Judge attempts and immutable wall clock."""
import os,json,time,fcntl,contextlib,subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 from storage import Store
 p=Path(p);Store(ROOT).write(str(p.relative_to(ROOT)),json.dumps(d,ensure_ascii=False,indent=2).encode())
@contextlib.contextmanager
def lock():
 with (ROOT/'RESOURCE_LEDGER.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);d=read(ROOT/'RESOURCE_LEDGER.json');yield d;write(ROOT/'RESOURCE_LEDGER.json',d)
def limits_waived():
 p=ROOT/'USER_LIMIT_WAIVER.json'
 return p.exists() and read(p).get('wallclock_gpu_limits_waived',False) is True

def check(training=False):
 m=read(ROOT/'EXPERIMENT_LOCK.json');now=time.time()
 if (ROOT/'STOP').exists() or (not limits_waived() and now>=datetime.fromisoformat(m['no_new_training_after' if training else 'deadline_at']).timestamp()):raise TimeoutError('STOP or original deadline')
 d=read(ROOT/'RESOURCE_LEDGER.json');active=sum(now-s['started_epoch'] for s in d['gpu_sessions'] if not s.get('ended_epoch'))
 if not limits_waived() and d['current_gpu_seconds']+active>=m['gpu_seconds_limit']:raise TimeoutError('Phase GPU limit')
 for s in d['gpu_sessions']:
  if s['pid']==os.getpid() and not s.get('ended_epoch') and now-s['started_epoch']>=s['reserved_seconds']:raise TimeoutError('GPU lease limit')
@contextlib.contextmanager
def session(purpose,seconds=900):
 check();gpu=os.environ['CUDA_VISIBLE_DEVICES'];bindings=read(ROOT/'GPU_BINDINGS.json')['devices'];assert gpu in [x['uuid'] for x in bindings]
 m=read(ROOT/'EXPERIMENT_LOCK.json')
 with lock() as d:
  active=[s for s in d['gpu_sessions'] if not s.get('ended_epoch')]
  assert len(active)<3 and all(s['gpu_uuid']!=gpu for s in active)
  assert limits_waived() or d['current_gpu_seconds']+sum(s['reserved_seconds'] for s in active)+seconds<=m['gpu_seconds_limit']
  s=dict(pid=os.getpid(),gpu_uuid=gpu,purpose=purpose,started_epoch=time.time(),reserved_seconds=seconds);d['gpu_sessions'].append(s)
 try:yield
 finally:
  with lock() as d:
   q=next(x for x in d['gpu_sessions'] if x['pid']==s['pid'] and x['started_epoch']==s['started_epoch']);q.update(ended_epoch=time.time());q['resident_seconds']=q['ended_epoch']-q['started_epoch'];d['current_gpu_seconds']+=q['resident_seconds'];d['lifetime_gpu_seconds']=d['historical_gpu_seconds']+d['current_gpu_seconds']
