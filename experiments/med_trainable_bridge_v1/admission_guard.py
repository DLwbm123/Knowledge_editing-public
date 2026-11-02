"""Independent hard deadline and bounded storage guard for state admission."""
import json,os,resource,signal,subprocess,time,traceback
from pathlib import Path
root=Path(os.environ['EXPERIMENT_ROOT']);deadline=float(os.environ['EXPERIMENT_DEADLINE'])
def write(n,v):
 p=root/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v));t.replace(p)
def size():
 return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(100_000_000,100_000_000))
with (root/'ADMISSION_GUARD_ONCE').open('x'):pass
with (root/'admission.log').open('x') as f:
 p=subprocess.Popen([os.environ['TRAIN_PYTHON'],str(root/'a.py')],env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit)
try:
 while p.poll() is None:
  now=time.time();used=size();free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
  write('ADMISSION_GUARD_HEALTH.json',{'time':now,'hard_end':deadline,'bytes':used,'free':free})
  if now>=deadline-120 or used>=1_000_000_000 or free<8*1024**3:raise RuntimeError('budget stop')
  time.sleep(3)
 code=p.wait();write('ADMISSION_WORKER_EXIT.json',{'code':code,'time':time.time()})
 if code:raise RuntimeError('admission worker failed')
 write('ADMISSION_GUARD_EXIT.json',{'code':0,'time':time.time()})
except BaseException:
 write('ADMISSION_GUARD_FAILURE.private.json',{'traceback':traceback.format_exc()})
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 write('ADMISSION_GUARD_EXIT.json',{'code':1,'time':time.time()})
 raise
