import json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
root=Path(os.environ['FIXED_SCORE_ROOT']);deadline=float(os.environ['EXPERIMENT_DEADLINE']); neutral=Path(os.environ['NEUTRAL_ENTRY'])
def write(n,v):
 p=root/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v));t.replace(p)
def size(p):
 return sum(x.stat().st_blocks*512 for x in p.rglob('*') if x.is_file() and not x.is_symlink())
def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
try:
 if time.time()>=deadline-660:raise TimeoutError('original deadline too close')
 with (root/'GUARD_ONCE').open('x'):pass
 with (root/'worker.log').open('a') as log:
  child=subprocess.Popen([sys.executable,str(neutral)],env=dict(os.environ,ENTRY_MODE='worker'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit)
  write('WORKER_IDENTITY.json',{'pid':child.pid})
  while child.poll() is None:
   used=size(root)+size(neutral.parent);st=os.statvfs(root);free=st.f_bavail*st.f_frsize
   write('GUARD_HEALTH.json',{'bytes':used,'free':free,'hard_end':deadline,'time':time.time()})
   if used>=30_000_000_000 or free<8*1024**3 or time.time()>=deadline-45:
    write('STOP.json',{'reason':'resource or original deadline'});os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
   time.sleep(5)
  code=child.wait();write('GUARD_EXIT.json',{'code':code,'hard_end':deadline,'time':time.time()})
except BaseException:
 write('GUARD_FAILURE.json',{'error':traceback.format_exc(),'time':time.time()});raise
