"""Bound the identity-first preparation to the original experiment clock."""
import json, os, pathlib, resource, signal, subprocess, time, traceback
root=pathlib.Path(os.environ['EXPERIMENT_ROOT'])
neutral=pathlib.Path(os.environ['NEUTRAL_ROOT'])
end=float(os.environ['EXPERIMENT_DEADLINE'])
def write(name, obj):
 p=neutral/name; t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj)); t.replace(p)
def usage():
 total=0
 for base in (root,neutral):
  for p in base.rglob('*'):
   try:
    if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
   except FileNotFoundError:pass
 return total
def main():
 with (neutral/'CPU_GUARD_ONCE').open('x'):pass
 def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
 with (neutral/'cpu.log').open('x') as f:
  child=subprocess.Popen([os.environ['TRAIN_PYTHON'],str(neutral/'p.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=os.environ.copy(),start_new_session=True,preexec_fn=limit)
 write('CPU_IDENTITIES.private.json',{'worker':child.pid,'deadline':end})
 try:
  while child.poll() is None:
   used=usage();fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize
   write('CPU_HEALTH.json',{'time':time.time(),'bytes':used,'free':free,'deadline':end})
   if used>=30_000_000_000 or free<8*1024**3 or time.time()>=end-120:raise RuntimeError('resource or deadline stop')
   time.sleep(3)
  code=child.wait();write('CPU_EXIT.json',{'code':code,'time':time.time()})
  if code:raise RuntimeError('preparation failed')
  write('CPU_GUARD_EXIT.json',{'code':0,'time':time.time()})
 except BaseException:
  if child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  write('CPU_GUARD_FAILURE.private.json',{'traceback':traceback.format_exc()})
  write('CPU_GUARD_EXIT.json',{'code':1,'time':time.time()});raise
if __name__=='__main__':main()
