"""Independent wall-clock, file and storage protection for a bounded GPU diagnostic."""
import json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
root=Path(os.environ['DIAG_ROOT']);end=float(os.environ['EXPERIMENT_DEADLINE']);neutral=Path(os.environ['NEUTRAL_ENTRY'])
def write(name,obj):
 p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj));t.replace(p)
def allocated(p):
 size=0
 for f in p.rglob('*'):
  try:
   if f.is_file() and not f.is_symlink():size+=f.stat().st_blocks*512
  except FileNotFoundError:pass
 return size
def lim():resource.setrlimit(resource.RLIMIT_FSIZE,(100_000_000,100_000_000))
def main():
 if time.time()>=end-600:raise TimeoutError('deadline near')
 gpu=os.environ['CUDA_VISIBLE_DEVICES']
 if not gpu.isdigit():raise ValueError('physical GPU must be explicit')
 row=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 if row[0].strip()!=os.environ['GPU_UUID'] or int(row[1])<46080:raise RuntimeError('GPU identity or free-memory reserve')
 with (root/'GUARD_ONCE').open('x'):pass
 write('GUARD_HEALTH.json',{'time':time.time(),'bytes':allocated(root)+allocated(neutral.parent),'hard_end':end})
 with (root/'worker.log').open('a') as f:
  child=subprocess.Popen([sys.executable,str(neutral)],env=dict(os.environ,ENTRY_MODE='worker'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lim)
  write('WORKER_IDENTITY.json',{'pid':child.pid,'time':time.time()})
  try:
   while child.poll() is None:
    size=allocated(root)+allocated(neutral.parent);v=os.statvfs(root);free=v.f_bavail*v.f_frsize
    write('GUARD_HEALTH.json',{'time':time.time(),'bytes':size,'free':free,'hard_end':end})
    if size>=900_000_000 or free<=8*1024**3 or time.time()>=end-45:raise RuntimeError('storage reserve or deadline')
    time.sleep(5)
  except BaseException:
   write('STOP.json',{'time':time.time()});write('GUARD_FAILURE.json',{'error':traceback.format_exc()})
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  code=child.wait();write('GUARD_EXIT.json',{'code':code,'hard_end':end,'time':time.time()});return code
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception:
  write('GUARD_FAILURE.json',{'error':traceback.format_exc()});write('GUARD_EXIT.json',{'code':1,'hard_end':end});raise
