"""Original-clock and coupled-storage guard for one diagnostic GPU worker."""
import json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['DIAG_ROOT']);cfg=json.loads((ROOT/'CONFIG.private.json').read_text());child=None

def write(name,x):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x));t.replace(p)
def stop(*_):
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 raise InterruptedError('diagnostic resource guard')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 (ROOT/'GUARD_ONCE').touch(exist_ok=False)
 row=subprocess.check_output(['nvidia-smi','-i',cfg['gpu']['index'],'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).split(',')
 assert row[0].strip()==cfg['gpu']['uuid'] and int(row[1])>=60000
 with (ROOT/'worker.log').open('x') as log:
  child=subprocess.Popen([cfg['python'],str(Path(cfg['neutral'])/'w.py')],env=dict(os.environ,CURRENT_SOURCE=str(ROOT/'tools/worker.py')),cwd='/data/bmw',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
   preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,256*1024**2)))
 write('IDENTITY.private.json',dict(pid=child.pid,time=time.time()))
 while child.poll() is None:
  size=0
  for base in [ROOT,Path(cfg['neutral'])]+[Path(x) for x in cfg['additional_owned_roots']]:
   for directory,dirs,files in os.walk(base,followlinks=False):
    for name in files:
     p=Path(directory)/name
     try:
      if not p.is_symlink():size+=p.stat().st_blocks*512
     except FileNotFoundError:pass
  fs=os.statvfs(ROOT);write('HEALTH.json',dict(time=time.time(),bytes=size,hard_end=cfg['clock']['hard_end']))
  assert time.time()<cfg['clock']['hard_end']-90 and size+2*1024**3<12_000_000_000 and fs.f_bavail*fs.f_frsize>8*1024**3
  time.sleep(3)
 write('GUARD_EXIT.json',dict(code=child.wait(),time=time.time()));assert child.returncode==0
except BaseException:
 write('GUARD_FAILURE.private.json',dict(time=time.time(),error=traceback.format_exc()))
 if child and child.poll() is None:stop()
 raise
