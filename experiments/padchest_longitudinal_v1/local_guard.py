"""Bound local controller and its own descendant process tree independently."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(os.environ['LONG_LOCAL_CONFIG']).read_text());root=Path(cfg['control'])
(root/'LOCAL_GUARD_ONCE').touch(exist_ok=False)
with (root/'controller.log').open('x') as log:
 child=subprocess.Popen([sys.executable,str(root/'q.py')],env=dict(os.environ,CURRENT_SOURCE=str(Path(cfg['worktree'])/'experiments/padchest_longitudinal_v1/controller.py')),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'CONTROLLER_IDENTITY.private.json').write_text(json.dumps(dict(pid=child.pid,time=time.time())))
def terminate():
 rows=[r.split(None,2) for r in subprocess.check_output(['ps','-ww','-eo','pid=,ppid=,args='],text=True).splitlines()]
 family={child.pid}
 while True:
  new=family|{int(r[0]) for r in rows if int(r[1]) in family}
  if new==family:break
  family=new
 for pid in sorted(family,reverse=True):
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 try:child.wait(timeout=20)
 except subprocess.TimeoutExpired:
  for pid in family:
   try:os.kill(pid,signal.SIGKILL)
   except ProcessLookupError:pass
signal.signal(signal.SIGTERM,lambda *_:(terminate(),sys.exit(1)))
try:
 while child.poll() is None:
  used=0
  for base in [root,Path(cfg['private_root'])]:
   for directory,dirs,files in os.walk(base,followlinks=False):
    for name in files:
     p=Path(directory)/name
     try:
      if not p.is_symlink():used+=p.stat().st_blocks*512
     except FileNotFoundError:pass
  assert used+256*1024**2<2_000_000_000 and time.time()<cfg['clock']['hard_end']-60
  fs=os.statvfs(root);assert fs.f_bavail*fs.f_frsize>8*1024**3
  (root/'LOCAL_GUARD_HEALTH.json').write_text(json.dumps(dict(time=time.time(),bytes=used)))
  time.sleep(5)
 (root/'LOCAL_GUARD_EXIT.json').write_text(json.dumps(dict(code=child.returncode,time=time.time())))
except BaseException:
 terminate();raise
