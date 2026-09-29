"""Local scorer handoff; never overlaps predecessor scorer or restarts after STOP."""
import os,json,time,subprocess,fcntl
from pathlib import Path
base=Path(os.environ['SCOPE_LOCAL']);remote=os.environ['RUN_ROOT'];old=base.parent/'Knowledge_editing-decomp-24h-20260927'
f=(base/'.local_run/HANDOFF.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
def query():
 code="from pathlib import Path;import json;r=Path("+repr(remote)+");print(json.dumps(dict(stop=(r/'STOP').exists(),acquired=(r/'LEASE_ACQUIRED.json').exists(),status=json.loads((r/'RUN_STATUS.json').read_text())['status'])))"
 return json.loads(subprocess.check_output(['ssh','pro5000','python3 -'],input=code,text=True,timeout=30))
while True:
 s=query();(base/'.local_run/HANDOFF_STATUS.json').write_text(json.dumps(s))
 if s['stop'] or s['status'] in ['USER_STOPPED','BLOCKED','FAILED']:break
 if s['acquired']:
  pid=json.loads((old/'.local_run/APPENDIX_SCORE_PID.json').read_text())['pid']
  cmd=subprocess.run(['ps','-p',str(pid),'-o','args='],text=True,capture_output=True).stdout
  if '/tmp/e2.py' not in cmd:
   state=Path('/tmp/ss1');state.mkdir(mode=0o700,exist_ok=True)
   raw=subprocess.check_output(['ssh','pro5000','cat '+remote+'/private/judge/JUDGE_MISSING_LOCK.json'],text=True);(state/'blocked_keys.json').write_text(raw)
   Path('/tmp/s11.py').write_text('import os,sys,runpy\nsys.path[:0]=os.environ["SOURCE_PATHS"].split(os.pathsep)\nrunpy.run_module("scope_scorer",run_name="__main__")\n')
   env=os.environ.copy();env.update(SCORER_STATE=str(state),SCORER_RUNTIME=str(old/'.local_run/runtime'),SOURCE_PATHS=os.pathsep.join([str(base/'experiments/scope_safe_residual_20260929'),str(base/'experiments/decomp_24h_20260927')]),PYTHONUNBUFFERED='1')
   with (state/'score.log').open('ab') as log:p=subprocess.Popen(['/opt/homebrew/bin/python3','/tmp/s11.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (base/'.local_run/SCORE_PID.json').write_text(json.dumps(dict(pid=p.pid,entry='/tmp/s11.py',state=str(state))));(base/'.local_run/HANDOFF_STATUS.json').write_text(json.dumps(dict(status='SCORER_STARTED',pid=p.pid)));break
 time.sleep(30)
