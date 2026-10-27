"""One neutral GPU entry and one guarded local grading controller."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def remote(code):
    p=subprocess.run(['ssh','pro5000','/data/bmw/envs/v0/bin/python','-'],input=code,capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout)


def main():
    cfg=read(os.environ['DISPATCH_CONFIG']);local=Path(cfg['local'])
    prefix='import os,json,sys,subprocess,time\nfrom pathlib import Path\nq=Path('+repr(cfg['neutral'])+')\nos.environ.update(json.loads((q/\'ENV.json\').read_text()))\nr=Path(os.environ[\'EXPERIMENT_ROOT\'])\n'
    current=subprocess.check_output(['git','-C',cfg['workspace'],'rev-parse','HEAD'],text=True).strip()
    assert subprocess.check_output(['git','-C',cfg['workspace'],'branch','--show-current'],text=True).strip()==cfg['branch']
    started=remote(prefix+'''
assert not (r/'GUARD_ONCE').exists() and not (r/'DISPATCH.json').exists()
value=dict(commit=__COMMIT__,branch=__BRANCH__,protocol='MED_RA_V134')
(r/'SOURCE_VERSION.json').write_text(json.dumps(value,indent=2))
with (r/'guard.log').open('a') as log:
 child=subprocess.Popen([sys.executable,str(q/'q.py')],env=dict(os.environ,ENTRY_MODE='guard'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()[19]
value=dict(pid=child.pid,start_ticks=ticks,time=time.time(),source_commit=__COMMIT__)
(r/'DISPATCH.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''.replace('__COMMIT__',repr(current)).replace('__BRANCH__',repr(cfg['branch'])))
    time.sleep(2)
    health=remote(prefix+"print(json.dumps(dict(worker=json.loads((r/'WORKER_IDENTITY.json').read_text()) if (r/'WORKER_IDENTITY.json').exists() else None,exit=json.loads((r/'GUARD_EXIT.json').read_text()) if (r/'GUARD_EXIT.json').exists() else None)))")
    assert health['worker'] and health['exit'] is None,health
    env=dict(os.environ,SCORING_CONFIG=cfg['scoring_config'],JUDGE_SHARED_ROOT=str(local),
             LOCAL_NEUTRAL_ENTRY=str(local/'s.py'),LOCAL_ENTRY_MODE='guard')
    with (local/'score.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(local/'s.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=local)
    write(local/'DISPATCH_RECEIPT.json',dict(time=time.time(),remote_guard=started,
        remote_worker=health['worker'],local_guard_pid=child.pid,source_commit=current))
    state=read(cfg['state']);state.update(status='STARTED_MEDRA_GPU'+str(health['worker']['GPU']),
        remote_guard=started,remote_worker=health['worker'],local_guard_pid=child.pid,source_commit=current,
        monitoring='OFF',last_update=time.time());write(cfg['state'],state)
    print(json.dumps(dict(status='STARTED',guard=started,worker=health['worker'],local_guard=child.pid)))


if __name__=='__main__':main()
