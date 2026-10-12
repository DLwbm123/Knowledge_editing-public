"""Publish/freeze first; install one bounded GPU5 chain and four finite score consumers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

W=Path(os.environ['WORKSPACE'])
HERE=W/'experiments/purew_original146_v1'
R='/data/bmw/Knowledge_editing/outputs/purew-original146-v1-20261012/run'
BRANCH='experiment/purew-original146-v1-20261012'


def launch():
    assert not subprocess.check_output(['git','status','--porcelain','--','experiments/purew_original146_v1','reports/purew_original146_v1_20261012'],cwd=W,text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    protocol=json.loads((W/'reports/purew_original146_v1_20261012/PROTOCOL.json').read_text())
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ('bootstrap.py','deploy.py','finish.py','scorer146.py')}
    payload=dict(protocol=protocol,files=files,commit=commit,branch=BRANCH)
    code=json.dumps((HERE/'bootstrap.py').read_text())+'\n'+json.dumps(payload)+'\n'
    remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',"/data/bmw/envs/v0/bin/python -c 'import sys,json;exec(json.loads(sys.stdin.readline()))'"],input=code,text=True,capture_output=True,timeout=180,check=True)
    startup=json.loads(remote.stdout)
    start_consumers(startup)


def start_consumers(startup):
    state=W/'.local_run';state.mkdir(exist_ok=True)
    (state/'STARTUP.json').write_text(json.dumps(startup,indent=2))
    folder=Path(tempfile.mkdtemp(prefix='e.',dir='/private/tmp'));workers=[]
    old=W/'experiments/medtrace_full_method_comparison_v2_20261003/scorer.py'
    for i in range(4):
        leaf=folder/str(i);leaf.mkdir(mode=0o700)
        entry=folder/(str(i)+'.py');entry.write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        env=dict(os.environ,RUN_ROOT=R,SCORER_STATE=str(leaf),JUDGE_SHARED_ROOT=str(folder),
            SCORER_SOURCE=str(W/'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'),
            SCORER_BASE=str(old),TASK_FILE=str(HERE/'scorer146.py'),JUDGE_QUEUE_FILE='queue146.py',
            JUDGE_RESULT_FOLDER='judge_purew146/worker'+str(i),JUDGE_EPOCH='PUREW_ORIGINAL146_V1_20261012',JUDGE_EFFORT='medium',JUDGE_WORKER=str(i),
            HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
        child=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(leaf/'worker.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        identity=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','lstart=,args='],text=True).strip()
        assert not any(x in identity for x in ('wangbomin','Knowledge_editing','purew'))
        workers.append(dict(pid=child.pid,entry=str(entry),identity=identity,worker=i,log=str(leaf/'worker.log')))
        (state/'SCORERS.json').write_text(json.dumps(dict(root=str(folder),workers=workers),indent=2))
    entry=folder/'tail.py';entry.write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env=dict(os.environ,WORKSPACE=str(W),GROUP=str(folder),TASK_FILE=str(HERE/'finish.py'),RUN_ROOT=R,BRANCH=BRANCH,
             HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    tail=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(folder/'tail.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    identity=subprocess.check_output(['ps','-ww','-p',str(tail.pid),'-o','lstart=,args='],text=True).strip()
    (state/'FINITE_TAIL.json').write_text(json.dumps(dict(pid=tail.pid,entry=str(entry),identity=identity,log=str(folder/'tail.log')),indent=2))
    print(json.dumps(dict(startup=startup,scorers=workers,finite_tail=tail.pid),indent=2))


if __name__=='__main__':
    launch()
