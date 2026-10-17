"""Four isolated one-attempt scorers and one finite publication tail."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
W=Path(os.environ['WORKSPACE']);HERE=W/'experiments/purew_odds_hinge_first1_v1'
R='/data/bmw/Knowledge_editing/outputs/purew-odds-hinge-first1-v1-20261016/run'
BRANCH='experiment/purew-ecw-v1-20261014'

def start_consumers(startup):
    state=Path(os.environ.get('CONTINUATION_STATE',str(W/'.local_run/odds_hinge_first1_v1_20261016')));state.mkdir(parents=True,exist_ok=True)
    assert not (state/'SCORERS.json').exists(), 'Consumer launch requires identity review, not duplication'
    (state/'STARTUP.json').write_text(json.dumps(startup,indent=2))
    folder=Path(tempfile.mkdtemp(prefix='e.',dir='/private/tmp'));workers=[]
    old=W/'experiments/medtrace_full_method_comparison_v2_20261003/scorer.py'
    for i in range(4):
        leaf=folder/str(i);leaf.mkdir(mode=0o700)
        entry=folder/(str(i)+'.py');entry.write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        env=dict(os.environ,RUN_ROOT=R,SCORER_STATE=str(leaf),JUDGE_SHARED_ROOT=str(folder),
            SCORER_SOURCE=str(W/'experiments/medtrace_full_method_comparison_v2_20261003/execution_source'),
            SCORER_BASE=str(old),TASK_FILE=str(W/'experiments/purew_original146_v1/scorer146.py'),JUDGE_QUEUE_FILE='queue_revision.py',
            JUDGE_RESULT_FOLDER='judge_purew146/worker'+str(i),JUDGE_EPOCH='PUREW_ODDS_HINGE_FIRST1_V1_20261016',JUDGE_EFFORT='medium',JUDGE_WORKER=str(i),
            HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
        child=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(leaf/'worker.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
        identity=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','lstart=,args='],text=True).strip()
        assert not any(x in identity for x in ('wangbomin','Knowledge_editing','purew'))
        workers.append(dict(pid=child.pid,entry=str(entry),identity=identity,worker=i,log=str(leaf/'worker.log')))
        (state/'SCORERS.json').write_text(json.dumps(dict(root=str(folder),workers=workers),indent=2))
    entry=folder/'tail.py';entry.write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env=dict(os.environ,WORKSPACE=str(W),CONTINUATION_STATE=str(state),GROUP=str(folder),TASK_FILE=str(HERE/'finish.py'),RUN_ROOT=R,BRANCH=BRANCH,
             HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897',ALL_PROXY='http://127.0.0.1:7897',NO_PROXY='',no_proxy='')
    tail=subprocess.Popen([sys.executable,'-u',str(entry)],env=env,stdout=(folder/'tail.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    identity=subprocess.check_output(['ps','-ww','-p',str(tail.pid),'-o','lstart=,args='],text=True).strip()
    (state/'FINITE_TAIL.json').write_text(json.dumps(dict(pid=tail.pid,entry=str(entry),identity=identity,log=str(folder/'tail.log')),indent=2))
    print(json.dumps(dict(startup=startup,scorers=workers,finite_tail=tail.pid),indent=2))
