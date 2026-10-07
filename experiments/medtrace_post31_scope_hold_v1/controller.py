"""Bounded first dependency: CPU inventory then admitted GPU mechanism audit."""
import os,sys,time,subprocess,tempfile,json,traceback
from pathlib import Path
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from common import budget,used,read,write,available


def launch(action):
    fd,name=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
    Path(name).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['RUN_ROOT']+'/private/tools/mechanism.py',run_name='__main__')\n")
    env=dict(os.environ,ACTION=action,CUDA_VISIBLE_DEVICES='6' if action=='A_GPU' else '',GPU='6')
    p=subprocess.Popen([os.environ['TRAIN_PYTHON'],'-u',name],env=env,stdout=(RUN/'logs'/(action+'.log')).open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
    write(RUN/'private'/('START_'+action+'.json'),dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],entry=name,epoch=time.time()))
    while p.poll() is None:
        try:budget()
        except BaseException:
            p.terminate();p.wait(timeout=30);raise
        time.sleep(5)
    assert p.returncode==0,action+' failed; retained evidence, no blind retry'


def main():
    if not (RUN/'public/A_CPU_AUDIT.json').exists():launch('A_CPU')
    budget();assert available(6)
    # Conservative first-stage admission; later work requires measured feature
    # throughput and real scope/reference eligibility, not invented negatives.
    estimate=1800;remaining=read(RUN/'RUN_MANIFEST.json')['GPU_seconds_limit']-used()
    assert remaining>=estimate+600
    write(RUN/'private/A_ADMISSION.json',dict(estimated_gpu_seconds=estimate,remaining_gpu_seconds=remaining,status='ADMITTED',two_training_limit=2))
    launch('A_GPU')
    write(RUN/'public/PROGRESS.json',dict(status='A_MECHANICAL_COMPLETE_NEXT_DEPENDENCY_ADMISSION_PENDING',GPU_hours=used()/3600,A1_semantic='PENDING_BLIND_REVIEW',B='NOT_STARTED_PENDING_SCOPE_QUALIFICATION',C='NOT_STARTED_PENDING_A_B',C2='NOT_ADMITTED',confirmation='BLOCKED_CONFIRMATION',experiment_complete=False,no_monitor=True))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(RUN/'private/FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),no_blind_retry=True));raise
