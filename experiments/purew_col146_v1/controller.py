"""Finite O, reload, L, reload; each child is a fresh neutral process."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import common as c


def main():
    R=c.RUN
    assert c.read(R/'public/CPU_VALIDATION.json')['status']=='PASS'
    assert c.read(R/'private/BASELINE_REUSE_ADMISSION.json')['status']=='PASS'
    for arm in ('O','L'):
        A=R/'private/arms'/arm
        for action,barrier in (('gpu','single_R0_COMPLETE'),('reload','FINAL_CONSUMED')):
            c.write(R/'public/PROGRESS.json',dict(status='STAGE_PENDING',arm=arm,action=action,epoch=time.time()))
            assert not (A/'private'/('START_'+action+'.json')).exists(), 'No automatic replay of any attempted stage'
            while True:
                c.budget()
                uuid,free=subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(', ')
                assert uuid==c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['5']
                if int(free)>=45*1024:break
                time.sleep(30)
            fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
            Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
            env=dict(os.environ,COL_ARM=arm,ACTION=action,TASK_FILE=str(R/'private/tools/run.py'))
            child=subprocess.Popen([env['TRAIN_PYTHON'],'-u',entry],env=env,stdout=(R/'logs'/(arm+'_'+action+'.log')).open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
            argv=subprocess.check_output(['ps','-ww','-p',str(child.pid),'-o','args='],text=True).strip()
            start=dict(pid=child.pid,start_ticks=Path('/proc').joinpath(str(child.pid),'stat').read_text().split()[21],entry=entry,epoch=time.time(),argv=argv)
            c.write(A/'private'/('START_'+action+'.json'),start)
            assert not any(s in argv for s in ('wangbomin','Knowledge_editing','purew'))
            code=child.wait();assert code==0, 'Retained stage failure; no blind restart'
            assert (A/'private'/(barrier+'.json')).exists()
            assert c.read(A/'private'/('EXIT_'+action+'.json'))['exit_code']==0
    assert not any(not s.get('ended_epoch') for s in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    for n in ('TRAINING_GENERATION_COMPLETE','FINAL_CONSUMED','GENERATION_COMPLETE'):
        c.write(R/'private'/(n+'.json'),dict(status='COMPLETE',formal_updates=46720,epoch=time.time()))
    c.write(R/'private/EXIT_controller.json',dict(exit_code=0,epoch=time.time()))
    c.write(R/'public/PROGRESS.json',dict(status='GENERATION_COMPLETE_SCORING_PENDING',epoch=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        c.write(c.RUN/'private/failures/controller.json',dict(traceback=traceback.format_exc(),epoch=time.time()))
        raise
