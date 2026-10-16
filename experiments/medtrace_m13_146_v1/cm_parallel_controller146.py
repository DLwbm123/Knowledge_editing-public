"""Adopt completed single generation and finish three disjoint bank workers."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import cm_controller146 as original

RUN=original.RUN
original.TOOLS=Path(__file__).parent
read,write=original.read,original.write


def main():
    original.protocol.admit(RUN,check_weights=True)
    assert read(RUN/'private/CM_PARALLEL_CPU_CHECK1.json')['status']=='PASS'
    with (RUN/'private/CM_PARALLEL_CONTROLLER_ONCE1').open('x'):pass
    jobs=[]
    for part,gpu in enumerate((3,4,5)):
        fd,entry=tempfile.mkstemp(prefix='e.',suffix='.py',dir=os.environ['TMPDIR']);os.close(fd)
        Path(entry).write_text('import runpy\nrunpy.run_path('+repr(str(original.TOOLS/'cm_parallel146.py'))+',run_name="__main__")\n')
        env=dict(os.environ,ACTION='cm_parallel_bank',PARTITION=str(part),GPU=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu))
        with (RUN/f'logs/cm_parallel_bank_{part}.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u',entry],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append(child)
        write(RUN/f'private/CM_PARALLEL_START_{part}.json',dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],entry=entry,GPU=gpu,epoch=time.time()))
    exits=[child.wait() for child in jobs]
    assert exits==[0,0,0],f'Parallel workers failed; no retry: {exits}'
    for part in (0,1):
        path=RUN/f'private/CM_GPU_EXIT_m13_146_single_{part}.json'
        while not path.exists():
            assert not list((RUN/'private').glob('FAILURE_cm*.json'))
            original.protocol.project_budget(RUN);time.sleep(5)
        assert read(path)['exit_code']==0
    original.finish()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;write(RUN/'private/FAILURE_cm_parallel_controller1.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()));raise
    finally:
        for phase in ('GPU','Judge'):
            try:original.merge(phase)
            except BaseException as error:write(RUN/f'private/CM_PARALLEL_COST_TRANSFER_{phase}_ERROR.json',dict(error=repr(error)))
        write(RUN/'private/CM_PARALLEL_CONTROLLER_EXIT1.json',dict(exit_code=code,epoch=time.time()))
