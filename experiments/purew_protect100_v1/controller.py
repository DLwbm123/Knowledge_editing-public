"""One neutral GPU process, then one fresh native reload; no blind stage retry."""
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import common as c


def main():
    R = c.RUN
    assert c.read(R/'public/CPU_VALIDATION.json')['status'] == 'PASS'
    assert c.read(R/'public/DATA_QUALIFICATION.json')['status'] == 'PASS'
    for action, barrier in (('gpu', 'TRAINING_GENERATION_COMPLETE'), ('reload', 'FINAL_CONSUMED')):
        assert not (R/'private'/('START_'+action+'.json')).exists(), 'Review any attempted stage instead of replaying'
        while True:
            c.budget()
            uuid, free = subprocess.check_output(['nvidia-smi', '-i', '5', '--query-gpu=uuid,memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip().split(', ')
            assert uuid == c.read(R/'PLAN_CONFIG.json')['hardware']['UUIDs']['5']
            if int(free) >= 45*1024: break
            time.sleep(30)
        fd, entry = tempfile.mkstemp(prefix='e.', suffix='.py', dir=os.environ['TMPDIR']); os.close(fd)
        Path(entry).write_text("import os,runpy\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
        env = dict(os.environ, ACTION=action, TASK_FILE=str(R/'private/tools/run.py'))
        child = subprocess.Popen([env['TRAIN_PYTHON'], '-u', entry], env=env,
                stdout=(R/'logs'/(action+'.log')).open('ab'), stderr=subprocess.STDOUT, start_new_session=True)
        argv = subprocess.check_output(['ps', '-ww', '-p', str(child.pid), '-o', 'args='], text=True).strip()
        c.write(R/'private'/('START_'+action+'.json'), dict(pid=child.pid, entry=entry, argv=argv,
                start_ticks=Path('/proc').joinpath(str(child.pid), 'stat').read_text().rsplit(')',1)[1].split()[19], epoch=time.time()))
        assert not any(x in argv for x in ('wangbomin', 'Knowledge_editing', 'purew'))
        code = child.wait()
        assert code == 0 and (R/'private'/(barrier+'.json')).exists(), 'Preserve failed stage; never automatically replay'
        assert c.read(R/'private'/('EXIT_'+action+'.json'))['exit_code'] == 0
    assert not any(not x.get('ended_epoch') for x in c.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    c.write(R/'private/GENERATION_COMPLETE.json', dict(status='GENERATED_NOT_SCORED', formal_updates=16000, epoch=time.time()))
    c.write(R/'private/EXIT_controller.json', dict(exit_code=0, epoch=time.time()))
    c.write(R/'public/PROGRESS.json', dict(status='GENERATION_COMPLETE_SCORING_PENDING', epoch=time.time()))


if __name__ == '__main__':
    try: main()
    except BaseException:
        c.write(c.RUN/'private/failures/controller.json', dict(traceback=traceback.format_exc(), epoch=time.time()))
        raise
