"""A CPU-only conditional handoff; no new models before both admissions."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import prepare146 as prepare
from followup146 import gate

UP, RUN = prepare.UP, prepare.RUN


def main():
    with (RUN/'private/PREPARED_HANDOFF_ONCE').open('x'):
        pass
    while not (UP/'private/FULL_COMPLETE.json').exists():
        if list((UP/'private').glob('FAILURE_full*')):
            prepare.write(RUN/'private/PREPARED_HANDOFF_NOT_STARTED.json', dict(reason='UPSTREAM_FAILED', new_model_calls=0));return
        time.sleep(20)
    if not gate(prepare.read(UP/'public/FULL_RESULTS.json')):
        prepare.write(RUN/'private/PREPARED_HANDOFF_NOT_STARTED.json', dict(reason='UPSTREAM_COMPATIBILITY_FAILED', new_model_calls=0));return
    requirements = ['SOURCE_REVIEW146.json', 'FROZEN_TASKS146.json', 'FROZEN_ANCHORS146.json']
    while not all((RUN/'private'/name).exists() for name in requirements) or not (UP/'private/FULL_DELIVERY.json').exists():
        prepare.write(RUN/'public/FULL146_WAIT.json', dict(status='SOURCE_AND_DELIVERY_ADMISSION_PENDING',
            training_started=False, new_model_calls=0, epoch=time.time()))
        time.sleep(30)
    prepare.admit()
    fd, entry = tempfile.mkstemp(prefix='e.', suffix='.py', dir='/data/bmw/tmp');os.close(fd)
    Path(entry).write_text("import os,sys,runpy\nsys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')\nrunpy.run_path(os.environ['TASK_FILE'],run_name='__main__')\n")
    env = dict(os.environ, **prepare.read(RUN/'private/LAUNCH_ENV.json'))
    env.update(RUN_ROOT=str(RUN), FOLLOWUP_ROOT=str(RUN), UPSTREAM_ROOT=str(UP),
               TASK_FILE=str(RUN/'private/tools/full146_run.py'), ACTION='full146_controller')
    log = RUN/'logs/full146_controller.log'
    p = subprocess.Popen([env['TRAIN_PYTHON'], '-u', entry], env=env, stdout=log.open('ab'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    prepare.write(RUN/'private/FULL146_START1.json', dict(pid=p.pid, entry=entry,
        start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21], epoch=time.time()))
    code = p.wait()
    prepare.write(RUN/'private/FULL146_EXIT1.json', dict(exit_code=code, epoch=time.time()))
    assert code == 0, 'Actual146 failure; preserve state and never retry here'


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        prepare.write(RUN/'private/FAILURE_full146_handoff.json', dict(error=repr(error),
            traceback=traceback.format_exc(), epoch=time.time()))
        raise
