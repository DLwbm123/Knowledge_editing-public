"""Replace only the waiting CPU parent; preserve every running model worker."""
import os
from pathlib import Path
import signal
import time
import traceback
import refine_run as f
from module_recovery import live
import pipeline

c,RUN=f.c,f.RUN


def main():
    hold=c.read(RUN/'private/FULL_CONSUMER_CPU_HOLD1.json')
    parent=hold['controller']
    assert live(parent)
    assert Path('/proc',str(parent['pid']),'stat').read_text().split()[2]=='T'
    assert Path('/proc',str(parent['pid']),'cmdline').read_bytes().split(b'\0')[2].decode()==parent['entry']
    assert not (RUN/'private/REPORT_COMPLETE.json').exists()
    with (RUN/'private/FULL_HANDOFF_ONCE1').open('x'):pass
    c.write(RUN/'private/FULL_HANDOFF1.json',dict(status='CPU_PARENT_ONLY_SUPERSEDED',
        previous=parent,workers=hold['workers'],model_restarts=0,epoch=time.time()))
    os.kill(parent['pid'],signal.SIGTERM)
    os.kill(parent['pid'],signal.SIGCONT)
    while True:
        c.budget()
        assert not list((RUN/'private').glob('FAILURE*')), 'Retain failure; no retry'
        active=False
        for part,worker in enumerate(hold['workers']):
            if live(worker):active=True
            else:
                memory=c.read(RUN/'private'/f'MEMORY_guard_probe_{part}.json')
                assert memory['complete'] and memory['pid']==worker['pid']
                assert all((RUN/'private/PARENT_PARITY'/f'{owner}.json').exists() for owner in range(part+1,9,3))
        if not active:break
        time.sleep(10)
    f.report(retain_for_full=True)
    result=c.read(RUN/'public/RESULTS.json')
    if result['original_rule_witness_states']==0:return
    c.write(RUN/'public/PROGRESS.json',dict(status='REFINEMENT_PASS_FULL_CONSUMER_ADMISSION_PENDING',complete=False))
    ready=RUN/'private/FULL_CONSUMER_READY.json'
    while not ready.exists():
        c.budget();time.sleep(10)
    assert c.read(ready)['status']=='PASS'
    assert not (RUN/'private/START_CHAIN_full_controller_None.json').exists()
    pipeline.wait([pipeline.launch('full_run.py','full_controller')])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        c.write(RUN/'private/FAILURE_full_handoff1.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
