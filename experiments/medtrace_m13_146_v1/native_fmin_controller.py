"""Run only the two newly admitted Fmin consumers; no retry or new Fisher."""
import os
import time
import traceback
from native_followup_controller import launch,wait,c,RUN


def main():
    gate=c.read(RUN/'private/NATIVE_FMIN_ADMISSION1.json')
    assert gate['status']=='PASS' and gate['owners']==[1,2]
    assert c.read(RUN/'private/NATIVE_FMIN_CPU_CHECKS1.json')['status']=='PASS'
    assert c.read(RUN/'private/NATIVE_FOLLOWUP_EXIT1.json')['exit_code']==0
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    with (RUN/'private/NATIVE_FMIN_CONTROLLER_ONCE1').open('x'):pass
    c.write(RUN/'private/NATIVE_FMIN_PROGRESS1.json',dict(stage='FIRST_TWO_FMIN_RUNNING',epoch=time.time()))
    wait([launch('native_fmin',gpu,gpu+1,'native_fmin.py','NATIVE_FMIN_SOURCE1.json') for gpu in (0,1)])
    for owner in (1,2):assert c.read(RUN/'private'/f'NATIVE_FMIN_COMPLETE_{owner}.json')['steps']==160
    counts=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
    total={k:sum(v.get(k,0) for v in counts) for k in counts[0]}
    assert total['updates']==960 and total['generation_attempts']==0
    for key,cap in c.read(RUN/'PLAN_CONFIG.json')['maximum_counts'].items():
        if key!='Judge_attempts':assert total.get(key,0)<=cap
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    c.write(RUN/'private/NATIVE_FMIN_PROGRESS1.json',dict(stage='FIRST_TWO_METHODS_FROZEN_EVALUATION_PENDING',
        counts=total,full146_started=False,performance_evaluation_complete=False,epoch=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/'private/FAILURE_native_fmin_controller1.json',
            dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
    finally:c.write(RUN/'private/NATIVE_FMIN_EXIT1.json',dict(exit_code=code,epoch=time.time()))
