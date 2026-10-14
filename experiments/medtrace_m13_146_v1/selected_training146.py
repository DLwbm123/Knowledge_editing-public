"""Two persistent models consume only owners 3..146 after the pilot is frozen."""
import os
from pathlib import Path
import time
import traceback
import comparison146
import contract
import trajectory
import runtime146 as worker

r,c,RUN=worker.r,worker.c,worker.RUN
OWNERS=list(range(3,147))


def assignments(part):
    assert part in (0,1)
    return OWNERS[part::2]


def require_admission():
    gate=c.read(RUN/'private/SELECTED_RUNTIME_ADMISSION1.json')
    assert gate['status']=='PASS' and gate['execution_ready'] and gate['launch_authorized']
    assert gate['owners']==OWNERS and gate['physical_GPUs']==[0,1]
    assert gate['packet_binding']==c.digest(worker.packet())
    assert contract.source_admission(worker.packet())['status']=='PASS'
    scope=c.read(RUN/'private/SELECTED_SCOPE1.json')
    assert gate['scope_binding']==c.digest(scope)
    assert scope['primary_methods']==list(comparison146.METHODS)
    assert scope['source_recipe']==contract.SOURCE_RECIPE and scope['source_weights']==list(contract.SOURCE_WEIGHTS)
    assert scope['steps']==160 and scope['parameters']==7168 and not scope['alignment_rule']
    assert scope['M13_flags']==[True,False,True] and scope['FISHER_MIN_gain_fraction']==.9
    assert scope['FISHER_MIN_budget_fraction']==.1
    assert gate['source_commit']==c.read(RUN/'private/SELECTED_SOURCE1.json')['commit']
    assert gate['plan_binding']==c.digest(c.read(RUN/'PLAN_CONFIG.json'))
    assert c.read(RUN/'PLAN_CONFIG.json')['maximum_counts']==comparison146.CAPS
    assert gate['cpu_checks_binding']==c.digest(c.read(RUN/'private/SELECTED_TRAINING_CPU_CHECKS1.json'))
    assert c.read(RUN/'private/SELECTED_TRAINING_CPU_CHECKS1.json')['status']=='PASS'
    assert c.read(RUN/'private/NATIVE_FMIN_EXIT1.json')['exit_code']==0
    for owner in (1,2):
        for prefix in ('CALIBRATION','NATIVE_M13','NATIVE_FMIN'):
            assert c.read(RUN/'private'/f'{prefix}_COMPLETE_{owner}.json')['steps']==160
    assert c.read(RUN/'public/FISHER_ADMISSION.json')['status']=='PASS'
    assert c.read(RUN/'private/NATIVE_FISHER_COMPLETE1.json')['status']=='COMPLETE'
    return gate


def counts():
    rows=[c.read(p) for p in (RUN/'private').glob('COUNTS_*.json')]
    return {key:sum(row.get(key,0) for row in rows) for key in r.COUNTS}


def check_counts():
    totals=counts()
    for key,cap in comparison146.CAPS.items():
        if key not in ('evaluation_consumers','Judge_attempts'):
            assert totals.get(key,0)<=cap,(key,totals.get(key),cap)
    return totals


def train():
    import sys
    sys.path.insert(0,str(Path(__file__).parent))
    import native_calibration
    import native_followup
    import native_fmin
    require_admission()
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION'])
    assert gpu==part and gpu in (0,1)
    with (RUN/'private'/f'SELECTED_TRAIN_ONCE_{part}').open('x'):pass
    r.NATIVE_FORWARD_CAP=comparison146.CAPS['non_generation_forwards']
    worker.configure()
    with r.p.lease(gpu):
        runtime,bindings,handle=r.load(gpu)
        # Task-specific finite counters replace r.count; the model hook follows it.
        handle.remove()
        handle=runtime.model.register_forward_pre_hook(lambda *args:r.count(*args))
        try:
            for owner in assignments(part):
                os.environ['OWNER']=str(owner)
                task=worker.tasks()[owner-1]
                assert all(not worker.point(task,arm).exists() for arm in ('C',)+comparison146.METHODS)
                assert all(Path(v['image_path']).is_file() for v in trajectory.source_rows(worker.packet(),owner))
                for arm,run in (('C',native_calibration.main),('M13',native_followup.train),('FISHER_MIN',native_fmin.main)):
                    c.budget();check_counts()
                    c.write(RUN/'private'/f'SELECTED_PROGRESS_{part}.json',dict(owner=owner,arm=arm,epoch=time.time()))
                    run(full=True,loaded=(runtime,bindings))
                    result=c.read(RUN/'private/results'/arm/f'{owner}.json')
                    assert len(result['curve'])==160 and worker.point(task,arm).is_file()
                    r.save_counts();check_counts()
            c.write(RUN/'private'/f'SELECTED_TRAIN_DONE_{part}.json',dict(status='COMPLETE',owners=assignments(part),
                counts=dict(r.COUNTS),source=c.read(RUN/'private/SELECTED_SOURCE1.json')['commit'],epoch=time.time()))
        finally:handle.remove();r.save_counts()


if __name__=='__main__':
    try:train()
    except BaseException as error:
        r.save_counts()
        c.write(RUN/'private'/f'FAILURE_selected_train_{os.environ.get("PARTITION")}.json',
            dict(error=repr(error),traceback=traceback.format_exc(),counts=r.COUNTS,epoch=time.time(),automatic_retry=False))
        raise
