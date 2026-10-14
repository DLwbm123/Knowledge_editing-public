"""One selected full-lane controller; actual exits and coverage close every stage."""
import os
from pathlib import Path
import sys
import time
import traceback
import runtime146 as worker
sys.path.insert(0,str(Path(__file__).parent))
import selected_training146 as training
from native_followup_controller import launch,wait

c,r,RUN=worker.c,worker.r,worker.RUN


def retire(paths,last_consumer,name):
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    assert len(paths)==len(set(paths))
    assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve()/'private') for p in paths)
    c.write(RUN/'private'/name,dict(status='LAST_CONSUMER_VERIFIED',paths=list(map(str,paths)),
        bytes=sum(p.stat().st_size for p in paths),last_consumer=last_consumer,epoch=time.time()))
    for path in paths:path.unlink()


def main():
    training.require_admission()
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    with (RUN/'private/SELECTED_CONTROLLER_ONCE1').open('x'):pass
    def stage(action,file='selected_runtime146.py'):
        c.budget()
        jobs=[launch(action,gpu,gpu,file,'SELECTED_SOURCE1.json') for gpu in (0,1)]
        wait(jobs)
        assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
        training.check_counts()
    stage('selected_train')
    for part in (0,1):assert c.read(RUN/'private'/f'SELECTED_TRAIN_DONE_{part}.json')['owners']==training.assignments(part)
    for task in worker.tasks():
        for arm in ('C',)+training.comparison146.METHODS:
            data=c.read(RUN/'private/results'/arm/f"{task['order']}.json")
            assert data['save_load_exact'] and len(data['curve'])==160
            assert worker.point(task,arm).is_file()
    assert training.counts()['updates']==70080
    c.write(RUN/'private/SELECTED_CANDIDATES_FROZEN1.json',dict(candidates=292,methods=list(training.comparison146.METHODS),
        all_146_160_steps=True,pilot_owners_reused=2,C_calibrations=146,selection=False,epoch=time.time()))
    # C and optimizer/resume states have no remaining generation consumer.
    calibration=[worker.point(task,'C') for task in worker.tasks()]
    resumes=[RUN/'private/resume146'/f"{arm}_{task['order']}.pt" for task in worker.tasks() for arm in ('C','M13','FISHER_MIN')]
    retire(calibration+resumes,'complete C matching/Fmin budgets and all final save-load checks','SELECTED_CALIBRATION_RETIREMENT1.json')
    stage('m13_146_single');stage('m13_146_bank')
    plan=c.read(RUN/'private/EVAL_PLAN146.json')
    paths=list((RUN/'private/outputs').glob('*/*/*.json'))
    assert len(paths)==len(plan)==17081
    assert set(paths)=={worker.dest(row) for row in plan}
    # The queue performs the complete payload/binding validation once on ingest.
    totals=training.check_counts()
    assert totals['generation_attempts']==17081
    c.write(RUN/'private/SELECTED_GENERATION_COMPLETE1.json',dict(status='COMPLETE',outputs=17081,counts=totals,epoch=time.time()))
    weights=[worker.point(task,arm) for task in worker.tasks() for arm in training.comparison146.METHODS]+[RUN/'private/OPERATOR_FISHER.pt']
    retire(weights,'all real single/insertion/bank generation and durable payload bindings','SELECTED_WEIGHT_RETIREMENT1.json')
    wait([launch('selected_ingest',0,0,'selected_queue146.py','SELECTED_SOURCE1.json')])
    queue=RUN/'private/judge_native146_two_methods_astra_medium'
    while not ((queue/'ALL_WORKERS_COMPLETE.json').exists() and all((queue/'workers'/str(i)/'SCORER_DONE.json').exists() for i in range(4))):
        c.budget()
        assert not list((queue/'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(20)
    wait([launch('selected_report',0,0,'selected_report146.py','SELECTED_SOURCE1.json')])
    assert c.read(RUN/'public/NATIVE146_RESULTS.json')['status']=='SCORING_RESOLVED'
    c.write(RUN/'public/NATIVE146_EXECUTION_AUDIT.json',dict(status='PASS',counts=training.check_counts(),
        pilot_reused=True,automatic_neural_retries=0,all_GPU_leases_ended=True,owned_weights_released=True))
    c.write(RUN/'private/SELECTED_COMPLETE1.json',dict(status='COMPLETE',publication_pending=True,epoch=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1
        c.write(RUN/'private/FAILURE_selected_controller1.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False))
        raise
    finally:c.write(RUN/'private/SELECTED_EXIT1.json',dict(exit_code=code,epoch=time.time()))
