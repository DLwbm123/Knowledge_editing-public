"""Reuse the original real R0 single evaluation on an isolated GPU2 RUN."""
import os
import collections
from pathlib import Path
import time
import traceback
import runtime146 as worker
import interim20 as protocol

r,c,RUN=worker.r,worker.c,worker.RUN
PARENT=Path(os.environ['INTERIM20_PARENT'])


def admit():
    gate=c.read(RUN/'private/INTERIM20_ADMISSION1.json')
    assert gate['status']=='PASS' and gate['launch_authorized'] and gate['GPU']==2
    assert gate['owners']==list(range(1,21)) and gate['generation_cap']==protocol.CONSUMERS
    assert gate['max_GPU_seconds']==14400 and gate['training_updates']==0
    assert PARENT.resolve()!=RUN.resolve()
    assert c.read(PARENT/'private/SELECTED_PREFILL_RECOVERY_SOURCE1.json')['commit']==gate['parent_source']
    assert c.read(RUN/'private/EVAL_PLAN146.json')==protocol.plan_subset(c.read(PARENT/'private/EVAL_PLAN146.json'))
    return gate


def main():
    gate=admit()
    with (RUN/'private/INTERIM20_ONCE1').open('x'):pass
    # Only a CPU waiter exists until both methods have all twenty immutable finals.
    wait_until=min(c.read(RUN/'RUN_MANIFEST.json')['deadline_epoch'],time.time()+7200)
    while True:
        if all((PARENT/f'private/results/{a}/20.json').exists() for a in ('M13','FISHER_MIN')):break
        c.budget();assert time.time()<wait_until,'First20 not complete within bounded wait'
        assert not list((PARENT/'private').glob('FAILURE*prefill*'))
        time.sleep(20)
    protocol.frozen(PARENT,c.read)
    c.write(RUN/'private/INTERIM20_FROZEN1.json',dict(candidates=40,steps=160,epoch=time.time()))
    worker.point=lambda t,a:PARENT/'private/candidates'/a/f"{t['order']}.pt"
    worker.configure();r.NATIVE_FORWARD_CAP=5000
    original_count=r.count
    def bounded(*args):
        original_count(*args)
        assert r.COUNTS['generation_attempts']<=protocol.CONSUMERS
        assert r.COUNTS['updates']==r.COUNTS['backwards']==r.COUNTS['operator_backwards']==0
        ledger=c.read(PARENT/'RESOURCE_LEDGER.json')
        parent_used=ledger['gpu_seconds_used']+sum(time.time()-v['started_epoch'] for v in ledger['gpu_sessions'] if not v.get('ended_epoch'))
        own=c.used()-c.read(RUN/'private/INHERITED_COST.json')['gpu_seconds_used']
        assert parent_used+own+1488.5244522094727<c.read(PARENT/'RUN_MANIFEST.json')['GPU_seconds_limit']
    r.count=bounded
    import bank146
    totals=collections.Counter()
    for part in (0,1):
        for key in r.COUNTS:r.COUNTS[key]=0
        os.environ.update(GPU='2',PARTITION=str(part),ACTION='m13_146_single')
        bank146.evaluate(40,2,admit,'INTERIM20_FROZEN1.json','SELECTED_SCOPE1.json','INTERIM20_SOURCE1.json')
        totals.update(r.COUNTS)
    assert totals['generation_attempts']==protocol.CONSUMERS
    assert len(list((RUN/'private/outputs').glob('*/*/*.json')))==protocol.CONSUMERS
    c.write(RUN/'private/INTERIM20_GENERATION_COMPLETE1.json',dict(status='COMPLETE',outputs=protocol.CONSUMERS,counts=dict(totals),epoch=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/'private/FAILURE_interim20_runtime1.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()))
        raise
    finally:
        r.save_counts();c.write(RUN/'private/INTERIM20_EXIT1.json',dict(exit_code=code,epoch=time.time()))
