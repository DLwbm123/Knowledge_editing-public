"""One fresh process for the audited, entirely unattempted second partition."""
import os
import time
import traceback
import interim20_runtime as w

r,c,RUN,PARENT=w.r,w.c,w.RUN,w.PARENT


def admit():
    w.admit()
    gate=c.read(RUN/'private/INTERIM20_REMAINING_ADMISSION1.json')
    audit=c.read(RUN/'private/INTERIM20_REMAINING_CPU_AUDIT1.json')
    assert gate['status']==audit['status']=='PASS' and gate['launch_authorized']
    assert gate['partition']==1 and gate['retained']==539 and gate['unattempted']==538
    assert gate['GPU'] in (2,5) and gate['user_device_authorized']
    assert gate['original_GPU_limit']==c.read(RUN/'STAGE_CAP.json')['GPU_seconds_limit']
    assert c.read(RUN/'private/DONE_m13_146_single_0.json')['outputs']==539
    assert not (RUN/'private/DONE_m13_146_single_1.json').exists()
    assert all(v.get('ended_epoch') for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'])
    return gate


def worker():
    gate=admit()
    with (RUN/'private/INTERIM20_REMAINING_ONCE1').open('x'):pass
    w.protocol.frozen(PARENT,c.read)
    w.worker.point=lambda t,a:PARENT/'private/candidates'/a/f"{t['order']}.pt"
    w.worker.configure();r.NATIVE_FORWARD_CAP=5000
    original=r.count
    def count(*args):
        original(*args)
        assert r.COUNTS['generation_attempts']<=538
        assert r.COUNTS['backwards']==r.COUNTS['updates']==r.COUNTS['operator_backwards']==0
        ledger=c.read(PARENT/'RESOURCE_LEDGER.json')
        parent_used=ledger['gpu_seconds_used']+sum(time.time()-v['started_epoch'] for v in ledger['gpu_sessions'] if not v.get('ended_epoch'))
        own=c.used()-c.read(RUN/'private/INHERITED_COST.json')['gpu_seconds_used']
        assert parent_used+own+1488.5244522094727<c.read(PARENT/'RUN_MANIFEST.json')['GPU_seconds_limit']
    r.count=count
    os.environ.update(GPU=str(gate['GPU']),PARTITION='1',ACTION='m13_146_single')
    import bank146
    # Gate once before the lease; the original evaluator also calls it before load.
    bank146.evaluate(40,2,admit,'INTERIM20_FROZEN1.json','SELECTED_SCOPE1.json','INTERIM20_REMAINING_SOURCE1.json')
    assert r.COUNTS['generation_attempts']==538
    old=c.read(RUN/'private/COUNTS_m13_146_single_0.json')
    totals={key:old.get(key,0)+r.COUNTS.get(key,0) for key in old.keys()|r.COUNTS.keys()}
    assert totals['generation_attempts']==1077 and len(list((RUN/'private/outputs').glob('*/*/*.json')))==1077
    c.write(RUN/'private/INTERIM20_GENERATION_COMPLETE1.json',dict(status='COMPLETE',outputs=1077,counts=totals,
        retained_outputs=539,new_outputs=538,original_failure_preserved=True,epoch=time.time()))


def controller():
    admit()
    with (RUN/'private/INTERIM20_REMAINING_CONTROLLER_ONCE1').open('x'):pass
    import interim20_controller as original
    os.environ['ACTION']='interim20_remaining_worker'
    original.execute('interim20_remaining.py')
    assert c.read(RUN/'private/INTERIM20_REMAINING_EXIT1.json')['exit_code']==0
    original.finish()


if __name__=='__main__':
    control=os.environ['ACTION']=='interim20_remaining_controller';code=0
    try:controller() if control else worker()
    except BaseException as error:
        code=1;c.write(RUN/'private'/('FAILURE_interim20_remaining_controller1.json' if control else 'FAILURE_interim20_remaining_worker1.json'),
            dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()))
        raise
    finally:
        if not control:r.save_counts()
        c.write(RUN/'private'/('INTERIM20_REMAINING_CONTROLLER_EXIT1.json' if control else 'INTERIM20_REMAINING_EXIT1.json'),dict(exit_code=code,epoch=time.time()))
