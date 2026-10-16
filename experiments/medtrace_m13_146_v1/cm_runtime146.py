"""Fresh GPU1 process per fixed stage/partition; only unattempted consumers."""
import os
from pathlib import Path
import time
import sys
import traceback
import runtime146 as worker
import cm_eval146 as protocol

r,c,RUN=worker.r,worker.c,worker.RUN


def main():
    gate=protocol.admit(RUN,check_weights=True);parent=Path(gate['parent'])
    assert protocol.read(RUN/'private/CM_CPU_CHECK1.json')['status']=='PASS'
    assert protocol.read(RUN/'private/CM_SOURCE1.json')['commit']!='CPU_PREPARATION_NOT_LAUNCHED'
    action=os.environ['ACTION'];part=int(os.environ['PARTITION'])
    assert action in ('m13_146_single','m13_146_bank') and part in (0,1) and os.environ['GPU']=='1'
    label=action+'_'+str(part)
    with (RUN/'private'/('CM_GPU_ONCE_'+label)).open('x'):pass
    rows=c.read(RUN/'private/CM_NEW_PLAN1.json')
    assigned=[v for i,v in enumerate(rows) if i%2==part and ((v['mode'] in ('base','single')) if action=='m13_146_single' else (v['mode'] in ('bank','insertion')))]
    assert all(not worker.dest(row).exists() for row in assigned)
    worker.point=lambda task,arm:parent/f"private/candidates/{arm}/{task['order']}.pt"
    worker.configure();r.NATIVE_FORWARD_CAP=protocol.NON_GENERATION_FORWARDS
    original=r.count
    def count(*args):
        original(*args);protocol.project_budget(RUN)
        assert r.COUNTS['generation_attempts']<=len(assigned)
        assert r.COUNTS['updates']==r.COUNTS['backwards']==r.COUNTS['operator_backwards']==r.COUNTS['local_solves']==0
    r.count=count
    sys.path.insert(0,str(Path(__file__).parent))
    import bank146
    bank146.evaluate(292,2,lambda:protocol.admit(RUN),'CM_FROZEN1.json','CM_SCOPE1.json','CM_SOURCE1.json','CM_NEW_PLAN1.json')
    assert r.COUNTS['generation_attempts']==len(assigned)
    c.write(RUN/'private'/('CM_GPU_COMPLETE_'+label+'.json'),dict(status='COMPLETE',outputs=len(assigned),counts=r.COUNTS,epoch=time.time()))


if __name__=='__main__':
    code=0;label=os.environ['ACTION']+'_'+os.environ['PARTITION']
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/'private'/('FAILURE_cm_runtime_'+label+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,epoch=time.time()));raise
    finally:
        r.save_counts();c.write(RUN/'private'/('CM_GPU_EXIT_'+label+'.json'),dict(exit_code=code,epoch=time.time()))
