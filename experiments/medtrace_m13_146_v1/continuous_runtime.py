"""Actual prefix routing over retained single experts, without training."""
import os
import time
import traceback
import runtime146 as w
import continuous_protocol as f
r,c,RUN=w.r,w.c,w.RUN

def main():
    f.admit(RUN);w.configure();r.NATIVE_FORWARD_CAP=20000
    original=c.budget
    def budget():
        original();f.project_budget(RUN);assert not list((RUN/'private').glob('FAILURE_continuous*.json'))
    c.budget=budget
    with (RUN/'private'/('CONTINUOUS_ONCE_'+os.environ['PARTITION'])).open('x'):pass
    import bank146
    bank146.evaluate(146,2,lambda:f.admit(RUN),'CONTINUOUS_FROZEN.json','SCOPE_LOCK.json','GPU_SOURCE_VERSION.json','CONTINUOUS_NEW_PLAN.json')
    assert r.COUNTS['updates']==r.COUNTS['backwards']==r.COUNTS['operator_backwards']==0
    c.write(RUN/'private'/('CONTINUOUS_WORKER_COMPLETE_'+os.environ['PARTITION']+'.json'),dict(status='COMPLETE',counts=r.COUNTS,epoch=time.time()))
if __name__=='__main__':
    code=0
    try:main()
    except BaseException as error:
        code=1;c.write(RUN/'private'/('FAILURE_continuous_'+os.environ['PARTITION']+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
    finally:r.save_counts();c.write(RUN/'private'/('CONTINUOUS_WORKER_EXIT_'+os.environ['PARTITION']+'.json'),dict(exit_code=code,epoch=time.time()))
