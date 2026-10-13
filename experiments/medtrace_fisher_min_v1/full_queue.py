"""New isolated queue; reuse accepted or missing historical identities once."""
import os
import sys
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/source')
import m13_queue as base
base.q.ROOT=base.RUN/'private/judge_fisher_min_astra_medium'
base.q.BATCH_LEDGER='Astra_fisher_min_batches'
base.EPOCH='MEDTRACE_FISHER_MIN_20261014_V1'
base.MAX_JUDGE=base.CONSUMERS=1158
base.PARENTS+=('M13_PARENT','REFINE_PARENT','MULTICORE_PARENT')
q,RUN,MAX_JUDGE,CONSUMERS=base.q,base.RUN,base.MAX_JUDGE,base.CONSUMERS
original=base.request


def request(db,operation):
    if operation['action']=='next' and list((RUN/'private').glob('FAILURE_*')):
        return dict(state=dict(status='STOPPED',reason='OWN_STAGE_FAILED'),rows=[])
    return original(db,operation)
base.request=request
if __name__=='__main__':base.main()
