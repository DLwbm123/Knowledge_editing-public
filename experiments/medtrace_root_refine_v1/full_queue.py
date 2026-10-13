"""Reuse one-attempt full-payload scoring, including frozen M13 results."""
import m13_queue as base
base.q.ROOT=base.RUN/'private/judge_full_astra_medium'
base.q.BATCH_LEDGER='Astra_full_batches'
base.EPOCH='MEDTRACE_FULL_REFINE_20261013_V1'
base.PARENTS=base.PARENTS+('M13_PARENT',)
q,RUN,MAX_JUDGE,CONSUMERS=base.q,base.RUN,base.MAX_JUDGE,base.CONSUMERS
request_original=base.request

def request(db,operation):
    if operation['action']=='next' and list((RUN/'private').glob('FAILURE_full*')):
        return dict(state=dict(status='STOPPED',reason='OWN_STAGE_FAILED'),rows=[])
    return request_original(db,operation)

base.request=request
if __name__=='__main__':base.main()
