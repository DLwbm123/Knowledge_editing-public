"""Four sealed workers, exact reuse and audited retries for the new finite run."""
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import sys
sys.path.insert(0,str(Path(__file__).parent))
import balance_protocol as f
import cm_queue146 as shared
import queue146 as base

def main():
    import astra_queue as a
    from fine_queue import full_identity,selfcheck
    q,run=a.q,a.RUN;q.ROOT=run/'private'/f.QUEUE;q.BATCH_LEDGER='Astra_balance146_batches';q.EFFORT='medium';shared.protocol=f
    base.MAX_JUDGE=f.CONSUMERS;base.contract=SimpleNamespace(original=shared.contract.original,METHOD_CONSUMERS=f.METHOD_CONSUMERS)
    op=json.loads(os.environ['QUEUE_REQUEST']);q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda _:None
        if op['action'] in ('next','reserve','retry_charge') and (list((run/'private').glob('FAILURE_balance*.json')) or list((q.ROOT/'workers').glob('*/SCORER_FAILURE.json'))):
            assert op['action']=='next','Stage failure; stop new attempts'
            print(json.dumps(dict(state=dict(status='STOPPED',reason='OWN_STAGE_FAILED'),rows=[])));return
        f.admit(run)
        if not (run/'private/BALANCE_GENERATION_COMPLETE1.json').exists():
            assert op['action'] in ('next','status','reserved')
            print(json.dumps([] if op['action']=='reserved' else dict(state=dict(status='GENERATING'),rows=[])));return
        judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
        epoch=dict(epoch=f.EPOCH,**judge,workers=4,batch_size=50,failed_request_retries_authorized=True,valid_rejudging=False,selfcheck=selfcheck())
        manifest=q.ROOT/'EPOCH_MANIFEST.json'
        if manifest.exists():assert q.read(manifest)==epoch
        else:q.write(manifest,epoch)
        if not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
            known=base.known_scores(q,q.read(run/'private/M13_146_REUSE_QUEUES.json'),judge)
            outputs=((p,q.read(p)) for p in sorted((run/'private/outputs').glob('*/*/*.json')))
            with db:summary=base.ingest_generated(q,db,q.read(run/'private/EVAL_PLAN146.json'),outputs,full_identity,judge,q.digest(q.read(run/'private/SCOPE_LOCK.json')),known)
            pending=[v[0] for v in db.execute("SELECT key FROM payload WHERE status='PENDING'")]
            assert len(pending)<=2956
            q.write(q.ROOT/'RETRY_SCOPE.json',dict(pending_keys=pending,model_retry_authorized=True))
            q.write(q.ROOT/'READY.json',dict(summary,pending=len(pending),epoch=time.time()))
        if op['action']=='retry_charge':
            answer=shared.charge_retry(q,run,db,op,q.read(q.ROOT/'RETRY_SCOPE.json'))
        else:
            if op['action']=='reserve':
                scope=q.read(q.ROOT/'RETRY_SCOPE.json');shared.check_resume_reservation(db,op['keys'],scope)
                gate=q.read(run/'private/CM_ADMISSION1.json');parent=Path(gate['parent'])
                ledger=q.read(run/'RESOURCE_LEDGER.json');prior=q.read(run/'private/INHERITED_COST.json');p=q.read(parent/'RESOURCE_LEDGER.json')
                own=ledger['Judge_attempts']-prior['Judge_attempts'];assert own+len(op['keys'])<=5000
                assert p['Judge_attempts']+own-p.get('balance146_Judge_merged',{}).get('increment',0)+len(op['keys'])<=q.read(parent/'RUN_MANIFEST.json')['Judge_limit']
            answer=base.coverage(db) if op['action']=='ingest' else shared.dispatch(a,q,db,op,4)
        coverage=base.coverage(db)
        if coverage['all_attempts_resolved'] and shared.all_workers_done(q.ROOT,4):q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(coverage,workers=4,epoch=time.time()))
        print(json.dumps(answer))

if __name__=='__main__':main()
