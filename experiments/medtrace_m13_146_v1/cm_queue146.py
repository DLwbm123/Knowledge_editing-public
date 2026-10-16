"""Sealed scoring with optional four-worker dispatch; identities and caps unchanged."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parent))
import contract
import cm_eval146 as protocol
import queue146 as base

base.MAX_JUDGE=protocol.CONSUMERS
base.contract=SimpleNamespace(original=contract.original,METHOD_CONSUMERS=protocol.METHOD_CONSUMERS)


def dispatch(a,q,db,operation,workers):
    if workers==1:return q.request(db,operation)
    assert workers==4
    write=q.write
    def guarded_write(path,data):
        if Path(path).name!='ALL_WORKERS_COMPLETE.json':write(path,data)
    q.write=guarded_write
    try:return a.request(db,operation)
    finally:q.write=write


def all_workers_done(root,workers):
    return all((root/f'workers/{w}/SCORER_DONE.json').exists() for w in range(workers))


def recovery(run):
    path=run/'private/CM_SCORING_RESUME1.json'
    if not path.exists():return None
    data=json.loads(path.read_text())
    assert data['status']=='ADMITTED' and len(data['pending_keys'])==3603
    assert len(set(data['pending_keys']))==3603 and len(data['preserved'])==1399
    return data


def new_failures(run,resume):
    old={} if resume is None else resume['prior_failures']
    return [p for p in (run/'private').glob('FAILURE_cm*.json')
            if old.get(p.name)!=json.loads(p.read_text())]


def completion_root(root,resume):
    return root/'resume1' if resume is not None else root


def check_resume_reservation(db,keys,resume):
    if resume is None:return
    assert set(keys)<=set(resume['pending_keys']), 'Outside authorized unattempted set'
    assert all(tuple(db.execute('SELECT status,batch FROM payload WHERE key=?',(k,)).fetchone())==('PENDING',None) for k in keys)


def main():
    import astra_queue as a
    from fine_queue import full_identity,selfcheck
    q,run=a.q,a.RUN;q.ROOT=run/'private'/protocol.QUEUE
    q.BATCH_LEDGER='Astra_cm146_batches';q.EFFORT='medium'
    operation=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"status"}'))
    amendment=run/'private/CM_PARALLEL_SCORING1.json'
    workers=4 if amendment.exists() else 1
    if workers==4:
        assert q.read(amendment)['status']=='ADMITTED'
        assert q.read(run/'private/CM_PARALLEL_SCORING_CPU_CHECK1.json')['status']=='PASS'
    resume=recovery(run);completed=completion_root(q.ROOT,resume)
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda _:None
        if (new_failures(run,resume) or (resume is not None and list((completed/'workers').glob('*/SCORER_FAILURE.json')))) and operation['action'] in ('next','reserve'):
            assert operation['action']!='reserve','Own stage failed'
            print(json.dumps(dict(state=dict(status='STOPPED',reason='OWN_STAGE_FAILED'),rows=[])));return
        if (run/'private/CM_GENERATION_COMPLETE1.json').exists():
            gate=protocol.admit(run)
            judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
            epoch=dict(epoch=protocol.EPOCH,**judge,workers=workers,batch_size=50,attempts_per_payload=1,
                maximum_new_Judge=gate['new_Judge_cap'],scientific_lock=q.digest(q.read(run/'private/CM_SCOPE1.json')),selfcheck=selfcheck())
            path=q.ROOT/'EPOCH_MANIFEST.json'
            if path.exists():assert q.read(path)==epoch
            else:q.write(path,epoch)
            if not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
                known=base.known_scores(q,q.read(run/'private/M13_146_REUSE_QUEUES.json'),judge)
                outputs=((p,q.read(p)) for p in sorted((run/'private/outputs').glob('*/*/*.json')))
                with db:result=base.ingest_generated(q,db,q.read(run/'private/EVAL_PLAN146.json'),outputs,full_identity,judge,epoch['scientific_lock'],known)
                pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]
                q.write(q.ROOT/'READY.json',dict(result,new_pending=pending,epoch=time.time()))
                assert pending<=gate['new_Judge_cap'],'Scoring exceeds remaining budget; preserve queue without partial selection'
        else:assert operation['action'] in ('status','next','reserved')
        if operation['action']=='reserve':
            keys=operation['keys'];check_resume_reservation(db,keys,resume);assert 0<len(keys)<=50 and len(keys)==len(set(keys))
            gate=q.read(run/'private/CM_ADMISSION1.json');parent=Path(gate['parent'])
            own=q.read(run/'RESOURCE_LEDGER.json')['Judge_attempts']-q.read(run/'private/INHERITED_COST.json')['Judge_attempts']
            assert own+len(keys)<=gate['new_Judge_cap']
            parent_ledger=q.read(parent/'RESOURCE_LEDGER.json')
            merged=parent_ledger.get('cm_eval146_v1_Judge_merged',{}).get('increment',0)
            assert parent_ledger['Judge_attempts']+own-merged+len(keys)<=q.read(parent/'RUN_MANIFEST.json')['Judge_limit']
            protocol.project_budget(run)
        answer=base.coverage(db) if operation['action']=='ingest' else dispatch(a,q,db,operation,workers)
        result=base.coverage(db)
        if result['all_attempts_resolved'] and (workers==1 or all_workers_done(completed,workers)):
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(result,workers=workers,epoch=time.time()))
    if operation['action']=='ingest' and workers==4:
        # The existing controller waits here until every scorer has archived and exited its loop.
        while not all_workers_done(completed,workers):
            assert not list((completed/'workers').glob('*/SCORER_FAILURE.json'))
            protocol.project_budget(run);time.sleep(5)
        with (q.ROOT/'QUEUE.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);result=base.coverage(db)
            assert result['all_attempts_resolved']
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(result,workers=workers,epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))


if __name__=='__main__':main()
