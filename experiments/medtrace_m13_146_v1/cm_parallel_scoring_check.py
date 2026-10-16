"""Exercise the actual four-worker queue without GPU calls or Judge reservations."""
import json
import os
from pathlib import Path
import tempfile
from contextlib import contextmanager
import cm_queue146 as cm
import astra_queue as a


def main():
    q=a.q;run=a.RUN
    originals={name:getattr(q,name) for name in ('ROOT','read','write','reconcile','status','request','resources')}
    try:
        with tempfile.TemporaryDirectory(prefix='e.',dir=run/'private') as tmp:
            q.ROOT=Path(tmp);db=q.connect()
            with db:
                db.executemany('INSERT INTO payload(key,record,binding,status) VALUES(?,?,?,?)',[(str(i),json.dumps({'id':i}),'{}','PENDING') for i in range(1,454)])
            ledger=dict(Judge_attempts=0,Astra_medium_batches=[])
            q.BATCH_LEDGER='Astra_medium_batches'
            q.read=lambda path:dict(Judge_limit=10000) if Path(path).name=='RUN_MANIFEST.json' else ledger
            q.reconcile=lambda db:None
            q.status=lambda db:dict(status='GENERATION_COMPLETE')
            def fake_request(db,op):
                if op['action']=='reserve':
                    ledger['Astra_medium_batches'].append(dict(id=op['batch_id'],keys=op['keys']))
                    db.executemany("UPDATE payload SET status='RESERVED' WHERE key=?",[(k,) for k in op['keys']]);db.commit()
                    return dict(status='RESERVED')
                if op['action']=='reserved':return ledger['Astra_medium_batches']
                if op['action']=='publish':return dict(status='FORMAT_VALID')
                raise AssertionError(op)
            @contextmanager
            def resources():yield ledger
            q.request=fake_request;q.resources=resources
            seen=set();counts=[]
            for worker in range(4):
                own=[]
                while True:
                    rows=cm.dispatch(a,q,db,dict(action='next',worker=worker),4)['rows']
                    if not rows:break
                    keys=[r['key'] for r in rows]
                    assert not seen.intersection(keys)
                    try:cm.dispatch(a,q,db,dict(action='reserve',worker=(worker+1)%4,keys=keys,batch_id='wrong'),4)
                    except AssertionError:pass
                    else:raise AssertionError('Cross-worker reservation accepted')
                    bid=f'{worker}-{len(own)}'
                    cm.dispatch(a,q,db,dict(action='reserve',worker=worker,keys=keys,batch_id=bid),4)
                    seen.update(keys);own.extend(keys)
                assert all(b['worker']==worker for b in cm.dispatch(a,q,db,dict(action='reserved',worker=worker),4))
                counts.append(len(own))
            assert len(seen)==453 and counts==[150,103,100,100]
            with db:db.execute("UPDATE payload SET status='MISSING'")
            assert all(not cm.dispatch(a,q,db,dict(action='next',worker=w),4)['rows'] for w in range(4))
            try:cm.dispatch(a,q,db,dict(action='publish',worker=1,batch_id='0-0'),4)
            except AssertionError:pass
            else:raise AssertionError('Cross-worker publish accepted')
            cm.dispatch(a,q,db,dict(action='publish',worker=0,batch_id='0-0'),4)
            assert not (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
            for w in range(3):
                p=q.ROOT/f'workers/{w}/SCORER_DONE.json';p.parent.mkdir(parents=True);p.write_text('{}')
            assert not cm.all_workers_done(q.ROOT,4)
            p=q.ROOT/'workers/3/SCORER_DONE.json';p.parent.mkdir();p.write_text('{}')
            assert cm.all_workers_done(q.ROOT,4)
            db.close()
    finally:
        for k,v in originals.items():setattr(q,k,v)
    result=dict(status='PASS',synthetic_payloads=453,partition_sizes=counts,disjoint_complete=True,cross_worker_reserve_publish_rejected=True,permanent_missing_not_retried=True,four_done_required=True,production_Judge_attempts=0)
    (run/'private/CM_PARALLEL_SCORING_CPU_CHECK1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
