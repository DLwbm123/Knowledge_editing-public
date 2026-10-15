"""New M13-146 queue: original sealed scorer and exact declared consumer coverage."""
import fcntl
import collections
import json
import os
from pathlib import Path
import sqlite3
import time
import contract

EPOCH='MEDTRACE_M13_ORIGINAL146_20261014_V1'
MAX_JUDGE=contract.CAPS['Judge_attempts']


def known_scores(q,paths,judge):
    known={}
    for path in paths:
        with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as old:
            for key,binding,status,score in old.execute("SELECT key,binding,status,correct FROM payload WHERE status IN ('FORMAT_VALID','MISSING')"):
                binding=json.loads(binding)
                if binding.get('judge')!=judge or 'runtime' not in binding:continue
                assert q.digest(binding)==key
                assert (status=='MISSING' and score is None) or (status=='FORMAT_VALID' and score in (0,1))
                value=status,score
                assert key not in known or known[key]==value,'Conflicting history: declare a resolved authorized snapshot'
                known[key]=value
    return known


def ingest_generated(q,db,plan,outputs,identity,judge,scope_lock,known):
    expected={contract.original.consumer_id(row):row for row in plan}
    assert len(expected)==len(plan)==MAX_JUDGE
    assert collections.Counter(row['method'] for row in plan)==dict(M1=7494,M3=7494,M13=7494,BASE=2385)
    assert not any(row['query'].get('unfilled_source_slot') for row in plan)
    assert not db.execute('SELECT 1 FROM consumer LIMIT 1').fetchone()
    actual=set()
    for path,data in outputs:
        cid=data['consumer_id'];assert cid in expected and cid not in actual
        row=expected[cid];actual.add(cid)
        assert all(data[k]==row[k] for k in ('method','mode','prefix','owner'))
        assert data['binding']['input']['query_id']==row['query']['query_id']
        assert all(data['binding']['input'].get(k)==row['query'][k]
            for k in ('question','reference','dataset','image_path') if k in row['query'])
        assert data['lock']==scope_lock
        full=identity(data,judge)
        if data.get('alias_of'):
            assert identity(q.read(data['alias_of']),judge)==full
        key=q.payload(db,data['binding']['input'],data['binding']['judge_input'],data['R0'])
        assert key==q.digest(full)
        if key in known:
            status,score=known[key]
            db.execute('UPDATE payload SET status=?,correct=? WHERE key=?',(status,score,key))
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (q.digest(cid),data['method'],data['mode'],data['prefix'],data['owner'],data['role'],
             row['query']['query_id'],str(path),q.digest(data),key))
    assert actual==set(expected),'Incomplete generated coverage'
    db.execute("INSERT INTO done VALUES ('generation')")
    return coverage(db)


def coverage(db):
    counts=dict(db.execute('SELECT p.status,count(*) FROM consumer c JOIN payload p ON c.payload_key=p.key GROUP BY p.status'))
    consumers=db.execute('SELECT count(*) FROM consumer').fetchone()[0]
    return dict(consumers=consumers,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],
        consumer_status=counts,all_attempts_resolved=consumers==MAX_JUDGE and not any(counts.get(k,0) for k in ('PENDING','RESERVED')),
        all_scores_valid=consumers==MAX_JUDGE and counts.get('FORMAT_VALID',0)==MAX_JUDGE)


def request(a,q,run,db,operation):
    action=operation['action']
    if action in ('next','reserve') and list((run/'private').glob('FAILURE_m13_146*.json')):
        assert action!='reserve','Own stage failed; no new Judge attempt'
        return dict(state=dict(status='STOPPED',reason='OWN_STAGE_FAILED'),rows=[])
    if action=='reserve':
        keys=operation['keys'];assert keys and len(keys)==len(set(keys))<=50
        for key in keys:
            row=db.execute('SELECT status FROM payload WHERE key=?',(key,)).fetchone()
            assert row and row[0]=='PENDING','No repeated or permanent-missing attempt'
        prior=q.read(run/'private/INHERITED_COST.json')['Judge_attempts']
        assert q.read(run/'RESOURCE_LEDGER.json')['Judge_attempts']+len(keys)-prior<=MAX_JUDGE
    return a.request(db,operation)


def main():
    import astra_queue as a
    from fine_queue import full_identity,selfcheck
    q,run=a.q,a.RUN
    q.ROOT=run/'private/judge_m13_146_astra_medium';q.BATCH_LEDGER='Astra_m13_146_batches'
    operation=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"status"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda _:None
        generated=(run/'private/M13_146_GENERATION_COMPLETE.json').exists()
        if generated:
            assert q.read(run/'private/M13_146_RUNTIME_ADMISSION.json')['status']=='PASS'
            assert q.read(run/'private/SOURCE_ADMISSION146.json')['status']=='PASS'
            judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
            epoch=dict(epoch=EPOCH,**judge,scientific_lock=q.digest(q.read(run/'private/SCOPE_LOCK.json')),
                workers=4,batch_size=50,attempts_per_payload=1,maximum_new_Judge=MAX_JUDGE,selfcheck=selfcheck())
            receipt=q.ROOT/'EPOCH_MANIFEST.json'
            if receipt.exists():assert q.read(receipt)==epoch
            else:q.write(receipt,epoch)
            if not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
                paths=q.read(run/'private/M13_146_REUSE_QUEUES.json')
                known=known_scores(q,paths,judge)
                outputs=((p,q.read(p)) for p in sorted((run/'private/outputs').glob('*/*/*.json')))
                with db:
                    result=ingest_generated(q,db,q.read(run/'private/EVAL_PLAN146.json'),outputs,
                        full_identity,judge,epoch['scientific_lock'],known)
                q.write(q.ROOT/'READY.json',dict(result,epoch=time.time()))
        else:assert operation['action'] in ('status','next','reserved'),'Generation not admitted'
        answer=coverage(db) if operation['action']=='ingest' else request(a,q,run,db,operation)
        result=coverage(db)
        if generated and result['all_attempts_resolved']:
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(result,epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))


if __name__=='__main__':main()
