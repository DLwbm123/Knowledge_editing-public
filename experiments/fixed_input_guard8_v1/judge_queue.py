"""Output-only fixed8 scoring through the existing isolated, one-attempt Judge."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import time

R=Path(os.environ['RUN_ROOT'])
spec=importlib.util.spec_from_file_location('queue_base',R/'private/tools/queue_base.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.ROOT=R/'private/judge_fixed8';q.EFFORT='medium';q.BATCH_LEDGER='Fixed8_Judge_batches'
EPOCH='FIXED_INPUT_GUARD8_V1_20261016'


def ingest(db):
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert q.read(R/'private/GENERATION_COMPLETE.json')['status']=='PASS'
    assert all(q.read(R/'private'/('FINAL_CONSUMED_'+a+'.json'))['status']=='PASS' for a in ('U','P'))
    assert not any(not s.get('ended_epoch') for s in q.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    lock=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,batch_size=50)
    q.write(q.ROOT/'EPOCH_MANIFEST.json',lock)
    ledger=q.read(R/'private/EVAL_LEDGER.json')['queries']
    for arm in ('U','P'):
        for marker in sorted((R/'private/outputs'/arm).rglob('COMPLETE.json')):
            m=q.read(marker);phase=m['phase']
            if phase['mode']=='reload':continue
            paths=[p for p in marker.parent.glob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
            assert len(paths)==m['outputs']
            for p in paths:
                d=q.read(p);b=d['binding'];prior=b['phase']
                expected=dict(phase)
                if prior.get('code')!=phase.get('code'):
                    assert arm=='U' and phase['mode']=='base' and phase['prefix']==0
                    expected['code']=q.read(R/'private/RECOVERY1.json')['previous_commit']
                assert prior==expected and b['input']==ledger[p.stem]
                key=q.payload(db,b['input'],b['judge_input'],d['R0'])
                db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(p)),arm,phase['mode'],phase['prefix'],phase['owner'],'PANEL',p.stem,str(p),q.digest(d),key))
        assert db.execute('SELECT count(*) FROM consumer WHERE method=?',(arm,)).fetchone()[0]==382
    old_path=R.parent.parent/'purew-odds-hinge-first1-v1-20261016/run/private/judge_purew146/queue.sqlite'
    old=sqlite3.connect('file:'+str(old_path)+'?mode=ro',uri=True);old.row_factory=sqlite3.Row;reused=0
    for row in db.execute('SELECT key,binding FROM payload').fetchall():
        prior=old.execute('SELECT * FROM payload WHERE key=?',(row['key'],)).fetchone()
        if prior and prior['status'] in ('FORMAT_VALID','MISSING'):
            assert json.loads(prior['binding'])==json.loads(row['binding'])
            db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',(prior['status'],prior['correct'],'HISTORICAL_EXACT:'+str(prior['batch']),row['key']));reused+=1
    old.close();db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'INGEST.json',dict(consumers=764,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],historical_exact_reused=reused))


def status(db):
    return dict(status='STOPPED' if (R/'STOP').exists() else 'GENERATION_COMPLETE',payloads=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),resource_Judge_attempts=q.read(R/'RESOURCE_LEDGER.json')['Judge_attempts'])


def request(db,op):
    q.reconcile(db);worker=op.get('worker',0);assert type(worker)is int and 0<=worker<4
    action=op['action']
    if action=='next':
        rows=[] if status(db)['status']=='STOPPED' else [dict(key=r['key'],record=json.loads(r['record'])) for r in db.execute("SELECT key,record FROM payload WHERE status='PENDING' AND ((rowid-1)/50)%4=? ORDER BY rowid LIMIT 50",(worker,))]
        return dict(state=status(db),rows=rows)
    if action=='reserve':
        keys=op['keys'];assert keys and len(keys)==len(set(keys))<=50 and not (R/'STOP').exists()
        for k in keys:
            row=db.execute('SELECT rowid,status FROM payload WHERE key=?',(k,)).fetchone()
            assert row['status']=='PENDING' and ((row['rowid']-1)//50)%4==worker
        charges={a:sum(bool(db.execute('SELECT 1 FROM consumer WHERE method=? AND payload_key=?',(a,k)).fetchone()) for k in keys) for a in ('U','P')}
        budgets={a:q.read(R/'private/fixed8'/a/'evaluation-budget.json') for a in ('U','P')}
        assert all(budgets[a]['counts']['new_Judge']+charges[a]<=512 for a in charges)
        with q.resources() as ledger:
            assert ledger['Judge_attempts']+len(keys)<=1024
            attempts=ledger.setdefault(q.BATCH_LEDGER,[])
            consecutive=0
            for a in reversed(attempts):
                if a['status']=='FAILED_NO_RETRY' and a.get('transport_failure'):consecutive+=1
                else:break
            assert consecutive<3,'Three transport failures: no new requests'
            assert not any(a['id']==op['batch_id'] or set(a['keys'])&set(keys) for a in attempts)
            ledger['Judge_attempts']+=len(keys)
            attempts.append(dict(id=op['batch_id'],keys=keys,status='RESERVED',worker=worker,started_epoch=time.time(),model='gpt-6-astra',reasoning_effort='medium'))
        for a in charges:
            budgets[a]['counts']['new_Judge']+=charges[a];q.write(R/'private/fixed8'/a/'evaluation-budget.json',budgets[a])
        q.reconcile(db);return dict(status='RESERVED',items=len(keys))
    if action=='reserved':return [a for a in q.read(R/'RESOURCE_LEDGER.json').get(q.BATCH_LEDGER,[]) if a['status']=='RESERVED' and a['worker']==worker]
    if action=='publish':
        a=next(a for a in q.read(R/'RESOURCE_LEDGER.json')[q.BATCH_LEDGER] if a['id']==op['batch_id']);assert a['worker']==worker
    return q.request(db,op)


q.status=status
if __name__=='__main__':
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();ingest(db)
        answer=request(db,json.loads(os.environ.get('QUEUE_REQUEST','{"action":"status"}')))
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_TERMINAL',epoch=time.time()))
    print(json.dumps(answer))
