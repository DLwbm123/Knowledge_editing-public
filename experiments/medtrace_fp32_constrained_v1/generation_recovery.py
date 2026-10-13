"""One user-authorized attempt for the40 missing edit ratings; original queue immutable."""
import os
import sys
import json
import fcntl
import sqlite3
import time
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_generation_recovery1'
q.BATCH_LEDGER='Astra_constrained_recovery1_batches'
OLD=RUN/'private/judge_generation_astra_medium'
EPOCH='MEDTRACE_CONSTRAINED_RECOVERY1_20261009_V1'


def initialize(db):
    receipt=q.ROOT/'AUTHORIZATION.json'
    if receipt.exists():
        assert q.read(receipt)['epoch']==EPOCH
        assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==40
        return
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==0
    old=sqlite3.connect('file:'+str(OLD/'queue.sqlite')+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
    rows=list(old.execute("SELECT * FROM payload WHERE status='MISSING' ORDER BY rowid"))
    assert len(rows)==40 and not old.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    search=q.read(RUN/'private/GEN_RECOVERY_HISTORICAL_SEARCH.json');assert search['unique_semantic_matches']==0
    for r in rows:
        j=json.loads(r['binding'])['judge'];assert (j['model'],j['reasoning_effort'],j['prompt'],j['protocol'])==('gpt-6-astra','medium',q.PROMPT,q.PROTOCOL)
        db.execute("INSERT INTO payload VALUES (?,?,?,'PENDING',NULL,NULL)",(r['key'],r['record'],r['binding']))
    keys={r['key'] for r in rows}
    for r in old.execute('SELECT * FROM consumer'):
        if r['payload_key'] in keys:db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',tuple(r))
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==160
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(receipt,dict(epoch=EPOCH,payloads=40,consumers=160,keys=sorted(keys),extra_attempts_per_payload=1,
        model='gpt-6-astra',reasoning_effort='medium',historical_queues_checked=search['queues_scanned'],historical_matches=0,
        original_queue_readonly=True,original_records_and_bindings_preserved=True,
        Judge_before=q.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts'],epoch_started=time.time()))


def report(db):
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    old=sqlite3.connect('file:'+str(OLD/'queue.sqlite')+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
    recovered={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')};assert len(recovered)==40
    assert all(r['status'] in ('FORMAT_VALID','MISSING') for r in recovered.values())
    rows=[]
    for r in old.execute("SELECT c.*,p.binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE c.folder='EDIT'"):
        s=recovered[r['payload_key']];assert r['status']=='MISSING' and s['binding']==r['binding']
        d=q.read(r['path']);rows.append(dict(path=r['path'],expert_order=d['expert_order'],index=d['index'],arm=d['arm'],
            correct=s['correct'] if s['status']=='FORMAT_VALID' else None,status=s['status'],payload=r['payload_key']))
    assert len(rows)==160
    base=[r for r in rows if r['arm']=='BASE'];assert len(base)==40
    panels={}
    for name,rs in [('NATIVE',[r for r in base if r['index']==0]),('FIT',[r for r in base if r['index']>0]),('ALL_EDIT',base)]:
        panels[name]=dict(queries=len(rs),correct=sum(r['correct']==1 for r in rs),wrong=sum(r['correct']==0 for r in rs),missing=sum(r['correct'] is None for r in rs))
    q.write(RUN/'private/EDIT_BASELINE_QUALIFICATION.json',dict(rows=rows,original_experts_preserved=True,source_only=True))
    attempts=q.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts'];before=q.read(q.ROOT/'AUTHORIZATION.json')['Judge_before'];assert attempts-before==40
    result=dict(status='COMPLETE',decision='EDIT_BASELINE_QUALIFIED' if all(r['correct'] is not None for r in base) else 'EDIT_BASELINE_STILL_INCOMPLETE',
        panels=panels,per_expert=[dict(expert_order=o,native_correct=next(r['correct'] for r in base if r['expert_order']==o and r['index']==0),
            FIT_correct=sum(r['correct']==1 for r in base if r['expert_order']==o and r['index']>0),FIT_missing=sum(r['correct'] is None for r in base if r['expert_order']==o and r['index']>0)) for o in range(1,9)],
        payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),new_Judge=40,cumulative_Judge=attempts,
        new_generations=0,new_GPU_hours=0,original_failures_preserved=40,new_answer_repairs=0)
    q.write(RUN/'public/GEN_RECOVERY_RESULTS.json',result)
    q.write(RUN/'private/GEN_RECOVERY_COMPLETE.json',dict(status='COMPLETE',decision=result['decision'],epoch=time.time()))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"status"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();initialize(db);q.ingest=lambda db:None
        if op['action']=='report':report(db);answer={'status':'REPORT_COMPLETE'}
        else:answer=a.request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
