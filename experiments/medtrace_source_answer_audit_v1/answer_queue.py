"""Existing isolated Astra protocol for the new, fixed Base-FIT answer queue."""
import os
import sys
import json
import fcntl
import time
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q, RUN = a.q, a.RUN
q.ROOT = RUN/'private/judge_answer_astra_medium'
q.BATCH_LEDGER = 'Astra_source_answer_batches'
EPOCH = 'MEDTRACE_SOURCE_ANSWER_ASTRA_MEDIUM_20261009_V1'


def initialize():
    lock = dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,
        prompt=q.PROMPT,batch_size=50,workers=4,attempts_per_payload=1,
        scientific_lock=q.digest(q.read(RUN/'private/ANSWER_LOCK.json')))
    path = q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)


def ingest(db):
    if not (RUN/'private/GENERATION_COMPLETE.json').exists() or db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    rows=q.read(RUN/'private/ANSWER_ROWS.json');assert len(rows)==192
    for row in rows:
        path=RUN/'private/outputs'/(q.digest(row['query_id'])+'.json');d=q.read(path);b=d['binding']
        assert d['unedited_Base'] and b['input']['role']=='REPLAY_FIT' and b['input']['query_id']==row['query_id']
        assert b['execution']==q.read(RUN/'private/GPU_SOURCE_VERSION.json') and b['lock']==q.digest(q.read(RUN/'private/ANSWER_LOCK.json'))
        key=q.payload(db,b['input'],b['judge_input'],d['R0'])
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (q.digest([row['query_id'],str(path)]),'BASE','source_FIT',0,0,row['audit_role'],row['query_id'],str(path),q.digest(d),key))
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==192
    q.write(q.ROOT/'READY.json',dict(consumers=192,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited=0))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();initialize();q.ingest=lambda db:None
        ingest(db)
        answer=(q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status':'GENERATING'}) if op['action']=='ingest' else a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
