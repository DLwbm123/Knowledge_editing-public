"""Existing medium one-attempt queue with exact baseline reuse and durable streaming panels."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
from contract import ARM
from pause100 import planned_controller_failure

q,R=a.q,a.RUN
q.ROOT=R/'private/judge_purew146'
q.EFFORT='medium'
q.BATCH_LEDGER='Astra_purew146_batches'
P=Path(q.read(R/'private/FROZEN_PROTOCOL.json')['common_parent'])


def terminal_write_allowed(path, generation_done):
    return Path(path)!=q.ROOT/'ALL_WORKERS_COMPLETE.json' or generation_done


def ro(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    return db


def ingest(db):
    if not (R/'private/BASELINE_REUSE_ADMISSION.json').exists():
        return
    if not db.execute("SELECT 1 FROM done WHERE name='baselines'").fetchone():
        old=ro(P/'private/judge_common146_medium/queue.sqlite')
        recovered={x['key']:x for x in ro(P/'private/judge_common146_medium_recovery1/queue.sqlite').execute('SELECT * FROM payload')}
        selected=list(old.execute("SELECT * FROM consumer WHERE method IN ('TT88_W0_RETRO146','balancedit')"))
        assert len(selected)==11444
        for row in selected:
            key=row['payload_key']
            prior=recovered.get(key) or old.execute('SELECT * FROM payload WHERE key=?',(key,)).fetchone()
            assert prior['status']=='FORMAT_VALID'
            full=json.loads(prior['binding'])
            assert full['judge']==dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
            db.execute('INSERT OR IGNORE INTO payload VALUES (?,?,?,?,?,?)',tuple(prior))
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',tuple(row))
        db.execute("INSERT INTO done VALUES ('baselines')")
        db.commit()
    for marker in sorted((R/'private/outputs').glob('*/*/*/COMPLETE.json')):
        name='panel:'+str(marker.relative_to(R))
        if db.execute('SELECT 1 FROM done WHERE name=?',(name,)).fetchone():
            continue
        receipt=q.read(marker)
        assert receipt['phase']['protocol']==q.digest(q.read(R/'private/FROZEN_PROTOCOL.json'))
        files=[p for p in marker.parent.glob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
        assert len(files)==receipt['outputs']
        for path in files:
            d=q.read(path);b=d['binding'];phase=b['phase']
            assert phase==receipt['phase'] and phase['arm']==ARM
            key=q.payload(db,b['input'],b['judge_input'],d['R0'])
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (q.digest(str(path)),ARM,phase['mode'],phase['prefix'],phase['owner'],
                        'INSERTION' if phase['mode']=='insertion_R0' else 'PANEL',
                        b['input']['query_id'],str(path),q.digest(d),key))
        db.execute('INSERT INTO done VALUES (?)',(name,));db.commit()
    if (R/'private/GENERATION_COMPLETE.json').exists():
        assert db.execute('SELECT count(*) FROM consumer WHERE method=?',(ARM,)).fetchone()[0]==5722
        db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumer_counts=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method')),
                statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))))


if __name__=='__main__':
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db=q.connect()
        receipt=q.ROOT/'EPOCH_MANIFEST.json'
        if not receipt.exists():
            q.write(receipt,dict(epoch='PUREW_ORIGINAL146_V1_20261012',model='gpt-6-astra',reasoning_effort='medium',
                protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
        ingest(db);q.ingest=lambda db:None
        op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"next"}'))
        failures=[p for p in (R/'private/failures').glob('*.json') if not planned_controller_failure(p,R)]
        if failures and op['action']=='next':
            answer=dict(state=dict(status='STOPPED',failure='Retained GPU/CPU failure requires review'),rows=[])
        else:
            write=q.write
            generation_done=bool(db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone())
            # The inherited batch publisher assumes generation has already ended.
            q.write=lambda path,data:write(path,data) if terminal_write_allowed(path,generation_done) else None
            try:
                answer=a.request(db,op)
            finally:
                q.write=write
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_TERMINAL',epoch=time.time(),statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))))
    print(json.dumps(answer,ensure_ascii=False))
