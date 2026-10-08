"""One explicitly authorized retry of the302 missing inputs; old queue immutable."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import time

RUN = Path(os.environ['RUN_ROOT'])
import sys
sys.path.insert(0, str(RUN/'private/tools'))
import replay_queue as original
q, a = original.q, original.a
q.ROOT = RUN/'private/judge_replay_astra_medium_recovery1'
q.EFFORT = 'medium'
q.BATCH_LEDGER = 'Astra_replay_recovery_batches'
EPOCH = 'MEDTRACE_SOURCE_REPLAY_ASTRA_MEDIUM_RECOVERY1_20261008'


def initialize(db):
    receipt = q.ROOT/'AUTHORIZATION.json'
    if receipt.exists():
        lock = q.read(receipt)
        assert lock['epoch'] == EPOCH and lock['extra_attempts_per_payload'] == 1
        assert db.execute('SELECT count(*) FROM payload').fetchone()[0] == 302
        return
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0] == 0
    old = sqlite3.connect('file:'+str(RUN/'private/judge_replay_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    old.row_factory = sqlite3.Row
    rows = old.execute("SELECT * FROM payload WHERE status='MISSING' ORDER BY rowid").fetchall()
    assert len(rows) == 302 and not old.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    keys = {r['key'] for r in rows}
    for row in rows:
        judge = json.loads(row['binding'])['judge']
        assert (judge['model'],judge['reasoning_effort'],judge['prompt']) == ('gpt-6-astra','medium',q.PROMPT)
        db.execute('INSERT INTO payload VALUES (?,?,?,\'PENDING\',NULL,NULL)',(row['key'],row['record'],row['binding']))
    for c in old.execute('SELECT * FROM consumer'):
        if c['payload_key'] in keys:
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',tuple(c))
    db.execute('INSERT INTO done VALUES (\'generation\')')
    db.commit()
    ledger = q.read(RUN/'RESOURCE_LEDGER.json')
    assert ledger['Judge_attempts'] + 302 <= q.read(RUN/'RUN_MANIFEST.json')['Judge_limit']
    q.write(receipt,dict(epoch=EPOCH,authorization='User explicitly requested completing the302 missing Astra scores',
        payloads=302,keys=[r['key'] for r in rows],original_failed_batches=sorted({r['batch'] for r in rows}),
        extra_attempts_per_payload=1,original_queue_readonly=True,original_input_order_and_records_preserved=True,
        model='gpt-6-astra',reasoning_effort='medium',Judge_before=ledger['Judge_attempts'],created_epoch=time.time()))


if __name__ == '__main__':
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db=q.connect()
        initialize(db)
        q.ingest=lambda db: None
        operation=json.loads(os.environ['QUEUE_REQUEST'])
        answer=a.request(db,operation)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(epoch=time.time(),state=q.status(db)))
        print(json.dumps(answer,ensure_ascii=False))
