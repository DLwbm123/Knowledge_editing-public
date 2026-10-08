"""Same isolated Astra medium protocol; reuse exact historical payloads only."""
import os
import sys
import json
import sqlite3
import fcntl
import time
from pathlib import Path
sys.path.insert(0, os.environ['RUN_ROOT']+'/private/tools')
import trajectory_queue as previous
a, q, RUN = previous.a, previous.q, previous.RUN
q.ROOT = RUN/'private/judge_scoped_astra_medium'; q.BATCH_LEDGER = 'Astra_scoped_batches'
EPOCH = 'MEDTRACE_TT_SCOPED_NO_REPLAY_ASTRA_MEDIUM_20261008_V1'


def parents():
    return previous.parents()


def initialize(db):
    lock = dict(epoch=EPOCH, model='gpt-6-astra', reasoning_effort='medium', protocol=q.PROTOCOL,
        prompt=q.PROMPT, batch_size=50, workers=4, attempts_per_payload=1,
        scientific_lock=q.digest(q.read(RUN/'private/SCOPED_LOCK.json')))
    path = q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():
        assert q.read(path) == lock
    else:
        q.write(path, lock)
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_folder TEXT,parent_batch TEXT)')


def ingest(db):
    if not (RUN/'private/GENERATION_COMPLETE.json').exists():
        return
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
        return
    parent = parents(); evidence = {}
    declared = q.read(RUN/'private/CONSUMERS.json'); assert len(declared) == q.read(RUN/'public/ADMISSION.json')['consumers']
    source = q.read(RUN/'private/GPU_SOURCE_VERSION.json')
    scientific_lock = q.digest(q.read(RUN/'private/SCOPED_LOCK.json'))
    for item in declared:
        path = Path(item['path']); d = q.read(path); b = d['binding']; row = b['input']
        assert row['query_id'] == item['query_id']
        if path.is_relative_to(RUN/'private/outputs'):
            assert b['phase']['execution'] == source and b['phase']['scoped_lock'] == scientific_lock
            assert b['arm'] == item['produced_arm'] and b['mode'] == item['mode']
        cid = q.digest(item)
        if db.execute('SELECT 1 FROM consumer WHERE id=?', (cid,)).fetchone():
            continue
        key = q.payload(db, row, b['judge_input'], d['R0'])
        if key in parent and not db.execute('SELECT 1 FROM inherited WHERE key=?', (key,)).fetchone():
            old, root = parent[key]
            assert json.loads(db.execute('SELECT binding FROM payload WHERE key=?', (key,)).fetchone()[0]) == json.loads(old['binding'])
            ek = str(root), old['batch']
            if ek not in evidence:
                saved = q.read(root/'evidence'/(old['batch']+'.json')); ev = saved['evidence']
                if old['status'] == 'FORMAT_VALID':
                    assert ev['status'] == 'FORMAT_VALID' and ev['actual_model'] == 'gpt-6-astra' and ev['reasoning_effort'] == 'medium'
                    assert ev['exit_code'] == 0 and not ev.get('errors') and not ev['tool_event_types']
                    assert all(ev['isolation_checks'].values()) and ev['input_binding'] == q.digest(saved['batch'])
                    evidence[ek] = saved['batch']['records'], {x['opaque_query_id']: x['is_correct'] for x in q.validate(saved['batch'], saved['response'])}
                else:
                    assert ev['status'] != 'FORMAT_VALID'; evidence[ek] = None
            if old['status'] == 'FORMAT_VALID':
                records, decisions = evidence[ek]; rec = json.loads(old['record'])
                assert rec in records and int(decisions[rec['opaque_query_id']]) == old['correct']
            db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',
                (old['status'], old['correct'], 'INHERITED_'+old['batch'], key))
            db.execute('INSERT INTO inherited VALUES (?,?,?)', (key, str(root), old['batch']))
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (cid, item['arm'], item['mode'], 146, 146, row.get('role', 'PANEL'), row['query_id'], str(path), q.digest(d), key))
    db.execute("INSERT INTO done VALUES ('generation')"); db.commit()
    q.write(q.ROOT/'READY.json', dict(consumers=db.execute('SELECT count(*) FROM consumer').fetchone()[0],
        payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],
        inherited=db.execute('SELECT count(*) FROM inherited').fetchone()[0], generation_complete=True))


if __name__ == '__main__':
    op = json.loads(os.environ.get('QUEUE_REQUEST', '{"action":"ingest"}')); q.ROOT.mkdir(parents=True, exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); db = q.connect(); initialize(db); q.ingest = lambda db: None
        if op['action'] == 'ingest':
            ingest(db); answer = q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status': 'GENERATING'}
        else:
            original = q.write
            def write(path, data):
                if Path(path).name == 'ALL_WORKERS_COMPLETE.json' and not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
                    return
                original(path, data)
            q.write = write; answer = a.request(db, op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json', dict(status='SCORING_COMPLETE', epoch=time.time()))
    print(json.dumps(answer, ensure_ascii=False))
