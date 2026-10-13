"""Isolated one-attempt queue for all fixed146 output consumers."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
import astra_queue as a
from fine_queue import full_identity, selfcheck
import full146_contract as contract

q, RUN = a.q, a.RUN
q.ROOT = RUN/'private/judge_full146_astra_medium'
q.BATCH_LEDGER = 'Astra_full146_batches'
EPOCH = 'MEDTRACE_FULL_REFINE146_20261013_V1'


def ingest(db):
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
        return
    assert (RUN/'private/MODULE_GENERATION_COMPLETE.json').exists()
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0] == 0
    judge = dict(model='gpt-6-astra', reasoning_effort='medium', protocol=q.PROTOCOL, prompt=q.PROMPT)
    lock = dict(epoch=EPOCH, **judge, workers=4, batch_size=50, attempts_per_payload=1,
        maximum_new_Judge=17373, scientific_lock=q.digest(q.read(RUN/'private/SCOPE_LOCK.json')),
        selfcheck=selfcheck())
    receipt = q.ROOT/'EPOCH_MANIFEST.json'
    if receipt.exists():
        assert q.read(receipt) == lock
    else:
        q.write(receipt, lock)
    env = q.read(RUN/'private/LAUNCH_ENV.json')
    known = {}
    # The completed upstream queue already resolves authorized recoveries. Do not
    # silently choose between conflicting old valid and permanent-missing scores.
    for root in (Path(env['UPSTREAM_ROOT']), Path(env['MODULE_PARENT'])):
        for path in sorted((root/'private').glob('judge*/queue.sqlite')):
            with sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True) as old:
                for key, binding, status, score in old.execute("SELECT key,binding,status,correct FROM payload WHERE status IN ('FORMAT_VALID','MISSING')"):
                    binding = json.loads(binding)
                    if binding.get('judge') != judge or 'runtime' not in binding:
                        continue
                    assert q.digest(binding) == key
                    if key in known:
                        assert known[key] == (status, score), 'Conflicting inherited payload; no automatic rescore'
                    known[key] = status, score
    expected = {contract.consumer_id(v) for v in q.read(RUN/'private/EVAL_PLAN146.json')}
    paths = sorted((RUN/'private/outputs').glob('*/*/*.json'))
    assert len(paths) == len(expected) == 17373
    actual = set()
    for path in paths:
        data = q.read(path)
        assert data['consumer_id'] not in actual and data['consumer_id'] in expected
        actual.add(data['consumer_id'])
        assert data['lock'] == lock['scientific_lock']
        key = q.payload(db, data['binding']['input'], data['binding']['judge_input'], data['R0'])
        assert key == q.digest(full_identity(data, judge))
        if key in known:
            status, score = known[key]
            db.execute('UPDATE payload SET status=?,correct=? WHERE key=?', (status, score, key))
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (q.digest(data['consumer_id']), data['method'], data['mode'], data['prefix'],
             data['eval_owner'], data['role'], data['binding']['input']['query_id'],
             str(path), q.digest(data), key))
    assert actual == expected
    pending = db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]
    assert pending <= 17373
    db.execute("INSERT INTO done VALUES ('generation')"); db.commit()
    q.write(q.ROOT/'READY.json', dict(consumers=17373, new_pending=pending,
        payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0], epoch=time.time()))


def request(db, operation):
    if operation['action'] in ('next', 'reserve'):
        if list((RUN/'private').glob('FAILURE_full146*')):
            assert operation['action'] != 'reserve'
            return dict(state=dict(status='STOPPED', reason='OWN_STAGE_FAILED'), rows=[])
    if operation['action'] == 'reserve':
        prior = q.read(RUN/'private/INHERITED_COST.json')['Judge_attempts']
        used = q.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']
        assert used+len(operation['keys'])-prior <= 17373
    return a.request(db, operation)


def main():
    operation = json.loads(os.environ.get('QUEUE_REQUEST', '{"action":"ingest"}'))
    q.ROOT.mkdir(exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = q.connect(); q.ingest = lambda _: None
        generated = (RUN/'private/MODULE_GENERATION_COMPLETE.json').exists()
        if generated:
            ingest(db)
        else:
            assert operation['action'] in ('reserved', 'next', 'status')
        answer = q.read(q.ROOT/'READY.json') if operation['action'] == 'ingest' else request(db, operation)
        if generated and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json', dict(status='SCORING_COMPLETE', epoch=time.time()))
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == '__main__':
    main()
