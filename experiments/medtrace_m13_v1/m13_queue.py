"""One-attempt Astra queue with exact historical payload reuse."""
import fcntl
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
sys.path.insert(0, os.environ['RUN_ROOT'] + '/private/tools')
import astra_queue as a
from fine_queue import full_identity, selfcheck

q, RUN = a.q, a.RUN
q.ROOT = RUN / 'private/judge_m13_astra_medium'
q.BATCH_LEDGER = 'Astra_m13_batches'
MAX_JUDGE = 1158
CONSUMERS = 1158
EPOCH = "MEDTRACE_M13_20261012_V1"
PARENTS = ("PROTECTION_PARENT", "DAMAGE_PARENT", "GENERATION_PARENT", "FP32_BASELINE_PARENT",
           "OPTIMIZER160_PARENT", "SCOPE_PARENT", "ENDPOINT_PARENT", "MODULE_PARENT")


def ingest(db):
    lock = dict(epoch=EPOCH, model='gpt-6-astra', reasoning_effort='medium',
        protocol=q.PROTOCOL, prompt=q.PROMPT, batch_size=50, workers=4, attempts_per_payload=1,
        scientific_lock=q.digest(q.read(RUN / 'private/SCOPE_LOCK.json')),
        maximum_new_Judge=MAX_JUDGE, inheritance='EXACT_FULL_PAYLOAD_IDENTITY_ONLY', selfcheck=selfcheck())
    path = q.ROOT / 'EPOCH_MANIFEST.json'
    if path.exists():
        assert q.read(path) == lock
    else:
        q.write(path, lock)
    assert (RUN / 'private/MODULE_GENERATION_COMPLETE.json').exists()
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
        return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0] == 0
    env = q.read(RUN / 'private/LAUNCH_ENV.json')
    judge = {k: lock[k] for k in ('model', 'reasoning_effort', 'protocol', 'prompt')}
    recovery = q.read(Path(env['GENERATION_PARENT']) / 'private/judge_generation_recovery1/AUTHORIZATION.json')
    assert recovery['epoch'] == 'MEDTRACE_CONSTRAINED_RECOVERY1_20261009_V1' and recovery['payloads'] == 40
    allowed_recovery = set(recovery['keys'])
    known, sources = {}, []
    for key in PARENTS:
        for path in sorted((Path(env[key]) / 'private').glob('judge*/queue.sqlite')):
            with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as previous:
                count = 0
                for payload, binding, status, correct in previous.execute("SELECT key,binding,status,correct FROM payload WHERE status IN ('FORMAT_VALID','MISSING')"):
                    binding = json.loads(binding)
                    if binding['judge'] != judge:
                        continue
                    assert q.digest(binding) == payload
                    value = (status, correct)
                    if payload in known and known[payload] != value:
                        assert payload in allowed_recovery and key == 'GENERATION_PARENT'
                        assert path.parent.name == 'judge_generation_recovery1' and known[payload] == ('MISSING', None) and status == 'FORMAT_VALID'
                    known[payload] = value; count += 1
            sources.append(dict(path=str(path), accepted_or_permanent_missing=count))
    paths = sorted((RUN / 'private/outputs').glob('*/*/*.json')); assert len(paths) == CONSUMERS
    inherited = missing = 0
    for path in paths:
        data = q.read(path); row = data['binding']['input']
        if not data.get('alias_of'):
            assert data['lock'] == lock['scientific_lock']
        else:
            assert full_identity(q.read(data['alias_of']), judge) == full_identity(data, judge)
        key = q.payload(db, row, data['binding']['judge_input'], data['R0'])
        assert key == q.digest(full_identity(data, judge))
        if key in known:
            status, score = known[key]
            db.execute('UPDATE payload SET status=?,correct=? WHERE key=?', (status, score, key))
            inherited += status == 'FORMAT_VALID'; missing += status == 'MISSING'
        natural = data['arm'].startswith('NATURAL_')
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)', (q.digest(str(path)), data['arm'],
            'natural' if natural else 'forced_owner', 8 if natural else 1, data['expert_order'], data['role'],
            row['query_id'], str(path), q.digest(data), key))
    assert not db.execute("SELECT 1 FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE c.method IN ('C','NATURAL_C') AND p.status!='FORMAT_VALID'").fetchone()
    pending = db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]
    assert pending <= MAX_JUDGE
    db.execute("INSERT INTO done VALUES ('generation')"); db.commit()
    q.write(q.ROOT / 'REUSE_SOURCES.json', sources)
    q.write(q.ROOT / 'READY.json', dict(consumers=CONSUMERS,
        payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0], inherited_consumers=inherited,
        permanent_missing_consumers=missing, new_pending=pending, epoch=time.time()))


def request(db, operation):
    if operation['action'] == 'next':
        from module_recovery import expected_parent_failure
        admitted = (RUN / 'private/engineering_admission_recovery1/ADMISSION.json').exists()
        failures = [path for path in list((RUN / 'private').glob('FAILURE_module*.json')) + list((RUN / 'private').glob('FAILURE_m13*.json'))
                    if not (admitted and path.name.startswith('FAILURE_module_controller_')
                            and expected_parent_failure(q.read(path)))]
        if failures:
            return dict(state=dict(status='STOPPED', reason='OWN_STAGE_FAILED'), rows=[])
    if operation['action'] == 'reserve':
        prior = q.read(RUN / 'private/INHERITED_COST.json')['Judge_attempts']
        assert q.read(RUN / 'RESOURCE_LEDGER.json')['Judge_attempts'] + len(operation['keys']) - prior <= MAX_JUDGE
    return a.request(db, operation)


def main():
    operation = json.loads(os.environ.get('QUEUE_REQUEST', '{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True, exist_ok=True)
    with (q.ROOT / 'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = q.connect(); q.ingest = lambda db: None
        generated = (RUN / 'private/MODULE_GENERATION_COMPLETE.json').exists()
        if generated:
            ingest(db)
        else:
            assert operation['action'] in ('reserved', 'next', 'status'), 'Scoring input not ready'
        answer = q.read(q.ROOT / 'READY.json') if operation['action'] == 'ingest' else request(db, operation)
        if generated and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT / 'ALL_WORKERS_COMPLETE.json', dict(status='SCORING_COMPLETE', epoch=time.time()))
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    main()
