"""One authorized extra attempt for the 200 missing scores; original DB read-only."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
RUN = Path(os.environ['RUN_ROOT'])
ORIGINAL = RUN.parent/'run'
import astra_queue as a
q = a.q
q.ROOT = RUN/'private/judge_interim20_recovery1'
q.EFFORT = 'medium'
q.BATCH_LEDGER = 'Astra_interim20_recovery1_batches'


def seed(db, old, groups):
    assert not db.execute('SELECT 1 FROM payload').fetchone()
    rows = {r['key']: r for r in old.execute("SELECT * FROM payload WHERE status='MISSING'")}
    keys = [key for group in groups for key in group]
    assert len(groups) == 4 and all(len(g) == 50 for g in groups)
    assert len(keys) == len(set(keys)) == len(rows) == 200 and set(keys) == set(rows)
    assert not old.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    for group in groups:
        assert len({rows[key]['batch'] for key in group}) == 1
    for key in keys:
        row = rows[key]
        binding = json.loads(row['binding']); judge = binding['judge']
        assert q.digest(binding) == key
        assert (judge['model'], judge['reasoning_effort'], judge['prompt'], judge['protocol']) == (
            'gpt-6-astra', 'medium', q.PROMPT, q.PROTOCOL)
        assert row['correct'] is None
        db.execute("INSERT INTO payload VALUES (?,?,?,'PENDING',NULL,NULL)",
                   (key, row['record'], row['binding']))
    for row in old.execute('SELECT * FROM consumer ORDER BY rowid'):
        if row['payload_key'] in rows:
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)', tuple(row))
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0] == 303
    db.execute("INSERT INTO done VALUES ('generation')")


def main():
    gate = q.read(RUN/'private/RECOVERY_ADMISSION1.json')
    assert gate['status'] == 'PASS' and gate['user_authorized'] and gate['extra_attempts'] == 200
    assert gate['parent_training_modified'] is False
    q.ROOT.mkdir(parents=True, exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = q.connect()
        if not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():
            with sqlite3.connect('file:'+str(ORIGINAL/'private/judge_interim20_astra_medium/queue.sqlite')+'?mode=ro', uri=True) as old:
                old.row_factory = sqlite3.Row
                with db: seed(db, old, gate['original_failed_groups'])
            q.write(q.ROOT/'AUTHORIZATION.json', gate)
            q.write(q.ROOT/'EPOCH_MANIFEST.json', dict(epoch='MEDTRACE_NATIVE146_INTERIM20_RECOVERY1_20261015_V1',
                model='gpt-6-astra', reasoning_effort='medium', prompt=q.PROMPT, protocol=q.PROTOCOL,
                attempts_per_payload=1, original_missing_preserved=True, workers=1, batch_size=50))
        else:
            assert q.read(q.ROOT/'AUTHORIZATION.json') == gate
        assert db.execute('SELECT count(*) FROM payload').fetchone()[0] == 200
        q.ingest = lambda _: None
        answer = q.request(db, json.loads(os.environ['QUEUE_REQUEST']))
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json', dict(status='RESOLVED', state=q.status(db), epoch=time.time()))
        print(json.dumps(answer, ensure_ascii=False))


if __name__ == '__main__': main()
