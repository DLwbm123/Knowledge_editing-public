"""One-attempt first-three diagnostic scoring after both trajectories/reloads freeze."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, os.environ["RUN_ROOT"] + "/private/tools")
import base_queue as parent
from panels import expected_ids
q, R, a = parent.q, parent.R, parent.a
protocol = q.read(R/'private/FROZEN_PROTOCOL.json')


def ingest(db):
    if not db.execute("SELECT 1 FROM done WHERE name='historical'").fetchone():
        for root in (protocol['reference_run'],protocol['K_reference_run'],protocol['M_reference_run']):
            old=parent.ro(Path(root)/'private/judge_purew146/queue.sqlite')
            for row in old.execute('SELECT * FROM payload'):
                assert row['status'] in ('FORMAT_VALID','MISSING')
                binding=json.loads(row['binding'])
                assert binding['judge']==dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
                assert q.digest(binding)==row['key']
                db.execute('INSERT OR IGNORE INTO payload VALUES (?,?,?,?,?,?)',tuple(row))
            old.close()
        db.execute("INSERT INTO done VALUES ('historical')");db.commit()
    if not (R/'private/GENERATION_COMPLETE.json').exists():return
    assert (R/'private/FINAL_CONSUMED.json').exists()
    assert all(s.get('ended_epoch') for s in q.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    tasks=q.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=q.read(R/'private/EVAL_LEDGER.json')
    for marker in sorted((R/'private/outputs').glob('*/*/*/*/COMPLETE.json')):
        key='panel:'+str(marker.relative_to(R))
        if db.execute('SELECT 1 FROM done WHERE name=?',(key,)).fetchone():continue
        receipt=q.read(marker);phase=receipt['phase'];assert phase['arm'] in ('M','F','S')
        allowed={q.digest(protocol)}
        if phase['arm']=='M':allowed.add(q.read(R/'private/M_PARENT_PROTOCOL_BINDING.json')['protocol'])
        assert phase['protocol'] in allowed
        files=[p for p in marker.parent.glob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
        assert len(files)==receipt['outputs'] and {p.stem for p in files}==set(expected_ids(tasks,ledger,phase['mode'],phase['owner'],phase['prefix']))
        for path in files:
            d=q.read(path);bb=d['binding'];assert bb['phase']==phase and bb['input']==ledger['queries'][bb['input']['query_id']]
            payload=q.payload(db,bb['input'],bb['judge_input'],d['R0'])
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),phase['arm'],phase['mode'],phase['prefix'],phase['owner'],
                'INSERTION' if phase['mode']=='insertion_R0' else 'PANEL',bb['input']['query_id'],str(path),q.digest(d),payload))
        db.execute('INSERT INTO done VALUES (?)',(key,));db.commit()
    assert all(db.execute('SELECT count(*) FROM consumer WHERE method=?',(arm,)).fetchone()[0]==161 for arm in ('M','F','S'))
    db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(coverage=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method'))))


if __name__=='__main__':
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();ingest(db);q.ingest=lambda db:None
        op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"next"}'));write=q.write
        q.write=lambda path,data:write(path,data) if Path(path)!=q.ROOT/'ALL_WORKERS_COMPLETE.json' or parent.terminal_allowed(db) else None
        try:answer=a.request(db,op)
        finally:q.write=write
        if parent.terminal_allowed(db):q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_TERMINAL',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
