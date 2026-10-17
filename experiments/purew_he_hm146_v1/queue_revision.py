"""One-attempt first-three diagnostic scoring after both trajectories/reloads freeze."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, os.environ["RUN_ROOT"] + "/private/tools")
import astra_queue as a
from panels import expected_ids
q,R=a.q,a.RUN
q.ROOT=R/'private/judge_purew146'
q.EFFORT='medium';q.BATCH_LEDGER='Astra_purew146_batches'
protocol = q.read(R/'private/FROZEN_PROTOCOL.json')
original_status=q.status

def terminal_allowed(db):
    return bool(db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone()) and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()

def status(db):
    state=original_status(db)
    if terminal_allowed(db):state['status']='GENERATION_COMPLETE'
    elif list((R/'private/failures').glob('*.json')):state.update(status='STOPPED',failure='Retain failure; no replay')
    elif not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():state['status']='GENERATING'
    return state
q.status=status


def ensure_manifest():
    from configuration import VERSION
    expected=dict(epoch=VERSION,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50)
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==expected
    else:q.write(path,expected)


def import_history(db):
    if db.execute("SELECT 1 FROM done WHERE name='historical_cache'").fetchone():return
    for root in protocol['cache_roots']:
        old=sqlite3.connect('file:'+str(Path(root)/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
        try:
            for row in old.execute('SELECT * FROM payload'):
                assert row['status'] in ('FORMAT_VALID','MISSING')
                binding=json.loads(row['binding'])
                assert binding['judge']==dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
                assert q.digest(binding)==row['key']
                db.execute('INSERT OR IGNORE INTO payload VALUES (?,?,?,?,?,?)',tuple(row))
        finally:old.close()
    db.execute("INSERT INTO done VALUES ('historical_cache')");db.commit()


def ingest(db):
    import_history(db)
    if not (R/'private/GENERATION_COMPLETE.json').exists():return
    assert (R/'private/FINAL_CONSUMED.json').exists()
    assert all(s.get('ended_epoch') for s in q.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    tasks=q.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=q.read(R/'private/EVAL_LEDGER.json')
    for marker in sorted((R/'private/outputs').glob('*/*/*/*/COMPLETE.json')):
        key='panel:'+str(marker.relative_to(R))
        if db.execute('SELECT 1 FROM done WHERE name=?',(key,)).fetchone():continue
        receipt=q.read(marker);phase=receipt['phase'];assert phase['arm'] in ('HM','HE')
        allowed={q.digest(protocol)}
        assert phase['protocol'] in allowed and phase['code']==q.read(R/'private/GPU_SOURCE_VERSION.json')['commit']
        files=[p for p in marker.parent.glob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
        assert len(files)==receipt['outputs'] and {p.stem for p in files}==set(expected_ids(tasks,ledger,phase['mode'],phase['owner'],phase['prefix']))
        for path in files:
            d=q.read(path);bb=d['binding'];assert bb['phase']==phase and bb['input']==ledger['queries'][bb['input']['query_id']]
            payload=q.payload(db,bb['input'],bb['judge_input'],d['R0'])
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),phase['arm'],phase['mode'],phase['prefix'],phase['owner'],
                'INSERTION' if phase['mode']=='insertion_R0' else 'PANEL',bb['input']['query_id'],str(path),q.digest(d),payload))
        db.execute('INSERT INTO done VALUES (?)',(key,));db.commit()
    terminals=q.read(R/'private/ARM_TERMINALS.json');assert set(terminals)==set(('HM','HE'))
    assert all(db.execute('SELECT count(*) FROM consumer WHERE method=?',(arm,)).fetchone()[0]==t['consumers'] for arm,t in terminals.items())
    assert sum(t['consumers'] for t in terminals.values())==q.read(R/'private/GENERATION_COMPLETE.json')['new_consumers']
    db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(coverage=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method'))))


if __name__=='__main__':
    from configuration import require_execution
    require_execution(q.read(R/'public/PROTOCOL.json'),q.read(R/'public/EXECUTION_AUTHORIZATION.json'),q.read(R/'public/HARDWARE_AUTHORIZATION.json'))
    ensure_manifest()
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();ingest(db);q.ingest=lambda db:None
        op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"next"}'));write=q.write
        q.write=lambda path,data:write(path,data) if Path(path)!=q.ROOT/'ALL_WORKERS_COMPLETE.json' or terminal_allowed(db) else None
        try:answer=a.request(db,op)
        finally:q.write=write
        if terminal_allowed(db):q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_TERMINAL',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
