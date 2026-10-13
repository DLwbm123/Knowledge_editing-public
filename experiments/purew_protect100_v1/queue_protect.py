"""Exact historical score cache; new requests wait for full trajectory/reload freeze."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, os.environ['RUN_ROOT'] + '/private/tools')
import astra_queue as a
q, R = a.q, a.RUN
q.ROOT = R/'private/judge_purew146'
q.EFFORT = 'medium'; q.BATCH_LEDGER = 'Astra_purew146_batches'
P = Path(q.read(R/'private/FROZEN_PROTOCOL.json')['reference_run'])
original_status = q.status


def status(db):
    state = original_status(db)
    generation = bool(db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone())
    unresolved = db.execute("SELECT count(*) FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()[0]
    failed = bool(list((R/'private/failures').glob('*.json')))
    if generation and not unresolved:
        state['status'] = 'GENERATION_COMPLETE'
    elif failed:
        state['status'] = 'STOPPED'; state['failure'] = 'Retained stage failure needs review'
    elif not generation:
        state['status'] = 'GENERATING'
    elif not db.execute("SELECT 1 FROM payload WHERE status='PENDING'").fetchone() and state['status'] != 'STOPPED':
        state['status'] = 'SCORING'
    return state


q.status = status


def terminal_allowed(db):
    return bool(db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone()) and not db.execute(
            "SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()


def ro(path):
    db = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True); db.row_factory=sqlite3.Row
    return db


def selected_history(db):
    return list(db.execute("""SELECT * FROM consumer WHERE
       (method='PUREW_H146' AND ((mode='bank_R0' AND prefix IN (1,50,100))
         OR (mode='single_R0' AND edit_order<=100) OR (mode='insertion_R0' AND edit_order<=100)))
       OR (method IN ('TT88_W0_RETRO146','balancedit') AND mode='bank_R0' AND prefix=100)"""))


def expected_ids(tasks, ledger, phase):
    if phase['mode']=='bank_R0':
        assert phase['owner']==0 and phase['prefix'] in (1,50,100)
        return {q for t in tasks[:phase['prefix']] for e in t['events'] for q in e['all_probe_query_ids']}
    assert phase['mode']=='insertion_R0' and phase['owner']==phase['prefix'] and 1<=phase['owner']<=100
    t=tasks[phase['owner']-1]
    return {t['edit_id']} | {q for e in t['events'] if e['task']=='T2G'
                           for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]}


def ingest(db):
    if not db.execute("SELECT 1 FROM done WHERE name='historical'").fetchone():
        old=ro(P/'private/judge_purew146/queue.sqlite')
        if os.environ.get('PROTECT_CPU_TRACE'): print('CACHE_ROWS',old.execute('SELECT count(*) FROM payload').fetchone()[0],file=sys.stderr,flush=True)
        prior_by_key={}
        # Cache all terminal old bindings so an identical new response never retries a prior missing payload.
        for index,prior in enumerate(old.execute('SELECT * FROM payload')):
            assert prior['status'] in ('FORMAT_VALID','MISSING')
            full=json.loads(prior['binding'])
            assert full['judge']==dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
            assert q.digest(full)==prior['key']
            prior_by_key[prior['key']]=full
            db.execute('INSERT OR IGNORE INTO payload VALUES (?,?,?,?,?,?)',tuple(prior))
            if os.environ.get('PROTECT_CPU_TRACE') and index%500==0:print('CACHE',index,file=sys.stderr,flush=True)
        selected=selected_history(old)
        assert q.read(P/'public/BASELINE_REUSE_ADMISSION.json')['status']=='PASS'
        assert q.read(P/'public/PAUSE100_RESULTS.json')['status']=='PAUSED_PARTIAL_PREFIX100_TERMINAL_SCORING'
        ledger=q.read(P/'private/EVAL_LEDGER.json');bindings=q.read(P/'private/EVAL_BINDINGS.json')
        if os.environ.get('PROTECT_CPU_TRACE'): print('SELECTED_ROWS',len(selected),file=sys.stderr,flush=True)
        for index,row in enumerate(selected):
            full=prior_by_key[row['payload_key']]
            # Parent binding audits are already terminal; preserve their output hashes, verify only the reusable full payload contract.
            query=ledger['queries'][row['query_id']];bound=bindings[query['opaque_Base_id']]
            assert Path(row['path']).is_file()
            assert full['query_id']==query['query_id']
            assert all(full[k]==query[k]==bound[k] for k in ('question','reference','image_sha256'))
            assert all(full[k]==bound[k] for k in ('image_path','prompt_ids','attention_mask','runtime','generation'))
            values=list(row)
            if values[1]=='PUREW_H146': values[1]='C'
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',values)
            if os.environ.get('PROTECT_CPU_TRACE') and index%500==0:print('BOUND',index,file=sys.stderr,flush=True)
        old.close(); db.execute("INSERT INTO done VALUES ('historical')"); db.commit()
    # Evaluation feedback is unavailable while any new training/reload stage remains.
    if not (R/'private/GENERATION_COMPLETE.json').exists(): return
    assert (R/'private/FINAL_CONSUMED.json').exists()
    assert not any(not s.get('ended_epoch') for s in q.read(R/'RESOURCE_LEDGER.json')['gpu_sessions'])
    tasks=q.read(R/'private/BENCHMARK146_QUEUE.json')['tasks']; ledger=q.read(R/'private/EVAL_LEDGER.json')
    for marker in sorted((R/'private/outputs').glob('*/*/*/COMPLETE.json')):
        name='panel:'+str(marker.relative_to(R))
        if db.execute('SELECT 1 FROM done WHERE name=?',(name,)).fetchone(): continue
        receipt=q.read(marker); phase=receipt['phase']
        assert phase['arm']=='K' and phase['protocol']==q.digest(q.read(R/'private/FROZEN_PROTOCOL.json'))
        files=[p for p in marker.parent.glob('*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
        assert len(files)==receipt['outputs'] and {p.stem for p in files}==expected_ids(tasks,ledger,phase)
        for path in files:
            d=q.read(path); bb=d['binding']
            assert bb['phase']==phase and bb['input']==ledger['queries'][bb['input']['query_id']]
            key=q.payload(db,bb['input'],bb['judge_input'],d['R0'])
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),'K',phase['mode'],
                    phase['prefix'],phase['owner'],'INSERTION' if phase['mode']=='insertion_R0' else 'PANEL',
                    bb['input']['query_id'],str(path),q.digest(d),key))
        db.execute('INSERT INTO done VALUES (?)',(name,));db.commit()
    assert db.execute("SELECT count(*) FROM consumer WHERE method='K'").fetchone()[0]==2181
    db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(coverage=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method')),
                                  statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))))


if __name__=='__main__':
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect()
        if not (q.ROOT/'EPOCH_MANIFEST.json').exists():
            q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch='PUREW_PROTECT100_V1_20261013',model='gpt-6-astra',
                    reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,workers=4,batch_size=50))
        ingest(db);q.ingest=lambda db:None
        op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"next"}'))
        write=q.write
        q.write=lambda path,data:write(path,data) if Path(path)!=q.ROOT/'ALL_WORKERS_COMPLETE.json' or terminal_allowed(db) else None
        try: answer=a.request(db,op)
        finally: q.write=write
        if terminal_allowed(db):
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_TERMINAL',epoch=time.time(),
                    statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))))
    print(json.dumps(answer,ensure_ascii=False))
