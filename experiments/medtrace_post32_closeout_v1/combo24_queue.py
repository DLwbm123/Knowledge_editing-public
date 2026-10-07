"""Separate paired-U and fit-anchor research outputs; immutable parent scores."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import time

RUN = Path(os.environ['RUN_ROOT'])
spec = importlib.util.spec_from_file_location('parallel_astra', RUN/'private/tools/astra_queue.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
q = a.q
q.ROOT = RUN/'private/judge_combo24_astra_medium'
q.BATCH_LEDGER = 'Astra_combo24_batches'
EPOCH = 'MEDTRACE_POST32_COMBO24_ASTRA_MEDIUM_20261008_V1'
ARMS = ('CE_ONLY','CE_U_MULTI','COMBO24_CE_FIT5','COMBO24_CEU_FIT5')


def initialize(db):
    lock = dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=4,attempts_per_payload=1,Judge_limit_enabled=False,
        deadline_epoch=q.read(RUN/'RUN_MANIFEST.json')['deadline_epoch'],
        scientific_lock=q.digest(q.read(RUN/'private/COMBO24_LOCK.json')))
    p = q.ROOT/'EPOCH_MANIFEST.json'
    if p.exists(): assert q.read(p)==lock
    else: q.write(p,lock)
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_folder TEXT,parent_batch TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS projection (consumer_id TEXT PRIMARY KEY,active_target INTEGER,effective_expert TEXT)')


def parent_rows():
    def rows(folder):
        c = sqlite3.connect('file:'+str(RUN/'private'/folder/'queue.sqlite')+'?mode=ro',uri=True)
        c.row_factory = sqlite3.Row
        result = {r['key']:(dict(r),folder) for r in c.execute('SELECT * FROM payload')}
        c.close()
        return result
    original = rows('judge_astra_medium')
    recovery = rows('judge_astra_medium_recovery1')
    missing = {k for k,(r,_) in original.items() if r['status']=='MISSING'}
    assert set(recovery)==missing and len(missing)==50 and len(original)==2137
    for k,(r,_) in recovery.items():
        assert r['binding']==original[k][0]['binding'] and r['record']==original[k][0]['record']
    original.update(recovery)
    assert all(r['status']=='FORMAT_VALID' for r,_ in original.values())
    for folder in ('judge_research_astra_medium','judge_lora146_astra_medium'):
        for key,(r,origin) in rows(folder).items():
            assert r['status'] in ('FORMAT_VALID','MISSING')
            if key in original:
                prior=original[key][0]
                assert prior['binding']==r['binding'] and prior['status']==r['status'] and prior['correct']==r['correct']
            else:
                assert not r['batch'].startswith('INHERITED_'), 'Inherited keys must resolve to existing original evidence'
                original[key]=(r,origin)
    return original


def ingest(db):
    if (q.ROOT/'READY.json').exists(): return
    assert (RUN/'private/COMBO24_GENERATION_COMPLETE.json').exists()
    paths=[Path(p) for p in q.read(RUN/'private/COMBO24_BASELINE_PATHS.json')]
    for arm in ARMS[2:]:
        paths+=sorted((RUN/'private/outputs/bank').glob('*/s0/'+arm+'/n160/p24/R0/*.json'))
    assert len(paths)==1132 and len(set(paths))==1132
    rows=[]
    for path in paths:
        d=q.read(path);b=d['binding'];p=b['phase']
        rows.append(dict(path=str(path),mode=b['mode'],prefix=b['prefix'],edit_order=b['owner_order'],query_id=b['input']['query_id'],arm=b['arm'],slot=p['slot']))
    assert all(sum(c['arm']==arm for c in rows)==283 for arm in ARMS)
    parents, evidence = parent_rows(), {}
    tasks = q.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks']
    ledger = q.read(RUN/'private/EVAL_LEDGER.json')
    bindings = q.read(RUN/'private/EVAL_BINDINGS.json')
    model_lock = q.read(RUN/'private/GPU_SOURCE_VERSION.json')
    for i,c in enumerate(rows,1):
        cid = q.digest(c)
        if db.execute('SELECT 1 FROM consumer WHERE id=?',(cid,)).fetchone(): continue
        d=q.read(c['path']);b=d['binding'];phase=b['phase'];row=b['input']
        if c['arm'] in ARMS[2:]:
            assert d['research_lock']==q.digest(q.read(RUN/'private/COMBO24_LOCK.json')) and phase['execution']==model_lock
        assert b['arm']==c['arm'] and phase['slot']==0 and phase['node']==160 and phase['prefix']==24
        if not row.get('role'):
            assert row==ledger['queries'][c['query_id']] and b['judge_input']==bindings[row['opaque_Base_id']]
        arm=ARMS[0] if c['arm'] in (ARMS[0],ARMS[2]) else ARMS[1]
        assert phase['weights']==q.read(RUN/'private/COMBO24_WEIGHT_BINDINGS.json')[arm]
        assert d['TT_parameters_per_expert']==7168
        key = q.payload(db,row,b['judge_input'],d['R0'])
        if key in parents and not db.execute('SELECT 1 FROM inherited WHERE key=?',(key,)).fetchone():
            p,folder = parents[key]
            full = json.loads(p['binding'])
            assert json.loads(db.execute('SELECT binding FROM payload WHERE key=?',(key,)).fetchone()[0])==full
            if p['status']=='MISSING':
                saved=q.read(RUN/'private'/folder/'evidence'/(p['batch']+'.json'))
                assert saved['evidence']['status']!='FORMAT_VALID'
                db.execute("UPDATE payload SET status='MISSING',correct=NULL,batch=? WHERE key=?",('INHERITED_'+p['batch'],key))
            else:
                ek = (folder,p['batch'])
                if ek not in evidence:
                    saved = q.read(RUN/'private'/folder/'evidence'/(p['batch']+'.json'))
                    ev = saved['evidence']
                    assert ev['status']=='FORMAT_VALID' and ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
                    assert ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
                    assert ev['input_binding']==q.digest(saved['batch'])
                    decisions = q.validate(saved['batch'],saved['response'])
                    evidence[ek] = (saved['batch']['records'],{v['opaque_query_id']:v['is_correct'] for v in decisions})
                records, decisions = evidence[ek]
                rec = json.loads(p['record'])
                assert rec in records and int(decisions[rec['opaque_query_id']])==p['correct']
                db.execute("UPDATE payload SET status='FORMAT_VALID',correct=?,batch=? WHERE key=?",(p['correct'],'INHERITED_'+p['batch'],key))
            db.execute('INSERT INTO inherited VALUES (?,?,?)',(key,folder,p['batch']))
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (cid,c['arm'],c['mode'],c['prefix'],c['edit_order'],row.get('role','PANEL'),c['query_id'],c['path'],q.digest(d),key))
        db.execute('INSERT INTO projection VALUES (?,?,?)',(cid,int(d['active_target']),d['effective_expert']))
        if i%50==0:
            db.commit()
            print('INGESTED',i,len(rows),flush=True)
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==len(rows)
    db.execute("INSERT OR IGNORE INTO done VALUES ('generation')")
    db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=len(rows),payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],
        inherited=db.execute('SELECT count(*) FROM inherited').fetchone()[0],epoch=time.time()))
    if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED') LIMIT 1").fetchone():
        q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))


if __name__=='__main__':
    operation = json.loads(os.environ['QUEUE_REQUEST']) if 'QUEUE_REQUEST' in os.environ else dict(action='ingest')
    q.ROOT.mkdir(parents=True,exist_ok=True)
    if operation['action']=='next' and not (q.ROOT/'READY.json').exists():
        # Ingestion holds the write lock; waiting scorers must not block on its disk scan.
        stopped = time.time()>=q.read(RUN/'RUN_MANIFEST.json')['deadline_epoch']-600 or (RUN/'STOP').exists()
        stopped = stopped or (RUN/'private/FAILURE_combo24_controller_all.json').exists()
        print(json.dumps(dict(state=dict(status='STOPPED' if stopped else 'GENERATING',payloads={},stage='24-edit interaction generation and binding audit'),rows=[])))
        raise SystemExit(0)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db = q.connect()
        initialize(db)
        if operation['action']=='ingest':
            ingest(db)
            answer = q.read(q.ROOT/'READY.json')
        elif operation['action']=='next' and not (q.ROOT/'READY.json').exists():
            answer = dict(state=dict(status='GENERATING',payloads={},stage='24-edit interaction generation and binding audit'),rows=[])
        else:
            q.ingest = lambda db: None
            answer = a.request(db,operation)
    print(json.dumps(answer,ensure_ascii=False))
