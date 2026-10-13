"""Separate Astra queue for new LoRA outputs; accepted TT scores stay immutable."""
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
q.ROOT = RUN/'private/judge_lora146_astra_medium'
q.BATCH_LEDGER = 'Astra_LoRA146_batches'
EPOCH = 'MEDTRACE_POST32_LORA146_ASTRA_MEDIUM_20261007_V1'
ARM = 'LORA_R1_W0_RETRO146'


def initialize(db):
    lock = dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=4,attempts_per_payload=1,Judge_limit_enabled=False,
        deadline_epoch=q.read(RUN/'RUN_MANIFEST.json')['deadline_epoch'],
        scientific_lock=q.digest(q.read(RUN/'private/LORA146_LOCK.json')))
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
    return original


def ingest(db):
    if (q.ROOT/'READY.json').exists(): return
    assert q.read(RUN/'private/LORA146_GENERATION_COMPLETE.json')['consumers']==5722
    rows = q.read(RUN/'private/LORA146_CONSUMERS.json')
    assert len(rows)==5722
    parents, evidence = parent_rows(), {}
    tasks = q.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks']
    ledger = q.read(RUN/'private/EVAL_LEDGER.json')
    bindings = q.read(RUN/'private/EVAL_BINDINGS.json')
    model_lock = q.read(RUN/'private/GPU_SOURCE_VERSION.json')
    for i,c in enumerate(rows,1):
        cid = q.digest(c)
        if db.execute('SELECT 1 FROM consumer WHERE id=?',(cid,)).fetchone(): continue
        d = q.read(c['path']); b = d['binding']; phase = b['phase']; row = ledger['queries'][c['query_id']]
        assert b['input']==row and b['judge_input']==bindings[row['opaque_Base_id']]
        assert (b['arm'],b['mode'],b['prefix'],b['owner_order'],b['panel'])==(ARM,c['mode'],c['prefix'],c['edit_order'],'PANEL')
        assert phase['arm']==ARM and phase['node']==0 and phase['slot']==0 and not phase['forced'] and phase['execution']==model_lock
        active = tasks[:c['prefix']] if c['mode']=='bank_R0' else []
        assert d['active_target']==((row['image_sha256'],row['question']) in {(t['native']['image_sha256'],t['native']['question']) for t in active})
        assert d['expert_parameters_per_expert']==18432 and d['representation']=='NORMALIZED_LORA_R1'
        inserted = [t for t in tasks if t['order']==c['edit_order']] if c['mode']=='single_R0' else tasks[:c['prefix']]
        assert list(phase['weights'])==[t['edit_id'] for t in inserted]
        assert [r['logical_edit_id'] for r in phase['router']]==[t['edit_id'] for t in inserted]
        key = q.payload(db,row,b['judge_input'],d['R0'])
        if key in parents and not db.execute('SELECT 1 FROM inherited WHERE key=?',(key,)).fetchone():
            p,folder = parents[key]
            full = json.loads(p['binding'])
            assert json.loads(db.execute('SELECT binding FROM payload WHERE key=?',(key,)).fetchone()[0])==full
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
            (cid,ARM,c['mode'],c['prefix'],c['edit_order'],'PANEL',c['query_id'],c['path'],q.digest(d),key))
        db.execute('INSERT INTO projection VALUES (?,?,?)',(cid,int(d['active_target']),d['effective_expert']))
        if i%50==0:
            db.commit()
            print('INGESTED',i,len(rows),flush=True)
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==5722
    db.execute("INSERT OR IGNORE INTO done VALUES ('generation')")
    db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=5722,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],
        inherited=db.execute('SELECT count(*) FROM inherited').fetchone()[0],epoch=time.time()))
    if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED') LIMIT 1").fetchone():
        q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))


if __name__=='__main__':
    operation = json.loads(os.environ['QUEUE_REQUEST']) if 'QUEUE_REQUEST' in os.environ else dict(action='ingest')
    q.ROOT.mkdir(parents=True,exist_ok=True)
    if operation['action']=='next' and not (q.ROOT/'READY.json').exists():
        # Ingestion holds the write lock; waiting scorers must not block on its disk scan.
        stopped = time.time()>=q.read(RUN/'RUN_MANIFEST.json')['deadline_epoch']-600 or (RUN/'STOP').exists()
        stopped = stopped or (RUN/'private/FAILURE_lora146_controller_all.json').exists()
        print(json.dumps(dict(state=dict(status='STOPPED' if stopped else 'GENERATING',payloads={},stage='LoRA146 generation and binding audit'),rows=[])))
        raise SystemExit(0)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db = q.connect()
        initialize(db)
        if operation['action']=='ingest':
            ingest(db)
            answer = q.read(q.ROOT/'READY.json')
        elif operation['action']=='next' and not (q.ROOT/'READY.json').exists():
            answer = dict(state=dict(status='GENERATING',payloads={},stage='LoRA146 generation and binding audit'),rows=[])
        else:
            q.ingest = lambda db: None
            answer = a.request(db,operation)
    print(json.dumps(answer,ensure_ascii=False))
