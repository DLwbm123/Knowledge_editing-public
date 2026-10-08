"""Incremental frozen-phase ingestion; exact historical score inheritance."""
import os
import json
import sqlite3
import fcntl
import time
from pathlib import Path
import sys
sys.path.insert(0,os.environ["RUN_ROOT"]+"/private/tools")
import astra_queue as a
q=a.q;RUN=q.RUN;PARENT=Path(q.read(RUN/'PLAN_CONFIG.json')['parent_run'])
q.ROOT=RUN/'private/judge_paper_astra_medium';q.BATCH_LEDGER='Astra_paper_batches'
EPOCH='MEDTRACE_PAPER_TRANSFER_ASTRA_MEDIUM_20261008_V1'

def parents():
    result={}
    for folder in ('judge_astra_medium','judge_astra_medium_recovery1','judge_research_astra_medium','judge_lora146_astra_medium','judge_combo24_astra_medium','judge_weak24_astra_medium'):
        db=sqlite3.connect('file:'+str(PARENT/'private'/folder/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM payload'):
            d=dict(row);key=d['key'];assert d['status'] in ('FORMAT_VALID','MISSING')
            if d['batch'].startswith('INHERITED_'):assert key in result;continue
            if key in result:
                old=result[key][0];assert old['binding']==d['binding']
                if folder!='judge_astra_medium_recovery1':assert old['status']==d['status'] and old['correct']==d['correct']
            result[key]=(d,folder)
        db.close()
    return result

def initialize(db):
    lock=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,batch_size=50,workers=4,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/PAPER_LOCK.json')))
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    scope=q.read(RUN/'private/TT_ONLY_AMENDMENT.json');assert scope['CP_enabled'] is False and scope['consumers']['total']==10395
    scope_path=q.ROOT/'TT_ONLY_AMENDMENT.json'
    if scope_path.exists():assert q.read(scope_path)==scope
    else:q.write(scope_path,scope)
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_folder TEXT,parent_batch TEXT)')

def ingest(db):
    parent=parents();evidence={}
    ledger=q.read(RUN/'private/EVAL_LEDGER.json');bindings=q.read(RUN/'private/EVAL_BINDINGS.json');source=q.read(RUN/'private/GPU_SOURCE_VERSION.json');scientific_lock=q.digest(q.read(RUN/'private/PAPER_LOCK.json'))
    for stage,count,workers in [('A',7565,3),('B',2547,6),('D',283,6)]:
        if db.execute('SELECT 1 FROM done WHERE name=?',(stage,)).fetchone():continue
        if not all((RUN/'private'/('EVAL_'+stage+'_'+str(i)+'.json')).exists() for i in range(workers)):continue
        paths=sorted((RUN/'private/outputs').glob(stage+'_*/*.json'));assert len(paths)==count
        for index,path in enumerate(paths):
            d=q.read(path);b=d['binding'];row=b['input'];phase=b['phase']
            expected_source=source if stage=='A' else q.read(RUN/'private/PAPER_TT_SOURCE_VERSION.json')
            assert d['research_lock']==scientific_lock and phase['execution']==expected_source
            if stage!='A':
                assert phase['tt_only_amendment']==q.digest(q.read(RUN/'private/TT_ONLY_AMENDMENT.json'))
                assert b['arm'] in (['B_'+arm+'_'+route for arm in ('TT_CE','TT_ALIGN','TT_FIXED_A') for route in ('R0','MODAL','INTRINSIC_MODAL')] if stage=='B' else ['D_TT_ALIGN_INTRINSIC_MODAL_MIX'])
            assert phase['slot']==0 and phase['prefix']==(146 if stage=='A' else 24)
            if not row.get('role'):
                assert row==ledger['queries'][row['query_id']]
                assert b['judge_input']==bindings[row['opaque_Base_id']]
            cid=q.digest([str(path),b['arm'],row['query_id']])
            if db.execute('SELECT 1 FROM consumer WHERE id=?',(cid,)).fetchone():continue
            key=q.payload(db,row,b['judge_input'],d['R0'])
            if key in parent and not db.execute('SELECT 1 FROM inherited WHERE key=?',(key,)).fetchone():
                p,folder=parent[key]
                assert json.loads(db.execute('SELECT binding FROM payload WHERE key=?',(key,)).fetchone()[0])==json.loads(p['binding'])
                ek=(folder,p['batch'])
                if ek not in evidence:
                    saved=q.read(PARENT/'private'/folder/'evidence'/(p['batch']+'.json'));ev=saved['evidence']
                    if p['status']=='FORMAT_VALID':
                        assert ev['status']=='FORMAT_VALID' and ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
                        assert ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values()) and ev['input_binding']==q.digest(saved['batch'])
                        evidence[ek]=(saved['batch']['records'],{x['opaque_query_id']:x['is_correct'] for x in q.validate(saved['batch'],saved['response'])})
                    else:assert ev['status']!='FORMAT_VALID';evidence[ek]=None
                if p['status']=='FORMAT_VALID':
                    records,decisions=evidence[ek];rec=json.loads(p['record']);assert rec in records and int(decisions[rec['opaque_query_id']])==p['correct']
                db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',(p['status'],p['correct'],'INHERITED_'+p['batch'],key))
                db.execute('INSERT INTO inherited VALUES (?,?,?)',(key,folder,p['batch']))
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(cid,b['arm'],b['mode'],b['prefix'],b['owner_order'],row.get('role','PANEL'),row['query_id'],str(path),q.digest(d),key))
            if index%100==0:db.commit();print('INGEST',stage,index,flush=True)
        db.execute('INSERT INTO done VALUES (?)',(stage,));db.commit()
    if (RUN/'private/GENERATION_COMPLETE.json').exists():
        assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==10395
        db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=db.execute('SELECT count(*) FROM consumer').fetchone()[0],payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited=db.execute('SELECT count(*) FROM inherited').fetchone()[0],generation_complete=bool(db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone())))

def complete(db):
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))

if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);db=q.connect();initialize(db);q.ingest=lambda db:None
        if op['action']=='ingest':ingest(db);answer=q.read(q.ROOT/'READY.json')
        else:
            # The original helper seals on an empty queue; defer sealing until all phases exist.
            original=q.write
            def write(path,data):
                if Path(path).name=='ALL_WORKERS_COMPLETE.json' and not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
                original(path,data)
            q.write=write;answer=a.request(db,op)
        complete(db)
    print(json.dumps(answer,ensure_ascii=False))
