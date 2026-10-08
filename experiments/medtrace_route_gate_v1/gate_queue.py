"""One-attempt Astra queue; inherit exact prior bindings, including missing scores."""
import os
import sys
import json
import sqlite3
import fcntl
import time
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import paper_queue as previous
a,q,RUN=previous.a,previous.q,previous.RUN
q.ROOT=RUN/'private/judge_gate_astra_medium';q.BATCH_LEDGER='Astra_gate_batches'
EPOCH='MEDTRACE_ROUTE_GATE_ASTRA_MEDIUM_20261008_V1'

def parents():
    result={key:(record,previous.PARENT/'private'/folder) for key,(record,folder) in previous.parents().items()}
    root=Path(q.read(RUN/'PLAN_CONFIG.json')['gate_previous'])/'private/judge_paper_astra_medium'
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    for row in db.execute('SELECT * FROM payload'):
        d=dict(row);key=d['key'];assert d['status'] in ('FORMAT_VALID','MISSING')
        if d['batch'].startswith('INHERITED_'):
            old=result[key][0];assert json.loads(old['binding'])==json.loads(d['binding']) and old['status']==d['status'] and old['correct']==d['correct']
        else:
            if key in result:assert result[key][0]['status']==d['status'] and result[key][0]['correct']==d['correct']
            result[key]=(d,root)
    db.close();return result

def initialize(db):
    lock=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,batch_size=50,workers=4,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/GATE_LOCK.json')))
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_folder TEXT,parent_batch TEXT)')

def ingest(db):
    parent=parents();evidence={}
    ledger=q.read(RUN/'private/EVAL_LEDGER.json');bindings=q.read(RUN/'private/EVAL_BINDINGS.json')
    source=q.read(RUN/'private/GPU_SOURCE_VERSION.json');lock=q.digest(q.read(RUN/'private/GATE_LOCK.json'))
    for stage,count in [('G',6052),('P',4539)]:
        if db.execute('SELECT 1 FROM done WHERE name=?',(stage,)).fetchone():continue
        if not all((RUN/'private'/('EVAL_'+stage+'_'+str(i)+'.json')).exists() for i in range(6)):continue
        paths=sorted((RUN/'private/outputs').glob(stage+'_*/*.json'));assert len(paths)==count
        for index,path in enumerate(paths):
            d=q.read(path);b=d['binding'];row=b['input'];phase=b['phase']
            assert d['research_lock']==lock and phase['gate_lock']==lock and phase['execution']==source
            assert phase['slot']==phase['node']==0 and phase['prefix']==146
            if not row.get('role'):
                assert row==ledger['queries'][row['query_id']] and b['judge_input']==bindings[row['opaque_Base_id']]
            cid=q.digest([str(path),b['arm'],row['query_id']])
            if db.execute('SELECT 1 FROM consumer WHERE id=?',(cid,)).fetchone():continue
            key=q.payload(db,row,b['judge_input'],d['R0'])
            if key in parent and not db.execute('SELECT 1 FROM inherited WHERE key=?',(key,)).fetchone():
                old,root=parent[key]
                assert json.loads(db.execute('SELECT binding FROM payload WHERE key=?',(key,)).fetchone()[0])==json.loads(old['binding'])
                ek=(str(root),old['batch'])
                if ek not in evidence:
                    saved=q.read(root/'evidence'/(old['batch']+'.json'));ev=saved['evidence']
                    if old['status']=='FORMAT_VALID':
                        assert ev['status']=='FORMAT_VALID' and ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
                        assert ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values()) and ev['input_binding']==q.digest(saved['batch'])
                        evidence[ek]=(saved['batch']['records'],{x['opaque_query_id']:x['is_correct'] for x in q.validate(saved['batch'],saved['response'])})
                    else:assert ev['status']!='FORMAT_VALID';evidence[ek]=None
                if old['status']=='FORMAT_VALID':
                    records,decisions=evidence[ek];rec=json.loads(old['record']);assert rec in records and int(decisions[rec['opaque_query_id']])==old['correct']
                db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',(old['status'],old['correct'],'INHERITED_'+old['batch'],key))
                db.execute('INSERT INTO inherited VALUES (?,?,?)',(key,str(root),old['batch']))
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(cid,b['arm'],b['mode'],b['prefix'],b['owner_order'],row.get('role','PANEL'),row['query_id'],str(path),q.digest(d),key))
            if index%100==0:db.commit();print('INGEST',stage,index,flush=True)
        db.execute('INSERT INTO done VALUES (?)',(stage,));db.commit()
    if (RUN/'private/GENERATION_COMPLETE.json').exists():
        assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==10591
        db.execute("INSERT OR IGNORE INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=db.execute('SELECT count(*) FROM consumer').fetchone()[0],payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited=db.execute('SELECT count(*) FROM inherited').fetchone()[0],generation_complete=bool(db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone())))

if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'));q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);db=q.connect();initialize(db);q.ingest=lambda db:None
        if op['action']=='ingest':ingest(db);answer=q.read(q.ROOT/'READY.json')
        else:
            original=q.write
            def write(path,data):
                if Path(path).name=='ALL_WORKERS_COMPLETE.json' and not db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
                original(path,data)
            q.write=write;answer=a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
