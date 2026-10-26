"""Private source-agreement queue; accepted judgments and predictions are reused.

Cache identity includes the original query, image, prompt, complete answer and
tokens, and the complete Judge contract. Method/runtime provenance stays on each
consumer; it is not shown to a Judge that only receives Q/reference/answer/ID.
"""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path(os.environ['V131_ROOT'])
OUT = ROOT / 'judge_v131'
CFG = json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())
SOURCES = json.loads((ROOT/'JUDGE_CACHE_SOURCES.json').read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def payload_key(qid, binding, answer, effort, prompt):
    return digest(dict(query_id=qid,question=binding['question'],reference=binding['reference'],
        image_sha256=binding['image_sha256'],prompt_ids=binding['prompt_ids'],output=answer,
        judge=dict(model='gpt-6-astra',reasoning_effort=effort,
            protocol=CFG['scoring']['protocol'],prompt=prompt)))


def initialize(db):
    if db.execute("SELECT count(*) FROM sqlite_master WHERE name='meta'").fetchone()[0]:
        assert db.execute("SELECT value FROM meta WHERE key='prompt'").fetchone(), 'Incomplete queue initialization; inspect before recovery'
        return
    db.executescript('''
      CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
      CREATE TABLE cache(key TEXT PRIMARY KEY,correct INTEGER,conflict INTEGER,source TEXT);
      CREATE TABLE historical_missing(key TEXT PRIMARY KEY,source TEXT);
      CREATE TABLE direct(path TEXT,effort TEXT,key TEXT,correct INTEGER,source TEXT,PRIMARY KEY(path,effort,key));
      CREATE TABLE payload(key TEXT PRIMARY KEY,record TEXT,binding TEXT,effort TEXT,state TEXT,
                           correct INTEGER,source TEXT,attempts INTEGER DEFAULT 0);
      CREATE TABLE consumer(id TEXT PRIMARY KEY,path TEXT UNIQUE,method TEXT,mode TEXT,
                            prefix INTEGER,owner INTEGER,query_id TEXT,payload_key TEXT);
      CREATE TABLE batch(id TEXT PRIMARY KEY,keys TEXT,effort TEXT,state TEXT,attempt INTEGER,worker TEXT);
      CREATE TABLE attempt(batch_id TEXT,number INTEGER,evidence TEXT,response TEXT,
                           PRIMARY KEY(batch_id,number));
    ''')
    prompt = None
    imported = collections.Counter()
    for source in SOURCES:
        manifest = read(Path(source['database']).parent/'EPOCH_MANIFEST.json')
        if prompt is None:
            prompt = manifest['prompt']
        assert manifest['prompt'] == prompt and manifest['protocol'] == CFG['scoring']['protocol']
        old = sqlite3.connect('file:'+source['database']+'?mode=ro',uri=True)
        old.row_factory = sqlite3.Row
        for row in old.execute('SELECT * FROM payload'):
            binding = json.loads(row['binding'])
            judge = binding['judge']
            effort = judge['reasoning_effort']
            assert judge['model']=='gpt-6-astra' and judge['protocol']==CFG['scoring']['protocol']
            assert judge['prompt']==prompt
            key = payload_key(binding['query_id'],binding,binding['output'],effort,prompt)
            proof = json.dumps(dict(database=source['database'],old_key=row['key']))
            if row['status'] == 'FORMAT_VALID':
                assert row['correct'] in (0,1)
                prior = db.execute('SELECT correct FROM cache WHERE key=?',(key,)).fetchone()
                if prior is None:
                    db.execute('INSERT INTO cache VALUES (?,?,0,?)',(key,row['correct'],proof))
                elif prior[0] != row['correct']:
                    db.execute('UPDATE cache SET conflict=1 WHERE key=?',(key,))
                imported['accepted_payloads'] += 1
            elif row['status'] == 'MISSING':
                db.execute('INSERT OR IGNORE INTO historical_missing VALUES (?,?)',(key,proof))
                imported['historical_missing_payloads'] += 1
        for row in old.execute('SELECT c.path,p.key,p.binding,p.correct FROM consumer c JOIN payload p ON p.key=c.payload_key WHERE p.status=\'FORMAT_VALID\''):
            binding = json.loads(row['binding'])
            effort = binding['judge']['reasoning_effort']
            key = payload_key(binding['query_id'],binding,binding['output'],effort,prompt)
            proof = json.dumps(dict(database=source['database'],old_key=row['key'],consumer_path=row['path']))
            # Legacy files may be bundles containing many different queries.
            prior = db.execute('SELECT key,correct FROM direct WHERE path=? AND effort=? AND key=?',(row['path'],effort,key)).fetchone()
            assert prior is None or tuple(prior)==(key,row['correct'])
            db.execute('INSERT OR IGNORE INTO direct VALUES (?,?,?,?,?)',(row['path'],effort,key,row['correct'],proof))
        old.close()
    db.executemany('INSERT INTO meta VALUES (?,?)',[
        ('prompt',prompt),('new_payloads','0'),('charged_attempt_items','0'),('charged_requests','0')])
    (OUT/'CACHE_IMPORT.json').write_text(json.dumps(dict(imported,time=time.time()),indent=2)+'\n')
    db.commit()


def ingest(db):
    prompt = db.execute("SELECT value FROM meta WHERE key='prompt'").fetchone()[0]
    seen = {r[0] for r in db.execute('SELECT path FROM consumer')}
    for method,effort in CFG['scoring']['effort'].items():
        for path in sorted((ROOT/method/'outputs').glob('*.json')):
            if str(path) in seen:
                continue
            value = read(path)
            binding,answer = value['binding']['judge_input'],value['R0']
            qid = value['binding']['input']['query_id']
            assert answer['raw_token_ids'] and isinstance(answer['raw_answer'],str)
            key = payload_key(qid,binding,answer,effort,prompt)
            source_path = value.get('provenance',{}).get('prediction')
            direct = db.execute('SELECT key,correct,source FROM direct WHERE path=? AND effort=? AND key=?',
                                (source_path,effort,key)).fetchone() if source_path else None
            if direct is not None:
                assert direct[0] == key, 'Changed scientific input cannot inherit a path judgment'
                state,correct,proof = 'REUSED',direct[1],direct[2]
            else:
                cached = db.execute('SELECT correct,conflict,source FROM cache WHERE key=?',(key,)).fetchone()
                missing = db.execute('SELECT source FROM historical_missing WHERE key=?',(key,)).fetchone()
                if cached is not None and not cached[1]:
                    state,correct,proof = 'REUSED',cached[0],cached[2]
                elif cached is not None:
                    state,correct,proof = 'CACHE_CONFLICT',None,cached[2]
                elif missing is not None:
                    state,correct,proof = 'HISTORICAL_MISSING',None,missing[0]
                else:
                    state,correct,proof = 'PENDING',None,None
            previous = db.execute('SELECT state,correct FROM payload WHERE key=?',(key,)).fetchone()
            if previous is None:
                if state == 'PENDING':
                    number = int(db.execute("SELECT value FROM meta WHERE key='new_payloads'").fetchone()[0])+1
                    assert number <= CFG['scoring']['max_new_unique_payloads']
                    db.execute("UPDATE meta SET value=? WHERE key='new_payloads'",(str(number),))
                record = dict(opaque_query_id=key,question=binding['question'],
                    gold_answer=binding['reference'],raw_base_answer=answer['raw_answer'])
                db.execute('INSERT INTO payload VALUES (?,?,?,?,?,?,?,0)',(key,json.dumps(record),
                    json.dumps(dict(query_id=qid,binding=binding,output=answer)),effort,state,correct,proof))
            elif direct is not None:
                # Preserve original accepted M13P semantics if another consumer
                # introduced a conservative global-cache conflict first.
                if previous[0] in ('CACHE_CONFLICT','HISTORICAL_MISSING'):
                    db.execute('UPDATE payload SET state=\'REUSED\',correct=?,source=? WHERE key=?',(correct,proof,key))
                elif previous[1] is not None:
                    assert previous[1] == correct
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?)',(
                value['consumer_id'],str(path),method,value['mode'],value['prefix'],value['owner'],qid,key))
    db.commit()


def charge(db,count):
    charged = int(db.execute("SELECT value FROM meta WHERE key='charged_attempt_items'").fetchone()[0])+count
    assert charged <= CFG['scoring']['max_charged_attempt_items']
    db.execute("UPDATE meta SET value=? WHERE key='charged_attempt_items'",(str(charged),))
    db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='charged_requests'")


def producer_status():
    return {method:dict(complete=(ROOT/method/'COMPLETE.json').exists() or
        (method=='ALPHAEDIT_MM' and (ROOT/method/'GENERATION_COMPLETE.json').exists()) or
        (method=='MEMIT_MM' and (ROOT/method/'INCOMPLETE_GENERATION.json').exists()),
        failure=(ROOT/method/'FAILURE.json').exists()) for method in CFG['scoring']['effort']}


def snapshot(db):
    return dict(payload_states=dict(db.execute('SELECT state,count(*) FROM payload GROUP BY state')),
        ready_by_effort={effort:len(candidates(db,effort)) for effort in ('high','medium')},
        consumers=dict(db.execute('SELECT method,count(*) FROM consumer GROUP BY method')),
        meta={r[0]:r[1] for r in db.execute('SELECT * FROM meta') if r[0]!='prompt'},
        producers=producer_status(),time=time.time())


def candidates(db,effort):
    rows = db.execute('SELECT key,record FROM payload WHERE state=\'PENDING\' AND effort=? ORDER BY key LIMIT 50',(effort,)).fetchall()
    if rows and len(rows)<50:
        placeholders = ','.join('?' for _ in rows)
        methods = [r[0] for r in db.execute('SELECT DISTINCT method FROM consumer WHERE payload_key IN ('+placeholders+')',[r[0] for r in rows])]
        states = producer_status()
        if not all(states[method]['complete'] or states[method]['failure'] for method in methods):
            return []
    return rows


def transport_only(evidence):
    errors = evidence.get('errors',[])
    messages = [str(e.get('message',e.get('error',{}).get('message',''))) for e in errors]
    allowed = ['Transport error: network error: error decoding response body',
               'stream disconnected before completion: idle timeout waiting for SSE',
               'error sending request for url', 'Connection failed: error sending request',
               'connection reset', 'Connection reset']
    return (evidence.get('status')=='FAILED_NO_RETRY' and evidence.get('exit_code')==1 and
        bool(messages) and all(any(t in m for t in allowed) for m in messages) and
        not evidence.get('tool_event_types') and bool(evidence.get('isolation_checks')) and
        all(evidence['isolation_checks'].values()))


def selfcheck():
    binding = dict(question='Q',reference='R',image_sha256='image1',prompt_ids=[1,2])
    answer = dict(raw_answer='R',raw_token_ids=[3,4])
    original = payload_key('q1',binding,answer,'high','prompt')
    assert original == payload_key('q1',dict(binding),dict(answer),'high','prompt')
    assert original != payload_key('q1',dict(binding,image_sha256='image2'),answer,'high','prompt')
    assert original != payload_key('q1',binding,dict(answer,raw_token_ids=[3]),'high','prompt')
    assert original != payload_key('q1',binding,answer,'medium','prompt')
    assert original != payload_key('q2',binding,answer,'high','prompt')
    failure = dict(status='FAILED_NO_RETRY',exit_code=1,
        errors=[dict(message='Connection failed: error sending request')],
        tool_event_types=[],isolation_checks=dict(own_input=True,operator=True))
    assert transport_only(failure)
    assert not transport_only(dict(failure,tool_event_types=['command_execution']))
    assert not transport_only(dict(failure,errors=[dict(message='Invalid score format')]))
    return dict(status='PASS',changed_image_tokens_effort_or_query_cannot_inherit=True,
        transport_recovery_excludes_tools_and_format=True)


def action(request):
    OUT.mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        db = sqlite3.connect(OUT/'queue.sqlite')
        try:
            initialize(db)
            kind = request['action']
            if kind in ('reserve','status','initialize'):
                ingest(db)
            if kind == 'reserve':
                if not CFG.get('run_until_completion') and time.time() >= CFG['hard_end'] - 620:
                    return dict(status='DEADLINE_NO_NEW_CALL',summary=snapshot(db))
                effort = request['effort']
                rows = candidates(db,effort)
                if not rows:
                    return dict(status='WAIT',summary=snapshot(db))
                keys = [r[0] for r in rows]
                bid = digest([CFG['epoch'],effort,keys])
                charge(db,len(keys))
                db.execute('INSERT INTO batch VALUES (?,?,?,\'RESERVED\',0,?)',(bid,json.dumps(keys),effort,request['worker']))
                db.executemany('UPDATE payload SET state=\'RESERVED\',attempts=attempts+1 WHERE key=?',[(k,) for k in keys])
                db.commit()
                return dict(status='RESERVED',effort=effort,batch=dict(batch_id=bid,records=[json.loads(r[1]) for r in rows]),prompt=db.execute("SELECT value FROM meta WHERE key='prompt'").fetchone()[0],attempt=0)
            if kind == 'recover_transport':
                # Recover only the unused preregistered transport attempt. The
                # original failed attempt and MISSING evidence stay immutable.
                bid = request['batch_id']
                row = db.execute('SELECT keys,effort,state,attempt FROM batch WHERE id=?',(bid,)).fetchone()
                assert row and row[2]=='MISSING' and row[3]==0
                old = db.execute('SELECT evidence,response FROM attempt WHERE batch_id=? AND number=0',(bid,)).fetchone()
                evidence = json.loads(old[0])
                assert json.loads(old[1]) is None and request['no_valid_response'] is True
                assert transport_only(evidence) and CFG['scoring']['transport_retries']==1
                assert evidence['actual_model']=='gpt-6-astra' and evidence['reasoning_effort']==row[1]
                keys = json.loads(row[0])
                payloads = [db.execute('SELECT state,attempts,record FROM payload WHERE key=?',(key,)).fetchone() for key in keys]
                assert all(item[0]=='MISSING' and item[1]==1 for item in payloads)
                charge(db,len(keys))
                db.execute('UPDATE batch SET state=\'RESERVED\',attempt=1,worker=? WHERE id=?',(request['worker'],bid))
                db.executemany('UPDATE payload SET state=\'RESERVED\',attempts=attempts+1 WHERE key=?',[(key,) for key in keys])
                db.commit()
                return dict(status='RESERVED',effort=row[1],batch=dict(batch_id=bid,records=[json.loads(item[2]) for item in payloads]),
                    prompt=db.execute("SELECT value FROM meta WHERE key='prompt'").fetchone()[0],attempt=1,
                    preserved_failed_attempt=True)
            if kind in ('finish','retry'):
                bid = request['batch_id']
                row = db.execute('SELECT keys,effort,state,attempt FROM batch WHERE id=?',(bid,)).fetchone()
                assert row and row[2]=='RESERVED' and row[3]==request['attempt']
                keys,effort,_,number = json.loads(row[0]),row[1],row[2],row[3]
                evidence = request['evidence']
                assert evidence.get('reasoning_effort')==effort
                assert evidence.get('actual_model')=='gpt-6-astra' and evidence.get('protocol')==CFG['scoring']['protocol']
                response = request.get('response')
                db.execute('INSERT INTO attempt VALUES (?,?,?,?)',(bid,number,json.dumps(evidence),json.dumps(response)))
                if kind == 'retry':
                    assert number==0 and CFG['scoring']['transport_retries']==1 and transport_only(evidence)
                    assert request['no_valid_response'] is True and response is None
                    charge(db,len(keys))
                    db.execute('UPDATE batch SET attempt=1 WHERE id=?',(bid,))
                    db.executemany('UPDATE payload SET attempts=attempts+1 WHERE key=?',[(k,) for k in keys])
                elif evidence.get('status')=='FORMAT_VALID':
                    assert evidence.get('exit_code')==0 and not evidence.get('errors') and not evidence.get('tool_event_types')
                    assert evidence.get('isolation_checks') and all(evidence['isolation_checks'].values())
                    decisions = response['decisions']
                    assert response['batch_id']==bid and [x['opaque_query_id'] for x in decisions]==keys
                    assert all(set(x)=={'opaque_query_id','is_correct'} and type(x['is_correct']) is bool for x in decisions)
                    db.executemany('UPDATE payload SET state=\'FORMAT_VALID\',correct=? WHERE key=?',[(int(x['is_correct']),x['opaque_query_id']) for x in decisions])
                    db.execute('UPDATE batch SET state=\'FORMAT_VALID\' WHERE id=?',(bid,))
                else:
                    db.executemany('UPDATE payload SET state=\'MISSING\' WHERE key=?',[(k,) for k in keys])
                    db.execute('UPDATE batch SET state=\'MISSING\' WHERE id=?',(bid,))
                db.commit()
                return dict(status='RECORDED',summary=snapshot(db))
            return dict(status='READY',summary=snapshot(db))
        finally:
            db.close()
