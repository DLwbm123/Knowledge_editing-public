"""Score every frozen consumer once, reusing only complete accepted payload identities."""
import os
import sys
import json
import time
import fcntl
import sqlite3
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_scope_astra_medium'
q.BATCH_LEDGER='Astra_scope_batches'


def full_identity(d,judge):
    row=d['binding']['input'];b=d['binding']['judge_input']
    assert all(row[k]==b[k] for k in ('question','reference','image_sha256'))
    return dict(query_id=row['query_id'],question=row['question'],reference=row['reference'],image_sha256=row['image_sha256'],
        image_path=b['image_path'],prompt_ids=b['prompt_ids'],attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=d['R0'],judge=judge)


def selfcheck():
    import copy
    judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
    row=dict(query_id='test',question='Q',reference='A',image_sha256='image')
    binding=dict(row,image_path='/private/example',prompt_ids=[1],attention_mask=[1],runtime={'precision':'FP32'},generation={'max_new_tokens':1024})
    d=dict(binding=dict(input=row,judge_input=binding),R0=dict(raw_answer='A',raw_token_ids=[1,2]))
    identity=q.digest(full_identity(d,judge))
    for field,value in [('runtime',{'precision':'FP16'}),('prompt_ids',[2]),('image_path','/private/other')]:
        altered=copy.deepcopy(d);altered['binding']['judge_input'][field]=value
        assert q.digest(full_identity(altered,judge))!=identity
    changed=copy.deepcopy(d);changed['R0']['raw_token_ids']=[3,2]
    assert q.digest(full_identity(changed,judge))!=identity
    return dict(status='PASS',runtime_prompt_image_path_and_token_changes_not_reused=True)


def ingest(db):
    checks=selfcheck();env=q.read(RUN/'private/LAUNCH_ENV.json')
    lock=dict(epoch='MEDTRACE_SCOPE_CD160_20261010_V1',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=4,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/SCOPE_LOCK.json')),
        maximum_new_Judge=2567,inheritance='EXACT_FULL_PAYLOAD_IDENTITY_ONLY',selfcheck=checks)
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    assert (RUN/'private/SCOPE_GENERATION_COMPLETE.json').exists()
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    judge={k:lock[k] for k in ('model','reasoning_effort','protocol','prompt')};known={};sources=[]
    recovery=q.read(Path(env['GENERATION_PARENT'])/'private/judge_generation_recovery1/AUTHORIZATION.json')
    assert recovery['epoch']=='MEDTRACE_CONSTRAINED_RECOVERY1_20261009_V1' and recovery['payloads']==40
    authorized_recovery=set(recovery['keys'])
    for key in ('PROTECTION_PARENT','DAMAGE_PARENT','GENERATION_PARENT','FP32_BASELINE_PARENT','OPTIMIZER160_PARENT'):
        for path in sorted((Path(env[key])/'private').glob('judge*/queue.sqlite')):
            with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as previous:
                count=0
                for payload,binding,status,correct in previous.execute("SELECT key,binding,status,correct FROM payload WHERE status IN ('FORMAT_VALID','MISSING')"):
                    b=json.loads(binding)
                    if b['judge']!=judge:continue
                    assert q.digest(b)==payload
                    value=(status,correct)
                    if payload in known and known[payload]!=value:
                        assert payload in authorized_recovery and key=='GENERATION_PARENT'
                        assert path.parent.name=='judge_generation_recovery1' and known[payload]==('MISSING',None) and status=='FORMAT_VALID'
                    known[payload]=value;count+=1
            sources.append(dict(path=str(path),accepted_or_permanent_missing=count))
    paths=sorted((RUN/'private/outputs').glob('*/*/*.json'));assert len(paths)==2567
    inherited=0;missing=0
    for path in paths:
        d=q.read(path);row=d['binding']['input']
        if not d.get('alias_of'):assert d['lock']==lock['scientific_lock']
        else:
            parent=q.read(d['alias_of'])
            assert full_identity(parent,judge)==full_identity(d,judge),'Alias input/output identity changed'
        key=q.payload(db,row,d['binding']['judge_input'],d['R0']);assert key==q.digest(full_identity(d,judge))
        if key in known:
            status,score=known[key];db.execute('UPDATE payload SET status=?,correct=? WHERE key=?',(status,score,key))
            inherited+=status=='FORMAT_VALID';missing+=status=='MISSING'
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),d['arm'],'natural' if d['arm'].startswith('NATURAL_') else 'forced_owner',
            8 if d['arm'].startswith('NATURAL_') else 1,d['expert_order'],d['role'],row['query_id'],str(path),q.digest(d),key))
    pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0];assert pending<=2567
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'REUSE_SOURCES.json',sources)
    q.write(q.ROOT/'READY.json',dict(consumers=2567,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],
        inherited_consumers=inherited,permanent_missing_consumers=missing,new_pending=pending,epoch=time.time()))


def request(db,op):
    if op['action']=='reserve':
        before=q.read(RUN/'private/INHERITED_COST.json')['Judge_attempts']
        assert q.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts']+len(op['keys'])-before<=2567
    return a.request(db,op)


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        if op['action']=='ingest':answer=q.read(q.ROOT/'READY.json')
        else:answer=request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
