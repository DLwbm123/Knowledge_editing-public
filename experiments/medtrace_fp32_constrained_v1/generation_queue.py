"""One-attempt Astra queue with explicit exact-text qualification inheritance."""
import os
import json
import fcntl
import time
import sys
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_generation_astra_medium'
q.BATCH_LEDGER='Astra_constrained_generation_batches'


def semantic_key(row,output):
    return q.digest([row['query_id'],row['image_sha256'],row['question'],row['reference'],output['raw_answer']])


def selfcheck():
    r=dict(query_id='a',image_sha256='i',question='q',reference='g');out=dict(raw_answer='yes',raw_token_ids=[1])
    assert semantic_key(r,out)==semantic_key(r,dict(out,raw_token_ids=[2]))
    assert semantic_key(r,out)!=semantic_key(dict(r,question='other'),out)
    assert semantic_key(r,out)!=semantic_key(r,dict(out,raw_answer='no'))


def ingest(db):
    selfcheck()
    lock=dict(epoch='MEDTRACE_CONSTRAINED_GENERATION_20261009_V1',model='gpt-6-astra',reasoning_effort='medium',
        protocol=q.PROTOCOL,prompt=q.PROMPT,batch_size=50,workers=4,attempts_per_payload=1,
        scientific_lock=q.digest(q.read(RUN/'private/GEN_LOCK.json')))
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    if not (RUN/'private/GENERATION_COMPLETE.json').exists() or db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    known={}
    baseline=q.read(RUN/'private/LAUNCH_ENV.json')['FP32_BASELINE_PARENT']
    qualification=q.read(Path(baseline)/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    for r in qualification:
        d=q.read(r['path']);key=semantic_key(d['binding']['input'],d['R0'])
        assert r['correct'] in (0,1);known[key]=dict(correct=r['correct'],path=r['path'])
        old=q.read(d['identity_parent']);key=semantic_key(old['binding']['input'],old['R0'])
        if key in known:assert known[key]['correct']==int(r['previously_correct'])
        known[key]=dict(correct=int(r['previously_correct']),path=d['identity_parent'])
    inherited=[]
    for path in sorted((RUN/'private/outputs').glob('*/*/*.json')):
        d=q.read(path);row=d['binding']['input'];b=d['binding']['judge_input']
        assert d['lock']==lock['scientific_lock']
        key=q.payload(db,row,b,d['R0']);semantic=semantic_key(row,d['R0'])
        if semantic in known:
            old=known[semantic];current=db.execute('SELECT status,correct FROM payload WHERE key=?',(key,)).fetchone()
            assert current['status']=='PENDING' or current['correct']==old['correct']
            db.execute("UPDATE payload SET status='FORMAT_VALID',correct=? WHERE key=?",(old['correct'],key))
            inherited.append(dict(path=str(path),parent=old['path'],payload=key,correct=old['correct']))
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (q.digest(str(path)),d['arm'],'forced_frozen_candidate',1,d['expert_order'],d['role'],row['query_id'],str(path),q.digest(d),key))
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==2464
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'TEXT_INHERITANCE.json',inherited)
    q.write(q.ROOT/'READY.json',dict(consumers=2464,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],
        pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0],inherited_consumers=len(inherited)))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        answer=(q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status':'GENERATING'}) if op['action']=='ingest' else a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
