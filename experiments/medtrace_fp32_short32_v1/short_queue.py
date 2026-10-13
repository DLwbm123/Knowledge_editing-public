"""Score the fixed RAW32 source endpoint once, then decide the competence gate."""
import os
import sys
import json
import fcntl
import sqlite3
import time
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_raw32_astra_medium'
q.BATCH_LEDGER='Astra_RAW32_batches'


def identity(row,output):return (row['query_id'],row['image_sha256'],row['question'],row['reference'],output['raw_answer'])


def gate(native,gains):
    assert len(native)==len(gains)==8
    if any(x is None for x in native):return 'INCOMPLETE_RAW32_NATIVE_EVIDENCE'
    if not any(x==1 for x in native):return 'STOP_RAW32_NO_NATIVE_REPAIR'
    if not all(x>1e-8 for x in gains):return 'STOP_RAW32_SOURCE_GAIN_NOT_POSITIVE'
    return 'RAW32_NATIVE_REPAIR_PRESENT'


def selfcheck():
    assert gate([0]*8,[1.]*8)=='STOP_RAW32_NO_NATIVE_REPAIR'
    assert gate([None]+[0]*7,[1.]*8)=='INCOMPLETE_RAW32_NATIVE_EVIDENCE'
    assert gate([1]+[0]*7,[1.]*8)=='RAW32_NATIVE_REPAIR_PRESENT'
    assert gate([1]+[0]*7,[0.]+[1.]*7)=='STOP_RAW32_SOURCE_GAIN_NOT_POSITIVE'


def ingest(db):
    selfcheck()
    lock=dict(epoch='MEDTRACE_FP32_RAW32_20261009_V1',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=1,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/SHORT32_LOCK.json')))
    manifest=q.ROOT/'EPOCH_MANIFEST.json'
    if manifest.exists():assert q.read(manifest)==lock
    else:q.write(manifest,lock)
    if not (RUN/'private/RAW32_GENERATION_COMPLETE.json').exists() or db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    parent=Path(q.read(RUN/'private/LAUNCH_ENV.json')['GENERATION_PARENT'])
    known={}
    for r in q.read(parent/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if r['arm']!='BASE':continue
        d=q.read(r['path']);assert r['correct'] in (0,1);known[identity(d['binding']['input'],d['R0'])]=r['correct']
    inherited=[]
    for path in sorted((RUN/'private/outputs/RAW32').glob('*/*.json')):
        d=q.read(path);row=d['binding']['input'];b=d['binding']['judge_input'];assert d['lock']==lock['scientific_lock']
        key=q.payload(db,row,b,d['R0']);old=known.get(identity(row,d['R0']))
        if old is not None:
            db.execute("UPDATE payload SET status='FORMAT_VALID',correct=? WHERE key=?",(old,key));inherited.append(key)
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),'RAW32','source_edit',1,d['expert_order'],'NATIVE' if d['index']==0 else 'FIT',row['query_id'],str(path),q.digest(d),key))
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==40
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=40,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited=len(inherited),new_pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]))


def report(db):
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    rows=[]
    for r in db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        assert r['status'] in ('FORMAT_VALID','MISSING');d=q.read(r['path']);rows.append(dict(order=r['edit_order'],index=d['index'],correct=r['correct'] if r['status']=='FORMAT_VALID' else None,path=r['path']))
    assert len(rows)==40
    source=[q.read(RUN/'private/raw_results'/f'{i}.json') for i in range(1,9)]
    native=[next(r['correct'] for r in rows if r['order']==i and r['index']==0) for i in range(1,9)]
    decision=gate(native,[s['edit_gain'] for s in source])
    panels={name:dict(queries=len(rs),correct=sum(r['correct']==1 for r in rs),missing=sum(r['correct'] is None for r in rs)) for name,rs in [('NATIVE',[r for r in rows if r['index']==0]),('FIT',[r for r in rows if r['index']>0])]}
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision,panels=panels,baseline_native_correct=0,baseline_native_wrong=8,
        per_expert=[dict(order=i,native_correct=native[i-1],edit_gain=source[i-1]['edit_gain'],all_four_cores_updated=source[i-1]['all_four_cores_updated'],mechanical=source[i-1]['mechanical']) for i in range(1,9)],
        all_experts_preserved=True,heldout_evaluated=False,protected_training_started=False,
        generation=q.read(RUN/'private/RAW32_GENERATION_COMPLETE.json'),queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    q.write(RUN/'private/RAW32_SEMANTIC.json',rows);q.write(RUN/'public/RAW32_RESULTS.json',result)
    paths=[RUN/'private/raw_candidates'/f'{i}.pt' for i in range(1,9)]
    assert all(p.is_file() and not p.is_symlink() and p.resolve().parent==(RUN/'private/raw_candidates').resolve() for p in paths)
    size=sum(p.stat().st_size for p in paths)
    if decision=='RAW32_NATIVE_REPAIR_PRESENT':
        lifecycle=dict(status='RETAINED_FOR_REGISTERED_PROTECTED_COMPARISON',packages=8,bytes=size)
    else:
        q.write(RUN/'private/RAW32_DELETION_PLAN.json',dict(paths=[str(p) for p in paths],bytes=size,reason=decision))
        for p in paths:p.unlink()
        lifecycle=dict(status='COMPLETE',packages_deleted=8,bytes_deleted=size,historical_artifacts_deleted=0)
    q.write(RUN/'public/RAW32_LIFECYCLE.json',lifecycle)
    q.write(RUN/'private/RAW32_REPORT_COMPLETE.json',dict(status='COMPLETE',decision=decision,epoch=time.time()))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"report"}' if os.environ.get('ACTION')=='raw32_report' else '{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        if op['action']=='report':report(db);answer={'status':'REPORT_COMPLETE'}
        elif op['action']=='ingest':answer=q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status':'GENERATING'}
        else:answer=a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
