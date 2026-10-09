"""Requalify only changed FP32 answers without changing original panel membership."""
import os
import sys
import json
import fcntl
import time
from pathlib import Path
sys.path.insert(0, os.environ["RUN_ROOT"]+"/private/tools")
import astra_queue as a
q, RUN = a.q, a.RUN
q.ROOT = RUN/'private/judge_fp32_astra_medium'
q.BATCH_LEDGER = 'Astra_FP32_requalification_batches'
EPOCH = 'MEDTRACE_FP32_REQUALIFICATION_20261009_V1'


def eligible(rows):
    return all(r['correct']==1 for r in rows if r['previously_correct'])


def selfcheck():
    assert eligible([dict(previously_correct=True,correct=1),dict(previously_correct=False,correct=0)])
    assert not eligible([dict(previously_correct=True,correct=None)])
    assert not eligible([dict(previously_correct=True,correct=0)])


def ingest(db):
    selfcheck()
    result=q.read(RUN/'public/RESULTS.json')
    assert result['decision']=='BASELINE_REQUALIFICATION_REQUIRED' and result['changed_texts']==6
    lock=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,
        prompt=q.PROMPT,batch_size=50,workers=1,attempts_per_payload=1,max_new_Judge=6,
        baseline_source=q.read(RUN/'private/GPU_SOURCE_VERSION.json'),
        qualification_source=q.read(RUN/'private/REQUALIFICATION_SOURCE.json'))
    manifest=q.ROOT/'EPOCH_MANIFEST.json'
    if manifest.exists():assert q.read(manifest)==lock
    else:q.write(manifest,lock)
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM payload').fetchone()[0]==0
    changed=[]
    for path in sorted((RUN/'private/outputs').glob('*.json')):
        d=q.read(path)
        if d['identity']['text_equal']:continue
        old=q.read(d['identity_parent'])
        assert d['binding']['input']==old['binding']['input'] and d['R0']['raw_answer']!=old['R0']['raw_answer']
        b=dict(d['binding']['judge_input'])
        b['runtime']=dict(inherited_runtime=b['runtime'],actual_precision=d['precision'],frozen_FP16_prefill=True)
        key=q.payload(db,d['binding']['input'],b,d['R0'])
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
            (q.digest([d['identity']['index'],str(path)]),'FP32_Base','requalification',0,0,d['identity']['role'],
             d['binding']['input']['query_id'],str(path),q.digest(d),key))
        changed.append(d['identity']['index'])
    assert len(changed)==6 and db.execute('SELECT count(*) FROM payload').fetchone()[0]==6
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=6,payloads=6,inherited_unchanged=151))


def report(db):
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    scored={r['path']:dict(r) for r in db.execute('SELECT c.path,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key')}
    assert len(scored)==6 and all(r['status'] in ('FORMAT_VALID','MISSING') for r in scored.values())
    rows=[]
    for path in sorted((RUN/'private/outputs').glob('*.json')):
        d=q.read(path);i=d['identity']
        s=scored.get(str(path))
        assert (s is None)==i['text_equal']
        correct=(s['correct'] if s['status']=='FORMAT_VALID' else None) if s else int(i['previously_correct'])
        rows.append(dict(row=d['binding']['input'],role=i['role'],index=i['index'],path=str(path),
            previously_correct=i['previously_correct'],correct=correct,
            status=s['status'] if s else 'INHERITED_EXACT_TEXT',text_equal=i['text_equal']))
    assert len(rows)==157 and sum(r['previously_correct'] for r in rows)==124
    decision='ORIGINAL_QUALIFIED_PANELS_PRESERVED' if eligible(rows) else 'ORIGINAL_QUALIFICATION_NOT_PRESERVED'
    panels={}
    for role in ('BASIS','HELDOUT','PRIMARY63'):
        rs=[r for r in rows if r['role']==role or role=='PRIMARY63' and r['role']=='HELDOUT' and r['previously_correct']]
        panels[role]=dict(queries=len(rs),correct=sum(r['correct']==1 for r in rs),missing=sum(r['correct'] is None for r in rs),
            previously_correct_now_wrong=sum(r['previously_correct'] and r['correct']==0 for r in rs))
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');new=ledger['Judge_attempts']-12406;assert new==6
    q.write(RUN/'private/FP32_SEMANTIC_QUALIFICATION.json',dict(rows=rows,decision=decision,original_membership_preserved=True))
    q.write(RUN/'public/REQUALIFICATION_RESULTS.json',dict(status='COMPLETE',decision=decision,panels=panels,
        new_Judge=new,inherited_exact_text=151,payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        cumulative_Judge=ledger['Judge_attempts'],annotation_source_only=True,independent_confirmation=False))
    q.write(RUN/'private/REQUALIFICATION_COMPLETE.json',dict(status='COMPLETE',decision=decision,epoch=time.time()))


if __name__=='__main__':
    operation=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        if operation['action']=='report':report(db);answer={'status':'REPORT_COMPLETE'}
        elif operation['action']=='ingest':answer=q.read(q.ROOT/'READY.json')
        else:answer=a.request(db,operation)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
