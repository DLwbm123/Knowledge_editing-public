"""One-attempt Astra medium audit; fixed paired denominators including missing scores."""
import os
import sys
import json
import fcntl
import sqlite3
import time
import statistics
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_damage_astra_medium'
q.BATCH_LEDGER='Astra_damage_batches'


def identity(row,out):return (row['query_id'],row['image_sha256'],row['question'],row['reference'],out['raw_answer'])


def summary(pairs):
    result=dict(queries=len(pairs),Base_correct=0,candidate_correct=0,retained=0,new_damage=0,repaired=0,still_wrong=0,missing=0,same_text=0)
    for before,after,same in pairs:
        result['Base_correct']+=before==1;result['candidate_correct']+=after==1;result['same_text']+=same
        key='missing' if before is None or after is None else ('retained' if after else 'new_damage') if before else ('repaired' if after else 'still_wrong')
        result[key]+=1
    return result


def selfcheck():
    result=summary([(1,1,True),(1,0,False),(0,1,False),(0,0,True),(None,1,False),(1,None,False)])
    assert result==dict(queries=6,Base_correct=3,candidate_correct=3,retained=1,new_damage=1,repaired=1,still_wrong=1,missing=2,same_text=2)
    assert summary([])['queries']==0


def ingest(db):
    selfcheck();lock=dict(epoch='MEDTRACE_FP32_DAMAGE_20261009_V1',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=4,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/DAMAGE_LOCK.json')))
    manifest=q.ROOT/'EPOCH_MANIFEST.json'
    if manifest.exists():assert q.read(manifest)==lock
    else:q.write(manifest,lock)
    if not (RUN/'private/DAMAGE_GENERATION_COMPLETE.json').exists() or db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    known={};env=q.read(RUN/'private/LAUNCH_ENV.json')
    # Reuse only accepted matching Astra medium protocol and exact semantic input/output.
    source=Path(env['PAPER_PARENT'])/'private/judge_astra_medium/queue.sqlite'
    with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as previous:
        for binding,correct in previous.execute("SELECT binding,correct FROM payload WHERE status='FORMAT_VALID'"):
            b=json.loads(binding)
            if b['judge']!={k:lock[k] for k in ('model','reasoning_effort','protocol','prompt')}:continue
            key=identity(b,b['output']);assert key not in known or known[key]==correct;known[key]=correct
    base=q.read(RUN/'private/HELD_BASE.json')
    for item in base.values():
        d=q.read(item['path']);known[identity(d['binding']['input'],d['R0'])]=item['correct']
    paths=list(sorted((RUN/'private/audit_outputs').glob('*/*/*.json')));assert len(paths)==1881
    inherited=0
    for path in paths:
        d=q.read(path);assert d['lock']==lock['scientific_lock'];row=d['binding']['input']
        key=q.payload(db,row,d['binding']['judge_input'],d['R0']);score=known.get(identity(row,d['R0']))
        if score is not None:
            db.execute("UPDATE payload SET status='FORMAT_VALID',correct=? WHERE key=?",(score,key));inherited+=1
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),d['arm'],'forced_owner',1,d['expert_order'],d['role'],row['query_id'],str(path),q.digest(d),key))
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==1881
    assert db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]<=1881
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=1881,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited_consumers=inherited,new_pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]))


def report(db):
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists();selfcheck()
    scored=[]
    for r in db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        assert r['status'] in ('FORMAT_VALID','MISSING');d=q.read(r['path'])
        scored.append(dict(path=r['path'],correct=r['correct'] if r['status']=='FORMAT_VALID' else None,output=d))
    assert len(scored)==1881
    scores={r['path']:r['correct'] for r in scored}
    for item in q.read(RUN/'private/HELD_BASE.json').values():scores[item['path']]=item['correct']
    panels={};per_expert=[]
    for arm in ('RAW','ADAM'):
        panels[arm]={}
        for role in ('T1G','T2G','T1L','T2L','HELDOUT','ORIGINAL_PRIMARY','FP32_QUALIFIED','T1L_SAME_REFERENCE','T1L_DIFFERENT_REFERENCE'):
            chosen=[]
            for r in scored:
                d=r['output']
                if d['arm']!=arm:continue
                qualifies=d['role']==role
                if role=='ORIGINAL_PRIMARY':qualifies=d['role']=='HELDOUT' and d['original_primary']
                if role=='FP32_QUALIFIED':qualifies=d['role']=='HELDOUT' and scores[d['baseline_path']]==1
                if role.startswith('T1L_'):qualifies=d['role']=='T1L' and d['same_reference']==(role=='T1L_SAME_REFERENCE')
                if qualifies:chosen.append(r)
            def panel(rs):
                pairs=[]
                for r in rs:
                    d=r['output'];b=q.read(d['baseline_path']);pairs.append((scores[d['baseline_path']],r['correct'],d['R0']['raw_answer']==b['R0']['raw_answer']))
                return summary(pairs)
            panels[arm][role]=panel(chosen)
            for owner in range(1,9):
                rs=[r for r in chosen if r['output']['expert_order']==owner]
                if rs:per_expert.append(dict(arm=arm,role=role,owner=owner,**panel(rs)))
    kl=[]
    for arm in ('RAW','ADAM'):
        for owner in range(1,9):
            d=q.read(RUN/'private/held_diagnostics'/arm/f'{owner}.json')
            for label,mask in [('ALL96',[True]*96),('ORIGINAL63',d['original_primary'])]:
                selected=[v for v,m in zip(d['values'],mask) if m]
                kl.append(dict(arm=arm,owner=owner,panel=label,queries=len(selected),KL=statistics.mean(v['KL'] for v in selected),content_KL=statistics.mean(v['content']['KL'] for v in selected),NLL_change=statistics.mean(v['NLL_change'] for v in selected)))
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==12 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    candidates=[r['output'] for r in scored if r['output']['arm']!='BASE'];assert len(candidates)==1766
    missing=sum(r['correct'] is None for r in scored)
    result=dict(status='COMPLETE',decision='INCOMPLETE_SCORING' if missing else 'QUALIFIED_UNPROTECTED_REFERENCE_AUDITED',panels=panels,per_expert=per_expert,held_prefix_diagnostics=kl,
        source_rebuild=q.read(RUN/'private/CANDIDATES_FROZEN.json'),queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        missing_consumers=missing,candidate_at_cap=sum(d['at_cap'] for d in candidates),candidate_EOS=sum(d['EOS'] for d in candidates),
        source_native=dict(RAW=8,ADAM=8,denominator=8),source_FIT=dict(RAW=32,ADAM=32,denominator=32),
        limitations=['development cases only','forced owner expert, not routed bank','T2L only2 queries from owner1','original precision qualification63 to62 preserved','no automatic protection training'],
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    assert result['resource']['new_Judge']<=1881
    q.write(RUN/'private/DAMAGE_SEMANTIC.json',[dict(path=r['path'],correct=r['correct']) for r in scored])
    q.write(RUN/'public/RESULTS.json',result)
    q.write(RUN/'private/DAMAGE_REPORT_COMPLETE.json',dict(status='COMPLETE',decision=result['decision'],epoch=time.time()))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"report"}' if os.environ.get('ACTION')=='damage_report' else '{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        if op['action']=='report':report(db);answer={'status':'REPORT_COMPLETE'}
        elif op['action']=='ingest':answer=q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status':'GENERATING'}
        else:answer=a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
