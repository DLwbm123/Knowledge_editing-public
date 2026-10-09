"""One fixed protection comparison; accepted semantic reuse and no retries."""
import os
import sys
import json
import time
import fcntl
import sqlite3
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
import damage_queue as audit_helpers
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_protection_astra_medium';q.BATCH_LEDGER='Astra_protection_batches'
identity=audit_helpers.identity
summary=audit_helpers.summary


def decide(complete,source_ok,generalization_ok,less_damage,no_owner_worse,primary_nonworse,intervals_positive):
    if not complete:return 'INCOMPLETE_SCORING'
    if not source_ok or not generalization_ok:return 'EDIT_OR_GENERALIZATION_NOT_PRESERVED'
    if not less_damage or not no_owner_worse or not primary_nonworse:return 'NO_QUALIFIED_PROTECTION_ADVANTAGE'
    return 'DEVELOPMENT_PROTECTION_SIGNAL' if intervals_positive else 'PROTECTION_POINT_SIGNAL_INCONCLUSIVE'


def selfcheck():
    audit_helpers.selfcheck()
    assert decide(False,True,True,True,True,True,True)=='INCOMPLETE_SCORING'
    assert decide(True,True,False,True,True,True,True)=='EDIT_OR_GENERALIZATION_NOT_PRESERVED'
    assert decide(True,True,True,False,True,True,True)=='NO_QUALIFIED_PROTECTION_ADVANTAGE'
    assert decide(True,True,True,True,False,True,True)=='NO_QUALIFIED_PROTECTION_ADVANTAGE'
    assert decide(True,True,True,True,True,True,False)=='PROTECTION_POINT_SIGNAL_INCONCLUSIVE'
    assert decide(True,True,True,True,True,True,True)=='DEVELOPMENT_PROTECTION_SIGNAL'


def ingest(db):
    selfcheck();lock=dict(epoch='MEDTRACE_FP32_PROTECTION160_20261009_V1',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=4,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/PROTECTION_LOCK.json')))
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    if not (RUN/'private/PROTECTION_GENERATION_COMPLETE.json').exists() or db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    env=q.read(RUN/'private/LAUNCH_ENV.json');parent=Path(env['DAMAGE_PARENT']);known={}
    def add(row,out,correct):
        key=identity(row,out);assert correct in (0,1)
        assert key not in known or known[key]==correct,'Conflicting accepted semantic labels'
        known[key]=correct
    with sqlite3.connect('file:'+str(parent/'private/judge_damage_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as old:
        for binding,correct in old.execute("SELECT binding,correct FROM payload WHERE status='FORMAT_VALID'"):
            b=json.loads(binding);assert b['judge']=={k:lock[k] for k in ('model','reasoning_effort','protocol','prompt')}
            add(b,b['output'],correct)
    for item in q.read(RUN/'private/HELD_BASE.json').values():
        d=q.read(item['path']);add(d['binding']['input'],d['R0'],item['correct'])
    for path in (parent/'private/outputs/RAW').glob('*/*.json'):
        d=q.read(path);add(d['binding']['input'],d['R0'],1)
    gen=Path(env['GENERATION_PARENT'])
    for item in q.read(gen/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if item['arm']=='BASE':
            d=q.read(item['path']);add(d['binding']['input'],d['R0'],item['correct'])
    inherited=0;paths=list(sorted((RUN/'private/outputs/PROTECTED').glob('*/*.json')));assert len(paths)==923
    for path in paths:
        d=q.read(path);assert d['lock']==lock['scientific_lock'];row=d['binding']['input']
        key=q.payload(db,row,d['binding']['judge_input'],d['R0']);score=known.get(identity(row,d['R0']))
        if score is not None:db.execute("UPDATE payload SET status='FORMAT_VALID',correct=? WHERE key=?",(score,key));inherited+=1
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),'PROTECTED','forced_owner',1,d['expert_order'],d['role'],row['query_id'],str(path),q.digest(d),key))
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=923,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited_consumers=inherited,new_pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]))


def report(db):
    import statistics
    import probe_report as stats
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists();selfcheck()
    env=q.read(RUN/'private/LAUNCH_ENV.json');parent=Path(env['DAMAGE_PARENT']);reference=q.read(parent/'public/RESULTS.json')
    scores={item['path']:item['correct'] for item in q.read(parent/'private/DAMAGE_SEMANTIC.json')}
    bases=q.read(RUN/'private/HELD_BASE.json');scores.update({v['path']:v['correct'] for v in bases.values()})
    for item in q.read(Path(env['GENERATION_PARENT'])/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if item['arm']=='BASE':scores[item['path']]=item['correct']
    rows=[]
    for r in db.execute('SELECT c.path,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        assert r['status'] in ('FORMAT_VALID','MISSING');d=q.read(r['path'])
        rows.append(dict(path=r['path'],output=d,correct=r['correct'] if r['status']=='FORMAT_VALID' else None))
    assert len(rows)==923
    def selected(role,owner=None):
        result=[]
        for r in rows:
            d=r['output'];match=d['role']==role
            if role=='ORIGINAL_PRIMARY':match=d['role']=='HELDOUT' and d['original_primary']
            if role=='FP32_QUALIFIED':match=d['role']=='HELDOUT' and scores[d['baseline_path']]==1
            if role.startswith('T1L_'):match=d['role']=='T1L' and d['same_reference']==(role=='T1L_SAME_REFERENCE')
            if match and (owner is None or owner==d['expert_order']):result.append(r)
        return result
    def panel(rs):
        return summary([(scores[r['output']['baseline_path']],r['correct'],r['output']['R0']['raw_answer']==q.read(r['output']['baseline_path'])['R0']['raw_answer']) for r in rs])
    roles=('NATIVE','FIT','T1G','T2G','T1L','T2L','HELDOUT','ORIGINAL_PRIMARY','FP32_QUALIFIED','T1L_SAME_REFERENCE','T1L_DIFFERENT_REFERENCE')
    panels={role:panel(selected(role)) for role in roles}
    owners=[dict(owner=i,panels={role:panel(selected(role,i)) for role in roles}) for i in range(1,9)]
    source=[q.read(RUN/'private/source_results'/f'{i}.json') for i in range(1,9)]
    source_ok=panels['NATIVE']['candidate_correct']==8 and panels['FIT']['candidate_correct']==32 and all(s['endpoint']['gain_ratio']>=.9 for s in source)
    raw_t2=q.read(RUN/'private/PROTECTION_LOCK.json')['baseline_T2G_by_owner']
    generalization_ok=panels['T1G']['candidate_correct']==32 and all(v['panels']['T2G']['candidate_correct']>=raw_t2[v['owner']-1] for v in owners)
    raw_damage=[next(v['new_damage'] for v in reference['per_expert'] if v['arm']=='RAW' and v['role']=='HELDOUT' and v['owner']==i) for i in range(1,9)]
    complete=all(r['correct'] is not None for r in rows);contrast=None
    if complete:
        primary=selected('ORIGINAL_PRIMARY');deltas=[]
        for row in primary:
            d=row['output'];query=d['binding']['input']['query_id'];oldpath=parent/'private/audit_outputs/RAW'/str(d['expert_order'])/(query+'.json')
            deltas.append(dict(owner=d['expert_order'],group=d['binding']['input']['source_group'],difference=row['correct']-scores[str(oldpath)]))
        assert len(deltas)==504 and len({r['group'] for r in deltas})==29
        contrast=dict(mean=stats.mean([r['difference'] for r in deltas]),
            expert_CI=stats.interval([stats.mean([r['difference'] for r in deltas if r['owner']==i]) for i in range(1,9)]),
            source_CI=stats.interval([stats.mean([r['difference'] for r in deltas if r['group']==g]) for g in sorted({r['group'] for r in deltas})]),
            limitation='Two separate cluster summaries; not independent paired observations or external confirmation')
    less=panels['FP32_QUALIFIED']['new_damage']<reference['panels']['RAW']['FP32_QUALIFIED']['new_damage']
    none_worse=all(v['panels']['HELDOUT']['new_damage']<=raw_damage[v['owner']-1] for v in owners)
    primary_nonworse=panels['ORIGINAL_PRIMARY']['candidate_correct']>=reference['panels']['RAW']['ORIGINAL_PRIMARY']['candidate_correct']
    positive=bool(contrast and contrast['expert_CI'][0]>0 and contrast['source_CI'][0]>0)
    decision=decide(complete,source_ok,generalization_ok,less,none_worse,primary_nonworse,positive)
    kl=[]
    for i in range(1,9):
        d=q.read(RUN/'private/held_diagnostics'/f'{i}.json')
        for label,mask in [('ALL96',[True]*96),('ORIGINAL63',d['original_primary'])]:
            vs=[v for v,m in zip(d['values'],mask) if m]
            kl.append(dict(owner=i,panel=label,queries=len(vs),KL=statistics.mean(v['KL'] for v in vs),content_KL=statistics.mean(v['content']['KL'] for v in vs)))
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==18 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision,panels=panels,per_owner=owners,reference_panels=reference['panels'],
        source_gate=source_ok,generalization_gate=generalization_ok,less_damage=less,no_owner_worse=none_worse,primary_nonworse=primary_nonworse,
        primary_contrast=contrast,source_endpoints=[dict(owner=s['order'],**s['endpoint']) for s in source],held_prefix_diagnostics=kl,
        protected_step_fallbacks=sum(v['mode']=='RAW_NONPOSITIVE_GAIN_FALLBACK' for s in source for v in s['curve']),
        restored_steps=sum(v['mode']=='PROTECTED_OR_RESTORED' and v['restoration_alpha']>0 for s in source for v in s['curve']),
        queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        missing=sum(r['correct'] is None for r in rows),at_cap=sum(r['output']['at_cap'] for r in rows),
        generation=q.read(RUN/'private/PROTECTION_GENERATION_COMPLETE.json'),automatic_next_experiment=False,independent_confirmation=False,
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    assert result['resource']['new_Judge']<=923
    q.write(RUN/'private/PROTECTION_SEMANTIC.json',[dict(path=r['path'],correct=r['correct']) for r in rows])
    q.write(RUN/'public/RESULTS.json',result);q.write(RUN/'private/PROTECTION_REPORT_COMPLETE.json',dict(status='COMPLETE',decision=decision,epoch=time.time()))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"report"}' if os.environ.get('ACTION')=='protection_report' else '{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        if op['action']=='report':report(db);answer={'status':'REPORT_COMPLETE'}
        elif op['action']=='ingest':answer=q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status':'GENERATING'}
        else:answer=a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
