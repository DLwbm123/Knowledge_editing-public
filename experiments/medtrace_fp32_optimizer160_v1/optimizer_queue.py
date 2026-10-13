"""Score both fixed optimization endpoints, retaining paired source evidence."""
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
q.ROOT=RUN/'private/judge_optimizer160_astra_medium'
q.BATCH_LEDGER='Astra_optimizer160_batches'


def identity(row,output):return (row['query_id'],row['image_sha256'],row['question'],row['reference'],output['raw_answer'])


def decision_for(paired):
    assert len(paired)==8
    if any(r[a] is None for r in paired for a in ('RAW','ADAM')):return 'INCOMPLETE_NATIVE_SCORING'
    repaired={a:any(r[a]==1 for r in paired) for a in ('RAW','ADAM')}
    return {(False,False):'NO_NATIVE_REPAIR_AT_FIXED160',(True,False):'RAW_ONLY_NATIVE_REPAIR',
        (False,True):'ADAM_ONLY_NATIVE_REPAIR',(True,True):'BOTH_HAVE_NATIVE_REPAIR'}[repaired['RAW'],repaired['ADAM']]


def selfcheck():
    for values,expected in [((0,0),'NO_NATIVE_REPAIR_AT_FIXED160'),((1,0),'RAW_ONLY_NATIVE_REPAIR'),
            ((0,1),'ADAM_ONLY_NATIVE_REPAIR'),((1,1),'BOTH_HAVE_NATIVE_REPAIR'),((None,1),'INCOMPLETE_NATIVE_SCORING')]:
        assert decision_for([dict(RAW=values[0],ADAM=values[1])]+[dict(RAW=0,ADAM=0) for _ in range(7)])==expected


def ingest(db):
    selfcheck()
    lock=dict(epoch='MEDTRACE_FP32_OPTIMIZER160_20261009_V1',model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
        batch_size=50,workers=2,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/OPTIMIZER160_LOCK.json')))
    manifest=q.ROOT/'EPOCH_MANIFEST.json'
    if manifest.exists():assert q.read(manifest)==lock
    else:q.write(manifest,lock)
    if not (RUN/'private/OPTIMIZER160_GENERATION_COMPLETE.json').exists() or db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    parent=Path(q.read(RUN/'private/LAUNCH_ENV.json')['GENERATION_PARENT'])
    known={}
    for r in q.read(parent/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if r['arm']!='BASE':continue
        d=q.read(r['path']);assert r['correct'] in (0,1);known[identity(d['binding']['input'],d['R0'])]=r['correct']
    short=Path(q.read(RUN/'private/LAUNCH_ENV.json')['SHORT32_PARENT'])
    for r in q.read(short/'private/RAW32_SEMANTIC.json'):
        d=q.read(r['path']);key=identity(d['binding']['input'],d['R0']);assert r['correct'] in (0,1)
        assert key not in known or known[key]==r['correct']
        known[key]=r['correct']
    inherited=[]
    for path in sorted((RUN/'private/outputs').glob('*/*/*.json')):
        d=q.read(path);row=d['binding']['input'];b=d['binding']['judge_input'];assert d['lock']==lock['scientific_lock']
        key=q.payload(db,row,b,d['R0']);old=known.get(identity(row,d['R0']))
        if old is not None:
            db.execute("UPDATE payload SET status='FORMAT_VALID',correct=? WHERE key=?",(old,key));inherited.append(key)
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),d['arm'],'source_edit',1,d['expert_order'],'NATIVE' if d['index']==0 else 'FIT',row['query_id'],str(path),q.digest(d),key))
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==80
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=80,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],inherited=len(inherited),new_pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]))


def report(db):
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    rows=[]
    for r in db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        assert r['status'] in ('FORMAT_VALID','MISSING');d=q.read(r['path'])
        rows.append(dict(arm=d['arm'],order=r['edit_order'],index=d['index'],correct=r['correct'] if r['status']=='FORMAT_VALID' else None,path=r['path']))
    assert len(rows)==80
    panels={};source=[]
    parent=Path(q.read(RUN/'private/LAUNCH_ENV.json')['GENERATION_PARENT'])
    before_scores={r['path']:r['correct'] for r in q.read(parent/'private/EDIT_BASELINE_QUALIFICATION.json')['rows'] if r['arm']=='BASE'}
    for arm in ('RAW','ADAM'):
        panels[arm]={}
        for name,rs in [('NATIVE',[r for r in rows if r['arm']==arm and r['index']==0]),('FIT',[r for r in rows if r['arm']==arm and r['index']>0])]:
            transitions=dict(retained=0,new_damage=0,repaired=0,still_wrong=0,missing=0,same_text=0,changed_text=0)
            for r in rs:
                d=q.read(r['path']);b=q.read(d['baseline_path']);before=before_scores[d['baseline_path']];after=r['correct']
                transitions['same_text' if d['R0']['raw_answer']==b['R0']['raw_answer'] else 'changed_text']+=1
                transitions['missing' if after is None else (('retained' if after else 'new_damage') if before else ('repaired' if after else 'still_wrong'))]+=1
            panels[arm][name]=dict(queries=len(rs),correct=sum(r['correct']==1 for r in rs),missing=sum(r['correct'] is None for r in rs),transitions=transitions)
        for i in range(1,9):
            s=q.read(RUN/'private/results'/arm/f'{i}.json')
            checkpoints=[]
            for node in s['checkpoints']:
                values=[]
                for v in node['values']:
                    td=v['token_diagnostics']
                    values.append(dict(NLL=v['NLL'],NLL_change=v['NLL_change'],tokens=v['tokens'],correct_tokens=v['correct_tokens'],all_argmax_correct=v['all_argmax_correct'],
                        wrong_tokens=td['wrong_tokens'],first_wrong_index=td['first_wrong_index'],minimum_margin=td['minimum_margin'],mean_margin=td['mean_margin']))
                checkpoints.append(dict(step=node['step'],values=values))
            source.append(dict(arm=arm,order=i,edit_gain=s['edit_gain'],all_four_cores_updated=s['all_four_cores_updated'],
                mechanical_status=s['mechanical']['status'],terminal_prefix_status=s['terminal_prefix_check']['status'],
                terminal_prefix_max_difference=s['terminal_prefix_check']['maximum_logit_difference'],checkpoints=checkpoints,
                native_correct=next(r['correct'] for r in rows if r['arm']==arm and r['order']==i and r['index']==0)))
    paired=[]
    for i in range(1,9):
        values={a:next(r['correct'] for r in rows if r['arm']==a and r['order']==i and r['index']==0) for a in ('RAW','ADAM')}
        paired.append(dict(order=i,**values))
    decision=decision_for(paired)
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==6 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision,panels=panels,paired_native=paired,per_expert=source,
        baseline_native_correct=0,baseline_FIT_correct=2,all_experts_preserved=True,heldout_evaluated=False,protected_training_started=False,
        automatic_next_experiment=False,generation=q.read(RUN/'private/OPTIMIZER160_GENERATION_COMPLETE.json'),
        queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    q.write(RUN/'private/OPTIMIZER160_SEMANTIC.json',rows);q.write(RUN/'public/RESULTS.json',result)
    assert len(list((RUN/'private').glob('OPTIMIZER160_CONSUMED_*.json')))==16
    assert not list((RUN/'private/results').rglob('*.pt')) and not list((RUN/'private/outputs').rglob('*.pt'))
    q.write(RUN/'public/LIFECYCLE.json',dict(status='COMPLETE',trained_experts_consumed=16,persistent_checkpoints_created=0,
        weights_retained=0,all_registered_consumers_completed=True,historical_artifacts_deleted=0,reconstruction_requires_new_training=True))
    q.write(RUN/'private/OPTIMIZER160_REPORT_COMPLETE.json',dict(status='COMPLETE',decision=decision,epoch=time.time()))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"report"}' if os.environ.get('ACTION')=='optimizer160_report' else '{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;ingest(db)
        if op['action']=='report':report(db);answer={'status':'REPORT_COMPLETE'}
        elif op['action']=='ingest':answer=q.read(q.ROOT/'READY.json') if (q.ROOT/'READY.json').exists() else {'status':'GENERATING'}
        else:answer=a.request(db,op)
        if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
