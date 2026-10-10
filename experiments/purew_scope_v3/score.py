"""Score the completed frozen outputs once using the existing isolated Judge."""
import os
import sys
import json
import time
import fcntl
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
from damage_queue import summary

q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_scope_v3_astra_medium'
q.BATCH_LEDGER='Astra_scope_v3_batches'
EPOCH='PUREW_SCOPE_V3_20261010'
ARMS=('A_LOW','B_NORM','C_FULL_FIXED','D_FULL_REFRESH','E_RLS')
ROLES=('NATIVE','FIT','GFIT','T1G','T2G','T1L','T2L','HELDOUT')
PARENT=Path(q.read(RUN/'private/INHERITED_RUN.json')['root'])


def inherited(db):
    import sqlite3
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_batch TEXT)')
    prior={}
    for name,queue in (('purew-scope-v1-20261009','judge_purew_astra_medium'),('purew-scope-v2-20261010','judge_scope_v2_astra_medium')):
        root=Path('/data/bmw/Knowledge_editing/outputs')/name/'run/private'/queue
        old=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
        for r in old.execute('SELECT * FROM payload'):
            if r['batch'].startswith('INHERITED_'):continue
            assert r['status'] in ('FORMAT_VALID','MISSING')
            if r['key'] not in prior or r['status']=='FORMAT_VALID':prior[r['key']]=(dict(r),root)
        old.close()
    checked={}
    for current in list(db.execute('SELECT * FROM payload')):
        pair=prior.get(current['key'])
        if pair is None:continue
        old,root=pair
        assert json.loads(old['binding'])==json.loads(current['binding'])
        old_record=json.loads(old['record']);new_record=json.loads(current['record'])
        assert {k:v for k,v in old_record.items() if k!='opaque_query_id'}=={k:v for k,v in new_record.items() if k!='opaque_query_id'}
        bid=old['batch'];cache=(str(root),bid)
        if cache not in checked:
            saved=q.read(root/'evidence'/(bid+'.json'));ev=saved['evidence']
            assert ev['input_binding']==q.digest(saved['batch']) and old_record in saved['batch']['records']
            if old['status']=='FORMAT_VALID':
                assert ev['status']=='FORMAT_VALID' and ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
                assert ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
                checked[cache]={v['opaque_query_id']:v['is_correct'] for v in q.validate(saved['batch'],saved['response'])}
            else:
                assert ev['status']!='FORMAT_VALID';checked[cache]=None
        if old['status']=='FORMAT_VALID':assert int(checked[cache][old_record['opaque_query_id']])==old['correct']
        else:assert old['correct'] is None
        db.execute('UPDATE payload SET status=?,correct=?,batch=?,record=? WHERE key=?',(old['status'],old['correct'],'INHERITED_'+bid,old['record'],old['key']))
        db.execute('INSERT OR IGNORE INTO inherited VALUES (?,?)',(old['key'],bid))


def ingest(db):
    lock=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
              batch_size=50,workers=4,attempts_per_payload=1,maximum_new_attempts=1880,
              scientific_lock=q.digest(q.read(RUN/'private/LOCK.json')),old_scores_reused='EXACT_BINDING_ONLY')
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert q.read(RUN/'private/GENERATION_COMPLETE.json')['outputs']==2147
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    paths=sorted(p for p in (RUN/'private/outputs').glob('*/*/*.json') if p.parts[-3]!='BASE_IDENTITY')
    assert len(paths)==2147
    for path in paths:
        d=q.read(path);row=d['binding']['input'];b=d['binding']['judge_input']
        assert d['lock']==lock['scientific_lock'] and row['owner']==d['owner'] and row['query_id']==d['query_id']
        assert (row['question'],row['reference'])==(b['question'],b['reference'])
        assert b['runtime']['actual_precision']=='NATIVE_FULL_FP32' and b['runtime']['native_forward']
        if row.get('image_sha256'):assert row['image_sha256']==b['image_sha256']
        key=q.payload(db,dict(row,image_sha256=b['image_sha256']),b,d['R0'])
        stage=d['stage'];prefix=int(stage.rsplit('_',1)[1]) if '_PREFIX_' in stage else 0 if stage=='BASE' else 8
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',
                   (q.digest(str(path)),stage,'native',prefix,d['owner'],d['role'],d['query_id'],str(path),q.digest(d),key))
    inherited(db)
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]
    assert 0<=pending<=1880
    q.write(q.ROOT/'READY.json',dict(consumers=2147,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],new_pending=pending,old_scores_reused=db.execute("SELECT count(*) FROM payload WHERE status='FORMAT_VALID' AND key IN (SELECT key FROM inherited)").fetchone()[0],old_missing_retained=db.execute("SELECT count(*) FROM payload WHERE status='MISSING' AND key IN (SELECT key FROM inherited)").fetchone()[0]))


def continual_loss(observations):
    losses=[]
    for key,values in observations.items():
        seen_correct=False
        for prefix,correct in sorted(values):
            if seen_correct and correct==0:losses.append(dict(owner=key[0],prefix=prefix))
            seen_correct=seen_correct or correct==1
    return losses


def gates(complete,panels,owners,losses,structure):
    return dict(scoring_complete=complete,
                final_native=panels['NATIVE']['candidate_correct']==8,
                final_trained_anchors=panels['GFIT']['candidate_correct']==16,
                continual_retention=not losses,
                fresh_Base_held_preserved=panels['HELDOUT']['new_damage']==0,
                untrained_generalization_preserved=all(x['panels'][r]['new_damage']==0 for x in owners for r in ('T1G','T2G')),
                untrained_T2G_improved=panels['T2G']['candidate_correct']>panels['T2G']['Base_correct'],
                unchanged_parameter_structure_and_native_reload=structure)


def selfcheck():
    assert summary([(1,0,False),(0,1,False),(1,None,False)])['new_damage']==1
    assert continual_loss({(1,'a'):[(1,1),(2,0),(3,1)]})==[dict(owner=1,prefix=2)]
    assert not continual_loss({(1,'a'):[(1,0),(2,1),(3,1)]})
    panels={r:dict(candidate_correct=0,Base_correct=0,new_damage=0) for r in ROLES}
    panels['NATIVE']['candidate_correct']=8;panels['GFIT']['candidate_correct']=16;panels['T2G']['candidate_correct']=1
    owners=[dict(panels={r:dict(new_damage=0) for r in ROLES})]
    assert all(gates(True,panels,owners,[],True).values())
    panels['HELDOUT']['new_damage']=1
    assert not all(gates(True,panels,owners,[],True).values())
    assert not all(gates(False,panels,owners,[],True).values())


def report(db):
    selfcheck()
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    records=[]
    for r in db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        assert r['status'] in ('FORMAT_VALID','MISSING')
        d=q.read(r['path']);assert q.digest(d)==r['output_binding']
        records.append(dict(output=d,correct=r['correct'] if r['status']=='FORMAT_VALID' else None,path=r['path'],payload_key=r['payload_key']))
    assert len(records)==2147
    stages={}
    for r in records:
        d=r['output'];stages.setdefault(d['stage'],{})[d['owner'],d['query_id']]=r
    base=stages['BASE'];assert len(base)==267
    complete=all(r['correct'] is not None for r in records)
    def panel(chosen):
        return summary([(base[r['output']['owner'],r['output']['query_id']]['correct'],r['correct'],
                        base[r['output']['owner'],r['output']['query_id']]['output']['R0']['raw_answer']==r['output']['R0']['raw_answer']) for r in chosen])
    def selection(arm,role,owner=None):
        out=[]
        for r in stages[arm].values():
            d=r['output'];match=d['role']==role
            if role=='ORIGINAL63':match=d['role']=='HELDOUT' and d['original_primary']
            if role=='FRESH_BASE_CORRECT_HELD':match=d['role']=='HELDOUT' and base[d['owner'],d['query_id']]['correct']==1
            if role.startswith('T1L_'):match=d['role']=='T1L' and d['binding']['input'].get('same_reference')==(role=='T1L_SAME_REFERENCE')
            if match and (owner is None or d['owner']==owner):out.append(r)
        return out
    panels={};owners={};checks={};histories={}
    allroles=ROLES+('ORIGINAL63','FRESH_BASE_CORRECT_HELD','T1L_SAME_REFERENCE','T1L_DIFFERENT_REFERENCE')
    for arm in ARMS:
        assert len(stages[arm])==267
        panels[arm]={role:panel(selection(arm,role)) for role in allroles}
        assert panels[arm]['ORIGINAL63']['queries']==63 and panels[arm]['HELDOUT']['queries']==96
        owners[arm]=[dict(owner=i,panels={role:panel(selection(arm,role,i)) for role in ROLES if role!='HELDOUT'}) for i in range(1,9)]
        observations={};prefixes=[]
        for prefix in range(1,9):
            rs=stages[f'{arm}_PREFIX_{prefix}'];assert len(rs)==prefix*3
            prefixes.append(dict(prefix=prefix,queries=len(rs),correct=sum(r['correct']==1 for r in rs.values()),missing=sum(r['correct'] is None for r in rs.values())))
            for key,r in rs.items():observations.setdefault(key,[]).append((prefix,r['correct']))
        for key,r in stages[f'{arm}_PREFIX_8'].items():assert r['output']['R0']==stages[arm][key]['output']['R0']
        losses=continual_loss(observations)
        tr=q.read(RUN/'private'/f'TRAIN_COMPLETE_{arm}.json');ev=q.read(RUN/'private'/f'EVAL_COMPLETE_{arm}.json')
        structure=tr['parameters_before']==tr['parameters_after']==7566219264 and tr['added_parameters']==ev['added_parameters']==0 and ev['native_fresh_reload_tokens_equal'] and ev['inference_structure_unchanged'] and not ev['editor_math_imported']
        checks[arm]=gates(complete,panels[arm],owners[arm],losses,structure)
        checks[arm]['insertion_correct']=all(stages[f'{arm}_PREFIX_{owner}'][owner,query]['correct']==1 for owner,query in stages[f'{arm}_PREFIX_8'])
        histories[arm]=dict(prefixes=prefixes,lost_correct_observations=losses)
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    attempts=sum(len(x['keys']) for x in ledger.get(q.BATCH_LEDGER,[]));assert attempts<=1880
    pairs={}
    for label,first,second in [('amplitude','B_NORM','A_LOW'),('directions_conditional_on_norm','C_FULL_FIXED','B_NORM'),('refresh','D_FULL_REFRESH','C_FULL_FIXED'),('RLS_removal','A_LOW','E_RLS')]:
        values={}
        for role in ('NATIVE','FIT','GFIT','T1G','T2G','HELDOUT'):
            aa={ (r['output']['owner'],r['output']['query_id']):r for r in selection(first,role)}
            bb={ (r['output']['owner'],r['output']['query_id']):r for r in selection(second,role)}
            assert aa.keys()==bb.keys();win=loss=tie=missing=lo=hi=0
            for k in aa:
                x,y=aa[k]['correct'],bb[k]['correct']
                if x is None or y is None:
                    missing+=1
                    if x is None and y is None and aa[k]['payload_key']==bb[k]['payload_key']:continue
                    lo+=(x if x is not None else 0)-(y if y is not None else 1)
                    hi+=(x if x is not None else 1)-(y if y is not None else 0)
                else:
                    win+=x>y;loss+=x<y;tie+=x==y;lo+=x-y;hi+=x-y
            values[role]=dict(queries=len(aa),both_scored=win+loss+tie,win=win,loss=loss,tie=tie,at_least_one_missing=missing,correct_count_difference_bounds=[lo,hi])
        pairs[label]=dict(first=first,second=second,panels=values)
    metrics={}
    for arm in ARMS:
        events=[q.read(p) for p in sorted((RUN/'private/training_diagnostics'/arm).glob('*.json'))];assert len(events)==1280
        metrics[arm]=dict(events=len(events),by_edit=[])
        for owner in range(1,9):
            ee=[e for e in events if e['owner']==owner]
            def ce(e):return .25*e['losses']['0']+.25*e['losses'][str(e['source_indices'][1])]+.5*e['losses'][str(e['source_indices'][2])]
            metrics[arm]['by_edit'].append(dict(owner=owner,weighted_CE_first=ce(ee[0]),weighted_CE_last=ce(ee[-1]),BASIS_KL_last=ee[-1]['losses']['BASIS_KL'],maximum_step_norm=max(e['actual_step_norm'] for e in ee),sampled_diagnostics=[dict(step=e['step'],**e['diagnostics']) for e in ee if e['diagnostics']]))
    inherited_counts=dict(db.execute('SELECT status,count(*) FROM payload WHERE key IN (SELECT key FROM inherited) GROUP BY status'))
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE_SCORING',decision='FINITE_ATTRIBUTION_EXPERIMENT_COMPLETE',
        DEV_success_by_arm={a:all(v.values()) for a,v in checks.items()},panels=panels,per_owner=owners,gates=checks,continual_history=histories,registered_contrasts=pairs,training_diagnostics=metrics,
        payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),queue=q.read(q.ROOT/'READY.json'),inherited_payload_status=inherited_counts,
        missing_consumers=sum(r['correct'] is None for r in records),generation_at_cap=sum(r['output']['at_cap'] for r in records),
        generation={k:v for k,v in q.read(RUN/'private/GENERATION_COMPLETE.json').items() if k not in ('counts','lock','epoch')},
        structural_checks_passed=all(checks[a]['unchanged_parameter_structure_and_native_reload'] for a in ARMS),independent_confirmation=False,full_paper_reproduction=False,
        historical_round_one=q.read(Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v1-20261009/run/public/RESULTS.json'))['panels'],historical_round_two=q.read(PARENT/'public/RESULTS.json')['panels'],
        limitation='One exposed DEV order; not independent clinical/SOTA; no causal percentages from retrospective differences; missing is not wrong',
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=attempts,cumulative_Judge=ledger['Judge_attempts'],owned_weights_remaining=0),
        deletion={k:q.read(RUN/'private/DELETION.json')[k] for k in ('files','bytes','retained_copy','diagnostics_retained')})
    q.write(RUN/'private/SEMANTIC.json',[dict(path=r['path'],correct=r['correct']) for r in records])
    q.write(RUN/'public/RESULTS.json',result)
    q.write(RUN/'private/REPORT_COMPLETE.json',dict(status=result['status'],decision=result['decision'],epoch=time.time()))


if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=q.connect();q.ingest=lambda db:None;selfcheck();ingest(db)
        if op['action']=='report':report(db);answer=dict(status='REPORT_COMPLETE')
        elif op['action'] in ('ingest','selfcheck'):answer=q.read(q.ROOT/'READY.json')
        else:
            if op['action']=='reserve':assert sum(len(x['keys']) for x in q.read(RUN/'RESOURCE_LEDGER.json').get(q.BATCH_LEDGER,[]))+len(op['keys'])<=1880
            answer=a.request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
