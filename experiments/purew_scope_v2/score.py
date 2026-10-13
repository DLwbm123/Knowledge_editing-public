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
q.ROOT=RUN/'private/judge_scope_v2_astra_medium'
q.BATCH_LEDGER='Astra_scope_v2_batches'
EPOCH='PUREW_SCOPE_V2_20261010'
ARMS=('GATED','ALWAYS_ON')
ROLES=('NATIVE','FIT','GFIT','T1G','T2G','T1L','T2L','HELDOUT')
PARENT=Path(q.read(RUN/'private/INHERITED_RUN.json')['root'])


def inherited(db):
    import sqlite3
    parent=PARENT/'private/judge_purew_astra_medium'
    old=sqlite3.connect('file:'+str(parent/'queue.sqlite')+'?mode=ro',uri=True)
    old.row_factory=sqlite3.Row
    prior={r['key']:dict(r) for r in old.execute('SELECT * FROM payload')};old.close()
    db.execute('CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_batch TEXT)')
    checked={}
    for current in list(db.execute('SELECT * FROM payload')):
        old=prior.get(current['key'])
        if old is None:continue
        assert old['status']=='FORMAT_VALID'
        assert json.loads(old['binding'])==json.loads(current['binding'])
        old_record=json.loads(old['record']);current_record=json.loads(current['record'])
        assert {k:v for k,v in old_record.items() if k!='opaque_query_id'}=={k:v for k,v in current_record.items() if k!='opaque_query_id'}
        batch=old['batch']
        if batch not in checked:
            saved=q.read(parent/'evidence'/(batch+'.json'));ev=saved['evidence']
            assert ev['status']=='FORMAT_VALID' and ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='medium'
            assert ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
            assert ev['input_binding']==q.digest(saved['batch'])
            checked[batch]=(saved['batch']['records'],{v['opaque_query_id']:v['is_correct'] for v in q.validate(saved['batch'],saved['response'])})
        records,decisions=checked[batch];record=json.loads(old['record'])
        assert record in records and int(decisions[record['opaque_query_id']])==old['correct']
        db.execute("UPDATE payload SET status='FORMAT_VALID',correct=?,batch=?,record=? WHERE key=?",(old['correct'],'INHERITED_'+batch,old['record'],old['key']))
        db.execute('INSERT OR IGNORE INTO inherited VALUES (?,?)',(old['key'],batch))


def ingest(db):
    lock=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,
              batch_size=50,workers=4,attempts_per_payload=1,maximum_new_attempts=752,
              scientific_lock=q.digest(q.read(RUN/'private/LOCK.json')),old_scores_reused='EXACT_BINDING_ONLY')
    path=q.ROOT/'EPOCH_MANIFEST.json'
    if path.exists():assert q.read(path)==lock
    else:q.write(path,lock)
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert q.read(RUN/'private/GENERATION_COMPLETE.json')['outputs']==1019
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    paths=sorted(p for p in (RUN/'private/outputs').glob('*/*/*.json') if p.parts[-3]!='BASE_IDENTITY')
    assert len(paths)==1019
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
    assert 0<=pending<=752
    q.write(q.ROOT/'READY.json',dict(consumers=1019,payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],new_pending=pending,old_scores_reused=db.execute('SELECT count(*) FROM inherited').fetchone()[0]))


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
        records.append(dict(output=d,correct=r['correct'] if r['status']=='FORMAT_VALID' else None,path=r['path']))
    assert len(records)==1019
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
    attempts=sum(len(x['keys']) for x in ledger.get(q.BATCH_LEDGER,[]));assert attempts<=752
    paired_regressions={r:0 for r in ('NATIVE','GFIT','T1G','T2G','BASE_CORRECT_HELD')}
    for key,control in stages['ALWAYS_ON'].items():
        role=control['output']['role']
        if role=='HELDOUT' and base[key]['correct']==1:role='BASE_CORRECT_HELD'
        if role in paired_regressions:
            paired_regressions[role]+=int(control['correct']==1 and stages['GATED'][key]['correct']!=1)
    improvements=[r for r in ROLES if panels['GATED'][r]['candidate_correct']>panels['ALWAYS_ON'][r]['candidate_correct']]
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE_SCORING',
                decision='DEV_SUCCESS' if all(checks['GATED'].values()) else 'FROZEN_SUCCESS_CRITERIA_NOT_MET',
                panels=panels,per_owner=owners,gates=checks,continual_history=histories,
                scope_gate_difference={r:panels['GATED'][r]['candidate_correct']-panels['ALWAYS_ON'][r]['candidate_correct'] for r in allroles},
                scope_contribution=dict(established=complete and bool(improvements) and not any(paired_regressions.values()),
                    improved_panels=improvements,paired_query_regressions=paired_regressions),
                payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
                queue=q.read(q.ROOT/'READY.json'),missing_consumers=sum(r['correct'] is None for r in records),
                generation_at_cap=sum(r['output']['at_cap'] for r in records),generation=q.read(RUN/'private/GENERATION_COMPLETE.json'),
                insertion_correct=all(checks[a]['insertion_correct'] for a in ARMS),
                structural_checks_passed=all(checks[a]['unchanged_parameter_structure_and_native_reload'] for a in ARMS),
                independent_confirmation=False,full_paper_reproduction=False,
                mechanism='paper ScopeEdit core adapted to direct existing W; fixed single-layer weight-nullspace coordinates',
                diagnostics={a:dict(events=len(list((RUN/'private/training_diagnostics'/a).glob('*.json'))),geometry=[q.read(RUN/'private/geometry_diagnostics'/f'{a}_{i}.json') for i in range(1,9)]) for a in ARMS},
                historical_round_one=q.read(PARENT/'public/RESULTS.json')['panels'],
                limitation='One exposed development order/seed; trained GFIT is not unseen generalization; no clinical or SOTA confirmation',
                resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,
                              new_Judge=attempts,cumulative_Judge=ledger['Judge_attempts'],owned_weights_remaining=0),
                deletion={k:q.read(RUN/'private/DELETION.json')[k] for k in ('files','bytes','retained_copy','reconstruction','diagnostics_retained')})
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
            if op['action']=='reserve':assert sum(len(x['keys']) for x in q.read(RUN/'RESOURCE_LEDGER.json').get(q.BATCH_LEDGER,[]))+len(op['keys'])<=752
            answer=a.request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():
            q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
