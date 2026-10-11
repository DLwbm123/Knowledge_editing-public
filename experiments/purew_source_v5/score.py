"""Two isolated one-attempt queues; evaluation scores are unavailable to training."""
import os,json,fcntl,time,sqlite3,sys
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
from damage_queue import summary
q,RUN=a.q,a.RUN
PHASE=os.environ.get('SCORING_PHASE','formal')
assert PHASE in ('qualification','formal')
q.ROOT=RUN/'private'/('judge_source_v5_'+PHASE)
q.EFFORT='medium';q.BATCH_LEDGER='Astra_source_v5_'+PHASE
EPOCH='PUREW_SOURCE_V5_20261011_'+PHASE.upper()

def bounded_summary(pairs):
    result=summary(pairs)
    metrics=dict(Base_correct=lambda b,c:b==1,candidate_correct=lambda b,c:c==1,retained=lambda b,c:b==c==1,new_damage=lambda b,c:b==1 and c==0,repaired=lambda b,c:b==0 and c==1,still_wrong=lambda b,c:b==c==0)
    bounds={k:[0,0] for k in metrics}
    for before,after,shared in pairs:
        options=[(b,c) for b in ((0,1) if before is None else (before,)) for c in ((0,1) if after is None else (after,)) if not shared or b==c]
        assert options
        for key,metric in metrics.items():
            values=[int(metric(b,c)) for b,c in options];bounds[key][0]+=min(values);bounds[key][1]+=max(values)
    result['bounds']=bounds
    return result

def ingest(db):
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    marker='QUAL_GENERATION_COMPLETE.json' if PHASE=='qualification' else 'GENERATION_COMPLETE.json'
    assert (RUN/'private'/marker).is_file()
    manifest=dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,scientific_lock=q.digest(q.read(RUN/'private/LOCK.json')))
    q.write(q.ROOT/'EPOCH_MANIFEST.json',manifest)
    paths=sorted((RUN/'private/outputs').glob('*/*/*.json'))
    if PHASE=='qualification':paths=[p for p in paths if p.parts[-3]=='QUAL_BASE']
    else:paths=[p for p in paths if p.parts[-3] not in ('QUAL_BASE','BASE_IDENTITY','U_RELOAD','H_SOURCE_RELOAD')]
    assert len(paths)==(84 if PHASE=='qualification' else 3783)
    for path in paths:
        d=q.read(path);row=d['binding']['input'];b=d['binding']['judge_input']
        assert d['lock']==manifest['scientific_lock']
        assert row['owner']==d['owner'] and row['query_id']==d['query_id']
        key=q.payload(db,dict(row,image_sha256=b['image_sha256']),b,d['R0'])
        stage=d['stage'];prefix=int(stage.rsplit('_',1)[1]) if '_PREFIX_' in stage else 0 if stage in ('BASE','QUAL_BASE') else 24
        db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),stage,'native',prefix,d['owner'],d['role'],d['query_id'],str(path),q.digest(d),key))
    if PHASE=='formal':
        previous=[RUN/'private/judge_source_v5_qualification/queue.sqlite',Path('/data/bmw/Knowledge_editing/outputs/purew-scope-v4-20261011/run/private/judge_scope_v4_astra_medium/queue.sqlite')]
        for path in previous:
            old=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
            for r in old.execute('SELECT * FROM payload'):
                current=db.execute('SELECT * FROM payload WHERE key=?',(r['key'],)).fetchone()
                if current is None:continue
                assert json.loads(current['binding'])==json.loads(r['binding'])
                assert {k:v for k,v in json.loads(current['record']).items() if k!='opaque_query_id'}=={k:v for k,v in json.loads(r['record']).items() if k!='opaque_query_id'}
                assert r['status'] in ('FORMAT_VALID','MISSING')
                db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',(r['status'],r['correct'],'INHERITED_'+r['batch'],r['key']))
            old.close()
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    q.write(q.ROOT/'READY.json',dict(consumers=len(paths),payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))))

def report(db):
    assert PHASE=='formal' and (q.ROOT/'ALL_WORKERS_COMPLETE.json').is_file()
    stages={};scored=[]
    for x in db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        d=q.read(x['path']);assert q.digest(d)==x['output_binding']
        assert x['status'] in ('FORMAT_VALID','MISSING')
        item=dict(output=d,correct=x['correct'] if x['status']=='FORMAT_VALID' else None,payload_key=x['payload_key'])
        stages.setdefault(d['stage'],{})[d['owner'],d['query_id']]=item;scored.append(item)
    base=stages['BASE'];assert len(base)==261
    def panel(items):
        return bounded_summary([(base[(x['output']['owner'],x['output']['query_id'])]['correct'],x['correct'],base[(x['output']['owner'],x['output']['query_id'])]['payload_key']==x['payload_key']) for x in items])
    panels={};history={};gates={};allscored=all(x['correct'] is not None for x in scored)
    for arm in ('U','H_SOURCE'):
        assert len(stages[arm])==261
        panels[arm]={role:panel([x for x in stages[arm].values() if x['output']['role']==role]) for role in ('NATIVE','T2G','HELDOUT','NEW_HELDOUT')}
        panels[arm]['T2G_by_formulation']={kind:panel([x for x in stages[arm].values() if x['output']['role']=='T2G' and x['output']['binding']['input']['variant_kind']==kind]) for kind in q.read(RUN/'private/DATA_ADMISSION.json')['variant_types']}
        prefixes=[];seen=set();losses=[]
        for prefix in range(1,25):
            xs=stages[f'{arm}_PREFIX_{prefix}'];assert len(xs)==5*prefix
            for key,x in xs.items():
                if key in seen and x['correct']==0:losses.append(dict(owner=key[0],prefix=prefix,role=x['output']['role']))
                if x['correct']==1:seen.add(key)
            prefixes.append(dict(prefix=prefix,by_role={role:panel([x for x in xs.values() if x['output']['role']==role]) for role in ('NATIVE','T2G')}))
        history[arm]=dict(prefixes=prefixes,lost_correct_observations=losses)
        insertion=all(stages[f'{arm}_PREFIX_{owner}'][owner,row['query_id']]['correct']==1 for owner in range(1,25) for row in q.read(RUN/'private/ROWS.json') if row['owner']==owner and row['audit_role']=='NATIVE')
        structural=q.read(RUN/'private'/f'EVAL_COMPLETE_{arm}.json')
        gates[arm]=dict(full_scoring=allscored,source_insertion=insertion,source_continual=not any(x['role']=='NATIVE' for x in losses),untrained_Base_correct_T2G_preserved=all(x['by_role']['T2G']['bounds']['new_damage'][1]==0 for x in prefixes),old_protection=panels[arm]['HELDOUT']['new_damage']==0,new_protection=panels[arm]['NEW_HELDOUT']['new_damage']==0 and panels[arm]['NEW_HELDOUT']['Base_correct']>0,structural=structural['native_fresh_reload_tokens_equal'] and structural['added_parameters']==0 and structural['inference_structure_unchanged'])
    pairs={}
    for role in ('NATIVE','T2G','HELDOUT','NEW_HELDOUT'):
        win=loss=tie=missing=lo=hi=0
        for key,x in stages['H_SOURCE'].items():
            if x['output']['role']!=role:continue
            y=stages['U'][key];a0,b0=x['correct'],y['correct']
            if a0 is None or b0 is None:
                missing+=1
                if a0 is None and b0 is None and x['payload_key']==y['payload_key']:continue
                lo+=(a0 if a0 is not None else 0)-(b0 if b0 is not None else 1);hi+=(a0 if a0 is not None else 1)-(b0 if b0 is not None else 0)
            else:win+=a0>b0;loss+=a0<b0;tie+=a0==b0;lo+=a0-b0;hi+=a0-b0
        pairs[role]=dict(win=win,loss=loss,tie=tie,missing=missing,correct_count_difference_bounds=[lo,hi])
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE' if allscored else 'INCOMPLETE_SCORING',panels=panels,continual_history=history,gates=gates,DEV_success={a:all(v.values()) for a,v in gates.items()},H_minus_U=pairs,missing_consumers=sum(x['correct'] is None for x in scored),source_only_variant=True,added_inference_parameters=0,independent_clinical_confirmation=False,untrained_question_formulation_types=q.read(RUN/'private/DATA_ADMISSION.json')['variant_types'],resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']),deletion={k:q.read(RUN/'private/DELETION.json')[k] for k in ('files','bytes','retained_copy')})
    q.write(RUN/'public/RESULTS.json',result);q.write(RUN/'private/REPORT_COMPLETE.json',dict(status=result['status']))

if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'))
    q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);db=q.connect();ingest(db);q.ingest=lambda db:None
        if op['action']=='report':report(db);answer=dict(status='REPORT_COMPLETE')
        elif op['action']=='ingest':answer=q.read(q.ROOT/'READY.json')
        else:answer=a.request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
