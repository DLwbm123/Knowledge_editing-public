"""Shared one-attempt scorer with immutable reuse and missing-aware reports."""
import os,json,fcntl,time,sqlite3,sys,statistics,itertools
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_queue as a
from damage_queue import summary
q,RUN=a.q,a.RUN
q.ROOT=RUN/'private/judge_source24_baselines';q.EFFORT='medium';q.BATCH_LEDGER='Astra_source24_baselines'
EPOCH='SOURCE24_BASELINES_V1_20261011'
PARENT=Path('/data/bmw/Knowledge_editing/outputs/purew-source-v5-20261011/run')
ARMS=('U','H_SOURCE','TT88_SOURCE','BALANCEDIT_SOURCE')

def bounds(pairs):
    result=summary(pairs)
    functions=dict(Base_correct=lambda b,c:b==1,candidate_correct=lambda b,c:c==1,retained=lambda b,c:b==c==1,new_damage=lambda b,c:b==1 and c==0,repaired=lambda b,c:b==0 and c==1,still_wrong=lambda b,c:b==c==0)
    limits={k:[0,0] for k in functions}
    for before,after,shared in pairs:
        options=[(b,c) for b in ((0,1) if before is None else (before,)) for c in ((0,1) if after is None else (after,)) if not shared or b==c]
        assert options
        for k,f in functions.items():
            x=[int(f(b,c)) for b,c in options];limits[k][0]+=min(x);limits[k][1]+=max(x)
    result['bounds']=limits;return result

def ingest(db):
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    assert (RUN/'private/GENERATION_COMPLETE.json').is_file()
    parent_queue=PARENT/'private/judge_source_v5_formal'
    assert (parent_queue/'ALL_WORKERS_COMPLETE.json').is_file()
    scientific_lock=q.digest(q.read(RUN/'private/LOCK.json'))
    q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=EPOCH,model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT,attempts_per_payload=1,scientific_lock=scientific_lock))
    files=[]
    for root,is_parent in ((PARENT,True),(RUN,False)):
        for path in sorted((root/'private/outputs').glob('*/*/*.json')):
            stage=path.parts[-3]
            if is_parent and stage not in ('BASE','U','H_SOURCE') and not stage.startswith(('U_PREFIX_','H_SOURCE_PREFIX_')):continue
            if not is_parent and stage not in ARMS[2:] and not any(stage.startswith(arm+'_PREFIX_') or stage.startswith(arm+'_PROTECT_') for arm in ARMS[2:]):continue
            d=q.read(path)
            assert d['lock']==(q.read(RUN/'private/PARENT_BINDING.json')['scientific_lock'] if is_parent else scientific_lock)
            row=d['binding']['input'];binding=d['binding']['judge_input']
            key=q.payload(db,dict(row,image_sha256=binding['image_sha256']),binding,d['R0'])
            prefix=int(stage.rsplit('_',1)[1]) if '_PREFIX_' in stage or '_PROTECT_' in stage else 0 if stage=='BASE' else 24
            db.execute('INSERT INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(q.digest(str(path)),stage,'native',prefix,d['owner'],d['role'],d['query_id'],str(path),q.digest(d),key));files.append(path)
    assert len(files)==8433
    old=sqlite3.connect('file:'+str(parent_queue/'queue.sqlite')+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
    reused=0
    for r in old.execute('SELECT * FROM payload'):
        current=db.execute('SELECT * FROM payload WHERE key=?',(r['key'],)).fetchone()
        if current is None:continue
        assert json.loads(current['binding'])==json.loads(r['binding'])
        assert {k:v for k,v in json.loads(current['record']).items() if k!='opaque_query_id'}=={k:v for k,v in json.loads(r['record']).items() if k!='opaque_query_id'}
        assert r['status'] in ('FORMAT_VALID','MISSING')
        db.execute('UPDATE payload SET status=?,correct=?,batch=? WHERE key=?',(r['status'],r['correct'],'INHERITED_'+r['batch'],r['key']));reused+=1
    old.close();db.execute("INSERT INTO done VALUES ('generation')");db.commit()
    pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0];assert pending<=4500
    q.write(q.ROOT/'READY.json',dict(consumers=len(files),payloads=db.execute('SELECT count(*) FROM payload').fetchone()[0],reused_payloads=reused,new_pending=pending,statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))))

def report(db):
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').is_file()
    stages={};allrows=[]
    for x in db.execute('SELECT c.*,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        d=q.read(x['path']);assert q.digest(d)==x['output_binding'] and x['status'] in ('FORMAT_VALID','MISSING')
        item=dict(role=d['role'],correct=x['correct'] if x['status']=='FORMAT_VALID' else None,key=x['payload_key'],kind=d['binding']['input'].get('variant_kind'))
        stages.setdefault(d['stage'],{})[d['owner'],d['query_id']]=item;allrows.append(item)
    base=stages['BASE'];assert len(base)==261
    def panel(items):return bounds([(base[k]['correct'],x['correct'],base[k]['key']==x['key']) for k,x in items])
    def correct_options(x):return (0,1) if x['correct'] is None else (x['correct'],)
    panels={};continual={};protections={};gates={}
    for arm in ARMS:
        final=stages[arm];assert len(final)==261
        panels[arm]={role:panel([(k,x) for k,x in final.items() if x['role']==role]) for role in ('NATIVE','T2G','HELDOUT','NEW_HELDOUT')}
        panels[arm]['T2G_by_formulation']={kind:panel([(k,x) for k,x in final.items() if x['role']=='T2G' and x['kind']==kind]) for kind in ('VERBATIM_REFRAME','LOCATION_REWRITE','TYPE_REWRITE')}
        prefixes=[];owners=[]
        for prefix in range(1,25):
            xs=stages[f'{arm}_PREFIX_{prefix}'];assert len(xs)==5*prefix
            prefixes.append(dict(prefix=prefix,by_role={role:panel([(k,x) for k,x in xs.items() if x['role']==role]) for role in ('NATIVE','T2G')}))
        for owner in range(1,25):
            native=next(k for k,x in final.items() if k[0]==owner and x['role']=='NATIVE')
            history=[stages[f'{arm}_PREFIX_{p}'][native] for p in range(owner,25)]
            # Full-denominator all-prefix retention bounds; shared payload is one unknown.
            inserted=history[0];unique={x['key']:x for x in history}
            all_correct=[int(all(x['correct']==1 for x in unique.values())),int(all(x['correct']!=0 for x in unique.values()))]
            losses=[0,0]
            for prior,after in zip(history,history[1:]):
                options=[int(a==1 and z==0) for a in correct_options(prior) for z in correct_options(after) if prior['key']!=after['key'] or a==z]
                losses[0]+=min(options);losses[1]+=max(options)
            t2g_history=[]
            for key,value in final.items():
                if key[0]!=owner or value['role']!='T2G':continue
                zs=[stages[f'{arm}_PREFIX_{p}'][key] for p in range(owner,25)];unique_t2={x['key']:x for x in zs};lost=[0,0]
                for prior,after in zip(zs,zs[1:]):
                    opts=[int(a==1 and z==0) for a in correct_options(prior) for z in correct_options(after) if prior['key']!=after['key'] or a==z];lost[0]+=min(opts);lost[1]+=max(opts)
                t2g_history.append(dict(variant_slot=len(t2g_history),formulation=value['kind'],all_prefix_correct_bounds=[int(all(x['correct']==1 for x in unique_t2.values())),int(all(x['correct']!=0 for x in unique_t2.values()))],adjacent_loss_event_bounds=lost,correct_by_prefix=[x['correct'] for x in zs]))
            owners.append(dict(owner=owner,T2G_history=t2g_history,T0_insertion_correct_bounds=[min(correct_options(inserted)),max(correct_options(inserted))],T0_all_later_prefixes_correct_bounds=all_correct,T0_adjacent_loss_event_bounds=losses,T2G_final=panel([(k,x) for k,x in final.items() if k[0]==owner and x['role']=='T2G'])))
        continual[arm]=dict(prefixes=prefixes,owners=owners,T0_all_prefix_retained_owner_count_bounds=[sum(x['T0_all_later_prefixes_correct_bounds'][i] for x in owners) for i in (0,1)])
        protections[arm]={str(prefix):{role:panel([(k,x) for k,x in stages[f'{arm}_PROTECT_{prefix}'].items() if x['role']==role]) for role in ('HELDOUT','NEW_HELDOUT')} if arm in ARMS[2:] else 'NA_NOT_MEASURED' for prefix in (1,8,16)}
        protections[arm]['24']={role:panels[arm][role] for role in ('HELDOUT','NEW_HELDOUT')}
        gates[arm]=dict(full_scoring=all(x['correct'] is not None for stage,xs in stages.items() if stage.startswith(arm) for x in xs.values()),all_source_insertions_and_retention=all(x['T0_all_later_prefixes_correct_bounds'][0]==1 for x in owners),Base_correct_T2G_no_new_damage=all(x['by_role']['T2G']['bounds']['new_damage'][1]==0 for x in prefixes),old_protection_no_new_damage=panels[arm]['HELDOUT']['bounds']['new_damage'][1]==0,new_protection_no_new_damage=panels[arm]['NEW_HELDOUT']['bounds']['new_damage'][1]==0 and panels[arm]['NEW_HELDOUT']['bounds']['Base_correct'][0]>0)
    paired={}
    for left,right in itertools.combinations(ARMS,2):
        metrics={}
        for role in ('NATIVE','T2G','HELDOUT','NEW_HELDOUT'):
            lo=hi=win=loss=tie=missing=0
            for key,x in stages[left].items():
                if x['role']!=role:continue
                y=stages[right][key];opts=[a-z for a in correct_options(x) for z in correct_options(y) if x['key']!=y['key'] or a==z];lo+=min(opts);hi+=max(opts)
                if x['correct'] is None or y['correct'] is None:missing+=1
                else:win+=x['correct']>y['correct'];loss+=x['correct']<y['correct'];tie+=x['correct']==y['correct']
            metrics[role]=dict(left_minus_right_bounds=[lo,hi],left_wins=win,right_wins=loss,ties=tie,missing=missing)
        paired[left+'_minus_'+right]=metrics
    performance=q.read(RUN/'public/PERFORMANCE.json');timing={}
    for arm,value in performance.items():
        events=[x for x in value['events'] if not x['warmup']];medians=[statistics.median(x['seconds'] for x in events if x['query_slot']==slot) for slot in range(12)]
        timing[arm]=dict(structure=value['summary'],paired_query_medians_seconds=medians,total_seconds=sum(x['seconds'] for x in events),total_tokens=sum(x['tokens'] for x in events),tokens_per_end_to_end_second=sum(x['tokens'] for x in events)/sum(x['seconds'] for x in events),median_prefill_seconds=statistics.median(x['forward_seconds'][0] for x in events),median_decode_seconds=statistics.median(sum(x['forward_seconds'][1:]) for x in events),median_route_seconds=statistics.median(x['route_seconds'] for x in events),lengths=[statistics.median(x['tokens'] for x in events if x['query_slot']==slot) for slot in range(12)])
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');parent=q.read(PARENT/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    result=dict(status='COMPLETE' if all(x['correct'] is not None for x in allrows) else 'COMPLETE_WITH_MISSING_SCORES',panels=panels,continual=continual,protection_prefixes=protections,paired_final=paired,gates=gates,efficacy_DEV_gates={a:all(g.values()) for a,g in gates.items()},missing_consumers=sum(x['correct'] is None for x in allrows),scoring_statuses=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),inference=timing,resource=dict(new_GPU_process_hours=ledger['gpu_seconds_used']/3600,prior_cumulative_GPU_process_hours=parent['gpu_seconds_used']/3600,total_GPU_process_hours=(ledger['gpu_seconds_used']+parent['gpu_seconds_used'])/3600,new_Judge=ledger['Judge_attempts'],prior_cumulative_Judge=parent['Judge_attempts'],total_Judge=ledger['Judge_attempts']+parent['Judge_attempts'],new_weight_peak_bytes=ledger['weights_observed_peak_bytes']),deletion={k:x for k,x in q.read(RUN/'private/DELETION.json').items() if k!='paths'},calls={k:sum(q.read(p)[k] for p in (RUN/'private/attempts').glob('*.json')) for k in ('forwards','backwards','updates','generations','generation_attempts','generation_forwards')},failure_receipts=len(list((RUN/'private/failures').glob('*.json'))),structural={a:q.read(RUN/'private'/('RELOAD_'+a+'.json'))['summary'] for a in ARMS[2:]},execution_version=dict(new_gpu_code=q.read(RUN/'private/GPU_SOURCE_VERSION.json')['commit'],inherited_parent_installation=q.read(RUN/'private/PARENT_EXECUTION_VERSIONS.json')['installation']['commit'],parent_branch_head=q.read(RUN/'private/FROZEN_PROTOCOL.json')['parent_source_commit']),limitations=['source-only positive/FIT adaptations','different supervision and fixed method-specific optimization budgets','same-image question variants, no T1G','patient independence UNKNOWN','pureW early protection NA','no clinical or SOTA confirmation'])
    q.write(RUN/'public/RESULTS.json',result);q.write(RUN/'private/REPORT_COMPLETE.json',dict(status=result['status']))

if __name__=='__main__':
    op=json.loads(os.environ.get('QUEUE_REQUEST','{"action":"ingest"}'));q.ROOT.mkdir(parents=True,exist_ok=True)
    with (q.ROOT/'QUEUE.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);db=q.connect();ingest(db);q.ingest=lambda db:None
        if op['action']=='report':report(db);answer=dict(status='REPORT_COMPLETE')
        elif op['action']=='ingest':answer=q.read(q.ROOT/'READY.json')
        else:answer=a.request(db,op)
        if not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone():q.write(q.ROOT/'ALL_WORKERS_COMPLETE.json',dict(status='SCORING_COMPLETE',epoch=time.time()))
    print(json.dumps(answer,ensure_ascii=False))
