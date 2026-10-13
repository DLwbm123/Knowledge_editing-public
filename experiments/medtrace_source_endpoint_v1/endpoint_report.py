"""Complete frozen E-C/E-D audit; no model calls or post-hoc selection."""
import os,sys,json,sqlite3,time
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import endpoint_queue as queue
import scope_report as prior
q,RUN=queue.q,queue.RUN
q.ROOT=RUN/'private/judge_endpoint_astra_medium'
ROLES=prior.ROLES

def main():
    queue.selfcheck();prior.selfcheck()
    env=q.read(RUN/'private/LAUNCH_ENV.json');parent=Path(env['SCOPE_PARENT'])
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    judge={k:q.read(q.ROOT/'EPOCH_MANIFEST.json')[k] for k in ('model','reasoning_effort','protocol','prompt')}
    db=sqlite3.connect('file:'+str(q.ROOT/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    scores={};cache={}
    def read(path):
        path=str(path)
        if path not in cache:cache[path]=q.read(path)
        return cache[path]
    for key,name in [('DAMAGE_PARENT','DAMAGE_SEMANTIC.json'),('OPTIMIZER160_PARENT','OPTIMIZER160_SEMANTIC.json'),('SCOPE_PARENT','SCOPE_SEMANTIC.json')]:
        for row in q.read(Path(env[key])/'private'/name):scores[row['path']]=row['correct']
    for row in q.read(Path(env['GENERATION_PARENT'])/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if row['arm']=='BASE':scores[row['path']]=row['correct']
    for row in q.read(RUN/'private/HELD_BASE.json').values():scores[row['path']]=row['correct']
    current=[]
    for row in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        d=read(row['path']);assert q.digest(d)==row['output_binding'];score=row['correct'] if row['status']=='FORMAT_VALID' else None
        scores[row['path']]=score;current.append(dict(path=row['path'],d=d,score=score))
    assert len(current)==1270
    groups={a:[r for r in current if r['d']['arm']==a] for a in ('E','NATURAL_E')}
    for arm in ('C','D','NATURAL_C','NATURAL_D'):
        groups[arm]=[dict(path=str(p),d=read(p),score=scores[str(p)]) for p in sorted((parent/'private/outputs'/arm).glob('*/*.json'))]
    assert all(len(groups[a])==939 for a in ('C','D','E'))
    def base(r):
        path=r['d']['baseline_path'];assert path in scores
        assert prior.input_identity(r['d'],judge)==prior.input_identity(read(path),judge)
        return scores[path]
    def select(rows,role,owner=None):
        out=[]
        for r in rows:
            d=r['d'];ok=d['role']==role
            if role=='ORIGINAL_PRIMARY':ok=d['role']=='HELDOUT' and d.get('original_primary',False)
            if role=='FP32_QUALIFIED':ok=d['role']=='HELDOUT' and base(r)==1
            if role.startswith('T1L_'):ok=d['role']=='T1L' and d.get('same_reference',False)==(role=='T1L_SAME_REFERENCE')
            if ok and (owner is None or d['expert_order'] in owner):out.append(r)
        return out
    def summary(rows):return prior.helpers.summary([(base(r),r['score'],r['d']['R0']['raw_answer']==read(r['d']['baseline_path'])['R0']['raw_answer']) for r in rows])
    panels={a:{role:summary(select(rs,role)) for role in ROLES} for a,rs in groups.items()}
    owners={a:[dict(owner=i,panels={role:summary(select(groups[a],role,{i})) for role in ROLES}) for i in range(1,9)] for a in ('C','D','E')}
    def paired(left,right,role):
        l={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups[left],role)}
        rr={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups[right],role)}
        assert l.keys()==rr.keys()
        for k in l:assert prior.input_identity(l[k]['d'],judge)==prior.input_identity(rr[k]['d'],judge),'Input version mismatch'
        return dict(left_arm=left,right_arm=right,**prior.paired([(l[k]['score'],rr[k]['score']) for k in l]))
    contrasts={left+'_E':{role:paired(left,'E',role) for role in ROLES} for left in ('C','D')}
    def preserved(roles):return all(owners['E'][i]['panels'][role]['candidate_correct']>=owners['C'][i]['panels'][role]['candidate_correct'] for i in range(8) for role in roles)
    endpoints=[dict(owner=i,**q.read(RUN/'private/results/E'/f'{i}.json')) for i in range(1,9)]
    complete=all(r['score'] is not None and base(r) is not None for rs in groups.values() for r in rs)
    gates=dict(complete=complete,endpoint=all(r['endpoint_qualified'] for r in endpoints),source=preserved(('NATIVE','FIT','GFIT')),generalization=preserved(('T1G','T2G')),less_damage=panels['E']['FP32_QUALIFIED']['new_damage']<panels['C']['FP32_QUALIFIED']['new_damage'],no_owner_worse=all(owners['E'][i]['panels']['FP32_QUALIFIED']['new_damage']<=owners['C'][i]['panels']['FP32_QUALIFIED']['new_damage'] for i in range(8)),primary_nonworse=panels['E']['ORIGINAL_PRIMARY']['candidate_correct']>=panels['C']['ORIGINAL_PRIMARY']['candidate_correct'])
    decision=prior.decision(**gates)
    ci={}
    if complete:
        import probe_report as stats
        for arm in ('C','D'):
            a={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups[arm],'ORIGINAL_PRIMARY')}
            e={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups['E'],'ORIGINAL_PRIMARY')}
            rows=[dict(owner=k[0],group=a[k]['d']['binding']['input']['source_group'],difference=e[k]['score']-a[k]['score']) for k in a]
            assert len(rows)==504
            ci[arm+'_E']=dict(mean=stats.mean([r['difference'] for r in rows]),expert_CI=stats.interval([stats.mean([r['difference'] for r in rows if r['owner']==i]) for i in range(1,9)]),source_CI=stats.interval([stats.mean([r['difference'] for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})]))
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json');assert len(ledger['gpu_sessions'])==9 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    routes=q.read(RUN/'private/ROUTES.json')['rows'];assert len(routes)==219
    result=dict(status='COMPLETE',decision=decision,gates=gates,panels=panels,per_owner=owners,paired=contrasts,primary_contrast=ci,endpoints=endpoints,
        version_panels={a:{v:{role:summary(select(groups[a],role,ids)) for role in ROLES} for v,ids in [('ORIGINAL_SEVEN',set(range(1,8))),('REVISED_OWNER8',{8})]} for a in ('C','D','E')},
        natural_route=dict(queries=219,active=sum(r['route']['activated'] for r in routes),original_R0_unchanged=True),
        queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),missing=sum(r['score'] is None for r in current),
        generation=q.read(RUN/'private/ENDPOINT_GENERATION_COMPLETE.json'),reference_exact=q.read(RUN/'private/REFERENCE_EXACT.json'),independent_confirmation=False,
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    assert result['resource']['new_Judge']<=1158
    q.write(RUN/'private/ENDPOINT_SEMANTIC.json',[dict(path=r['path'],correct=r['score']) for r in current]);q.write(RUN/'public/RESULTS.json',result)
    q.write(RUN/'private/ENDPOINT_REPORT_COMPLETE.json',dict(epoch=time.time(),decision=decision))
    print(json.dumps(dict(decision=decision,gates=gates,resource=result['resource'],E={k:panels['E'][k] for k in ('NATIVE','FIT','GFIT','T1G','T2G','FP32_QUALIFIED')})))

if __name__=='__main__':main()
