"""Frozen four-arm mechanism and NT primary analysis; no selection or model calls."""
import os,sys,json,sqlite3,time
from pathlib import Path
from collections import Counter
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import fine_queue as queue
import scope_report as prior
q,RUN=queue.q,queue.RUN
q.ROOT=RUN/'private/judge_fine_astra_medium'
ROLES=prior.ROLES
ARMS=('D','N','T','NT')


def decision(complete,endpoint,source,generalization,less_damage,no_owner_worse,positive_intervals):
    if not complete:return 'INCOMPLETE_SEMANTIC_EVIDENCE'
    if not endpoint:return 'ENDPOINT_GROUP_CONSTRAINT_FAILED'
    if not source or not generalization:return 'EDIT_OR_GENERALIZATION_NOT_PRESERVED'
    if not less_damage or not no_owner_worse or not positive_intervals:return 'NO_STRONGER_DEVELOPMENT_PROTECTION_EVIDENCE'
    return 'STRONGER_DEVELOPMENT_SIGNAL_NOT_INDEPENDENT_CONFIRMATION'


def selfcheck():
    queue.selfcheck();prior.selfcheck()
    assert decision(True,True,True,True,True,True,False)=='NO_STRONGER_DEVELOPMENT_PROTECTION_EVIDENCE'
    assert decision(False,True,True,True,True,True,True)=='INCOMPLETE_SEMANTIC_EVIDENCE'
    assert decision(True,True,True,True,True,True,True)=='STRONGER_DEVELOPMENT_SIGNAL_NOT_INDEPENDENT_CONFIRMATION'
    return dict(status='PASS',zero_or_crossing_interval_not_success=True,missing_not_success=True)


def main():
    checks=selfcheck();env=q.read(RUN/'private/LAUNCH_ENV.json');parent=Path(env['SCOPE_PARENT']);ep=Path(env['ENDPOINT_PARENT'])
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    judge={k:q.read(q.ROOT/'EPOCH_MANIFEST.json')[k] for k in ('model','reasoning_effort','protocol','prompt')}
    db=sqlite3.connect('file:'+str(q.ROOT/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    scores={};cache={}
    def read(path):
        path=str(path)
        if path not in cache:cache[path]=q.read(path)
        return cache[path]
    for key,name in [('DAMAGE_PARENT','DAMAGE_SEMANTIC.json'),('OPTIMIZER160_PARENT','OPTIMIZER160_SEMANTIC.json'),('SCOPE_PARENT','SCOPE_SEMANTIC.json'),('ENDPOINT_PARENT','ENDPOINT_SEMANTIC.json')]:
        for row in q.read(Path(env[key])/'private'/name):scores[row['path']]=row['correct']
    for row in q.read(Path(env['GENERATION_PARENT'])/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if row['arm']=='BASE':scores[row['path']]=row['correct']
    for row in q.read(RUN/'private/HELD_BASE.json').values():scores[row['path']]=row['correct']
    current=[]
    for row in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        d=read(row['path']);assert q.digest(d)==row['output_binding'];score=row['correct'] if row['status']=='FORMAT_VALID' else None
        scores[row['path']]=score;current.append(dict(path=row['path'],d=d,score=score))
    assert len(current)==3586
    groups={a:[r for r in current if r['d']['arm']==a] for a in ('N','T','NT','NATURAL_N','NATURAL_T','NATURAL_NT')}
    for root,arms in [(parent,('C','D','NATURAL_C','NATURAL_D')),(ep,('E','NATURAL_E'))]:
        for arm in arms:groups[arm]=[dict(path=str(p),d=read(p),score=scores[str(p)]) for p in sorted((root/'private/outputs'/arm).glob('*/*.json'))]
    assert all(len(groups[a])==939 for a in ('C','D','E','N','T','NT'))
    def base(r):
        path=r['d']['baseline_path'];assert path in scores
        assert prior.input_identity(r['d'],judge)==prior.input_identity(read(path),judge)
        return scores[path]
    def select(rows,role,owners=None):
        out=[]
        for r in rows:
            d=r['d'];ok=d['role']==role
            if role=='ORIGINAL_PRIMARY':ok=d['role']=='HELDOUT' and d.get('original_primary',False)
            if role=='FP32_QUALIFIED':ok=d['role']=='HELDOUT' and base(r)==1
            if role.startswith('T1L_'):ok=d['role']=='T1L' and d.get('same_reference',False)==(role=='T1L_SAME_REFERENCE')
            if ok and (owners is None or d['expert_order'] in owners):out.append(r)
        return out
    def summary(rows):return prior.helpers.summary([(base(r),r['score'],r['d']['R0']['raw_answer']==read(r['d']['baseline_path'])['R0']['raw_answer']) for r in rows])
    panels={a:{role:summary(select(rs,role)) for role in ROLES} for a,rs in groups.items()}
    owners={a:[dict(owner=i,panels={role:summary(select(groups[a],role,{i})) for role in ROLES}) for i in range(1,9)] for a in ('C','E')+ARMS}
    def mapped(arm,role):return {(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups[arm],role)}
    def pair(left,right,role):
        a,b=mapped(left,role),mapped(right,role);assert a.keys()==b.keys()
        for k in a:assert prior.input_identity(a[k]['d'],judge)==prior.input_identity(b[k]['d'],judge)
        return dict(left_arm=left,right_arm=right,**prior.paired([(a[k]['score'],b[k]['score']) for k in a]))
    comparisons={a+'_'+b:{role:pair(a,b,role) for role in ROLES} for a,b in [('D','N'),('D','T'),('D','NT'),('C','NT'),('E','NT')]}
    complete=all(r['score'] is not None and base(r) is not None for rows in groups.values() for r in rows)
    ci={};interaction={}
    if complete:
        import probe_report as stats
        def intervals(rows):return dict(mean=stats.mean([r['delta'] for r in rows]),expert_CI=stats.interval([stats.mean([r['delta'] for r in rows if r['owner']==i]) for i in range(1,9)]),source_CI=stats.interval([stats.mean([r['delta'] for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})]))
        for a,b in [('D','N'),('D','T'),('D','NT'),('C','NT'),('E','NT')]:
            aa,bb=mapped(a,'ORIGINAL_PRIMARY'),mapped(b,'ORIGINAL_PRIMARY')
            rows=[dict(owner=k[0],group=aa[k]['d']['binding']['input']['source_group'],delta=bb[k]['score']-aa[k]['score']) for k in aa];assert len(rows)==504
            ci[a+'_'+b]=intervals(rows)
        for role in ROLES:
            maps={a:mapped(a,role) for a in ARMS};assert all(maps['D'].keys()==m.keys() for m in maps.values())
            rows=[dict(owner=k[0],group=maps['D'][k]['d']['binding']['input']['source_group'],delta=maps['NT'][k]['score']-maps['N'][k]['score']-maps['T'][k]['score']+maps['D'][k]['score']) for k in maps['D']]
            interaction[role]=dict(queries=len(rows),sum=sum(r['delta'] for r in rows),histogram=dict(Counter(r['delta'] for r in rows)))
            if role=='ORIGINAL_PRIMARY':interaction[role].update(intervals(rows))
    def preserved(roles):return all(owners['NT'][i]['panels'][role]['candidate_correct']>=owners['C'][i]['panels'][role]['candidate_correct'] for i in range(8) for role in roles)
    mechanism=[]
    for arm in ARMS:
        for i in range(1,9):
            d=read(RUN/'private/results'/arm/f'{i}.json');pre=read(RUN/'private/pre_endpoint'/arm/f'{i}.json');assert len(d['curve'])==160
            fine=read(RUN/'private/fine_checks'/arm/f'{i}.json') if arm!='D' else []
            if arm!='D':assert len(fine)==160
            mechanism.append(dict(arm=arm,owner=i,endpoint=d['endpoint'],all_four_cores_updated=d['all_four_cores_updated'],modes=dict(Counter(x['mode'] for x in d['curve'])),fine_checks=sum(x['checks'] for x in fine),NLL_failures=sum(x['nll_failures'] for x in fine),token_failures=sum(x['token_failures'] for x in fine),pre_all_argmax=sum(v['all_argmax_correct'] for v in pre['values']),final_all_argmax=sum(v['all_argmax_correct'] for v in d['values']),pre_correct_tokens=sum(v['correct_tokens'] for v in pre['values']),final_correct_tokens=sum(v['correct_tokens'] for v in d['values'])))
    gates=dict(complete=complete,endpoint=all(x['endpoint']['qualified'] for x in mechanism if x['arm']=='NT'),source=preserved(('NATIVE','FIT','GFIT')),generalization=preserved(('T1G','T2G')),less_damage=panels['NT']['FP32_QUALIFIED']['new_damage']<panels['E']['FP32_QUALIFIED']['new_damage'],no_owner_worse=all(owners['NT'][i]['panels']['FP32_QUALIFIED']['new_damage']<=owners['E'][i]['panels']['FP32_QUALIFIED']['new_damage'] for i in range(8)),positive_intervals=complete and ci['E_NT']['expert_CI'][0]>0 and ci['E_NT']['source_CI'][0]>0)
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json');assert len(ledger['gpu_sessions'])==13 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision(**gates),gates=gates,panels=panels,per_owner=owners,paired=comparisons,primary_contrast=ci,interaction=interaction,mechanism=mechanism,selfcheck=checks,
        version_panels={a:{v:{role:summary(select(groups[a],role,ids)) for role in ROLES} for v,ids in [('ORIGINAL_SEVEN',set(range(1,8))),('REVISED_OWNER8',{8})]} for a in ('C','E')+ARMS},
        queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),missing=sum(r['score'] is None for r in current),generation=q.read(RUN/'private/FINE_GENERATION_COMPLETE.json'),reference_exact=q.read(RUN/'private/REFERENCE_EXACT.json'),independent_confirmation=False,
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    assert result['resource']['new_Judge']<=3474 and result['resource']['new_GPU_process_hours']<=64.01
    q.write(RUN/'private/FINE_SEMANTIC.json',[dict(path=r['path'],correct=r['score']) for r in current]);q.write(RUN/'public/RESULTS.json',result);q.write(RUN/'private/FINE_REPORT_COMPLETE.json',dict(decision=result['decision'],epoch=time.time()))
    print(json.dumps(dict(decision=result['decision'],gates=gates,resource=result['resource'])))

if __name__=='__main__':main()
