"""Fixed paired source/generalization/protection report; no model calls or selection."""
import os
import sys
import json
import sqlite3
import time
import statistics
from collections import Counter
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import scope_queue as queue
import damage_queue as helpers
q,RUN=queue.q,queue.RUN
q.ROOT=RUN/'private/judge_scope_astra_medium'
ROLES=('NATIVE','FIT','GFIT','T1G','T2G','T1L','T2L','HELDOUT','ORIGINAL_PRIMARY','FP32_QUALIFIED','T1L_SAME_REFERENCE','T1L_DIFFERENT_REFERENCE')


def input_identity(d,judge):
    value=queue.full_identity(d,judge);value.pop('output')
    if d.get('precision')=='MATH_FP32_FROZEN_FP16_PREFILL' and 'inherited_runtime' not in value['runtime']:
        assert d.get('unedited_Base') is True
        value['runtime']=dict(inherited_runtime=value['runtime'],actual_precision=d['precision'],frozen_FP16_prefill=True)
    return value


def decision(complete,endpoint,source,generalization,less_damage,no_owner_worse,primary_nonworse):
    if not complete:return 'INCOMPLETE_SCORING'
    if not endpoint:return 'ENDPOINT_GROUP_CONSTRAINT_FAILED'
    if not source or not generalization:return 'EDIT_OR_GENERALIZATION_NOT_PRESERVED'
    if not less_damage or not no_owner_worse or not primary_nonworse:return 'NO_QUALIFIED_PROTECTION_ADVANTAGE'
    return 'DEVELOPMENT_PROTECTION_SIGNAL_NOT_INDEPENDENT_CONFIRMATION'


def paired(pairs):
    c=Counter('missing' if a is None or b is None else 'both_correct' if a and b else 'right_only' if b else 'left_only' if a else 'both_wrong' for a,b in pairs)
    return dict(queries=len(pairs),**{k:c[k] for k in ('both_correct','right_only','left_only','both_wrong','missing')})


def selfcheck():
    helpers.selfcheck();queue.selfcheck()
    assert decision(False,True,True,True,True,True,True)=='INCOMPLETE_SCORING'
    assert decision(True,False,True,True,True,True,True)=='ENDPOINT_GROUP_CONSTRAINT_FAILED'
    assert decision(True,True,True,False,True,True,True)=='EDIT_OR_GENERALIZATION_NOT_PRESERVED'
    assert decision(True,True,True,True,True,False,True)=='NO_QUALIFIED_PROTECTION_ADVANTAGE'
    assert paired([(1,1),(0,1),(1,0),(0,0),(None,1)])==dict(queries=5,both_correct=1,right_only=1,left_only=1,both_wrong=1,missing=1)
    return dict(status='PASS',missing_not_wrong=True,failed_group_and_generalization_gate_blocks_success=True)


def main():
    checks=selfcheck();env=q.read(RUN/'private/LAUNCH_ENV.json');judge={k:q.read(q.ROOT/'EPOCH_MANIFEST.json')[k] for k in ('model','reasoning_effort','protocol','prompt')}
    assert (q.ROOT/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(q.ROOT/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    old_scores={}
    for key,filename in [('DAMAGE_PARENT','DAMAGE_SEMANTIC.json'),('OPTIMIZER160_PARENT','OPTIMIZER160_SEMANTIC.json')]:
        for r in q.read(Path(env[key])/'private'/filename):old_scores[r['path']]=r['correct']
    for r in q.read(Path(env['GENERATION_PARENT'])/'private/EDIT_BASELINE_QUALIFICATION.json')['rows']:
        if r['arm']=='BASE':old_scores[r['path']]=r['correct']
    for r in q.read(RUN/'private/HELD_BASE.json').values():old_scores[r['path']]=r['correct']
    data={};scores=dict(old_scores);records=[]
    def read(path):
        path=str(path)
        if path not in data:data[path]=q.read(path)
        return data[path]
    for r in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key'):
        d=read(r['path']);assert q.digest(d)==r['output_binding']
        score=r['correct'] if r['status']=='FORMAT_VALID' else None;scores[r['path']]=score
        records.append(dict(path=r['path'],d=d,score=score))
    assert len(records)==2567
    groups={a:[r for r in records if r['d']['arm']==a] for a in ('C','D','A8','B8','BASE','NATURAL_C','NATURAL_D')}
    current={(r['d']['arm'],r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in records if not r['d']['arm'].startswith('NATURAL_')}
    matched=[]
    for r in groups['C']:
        d=r['d'];owner=d['expert_order'];qid=d['binding']['input']['query_id'];role=d['role']
        if role=='GFIT':continue
        if owner==8:ref=current['A8',8,qid]
        else:
            parent=Path(env['OPTIMIZER160_PARENT'])/'private/outputs/RAW' if role in ('NATIVE','FIT') else Path(env['DAMAGE_PARENT'])/'private/audit_outputs/RAW'
            path=str(parent/str(owner)/(qid+'.json'));old=read(path)
            old=dict(old,role=role,original_primary=d.get('original_primary',False),same_reference=d.get('same_reference',False))
            ref=dict(path=path,d=old,score=scores[path])
        assert input_identity(d,judge)==input_identity(ref['d'],judge),'Version-mismatched reference comparison'
        matched.append(ref)
    groups['A_MATCHED']=matched
    def base_score(r):
        path=r['d']['baseline_path'];assert path in scores
        assert input_identity(r['d'],judge)==input_identity(read(path),judge),'Baseline input binding mismatch'
        return scores[path]
    def select(rows,role,owners=None):
        out=[]
        for r in rows:
            d=r['d'];match=d['role']==role
            if role=='ORIGINAL_PRIMARY':match=d['role']=='HELDOUT' and d.get('original_primary',False)
            if role=='FP32_QUALIFIED':match=d['role']=='HELDOUT' and base_score(r)==1
            if role.startswith('T1L_'):match=d['role']=='T1L' and d.get('same_reference',False)==(role=='T1L_SAME_REFERENCE')
            if match and (owners is None or d['expert_order'] in owners):out.append(r)
        return out
    def summary(rows):
        return helpers.summary([(base_score(r),r['score'],r['d']['R0']['raw_answer']==read(r['d']['baseline_path'])['R0']['raw_answer']) for r in rows])
    panels={a:{role:summary(select(rs,role)) for role in ROLES} for a,rs in groups.items() if a!='BASE'}
    owners={a:[dict(owner=i,panels={role:summary(select(rs,role,{i})) for role in ROLES}) for i in (range(1,9) if a not in ('A8','B8') else (8,))] for a,rs in groups.items() if a in ('C','D','A8','B8','A_MATCHED')}
    version_panels={a:{name:{role:summary(select(rs,role,ids)) for role in ROLES} for name,ids in [('ORIGINAL_SEVEN',set(range(1,8))),('REVISED_OWNER8',{8})]} for a,rs in groups.items() if a in ('C','D','A_MATCHED')}
    def compare(left,right,role):
        l={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups[left],role)}
        rr={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups[right],role)}
        assert set(l)==set(rr)
        return dict(left_arm=left,right_arm=right,**paired([(l[k]['score'],rr[k]['score']) for k in l]))
    paired_cd={role:compare('C','D',role) for role in ROLES}
    paired_ca={role:compare('A_MATCHED','C',role) for role in ROLES if role!='GFIT'}
    endpoints=[];steps=[]
    for arm,n in [('C',range(1,9)),('D',range(1,9)),('A8',[8]),('B8',[8])]:
        for i in n:
            result=q.read(RUN/'private/results'/arm/f'{i}.json');assert len(result['curve'])==160
            endpoints.append(dict(arm=arm,owner=i,**result['endpoint'],final_gain=result['gain'],all_four_cores_updated=result['all_four_cores_updated']))
            steps.append(dict(arm=arm,owner=i,modes=dict(Counter(x['mode'] for x in result['curve'])),
                solver_status=dict(Counter((x.get('solver') or {}).get('status','NA') for x in result['curve'])),
                negative_RAW_group_proposals=sum(any(v<0 for v in x['gains']['raw']) for x in result['curve'] if x.get('gains'))))
    complete=all(r['score'] is not None and base_score(r) is not None for r in records)
    endpoint=all(x['qualified'] for x in endpoints if x['arm']=='D')
    def preserved(roles):
        return all(owners['D'][i]['panels'][role]['candidate_correct']>=owners['C'][i]['panels'][role]['candidate_correct'] for i in range(8) for role in roles)
    source=preserved(('NATIVE','FIT','GFIT'));generalization=preserved(('T1G','T2G'))
    less=panels['D']['FP32_QUALIFIED']['new_damage']<panels['C']['FP32_QUALIFIED']['new_damage']
    none_worse=all(owners['D'][i]['panels']['FP32_QUALIFIED']['new_damage']<=owners['C'][i]['panels']['FP32_QUALIFIED']['new_damage'] for i in range(8))
    primary_nonworse=panels['D']['ORIGINAL_PRIMARY']['candidate_correct']>=panels['C']['ORIGINAL_PRIMARY']['candidate_correct']
    primary=None
    if complete:
        import probe_report as stats
        cc={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups['C'],'ORIGINAL_PRIMARY')}
        dd={(r['d']['expert_order'],r['d']['binding']['input']['query_id']):r for r in select(groups['D'],'ORIGINAL_PRIMARY')}
        deltas=[dict(owner=k[0],group=cc[k]['d']['binding']['input']['source_group'],difference=dd[k]['score']-cc[k]['score']) for k in cc]
        assert len(deltas)==504 and len({r['group'] for r in deltas})==29
        primary=dict(queries=504,mean=stats.mean([r['difference'] for r in deltas]),
            expert_CI=stats.interval([stats.mean([r['difference'] for r in deltas if r['owner']==i]) for i in range(1,9)]),
            source_CI=stats.interval([stats.mean([r['difference'] for r in deltas if r['group']==g]) for g in sorted({r['group'] for r in deltas})]),
            independent_confirmation=False)
    route=q.read(RUN/'private/ROUTES.json');assert len(route['rows'])==219
    ledger=q.read(RUN/'RESOURCE_LEDGER.json');before=q.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==14 and all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result=dict(status='COMPLETE',decision=decision(complete,endpoint,source,generalization,less,none_worse,primary_nonworse),
        gates=dict(complete=complete,endpoint=endpoint,source=source,generalization=generalization,less_damage=less,no_owner_worse=none_worse,primary_nonworse=primary_nonworse),
        panels=panels,per_owner=owners,version_panels=version_panels,paired_CD=paired_cd,paired_CA=paired_ca,primary_contrast=primary,
        endpoints=endpoints,training_modes=steps,new_Base={role:dict(queries=len(select(groups['BASE'],role)),correct=sum(r['score']==1 for r in select(groups['BASE'],role)),missing=sum(r['score'] is None for r in select(groups['BASE'],role))) for role in ROLES[:5]},
        natural_route=dict(queries=219,active=sum(r['route']['activated'] for r in route['rows']),inactive=sum(not r['route']['activated'] for r in route['rows']),
            original_R0_unchanged=True,revised_owner8_uses_original_key=True),
        queue=q.read(q.ROOT/'READY.json'),payload_status=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status')),
        missing=sum(r['score'] is None for r in records),at_cap=sum(r['d']['at_cap'] for r in records),
        generation=q.read(RUN/'private/SCOPE_GENERATION_COMPLETE.json'),selfcheck=checks,independent_confirmation=False,
        resource=dict(new_GPU_process_hours=(ledger['gpu_seconds_used']-before['gpu_seconds_used'])/3600,cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,
            new_Judge=ledger['Judge_attempts']-before['Judge_attempts'],cumulative_Judge=ledger['Judge_attempts']))
    assert result['resource']['new_Judge']<=2567
    q.write(RUN/'private/SCOPE_SEMANTIC.json',[dict(path=r['path'],correct=r['score']) for r in records])
    q.write(RUN/'public/RESULTS.json',result)
    q.write(RUN/'private/SCOPE_REPORT_COMPLETE.json',dict(status='COMPLETE',decision=result['decision'],epoch=time.time()))
    print(json.dumps(dict(status=result['status'],decision=result['decision'],gates=result['gates'],resource=result['resource'])))


if __name__=='__main__':main()
