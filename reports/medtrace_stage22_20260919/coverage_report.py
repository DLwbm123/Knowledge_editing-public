"""All valid coverage arms, matching-U comparison and fixed DEV selection."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
from scripts.medtrace.stage18_score import score_key,query_id
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.astra_judge_bundle import write_new
from report import metric,WEIGHTS


def select(arms,control,complete):
    if not complete:return dict(status='WAIT_ALL_SUPPORTED_COVERAGE_ARMS',selected=None,promoted=False)
    feasible=[]
    for arm,a in arms.items():
        pairs=[(a['U_Retention']['source_macro'],control['U_Retention']['source_macro']),(a['H_accuracy']['accuracy'],control['H_accuracy']['accuracy']),(a['positive_accuracy']['accuracy'],control['positive_accuracy']['accuracy'])]
        if a['native_accuracy']['accuracy']==1 and all(x is not None and y is not None and x>=y for x,y in pairs) and a['H_Retention']['source_macro'] is not None:feasible.append(arm)
    feasible.sort(key=lambda arm:(-arms[arm]['H_Retention']['source_macro'],-arms[arm]['H_accuracy']['source_macro'],int(arm[1:])))
    return dict(status='DEV_COVERAGE_CANDIDATE_SELECTED' if feasible else 'NO_FEASIBLE_H_COVERAGE',selected=feasible[0] if feasible else 'S0',promoted=bool(feasible),feasible=feasible,fallback='Original support at fixed weights for negative-regression reporting; not a winner' if not feasible else None,CONFIRM_used=False)


def lock_weights(directory):
    d=Path(directory);root=d/'private/run';pub=root/'public'
    assert read(pub/'REPORT_STATUS.json')['all_six_scored'] and read(pub/'EXIT_22B.json')['exit_code']==0
    result=read(pub/'ALL_ARM_RESULTS.json');promotion=result['promotion'];arm=promotion['selected'] or 'E1';H,U=WEIGHTS[arm]
    lock=dict(S0_arm=arm,control_arm='E0' if U==.01 else 'E3',weights=dict(H=H,U=U),parent_B_result_binding=digest(result),B_promotion=promotion,pool_binding=read(d/'private/COVERAGE_POOL_FREEZE.json')['binding'],new_student_judgment_reservation=132,new_reference_reservation_22A=7,steps=320,N=19,order=['S1','S2'],no_winner_fallback=promotion['selected'] is None)
    lock['binding']=digest(lock);path=pub/'COVERAGE_WEIGHT_LOCK.json'
    if path.exists():assert read(path)==lock
    else:write_new(path,lock)
    return lock


def report(directory):
    d=Path(directory);root=d/'private/run';p=root/'private';pub=root/'public';lock=read(pub/'COVERAGE_WEIGHT_LOCK.json')
    original=read(pub/'ALL_ARM_RESULTS.json');assert digest(original)==lock['parent_B_result_binding']
    scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];records=outputs(root);dev=read(p/'DEV_PANEL.json');expected={query_id(r) for r in dev['natives']+dev['rows']}
    arms={'S0':original['arms'][lock['S0_arm']]};private={}
    for arm in ('S1','S2'):
        rows=[r for r in records if r['arm']==arm and r['mode']=='DEV'];assert len(rows)==len({r['query_id'] for r in rows})
        if {r['query_id'] for r in rows}!=expected or any(score_key(r['source'],r['output']) not in scores for r in rows):continue
        panels={'native':[r for r in rows if r['source']['role']=='native'],'H':[r for r in rows if r['source']['role']=='H_eval'],'U':[r for r in rows if r['source']['role']=='U_eval'],'rewrite':[r for r in rows if r['source']['role']=='native_text_extension'],'positive':[r for r in rows if r['source']['role']=='positive_image']}
        a={name+'_accuracy':metric(rr,scores) for name,rr in panels.items()}
        for name in ('native','H','U'):
            a[name+'_Fix']=metric(panels[name],scores,False)
            if name!='native':a[name+'_Retention']=metric(panels[name],scores,True)
        private[arm]=a;arms[arm]={k:{x:y for x,y in v.items() if x!='per_source'} for k,v in a.items()}
    unsupported=(pub/'GENERATED_22C.json').exists() and read(pub/'GENERATED_22C.json')['status']=='UNSUPPORTED_NO_SELECTION_DIFFERENCE'
    complete=(unsupported or set(arms)=={'S0','S1','S2'}) and (pub/'EXIT_22C.json').exists() and read(pub/'EXIT_22C.json')['exit_code']==0
    promotion=select(arms,original['arms'][lock['control_arm']],complete)
    value=dict(arms=arms,control_arm=lock['control_arm'],weights=lock['weights'],S0_exact_reuse=lock['S0_arm'],complete=complete,unsupported_no_selection_difference=unsupported,promotion=promotion,DEV_only=True,patient_study='UNKNOWN')
    write(pub/'COVERAGE_ARM_RESULTS.json',value);write(p/'COVERAGE_SOURCE_METRICS_PRIVATE.json',private)
    # Paired source counts use stable anonymous indices, not raw source identities.
    pairs=[]
    for left,right in [('S1',lock['S0_arm']),('S2','S1'),('S1',lock['control_arm']),('S2',lock['control_arm'])]:
        aa={r['query_id']:r for r in records if r['arm']==left and r['mode']=='DEV'};bb={r['query_id']:r for r in records if r['arm']==right and r['mode']=='DEV'}
        if set(aa)!=expected or set(bb)!=expected or any(score_key(r['source'],r['output']) not in scores for r in list(aa.values())+list(bb.values())):continue
        from collections import Counter,defaultdict
        source_names=sorted({r['source']['source_group'] for r in aa.values()});counts=defaultdict(Counter)
        for q,r in aa.items():
            key=(r['source']['role'],source_names.index(r['source']['source_group'])+1);counts[key][f"{int(scores[score_key(r['source'],r['output'])])}->{int(scores[score_key(bb[q]['source'],bb[q]['output'])])}"]+=1
        pairs.append(dict(A=left,B=right,rows=[dict(role=k[0],source_index=k[1],counts=dict(v)) for k,v in sorted(counts.items())],different_tokens=sum(aa[q]['output']['raw_token_ids']!=bb[q]['output']['raw_token_ids'] for q in aa),same_route=sum(aa[q]['route']['logical_edit_id']==bb[q]['route']['logical_edit_id'] for q in aa)))
    write(pub/'COVERAGE_PAIRED_SOURCE_RESULTS.json',dict(pairs=pairs,natural_routing=True,conditioning_does_not_replace_main=True))
    return value


def freeze_stage23(directory):
    d=Path(directory);pub=d/'private/run/public';c=read(pub/'COVERAGE_ARM_RESULTS.json');w=read(pub/'COVERAGE_WEIGHT_LOCK.json')
    assert c['complete'];confirm=read(d/'private/CONFIRM_QUALIFIED_FREEZE.json')
    candidate=w['S0_arm'] if c['promotion']['selected']=='S0' else c['promotion']['selected'];H,U=w['weights']['H'],w['weights']['U']
    controls=[w['control_arm'],'E1' if U==.01 else 'E4'];banks=list(dict.fromkeys(controls+[candidate]))
    lock=dict(candidate=candidate,candidate_support=c['promotion']['selected'],weights=dict(H=H,U=U),banks=banks,NOH=controls[0],H1=controls[1],candidate_promoted=c['promotion']['promoted'],DEV_selection=digest(c),coverage_lock=w['binding'],CONFIRM_binding=confirm['binding'],N=45,prefixes=[11,19,32,45],new_GPU_limit_seconds=9600,cumulative_GPU_limit_seconds=28800,conservative_new_judgments=len(banks)*298+len(confirm['rows'])*(len(banks)+1),regression_before_CONFIRM=True,semantic_retries=0,scope='Fixed45 DEV regression plus one source-held-out probe panel; no unseen-edit/clinical claim')
    lock['binding']=digest(lock);write_new(d/'private/FINAL_STAGE23_LOCK.json',lock)
    return lock


def selfcheck():
    import copy
    a={k:dict(accuracy=1.,source_macro=1.) for k in ('native_accuracy','U_Retention','H_accuracy','H_Retention','positive_accuracy')}
    assert select({'S0':a,'S1':a,'S2':a},a,False)['selected'] is None
    assert select({'S0':a,'S1':a,'S2':a},a,True)['selected']=='S0'
    bad=copy.deepcopy(a);bad['U_Retention']['source_macro']=.5
    r=select({'S0':bad,'S1':bad,'S2':bad},a,True);assert r['selected']=='S0' and not r['promoted']
    control=copy.deepcopy(a);control['H_Retention']['source_macro']=.3
    s=copy.deepcopy(a);s['H_Retention']['source_macro']=.4
    assert select({'S0':control,'S1':s,'S2':s},control,True)['selected']=='S1'
    print('Coverage promotion selfcheck passed')


if __name__=='__main__':
    if sys.argv[1:] == ['--selfcheck']:selfcheck()
    else:report(sys.argv[1])
