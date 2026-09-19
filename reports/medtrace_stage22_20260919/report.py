"""All-arm DEV report and preregistered matching-U promotion, without CONFIRM access."""
import json,sys
from pathlib import Path
from collections import defaultdict
from statistics import mean
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage18_score import score_key,query_id
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs
WEIGHTS={'E0':(0.,.01),'E1':(1.,.01),'E2':(.25,.01),'E3':(0.,.05),'E4':(1.,.05),'E5':(.25,.05)}


def metric(rows,scores,base_filter=None):
    rows=[r for r in rows if base_filter is None or scores.get(score_key(r['source'],r['Base'])) is base_filter]
    values=[scores.get(score_key(r['source'],r['output'])) for r in rows];complete=all(type(v) is bool for v in values)
    groups=defaultdict(list)
    for r,v in zip(rows,values):groups[r['source']['source_group']].append(v)
    sources={g:mean(v) for g,v in groups.items()} if complete else {}
    loo=[mean([v for h,v in sources.items() if h!=g]) for g in sources] if len(sources)>1 else []
    return dict(leave_one_source_out=[min(loo),max(loo)] if loo else None,correct=sum(v is True for v in values),N=len(rows),coverage=sum(type(v) is bool for v in values),accuracy=mean(values) if complete and rows else None,source_macro=mean(sources.values()) if sources else None,sources=len(groups),per_source=sources)


def select(arms):
    if set(arms)!=set(WEIGHTS):return dict(status='WAIT_ALL_SIX_VALID_ARMS',selected=None)
    feasible=[]
    for arm in ('E1','E2','E4','E5'):
        ref=arms['E0' if WEIGHTS[arm][1]==.01 else 'E3'];a=arms[arm]
        comparisons=[(a['U_Retention']['source_macro'],ref['U_Retention']['source_macro']),(a['H_accuracy']['accuracy'],ref['H_accuracy']['accuracy']),(a['positive_accuracy']['accuracy'],ref['positive_accuracy']['accuracy'])]
        if a['native_accuracy']['accuracy']==1 and all(x is not None and y is not None and x>=y for x,y in comparisons) and a['H_Retention']['source_macro'] is not None:feasible.append(arm)
    feasible.sort(key=lambda a:(-arms[a]['H_Retention']['source_macro'],-arms[a]['H_accuracy']['source_macro'],WEIGHTS[a][0],WEIGHTS[a][1],a))
    return dict(status='DEV_H_CANDIDATE_SELECTED' if feasible else 'NO_FEASIBLE_H_WEIGHT',selected=feasible[0] if feasible else None,feasible=feasible,coverage_fallback=dict(H=1.,U=.01) if not feasible else None,CONFIRM_used=False)


def report(root):
    root=Path(root);p=root/'private';pub=root/'public';dev=read(p/'DEV_PANEL.json');scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];allrows=outputs(root);arms={};private={};routes=[]
    expected={query_id(r) for r in dev['natives']+dev['rows']}
    for arm in WEIGHTS:
        rr=[r for r in allrows if r['arm']==arm];assert len(rr)==len({r['query_id'] for r in rr})
        if {r['query_id'] for r in rr}!=expected:continue
        if any(score_key(r['source'],r['output']) not in scores for r in rr):continue
        panels={'native':[r for r in rr if r['source']['role']=='native'],'H':[r for r in rr if r['source']['role']=='H_eval'],'U':[r for r in rr if r['source']['role']=='U_eval'],'rewrite':[r for r in rr if r['source']['role']=='native_text_extension'],'positive':[r for r in rr if r['source']['role']=='positive_image']}
        result={name+'_accuracy':metric(rows,scores) for name,rows in panels.items()}
        for name in ('native','H','U'):
            result[name+'_Fix']=metric(panels[name],scores,False)
            if name!='native':result[name+'_Retention']=metric(panels[name],scores,True)
        private[arm]=result;arms[arm]={k:{x:y for x,y in v.items() if x!='per_source'} for k,v in result.items()};routes.append(dict(arm=arm,queries=66,activated=sum(r['route']['activated'] for r in rr)))
    promotion=select(arms);write(pub/'ALL_ARM_RESULTS.json',dict(N=19,arms=arms,promotion=promotion,exposure='VIEWED_DEV',patient_study='UNKNOWN',p_values=None))
    write(p/'SOURCE_METRICS_PRIVATE.json',private)
    comparisons=[]
    for arm in ('E1','E2','E4','E5'):
        control='E0' if WEIGHTS[arm][1]==.01 else 'E3'
        aa={r['query_id']:r for r in allrows if r['arm']==arm};bb={r['query_id']:r for r in allrows if r['arm']==control}
        if arm not in arms or control not in arms:continue
        counts=defaultdict(lambda:defaultdict(int));source_groups=sorted({r['source']['source_group'] for r in aa.values()})
        for q,a in aa.items():
            b=bb[q];role=a['source']['role'];key=f"{int(scores[score_key(a['source'],a['output'])])}->{int(scores[score_key(b['source'],b['output'])])}";source=source_groups.index(a['source']['source_group'])+1;counts[(role,source)][key]+=1
        comparisons.append(dict(arm=arm,control=control,same_route=sum((aa[q]['route']['logical_edit_id'],aa[q]['route']['activated'])==(bb[q]['route']['logical_edit_id'],bb[q]['route']['activated']) for q in aa),different_tokens=sum(aa[q]['output']['raw_token_ids']!=bb[q]['output']['raw_token_ids'] for q in aa),paired_source=[dict(role=k[0],source_index=k[1],counts=v,orientation='H_arm correctness -> matching_NOH correctness') for k,v in sorted(counts.items())]))
    write(pub/'PAIRED_SOURCE_RESULTS.json',dict(comparisons=comparisons,natural_routing=True,activation=routes))
    write(pub/'REPORT_STATUS.json',dict(complete_arms=list(arms),all_six_scored=len(arms)==6,promotion=promotion,stage23_started=False))
    return arms

if __name__=='__main__':report(Path(sys.argv[1]))
