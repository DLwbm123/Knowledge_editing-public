"""Stage23 counts and source-level comparisons; CONFIRM never changes configuration."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.stage20_closeout import outputs,expected
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from report import metric


def gate(candidate,noh,h1,promoted):
    if not promoted:return dict(status='NO_DEV_CANDIDATE_PROMOTED',passed=False)
    keys=['H_eval_accuracy','H_eval_Retention','U_eval_strict_Retention','positive_image_accuracy']
    if any(a.get(k,{}).get('source_macro') is None for a in (candidate,noh,h1) for k in keys):return dict(status='INSUFFICIENT_CONFIRM_SUPPORT',passed=None)
    passed=(candidate['H_eval_Retention']['source_macro']>=noh['H_eval_Retention']['source_macro']+.1-1e-12
            and candidate['H_eval_Retention']['source_macro']>=h1['H_eval_Retention']['source_macro']
            and candidate['H_eval_accuracy']['source_macro']>=max(noh['H_eval_accuracy']['source_macro'],h1['H_eval_accuracy']['source_macro'])
            and candidate['U_eval_strict_Retention']['source_macro']>=noh['U_eval_strict_Retention']['source_macro']
            and candidate['positive_image_accuracy']['source_macro']>=noh['positive_image_accuracy']['source_macro'])
    return dict(status='PROBE_THRESHOLDS_MET' if passed else 'TRADEOFF_NOT_CONFIRMED',passed=passed,clinical_or_unseen_edit_confirmation=False)


def report(directory):
    d=Path(directory);root=d/'private/run';p=root/'private';pub=root/'public'
    lock=read(d/'private/FINAL_STAGE23_LOCK.json');stream=read(p/'STREAM.json');panel=read(d/'private/CONFIRM_QUALIFIED_FREEZE.json')
    scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];allrows=outputs(root);records=[r for r in allrows if r['arm'].startswith('R_')]
    def complete(rows,ids):
        return len(rows)==len(ids) and {r['query_id'] for r in rows}==ids and all(type(scores.get(score_key(r['source'],r[field]))) is bool for r in rows for field in ('Base','output'))
    def unrelated(row,n):
        q=row['source']['question'];a=reviewed_attribute(q)
        return not any(normalized(q)==normalized(t['native']['question']) or a is not None and a==reviewed_attribute(t['native']['question']) for t in stream['tasks'][:n])
    def metrics(rows,n):
        result={}
        for role in ('native','H_eval','U_eval','native_text_extension','positive_image'):
            rr=[r for r in rows if r['source']['role']==role]
            result[role+'_accuracy']=metric(rr,scores)
            if role in ('native','H_eval','U_eval'):
                result[role+'_Fix']=metric(rr,scores,False)
                if role!='native':result[role+'_Retention']=metric(rr,scores,True)
            if role=='U_eval':result[role+'_strict_Retention']=metric([r for r in rr if unrelated(r,n)],scores,True)
        return result
    public=[];private=[];available={arm:[] for arm in lock['banks']};confirm={}
    def store(arm,n,panel_name,rr):
        value=metrics(rr,n);private.append(dict(arm=arm,N=n,panel=panel_name,metrics=value))
        public.append(dict(arm=arm,N=n,panel=panel_name,metrics={k:{x:y for x,y in v.items() if x!='per_source'} for k,v in value.items()}));return value
    for arm in lock['banks']:
        rr=[r for r in records if r['arm']=='R_'+arm]
        for n in lock['prefixes']:
            endpoint=[r for r in rr if r['mode']=='endpoint' and r['prefix']==n]
            if not complete(endpoint,set(expected(stream,n,final=n==45))):continue
            available[arm].append(n)
            store(arm,n,'native',[r for r in endpoint if r['query_id'] in {query_id(t['native']) for t in stream['tasks'][:n]}])
            for name,key in [('old_DEV','core_rows'),('new_viewed_DEV','new_rows'),('positive_DEV','positive_rows')]:
                ids={query_id(r) for r in stream[key]};subset=[r for r in endpoint if r['query_id'] in ids]
                if subset:store(arm,n,name,subset)
        inserted=[r for r in rr if r['mode']=='insertion']
        if complete(inserted,{query_id(t['native']) for t in stream['tasks']}):store(arm,45,'insertion',inserted)
        probe=[r for r in rr if r['mode']=='CONFIRM']
        if (pub/f'CONFIRM_23_{arm}.json').exists() and complete(probe,{query_id(r) for r in panel['rows']}):confirm[arm]=store(arm,45,'CONFIRM_source_probe',probe)
    common=sorted(set.intersection(*(set(v) for v in available.values())))
    full=all(45 in available[a] for a in lock['banks']) and set(confirm)==set(lock['banks']) and all((p/f'23_{a}_insertions_COMPLETE_SCORE_RECEIPT.json').exists() for a in lock['banks'])
    value=dict(results=public,highest_common_scored_N=max(common,default=0),complete=full,configuration_locked=True,patient_study='UNKNOWN',scope='Viewed DEV regression and limited source-held-out probes; no independent patient/unseen-edit/clinical claim')
    if full:
        value['engineering_gate']=gate(confirm[lock['candidate']],confirm[lock['NOH']],confirm[lock['H1']],lock['candidate_promoted'])
        value['all_native_45_correct']=all(next(r['metrics']['native_accuracy']['correct'] for r in public if r['arm']==a and r['N']==45 and r['panel']=='native')==45 for a in lock['banks'])
        if not value['all_native_45_correct']:value['engineering_gate']=dict(status='NATIVE_REGRESSION_FAILED',passed=False)
    write(pub/'RESULTS_23.json',value);write(p/'SOURCE_RESULTS_23_PRIVATE.json',dict(results=private))
    pairs=[]
    for mode in ('endpoint','CONFIRM'):
        grouped={a:{r['query_id']:r for r in records if r['arm']=='R_'+a and r['prefix']==45 and r['mode']==mode} for a in lock['banks']}
        candidate=grouped[lock['candidate']]
        for control in dict.fromkeys((lock['NOH'],lock['H1'])):
            other=grouped[control]
            if not candidate or set(candidate)!=set(other) or not complete(list(candidate.values()),set(candidate)) or not complete(list(other.values()),set(other)):continue
            from collections import defaultdict,Counter
            sources=sorted({r['source']['source_group'] for r in candidate.values()});counts=defaultdict(Counter)
            for q,r in candidate.items():
                key=(r['source']['role'],sources.index(r['source']['source_group'])+1)
                counts[key][f"{int(scores[score_key(r['source'],r['output'])])}->{int(scores[score_key(other[q]['source'],other[q]['output'])])}"]+=1
            pairs.append(dict(mode=mode,candidate=lock['candidate'],control=control,orientation='candidate correctness -> control correctness',rows=[dict(role=k[0],source_index=k[1],counts=dict(v)) for k,v in sorted(counts.items())],different_tokens=sum(candidate[q]['output']['raw_token_ids']!=other[q]['output']['raw_token_ids'] for q in candidate),route_switches=sum(candidate[q]['route']['logical_edit_id']!=other[q]['route']['logical_edit_id'] for q in candidate)))
    write(pub/'PAIRED_SOURCE_23.json',dict(pairs=pairs,natural_routing=True,conditioning_replaces_main=False))
    return value


def selfcheck():
    a={k:dict(source_macro=.5) for k in ('H_eval_accuracy','H_eval_Retention','U_eval_strict_Retention','positive_image_accuracy')}
    assert gate(a,a,a,True)['passed'] is False
    b={k:dict(v) for k,v in a.items()};b['H_eval_Retention']['source_macro']=.6
    assert gate(b,a,a,True)['passed'] is True
    b['positive_image_accuracy']['source_macro']=None
    assert gate(b,a,a,True)['passed'] is None
    assert gate(a,a,a,False)['status']=='NO_DEV_CANDIDATE_PROMOTED'
    print('Stage23 gate selfcheck passed')


if __name__=='__main__':
    if sys.argv[1:]==['--selfcheck']:selfcheck()
    else:report(sys.argv[1])
