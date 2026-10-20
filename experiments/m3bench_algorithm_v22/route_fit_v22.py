"""Frozen DEV-only fitting and leave-image-group-out threshold calibration."""
import torch
from route_math_v22 import features,projection,fit_heads,select_threshold,choose,risk_target

def fit_predict(plan,keys,dev_verdict,r0,budget=lambda:None):
    experts=[t['key'] for t in plan['tasks']];queries=sorted(plan['queries']);native={t['key']:t['source']['query_id'] for t in plan['tasks']}
    dev=[q for q in queries if plan['query_roles'][q]=='DEV'];held=[q for q in queries if q not in dev]
    assert dev and held and set(keys)==set(queries)
    # This function cannot access a held-out answer, even accidentally.
    expected={(e,q) for q in dev for e in ['BASE']+experts};assert set(dev_verdict)==expected
    matrix=projection(keys[queries[0]]['image'].numel());audit=[]
    def model(kind,excluded_group=None):
        budget()
        train=[q for q in dev if plan['query_groups'][q]!=excluded_group];assert train
        prototypes={}
        for e in experts:
            budget()
            yes={native[e]}|{q for q in train if plan['relations'][e+'|'+q]=='YES'}
            no={q for q in train if plan['relations'][e+'|'+q]=='NO'}
            prototypes[e]=([keys[q] for q in sorted(yes)],[keys[q] for q in sorted(no)])
        def vector(e,q):
            positive,negative=prototypes[e]
            return features(keys[q],keys[native[e]],positive,negative,kind,matrix)
        rows=[];app=[];risk=[];risk_pairs=[]
        for e in experts:
            # Native anchor labels are common to all routers; native answers are not extra risk labels.
            source=native[e];rows.append(vector(e,source));app.append(1.);risk.append(None)
            for q in train:
                relation=plan['relations'][e+'|'+q]
                rows.append(vector(e,q));app.append(None if q==source else 1. if relation=='YES' else -1. if relation=='NO' else None)
                risk.append(risk_target(dev_verdict['BASE',q],dev_verdict[e,q]));risk_pairs.append((e,q))
        comp,harm,status=fit_heads(torch.stack(rows),app,risk)
        audit.append(dict(kind=kind,excluded_group=excluded_group,training_query_ids=train,risk_pairs=risk_pairs,risk_status=status,dimensions=len(comp)))
        def scores(q):
            return [float(vector(e,q)@comp)-max(0.,float(vector(e,q)@harm)) for e in experts]
        return scores
    result={}
    for kind in ('SIMPLE','CONDITIONAL'):
        scores={}
        for group in sorted({plan['query_groups'][q] for q in dev}):
            predict=model(kind,group)
            for q in dev:
                if plan['query_groups'][q]==group:scores[q]=predict(q)
        threshold=select_threshold(dev,scores,experts,dev_verdict,r0)
        predict=model(kind);all_scores={q:predict(q) for q in queries}
        selected={q:choose(all_scores[q],threshold['tau']) for q in queries} if threshold['status']=='PASS' else None
        result[kind]=dict(threshold=threshold,dev_leave_group_out_scores=scores,all_scores=all_scores,selected=selected)
    return dict(methods=result,fit_audit=audit,experts=experts,dev_queries=dev,heldout_queries=held,no_heldout_answer_access=True)

def selfcheck():
    # Multiple images per group; changing a held-out key may change its prediction,
    # but cannot change the fitted DEV threshold or its leave-group-out scores.
    from route_math_v22 import risk_target
    qids=['n0','n1','d0','d1','h0'];roles={q:'DEV' if q in ('d0','d1') else 'HELDOUT_DEVELOPMENT' for q in qids}
    plan=dict(tasks=[dict(key='E0',source=dict(query_id='n0')),dict(key='E1',source=dict(query_id='n1'))],queries={q:{} for q in qids},query_roles=roles,query_groups={q:q for q in qids},relations={e+'|'+q:('YES' if (e,q) in [('E0','d0'),('E1','d1')] else 'NO') for e in ('E0','E1') for q in qids})
    keys={q:{k:torch.tensor([1.+i,.2+i/10,.5]) for k in ('joint','image','text')} for i,q in enumerate(qids)}
    verdict={(e,q):'INCORRECT' for e in ('BASE','E0','E1') for q in ('d0','d1')};verdict['E0','d0']='CORRECT';r0={'d0':0,'d1':None}
    one=fit_predict(plan,keys,verdict,r0)
    keys['h0']={k:torch.tensor([-1.,.2,.5]) for k in ('joint','image','text')};two=fit_predict(plan,keys,verdict,r0)
    for kind in ('SIMPLE','CONDITIONAL'):
        assert one['methods'][kind]['threshold']==two['methods'][kind]['threshold']
        assert one['methods'][kind]['dev_leave_group_out_scores']==two['methods'][kind]['dev_leave_group_out_scores']
    for audit in one['fit_audit']:
        assert all(roles[q]=='DEV' and plan['query_groups'][q]!=audit['excluded_group'] for _,q in audit['risk_pairs'])
    try:fit_predict(plan,keys,dict(verdict,extra='CORRECT'),r0)
    except AssertionError:pass
    else:raise AssertionError('Unexpected answer access accepted')
    return dict(status='PASS',whole_group_exclusion=True,heldout_feature_change_cannot_change_DEV_calibration=True,heldout_verdicts_rejected=True,new_model_calls=0)
if __name__=='__main__':
    torch.set_num_threads(1);print(selfcheck())
