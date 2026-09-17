"""Stage19 existing-output audits, deduplicated counting and reference sensitivities."""
from collections import Counter,defaultdict
import csv
from pathlib import Path
from statistics import mean
from scripts.medtrace.astra_judge_bundle import read,write_new,validate
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import METHODS,query_id,score_key


def old_data(old):
    packages=read(old/'private/DEV_EVALUATION.json')['candidate_packages']
    rows=[r for m in METHODS for r in read(old/'private'/f'{m}_EVALUATION.json')['records']]
    source=read(old/'private/STUDENT_JUDGE_INPUTS.json');verdict=read(old/'STUDENT_JUDGE/operator/VERDICTS.json')
    if verdict['source_binding']!=digest(source):raise ValueError('Judge source changed')
    batch=dict(batch_id='audit',records=[dict(opaque_query_id=digest(r)) for r in source['records']])
    new={r['opaque_query_id']:r['is_correct'] for r in validate(batch,dict(batch_id='audit',decisions=verdict['decisions']))}
    scores={k:v['correct'] for k,v in read(old/'private/SCORE_CACHE.json')['cached'].items()}
    scores.update({score_key(r['source'],r['output']):new[digest(r)] for r in source['records']})
    return packages,rows,scores


def audit(old,private,public):
    packages,rows,scores=old_data(old);ids=[p['candidate_id'] for p in packages]
    score=lambda row,out:scores[score_key(row,out)]
    lookup={(r['method'],r['mode'],r['prefix'],r['inserted'][0] if r['mode']=='single' else '',r['query_id']):r for r in rows}
    h={query_id(r):r for p in packages for r in p['evaluation'] if r['role']=='H_eval'}
    qnames={q:f'H{i:02d}' for i,q in enumerate(sorted(h),1)};images=sorted({r['image_sha256'] for r in h.values()});inames={v:f'I{i:02d}' for i,v in enumerate(images,1)}
    aliases={e:f'E{i:02d}' for i,e in enumerate(ids,1)};links=[];weights=defaultdict(float)
    for i,p in enumerate(packages):
        edit=p['candidate_id'];hs=[r for r in p['evaluation'] if r['role']=='H_eval']
        for r in hs:
            q=query_id(r);weights[q]+=1/len(packages)/len(hs)
            for method in METHODS:
                single=lookup[method,'single',1,edit,q];final=lookup[method,'sequential',16,'',q]
                prefix={str(n):lookup[method,'sequential',n,'',q] for n in [1,10,16] if i<n}
                links.append(dict(edit=edit,method=method,query_id=q,source=r,native=p['training']['native'],training_H=p['training']['H_fit'],
                    Base=score(r,final['Base']),single=score(r,single['R0']),final=score(r,final['R0']),
                    single_route=single['route'],final_route=final['route'],prefix={n:dict(correct=score(r,v['R0']),route=v['route']) for n,v in prefix.items()}))
    transitions={};finals={};qa_rows=[]
    for method in METHODS:
        ls=[r for r in links if r['method']==method];c=Counter((r['single'],r['final']) for r in ls)
        transitions[method]=dict(correct_to_correct=c[True,True],correct_to_wrong=c[True,False],wrong_to_correct=c[False,True],wrong_to_wrong=c[False,False],
            selected_expert_changed=sum(r['single_route']['nearest_logical_edit_id']!=r['final_route']['nearest_logical_edit_id'] for r in ls),slots=len(ls))
        values={q:score(r,lookup[method,'sequential',16,'',q]['R0']) for q,r in h.items()};finals[method]=values
        source_values=[mean(v for q,v in values.items() if h[q]['image_sha256']==im) for im in images]
        for q,r in h.items():
            contexts=[x for x in ls if x['query_id']==q]
            qa_rows.append(dict(method=method,QA=qnames[q],image=inames[r['image_sha256']],associations=len(contexts),edit_macro_weight=weights[q],
                Base=contexts[0]['Base'],single_context_accuracy=mean(x['single'] for x in contexts),final=values[q],final_expert=aliases[lookup[method,'sequential',16,'',q]['route']['nearest_logical_edit_id']]))
        transitions[method].update(final_unique_QA_accuracy=mean(values.values()),final_source_macro=mean(source_values),
            final_original_weighted=sum(weights[q]*v for q,v in values.items()),single_unique_QA_context_macro=mean(mean(x['single'] for x in ls if x['query_id']==q) for q in h))
    improvements=[]
    for control in METHODS:
        if control=='C_FACT':continue
        changes=[]
        for p in packages:
            qs=[query_id(r) for r in p['evaluation'] if r['role']=='H_eval']
            if mean(finals['C_FACT'][q] for q in qs)>mean(finals[control][q] for q in qs):changes.append(p['candidate_id'])
        improved={q for q in h if finals['C_FACT'][q] and not finals[control][q]}
        improvements.append(dict(control=control,improved_edits=len(changes),distinct_improved_H_QA=len(improved),distinct_H_images=len({h[q]['image_sha256'] for q in improved}),
            improved_QA=[qnames[q] for q in sorted(improved)],improved_expert_aliases=[aliases[e] for e in changes]))
    sensitivity=[]
    def measure(method,excluded,tag):
        keep={q for q,r in h.items() if q not in excluded};values=finals[method];peredit=[];slots=0
        for p in packages:
            qs=[query_id(r) for r in p['evaluation'] if r['role']=='H_eval' and query_id(r) in keep]
            if qs:peredit.append(mean(values[q] for q in qs));slots+=len(qs)
        ims={h[q]['image_sha256'] for q in keep}
        return dict(method=method,excluded=tag,remaining_edits=len(peredit),H_slots=slots,unique_QA=len(keep),source_images=len(ims),
            edit_macro=mean(peredit) if peredit else None,unique_QA_macro=mean(values[q] for q in keep) if keep else None,
            source_macro=mean(mean(values[q] for q in keep if h[q]['image_sha256']==im) for im in ims) if ims else None,
            original_weights_renormalized=sum(weights[q]*values[q] for q in keep)/sum(weights[q] for q in keep) if keep else None)
    for method in METHODS:
        sensitivity.append(measure(method,set(),'NONE'))
        for im in images:sensitivity.append(measure(method,{q for q in h if h[q]['image_sha256']==im},inames[im]))
    review=read(private/'BLIND_REFERENCE_REVIEW.json');uncertain={query_id(r) for r in review['records'] if r['verdict']!='SUPPORTED'}&set(h)
    ref_sensitivity=[measure(m,uncertain,'UNVERIFIED_REFERENCE_EXCLUDED_DESCRIPTIVE_ONLY') for m in METHODS]
    u=next(r for r in packages[0]['evaluation'] if r['role']=='U_eval');uq=query_id(u);urows=[];private_u=[]
    for method in METHODS:
        r=lookup[method,'sequential',16,'',uq]
        urows.append(dict(method=method,unique_inputs=1,Base=score(u,r['Base']),final=score(u,r['R0']),selected_expert=aliases[r['route']['nearest_logical_edit_id']],activated=r['route']['activated']))
        private_u.append(dict(method=method,source=u,output=r['R0'],route=r['route']))
    anchors=[]
    for method in METHODS:
        rs=[x for x in links if x['method']==method and x['edit']==ids[0]]
        anchors.append(dict(method=method,anchor_edits=1,H_slots=len(rs),prefix={n:mean(x['prefix'][str(n)]['correct'] for x in rs) for n in [1,10,16]},
            expert_changes=[dict(prefix=n,selected=[aliases[x['prefix'][str(n)]['route']['nearest_logical_edit_id']] for x in rs]) for n in [1,10,16]]))
    result=dict(status='EXISTING_OUTPUT_AUDIT_COMPLETE',old_results_unchanged=True,N=16,H_unique_QA=7,H_images=4,
        transitions=transitions,FACT_improvement_dependencies=improvements,QA_weights=qa_rows,U_unique=urows,anchors=anchors,
        reference_review=dict(type='AI',supported=sum(r['verdict']=='SUPPORTED' for r in review['records']),unverified=sum(r['verdict']=='UNVERIFIED' for r in review['records']),
            affected_H_QA=[qnames[q] for q in sorted(uncertain)],all_method_sensitivity=ref_sensitivity,original_references_and_scores_unchanged=True),
        inference_boundary='No patient-population CI from four evaluation images; leave-one changes are sensitivity only; single contexts are not identical inputs to identical banks')
    result['early10_fixed_anchors']=[]
    for method in METHODS:
        fixed=[r for r in links if r['method']==method and r['edit'] in ids[:10]]
        result['early10_fixed_anchors'].append(dict(method=method,edits=10,H_slots=len(fixed),unique_QA=len({r['query_id'] for r in fixed}),
            prefix={str(n):dict(H_slot_accuracy=mean(r['prefix'][str(n)]['correct'] for r in fixed),edit_macro=mean(mean(r['prefix'][str(n)]['correct'] for r in fixed if r['edit']==e) for e in ids[:10])) for n in (10,16)},
            selected_expert_changes=sum(r['prefix']['10']['route']['nearest_logical_edit_id']!=r['prefix']['16']['route']['nearest_logical_edit_id'] for r in fixed)))
    write_new(private/'QA_ALIASES.json',dict(queries=qnames,images=inames,experts=aliases))
    write_new(private/'DEV16_ROUTE_TRANSITIONS_PRIVATE.json',dict(links=links,U=private_u))
    write_new(public/'DEV16_ROUTE_AND_TRANSITION_AUDIT.json',result)
    with (public/'DEV16_UNIQUE_SOURCE_SENSITIVITY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sensitivity[0]));w.writeheader();w.writerows(sensitivity)
    return result,sensitivity


def matrix_report(old,private,public):
    packages,rows,scores=old_data(old);plan=read(private/'DIAGNOSTIC_PLAN.json')
    source=read(private/'MATRIX_JUDGE_INPUTS.json');v=read(private/'MATRIX_JUDGE/operator/VERDICTS.json')
    ev=read(private/'MATRIX_JUDGE/operator/execution_evidence/b000.json')
    if v['source_binding']!=digest(source) or ev['status']!='FORMAT_VALID' or ev['tool_event_types'] or not all(ev['isolation_checks'].values()):raise ValueError('Judge binding/isolation failed')
    verdict={r['opaque_query_id']:r['is_correct'] for r in validate(dict(batch_id='b000',records=[dict(opaque_query_id=digest(r)) for r in source['records']]),dict(batch_id='b000',decisions=v['decisions']))}
    for r in source['records']:
        key=score_key(r['source'],r['output'])
        if key in scores:raise ValueError('Old text judged again')
        scores[key]=verdict[digest(r)]
    new=read(private/'diagnostic_run/NEW_MATRIX_OUTPUTS.json')
    if new['plan']!=plan['freeze_id']:raise ValueError('Matrix plan mismatch')
    matrix=plan['reused']+new['records'];by={(r['query_id'],r['expert']):r for r in matrix}
    expected={(q,e) for q in plan['queries'] for e in plan['experts']}
    if len(by)!=128 or set(by)!=expected:raise ValueError('Incomplete matrix')
    aliases=read(private/'QA_ALIASES.json');qnames=aliases['queries'];enames=aliases['experts']
    scope=read(private/'BLIND_SCOPE_REVIEW.json');relations={(str(r['source_qid']),r['expert']):r['relation'] for r in scope['records']}
    links=read(private/'DEV16_ROUTE_TRANSITIONS_PRIVATE.json')['links'];summaries=[];slot_cases=[];cells=[]
    for q,row in plan['queries'].items():
        results={e:scores[score_key(row,by[q,e]['output'])] for e in plan['experts']}
        final=next(r for r in rows if r['method']=='C_FACT' and r['mode']=='sequential' and r['prefix']==16 and r['query_id']==q)
        selected=final['route']['nearest_logical_edit_id'];good=[enames[e] for e in plan['experts'] if results[e]]
        if by[q,selected]['output']['raw_token_ids']!=final['R0']['raw_token_ids']:raise ValueError('Final selected expert differs')
        base=scores[score_key(row,final['Base'])]
        summaries.append(dict(QA=qnames.get(q,'U01'),role=row['role'],Base=base,selected=enames[selected],final=results[selected],correct_experts=len(good),experts=16,correct_expert_aliases=good,all_experts_wrong=not good,
            oracle_only=bool(good),selected_scope=relations.get((str(row['source_qid']),selected),'NOT_REVIEWED_U')))
        for e,ok in results.items():cells.append(dict(QA=qnames.get(q,'U01'),expert=enames[e],correct=ok,scope=relations.get((str(row['source_qid']),e),'NOT_REVIEWED_U'),reused=(q,e) not in {(r['query_id'],r['expert']) for r in new['records']}))
        for link in [r for r in links if r['method']=='C_FACT' and r['query_id']==q]:
            own=link['edit'];old_correct=results[own]
            if old_correct!=link['single']:raise ValueError('Restored own score drift')
            case='PRESERVED_ORIGINAL_CORRECT' if old_correct and results[selected] else 'COMPETITION_CORRECT_OWN_REPLACED' if old_correct else 'CROSS_EXPERT_CORRECT_AVAILABLE' if good else 'ALL_EXPERTS_WRONG'
            slot_cases.append(dict(QA=qnames[q],original=enames[own],selected=enames[selected],own_correct=old_correct,final_correct=results[selected],case=case))
    result=dict(status='SCORED_COMPLETE',matrix_combinations=128,reused_combinations=41,new_combinations=87,new_Judge_texts=len(source['records']),queries=summaries,
        slot_classifications=dict(Counter(r['case'] for r in slot_cases)),slots=slot_cases,scope_counts=dict(Counter(r['relation'] for r in scope['records'])),
        clinical_signoff=False,reference_sensitivity='H01 unverified; original scores preserved',oracle_deployment=False,new_training=False,
        gpu=read(private/'diagnostic_run/public/DIAGNOSTIC_RESULT.json'),judge_seconds=ev.get('wall_seconds'),judge_usage=ev.get('usage'))
    write_new(public/'EXPERT_MATRIX_DIAGNOSTIC.json',result)
    with (public/'EXPERT_MATRIX_DIAGNOSTIC.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    write_new(private/'MATRIX_SCORE_CLOSEOUT.json',dict(scores=scores,source_binding=digest(source),verdict_binding=digest(v),plan=plan['freeze_id'],complete=True))
    return result
