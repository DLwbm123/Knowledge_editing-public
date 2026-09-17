"""One bounded cache-first source pass, then a fresh Base mask; no student access."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import train_row, conflict, validate_task
from scripts.medtrace.prepare_stage2_sources import normalized, reviewed_attribute
from scripts.medtrace.stage18_score import query_id, score_key


def prepare(old19,old18,root):
    old19,old18,root=map(Path,(old19,old18,root));p=root/'private'
    ledger=read(old19/'DEV_SOURCE_ROLE_LEDGER.json');frozen=read(p/'CANDIDATE_FREEZE.json')
    reviews={r['source_qid']:r for f in (old19/'V3_TRAIN_REFERENCE_REVIEW.json',p/'REFERENCE_REUSED.json',p/'REFERENCE_REVIEW_RESULT.json') for r in read(f)['records']}
    natives=[r for r in frozen['rows'] if reviews[r['source_qid']]['verdict']=='SUPPORTED']
    auxiliary=[r for r in reviews.values() if r['source_group'] in ledger['auxiliary_groups'] and r['source_group'].split(':')[-1] not in ledger['quarantined_groups'] and r['verdict']=='SUPPORTED']
    packet=read(old19/'DEV_EVALUATION_V3.json');extensions=read(old19/'EXTENSIONS_V3.json')
    anchors=[p['training'] for p in packet['candidate_packages']];byqid={t['native']['source_qid']:t for t in anchors}
    hu=Counter();uu=Counter();tasks=[]
    for n in [t['native'] for t in anchors]+[r for r in natives if r['source_qid'] not in byqid]:
        if n['source_qid'] in byqid: tasks.append(byqid[n['source_qid']]);continue
        us=[r for r in auxiliary if normalized(r['question'])!=normalized(n['question']) and reviewed_attribute(r['question'])!=reviewed_attribute(n['question'])]
        hs=[r for r in auxiliary if conflict(n,r) and hu[r['source_qid']]<3]
        if not us:continue
        u=min(us,key=lambda r:(uu[r['source_group']],digest([19001,r['source_qid']])));uu[u['source_group']]+=1
        h=min(hs,key=lambda r:(hu[r['source_qid']],digest([19001,r['source_qid']]))) if hs else None
        if h:hu[h['source_qid']]+=1
        edit='stage19-fasttrack:'+str(n['source_qid']);q=n['question']
        t=dict(canonical_edit_id=edit,order=len(tasks)+1,seed=int(digest([20260912,edit])[:8],16),native=train_row(n,'native'),
               fit_questions=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}'],
               H_fit=[train_row(h,'H_fit')] if h else [],G_fit=[],U_fit=[train_row(u,'U_fit')])
        validate_task(t,fasttrack_branch='C_FACT' if h else 'C_NO_H');tasks.append(t)
    core={query_id(r):r for package in packet['candidate_packages'] for r in package['evaluation']}
    for link in extensions['text_links']:
        if link['role']=='native_text_extension':
            r=dict(extensions['text_queries'][link['query_id']],role=link['role']);core[link['query_id']]=r
    if len(core)!=26:raise ValueError('Fixed core panel changed')
    protected=set(ledger['evaluation_sources']);ng=set(ledger['native_groups']);ag=set(ledger['auxiliary_groups'])
    if ng&ag or ng&protected or ag&protected:raise ValueError('Global source role overlap')
    for task in tasks:
        if task['native']['source_group'] not in ng:raise ValueError('Native source outside global partition')
        if any(r['source_group'] not in ag for role in ('H_fit','G_fit','U_fit') for r in task[role]):raise ValueError('Auxiliary source outside global partition')
    base={r['query_id']:r for f in (old18/'QUALIFICATION_BASE_OUTPUTS.json',old19/'base_run/BASE_OUTPUTS.json',old19/'extension_base_run/BASE_OUTPUTS.json') for r in read(f)['records']}
    rows={query_id(t['native']):t['native'] for t in tasks};rows.update(core)
    write_new(p/'BASE_QUEUE.json',dict(rows=list(rows.values()),freeze_id=digest(list(rows.values())),source_only=True))
    write_new(p/'STRUCTURAL_TASKS.json',dict(tasks=tasks,anchors=[t['canonical_edit_id'] for t in anchors],core_rows=list(core.values()),anchor_packages=packet['candidate_packages'],
        tasks_binding=digest(tasks),source_role_binding=digest(ledger),candidate_binding=frozen['freeze_id'],source_review_binding=digest(reviews),no_student_access=True))
    write_new(p/'EXISTING_BASE_OUTPUTS.json',dict(records=[base[q] for q in rows]))
    scores={}
    for f in (old19/'BASE_SCORE_CLOSEOUT.json',old19/'EXTENSION_SCORE_CLOSEOUT.json',old19/'MATRIX_SCORE_CLOSEOUT.json'):
        d=read(f);scores.update(d.get('all_scores',d.get('scores',{})))
    write_new(p/'EXISTING_SCORE_CACHE.json',dict(scores=scores,protocol='MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1',reuse='Exact image/question/reference/raw answer; generation consumers retain full separate bindings'))
    report=dict(source_candidates=len(frozen['rows']),source_supported=len(natives),structural_tasks=len(tasks),new_reference_items=len(read(p/'REFERENCE_REVIEW_RESULT.json')['records']),
                reused_reference_items=len(read(p/'REFERENCE_REUSED.json')['records']),fresh_Base_queries=len(rows),sealed_access=False,clinical_signoff=False,patient_study='UNKNOWN')
    write_new(root/'public/BOUNDED_SOURCE_SCAN.json',report)
    return report


def finalize(root):
    root=Path(root);p=root/'private';structural=read(p/'STRUCTURAL_TASKS.json');fresh=read(p/'FRESH_BASE_OUTPUTS.json')
    scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];base={r['query_id']:r for r in fresh['records']}
    eligible=[];flow=[]
    for t in structural['tasks']:
        r=base[query_id(t['native'])];key=score_key(t['native'],r['output']);correct=scores.get(key)
        if correct is None:raise ValueError('Qualification incomplete')
        flow.append(dict(task=t['canonical_edit_id'],Base_correct=correct,H_supported=bool(t['H_fit']),source_group=t['native']['source_group'],Base_binding=digest(r)))
        if not correct:eligible.append(t)
    pure=[t for t in eligible if t['H_fit']];track='P' if len(pure)>=50 else 'S'
    selected=pure if track=='P' else eligible
    if [t['canonical_edit_id'] for t in selected[:11]]!=structural['anchors']:
        raise ValueError('Fresh migration invalidates an anchor; preserve paired DEV scope for explicit repair')
    selected=selected[:100]
    for i,t in enumerate(selected,1):
        t['order']=i
        if track=='S' and i>11:t['H_fit']=[];t['G_fit']=[]
    # Core H remains image-indexed, outside all training sources. U role follows frozen predicate matching.
    roles=[]
    for r in structural['core_rows']:
        affected=[t['canonical_edit_id'] for t in selected if r['role']=='U_eval' and (normalized(r['question'])==normalized(t['native']['question']) or reviewed_attribute(r['question'])==reviewed_attribute(t['native']['question']))]
        roles.append(dict(query_id=query_id(r),role=r['role'],U_scope_overlap_edits=affected,policy='Always generate; strict U retention excludes predeclared predicate-overlap rows at affected prefixes, full fixed-panel accuracy retained'))
    stream=dict(tasks=selected,anchors=structural['anchors'],track=track,core_rows=structural['core_rows'],anchor_packages=structural['anchor_packages'],role_transitions=roles,
                qualification_binding=digest(fresh),score_binding=digest(scores),order='11 frozen anchors then source-only candidate order',source_role_binding=structural['source_role_binding'])
    write_new(p/'STREAM.json',stream)
    write_new(p/'TRAINING_ELIGIBILITY.json',dict(flow=flow,stream_binding=digest(stream),selection_before_students=True))
    summary=dict(track=track,N=len(selected),K=11 if track=='S' else len(selected),pure_FACT_qualified=len(pure),all_Base_wrong_qualified=len(eligible),background_newly_trained=len(selected)-11 if track=='S' else 0,
                 native_QA=len(selected),native_sources=len({t['native']['source_group'] for t in selected}),compatible_old_background_reused=0,stream_binding=digest(stream),status='FROZEN_FOR_TRAINING',clinical_signoff=False,patient_study='UNKNOWN')
    write_new(root/'public/TRAINING_ELIGIBILITY.json',summary);write_new(root/'public/STREAM_MANIFEST.json',summary)
    return summary


if __name__=='__main__':
    if sys.argv[1]=='prepare':print(prepare(*sys.argv[2:]))
    elif sys.argv[1]=='finalize':print(finalize(sys.argv[2]))
    else:raise ValueError('Unknown operation')
