"""Frozen source-qualified queue -> current-Base-qualified pure stream, once."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from scripts.medtrace.stage19_fasttrack_budget import write


def proposition(row):
    return row['image_sha256'],reviewed_attribute(row['question']) or normalized(row['question'])


def select(original,candidates,base_correct,maximum=100):
    """Preserve legacy prefix even if cross-device Base correctness drifts."""
    tasks=deepcopy(original);seen={proposition(t['native']) for t in original};counts=Counter();flow=[]
    assert len(tasks)==19
    for t in candidates:
        n=t['native'];q=query_id(n);value=base_correct.get(q)
        if type(value) is not bool:raise ValueError('Missing semantic current Base verdict')
        why=('BASE_CORRECT' if value else 'DUPLICATE_NATIVE_PROPOSITION' if proposition(n) in seen else 'NEW_SOURCE_CAP2' if counts[n['source_group']]>=2 else 'MAXIMUM100' if len(tasks)>=maximum else 'ELIGIBLE')
        flow.append(dict(query_id=q,reason=why))
        if why=='ELIGIBLE':
            t=deepcopy(t);t['order']=len(tasks)+1;tasks.append(t);seen.add(proposition(n));counts[n['source_group']]+=1
    assert tasks[:19]==original and max(counts.values(),default=0)<=2
    return tasks,flow


def prepare_base(root):
    root=Path(root);p=root/'private';src=read(p/'SOURCE_QUALIFIED_CANDIDATES.json');panel=read(p/'NEW_SOURCE_PANEL.json');positive=read(p/'POSITIVE_PANEL.json')
    original=src['original_stream'];rows={query_id(t['native']):t['native'] for t in original['tasks']+src['tasks']}
    for r in original['core_rows']+panel['rows']+positive['rows']:rows[query_id(r)]=r
    queue=dict(rows=list(rows.values()),freeze_id=digest(list(rows.values())),source_only=True,source_qualified_binding=digest(src),new_panel_binding=digest(panel),positive_binding=digest(positive))
    # Preserve the source-review output as a predecessor; the added panel is frozen before Base/student.
    old=p/'BASE_QUEUE.json'
    if old.exists():write_new(p/'BASE_QUEUE_SOURCE_REVIEW.json',read(old))
    write(old,queue)
    return queue


def finalize(root):
    root=Path(root);p=root/'private';auth=read(root/'public/RUN_AUTHORIZATION.json');src=read(p/'SOURCE_QUALIFIED_CANDIDATES.json');queue=read(p/'BASE_QUEUE.json');fresh=read(p/'FRESH_BASE_OUTPUTS.json');dispatch=read(p/'DISPATCH_base.json')
    if dispatch['gpu_uuid']!=auth['gpu_uuid'] or str(dispatch['gpu'])!=str(auth['physical_gpu']):raise ValueError('Current authorized GPU required')
    if digest(queue['rows'])!=dispatch['base_queue_binding'] or fresh['code']!=dispatch['code_commit'] or fresh['runtime']!=dispatch['runtime_lock']:raise ValueError('Current Base queue/runtime lineage mismatch')
    base={r['query_id']:r for r in fresh['records']};scores=read(p/'QUALIFIED_SCORE_CACHE.json')['scores'];mask={}
    for r in queue['rows']:
        q=query_id(r)
        if q not in base or base[q]['source']!=r:raise ValueError('Incomplete/current Base source mismatch')
        mask[q]=scores.get(score_key(r,base[q]['output']))
        if type(mask[q]) is not bool:raise ValueError('Unscored Base item')
    original=src['original_stream'];tasks,flow=select(original['tasks'],src['tasks'],mask)
    new=read(p/'NEW_SOURCE_PANEL.json');positive=read(p/'POSITIVE_PANEL.json');ablate=read(root/'public/ABLATION_PREREGISTRATION.json')
    if len(tasks)==19 and not new['rows'] and not ablate['all_H_G_structurally_legal']:raise ValueError('No new executable consumer; do not rebuild19')
    stream=deepcopy(original);stream.update(tasks=tasks,track='P',stage=20,status='FROZEN_STAGE20_PURE',new_rows=new['rows'],positive_rows=positive['rows'],source_binding=digest(src),new_panel_binding=digest(new),positive_binding=digest(positive),Base_mask_binding=digest(mask),admission_policy_binding=digest(read(p/'ADMISSION_POLICY.json')))
    for r in stream['role_transitions']:
        source=next(x for x in stream['core_rows'] if query_id(x)==r['query_id'])
        r['U_scope_overlap_edits']=[t['canonical_edit_id'] for t in tasks if source['role']=='U_eval' and (normalized(source['question'])==normalized(t['native']['question']) or reviewed_attribute(source['question'])==reviewed_attribute(t['native']['question']))]
    train=[r for t in tasks for r in [t['native']]+t['H_fit']+t['U_fit']]
    evalrows=[r for r in stream['core_rows'] if r['role']!='native_text_extension']+new['rows']+positive['rows']
    assert not {r['source_group'] for r in train}&{r['source_group'] for r in evalrows}
    assert not {r['image_sha256'] for r in train}&{r['image_sha256'] for r in evalrows}
    write_new(p/'STREAM.json',stream);write_new(p/'TRAINING_ELIGIBILITY.json',dict(flow=flow,legacy_Base_mask={t['canonical_edit_id']:mask[query_id(t['native'])] for t in tasks[:19]},current_Base_mask=mask,stream_binding=digest(stream),selection_before_students=True))
    summary=dict(N=len(tasks),new_edits=len(tasks)-19,K=len(tasks),track='P',legacy_prefix_preserved=True,legacy19_current_Base_correct=sum(mask[query_id(t['native'])] for t in tasks[:19]),new_all_Base_wrong=True,native_unique_QA=len({query_id(t['native']) for t in tasks}),native_propositions=len({proposition(t['native']) for t in tasks}),native_sources=len({t['native']['source_group'] for t in tasks}),new_max_per_source=2,stream_binding=digest(stream),patient_study='UNKNOWN',clinical_signoff=False)
    summary['native_images']=len({t['native']['image_sha256'] for t in tasks})
    cohort=read(root/'public/COHORT_FLOW.json');cohort.update(admission_reasons=dict(Counter(r['reason'] for r in flow)),actual_qualified_N=len(tasks),new_qualified_edits=len(tasks)-19,status='CURRENT_BASE_AND_STREAM_FROZEN_BEFORE_STUDENTS')
    write(root/'public/COHORT_FLOW.json',cohort)
    write_new(root/'public/STREAM_MANIFEST.json',summary);return summary


def self_test():
    def row(i,g,attr='liver'):return dict(image_sha256=g,question='Does the picture contain '+attr+'?',source_group=g)
    old=[dict(native=row(i,'old'+str(i)),order=i+1) for i in range(19)]
    candidates=[dict(native=row(i,'new',attr),order=20+i) for i,attr in enumerate(['liver','kidney','spleen'])]
    mask={query_id(t['native']):False for t in candidates};tasks,flow=select(old,candidates,mask);assert len(tasks)==21 and flow[-1]['reason']=='NEW_SOURCE_CAP2' and tasks[:19]==old
    bad=dict(mask);bad.pop(next(iter(bad)))
    try:select(old,candidates,bad)
    except ValueError:pass
    else:raise AssertionError('Missing verdict accepted')
    print('Stage20 admission checks PASS')

if __name__=='__main__':
    if sys.argv[1]=='test':self_test()
    elif sys.argv[1]=='prepare-base':prepare_base(sys.argv[2])
    elif sys.argv[1]=='finalize':print(finalize(sys.argv[2]))
    else:raise ValueError('Unknown command')
