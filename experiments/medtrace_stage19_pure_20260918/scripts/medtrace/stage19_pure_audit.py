"""Reconstruct the pre-student pure candidates; never train or read student scores."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage18_support import validate_task
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute


def audit(previous, source, destination):
    previous,source,destination=map(Path,(previous,source,destination))
    p=previous/'private'; structural=read(p/'STRUCTURAL_TASKS.json')
    ledger=read(source/'DEV_SOURCE_ROLE_LEDGER.json')
    assert digest(ledger)==structural['source_role_binding']
    assert digest(structural['tasks'])==structural['tasks_binding']
    reviews={r['source_qid']:r for f in (source/'V3_TRAIN_REFERENCE_REVIEW.json',p/'REFERENCE_REUSED.json',p/'REFERENCE_REVIEW_RESULT.json') for r in read(f)['records']}
    # Reconstruct only the already-frozen Base qualification, not the final score cache.
    scores=dict(read(p/'EXISTING_SCORE_CACHE.json')['scores'])
    judged=read(p/'qualification_JUDGE_INPUT.json');verdict=read(p/'qualification_JUDGE/operator/VERDICTS.json')
    assert digest(judged)==verdict['source_binding']
    decisions={r['opaque_query_id']:r['is_correct'] for r in verdict['decisions']}
    for r in judged['records']:scores[score_key(r['source'],r['output'])]=decisions[digest(r)]
    fresh={r['query_id']:r for r in read(p/'FRESH_BASE_OUTPUTS.json')['records']}
    older={r['query_id']:r for r in read(p/'EXISTING_BASE_OUTPUTS.json')['records']}
    flow=read(p/'TRAINING_ELIGIBILITY.json')['flow']
    selected=[t for t in structural['tasks'] if t['H_fit'] and scores[score_key(t['native'],fresh[query_id(t['native'])]['output'])] is False]
    assert {t['canonical_edit_id'] for t in selected}=={r['task'] for r in flow if r['H_supported'] and not r['Base_correct']}
    assert len(selected)==read(previous/'public/STREAM_MANIFEST.json')['pure_FACT_qualified']==19
    ng,ag,eg=(set(ledger[k]) for k in ('native_groups','auxiliary_groups','evaluation_sources'))
    assert not (ng&ag or ng&eg or ag&eg)
    protected_hashes={r['image_sha256'] for r in ledger['evaluation_pool']}
    rows=[];members=[]
    for position,t in enumerate(selected,1):
        validate_task(t,fasttrack_branch='C_FACT')
        assert t['native']['source_group'] in ng
        for role in ('native','H_fit','U_fit'):
            for r in [t[role]] if role=='native' else t[role]:
                review=reviews[r['source_qid']]
                assert review['verdict']=='SUPPORTED'
                assert all(r[k]==review[k] for k in ('dataset','source_group','image_sha256','question','reference'))
                assert r['source_group'] not in eg and r['image_sha256'] not in protected_hashes
                assert r['source_group'].split(':')[-1] not in ledger['quarantined_groups']
                if role!='native':assert r['source_group'] in ag
                rows.append(r)
        for u in t['U_fit']:
            assert normalized(u['question'])!=normalized(t['native']['question'])
            assert reviewed_attribute(u['question'])!=reviewed_attribute(t['native']['question'])
        old=older[query_id(t['native'])]
        members.append(dict(position=position,canonical_edit_id=t['canonical_edit_id'],structural_order=t['order'],
            native=t['native'],H_fit=t['H_fit'],U_fit=t['U_fit'],source_checks='PASS',
            fresh_4090_Base_correct=False,prior_A100_Base_correct=scores[score_key(t['native'],old['output'])]))
    assert len({query_id(t['native']) for t in selected})==19
    native_hashes={t['native']['image_sha256'] for t in selected}
    assert not native_hashes&{r['image_sha256'] for t in selected for r in t['H_fit']+t['U_fit']}
    trained_questions={(r['image_sha256'],normalized(r['question'])) for r in rows}
    trained_questions|={(t['native']['image_sha256'],normalized(q)) for t in selected for q in t['fit_questions']}
    assert all((r['image_sha256'],normalized(r['question'])) not in trained_questions for r in structural['core_rows'] if r['role']=='native_text_extension')
    tasks=deepcopy(selected)
    for i,t in enumerate(tasks,1):t['order']=i
    transitions=[]
    for r in structural['core_rows']:
        affected=[t['canonical_edit_id'] for t in tasks if r['role']=='U_eval' and (normalized(r['question'])==normalized(t['native']['question']) or reviewed_attribute(r['question'])==reviewed_attribute(t['native']['question']))]
        transitions.append(dict(query_id=query_id(r),role=r['role'],U_scope_overlap_edits=affected,
            policy='Always generate full fixed panel; apply existing strict-U predicate rule and report its denominator'))
    stream=dict(tasks=tasks,anchors=structural['anchors'],track='P',core_rows=structural['core_rows'],
        anchor_packages=structural['anchor_packages'],role_transitions=transitions,source_role_binding=structural['source_role_binding'],
        order='Existing 11 anchors then original pre-student structural order; stable ID seeds',
        status='SOURCE_ELIGIBLE_CANDIDATES_NOT_GPU_AUTHORIZED',minimum_size_for_pure=1,
        qualification_note='19 Base-wrong on completed 4090 run; one was correct in prior A100 output; GPU3 qualification required before training')
    def support(role):
        rs=[r for t in tasks for r in t[role]]
        return dict(assignments=len(rs),unique_QA=len({r['source_qid'] for r in rs}),sources=len({r['source_group'] for r in rs}),max_QA_reuse=max(Counter(r['source_qid'] for r in rs).values()))
    summary=dict(status=stream['status'],candidate_N=19,native_sources=len(native_hashes),anchors=11,FACT_coverage_if_all_qualified='19/19',
        H_fit=support('H_fit'),U_fit=support('U_fit'),fixed_core_QA=26,H_eval_QA=6,H_eval_sources=2,U_eval_QA=9,native_rephrases=11,
        full_sequence_protected_source_overlaps=0,full_sequence_protected_image_overlaps=0,native_aux_image_overlaps=0,
        protected_source_groups=len(eg),all_author_references_supported=True,clinical_signoff=False,patient_study='UNKNOWN',
        prior_A100_Base_wrong=sum(not m['prior_A100_Base_correct'] for m in members),fresh_4090_Base_wrong=19,
        task_ids_order_seed_preserved=True,selection_uses_student_outputs=False,new_judgments=0,new_GPU_seconds=0,
        minimum_50_gate_removed=True,shared_background_allowed=False,forced_routing_as_main=False,stream_binding=digest(stream))
    (destination/'private').mkdir(parents=True,exist_ok=True);(destination/'public').mkdir(exist_ok=True)
    write_new(destination/'private/PURE_CANDIDATE_MEMBERS.json',dict(members=members,stream_binding=digest(stream)))
    write_new(destination/'private/STREAM_CANDIDATES.json',stream)
    write_new(destination/'public/PURE_CANDIDATE_AUDIT.json',summary)
    return summary


if __name__=='__main__':print(json.dumps(audit(*sys.argv[1:]),indent=2))
