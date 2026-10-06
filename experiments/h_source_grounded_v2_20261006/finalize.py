"""Freeze complete V2 results, expose only anonymous aggregates publicly."""
import collections
import csv
import json
import os
from pathlib import Path
import re
import shutil
from audit import read,write,save,accepted,intersections
from runner import digest

def csv_write(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def counts(rows):return dict(edits=len({r['edit_id'] for r in rows}),relations=len(rows),source_groups=len({r['source_group_id'] for r in rows}),images=len({r['image_sha256'] for r in rows}))
def classify_result(row):
    row=dict(row);b=row.get('generator_output');a=row.get('audit_output')
    if row['construction_status']!='MODEL_REVIEW_COMPLETED':return row
    disagree={'SUPPORTED':'PASS','REFUTED':'FAIL','UNKNOWN':'UNKNOWN'}.get(b['candidate_verdict'])!=a['independent_verdict']
    row['generator_auditor_disagreement']=disagree
    if disagree:row.update(evidence_level='UNKNOWN',final_status='UNKNOWN',qualification_note='Generator/auditor disagreement; no resampling or relabeling to PASS')
    row.update(clinical_verified=False,training_auto_admitted=False,formal_evaluation_auto_admitted=False)
    return row

def main():
    root=Path(os.environ['V2_ROOT']);audit=Path(os.environ['AUDIT_ROOT']);v1=Path(os.environ['V1_ROOT']);out=Path(os.environ['DELIVERY_ROOT'])
    assert not out.exists();shutil.copytree(audit,out)
    p=out/'private';src=root/'private'
    for name in ['PILOT_SELECTION.json','IMAGE_BINDINGS.json','ACQUISITION_PLAN.json','new_sources_ledger.jsonl','new_source_acquisition_details.jsonl','new_units.jsonl','v1_reused_units.jsonl','outbound_status.jsonl','outbound_status_qualified.jsonl','native_metadata_only.json']:
        shutil.copyfile(src/name,p/name)
    for name in ['model_calls','new_primary','exclusions']:
        shutil.copytree(src/name,p/name)
    for file in src.glob('QUEUE_*.json'):shutil.copyfile(file,p/file.name)
    shutil.copytree(root.parent/'fixture',p/'fixture_initial_preserved')
    shutil.copytree(root.parent/'fixture_corrected',p/'fixture_corrected')
    raw=read(src/'pilot_results.jsonl');rows=[classify_result(r) for r in raw]
    selection=json.loads((src/'PILOT_SELECTION.json').read_text());inventory=read(p/'edit_inventory.jsonl');em={e['edit_id']:e for e in inventory}
    units=read(p/'v1_reused_units.jsonl')+read(p/'new_units.jsonl');um={s['source_id']:s for s in units}
    native=json.loads((p/'native_metadata_only.json').read_text())
    remote=os.environ.get('REMOTE_DELIVERY_ROOT')
    for r in rows:
        s=um[r['source_id']];e=em[r['edit_id']]
        r.update(image_path=s['image_path'],evidence_origin=s['evidence_origin'],prompt_version='h-source-grounded-v2-20261006/v1-prompts+isolation-v2',visual_label_leakage=s.get('visual_label_leakage',False),clinical_verified=False,training_auto_admitted=False)
        if s['acquisition']=='NEW_PRIMARY' and remote:r['image_path']=remote+'/private/new_primary/'+Path(s['image_path']).name
        nb=native[e['source_identifier']]['body_part'];sb=s.get('body_part')
        r['same_body_region_metadata']=str(nb).lower()==str(sb).lower() or (nb=='HEAD' and sb=='brain')
        r['same_modality_confirmed']=False # No unverified native modality inferred from pixels.
        answer=r.get('h_answer','').lower()
        negative=bool(re.match(r'^(?:no|none|not|absence|without)\b',answer))
        r['hardness_category']='EASY_EXPLICIT_REFERENT_NEGATION' if negative else 'ANATOMIC_LOCATION_BOUNDARY_MODALITY_UNVERIFIED' if r['pilot_question_type']=='LOCALIZATION' and r['same_body_region_metadata'] else 'UNKNOWN'
        r['training_readiness']='BLOCKED_ANSWER_TEXT_IN_ORIGINAL_FIGURE' if r['visual_label_leakage'] else 'FUTURE_SEPARATE_REGISTRATION_REQUIRED'
    assert len(rows)<=120 and len({r['edit_id'] for r in rows})==len(selection['edits'])
    assert all(sum(1 for r in rows if r['edit_id']==e['edit_id'])<=10 and len({r['source_group_id'] for r in rows if r['edit_id']==e['edit_id']})==sum(1 for r in rows if r['edit_id']==e['edit_id']) for e in selection['edits'])
    write(p/'pilot_raw_execution_results.jsonl',raw);write(p/'pilot_all_results.jsonl',rows)
    write(p/'qualification_corrections.jsonl',[dict(edit_id=b['edit_id'],source_id=b['source_id'],before=b['final_status'],after=a['final_status'],reason=a.get('qualification_note')) for a,b in zip(rows,raw) if a['final_status']!=b['final_status']])
    reviewed=[r for r in rows if r['final_status']=='ACCEPTED_GROUNDED_REVIEWED']
    legacy=read(p/'v1_revised_candidates.jsonl')
    for r in legacy:
        r['v1_construction_status']=r['construction_status']
        if r['construction_status']=='ENGINE_NOT_IMPLEMENTED':r['construction_status']='MODEL_NOT_RUN';r['v2_engine_available']=True
    write(p/'v2_candidate_manifest.jsonl',legacy+rows)
    verified=[r for r in legacy if accepted(r,'FIT') and r['evidence_level']=='SOURCE_VERIFIED']
    oldgrounded=[r for r in legacy if accepted(r,'FIT') and r['evidence_level']=='SOURCE_GROUNDED']
    write(p/'h_fit_verified.jsonl',verified);write(p/'h_fit_grounded_reviewed.jsonl',reviewed)
    # Old grounded is separate; review completion cannot be inherited by it.
    write(p/'h_fit_grounded_pending_review.jsonl',oldgrounded)
    write(p/'v2_pending_manifest.jsonl',[r for r in legacy+rows if r['evidence_level']=='UNKNOWN' or r.get('evidence_status')=='SOURCE_OR_ROLE_BLOCKED'])
    ev=read(v1/'sealed_eval/h_eval_sealed.jsonl');fitany=verified+oldgrounded+reviewed
    original_fit=read(v1/'private/h_fit_verified.jsonl')+read(v1/'private/h_fit_grounded.jsonl')
    tables={}
    for label,fit in [('V1_frozen',original_fit),('V2_revised_rule_fit',verified+oldgrounded),('V2_with_grounded_reviewed',fitany)]:
        table=intersections(fit,ev);write(p/(label+'_FIT_EVAL_intersections.jsonl'),table)
        tables[label]={kind:sum(r['intersection']==kind for r in table) for kind in ['FIT_verified_EVAL_verified','FIT_verified_EVAL_grounded','FIT_any_EVAL_any']}
    write(p/'FIT_EVAL_intersection_per_edit.jsonl',[dict(table=label,**r) for label,fit in [('V1_frozen',original_fit),('V2_revised_rule_fit',verified+oldgrounded),('V2_with_grounded_reviewed',fitany)] for r in intersections(fit,ev)])
    assert not {r['source_group_id'] for r in fitany}&{r['source_group_id'] for r in ev}
    assert not {r['image_sha256'] for r in fitany}&{r['image_sha256'] for r in ev}
    receipts=sorted((p/'model_calls/receipts').glob('*.json'));calls=[json.loads(f.read_text()) for f in receipts]
    assert len(calls)+5<=300
    sessions=[r.get('session_id') for r in calls if r['status']=='COMPLETED'];assert len(sessions)==len(set(sessions))
    contexts=p/'execution_contexts';contexts.mkdir()
    for r in calls:
        context=contexts/r['call_id'];context.mkdir();work=Path(r['work_directory'])
        for n in ['input.json','prompt.txt','schema.json','final.json','stderr.txt','boundary.sb']:
            if (work/n).exists():shutil.copyfile(work/n,context/n)
        if r['stage']=='C':
            data=json.loads((context/'input.json').read_text());assert set(data)=={'edit_record','candidate_H','source_record'}
            assert set(data['candidate_H'])=={'h_question','h_answer','evidence_ids','evidence_spans'}
        assert not r['tool_event_types'] and all(r['isolation_checks'].values())
    token_totals=collections.Counter()
    for r in calls:
        for k,v in (r.get('usage') or {}).items():
            if isinstance(v,(int,float)):token_totals[k]+=v
    types=sorted({e['attribute_relation'] for e in inventory});funnel=[]
    allrows=legacy+rows
    for kind in types:
        edits=[e for e in inventory if e['attribute_relation']==kind];ids={e['edit_id'] for e in edits}
        relevant=[r for r in allrows if r['edit_id'] in ids and r.get('source_role_check')=='PASS' and r['assigned_role']=='FIT']
        def n(rs):return len({r['edit_id'] for r in rs})
        # Intake into a same-question alignment path, including unsuccessful
        # model attempts; final semantic alignment is a separate audit result.
        aligned=[r for r in relevant if r.get('same_question_operator') or r['construction_status']=='MODEL_REVIEW_COMPLETED']
        assessed=[r for r in aligned if r['construction_status'] in {'RULE_PATH_IMPLEMENTED','MODEL_REVIEW_COMPLETED'} and r['evidence_status']!='NOT_YET_ASSESSED']
        supported=[r for r in assessed if r.get('source_supports_h_answer')=='YES']
        excluded=[r for r in supported if r.get('target_relation')=='CONTRADICTS']
        isolated=[r for r in excluded if r['source_group_id'] not in {e['source_group_id'] for e in ev}]
        final=[r for r in isolated if r['evidence_level'] in {'SOURCE_VERIFIED','SOURCE_GROUNDED'}]
        funnel.append(dict(question_type=kind,edits=len(edits),legal_source_edits=n(relevant),aligned_candidate_edits=n(aligned),actually_assessed_edits=n(assessed),semantic_alignment_confirmed_edits=n([r for r in relevant if r.get('same_question_operator')]),model_review_completed_edits=n([r for r in relevant if r['construction_status']=='MODEL_REVIEW_COMPLETED']),answer_supported_edits=n(supported),target_excluded_edits=n(excluded),source_group_isolated_edits=n(isolated),finally_accepted_edits=n(final)))
    csv_write(out/'public/FUNNEL_BY_QUESTION_TYPE.csv',funnel)
    pilot=[]
    for e in selection['edits']:
        rr=[r for r in rows if r['edit_id']==e['edit_id']]
        pilot.append(dict(position=e['position'],question_type=e['pilot_question_type'],old_cache_relations=sum(r['origin']=='V1_CACHE_REUSE' for r in rr),old_candidate_source_rechecked=sum(r['old_candidate_reused'] for r in rr),new_source_relations=sum(r['origin']=='NEW_PRIMARY' for r in rr),distinct_source_groups=len({r['source_group_id'] for r in rr}),model_review_completed=sum(r['construction_status']=='MODEL_REVIEW_COMPLETED' for r in rr),accepted_grounded=sum(r['final_status']=='ACCEPTED_GROUNDED_REVIEWED' for r in rr),rejected=sum(r['final_status']=='REJECTED' for r in rr),unknown=sum(r['final_status']=='UNKNOWN' for r in rr)))
    csv_write(out/'public/PILOT_ANONYMOUS_RESULTS.csv',pilot)
    gaps=[]
    for e in inventory:
        rs=[r for r in allrows if r['edit_id']==e['edit_id']];ff=[r for r in fitany if r['edit_id']==e['edit_id']]
        gaps.append(dict(position=e['position'],edit_id=e['edit_id'],question_type=e['attribute_relation'],verified_relations=sum(r['edit_id']==e['edit_id'] for r in verified),grounded_reviewed_relations=sum(r['edit_id']==e['edit_id'] for r in reviewed),grounded_unreviewed_relations=sum(r['edit_id']==e['edit_id'] for r in oldgrounded),construction_statuses=dict(collections.Counter(r['construction_status'] for r in rs)),evidence_statuses=dict(collections.Counter(r['evidence_status'] for r in rs)),fit_covered=bool(ff),missing_evidence=[r.get('specific_review_gap') for r in rows if r['edit_id']==e['edit_id'] and r['final_status']!='ACCEPTED_GROUNDED_REVIEWED'] or [e['nontransfer_condition']],eval_exposed_frozen_source_only=True))
    write(p/'per_edit_coverage_and_gaps.jsonl',gaps)
    oldids={r['edit_id'] for r in verified+oldgrounded};newids={r['edit_id'] for r in reviewed}-oldids
    summary=dict(protocol='H_SOURCE_GROUNDED_V2',N=146,v1_rule_audit=json.loads((out/'public/RULE_AUDIT_SUMMARY.json').read_text()),fit_verified=counts(verified),fit_grounded_reviewed=counts(reviewed),fit_grounded_unreviewed=counts(oldgrounded),fit_any=counts(fitany),new_pilot_covered_edits=len(newids),new_nonmodality_noncontrast_covered_edits=len(newids),new_verified_covered_edits=0,pilot_edits=len(pilot),pilot_relations=len(rows),pilot_status_counts=dict(collections.Counter(r['final_status'] for r in rows)),pilot_question_type_counts=dict(collections.Counter(e['pilot_question_type'] for e in selection['edits'])),pilot_origin_counts=dict(collections.Counter(r['origin'] for r in rows)),old_candidate_sources_rechecked=sum(r['old_candidate_reused'] for r in rows),strict_near_boundary_verified=0,grounded_near_boundary_confirmed=0,grounded_same_region_modality_unverified=sum(r['final_status']=='ACCEPTED_GROUNDED_REVIEWED' and r['pilot_question_type']=='LOCALIZATION' for r in rows),model_calls=dict(collections.Counter(r['stage'] for r in calls)),model_calls_total=len(calls),model_call_statuses=dict(collections.Counter(r['status'] for r in calls)),model_usage_tokens=dict(token_totals),technical_retries=sum(r['attempt']>1 for r in calls),fixture_actual_model_calls=3,fixture_simulated_execution_failures=2,fixture_cache_hits=1,model_call_attempts_including_all_fixtures=len(calls)+5,actual_model='gpt-6.1-sol',interface='official Codex CLI 0.160.1 with existing ChatGPT login',model_roles='SAME_MODEL_SEPARATE_CONTEXT',upstream_immutable_model_snapshot='UNAVAILABLE',api_billing_usd=None,credit_purchase=False,quota_reset_used=False,intersections=tables,construction_status_counts=dict(collections.Counter(r['construction_status'] for r in allrows)),evidence_status_counts=dict(collections.Counter(r['evidence_status'] for r in allrows)),generator_auditor_disagreements=sum(r.get('generator_auditor_disagreement',False) for r in rows),source_acquisition=json.loads((root/'public/ACQUISITION_SUMMARY.json').read_text()),old_eval_unchanged=True,patient_independence='UNKNOWN',development_exposure=True,training_started=False,GPU_judge_started=False,TT_router_loss_changed=False,cohort_order_preserved=True,performance_claim=False)
    summary.update(grounded_near_boundary_candidates_new=sum(r['hardness_category']=='ANATOMIC_LOCATION_BOUNDARY_MODALITY_UNVERIFIED' for r in reviewed),legacy_grounded_near_boundary_candidates_fit=sum(r.get('hardness_category')=='ANATOMIC_LOCATION_BOUNDARY_MODALITY_UNVERIFIED' for r in oldgrounded),legacy_grounded_near_boundary_candidates_eval=sum(r.get('hardness_category')=='ANATOMIC_LOCATION_BOUNDARY_MODALITY_UNVERIFIED' and r['evidence_level']=='SOURCE_GROUNDED' for r in ev),accepted_visual_label_leakage_risks=sum(r['visual_label_leakage'] for r in reviewed),original_source_outbound_status_counts=dict(collections.Counter(r['outbound_status'] for r in read(p/'outbound_status_qualified.jsonl'))))
    save(out/'public/SUMMARY.json',summary)
    save(out/'FREEZE_RECEIPT.json',dict(status='V2_DATA_BUILD_AND_CONTEXT_ISOLATED_MODEL_PILOT_COMPLETE',cohort_order_digest=digest([e['edit_id'] for e in inventory]),student_outputs_read=False,old_eval_path=str(v1/'sealed_eval/h_eval_sealed.jsonl'),old_eval_modified=False,new_eval_created=False,formal_eval_admitted=False,private_records_publicly_published=False,model_calls=len(calls),quota_limit=300,covered_pilot_edits=len(newids)))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
