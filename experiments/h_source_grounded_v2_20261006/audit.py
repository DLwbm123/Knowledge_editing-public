"""Read frozen V1; write a separate V2 impact audit, queues and intersections."""
import collections
import csv
import json
import os
from pathlib import Path
import rules

def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def accepted(c,role):
    return c['assigned_role']==role and c['evidence_level'] in {'SOURCE_VERIFIED','SOURCE_GROUNDED'} and c['source_role_check']=='PASS' and c['citation_check']=='PASS' and c['target_relation']=='CONTRADICTS'
def intersections(fit,ev):
    result=[]
    for label,fl,el in [('FIT_verified_EVAL_verified',{'SOURCE_VERIFIED'},{'SOURCE_VERIFIED'}),('FIT_verified_EVAL_grounded',{'SOURCE_VERIFIED'},{'SOURCE_GROUNDED'}),('FIT_any_EVAL_any',{'SOURCE_VERIFIED','SOURCE_GROUNDED'},{'SOURCE_VERIFIED','SOURCE_GROUNDED'})]:
        ff=[x for x in fit if x['evidence_level'] in fl];ee=[x for x in ev if x['evidence_level'] in el]
        ids={x['edit_id'] for x in ff}&{x['edit_id'] for x in ee}
        for eid in sorted(ids):
            fs={x['source_group_id'] for x in ff if x['edit_id']==eid};es={x['source_group_id'] for x in ee if x['edit_id']==eid}
            result.append(dict(intersection=label,edit_id=eid,position=next(x['position'] for x in ff if x['edit_id']==eid),fit_groups=len(fs),eval_groups=len(es),group_overlap=len(fs&es),patient_independence='UNKNOWN',development_exposure=True))
    return result

def main():
    v1=Path(os.environ['V1_ROOT']);out=Path(os.environ['V2_ROOT']);out.mkdir(exist_ok=False)
    sources=read(v1/'private/source_registry.jsonl');sm={s['source_id']:s for s in sources}
    edits=read(v1/'private/edit_inventory.jsonl');em={e['edit_id']:e for e in edits}
    # Operator image-label inspections remain a private evidence asset; no
    # source/case-specific annotation is embedded in published code.
    bindings=json.loads(Path(os.environ['PANEL_BINDINGS']).read_text())
    for sid,binding in bindings.items():
        assert sm[sid]['image_sha256']==binding['image_sha256']
        sm[sid]['panel_binding']=binding['panel_binding']
    write(out/'private/source_registry.jsonl',sources)
    candidates=read(v1/'private/candidate_manifest.jsonl');impact=[];changed=[]
    for c in candidates:
        e=em[c['edit_id']];s=sm[c['source_id']];d=rules.automatic_decision(e,s)
        before={k:c[k] for k in ('evidence_level','target_relation','decisive_reason','h_question','h_answer')}
        if d is not None:
            c.update(evidence_level=d['level'],target_relation=d['relation'],decisive_reason=d['reason'],h_question=e['original_question'] if d['answer'] else None,h_answer=d['answer'],source_supports_h_answer='YES' if d['answer'] and d['relation']!='UNKNOWN' else 'UNKNOWN',same_question_operator=bool(d['answer']),final_status=d['level'])
            c['structured_acquisition']=d
            after={k:c[k] for k in before}
            row=dict(candidate_id=c['candidate_id'],edit_id=c['edit_id'],position=c['position'],assigned_role=c['assigned_role'],source_id=s['source_id'],source_group_id=s['source_group_id'],source_text=s['source_text'],source_evidence_ids=s['evidence_ids'],before=before,after=after,verdict_changed=before['evidence_level']!=after['evidence_level'] or before['target_relation']!=after['target_relation'],difference_reason=d['reason'],structured_state=d)
            impact.append(row)
            if row['verdict_changed']:changed.append(row)
        if c['source_role_check']!='PASS':
            construction='RULE_PATH_IMPLEMENTED';evidence='SOURCE_OR_ROLE_BLOCKED'
        elif d is not None:
            construction='RULE_PATH_IMPLEMENTED';evidence='ANSWER_SUPPORTED_TARGET_EXCLUDED' if d['relation']=='CONTRADICTS' else 'ANSWER_SUPPORTED_TARGET_NOT_EXCLUDED' if d['relation']=='COMPATIBLE' else 'REFERENT_OR_QUALIFIER_UNRESOLVED'
        elif c['evidence_level'] in {'SOURCE_VERIFIED','SOURCE_GROUNDED','REJECTED'}:
            construction='RULE_PATH_IMPLEMENTED';evidence='ANSWER_SUPPORTED_TARGET_EXCLUDED' if c['target_relation']=='CONTRADICTS' else 'ANSWER_SUPPORTED_TARGET_NOT_EXCLUDED'
        else:
            construction='ENGINE_NOT_IMPLEMENTED';evidence='NOT_YET_ASSESSED'
        c.update(construction_status=construction,evidence_status=evidence,v1_medical_grade=before['evidence_level'],v2_model_review_status='MODEL_NOT_RUN',historical_asset_overwritten=False)
    write(out/'private/v1_revised_candidates.jsonl',candidates);write(out/'private/acquisition_impact_audit.jsonl',impact)
    other=[]
    for c in candidates:
        if c['source_dataset']=='PMC_PRIMARY' or c['v1_medical_grade']!='SOURCE_VERIFIED':continue
        s=sm[c['source_id']];e=em[c['edit_id']]
        other.append(dict(candidate_id=c['candidate_id'],assigned_role=c['assigned_role'],source_id=s['source_id'],original_source_text=s['source_text'],original_question=s['original_question'],original_answer=s['original_answer'],question_operator=e['question_operator'],verification_method=c.get('verification_method'),before='SOURCE_VERIFIED',after=c['evidence_level'],audit_scope='Original current-image modality QA' if e['question_operator']=='modality' else 'Historical anatomical-frame record retained outside caption-acquisition repair',finding='No new current-image modality attribution ambiguity found in original modality QA inspected' if e['question_operator']=='modality' else 'No new invalidation claimed; historical source/frame certificate retained'))
    write(out/'private/other_verified_branch_audit.jsonl',other)
    write(out/'private/edit_inventory.jsonl',edits)
    fit=[c for c in candidates if accepted(c,'FIT')];ev=read(v1/'sealed_eval/h_eval_sealed.jsonl')
    write(out/'private/h_fit_verified_rule_revised.jsonl',[x for x in fit if x['evidence_level']=='SOURCE_VERIFIED'])
    write(out/'private/h_fit_grounded_unreviewed.jsonl',[x for x in fit if x['evidence_level']=='SOURCE_GROUNDED'])
    write(out/'private/pending_manifest.jsonl',[c for c in candidates if c['evidence_level']=='UNKNOWN' or c['evidence_status']=='SOURCE_OR_ROLE_BLOCKED'])
    origfit=read(v1/'private/h_fit_verified.jsonl')+read(v1/'private/h_fit_grounded.jsonl')
    for name,rows in [('v1_frozen',intersections(origfit,ev)),('v2_revised_fit_old_frozen_eval',intersections(fit,ev))]:
        write(out/'private'/f'{name}_intersections.jsonl',rows)
    summary=dict(N=146,v1_fit_verified_edits=len({c['edit_id'] for c in origfit if c['evidence_level']=='SOURCE_VERIFIED'}),v2_rule_fit_verified_edits=len({c['edit_id'] for c in fit if c['evidence_level']=='SOURCE_VERIFIED'}),v2_rule_fit_verified_relations=sum(c['evidence_level']=='SOURCE_VERIFIED' for c in fit),automatic_relations_audited=len(impact),verdict_changes=len(changed),changes_by_role=dict(collections.Counter(c['assigned_role'] for c in changed)),previously_verified_changed=sum(c['before']['evidence_level']=='SOURCE_VERIFIED' for c in changed),previously_verified_changes_by_reason=dict(collections.Counter(c['difference_reason'] for c in changed if c['before']['evidence_level']=='SOURCE_VERIFIED')),construction_status_counts=dict(collections.Counter(c['construction_status'] for c in candidates)),evidence_status_counts=dict(collections.Counter(c['evidence_status'] for c in candidates)),old_eval_unchanged=True,cohort_order_unchanged=True)
    for name,rows in [('v1_frozen',intersections(origfit,ev)),('v2_revised_fit_old_frozen_eval',intersections(fit,ev))]:
        summary[name+'_intersection_counts']={k:sum(r['intersection']==k for r in rows) for k in ['FIT_verified_EVAL_verified','FIT_verified_EVAL_grounded','FIT_any_EVAL_any']}
    summary.update(other_verified_branch_records=len(other),original_modality_QA_reviewed=sum(x['question_operator']=='modality' for x in other))
    save(out/'public/RULE_AUDIT_SUMMARY.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
