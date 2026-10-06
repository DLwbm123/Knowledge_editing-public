"""Freeze a reviewed delivery from staging, preserving staging and all failures."""
import collections
import csv
import json
import os
from pathlib import Path
import time
from PIL import Image
import build as b

def main():
    cpu=time.process_time();start=time.time();r=Path(os.environ['STAGING_ROOT']);out=Path(os.environ['RUN_ROOT']);assert not out.exists();out.mkdir()
    rules=b.load(os.environ['SEMANTIC_REVIEW_RULES'])
    src=b.jl(r/'private/source_registry.jsonl');cc=b.jl(r/'private/candidate_manifest.jsonl');inventory=b.jl(r/'private/edit_inventory.jsonl')
    uniq={s['image_sha256']:s for s in src};prints=[]
    for sha,s in uniq.items():
        with Image.open(s['image_path']) as im:
            v=list(im.convert('L').resize((9,8)).getdata());bits=0
            for i in range(64):bits=(bits<<1)|(v[(i//8)*9+i%8]>v[(i//8)*9+i%8+1])
        prints.append((sha,s,bits))
    pairs=[];flagged=set()
    for i,a in enumerate(prints):
        for z in prints[i+1:]:
            distance=(a[2]^z[2]).bit_count()
            if distance<=4:
                pairs.append(dict(image_a=a[1]['image_id'],image_b=z[1]['image_id'],group_a=a[1]['source_group_id'],group_b=z[1]['source_group_id'],role_a=a[1]['assigned_role'],role_b=z[1]['assigned_role'],distance=distance,status='QUARANTINE_PENDING_REVIEW_NOT_PATIENT_PROOF'))
                flagged.update([a[1]['source_group_id'],z[1]['source_group_id']])
    for s in src:
        s['near_duplicate_review']='PENDING' if s['source_group_id'] in flagged else 'NO_LOW_COST_FLAG_NOT_PROOF'
        if s['source_group_id'] in flagged:s['source_role_check']='UNKNOWN_NEAR_DUPLICATE_REVIEW'
    for c in cc:
        if c['source_group_id'] in flagged:
            c['source_role_check']='UNKNOWN_NEAR_DUPLICATE_REVIEW';c['final_status']='QUARANTINED';c['exclusion_reason']='NEAR_DUPLICATE_SOURCE_GROUP_PENDING_REVIEW'
        for rule in rules.get('final_downgrades',[]):
            if c['position']==rule['position'] and c['source_question_id'] in rule['qids'] and c['evidence_level']=='SOURCE_GROUNDED':
                c.update(evidence_level='UNKNOWN',target_relation='UNKNOWN',generator_verdict='UNKNOWN',final_status='UNKNOWN',exclusion_reason=rule['reason'],decisive_reason='Location descriptions need not be mutually exclusive; no direct target-exclusion evidence. Codex review correction preserved as a new version.')
        for rule in rules.get('hardness_rules',[]):
            if c['evidence_level']=='SOURCE_GROUNDED' and c['position'] in rule['positions'] and c['source_question_id'] in rule['qids']:c['hardness_category']=rule['category']
        if c['evidence_level']=='SOURCE_GROUNDED' and c['source_dataset']=='PMC_PRIMARY':c['hardness_category']='ANATOMIC_LOCATION_BOUNDARY_MODALITY_UNVERIFIED'
    fitv=[c for c in cc if b.admit(c,'FIT') and c['evidence_level']=='SOURCE_VERIFIED'];fitg=[c for c in cc if b.admit(c,'FIT') and c['evidence_level']=='SOURCE_GROUNDED'];ev=[c for c in cc if b.admit(c,'EVAL')];allfit=fitv+fitg
    # The FIT loader only accepts role-qualified FIT records and rejects sealed paths.
    for name,rows in [('source_registry',src),('candidate_manifest',cc),('edit_inventory',inventory),('h_fit_verified',fitv),('h_fit_grounded',fitg),('unknown_and_review_queue',[c for c in cc if c['evidence_level'] in {'UNKNOWN','SOURCE_GROUNDED'} or c['final_status']=='QUARANTINED'])]:b.lines(out/'private'/f'{name}.jsonl',rows)
    b.lines(out/'sealed_eval/h_eval_sealed.jsonl',ev);os.chmod(out/'sealed_eval',0o700)
    for name in ['blind_evidence_facts.jsonl','historical_unknown_outcomes.jsonl','api_review_queue.jsonl','API_LEDGER.json','exclusion_audit.json','PILOT_GATE.json','PILOT_RECEIPT.json']:
        p=out/'private'/name;p.write_bytes((r/'private'/name).read_bytes())
    b.save(out/'private/NEAR_DUPLICATE_REVIEW.json',dict(images_read=len(prints),pairs=pairs,quarantined_groups=len(flagged),role_change=False,patient_inference=False))
    # Query-level schema/role checks after the one source pass; not clinical validation.
    sm={s['source_id']:s for s in src};groups=collections.defaultdict(set)
    for s in src:groups[s['source_group_id']].add(s['assigned_role'])
    assert all(len(x)==1 for x in groups.values())
    fithashes={c['image_sha256'] for c in allfit};evalhashes={c['image_sha256'] for c in ev};assert not fithashes&evalhashes
    fitgroups={c['source_group_id'] for c in allfit};evalgroups={c['source_group_id'] for c in ev};assert not fitgroups&evalgroups
    for c in allfit+ev:
        assert b.support_check(c['evidence_spans'],sm[c['source_id']]['source_text'])
        assert c['auditor_verdict']=='NOT_RUN' and not c['external_api_executed']
        assert not c['clinical_verified']
    assert len(b.fit_load(out/'private/h_fit_verified.jsonl'))==len(fitv)
    assert len(b.fit_load(out/'private/h_fit_grounded.jsonl'))==len(fitg)
    summary=b.load(r/'public/SUMMARY.json')
    def counts(xs):return dict(edits=len({c['edit_id'] for c in xs}),relations=len(xs),original_QA=len({c['source_id'] for c in xs if c['evidence_origin']=='ORIGINAL_QA'}),images=len({c['image_sha256'] for c in xs}),source_groups=len({c['source_group_id'] for c in xs}))
    inherited=[c for c in allfit if c['origin']=='HISTORICAL_VERIFIED_INHERITED'];new=[c for c in allfit if c['origin']!='HISTORICAL_VERIFIED_INHERITED']
    summary.update(fit_verified=counts(fitv),fit_grounded=counts(fitg),fit_union=counts(allfit),fit_new=counts(new),eval_verified=counts([c for c in ev if c['evidence_level']=='SOURCE_VERIFIED']),eval_grounded=counts([c for c in ev if c['evidence_level']=='SOURCE_GROUNDED']),eval_union=counts(ev),all_candidate_status_counts=dict(collections.Counter(c['evidence_level'] for c in cc)),new_generated_QA_relations=sum(c['construction_type']=='EVIDENCE_TO_QA' for c in new),new_original_QA_pairings=sum(c['construction_type']!='EVIDENCE_TO_QA' for c in new),new_original_QA_sources=len({c['source_id'] for c in allfit if c['evidence_origin']=='ORIGINAL_QA'}-{c['source_id'] for c in inherited}),new_images=len(fithashes-{c['image_sha256'] for c in inherited}),new_source_groups=len(fitgroups-{c['source_group_id'] for c in inherited}),near_duplicate_flags=len(pairs),near_duplicate_cross_role_flags=sum(p['role_a']!=p['role_b'] for p in pairs),near_duplicate_quarantined_groups=len(flagged),grounded_hardness_counts=dict(collections.Counter(c['hardness_category'] for c in fitg+ev if c['evidence_level']=='SOURCE_GROUNDED')),source_isolated_eval_not_patient_independent=True,finalization_CPU_seconds=time.process_time()-cpu,finalization_wall_seconds=time.time()-start,qualified_role_checks='PASS',external_GPT_stage_not_completed=True,first_failed_build_CPU_unknown=True)
    rows=[]
    for e in inventory:
        fit=[c for c in allfit if c['edit_id']==e['edit_id']];ee=[c for c in ev if c['edit_id']==e['edit_id']];cand=[c for c in cc if c['edit_id']==e['edit_id']]
        gaps=[]
        if not fit:gaps+=['TARGET_NOT_EXCLUDED' if cand else 'NO_SEMANTIC_MATCH']
        if e['H_applicability']=='UNCERTAIN':gaps+=['H_APPLICABILITY_UNCERTAIN']
        if e['H_applicability']=='NOT_APPLICABLE_WITH_REASON':gaps+=['NOT_APPLICABLE_FIXED_ENTITY_GENERAL_PROPOSITION']
        if e['question_operator'] in {'treatment','prevention','symptom','cause'} and not fit:gaps+=['NEEDS_SPECIALIST_REVIEW']
        if not ee:gaps+=['NO_INDEPENDENT_H_EVAL']
        rows.append(dict(position=e['position'],edit_id=e['edit_id'],H_applicability=e['H_applicability'],fit_verified=sum(c['evidence_level']=='SOURCE_VERIFIED' for c in fit),fit_grounded=sum(c['evidence_level']=='SOURCE_GROUNDED' for c in fit),fit_independent_source_groups=len({c['source_group_id'] for c in fit}),eval_verified=sum(c['evidence_level']=='SOURCE_VERIFIED' for c in ee),eval_grounded=sum(c['evidence_level']=='SOURCE_GROUNDED' for c in ee),eval_source_groups=len({c['source_group_id'] for c in ee}),candidates=len(cand),gap_codes=';'.join(gaps),next_evidence_needed=e['nontransfer_condition']+'; '+('Original diagnosis/causal/treatment evidence and qualified review preserving side/severity/time.' if e['question_operator'] in {'treatment','prevention','symptom','cause'} else 'Explicit target negation/disjoint single referent evidence; not an incomplete list.')))
    with (out/'private/h_coverage_by_edit.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary['applicability_counts']=dict(collections.Counter(e['H_applicability'] for e in inventory));summary['fit_gap_code_counts']=dict(collections.Counter(g for x in rows for g in x['gap_codes'].split(';') if g))
    summary['pilot_phase']='CPU_SCHEMA_SOURCE_ROLE_GATE_PASS; external API pipeline pending'
    b.save(out/'public/SUMMARY.json',summary)
    b.save(out/'FREEZE_RECEIPT.json',dict(status='SOURCE_DATA_DELIVERY_FROZEN_API_AUDIT_PENDING',N=146,fit_coverage=summary['fit_union']['edits'],candidate_records=len(cc),source_groups_role_disjoint=True,qualified_images_role_disjoint=True,student_outputs_read=False,GPU_started=False,private_files_only=True,formal_H_eval_auto_admitted=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
