"""Read-only source audit; patient-level inputs and evidence stay private."""
import ast
from collections import Counter, defaultdict
import csv
from datetime import date
import io
import json
import os
from pathlib import Path
import re
import struct
import zipfile


def main():
    cfg = json.loads(Path(os.environ['AUDIT_CONFIG']).read_text())
    private, public = Path(cfg['private']), Path(cfg['public'])
    source, package = Path(cfg['source']), Path(cfg['package'])
    tasks = [json.loads(line) for line in (package/'T5_CANONICAL.private.jsonl').read_text().splitlines()]
    selected_ids = json.loads((private/'SELECTION_FROZEN.private.json').read_text())['case_ids']
    selected = [t for t in tasks if t['case_id'] in selected_ids]
    assert len(selected) == len(set(selected_ids)) == 34
    assert Counter(t['presence_code'] for t in selected) == {'01': 28, '10': 6}
    notes = json.loads((private/'MANUAL_REVIEW.private.json').read_text())
    assert set(notes) == set(selected_ids) and all(n['reviewed_en_es_and_report_context'] for n in notes.values())
    reports = {r['ImageID']: r for r in json.loads((source/'metadata/grounded_reports_20240819.json').read_text())}
    with zipfile.ZipFile(source/'metadata/master_table.csv.zip') as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open('master_table.csv'), encoding='utf-8-sig')))
    by_image = defaultdict(list)
    for row in rows:
        by_image[row['ImageID']].append(row)
    mask = json.loads(Path(cfg['base_mask']).read_text())
    eligibility = {e['event_id']: e for e in mask['events']}
    material = json.loads(Path(cfg['base_material']).read_text())['source']
    prepared = {t['edit_id']: t for t in material['tasks']}
    queries = {r['query']['query_id']: r['query'] for r in material['base_plan']}
    events = {e['event_id']: e for e in material['events']}
    admitted = [t for t in tasks if eligibility[t['case_id']]['prior_core'] is False]
    admitted_priors = {t['edit_case']['image_id'] for t in admitted}
    admitted_patients = {t['patient_id'] for t in admitted}
    conflicts = defaultdict(set)
    for t in tasks:
        for role in ('edit_case', 'eval_case'):
            v = t[role]
            conflicts[(v['image_id'], v['question'])].add(v['gold_answer'])
    audit, image_headers = [], {}
    for t in selected:
        a, b, cid = t['edit_case'], t['eval_case'], t['case_id']
        rr = reports[b['image_id']]
        source_rows = [rows[i-1] for i in t['source_master_rows']]
        ff = [f for f in rr['findings'] if t['label'] in (f.get('labels') or []) and f.get('progression') in t['progression_statuses']]
        checks = {
            'source_row_image_pair_label': all(r['ImageID'] == b['image_id'] and r['prior_imageID'] == a['image_id'] and r['label'] == t['label'] for r in source_rows),
            'source_row_current_study': all(str(r['StudyID']) == str(b['study_id']) == str(rr['StudyID']) for r in source_rows),
            'explicit_previous_pointer': rr['PreviousImageID'] == a['image_id'] and str(rr['PreviousStudyID']) == str(a['study_id']),
            'distinct_images': a['image_id'] != b['image_id'],
            'same_current_patient_and_date': len({r['PatientID'] for r in by_image[b['image_id']]}) == 1 and len({r['StudyDate_DICOM'] for r in by_image[b['image_id']]}) == 1,
            'source_progression_to_binary': all({'new':'01', 'resolved':'10'}.get(r['progression_status']) == t['presence_code'] for r in source_rows),
            'source_finding_en_es_location': all(any(f.get('sentence_en') == r['sentence_en'] and f.get('sentence_es') == r['sentence_es'] and sorted(f.get('locations') or []) == sorted(ast.literal_eval(r['locations'])) for f in ff) for r in source_rows),
            'same_question_and_references': a['question'] == b['question'] == f"Is there evidence of {t['label']} in this chest radiograph?" and a['gold_answer'] == ('Yes' if t['presence_code'][0] == '1' else 'No') and b['gold_answer'] == ('Yes' if t['presence_code'][1] == '1' else 'No'),
            'no_global_same_input_reference_conflict': all(len(conflicts[(v['image_id'],v['question'])]) == 1 for v in (a,b)),
        }
        e = events[cid]
        prior_q, future_q = queries[e['edit_query_id']], queries[e['followup_query_id']]
        checks['prepared_prior_followup_bindings'] = all(q['source_image_id'] == v['image_id'] and q['question'] == v['question'] and q['reference'] == v['gold_answer'] for q,v in [(prior_q,a),(future_q,b)])
        checks['prepared_raw_and_render_path_basenames'] = all(Path(q['original_image_path']).name == v['image_id'] and Path(q['image_path']).name == v['image_id'] for q,v in [(prior_q,a),(future_q,b)])
        checks['training_native_bound_to_prior'] = prepared[cid]['native'] == prior_q
        independent = a['image_id'] in by_image
        crosscheck = None
        if independent:
            pr = by_image[a['image_id']][0]
            crosscheck = pr['PatientID'] == source_rows[0]['PatientID'] and str(pr['StudyID']) == str(a['study_id']) and pr['StudyDate_DICOM'] < source_rows[0]['StudyDate_DICOM']
            assert date.fromisoformat(a['study_date']) < date.fromisoformat(b['study_date'])
        prior_report = reports.get(a['image_id'])
        other_same_label = [f for f in rr['findings'] if t['label'] in (f.get('labels') or []) and f not in ff]
        locations = sorted({loc for f in ff for loc in (f.get('locations') or [])})
        for v in (a,b):
            if v['image_id'] not in image_headers:
                p = package/v['relative_image_path']
                with p.open('rb') as f:
                    header = f.read(33)
                assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
                width,height,depth,color,_,_,_ = struct.unpack('>IIBBBBB',header[16:29])
                image_headers[v['image_id']] = dict(width=width,height=height,bit_depth=depth,color_type=color,bytes=p.stat().st_size)
        audit.append(dict(case_id=cid,source_master_rows=t['source_master_rows'],label=t['label'],presence_code=t['presence_code'],checks=checks,
            independent_prior_patient_date_available=independent,independent_patient_date_pass=crosscheck,
            source_current_image_id=b['image_id'],source_prior_image_id=a['image_id'],source_current_patient=source_rows[0]['PatientID'],
            source_findings=ff,source_context=rr['findings'],prior_report_findings=prior_report['findings'] if prior_report else None,
            same_label_other_current_findings=other_same_label,location_in_source=locations,question_has_location=False,
            own_sentence_explicit_new=any(re.search(r'\bnew\b|\bnewly\b|\bappearance\b',f['sentence_en'],re.I) for f in ff),
            followup_is_another_admitted_prior=b['image_id'] in admitted_priors,
            patient_has_any_admitted_edit=t['patient_id'] in admitted_patients,
            prior_core=eligibility[cid]['prior_core'],followup_core=eligibility[cid]['followup_core'],base_status=eligibility[cid]['status'],
            manual_review=notes[cid]))
    (private/'CASE_EVIDENCE.private.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (private/'IMAGE_HEADERS.private.json').write_text(json.dumps(image_headers,indent=2)+'\n')
    names = list(audit[0]['checks'])
    flags = Counter(flag for row in audit for flag in row['manual_review']['flags'])
    independent = [r for r in audit if r['independent_prior_patient_date_available']]
    result = dict(status='SOURCE_AUDIT_COMPLETE_WITH_SEMANTIC_CONCERNS',tasks=34,unique_pairs=len({(r['source_prior_image_id'],r['source_current_image_id']) for r in audit}),
        unique_patient_groups=len({t['patient_id'] for t in selected}),unique_images=len(image_headers),presence_codes={'01':28,'10':6},
        structural_checks={name:dict(pass_count=sum(r['checks'][name] for r in audit),fail_count=sum(not r['checks'][name] for r in audit)) for name in names},
        independent_prior_patient_date_tasks=len(independent),independent_prior_patient_date_pairs=len({r['source_prior_image_id'] for r in independent}),
        independent_patient_date_pass=sum(r['independent_patient_date_pass'] for r in independent),
        explicit_previous_pointer_only_tasks=34-len(independent),
        manual_en_es_context_reviewed=34,semantic_flags=dict(flags),
        tasks_with_specific_semantic_flags=sum(bool(r['manual_review']['flags']) for r in audit),
        new_tasks_own_sentence_explicit_new=sum(r['own_sentence_explicit_new'] for r in audit if r['presence_code']=='01'),
        new_tasks_without_explicit_new_word=28-sum(r['own_sentence_explicit_new'] for r in audit if r['presence_code']=='01'),
        prior_reports_available_tasks=sum(r['prior_report_findings'] is not None for r in audit),
        source_location_present_but_not_in_question_tasks=sum(bool(r['location_in_source']) for r in audit),
        another_current_finding_has_same_label_tasks=sum(bool(r['same_label_other_current_findings']) for r in audit),
        followup_is_another_admitted_prior_tasks=sum(r['followup_is_another_admitted_prior'] for r in audit),
        base_qualification_status_counts=dict(Counter(r['base_status'] for r in audit)),
        admitted_change_tasks=sum(r['prior_core'] is False for r in audit),
        specific_CT_comparison_concern_admitted_tasks=sum(r['prior_core'] is False and 'CT_COMPARISON_NOT_BOUND_TO_LINKED_CXR' in r['manual_review']['flags'] for r in audit),
        images='Existing selected34-task PNG headers/readability checked; no pixel-level radiologist validation or archive re-extraction.',
        changes=dict(model_calls=0,Judge_calls=0,UNKNOWN_retries=0,cohort_changes=0,label_changes=0,stopped_or_restarted_existing_jobs=0),
        limitation='Metadata and bilingual source consistency audit; generic disease presence, uncertain findings and actual clinical image truth are not independently validated. Lack of temporal words alone is not a labeling error.')
    assert len(audit) == 34 and sum(result['base_qualification_status_counts'].values()) == 34
    assert all(v['pass_count']+v['fail_count'] == 34 for v in result['structural_checks'].values())
    (public/'AUDIT_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
