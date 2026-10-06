"""Lock the pilot before model results; aggregate original QAs by image."""
import collections
import json
import os
from pathlib import Path
import re
from audit import read,write,save,accepted

def outbound_permission(s):
    if s['source_dataset']=='VQA-RAD':return 'ALLOWED','Official author CC0 license independently checked; anonymous QA text only'
    if s['source_dataset']!='PMC_PRIMARY':return 'UNKNOWN','Local research authorization does not establish external processing permission'
    lic=json.dumps(s['license']).lower()
    if re.search(r'noncommercial|non-commercial|by-nc|by-nd|no.?derivatives|share.?alike',lic):return 'BLOCKED','License not eligible under this external processing protocol'
    if re.search(r'reproduced|with permission|courtesy',s['source_text'],re.I):return 'UNKNOWN','Third-party figure rights need source-specific confirmation'
    if re.search(r'licenses/by/[\d.]|publicdomain/zero/',lic):return 'ALLOWED','Explicit retained CC-BY/CC0 license; original anonymous caption text only'
    return 'UNKNOWN','No verified outbound license in retained source metadata'

def category(e):
    if e['original_source']!='VQA-RAD' or e['question_type']!='IMAGE_DIRECT':return None
    q=e['original_question'].lower()
    if e['attribute_relation']=='acquisition_modality_or_contrast':return None
    if 'side of the body' in q or "patient's right side" in q or 'what two tributaries' in q:return 'SIDE_OR_COUNT'
    if (e['attribute_relation']=='localization_or_referent' or q.startswith('what part of the posterior brain')) and not e['target_answer'].lower().startswith('sella'):return 'LOCALIZATION'
    if any(k in q for k in ['pathology','abnormal findings','what is abnormal','what is the lesion','describe this lesion']):return 'LESION_OR_FINDING'
    return None

def main():
    root=Path(os.environ['V2_ROOT']);p=root/'private'
    ss=read(p/'source_registry.jsonl');edits=read(p/'edit_inventory.jsonl');cc=read(p/'v1_revised_candidates.jsonl')
    covered={c['edit_id'] for c in cc if accepted(c,'FIT')}
    # Fixed deterministic priority: literal negative evidence in the existing
    # source registry (either frozen role, solely for retrieval), then order.
    # No student/model outputs or scores. Unique target within finding category.
    negatives=[s['source_text'].lower() for s in ss if s.get('original_answer','') and str(s['original_answer']).lower() in {'no','none'} or re.search(r'\bno\b',s['source_text'],re.I)]
    def negative_score(e):
        terms=[x for x in ['ring','pulmonary edema','subdural','pleural effusion','hyperintensity'] if x in e['target_answer'].lower()]
        return int(any(all(x in s for x in terms) for s in negatives)) if terms else 0
    groups={k:[] for k in ['LESION_OR_FINDING','LOCALIZATION','SIDE_OR_COUNT']}
    for e in edits:
        k=category(e)
        if e['edit_id'] not in covered and k:groups[k].append(e)
    selected=[]
    for k,rows in groups.items():
        ordered=sorted(rows,key=lambda e:(-negative_score(e) if k=='LESION_OR_FINDING' else 0,e['position']))
        targets=set()
        for e in ordered:
            if k=='LESION_OR_FINDING' and e['target_answer'].lower() in targets:continue
            selected.append(dict(e,pilot_question_type=k));targets.add(e['target_answer'].lower())
            if sum(x['pilot_question_type']==k for x in selected)==4:break
    selected.sort(key=lambda e:e['position'])
    assert len(selected)<=12 and all(e['edit_id'] not in covered for e in selected)
    save(p/'PILOT_SELECTION.json',dict(rule='Four per category; finding category ranks literal negative target-token relevance in V1 registry then position, distinct target; location and side/count rank position. No modality edits. Membership locked before model calls.',positions=[e['position'] for e in selected],edits=selected,question_types=dict(collections.Counter(e['pilot_question_type'] for e in selected)),frozen_before_model_execution=True,student_outputs_read=False,max_distinct_sources_per_edit=10,max_new_papers=40,max_new_units=60,max_relations=120,max_model_calls_including_retries=300))
    # Explicit outbound decisions do not assume authorization from an old
    # local research label. All SLAKE remains UNKNOWN in this V2 pilot.
    license_record=json.loads(Path(os.environ['VQA_LICENSE']).read_text())
    license_name=license_record['data']['attributes']['name']
    assert 'CC0' in license_name
    units=[]
    byimage=collections.defaultdict(list)
    for s in ss:
        if s['source_dataset']=='VQA-RAD' and s['assigned_role']=='FIT' and s['source_role_check']=='PASS':byimage[s['image_id']].append(s)
    for image,rows in sorted(byimage.items()):
        s=rows[0];text='\n\n'.join('QA '+x['source_record_id']+':\n'+x['source_text'] for x in rows)
        units.append(dict(source_id='V2_QA_UNIT:'+image,source_dataset='VQA-RAD',source_group_id=s['source_group_id'],image_id=image,image_sha256=s['image_sha256'],image_path=s['image_path'],assigned_role='FIT',source_role_check='PASS',evidence_origin='ORIGINAL_QA',source_text=text,evidence_ids=[x['source_id'] for x in rows],original_QA_records=[dict(source_id=x['source_id'],question=x['original_question'],answer=x['original_answer']) for x in rows],image_binding='All original author QA rows share the same retained image ID; source-role and whole-case exclusions inherited from frozen V1.',license=license_name,permission_reference='https://api.osf.io/v2/nodes/89kps/ + linked official license endpoint',outbound_status='ALLOWED',outbound_reason='Public author VQA-RAD archive with CC0; only deidentified original question/answer text and anonymous binding IDs sent. No image bytes or raw case URLs sent.',acquisition='V1_CACHE_REUSE',patient_independence='UNKNOWN',modality=None,body_part=next((x['body_part'] for x in rows if x.get('body_part')),None)))
    permissions=[]
    for s in ss:
        status,reason=outbound_permission(s)
        permissions.append(dict(source_id=s['source_id'],source_group_id=s['source_group_id'],outbound_status=status,reason=reason))
    write(p/'outbound_status.jsonl',permissions);write(p/'v1_reused_units.jsonl',units)
    print(json.dumps(dict(selected_positions=[e['position'] for e in selected],types=dict(collections.Counter(e['pilot_question_type'] for e in selected)),legal_reused_fit_units=len(units)),indent=2))

if __name__=='__main__':main()
