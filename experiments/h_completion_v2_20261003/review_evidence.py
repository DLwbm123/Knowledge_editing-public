"""Six-check evidence packet assembly. No clinical labels or model outputs."""
import collections
import copy
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def jl(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def lines(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))


def blind(n):
    return {k:v for k,v in n.items() if k not in {'base_correct','opaque_Base_id'}}


def norm(s):
    return ' '.join(s.casefold().split()).strip(' .')


def target_clauses_supported(target,answer):
    # A literal subset is sufficient to reject this proposed contrast; not medical reasoning.
    parts=[norm(x) for x in re.split(r'[,;]',target) if norm(x)]
    return bool(parts) and all(p in norm(answer) for p in parts)


NEEDS={
 'disease_inventory':'完整疾病集合须由原报告/原QA明确枚举，不能把图注只提及一种病变当其它病变不存在。',
 'regional_disease_inventory':'逐项核对原问句指定区域/侧别及完整疾病集合；区域不同不能删除疾病或左右限定。',
 'symptom':'具名病变与此图指代的绑定，以及该病变症状的原KG或原报告；症状可能跨疾病共存，须专业确认target不适用。',
 'cause':'具名病变、此图指代及其病因的明确原文/KG；病例背景或关联不是因果，不以另一个病因排除native target。',
 'prevention':'具名病变与图像指代，以及原KG/专业来源的预防关系；共用预防措施可能仍属于应泛化范围。',
 'treatment':'图像中的确诊病变、严重程度及原治疗出处；病例采用的治疗不自动代表唯一正确方案，须专业审核。',
 'severity_conditioned_treatment':'首先证明最严重病变的唯一指代及排序依据，再核对该程度的治疗；现有检测框不能给出严重程度排序。',
 'function':'原位置QA/框/mask与完整颜色、方位、形态指代的唯一实体绑定，再绑定具名KG功能；已含target全部子句的答案不能作H。',
 'organ_system_inventory':'原QA和具名框/类别映射须支持完整器官集合；缺少某个框或mask字典不证明该器官不存在。',
 'localization_or_referent':'原图侧别/坐标及唯一病变或实体指代；病人的左右和画面左右必须分别记录。',
 'acquisition_modality_or_contrast':'原图精确subpanel及原报告/采集信息，区分CT/MRI与造影状态；混合图的其它panel不能替代所问图像。',
 'visual_shape':'原图上唯一结构、原标注或图注的形态证据；不能从另一结构的形状替换此图指代。',
 'visual_measurement_or_signal':'原图采集序列、参照结构、测量或原报告信号/密度记录；保留强度和程度限定，不凭像素新增诊断。',
 'clinical_inference':'原报告受累结构或原诊断及推断证据；原问句可能性限定不能升级为确定因果，需真实专业判断。',
 'image_explanation':'原图征象与解释关系的原报告或签核，不以未提及为排除其它原因。',
 'context_or_clinical_sign':'先确认本题是否图像局部事实或一般成像知识，再核对完整预设和征象。',
 'general_anatomy_or_imaging_comparison':'作者须明确本题的图像/患者变量与编辑作用域；字面通用命题换图没有相反真值，不能改写native或删题。',
 'visual_finding_or_entity':'所有原限定在新图成立的原QA/图注/结构证据；需唯一指代和原target排他证据，不能依靠不同字符串。'
}


def checks(a,source_answer=False):
    return {
      'QUESTION_VALID':{'status':'UNKNOWN','reason':NEEDS[a['relation_type']]},
      'ANSWER_SOURCE':{'status':'PASS' if source_answer else 'UNKNOWN','reason':'原始QA或具名KG值直接保留；不代表问题绑定已成立' if source_answer else 'PMC自动QA仅检索；原caption/report尚未形成已验证H答案'},
      'TARGET_INAPPLICABLE':{'status':'UNKNOWN','reason':'需target排他证据；未提及、文字不同、可能共存均不能通过'},
      'OUTSIDE_EDIT_GENERALIZATION':{'status':'UNKNOWN','reason':'须比较native与新实体/关系及真实泛化作用域，不以不同文件名或问题改写决定'},
      'GLOBAL_ISOLATION':{'status':'UNKNOWN','reason':'按完整native/future/eval/fit/CAL源身份排除，仍须跨来源派生和病例/患者证据；患者未知不声称独立'},
      'REVIEW_AUTHENTICITY':{'status':'UNKNOWN','reason':'Codex仅做结构/出处审计；需要医学判断的项目尚无真实专业签核'}
    }


def article_binding(root,c):
    meta=c['source_metadata'];paper=meta['paper'];base=root/'private/primary_evidence'
    p=base/(paper+'.PRIMARY_EVIDENCE.json')
    if not p.exists():return dict(status='PRIMARY_ACQUISITION_FAILED_OR_PENDING',paper=paper)
    d=json.loads(p.read_text());im=next((x for x in d['images'] if x['figure']==meta['figure_path']),None)
    corrected=base/'PRIMARY_BINDING_CORRECTIONS.json'
    correction=next((x for x in json.loads(corrected.read_text()) if x['candidate_id']==c['candidate_id']),None) if corrected.exists() else None
    if correction:
        im=dict(figure=correction['corrected_figure_path'],primary_figure_id=correction['primary_figure_id'],
                source_URL=correction['source_URL'],bytes=correction['bytes'],caption=correction['caption'],
                status='ACQUIRED_PRIMARY_BINDING_CORRECTION_PENDING_PANEL_AND_SIX_CHECKS')
    if im is None:return dict(status='IMAGE_NOT_ACQUIRED_LICENSE_OR_MAPPING',paper=paper,license=d['license'])
    if not im['status'].startswith('ACQUIRED'):return dict(status=im['status'],paper=paper,source=im)
    a=ET.parse(base/(paper+'.xml'));fid=im['primary_figure_id']
    paragraphs=[]
    for para in a.findall('.//p'):
        if any(fid in x.get('rid','').split() for x in para.findall('.//xref')):
            paragraphs.append(''.join(para.itertext()))
    return dict(status='SOURCE_IMAGE_AND_PRIMARY_CAPTION_BOUND',paper=paper,source=im,
                source_path=str(base/im['figure']),referencing_original_paragraphs=paragraphs,
                source_derived_draft=correction,literal_answer_extract=correction['literal_answer_extract'] if correction else None,
                article_license=d['license'],clinical_truth_assigned=False,patient_independence='UNKNOWN',
                subpanel_and_complete_question_verification='PENDING_PROFESSIONAL_REVIEW')


def main():
    started=time.time();cpu=time.process_time();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    root=Path(os.environ['RUN_ROOT']);manifest=json.loads((root/'RUN_MANIFEST.json').read_text())
    assert started<manifest['deadline_epoch'] and not (root/'STOP').exists()
    complete=root/'private/primary_evidence/PRIMARY_ACQUISITION_RECOVERY_COMPLETE.json';assert complete.exists()
    audits=jl(root/'private/H_GAP_AUDIT_146.jsonl');assert len(audits)==146
    original=root/'private/revisions/PRE_PRIMARY_REVIEW';original.mkdir(parents=True,exist_ok=True)
    for name in ('H_GAP_AUDIT_146.jsonl','H_COVERAGE_146.json','H_EVIDENCE_MANIFEST.jsonl'):
        p=root/'private'/name;q=original/name
        assert not q.exists();q.write_bytes(p.read_bytes())
    by_edit={a['edit_id']:a for a in audits}
    old=Path(manifest['previous_run'])
    unknown=json.loads((old/'private/H_SUPPORT_FINAL/UNKNOWN_REVIEW_QUEUE.json').read_text());assert len(unknown)==22
    inherited=jl(root/'private/PR20_INHERITED_EVIDENCE.jsonl');assert len(inherited)==12
    banned={tuple(x) for x in json.loads((root/'private/GLOBAL_SOURCE_EXCLUSION.json').read_text())['banned_images']}
    case_rows=json.loads(Path('/data/bmw/DataP/knowledge_editing/data/m3bench/VQA-RAD/VQA_RAD Dataset Public.json').read_text())
    banned_cases={x['image_case_url'] for x in case_rows if ('VQA-RAD',x['image_name'].removesuffix('.jpg').lower()) in banned}
    existing=jl(root/'private/H_CANDIDATE_REVIEW_PACKET/CANDIDATES.jsonl');pmc=jl(root/'private/H_CANDIDATE_REVIEW_PACKET/PMC_PRIMARY_CANDIDATES.jsonl')
    reviewed=[]
    for c in existing:
        c=copy.deepcopy(c);a=by_edit[c['edit_id']];c['native']=blind(c['native']);c['checks']=checks(a,True)
        s=c['source'];family=c.get('independent_annotation_family')
        c['source_level_role_check']='PASS' if (s['dataset'],s['image_id']) not in banned else 'FAIL'
        c['patient_independence']='UNKNOWN';c['training_admitted_v2']=False
        if c['evidence_type']=='ORIGINAL_QA' and norm(c['question_on_source'])!=norm(c['native']['question']):
            c['evidence_type']='SOURCE_DERIVED_QA';c['original_QA_retained_unchanged']=True
            c['question_on_source']=c['native']['question'];c['language_or_question_transfer_review']='Native preserved; source proposition correspondence requires individual review'
        answer=c.get('answer',s.get('answer',''))
        if target_clauses_supported(c['native']['reference'],answer):
            c['checks']['TARGET_INAPPLICABLE']=dict(status='FAIL',reason='源答案已逐字包含native target的全部逗号/分号子句；不能当成同问相反真值',test='literal target-clause containment only; not clinical inference')
            c['review_state']='REJECTED_TARGET_CONTRAST_NOT_ESTABLISHED'
        reviewed.append(c)
    for c in pmc:
        c=copy.deepcopy(c);a=by_edit[c['edit_id']];c['native']=blind(c['native']);c['evidence_type']='SOURCE_DERIVED_QA';c['checks']=checks(a)
        c['primary_binding']=article_binding(root,c);c['image_downloaded']=c['primary_binding']['status']=='SOURCE_IMAGE_AND_PRIMARY_CAPTION_BOUND'
        c['training_admitted_v2']=False;c['patient_independence']='UNKNOWN'
        c['raw_auto_QA_evidence_family']=c['source_metadata']['figure_path']+':'+norm(c['source_metadata']['author_generated_question'])
        c['source_group']=c['source_metadata']['paper'];c['primary_evidence_is_annotation']=False
        c['review_state']='PROFESSIONAL_QA_SCOPE_AND_CASE_REVIEW_REQUIRED' if c['image_downloaded'] else 'PRIMARY_BINDING_NOT_AVAILABLE'
        reviewed.append(c)
    counts=collections.Counter(c['edit_id'] for c in reviewed);assert max(counts.values())<=5
    unknown_reviews=[]
    for c in unknown:
        c=copy.deepcopy(c);c['native']=blind(c['native']);a=by_edit[c['edit_id']]
        c['v2_missing_evidence']=NEEDS[a['relation_type']];c['v2_checks']=checks(a,True)
        c['prior_verified_alternatives_retained']=a['prior_H_relations'];c['clinical_verification']=False;c['admitted_H_fit']=False
        c['duplicate_of_first_round_candidate']=any(x.get('prior_review_id')==c['review_id'] for x in reviewed)
        unknown_reviews.append(c)
    for a in audits:
        a['native']=blind(a['native']);a['missing_evidence'].append(NEEDS[a['relation_type']])
        a['full_qualifier_binding_required']=a['native']['question'];a['first_round_candidate_count']=counts[a['edit_id']]
        a['new_primary_images_bound']=sum(c['edit_id']==a['edit_id'] and c.get('image_downloaded',False) for c in reviewed)
        a['scope_decision']='AUTHOR_AND_PROFESSIONAL_SCOPE_CONFIRMATION_REQUIRED' if a['question_type']=='IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE' else 'NO_IMPLICIT_SCOPE_CHANGE'
        a['evidence_search_not_exhausted']=True;a['professionally_verified_new_H_fit']=0
    for e in inherited:
        e['native']=blind(e['native']);s=e['source'];case=s['raw_source'].get('image_case_url')
        e['source_and_known_case_exclusion']='PASS' if (s['dataset'],s['image_id']) not in banned and case not in banned_cases else 'FAIL'
        e['evidence_type']='ORIGINAL_QA' if norm(s['question'])==norm(e['native']['question']) else 'SOURCE_DERIVED_QA'
        e['v2_review_status']='PR20_QUALIFICATION_PRESERVED_V2_PATIENT_DERIVATION_REVIEW_PENDING';e['training_admitted_v2']=False
    out=root/'private/H_CANDIDATE_REVIEW_PACKET';lines(out/'SIX_CHECK_REVIEWED_CANDIDATES.jsonl',reviewed)
    lines(out/'PR20_UNKNOWN_ALL_22_REVIEW.jsonl',unknown_reviews)
    lines(out/'PROFESSIONAL_REVIEW_QUEUE.jsonl',[c for c in reviewed if c['checks']['TARGET_INAPPLICABLE']['status']!='FAIL'])
    lines(root/'private/PR20_INHERITED_EVIDENCE_V2_REVIEW.jsonl',inherited)
    lines(root/'private/H_GAP_AUDIT_146.jsonl',audits)
    with (out/'PROFESSIONAL_SIGNOFF_TEMPLATE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['candidate_id','position','reviewer_name','qualification','date','question_valid','answer_source','target_inapplicable','outside_edit_scope','global_isolation','answer_extract','evidence_locator','decision','clinical_signoff']);w.writeheader()
        for c in reviewed:w.writerow(dict(candidate_id=c['candidate_id'],position=c['position']))
    # A signed/admitted manifest cannot contain pending or failed evidence.
    assert not (root/'private/H_EVIDENCE_MANIFEST.jsonl').read_text().strip()
    coverage=json.loads((root/'private/H_COVERAGE_146.json').read_text())
    for row in coverage['rows']:
        a=by_edit[row['edit_id']];row.update(new_candidates=counts[row['edit_id']],new_primary_images_bound=a['new_primary_images_bound'],missing_evidence=a['missing_evidence'],scope_decision=a['scope_decision'])
    coverage.update(stage_status='FIRST_ROUND_DATA_MILESTONE_COMPLETE_PROFESSIONAL_SCOPE_AND_PATIENT_REVIEW_PENDING',
                    target_146_unchanged=True,original_PR20_qualified_H_coverage='8/146 preserved, not new clinical labels',
                    first_round_candidate_relations=len(reviewed),first_round_candidate_edits=len(counts),
                    new_v2_admitted_H_fit=0,full_model_comparison_data_gate='NOT_ADMITTED')
    save(root/'private/H_COVERAGE_146.json',coverage)
    ledger=json.loads((root/'private/PRIMARY_DOWNLOAD_LEDGER.json').read_text());article=list((root/'private/primary_evidence').glob('*.PRIMARY_EVIDENCE.json'))
    documents=[json.loads(p.read_text()) for p in article]
    images={x['figure'] for d in documents for x in d['images'] if x['status'].startswith('ACQUIRED')}
    summary=dict(N=146,status=coverage['stage_status'],all_146_H_fit_complete=False,sparse_H_substitution=False,
      question_types=dict(collections.Counter(a['question_type'] for a in audits)),original_PR20_qualified_edits=8,original_PR20_qualified_relations=12,
      new_v2_H_fit_admitted=0,new_clinical_signoffs=0,source_exhaustion_claim=False,
      candidates=len(reviewed),candidate_edits=len(counts),SLAKE_candidate_edits=len({c['edit_id'] for c in reviewed if c['native']['dataset']=='SLAKE'}),
      evidence_types_proposed=dict(collections.Counter(c['evidence_type'] for c in reviewed)),all_previous_UNKNOWN_reviewed=len(unknown_reviews),
      rejected_literal_target_containment=sum(c['checks']['TARGET_INAPPLICABLE']['status']=='FAIL' for c in reviewed),
      new_primary_papers_available=len(documents),eligible_article_license_groups=sum(d['license_is_CC_BY_or_CC0'] for d in documents),
      primary_images_acquired=len(images),source_groups_are_not_independent_patients=True,known_independent_new_patients=0,
      image_bound_candidate_relations=sum(c.get('image_downloaded',False) for c in reviewed),
      image_bound_edits=len({c['edit_id'] for c in reviewed if c.get('image_downloaded',False)}),
      different_auto_QA_retrieval_families=len({c['raw_auto_QA_evidence_family'] for c in reviewed if 'raw_auto_QA_evidence_family' in c}),
      automatic_QA_families_are_not_verified_annotations=True,
      different_cached_QA_families=len({c['independent_annotation_family'] for c in reviewed if 'independent_annotation_family' in c}),
      candidate_source_groups=len({c.get('source_group',c.get('source',{}).get('source_group','')) for c in reviewed}),
      metadata_download_bytes=ledger['metadata_download_bytes'],image_download_bytes=ledger['image_download_bytes'],
      CPU_review_seconds=time.process_time()-cpu,wall_review_seconds=time.time()-started,GPU_hours=0,Judge_requests=0,
      no_model_or_evaluation_changes=True,no_protected_test_QA_used=True,bilingual_new_independent_annotations=0,
      MIMIC_CXR='ACCESS_NOT_ESTABLISHED_NOT_USED',original_clock_preserved=True,original_comparison_budget_preserved=True)
    save(root/'public/MILESTONE_3_SUMMARY.json',summary)
    save(root/'private/REVIEW_RECEIPT.json',dict(pid=os.getpid(),argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),RUN_ROOT=str(root),summary=summary))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    assert target_clauses_supported('toy one, toy two','toy three; toy two; toy one')
    assert not target_clauses_supported('toy one, toy two','toy one')
    main()
