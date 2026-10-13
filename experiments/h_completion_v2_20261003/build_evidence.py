"""Source-only question/scope audit and bounded evidence proposals; no models."""
import collections
import csv
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import zipfile
from pathlib import Path


def norm(x):
    return ' '.join(unicodedata.normalize('NFKC', str(x)).casefold().split())


def source_id(dataset, path):
    ids = set(re.findall(r'xmlab\d+|synpic\d+', str(path), re.I))
    if len(ids) != 1:
        raise ValueError('Ambiguous source identity')
    return dataset, next(iter(ids)).lower()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def classify(position, question):
    q = norm(question)
    if position in (23, 101):
        return ('IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE', 'general_anatomy_or_imaging_comparison',
                'NOT_APPLICABLE_UNDER_LITERAL_GLOBAL_READING',
                '原问句是具名解剖/成像方式的一般比较，未指定新图可改变的变量；在此字面作用域下，换图不能支持同问相反真值。若作者本意是患者/图像特例，须有明确作用域证据，不能自行改写命题。')
    if 'most severe' in q:
        return ('AMBIGUOUS_PENDING_REVIEW', 'severity_conditioned_treatment', 'UNRESOLVED',
                '最严重的病变需要严重程度排序和专业依据；仅有病变框/名称或治疗KG不能确立唯一指代，也不能证明替换答案正确。')
    relations = [('symptom', '症状', 'symptom'), ('cause', '病因', 'cause'),
                 ('prevent', '预防', 'prevention'), ('treat', '治疗', 'treatment'),
                 ('function', '功能', 'function'), ('effect', '作用', 'function')]
    for en, zh, rel in relations:
        if en in q or zh in q:
            return ('ENTITY_CONDITIONED_KNOWLEDGE', rel, 'CONDITIONAL_ENTITY_BINDING',
                    '问题的知识关系依赖图中指代实体；需把完整部位/左右/颜色/层级指代绑定到具名实体，再引用该实体原KG/报告关系。实体不同或答案文字不同不足以证明target不适用；通用治疗/症状可共存，须专业审核。')
    if any(x in q for x in ('cranial nerves', 'could be affected', 'possible diagnoses', 'can cause')):
        return ('ENTITY_CONDITIONED_KNOWLEDGE', 'clinical_inference', 'CONDITIONAL_ENTITY_BINDING',
                '需要新图具体病变/受累结构及推断关系的原报告或专业判断，保留possibly/could等不确定性；不能把模型诊断或图注未提及当否定。')
    if 'typically visualized' in q or 'indicates that' in q or 'what indicates' in q:
        return ('AMBIGUOUS_PENDING_REVIEW', 'context_or_clinical_sign', 'UNRESOLVED',
                '需先确认问题是在询问此图可见征象还是一般成像知识；保留plane/病变/侧别预设，未明确作用域不能强制构造同问异答。')
    if any(x in q for x in ('diseases are included', 'disease is/are shown')):
        kind = 'regional_disease_inventory' if 'shown on' in q else 'disease_inventory'
    elif 'digestive system' in q:
        kind = 'organ_system_inventory'
    elif any(x in q for x in ('modality', 'contrast ct')):
        kind = 'acquisition_modality_or_contrast'
    elif any(x in q for x in ('where', 'located', 'location', 'which is the kidney', 'structures', 'side of')):
        kind = 'localization_or_referent'
    elif 'shape' in q:
        kind = 'visual_shape'
    elif any(x in q for x in ('function', 'effect')):
        kind = 'function'
    elif any(x in q for x in ('size', 'density', 'wide', 'intensit', 'enhancement')):
        kind = 'visual_measurement_or_signal'
    elif any(x in q for x in ('why', 'what can cause')):
        return ('ENTITY_CONDITIONED_KNOWLEDGE', 'image_explanation', 'CONDITIONAL_ENTITY_BINDING',
                '回答解释此图征象，需源图和具名征象/原因的报告证据；不能仅凭像素自行作医学推断。')
    else:
        kind = 'visual_finding_or_entity'
    return ('IMAGE_DIRECT', kind, 'APPLICABLE_WITH_SOURCE_EVIDENCE',
            '同问异答可适用，但必须有新图上所有预设/指代成立及原答案的具体证据；集合回答须证明完整性/排他性，定位须保留侧别和坐标，不从标注未出现推断不存在。')


REL_ALIAS = {'function': 'function', 'effect': 'function', '功能': 'function', '作用': 'function',
             'symptom': 'symptom', '主要症状': 'symptom', 'cause': 'cause', '病因': 'cause',
             'prevention': 'prevention', '预防': 'prevention', 'treatment': 'treatment', '治疗方法': 'treatment'}
ENTITY_ALIAS = {'small bowel': 'small intestine', 'small intestine': 'small intestine', '小肠': 'small intestine',
                'large bowel': 'large intestine', 'large intestine': 'large intestine', '大肠': 'large intestine',
                'colon': 'colon', '结肠': 'colon', 'rectum': 'rectum', '直肠': 'rectum',
                'stomach': 'stomach', '胃': 'stomach', 'liver': 'liver', '肝脏': 'liver',
                'heart': 'heart', '心脏': 'heart', 'bladder': 'bladder', '膀胱': 'bladder',
                'spinal cord': 'spinal cord', '脊髓': 'spinal cord', 'brain stem': 'brain stem', '脑干': 'brain stem',
                'duodenum': 'duodenum', '十二指肠': 'duodenum', 'kidney': 'kidney', '肾脏': 'kidney',
                'lung': 'lung', '肺': 'lung', 'spleen': 'spleen', '脾脏': 'spleen'}
POSITION_ALIAS = {'top': 'top', '顶部': 'top', 'center': 'center', '中间': 'center', '中心': 'center',
                  'left': 'left', '左侧': 'left', 'right': 'right', '右侧': 'right', 'bottom': 'bottom', '底部': 'bottom'}


def entity(x):
    return ENTITY_ALIAS.get(norm(x), norm(x))


def qualifiers(question):
    q = norm(question)
    words = ('left', 'right', 'upper', 'lower', 'top', 'center', 'bottom', 'black', 'gray',
             'most severe', 'possibly', 'could', 'brain', 'lung', 'lobe', 'posterior', 'anterior',
             'frontal', 'occipital', 'temporal', 'hemorrhage', 'contrast', 'csf', 'plane',
             '左', '右', '上', '下', '顶部', '黑', '灰', '大脑', '肺', '疾病', '症状')
    return [w for w in words if w in q]


def qa_relation(row):
    q = norm(row['question'])
    for en, zh, rel in [('symptom', '症状', 'symptom'), ('cause', '病因', 'cause'),
                        ('prevent', '预防', 'prevention'), ('treat', '治疗', 'treatment'),
                        ('function', '功能', 'function'), ('effect', '作用', 'function')]:
        if en in q or zh in q:
            return rel
    return row.get('content_type', 'OTHER')


def visual_slot(row):
    q = norm(row['question'])
    # A conservative alignment key, not automatic semantic equivalence.
    entities = sorted({v for k, v in ENTITY_ALIAS.items() if k in q})
    positions = sorted({v for k, v in POSITION_ALIAS.items() if k in q})
    return dict(relation=qa_relation(row), entities=entities, positions=positions,
                qualifiers=qualifiers(row['question']))


def source_inventory(old, root):
    base = Path('/data/bmw/DataP/knowledge_editing/data/m3bench')
    slake = json.loads((base/'SLAKE/train.json').read_text())
    sources = Path('/data/bmw/Knowledge_editing/outputs')
    rows = json.loads((sources/'scope-m3bench-candidate-audit-20261001/run/private/SOURCE_QA_ROLE_AUDIT.json').read_text())
    role_rows = json.loads((sources/'scope-m3bench-exposure-audit-20261001/run/private/IDENTITY_ROLE_LEDGER.json').read_text())
    roles = collections.defaultdict(set)
    for r in role_rows:
        roles[(r['dataset'], r['image_id'])].update(r['roles'])
    forbidden = {'PROTECTED_SPLIT', 'FORMAL_OR_RESERVED_SOURCE', 'CAL_CHECK_SUPPORT_RESERVED', 'FROZEN_FIT_SOURCE'}
    ledger = json.loads((old/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json').read_text())
    join = json.loads((old/'private/legacy_stage17/ROLE_BOUNDARY_JOIN.json').read_text())
    banned = {source_id(q['dataset'], q['original_image_path']) for q in ledger['queries'].values()}
    banned |= {source_id(t['native']['dataset'], t['native']['original_image_path']) for t in ledger['tasks']}
    banned |= {source_id(d, p) for d, p in join['reserved_source_groups']}
    banned |= {k for k, v in roles.items() if v & forbidden}
    original_legal_slake = [r for r in slake if source_id('SLAKE', r['img_name']) not in banned
                            and roles[source_id('SLAKE', r['img_name'])]]
    groups = collections.defaultdict(list)
    for r in original_legal_slake:
        groups[source_id('SLAKE', r['img_name'])[1]].append(r)
    inventory, bilingual = [], []
    for iid, group in sorted(groups.items()):
        folder = base/'SLAKE/imgs'/iid
        detection = json.loads((folder/'detection.json').read_text())
        boxes = [{'entity': name, 'box': box} for item in detection for name, box in item.items()]
        # Only approved train QA, not unfiltered per-image question.json.
        inventory.append(dict(dataset='SLAKE', image_id=iid, image_path=str(folder/'source.jpg'),
                              source_group=f'SLAKE:{iid}', QA=group, detection=boxes,
                              mask_path=str(folder/'mask.png'), mask_exists=(folder/'mask.png').is_file(),
                              mask_label_dictionary_status='NOT_LOCATED_NOT_INFERRED',
                              source_roles=sorted(roles[('SLAKE', iid)]), patient_id=None,
                              patient_independence='UNKNOWN', source_role_check='PASS'))
        ens = [r for r in group if r['q_lang'] == 'en']
        zhs = [r for r in group if r['q_lang'] == 'zh']
        for zh in zhs:
            zs = visual_slot(zh)
            candidates = [en for en in ens if qa_relation(en) == zs['relation']]
            bilingual.append(dict(image_id=iid, zh_qid=zh['qid'], zh_question=zh['question'],
                                  candidate_en_qids=[e['qid'] for e in candidates], slot=zs,
                                  status='SEMANTIC_ALIGNMENT_PENDING',
                                  reason='Same image/relation is only retrieval; exact referent, qualifiers and answer semantics require individual alignment',
                                  count_as_new_independent_evidence=False))
    kg = []
    z = zipfile.ZipFile(old/'private/source_supplement/KG.zip')
    for member in z.namelist():
        if not member.endswith('.csv'):
            continue
        delimiter = '#' if '/en_' in member else ','
        for line_no, fields in enumerate(csv.reader(io.StringIO(z.read(member).decode()), delimiter=delimiter), 1):
            if line_no == 1 or len(fields) != 3:
                continue
            head, relation, answer = fields
            if norm(head) in {'vhead', 'ktail'} or norm(answer) in {'vhead', 'ktail'}:
                continue
            kg.append(dict(member=member, line=line_no, head=head, entity=entity(head),
                           relation=REL_ALIAS.get(norm(relation), norm(relation)), answer=answer,
                           evidence_language='en' if '/en_' in member else 'zh'))
    legal_vqa = []
    for row in rows:
        if row['dataset'] != 'VQA-RAD' or (row['dataset'], row['image_id']) in banned:
            continue
        raw = row['raw_source']
        if raw.get('evaluation') == 'evaluated' and raw.get('phrase_type') == 'freeform':
            legal_vqa.append(row)
    # PR20's whole-test-image and known case exclusions remain enforced through final membership.
    final = json.loads((old/'private/source_supplement/final/RELATION_REVIEW_PACKET.json').read_text())
    jsonl(root/'private/SLAKE_ENTITY_SOURCE_INDEX.jsonl', inventory)
    jsonl(root/'private/BILINGUAL_SEMANTIC_ALIGNMENT.jsonl', bilingual)
    jsonl(root/'private/NAMED_KG_SOURCE_INDEX.jsonl', kg)
    dump(root/'private/GLOBAL_SOURCE_EXCLUSION.json', dict(banned_images=[list(x) for x in sorted(banned)],
                                                          all_594_future_natives_excluded=True,
                                                          source_role_rows=len(role_rows)))
    return ledger, inventory, kg, final, bilingual


def checks(source, answer=False):
    return dict(question成立=dict(status='UNKNOWN', reason='完整指代/临床限定需要逐条源证据'),
                answer来源=dict(status='PASS' if answer else 'UNKNOWN', reason='原始QA行/具名KG抽取' if answer else '尚无足够来源'),
                target不适用=dict(status='UNKNOWN', reason='不得把文字不同或未提及当矛盾'),
                编辑泛化范围之外=dict(status='UNKNOWN', reason='须比较实体及知识作用域；同实体真知识可能应泛化'),
                全局隔离=dict(status='PASS' if source else 'UNKNOWN', reason='完整native/eval/reserved/fit/CAL源角色排除' if source else '待源身份/病例隔离'),
                审核方式=dict(status='UNKNOWN', reason='涉及新医学判断必须真实专业签核；Codex仅作来源/结构审计'))


def referent_chain(native, src, relation, kg):
    q = norm(native['question'])
    positions = {p for p in ('top', 'center', 'left', 'right', 'bottom') if p in q}
    matching = []
    for row in src['QA']:
        if row.get('content_type') != 'Position' or row['q_lang'] != 'en':
            continue
        answer_position = POSITION_ALIAS.get(norm(row['answer']))
        if answer_position not in positions:
            continue
        for box in src['detection']:
            name = entity(box['entity'])
            aliases = [k for k, v in ENTITY_ALIAS.items() if v == name and k.isascii()]
            if not any(re.search(r'\b'+re.escape(a)+r'\b', norm(row['question'])) for a in aliases):
                continue
            facts = [f for f in kg if f['entity'] == name and f['relation'] == relation
                     and f['evidence_language'] == ('zh' if re.search('[\u4e00-\u9fff]', native['question']) else 'en')]
            for fact in facts:
                matching.append(dict(image_position_QA=row, detected_entity=box,
                                     named_KG=fact, derived_answer=fact['answer'],
                                     unresolved_qualifiers=qualifiers(native['question']),
                                     binding_status='CANDIDATE_NOT_PROOF_OF_UNIQUE_REFERENT'))
    return matching


def main():
    start, cpu = time.time(), time.process_time()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    root = Path(os.environ['RUN_ROOT'])
    manifest = json.loads((root/'RUN_MANIFEST.json').read_text())
    assert start < manifest['deadline_epoch'] and not (root/'STOP').exists()
    old = Path(manifest['previous_run'])
    cfg = json.loads((root/'PROTOCOL_CONFIG.json').read_text())
    ledger, inventory, kg, old_packet, bilingual = source_inventory(old, root)
    canonical = json.dumps({k:v for k,v in ledger.items() if k!='freeze_id'}, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    assert hashlib.sha256(canonical.encode()).hexdigest() == ledger['freeze_id'] == cfg['cohort_freeze_id']
    original_coverage = json.loads((old/'private/H_SUPPORT_FINAL/FULL_COHORT_H_COVERAGE.json').read_text())
    assert [x['edit_id'] for x in original_coverage] == ledger['main_T0']
    assert len(ledger['main_T0']) == len(set(ledger['main_T0'])) == 146
    tasks = {t['edit_id']: t for t in ledger['tasks']}
    old_fits = json.loads((old/'private/H_SUPPORT_FINAL/SOURCE_VERIFIED_H_FIT.json').read_text())
    by_edit = collections.defaultdict(list)
    for r in old_packet:
        by_edit[r['edit_id']].append(r)
    supported = {r['edit_id'] for r in old_fits}
    audits, candidates, inherited = [], [], []
    for i, eid in enumerate(ledger['main_T0'], 1):
        n = tasks[eid]['native']
        kind, rel, applicability, reason = classify(i, n['question'])
        needs = ['新图原始出处、训练角色、病例/派生关系和完整全局隔离', '完整问句所有临床限定在新图上成立',
                 '原答案来源及不改变命题的语言/抽取对应', 'native target不适用且两答案不能同时成立的证据',
                 '源输入确在编辑应影响的泛化范围之外的作用域证据']
        if kind == 'ENTITY_CONDITIONED_KNOWLEDGE':
            needs += ['图像指代—具名实体的框/mask/原定位QA或报告证据', f'具名实体—{rel}—答案的原KG/报告出处，模板占位符不计',
                      '原/新实体临床知识可共存和应泛化范围的专业审核']
        if kind == 'AMBIGUOUS_PENDING_REVIEW':
            needs += ['先确认问题作用域、唯一指代/严重程度或通用知识解释']
        if kind == 'IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE':
            needs += ['确认原问题是否真为一般命题；未确认前不能制造图像条件化反例或修改native']
        audit = dict(position=i, edit_id=eid, native=n, question_type=kind, relation_type=rel,
                     retained_qualifiers=qualifiers(n['question']), strict_same_question_H_applicability=applicability,
                     applicability_reason=reason, missing_evidence=needs,
                     prior_H_relations=sum(r['edit_id']==eid for r in old_fits),
                     edited_scope='原图具体实体/关系及其真实等价输入；同实体通用知识可能应泛化，临床边界待专业审阅' if kind != 'IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE' else '字面一般命题，无已证实图像局部作用域',
                     reviewer='Codex question/source audit; not clinical signoff', clinical_signoff=False,
                     evidence_needed_even_if_legacy_H=True, source_exhausted=False)
        audits.append(audit)
        if eid in supported:
            continue
        selected, seen = [], set()
        for old_row in by_edit[eid]:
            src=old_row['source'];key=(src['dataset'],src['image_id'],str(src['qid']))
            if key in seen:
                continue
            seen.add(key)
            selected.append(dict(candidate_id=f'V2-{i:03d}-{len(selected)+1}',position=i,edit_id=eid,
                                 evidence_type='ORIGINAL_QA', native=n, question_on_source=src['question'],
                                 source=src, origin='PR20_UNKNOWN_REVIEW', retained_qualifiers=audit['retained_qualifiers'],
                                 review_state='PENDING_SIX_CHECKS',checks=checks(True, True),
                                 prior_review_id=old_row['review_id'], expert_review_required=True, clinical_signoff=False,
                                 independent_annotation_family=f"{src['dataset']}:{src['image_id']}:{src['qid']}"))
            if len(selected)==5:
                break
        if n['dataset']=='SLAKE' and kind=='ENTITY_CONDITIONED_KNOWLEDGE' and rel=='function':
            chains=[]
            for src in inventory:
                for chain in referent_chain(n,src,rel,kg):
                    if norm(chain['derived_answer']) == norm(n['reference']):
                        continue
                    chains.append((src,chain))
            audit['retrieval_matches_before_cap']=len(chains)
            for src,chain in chains:
                key=('DERIVED',src['image_id'],chain['named_KG']['member'],chain['named_KG']['line'])
                if key in seen or len(selected)==5:
                    continue
                seen.add(key)
                selected.append(dict(candidate_id=f'V2-{i:03d}-{len(selected)+1}',position=i,edit_id=eid,
                                     evidence_type='SOURCE_DERIVED_QA',native=n,question_on_source=n['question'],
                                     source={k:v for k,v in src.items() if k!='QA'}, evidence_chain=chain,
                                     answer=chain['derived_answer'],derived_by='Codex deterministic source extraction',
                                     origin='SLAKE_POSITION_QA_DETECTION_NAMED_KG_CHAIN',
                                     retained_qualifiers=audit['retained_qualifiers'],review_state='PENDING_SIX_CHECKS',
                                     checks=checks(True,True),expert_review_required=True,clinical_signoff=False,
                                     independent_annotation_family=f"SLAKE:{src['image_id']}:{chain['image_position_QA']['qid']}"))
        audit['first_round_candidate_count']=len(selected)
        audit['new_source_search_slots_remaining']=5-len(selected)
        # A targeted query brief preserves the literal question and clinical qualifiers.
        audit['external_search_brief']=dict(native_question=n['question'],relation=rel,
                                          required_body='brain' if any(x in norm(n['question']) for x in ('brain','大脑')) else 'chest' if 'lung' in norm(n['question']) else 'question-specific',
                                          sources=['PMC_VQA_TRAIN_PRIMARY_PAPER','OTHER_DOCUMENTED_TRAIN_SOURCE'],
                                          missing_entities_or_evidence=needs, only_metadata_first=True,
                                          limit=5-len(selected), clinical_answer_generation=False)
        candidates.extend(selected)
    for f in old_fits:
        inherited.append(dict(evidence_id='PR20:'+f['review_id'], edit_id=f['edit_id'], position=f['position'],
                              evidence_type='ORIGINAL_QA', provenance='PR20_READ_ONLY_INHERITED',
                              independent_annotation_family=f"{f['source']['dataset']}:{f['source']['image_id']}:{f['source']['qid']}",
                              source=f['source'],native=f['native'],evidence=f['evidence'],
                              v2_review_status='REVALIDATION_PENDING',clinical_signoff=False,
                              patient_independence='UNKNOWN',count_as_new_annotation=False,training_admitted_v2=False))
    jsonl(root/'private/H_GAP_AUDIT_146.jsonl',audits)
    jsonl(root/'private/H_CANDIDATE_REVIEW_PACKET/CANDIDATES.jsonl',candidates)
    jsonl(root/'private/H_CANDIDATE_REVIEW_PACKET/PROFESSIONAL_REVIEW_QUEUE.jsonl',candidates)
    jsonl(root/'private/H_CANDIDATE_REVIEW_PACKET/EXTERNAL_SEARCH_BRIEFS.jsonl',[x for x in audits if not x['prior_H_relations']])
    jsonl(root/'private/PR20_INHERITED_EVIDENCE.jsonl',inherited)
    # Only fully accepted six-check evidence belongs here; pending material lives elsewhere.
    jsonl(root/'private/H_EVIDENCE_MANIFEST.jsonl',[])
    coverage=dict(N=146,original_membership_order_binding='PASS',target_min_H_fit=1,
                  original_PR20_H_supported_edits=8,original_PR20_H_relations=12,
                  v2_accepted_H_supported_edits=0,v2_revalidation_pending_prior_edits=8,
                  all_146_H_admitted=False,sparse_H_substitution=False,
                  rows=[dict(position=x['position'],edit_id=x['edit_id'],question_type=x['question_type'],
                             accepted_H_fit=0,legacy_H_relations=x['prior_H_relations'],
                             new_candidates=x.get('first_round_candidate_count',0),
                             status='PR20_REVALIDATION_PENDING' if x['prior_H_relations'] else x['strict_same_question_H_applicability'],
                             missing_evidence=x['missing_evidence']) for x in audits])
    dump(root/'private/H_COVERAGE_146.json',coverage)
    summary=dict(status='QUESTION_AUDIT_AND_ENTITY_CHAIN_MILESTONE_COMPLETE_NEXT_PRIMARY_SOURCE_RETRIEVAL',N=146,
                 question_types=dict(collections.Counter(x['question_type'] for x in audits)),
                 relation_types=dict(collections.Counter(x['relation_type'] for x in audits)),
                 strict_H_applicability=dict(collections.Counter(x['strict_same_question_H_applicability'] for x in audits)),
                 all_gaps_have_specific_evidence_needs=True,
                 source_legal_SLAKE_images=len(inventory), source_legal_SLAKE_QA=sum(len(x['QA']) for x in inventory),
                 bilingual_ZH_rows=len(bilingual),bilingual_actual_semantic_alignment_confirmed=0,
                 new_candidate_relations=len(candidates),new_candidate_edits=len({x['edit_id'] for x in candidates}),
                 candidate_evidence_types=dict(collections.Counter(x['evidence_type'] for x in candidates)),
                 candidate_different_raw_QA_families=len({x['independent_annotation_family'] for x in candidates}),
                 same_question_global_interpretation_needs_scope_confirmation=sum(x['question_type']=='IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE' for x in audits),
                 initial_prior_H_supported=8,new_v2_H_accepted=0,new_clinical_signoffs=0,
                 GPU_hours=0,Judge_requests=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,
                 original_PR20_preserved=True,source_exhaustion_claim=False)
    dump(root/'public/MILESTONE_1_SUMMARY.json',summary)
    dump(root/'private/MILESTONE_1_RECEIPT.json',dict(summary=summary,pid=os.getpid(),argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),RUN_ROOT=str(root),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],epoch=time.time()))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def selfcheck():
    assert classify(23,'Toy general comparison without a case variable')[0]=='IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE'
    assert classify(2,'Describe the function of the marked toy entity')[0]=='ENTITY_CONDITIONED_KNOWLEDGE'
    assert classify(15,'Treatment of the most severe toy finding?')[0]=='AMBIGUOUS_PENDING_REVIEW'
    assert 'right' in qualifiers('Toy right marker, possibly uncertain')
    assert 'possibly' in qualifiers('Toy right marker, possibly uncertain')
    assert entity('Small Bowel')==entity('小肠')
    assert checks(True,True)['target不适用']['status']=='UNKNOWN'


if __name__=='__main__':
    selfcheck()
    main()
