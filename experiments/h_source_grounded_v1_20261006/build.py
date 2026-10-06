"""One bounded CPU pass over original QA and cached, bound primary captions.
No student outputs, GPUs, paid APIs, or changes to historical assets.
"""
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

VERSION = 'h-source-grounded-v1-20261006'
FORBIDDEN = {'PROTECTED_SPLIT','FORMAL_OR_RESERVED_SOURCE','CAL_CHECK_SUPPORT_RESERVED','FROZEN_FIT_SOURCE'}

def norm(s):
    return ' '.join(unicodedata.normalize('NFKC',str(s)).casefold().split()).rstrip('?.')

def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def load(p): return json.loads(Path(p).read_text())
def jl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def lines(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def identity(ds,path):
    return ds, Path(path).parent.name if ds=='SLAKE' else Path(path).stem

def qualifiers(q):
    # Preserve the exact question as well; this list is retrieval/audit metadata only.
    return [x for x in ['left','right','upper','lower','top','center','central','posterior','anterior','most severe','possibly','possible','bilateral','contrast','non-contrast','灰','黑','左','右','上','下','顶部','最严重'] if x in norm(q)]

def operator(q):
    s=norm(q)
    if 'modality' in s or s in {'what type of image is this','what kind of image is this','what type of imaging is this'}:return 'modality'
    if 'contrast ct' in s or 'contrast or non contrast ct' in s:return 'ct_contrast'
    if re.fullmatch(r'(?:where (?:is|are|is/are)|the lesion is located where).*',s) or s.startswith('what is the location') or s.startswith('how would you describe the location'):
        return 'location'
    if re.match('which lobe is the lesion',s):return 'location'
    if 'function' in s or 'what is the effect' in s:return 'function'
    if 'diseases are' in s or 'disease is/are' in s:return 'disease_inventory'
    if 'digestive system' in s:return 'organ_inventory'
    if 'treat' in s:return 'treatment'
    if 'prevent' in s:return 'prevention'
    if 'symptom' in s or '症状' in s:return 'symptom'
    if 'cause' in s:return 'cause'
    if any(x in s for x in ['pathology','abnormal findings','what is abnormal','what is wrong','abnormality seen','describe this lesion','what is the lesion']):return 'finding'
    return 'other'

def modality(a):
    a=norm(a);c=set()
    if re.search(r'\b(?:mri|mr|magnetic resonance)\b',a):c.add('MRI')
    if re.search(r'\b(?:ct|computed tomography|computer tomography)\b',a):c.add('CT')
    if re.search(r'\b(?:xray|x-ray|x ray|radiograph)\b',a):c.add('XR')
    if re.search(r'\b(?:ultrasound|ultrasonograph)\w*\b',a):c.add('US')
    return next(iter(c)) if len(c)==1 else None

def compatible(a,b):
    a=norm(a);b=norm(b)
    if a==b:return True
    # Only synonyms actually relevant to this finite source pool.
    alias={'mr flair':'mri flair','mr-flair':'mri flair','suprasellar cistern':'suprasellar cistern'}
    a=alias.get(a,a);b=alias.get(b,b)
    if a==b:return True
    if 'suprasellar cistern' in a and 'suprasellar cistern' in b:return True
    return False

def support_check(spans, source_text):
    return bool(spans) and all(isinstance(x,str) and x and x in source_text for x in spans)

def admit(row,role):
    if role not in {'FIT','EVAL'} or row['assigned_role']!=role: return False
    if row['evidence_level'] not in {'SOURCE_VERIFIED','SOURCE_GROUNDED'}:return False
    return row['source_role_check']=='PASS' and row['source_supports_h_answer']=='YES' and row['target_relation']=='CONTRADICTS' and row['same_question_operator'] and row['citation_check']=='PASS'

def validate_role_isolation(rows):
    groups=collections.defaultdict(set);images=collections.defaultdict(set)
    for row in rows:
        groups[row['source_group_id']].add(row['assigned_role'])
        images[row['image_sha256']].add(row['assigned_role'])
    if any(len(x)>1 for x in list(groups.values())+list(images.values())):
        raise ValueError('Same source group or image cannot cross FIT/EVAL')
    return True

def fit_load(path):
    path=Path(path)
    if 'sealed_eval' in path.parts:raise ValueError('Sealed evaluation cannot be loaded for training')
    rows=jl(path)
    if any(not admit(x,'FIT') for x in rows):raise ValueError('Unqualified/role-incompatible training record')
    return rows

def classify(a):
    q=a['native']['question']; op=operator(q)
    if a['question_type']=='IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE':
        app='NOT_APPLICABLE_WITH_REASON';reason='Literal fixed-entity general proposition; changing image alone cannot establish opposite truth. Retain native and full denominator.'
    elif a['question_type']=='AMBIGUOUS_PENDING_REVIEW':
        app='UNCERTAIN';reason='Image dependency/referent or severity interpretation is unresolved, not merely a missing source.'
    else:app='APPLICABLE';reason='Image-dependent predicate or visual referent; all original qualifiers must bind on the new source.'
    n=a['native'];return dict(edit_id=a['edit_id'],position=a['position'],original_source=n['dataset'],source_identifier=identity(n['dataset'],n['original_image_path'])[1],original_question=q,target_answer=n['reference'],question_type=a['question_type'],question_operator=op,image_dependent=app!='NOT_APPLICABLE_WITH_REASON',visual_reference=bool(re.search(r'\b(this|image|picture|lesion|mass|abnormality)\b|图|这个',q,re.I)),target_entity_or_variable='original visual referent' if app!='NOT_APPLICABLE_WITH_REASON' else 'fixed entity in literal question',attribute_relation=a['relation_type'],answer_type='SET_OR_MULTI_CLAUSE' if ',' in n['reference'] else 'TEXT',answer_polarity='OTHER',qualifiers=qualifiers(q),qualifiers_full_text=q,transfer_condition='same referent and predicate under all original qualifiers; general fixed-entity fact may transfer',nontransfer_condition='source explicitly establishes incompatible acquisition, location of the same visual referent, or explicit negation of target; absence of mention is insufficient',H_applicability=app,applicability_reason=reason,parser='Codex finite semantic taxonomy + exact question preservation; not clinical signoff')

def provenance(base,previous,old,data):
    qa=load(base/'scope-m3bench-candidate-audit-20261001/run/private/SOURCE_QA_ROLE_AUDIT.json')
    roles=load(base/'scope-m3bench-exposure-audit-20261001/run/private/IDENTITY_ROLE_LEDGER.json')
    byrole=collections.defaultdict(set)
    for r in roles:byrole[(r['dataset'],r['image_id'])].update(r['roles'])
    ledger=load(old/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    banned={tuple(x) for x in load(previous/'private/GLOBAL_SOURCE_EXCLUSION.json')['banned_images']}
    for t in ledger['tasks']:
        banned.add(identity(t['native']['dataset'],t['native']['original_image_path']))
        for u in t.get('U_fit',[]):banned.add((u['dataset'],u['image_id']))
    banned|={k for k,v in byrole.items() if v&FORBIDDEN}
    sl=load(data/'SLAKE/train.json');vr=load(data/'VQA-RAD/VQA_RAD Dataset Public.json')
    banned|={identity('VQA-RAD',x['image_name']) for x in vr if x['phrase_type'].startswith('test_')}
    cases={x.get('image_case_url') for x in vr if identity('VQA-RAD',x['image_name']) in banned}-{None,'','NULL'}
    idx={('SLAKE',str(x['qid'])):x for x in sl};idx.update({('VQA-RAD',str(x['qid'])):x for x in vr})
    eligible=[]
    for row in qa:
        key=(row['dataset'],row['image_id']);raw=idx[(row['dataset'],str(row['qid']))]
        if key in banned or not byrole[key] or not Path(row['image_path']).is_file():continue
        if row['dataset']=='VQA-RAD' and (raw['evaluation']!='evaluated' or raw['phrase_type']!='freeform' or raw.get('image_case_url') in cases):continue
        assert raw==row['raw_source'] and str(raw['answer'])==str(row['answer']) and raw['question']==row['question']
        eligible.append(dict(row,raw_source=raw))
    for raw in sl:
        key=identity('SLAKE',raw['img_name'])
        if raw['q_lang']=='zh' and key not in banned and byrole[key]:
            image=data/'SLAKE/imgs'/raw['img_name']
            if image.is_file():eligible.append(dict(dataset='SLAKE',qid=str(raw['qid']),image_id=key[1],question=raw['question'],answer=str(raw['answer']),image_path=str(image),raw_source=raw))
    return ledger,eligible,banned,cases,idx

def registry(eligible,previous,inherited,idx):
    training_images={(x['source']['dataset'],x['source']['image_id']) for x in inherited}
    # Establish source groups before matching edits. Exact duplicate image identities are grouped.
    image_hash={};groups={};sources=[]
    for r in eligible:
        key=(r['dataset'],r['image_id']);raw=r['raw_source']
        if key not in image_hash:image_hash[key]=hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()
        case=raw.get('image_case_url');group=case if case not in (None,'','NULL') else ':'.join(key)
        groups[key]=group
    for key in list(training_images):
        if key in groups:training_images|={k for k,g in groups.items() if g==groups[key]}
    fit_hashes={image_hash[k] for k in training_images if k in image_hash}
    hash_roles={}
    for r in eligible:
        key=(r['dataset'],r['image_id']);sha=image_hash[key];raw=r['raw_source'];group=groups[key]
        role='FIT' if key in training_images or sha in fit_hashes else ('EVAL' if int(digest(group)[:8],16)%3==0 else 'FIT')
        if sha in hash_roles:role=hash_roles[sha]
        hash_roles[sha]=role
        sid=f"{r['dataset']}:{r['image_id']}:{r['qid']}"
        sources.append(dict(source_id=sid,source_dataset=r['dataset'],source_version='existing author release; exact original row retained',source_record_id=str(r['qid']),source_question_id=str(r['qid']),image_id=r['image_id'],image_path=r['image_path'],image_sha256=sha,case_or_study_id=raw.get('image_case_url'),source_group_id=group,original_split='train' if r['dataset']=='SLAKE' else 'evaluated/freeform; test-image/case exclusion passed',assigned_role=role,original_question=r['question'],original_answer=str(r['answer']),evidence_origin='ORIGINAL_QA',evidence_ids=[sid],evidence_spans=[r['question'],str(r['answer'])],source_text=r['question']+'\n'+str(r['answer']),raw_source=raw,body_part=raw.get('location',raw.get('image_organ')),modality=raw.get('modality'),metadata_origin='ORIGINAL_ANNOTATION',source_role_check='PASS',license='Existing authorized local research source; no raw public redistribution or external API assumed',patient_independence='UNKNOWN',image_binding='author QA image ID/path, exists, SHA256 requested by current protocol'))
    banned_papers=set()
    for name in ['test.csv.EXCLUSION_IDS.json','test_2.csv.EXCLUSION_IDS.json']:banned_papers.update(load(previous/'private/external_metadata'/name)['papers'])
    for p in sorted((previous/'private/primary_evidence').glob('*.PRIMARY_EVIDENCE.json')):
        d=load(p);paper=d['paper']
        if paper in banned_papers or not d.get('license_is_CC_BY_or_CC0'):continue
        for im in d.get('images',[]):
            path=previous/'private/primary_evidence'/im['figure'];caption=im.get('caption','')
            if im.get('status')!='ACQUIRED_PENDING_GLOBAL_ROLE_AND_QA_REVIEW' or not caption or not path.is_file():continue
            group='PMC:'+paper;role='EVAL' if int(digest(group)[:8],16)%3==0 else 'FIT';sid=paper+':'+im['primary_figure_id']
            sources.append(dict(source_id=sid,source_dataset='PMC_PRIMARY',source_version='cached author JATS/HTML acquired 2026-10-03; original article record retained',source_record_id=im['primary_figure_id'],source_question_id=None,image_id=im['figure'],image_path=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),case_or_study_id=None,source_group_id=group,original_split='PMC-VQA train index; whole-paper test exclusion',assigned_role=role,original_question=None,original_answer=None,evidence_origin='ORIGINAL_ANNOTATION',evidence_ids=[sid+':caption'],evidence_spans=[caption],source_text=caption,body_part=None,modality=modality(caption),metadata_origin='PRIMARY_FIGURE_CAPTION',source_role_check='PASS',license=d['license'],primary_url='https://pmc.ncbi.nlm.nih.gov/articles/'+paper+'/',patient_independence='UNKNOWN',image_binding='cached primary figure ID and acquired file; no crop or panel generated',composite=bool(re.search(r'\([a-f]\)|(?:^|[. ])(?:a|b),|(?:^|\. )(?:a|b) ',caption))))
    validate_role_isolation(sources)
    return sources

def blind_facts(sources):
    result=[]
    for s in sources:
        # Literal extraction only, independent of any edit/target. No imagined negatives.
        result.append(dict(source_id=s['source_id'],facts=[dict(fact_id=s['source_id']+':literal',entity='source visual referent' if s['original_question'] else 'caption-specific referent',relation=operator(s['original_question']) if s['original_question'] else 'PRIMARY_CAPTION_STATEMENT',value=s['original_answer'] if s['original_question'] else s['source_text'],polarity='EXPLICIT_NEGATIVE_ANSWER' if norm(s['original_answer']) in {'no','none'} else 'OTHER',qualifiers=qualifiers(s['original_question'] or s['source_text']),evidence_ids=s['evidence_ids'],evidence_spans=s['evidence_spans'],evidence_origin=s['evidence_origin'],scope='IMAGE' if not s.get('composite') else 'CAPTION_WITH_PANEL_LIMITATIONS',limitations=['Patient independence UNKNOWN','No unmentioned negative facts extracted'])],insufficient_evidence=False,extraction_method='Literal CPU extraction; separate GPT API not executed'))
    return result

# Private source-specific review table is supplied as an asset, never published.
REVIEW_RULES={}

def decide(edit,s):
    if s.get('evidence_origin')=='LLM_DERIVED' or s.get('metadata_origin')=='LLM_DERIVED':
        return 'MODEL_ONLY','UNKNOWN','Derived model labels are retrieval clues, not original answer evidence.',None,None
    q=edit['original_question'];t=edit['target_answer'];op=edit['question_operator'];sq=s['original_question'] or '';a=s['original_answer'];qid=s['source_question_id'];p=edit['position']
    if a is not None and compatible(t,a):return 'REJECTED','COMPATIBLE','Candidate answer overlaps/synonymous with target, not a counterexample.',None,None
    if s['source_dataset']=='PMC_PRIMARY':
        cap=s['source_text'];m=modality(cap)
        if op=='modality' and m and modality(t) and m!=modality(t) and not s.get('composite'):
            answer={'MRI':'MRI','CT':'CT','XR':'X-ray','US':'Ultrasound'}[m]
            return 'SOURCE_VERIFIED','CONTRADICTS','Primary caption explicitly names a single acquisition class incompatible with target major class.',q,answer
        if op=='ct_contrast' and m=='CT' and re.search(r'non[ -]contrast',cap,re.I) and 'contrast' in norm(t):
            return 'SOURCE_VERIFIED','CONTRADICTS','Caption explicitly states non-contrast CT for all described panels; target GI+IV contrast is incompatible.',q,'Non-contrast CT'
        if op=='location' and not s.get('composite'):
            rule=REVIEW_RULES.get('pmc_location_rules',{}).get(s['source_id'].split(':')[0],{})
            if p in rule.get('positions',[]):
                return 'SOURCE_GROUNDED','CONTRADICTS','Caption explicitly locates its described lesion/mass. Codex aligns generic visual-location operator; target location is incompatible for that referent. New semantic alignment needs independent review.',q,rule['answer']
        return 'UNKNOWN','UNKNOWN','Primary caption is available, but full referent/panel/clinical target exclusion is not established.',None,None
    if op=='modality' and operator(sq)=='modality' and modality(a) and modality(t) and modality(a)!=modality(t):
        return 'SOURCE_VERIFIED','CONTRADICTS','Original source QA identifies incompatible mutually exclusive major acquisition class.',sq,a
    if op=='location' and qid in REVIEW_RULES.get('location_rules',{}).get(str(p),[]):
        if compatible(t,a):return 'REJECTED','COMPATIBLE','Target and source location compatible.',None,None
        return 'SOURCE_GROUNDED','CONTRADICTS','Original QA explicitly locates the source visual referent; Codex reviewed same location operator and incompatible named region. Medical/anatomical alignment remains model-assisted, no independent audit.',q,a
    return 'UNKNOWN','UNKNOWN','Literal answer exists; target exclusion, exact qualifiers or task operator still needs evidence.',None,None

def similarity(q,s):
    # ponytail: finite corpus lexical/semantic-template retrieval, not a neural embedding claim.
    a=set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]',norm(q)));b=set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]',norm(s['original_question'] or s['source_text'])))
    return len(a&b)/math.sqrt(max(1,len(a)*len(b)))

def make(edit,s,decision,origin='NEW_RETRIEVAL'):
    level,relation,reason,hq,ha=decision
    source_answer=ha if ha is not None else s['original_answer']
    aligned=bool(hq)
    spans=s['evidence_spans'];citation=support_check(spans,s['source_text'])
    return dict(candidate_id='H-'+digest([VERSION,edit['edit_id'],s['source_id']])[:20],edit_id=edit['edit_id'],position=edit['position'],source_id=s['source_id'],**{k:s[k] for k in ['source_dataset','source_version','source_record_id','source_question_id','image_id','image_sha256','case_or_study_id','source_group_id','original_split','assigned_role','original_question','original_answer','evidence_ids','evidence_spans','evidence_origin']},image_path=s['image_path'],h_question=hq,h_answer=source_answer,construction_type='EVIDENCE_TO_QA' if s['original_question'] is None and aligned else ('SEMANTIC_ALIGNMENT' if aligned and norm(hq)!=norm(s['original_question']) else 'ORIGINAL_QA' if aligned else 'NONE'),answer_support_status='YES' if source_answer is not None and aligned else 'UNKNOWN',source_supports_h_answer='YES' if source_answer is not None and aligned else 'UNKNOWN',same_question_operator=aligned,qualifier_alignment={'native_full_question_preserved':hq==edit['original_question'],'source_full_question':s['original_question'],'review':'Codex finite semantic mapping; no qualifier removal from native'},target_relation=relation,evidence_level=level,generator_verdict='SUPPORTED' if level.startswith('SOURCE_') else 'REFUTED' if level=='REJECTED' else 'UNKNOWN',auditor_verdict='NOT_RUN',final_status='SOURCE_GROUNDED_PENDING_INDEPENDENT_REVIEW' if level=='SOURCE_GROUNDED' else 'ACCEPTED_RULE_VERIFIED' if level=='SOURCE_VERIFIED' else level,verification_method='ORIGINAL_SOURCE_BINDING_AND_EXPLICIT_ACQUISITION_LOGIC' if level=='SOURCE_VERIFIED' else 'CURRENT_CODEX_TEXT_SEMANTIC_REVIEW' if level=='SOURCE_GROUNDED' else 'CPU_SOURCE_RETRIEVAL_ONLY',hardness_category='EASY_MODALITY' if level=='SOURCE_VERIFIED' else 'UNKNOWN',exclusion_reason=reason if level in {'REJECTED','UNKNOWN'} else None,decisive_reason=reason,generator_model='Current Codex session; exact serving model snapshot unavailable' if level=='SOURCE_GROUNDED' else None,auditor_model=None,prompt_version=VERSION,input_hash=digest([edit,s['source_id'],s['source_text'],VERSION]),created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_role_check=s['source_role_check'],citation_check='PASS' if citation else 'FAIL',patient_independence='UNKNOWN',clinical_verified=False,external_api_executed=False,origin=origin,training_auto_admitted=False,formal_evaluation_auto_admitted=False)

def main():
    start=time.time();cpu=time.process_time();root=Path(os.environ['RUN_ROOT']);cfg=load(os.environ['BUILD_CONFIG'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not cfg['external_api_enabled']
    assert not (root/'private/candidate_manifest.jsonl').exists(),'Frozen outputs cannot be overwritten'
    global REVIEW_RULES
    assets=load(os.environ['ASSET_CONFIG'])
    REVIEW_RULES=load(assets['semantic_review_path'])
    base=Path(assets['outputs_root']);previous=Path(assets['previous_evidence_root']);old=Path(assets['original_comparison_root']);data=Path(assets['original_data_root'])
    ledger,pool,banned,cases,idx=provenance(base,previous,old,data)
    audits=jl(previous/'private/H_GAP_AUDIT_146.jsonl');assert [x['edit_id'] for x in audits]==ledger['main_T0'] and len(audits)==146
    canonical={k:v for k,v in ledger.items() if k!='freeze_id'}
    canonical_bytes=json.dumps(canonical,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    assert hashlib.sha256(canonical_bytes).hexdigest()==ledger['freeze_id']
    inherited=load(old/'private/H_SUPPORT_FINAL/SOURCE_VERIFIED_H_FIT.json');unknown=load(old/'private/H_SUPPORT_FINAL/UNKNOWN_REVIEW_QUEUE.json')
    assert len(inherited)==12 and len({x['edit_id'] for x in inherited})==8 and len(unknown)==22
    sources=registry(pool,previous,inherited,idx);sm={s['source_id']:s for s in sources};inventory=[classify(a) for a in audits];em={x['edit_id']:x for x in inventory}
    lines(root/'private/source_registry.jsonl',sources);lines(root/'private/blind_evidence_facts.jsonl',blind_facts(sources));lines(root/'private/edit_inventory.jsonl',inventory)
    save(root/'private/exclusion_audit.json',dict(banned_source_images=len(banned),known_case_exclusions=len(cases),all_future_594_natives_excluded=True,U_training_groups_excluded=True,source_pool_rows=len(pool),source_pool_images=len({(x['dataset'],x['image_id']) for x in pool}),cached_primary_source_figures=len(sources)-len(pool),raw_original_binding='PASS',cohort_freeze_id=ledger['freeze_id']))
    candidates=[];historical=[]
    for x in inherited:
        s=sm.get(f"{x['source']['dataset']}:{x['source']['image_id']}:{x['source']['qid']}")
        assert s and s['assigned_role']=='FIT'
        row=make(em[x['edit_id']],s,('SOURCE_VERIFIED','CONTRADICTS',x['evidence'].get('validation',json.dumps(x['evidence'],ensure_ascii=False)),x['source']['question'],str(x['source']['answer'])),'HISTORICAL_VERIFIED_INHERITED')
        row.update(verification_method=x['evidence']['kind'],hardness_category='EASY_MODALITY' if 'MODALITY' in x['evidence']['kind'] else 'EASY_CROSS_FRAME',historical_review_id=x['review_id'],historical_evidence=x['evidence'],new_annotation=False)
        candidates.append(row)
    for x in unknown:
        sid=f"{x['source']['dataset']}:{x['source']['image_id']}:{x['source']['qid']}";s=sm.get(sid)
        if not s:
            historical.append(dict(review_id=x['review_id'],edit_id=x['edit_id'],result='REJECTED',reason='SPLIT_OR_PROVENANCE_BLOCKED'));continue
        row=make(em[x['edit_id']],s,decide(em[x['edit_id']],s),'HISTORICAL_UNKNOWN_REVIEW');row['historical_review_id']=x['review_id'];candidates.append(row)
        historical.append(dict(review_id=x['review_id'],edit_id=x['edit_id'],result=row['evidence_level'],reason=row['decisive_reason'],new_real_evidence=False))
    pilot_ids={x['edit_id'] for x in inventory if x['position'] in cfg['pilot_positions']}
    ordered=sorted(inventory,key=lambda x:(x['edit_id'] not in pilot_ids,x['position']))
    pilot_gate=False
    for edit in ordered:
        if edit['edit_id'] not in pilot_ids and not pilot_gate:
            pilot=[c for c in candidates if c['edit_id'] in pilot_ids]
            assert all(c['citation_check']=='PASS' and c['assigned_role'] in {'FIT','EVAL'} and c['auditor_verdict']=='NOT_RUN' for c in pilot)
            save(root/'private/PILOT_GATE.json',dict(status='CPU_SCHEMA_AND_SOURCE_BINDING_PASS',positions=cfg['pilot_positions'],records=len(pilot),independent_API_review='NOT_EXECUTED',medical_validity_not_implied=True))
            pilot_gate=True
        existing={r['source_id'] for r in candidates if r['edit_id']==edit['edit_id']}
        room=cfg['max_candidates_per_edit']-len(existing)
        if edit['H_applicability']=='NOT_APPLICABLE_WITH_REASON':continue
        retrieved=[]
        for s in sources:
            if s['source_id'] in existing:continue
            score=similarity(edit['original_question'],s)
            sop=operator(s['original_question'] or '')
            # Author metadata is retrieval-only. The source QA/caption is the evidence.
            decision=decide(edit,s)
            if decision[0].startswith('SOURCE_'):priority=0
            elif sop==edit['question_operator'] and sop!='other':priority=1
            elif score>.20:priority=2
            else:continue
            retrieved.append((priority,-score,s['source_id'],s,decision))
        for _,_,_,s,decision in sorted(retrieved,key=lambda x:x[:3])[:max(0,room)]:candidates.append(make(edit,s,decision))
    # Count original/translated QA as one image source group, never as independent facts.
    seen=set();unique=[]
    for c in candidates:
        key=(c['edit_id'],c['source_id'])
        if key not in seen:unique.append(c);seen.add(key)
    candidates=unique
    assert max(collections.Counter(c['edit_id'] for c in candidates).values())<=cfg['max_candidates_per_edit']
    # Seal EVAL by source group before candidate construction; one group cannot cross roles.
    group_roles=collections.defaultdict(set)
    for s in sources:group_roles[s['source_group_id']].add(s['assigned_role'])
    assert all(len(v)==1 for v in group_roles.values())
    for row in candidates:
        if row['citation_check']!='PASS':row.update(evidence_level='REJECTED',final_status='REJECTED',exclusion_reason='EVIDENCE_SPAN_NOT_BOUND')
    fitv=[c for c in candidates if admit(c,'FIT') and c['evidence_level']=='SOURCE_VERIFIED']
    fitg=[c for c in candidates if admit(c,'FIT') and c['evidence_level']=='SOURCE_GROUNDED']
    ev=[c for c in candidates if admit(c,'EVAL')]
    lines(root/'private/candidate_manifest.jsonl',candidates);lines(root/'private/h_fit_verified.jsonl',fitv);lines(root/'private/h_fit_grounded.jsonl',fitg)
    lines(root/'sealed_eval/h_eval_sealed.jsonl',ev)
    os.chmod(root/'sealed_eval',0o700)
    lines(root/'private/unknown_and_review_queue.jsonl',[c for c in candidates if c['evidence_level']=='UNKNOWN' or c['evidence_level']=='SOURCE_GROUNDED'])
    lines(root/'private/historical_unknown_outcomes.jsonl',historical)
    lines(root/'private/api_review_queue.jsonl',[dict(candidate_id=c['candidate_id'],stage='BLIND_SOURCE_THEN_CONSTRUCTION_THEN_INDEPENDENT_AUDIT',status='NOT_SUBMITTED',source_id=c['source_id'],outbound_policy='UNCONFIRMED_NO_SEND',model=None) for c in candidates if c['evidence_level']!='REJECTED'])
    api=dict(status='NOT_EXECUTED_INTERFACE_PRICING_AND_OUTBOUND_POLICY_UNVERIFIED',new_requests=0,input_tokens=0,output_tokens=0,actual_cost_usd=0,estimated_cost_usd=0,unit_prices=None,cache_hits=0,failures=[],models=[],current_session_codex_source_review=True,independent_auditor_executed=False)
    save(root/'private/API_LEDGER.json',api)
    save(root/'private/PILOT_RECEIPT.json',dict(edit_positions=cfg['pilot_positions'],edits=12,records=sum(c['edit_id'] in pilot_ids for c in candidates),bound_citations=all(c['citation_check']=='PASS' for c in candidates if c['edit_id'] in pilot_ids),role_schema_checks='PASS',clinical_truth_validation=False,external_GPT_pilot='NOT_EXECUTED',expanded_CPU_retrieval=True))
    rows=[]
    for e in inventory:
        cc=[c for c in candidates if c['edit_id']==e['edit_id']];v=[c for c in cc if admit(c,'FIT') and c['evidence_level']=='SOURCE_VERIFIED'];g=[c for c in cc if admit(c,'FIT') and c['evidence_level']=='SOURCE_GROUNDED'];ee=[c for c in cc if admit(c,'EVAL')]
        gaps=[]
        if not v and not g:gaps.append('TARGET_NOT_EXCLUDED' if cc else 'NO_SEMANTIC_MATCH')
        if e['H_applicability']=='UNCERTAIN':gaps.append('H_APPLICABILITY_UNCERTAIN')
        if e['question_operator'] in {'treatment','prevention','symptom','cause','function','disease_inventory','organ_inventory'} and not v and not g:gaps.append('NEEDS_SPECIALIST_REVIEW' if e['question_operator'] in {'treatment','prevention','symptom','cause'} else 'NO_EXPLICIT_TARGET_EXCLUSION_EVIDENCE')
        if not ee:gaps.append('NO_INDEPENDENT_H_EVAL')
        rows.append(dict(position=e['position'],edit_id=e['edit_id'],H_applicability=e['H_applicability'],fit_verified_relations=len(v),fit_grounded_relations=len(g),fit_source_groups=len({c['source_group_id'] for c in v+g}),eval_verified_relations=sum(c['evidence_level']=='SOURCE_VERIFIED' for c in ee),eval_grounded_relations=sum(c['evidence_level']=='SOURCE_GROUNDED' for c in ee),eval_source_groups=len({c['source_group_id'] for c in ee}),candidate_count=len(cc),gap_codes=';'.join(gaps),next_evidence_needed=e['nontransfer_condition']+'; original referent/qualifiers must bind; independent source if EVAL missing'))
    with (root/'private/h_coverage_by_edit.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def coverage(xs):return len({x['edit_id'] for x in xs})
    def counts(xs):return dict(edits=coverage(xs),relations=len(xs),original_QA=len({x['source_id'] for x in xs if x['evidence_origin']=='ORIGINAL_QA'}),images=len({x['image_sha256'] for x in xs}),source_groups=len({x['source_group_id'] for x in xs}))
    new=[c for c in fitv+fitg if c['origin']!='HISTORICAL_VERIFIED_INHERITED'];verified_new=[c for c in new if c['evidence_level']=='SOURCE_VERIFIED']
    summary=dict(protocol=VERSION,N=146,baseline_verified=counts([c for c in candidates if c['origin']=='HISTORICAL_VERIFIED_INHERITED']),fit_verified=counts(fitv),fit_grounded=counts(fitg),fit_union=counts(fitv+fitg),fit_new=counts(new),new_verified_relations=len(verified_new),new_verified_edits_beyond_baseline=len({c['edit_id'] for c in fitv}-{x['edit_id'] for x in inherited}),new_fit_union_edits_beyond_baseline=len({c['edit_id'] for c in fitv+fitg}-{x['edit_id'] for x in inherited}),eval_verified=counts([c for c in ev if c['evidence_level']=='SOURCE_VERIFIED']),eval_grounded=counts([c for c in ev if c['evidence_level']=='SOURCE_GROUNDED']),all_candidate_status_counts=dict(collections.Counter(c['evidence_level'] for c in candidates)),all_candidates=len(candidates),historical_UNKNOWN_outcomes=dict(collections.Counter(x['result'] for x in historical)),historical_UNKNOWN_new_real_evidence=0,new_generated_QA_relations=sum(c['construction_type']=='EVIDENCE_TO_QA' for c in new),new_original_QA_pairings=sum(c['construction_type']!='EVIDENCE_TO_QA' for c in new),near_boundary_confirmed=0,near_boundary_note='Anatomical location sources prioritized; modality/region or referent evidence insufficient for a blanket same-part same-modality claim. Detailed hardness review pending.',patient_independence='UNKNOWN',API=api,source_pool_original_QA=len(pool),source_pool_images=len({(s['dataset'],s['image_id']) for s in pool}),source_registry_entries=len(sources),cached_primary_figures=len(sources)-len(pool),CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,GPU_hours=0,student_training=0,student_generation=0,performance_claim=False,independent_audit_missing=True,cohort_and_order_preserved=True)
    save(root/'public/SUMMARY.json',summary)
    save(root/'RUN_RECEIPT.json',dict(status='CPU_DATA_BUILD_COMPLETE_API_NOT_EXECUTED',argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),RUN_ROOT=str(root),CPU_seconds=summary['CPU_seconds'],wall_seconds=summary['wall_seconds'],outputs_frozen=True))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
