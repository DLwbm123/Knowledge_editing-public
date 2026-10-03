"""Metadata-only proposal retrieval; PMC-VQA synthetic answers are not H labels."""
import collections
import csv
import json
import os
import re
import time
from pathlib import Path


def jl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def body(q):
    q=q.lower()
    if any(x in q for x in ('brain','cranial','csf','cerebr','大脑','脑')):return 'brain'
    if any(x in q for x in ('lung','chest','cardiac','heart','pulmonary','radiograph','胸','肺','心脏')):return 'chest'
    if any(x in q for x in ('liver','bile','cystic','omental','appendic','肝','kidney','bowel','organ','rectum','腹','器官')):return 'abdomen'
    return 'unspecified'


TOKENS={'brain':('brain','cerebr','intracranial','thalam','frontal','occipital','mri','hemorrhage','stroke'),
        'chest':('lung','pulmonary','chest','thorac','cardiac','pneumonia','pleural','cardiomegaly'),
        'abdomen':('abdomen','abdominal','liver','kidney','bowel','pancrea','stomach','colon','rectum')}


def main():
    start,cpu=time.time(),time.process_time();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    root=Path(os.environ['RUN_ROOT']);meta=root/'private/external_metadata'
    audits=jl(root/'private/H_GAP_AUDIT_146.jsonl');old=jl(root/'private/H_CANDIDATE_REVIEW_PACKET/CANDIDATES.jsonl')
    banned_papers=set();banned_images=set()
    for n in ('test.csv.EXCLUSION_IDS.json','test_2.csv.EXCLUSION_IDS.json'):
        d=json.loads((meta/n).read_text());assert d['use']=='EXCLUSION_ONLY_NO_QA';banned_papers.update(d['papers']);banned_images.update(d['images'])
    index=collections.defaultdict(list);nrows=0;excluded=0
    with (meta/'train.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            nrows+=1;p=re.search(r'PMC\d+',row['Figure_path'])
            if not p or p.group() in banned_papers or row['Figure_path'] in banned_images:
                excluded+=1;continue
            text=row['Question'].lower()
            # Keep the author-generated answer only as an explicitly untrusted retrieval clue.
            for b,terms in TOKENS.items():
                s=sum(t in text for t in terms)
                if s:index[b].append(dict(row=row,paper=p.group(),body_score=s))
    prior_counts=collections.Counter(c['edit_id'] for c in old);global_papers=[];selected=[];search=[]
    for a in sorted(audits,key=lambda x:(x['native']['dataset']!='SLAKE',x['position'])):
        if a['prior_H_relations'] or a['question_type']=='IMAGE_INDEPENDENT_GENERAL_KNOWLEDGE':continue
        b=body(a['native']['question']);pool=index.get(b,[])
        if b=='unspecified':pool=[r for v in index.values() for r in v]
        q=a['native']['question'].lower();terms=set(re.findall(r'[a-z]{4,}',q))
        terms-={'what','which','this','image','picture','disease','diseases','there','shown','included','located','patient'}
        scored=[]
        for item in pool:
            text=item['row']['Question'].lower();score=item['body_score']+sum(t in text for t in terms)
            if a['relation_type'] in {'symptom','cause','prevention','treatment'}:
                score+=3*any(t in text for t in ('diagnos','patholog','lesion','hemorrhage','stroke','pneumonia','disease'))
            scored.append((-score,item['paper'],item['row']['Figure_path'],item['row']['Question'],item))
        scored.sort(key=lambda x:x[:4]);seen=set();chosen=[];cap=5-prior_counts[a['edit_id']]
        for *_,item in scored:
            p=item['paper']
            if p in seen:continue
            if p not in global_papers:
                if len(global_papers)>=40:continue
                global_papers.append(p)
            seen.add(p);row=item['row'];chosen.append(dict(candidate_id=f"V2P-{a['position']:03d}-{len(chosen)+1}",
                    position=a['position'],edit_id=a['edit_id'],native=a['native'],question_on_source=a['native']['question'],
                    proposed_evidence_type='SOURCE_DERIVED_QA',evidence_type_admitted=False,
                    source_metadata=dict(dataset='PMC-VQA',split='train',paper=p,figure_path=row['Figure_path'],
                                         author_generated_question=row['Question'],author_generated_answer_UNTRUSTED=row['Answer']),
                    primary_paper_URL=f'https://pmc.ncbi.nlm.nih.gov/articles/{p}/',
                    metadata_paper_test_exclusion='PASS',
                    image_case_patient_and_project_global_isolation='UNKNOWN_PENDING_PRIMARY_BINDING',
                    review_state='PENDING_PRIMARY_PAPER_IMAGE_AND_SIX_CHECKS',
                    missing_evidence=a['missing_evidence'],expert_review_required=True,clinical_signoff=False,
                    image_downloaded=False,answer_available_for_H_training=False,
                    source_exhausted=False,selection='question/body/relationship retrieval; no model performance'))
            if len(chosen)>=cap:break
        if cap==0:chosen=[]
        selected.extend(chosen);search.append(dict(position=a['position'],edit_id=a['edit_id'],body=b,
                metadata_matches_before_cap=len(scored),candidates_selected=len(chosen),
                prior_candidates_count=prior_counts[a['edit_id']],first_round_total=prior_counts[a['edit_id']]+len(chosen),
                source_exhausted=False,paper_source_budget_limited=len(global_papers)>=40))
    assert all(x['first_round_total']<=5 for x in search)
    out=root/'private/H_CANDIDATE_REVIEW_PACKET'
    with (out/'PMC_PRIMARY_CANDIDATES.jsonl').open('x') as f:
        for x in selected:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    write(out/'PMC_SEARCH_COVERAGE.json',search)
    write(root/'private/PRIMARY_PAPER_REQUESTS.json',dict(papers=global_papers,stage='METADATA_ONLY',
                access_policy='Primary XML/license first, then only relevant figure/subpanel',
                downloaded_image_archives=False))
    summary=dict(status='PMC_TRAIN_METADATA_SEARCH_COMPLETE_PRIMARY_EVIDENCE_PENDING',N=146,
            indexed_train_rows=nrows,rows_excluded_by_test_paper_identity=excluded,
            test_paper_identities_excluded=len(banned_papers),index_body_counts={b:len(v) for b,v in index.items()},
            proposed_primary_paper_groups=len(global_papers),primary_candidate_relations=len(selected),
            primary_candidate_edits=len({x['edit_id'] for x in selected}),
            SLAKE_primary_candidate_edits=len({x['edit_id'] for x in selected if x['native']['dataset']=='SLAKE'}),
            candidate_limit_5_per_initial_missing_edit='PASS',H_truth_auto_assigned=0,
            CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,GPU_hours=0,Judge_requests=0,
            source_exhaustion_claim=False)
    write(root/'public/MILESTONE_2_SUMMARY.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
