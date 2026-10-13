"""Read-only P0 source qualification, explicit hashed inventory and historical joins."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

RUN = Path(os.environ['RUN_ROOT'])
OLD = Path('/data/bmw/Knowledge_editing/outputs/medtrace-full-method-comparison-v2-20261003/run')
sys.path.insert(0, str(OLD/'private/source'))
from scripts.medtrace.stage17_prepare import digest, PROMPT, PROTOCOL


def read(p): return json.loads(Path(p).read_text())


def write(p, d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def relocate(p):
    p=str(p)
    if '/derived_inputs/' in p: return str(OLD/'private/derived_inputs'/p.split('/derived_inputs/',1)[1])
    if '/DataP/' in p: return '/data/bmw/DataP/'+p.split('/DataP/',1)[1]
    return p


def output(d):
    if 'raw_answer' in d: return dict(raw_answer=d['raw_answer'],raw_token_ids=d['raw_token_ids'])
    return dict(raw_answer=d['model_answer_raw'],raw_token_ids=d['raw_generated_token_ids'])


def payload(q,b,o):
    return dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],
        image_sha256=q['image_sha256'],image_path=b['image_path'],prompt_ids=b['prompt_ids'],
        attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=output(o),
        judge=dict(model='gpt-6-astra',reasoning_effort='high',protocol=PROTOCOL,prompt=PROMPT))


def main():
    began=time.time();plan=read(RUN/'PLAN_CONFIG.json');l=read(OLD/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json')
    assert digest({k:v for k,v in l.items() if k!='freeze_id'})==l['freeze_id']==plan['cohort_freeze']
    by={t['edit_id']:t for t in l['tasks']};tasks=[by[e] for e in l['main_T0']];assert len(tasks)==146
    H=read(OLD/'private/H_AVAILABLE.json');assert sorted(by[e]['order'] for e in H)==plan['H8_orders']
    assert sum(map(len,H.values()))==12
    banned=set(map(tuple,read('/data/bmw/Knowledge_editing/outputs/h-completion-v2-20261003/run/private/GLOBAL_SOURCE_EXCLUSION.json')['banned_images']))
    source_path=next(iter(H.values()))[0]['evidence']['source']['source_file'];source=read(source_path)
    originals={str(r['qid']):r for r in source}
    protected_cases={r['image_case_url'] for r in source if ('VQA-RAD',Path(r['image_name']).stem) in banned}
    semantics={31:'lesion location without additional qualifiers',35:'mass location without additional qualifiers',49:'image acquisition modality',71:'lesion location without additional qualifiers',87:'mass location without additional qualifiers',125:'image acquisition modality',126:'abnormality location without additional qualifiers',146:'image acquisition modality'}
    review=[];sources=Counter();cases=Counter();fail=[]
    for eid,rows in H.items():
        t=by[eid]
        for h in rows:
            e=h['evidence'];s=e['source'];raw=originals[str(s['qid'])]
            checks=dict(original_row=raw==s['original_source_row'],answer_unchanged=h['reference']==raw['answer'],native_question=h['question']==t['native']['question'],
                original_question=s['question']==raw['question'],image_identity=Path(h['image_path']).stem==s['image_id'],
                original_admission=e['admitted_H_fit'] and e['source_binding']=='PASS',
                global_image_exclusion=(s['dataset'],str(s['image_id'])) not in banned,
                global_case_exclusion=raw['image_case_url'] not in protected_cases,
                source_image_exists=Path(h['image_path']).is_file(),
                semantic_review=t['order'] in semantics)
            evidence=e['evidence']
            if evidence['kind']=='SOURCE_VERIFIED_BENCHMARK_MODALITY_CONTRAST':
                checks['exclusive_acquisition_class']=evidence['native_class']!=evidence['source_class'] and {evidence['native_class'],evidence['source_class']}=={'MRI','CT'}
                reason='Same unqualified image-modality proposition; original acquisition classes differ. Keep original sequence/contrast details in both answers.'
            else:
                checks['bound_frame_evidence']=evidence['source_pixel_frame_checked'] and evidence['source_answer_in_source_frame'] and evidence['source_evidence']['original_QA']==raw
                reason='Original source question refers to the same singular lesion/mass/abnormality location. Native question adds no anatomy, side, negation or time qualifier; original answer/side is preserved. Original frame evidence excludes the native target, without new clinical diagnosis.'
            status='PASS_SOURCE_BOUND_NOT_CLINICAL_SIGNOFF' if all(checks.values()) else 'BLOCKED'
            review.append(dict(edit_id=eid,order=t['order'],review_id=e['review_id'],checks=checks,status=status,reason=reason,proposition=semantics[t['order']],evidence=e,patient_independence='UNKNOWN'))
            sources[h['source_group']]+=1;cases[raw['image_case_url']]+=1
            if status=='BLOCKED':fail.append(e['review_id'])
    write(RUN/'private/H_SEMANTIC_BINDING_AUDIT.json',dict(rows=review,blocked=fail,unchanged_H8=True,H_eval='NA'))
    write(RUN/'public/H_SOURCE_SUMMARY.json',dict(H_edits=8,H_relations=12,distinct_original_QA=len(sources),distinct_images=len(sources),distinct_original_case_groups=len(cases),relations_per_source=sorted(sources.values()),patient_independence='UNKNOWN',clinical_signoff=False,H_eval='NA',blocked_count=len(fail)))
    assert not fail, 'H source/semantic qualification failed; original H8 cannot silently shrink'
    inv=[]
    def add(p,role):
        p=Path(p);inv.append(dict(path=str(p),role=role,status='AVAILABLE' if p.is_file() else 'UNAVAILABLE',bytes=p.stat().st_size if p.is_file() else None,sha256=sha(p) if p.is_file() else None))
    for p in (OLD/'private/legacy_stage17').iterdir():add(p,'original_input_role_Base_mask_or_Judge_evidence')
    add(OLD/'private/H_AVAILABLE.json','frozen_H8');add(source_path,'original_source_QA');add(RUN/'PLAN_CONFIG.json','new_frozen_plan')
    paths={relocate(q['image_path']) for q in l['queries'].values()}
    paths.update(relocate(u['image_path']) for t in tasks for u in t['U_fit'])
    paths.update(h['image_path'] for rs in H.values() for h in rs)
    for p in sorted(paths):add(p,'bound_input_image')
    for p in sorted((OLD/'private/edits').glob('e*/**/*.json')):add(p,'PR22_single_raw_route_or_training_receipt')
    for p in sorted((OLD/'private/sequential').glob('e*/**/*.json')):add(p,'PR22_sequential_raw_route_or_receipt')
    for p in sorted((RUN/'private/history').rglob('*.json*')):add(p,'historical_C_NO_H_raw_or_score')
    for p in sorted((OLD/'private/judge_common/evidence').glob('*.json')):add(p,'PR22_actual_Judge_batch_evidence')
    for p in sorted((OLD/'private/source').rglob('*.py')):add(p,'unchanged_legacy_execution_source')
    cleanup=read(OLD/'private/tools/CHECKPOINT_OWNED_INVENTORY.json')
    write(RUN/'private/PR22_DELETED_WEIGHT_INVENTORY.json',cleanup)
    for t in tasks:
        for name in ['AVAILABLE_H/latest.pt','ROUTER.pt','initial/W0_COMPLETE.pt']:
            add(OLD/'private/edits'/f"e{t['order']:03d}"/name,'prior_weight_reuse_candidate')
    with (RUN/'private/ARTIFACT_INVENTORY.jsonl').open('x') as f:
        for x in inv:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    write(RUN/'public/ARTIFACT_INVENTORY.json',dict(items=len(inv),available=sum(x['status']=='AVAILABLE' for x in inv),unavailable=sum(x['status']=='UNAVAILABLE' for x in inv),weights_require_reconstruction=True,private_hashed_manifest=True,old_artifacts_read_only=True))
    write(RUN/'private/CPU_ADMISSION.json',dict(status='PASS',N=146,H_covered=8,H_relations=12,source_checks=len(review),clock=read(RUN/'RUN_MANIFEST.json')['starting_epoch'],seconds=time.time()-began,legacy_source_read_only=True))
    print('P0_SOURCE_INVENTORY_PASS',len(inv),flush=True)


if __name__=='__main__':main()
