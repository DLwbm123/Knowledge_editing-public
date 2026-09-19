"""Bounded metadata-only inventory of the already-authorized 1154-QA pool."""
import json,sys
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage18_support import conflict
from scripts.medtrace.prepare_stage2_sources import reviewed_attribute,normalized
from scripts.medtrace.stage17_prepare import digest


def read(p):return json.loads(Path(p).read_text())

def source_rows(value):
    if isinstance(value,dict):
        if 'source_group' in value and 'image_sha256' in value and 'question' in value:yield value
        else:
            for v in value.values():yield from source_rows(v)
    elif isinstance(value,list):
        for v in value:yield from source_rows(v)


def inventory():
    downloads=Path.home()/'Downloads';old=downloads/'MedTRACE_Stage20_20260918/private';cfg=read(old/'PREPARE_CONFIG.json');data=Path(cfg['source18']);src=Path(cfg['source19']);raw=read(data/'CURRENT_DATA_ALL_TRAIN_V2.json');ledger=read(src/'DEV_SOURCE_ROLE_LEDGER.json');stream=read(old/'STREAM.json');prior=read(old/'SOURCE_ROLE_LEDGER.json')
    release=read(data/'SOURCE_OVERLAY.json')['rows'][0]['source_group'].split(':')[1]
    pool=[dict(dataset='SLAKE',source_group='SLAKE:'+release+':'+r['img_name'].split('/')[0],image_path=cfg['image_root']+'/'+r['img_name'],image_sha256=raw['image_hashes'][r['img_name'].split('/')[0]],source_qid=r['qid'],question=r['question'],reference=r['answer']) for r in raw['rows']]
    assert len(pool)==1154 and len({r['source_group'] for r in pool})==113
    training=[r for t in stream['tasks'] for r in [t['native']]+t['H_fit']+t['G_fit']+t['U_fit']]
    for f in [data/'TRAINING_TASKS.json',Path(cfg['old_fasttrack'])/'private/STREAM.json',Path(cfg['old_pure'])/'private/STREAM.json']:
        for t in read(f)['tasks']:training.extend([t['native']]+t['H_fit']+t['G_fit']+t['U_fit'])
    trained={r['source_group'] for r in training};train_images={r['image_sha256'] for r in training}
    protected=set(prior['protected_groups'])|set(ledger['evaluation_sources']);protected_images={r['image_sha256'] for r in pool if r['source_group'] in protected}|{r['image_sha256'] for r in ledger['evaluation_pool']};quarantine=set(prior['quarantined_groups'])|set(ledger['quarantined_groups'])
    exposure_files=[old/'FRESH_BASE_OUTPUTS.json',data/'BASE_OUTPUTS.json',data/'QUALIFICATION_BASE_OUTPUTS.json',src/'BASE_JUDGE_INPUTS.json',src/'EXTENSION_BASE_PACKET.json',Path(cfg['old_fasttrack'])/'private/FRESH_BASE_OUTPUTS.json',Path(cfg['old_pure'])/'private/FRESH_BASE_OUTPUTS.json']
    exposed=[]
    for f in exposure_files:exposed.extend(source_rows(read(f)))
    exposed_groups={r['source_group'] for r in exposed};exposed_images={r['image_sha256'] for r in exposed}
    references={}
    for f in [src/'V3_TRAIN_REFERENCE_REVIEW.json',src/'V3_EVAL_REFERENCE_REVIEW.json',src/'BLIND_REFERENCE_REVIEW.json',Path(cfg['old_fasttrack'])/'private/REFERENCE_REUSED.json',Path(cfg['old_fasttrack'])/'private/REFERENCE_REVIEW_RESULT.json']:
        for r in read(f)['records']:
            k=(r['image_sha256'],normalized(r['question']),r['reference']);v=references.setdefault(k,r['verdict']);assert v==r['verdict']
    for r in prior['reference_reused']:
        x=r['row'];references[(x['image_sha256'],normalized(x['question']),x['reference'])]=r['review']['verdict']
    for f in (old/'reference_review').glob('ref*/operator/RESULT.json'):
        decisions={r['id']:r['verdict'] for r in read(f)['records']}
        for r in prior['reference_novel']:
            if digest(r) in decisions:references[(r['image_sha256'],normalized(r['question']),r['reference'])]=decisions[digest(r)]
    flow=[];confirm=[];expanded=[]
    for r in pool:
        reasons=[]
        if r['source_group'] in trained or r['image_sha256'] in train_images:reasons.append('HISTORICAL_TRAINING_SOURCE')
        if r['source_group'] in protected or r['image_sha256'] in protected_images:reasons.append('HISTORICAL_EVALUATION_RESERVED_SOURCE')
        if r['source_group'] in exposed_groups or r['image_sha256'] in exposed_images:reasons.append('HISTORICAL_BASE_GENERATION_EXPOSURE')
        if r['source_group'].split(':')[-1] in quarantine:reasons.append('QUARANTINED_NEAR_DUPLICATE')
        verdict=references.get((r['image_sha256'],normalized(r['question']),r['reference']))
        rr=dict(r,reference_review=verdict)
        flow.append(dict(source_qid=r['source_qid'],source_group=r['source_group'],CONFIRM_exclusions=reasons))
        if not reasons:confirm.append(rr)
        if r['source_group'] not in protected and r['image_sha256'] not in protected_images and r['source_group'].split(':')[-1] not in quarantine:expanded.append(rr)
    # Candidate-source priority is frozen before any prospective training support assignment.
    cs={r['source_group'] for r in confirm};ci={r['image_sha256'] for r in confirm};expanded=[r for r in expanded if r['source_group'] not in cs and r['image_sha256'] not in ci]
    byedit=[]
    for t in stream['tasks'][:19]:
        hs=[r for r in expanded if conflict(t['native'],r)]
        byedit.append(dict(position=t['order'],candidate_QA=len(hs),candidate_sources=len({r['source_group'] for r in hs}),previously_supported_QA=sum(r['reference_review']=='SUPPORTED' for r in hs),previously_supported_sources=len({r['source_group'] for r in hs if r['reference_review']=='SUPPORTED'})))
    hs=[r for r in confirm if any(conflict(t['native'],r) for t in stream['tasks'])]
    us=[r for r in confirm if reviewed_attribute(r['question']) in ('modality','plane') and all(reviewed_attribute(r['question'])!=reviewed_attribute(t['native']['question']) and normalized(r['question'])!=normalized(t['native']['question']) for t in stream['tasks'])]
    public=dict(scanned_QA=1154,scanned_sources=113,existing_authorized_pool_only=True,new_downloads=0,student_scores_used=False,historical_training_sources=len(trained),historical_Base_exposed_sources=len(exposed_groups),protected_eval_reserved_sources=len(protected),CONFIRM_candidate_QA=len(confirm),CONFIRM_candidate_sources=len(cs),CONFIRM_H_structural_QA=len(hs),CONFIRM_H_structural_sources=len({r['source_group'] for r in hs}),CONFIRM_U_strict_structural_QA=len(us),CONFIRM_U_strict_structural_sources=len({r['source_group'] for r in us}),exclusion_counts_nonexclusive=dict(Counter(k for r in flow for k in r['CONFIRM_exclusions'])),train_H_candidates=byedit,qualification='Structural candidates only; no automatic medical-reference or relation approval',patient_study='UNKNOWN',new_GPU_seconds=0,new_judgments=0)
    return dict(public=public,flow=flow,CONFIRM_candidates=confirm,train_H_candidate_pool=expanded,exposure_files=[str(f) for f in exposure_files],pool_binding=digest(raw))

if __name__=='__main__':
    out=Path(__file__).parent;d=inventory();(out/'private/SOURCE_INVENTORY.json').write_text(json.dumps(d,indent=2)+'\n');(out/'SOURCE_INVENTORY.json').write_text(json.dumps(d['public'],indent=2)+'\n');print(json.dumps(d['public'],indent=2))
