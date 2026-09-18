"""Bounded Stage20 source-only construction. No student answers enter selection."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import conflict, train_row, validate_task
from scripts.medtrace.prepare_stage2_sources import normalized, reviewed_attribute
from scripts.medtrace.stage18_score import query_id


def build(cfg):
    root=Path(cfg['run']);p=root/'private';pub=root/'public';p.mkdir(parents=True,exist_ok=True);pub.mkdir(exist_ok=True)
    old=Path(cfg['old_pure']);source=Path(cfg['source19']);data=Path(cfg['source18']);fast=Path(cfg['old_fasttrack'])
    stream=read(old/'private/STREAM.json');ledger=read(source/'DEV_SOURCE_ROLE_LEDGER.json');raw=read(data/'CURRENT_DATA_ALL_TRAIN_V2.json')
    release=read(data/'SOURCE_OVERLAY.json')['rows'][0]['source_group'].split(':')[1]
    pool=[dict(dataset='SLAKE',source_group='SLAKE:'+release+':'+r['img_name'].split('/')[0],image_path=cfg['image_root']+'/'+r['img_name'],image_sha256=raw['image_hashes'][r['img_name'].split('/')[0]],source_qid=r['qid'],question=r['question'],reference=r['answer'],role='native') for r in raw['rows']]
    assert len(pool)==1154 and len({r['source_group'] for r in pool})==113
    references={};reference_files=[source/'V3_TRAIN_REFERENCE_REVIEW.json',source/'V3_EVAL_REFERENCE_REVIEW.json',source/'BLIND_REFERENCE_REVIEW.json',fast/'private/REFERENCE_REUSED.json',fast/'private/REFERENCE_REVIEW_RESULT.json']
    for f in reference_files:
        for r in read(f)['records']:
            key=(r['image_sha256'],normalized(r['question']),r['reference']);oldref=references.get(key)
            if oldref and oldref['verdict']!=r['verdict']:raise ValueError('Conflicting saved reference decisions')
            references[key]=dict(r,provenance_file=str(f))
    def review(r):return references.get((r['image_sha256'],normalized(r['question']),r['reference']))
    oldtrain=[r for t in stream['tasks'] for r in [t['native']]+t['H_fit']+t['G_fit']+t['U_fit']]
    used={r['source_group'] for r in oldtrain};usedhash={r['image_sha256'] for r in oldtrain}
    history=set(used)
    for f in [fast/'private/STREAM.json',data/'TRAINING_TASKS.json']:
        for t in read(f)['tasks']:
            history|={r['source_group'] for r in [t['native']]+t['H_fit']+t['G_fit']+t['U_fit']}
    protected=set(ledger['evaluation_sources']);quarantine=set(ledger['quarantined_groups'])
    near=read(source/'NEAR_DUPLICATE_SCREEN.json');blockedhash={r['image_sha256'] for r in ledger['evaluation_pool']}
    auxiliary=[r for r in pool if r['source_group'] in ledger['auxiliary_groups'] and r['source_group'].split(':')[-1] not in quarantine and review(r) and review(r)['verdict']=='SUPPORTED']
    def matches(r):return [h for h in auxiliary if conflict(r,h)]
    nativepool=[r for r in pool if r['source_group'] in ledger['native_groups'] and r['source_group'].split(':')[-1] not in quarantine]
    bygroup=defaultdict(list)
    for r in nativepool:bygroup[r['source_group']].append(r)
    # Reserve evaluation sources before assigning any new train sources. No Base scores used.
    eval_options=[]
    for g,rows in bygroup.items():
        if g in history or any(r['image_sha256'] in usedhash|blockedhash for r in rows):continue
        hs=[r for r in rows if any(conflict(t['native'],r) for t in stream['tasks'])]
        us=[r for r in rows if reviewed_attribute(r['question']) in ('modality','plane')]
        if hs and us:eval_options.append((g,hs,us))
    eval_options.sort(key=lambda z:(-min(3,len(z[1])),-min(2,len(z[2])),digest([20001,z[0]])))
    chosen=eval_options[:8];newgroups={g for g,_,_ in chosen};panels=[];links=[]
    for g,hs,us in chosen:
        for role,rows,cap in [('H_eval',hs,3),('U_eval',us,2)]:
            for r in sorted(rows,key=lambda r:(not bool(review(r) and review(r)['verdict']=='SUPPORTED'),digest([20001,r['source_qid']])) )[:cap]:
                row=train_row(r,role);panels.append(row)
                if role=='H_eval':
                    for t in stream['tasks']:
                        if conflict(t['native'],r):links.append(dict(native_id=t['canonical_edit_id'],native=t['native'],probe=row,relation='H_NEW_EVAL'))
    protected|=newgroups;blockedhash|={r['image_sha256'] for r in panels}
    oldq={query_id(t['native']) for t in stream['tasks']};oldnatives={t['native']['source_qid'] for t in stream['tasks']};flow=[];candidates=[];pergroup=Counter()
    ordered=sorted(pool,key=lambda r:(digest([20001,r['source_group']]),digest([20001,r['source_qid']])))
    eligible=[]
    for r in ordered:
        reasons=[]
        if query_id(r) in oldq:reasons.append('ORIGINAL19_PRESERVED')
        if r['source_group'] in protected or r['image_sha256'] in blockedhash:reasons.append('PROTECTED_EVALUATION_SOURCE')
        if r['source_group'].split(':')[-1] in quarantine:reasons.append('NEAR_DUPLICATE_QUARANTINE')
        if r['source_group'] not in ledger['native_groups']:reasons.append('AUXILIARY_OR_NON_NATIVE_SOURCE')
        hs=matches(r)
        if not hs:reasons.append('NO_SUPPORTED_DIFFERENT_IMAGE_H_FIT')
        rev=review(r)
        if rev and rev['verdict']!='SUPPORTED':reasons.append('REFERENCE_'+rev['verdict'])
        flow.append(dict(source_qid=r['source_qid'],reasons=reasons))
        if not reasons:eligible.append(r)
    # Diversity-first candidate order is frozen before all Base qualification.
    while eligible and len(candidates)<120:
        eligible.sort(key=lambda r:(pergroup[r['source_group']],not bool(review(r)),digest([20001,r['source_qid']])))
        n=eligible.pop(0);pergroup[n['source_group']]+=1;hs=matches(n)
        h=min(hs,key=lambda r:digest([20001,n['source_qid'],r['source_qid']]))
        us=[r for r in auxiliary if reviewed_attribute(r['question'])!=reviewed_attribute(n['question']) and normalized(r['question'])!=normalized(n['question'])]
        if not us:raise ValueError('No legal U fit')
        u=min(us,key=lambda r:digest([20001,n['source_qid'],r['source_qid']]))
        edit='stage20:'+str(n['source_qid']);q=n['question']
        task=dict(canonical_edit_id=edit,order=20+len(candidates),seed=int(digest([20260912,edit])[:8],16),native=train_row(n,'native'),fit_questions=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}'],H_fit=[train_row(h,'H_fit')],U_fit=[train_row(u,'U_fit')],G_fit=[])
        validate_task(task,fasttrack_branch='C_FACT');candidates.append(task);links.append(dict(native_id=edit,native=task['native'],probe=task['H_fit'][0],relation='H_FIT'))
    # One decision per unique source-question-reference; review once, no semantic retries.
    required={query_id(r):r for r in panels}
    for t in candidates:
        for r in [t['native']]+t['H_fit']+t['U_fit']:required[query_id(r)]=r
    reused=[dict(row=r,review=review(r)) for r in required.values() if review(r)]
    novel=[r for r in required.values() if not review(r)]
    relation_items=[]
    for link in links:
        # Same probe may relate to several old anchors: retain all mappings, review exact pair once.
        key=digest([query_id(link['native']),query_id(link['probe'])]);relation_items.append(dict(link,id=key))
    relation_items=list({r['id']:r for r in relation_items}.values())
    selectedids={t['native']['source_qid'] for t in candidates}
    for r in flow:
        if not r['reasons'] and r['source_qid'] not in selectedids:r['reasons']=['BOUNDED_120_CANDIDATE_LIMIT']
    assert not {r['source_group'] for t in candidates for r in [t['native']]+t['H_fit']+t['U_fit']} & protected
    assert not {r['image_sha256'] for t in candidates for r in [t['native']]+t['H_fit']+t['U_fit']} & blockedhash
    artifacts=dict(original_stream=stream,candidates=candidates,new_panel_candidates=panels,H_relation_candidates=relation_items,order_seed=20001,source_role_policy='Evaluation first; original19 unchanged; new qualified edits max2/source after Base mask; no student access',new_source_groups=sorted(newgroups),protected_groups=sorted(protected),quarantined_groups=sorted(quarantine),reference_novel=novel,reference_reused=reused,source_flow=flow,patient_study='UNKNOWN',historical_exposure='Existing project DEV pool; not unseen or patient-independent; Stage18/19 known trained sources excluded from new panel',prior_source_ledger_binding=digest(ledger),pool_binding=digest(raw),maximum_new_per_source=2)
    artifacts['freeze_id']=digest(artifacts)
    write_new(p/'SOURCE_ROLE_LEDGER.json',artifacts)
    write_new(p/'REFERENCE_REVIEW_QUEUE.json',dict(records=novel,freeze_id=digest(novel)))
    write_new(p/'RELATION_REVIEW_QUEUE.json',dict(records=relation_items,freeze_id=digest(relation_items)))
    write_new(p/'OLD_FIXED_PANEL.json',dict(rows=stream['core_rows'],anchor_packages=stream['anchor_packages'],exposure='VIEWED_DEV',binding=digest(stream['core_rows'])))
    write_new(p/'NEW_SOURCE_PANEL_CANDIDATES.json',dict(rows=panels,status='SOURCE_RESERVED_REFERENCE_REVIEW_PENDING',binding=digest(panels)))
    report=dict(scanned_QA=len(pool),scanned_sources=len(bygroup),all_scanned_sources=len({r['source_group'] for r in pool}),approved_pool_only=True,new_downloads=0,sealed_access=False,original_prefix=19,structural_candidate_count=len(candidates),candidate_sources=len(pergroup),new_reference_items=len(novel),exact_reference_reuses=len(reused),new_relation_items=len(relation_items),new_panel_H=sum(r['role']=='H_eval' for r in panels),new_panel_U=sum(r['role']=='U_eval' for r in panels),new_panel_sources=len(newgroups),exclusion_reasons=dict(Counter(reason for r in flow for reason in r['reasons'])),flow_counts_nonexclusive=True,base_qualification='PENDING_CURRENT_GPU3_BASE',maximum_new_per_source=2,source_freeze=digest(artifacts),status='PRE_STUDENT_CANDIDATES_NOT_TRAINING_QUALIFIED')
    if len(novel)+len(relation_items)+len(candidates)+len(panels)+19>500:raise ValueError('Conservative qualification reservation exceeds500')
    write_new(pub/'COHORT_FLOW.json',report)
    write_new(pub/'ABLATION_PREREGISTRATION.json',dict(configurations=['FACT_HSIC_MAIN_PURE11','NO_H_HSIC','EXTRA_HSIC'],candidate_edits=11,all_H_G_structurally_legal=all(validate_task(t) for t in stream['tasks'][:11]) is True,training_allowed_after_support_freeze=True,selection_uses_student_scores=False,priority='After main common endpoint; only resource and support gates',same_W0=True,same_native_layer_map=True,independent_optimizers=True,max_new_judgments=200,max_extra_GPU_seconds=3600,final_banks_retained=True))
    return report


def hsic_audit(cfg):
    root=Path(cfg['run']);old=Path(cfg['old_pure']);rows=[]
    for f in sorted((old/'private/edits').glob('e*/SELECTION.json')):
        d=read(f);s=sorted(((int(k),v) for k,v in d['scores'].items()),key=lambda x:(-x[1],x[0]));assert len(s)==30 and d['samples']==5
        rows.append(dict(position=int(f.parent.name[1:]),selected_layer=d['layer_id'],scores=d['scores'],invalid=d['invalid'],top1=s[0][0],top2=s[1][0],margin=s[0][1]-s[1][1],token_counts=d['token_counts'],component_terms=None,component_status='Not retained in Stage19; cannot recover two terms from one difference',binding=digest(d)))
    write_new(root/'public/HSIC_SCORE_AUDIT.json',dict(rows=rows,layer_distribution=dict(Counter(r['selected_layer'] for r in rows)),observations=5,observations_independent_facts=False,code_checked='native+4 exact wrappers; Base OFF before/after; down_proj input hooks1..30; router target independent of writer layer',implementation_bug_found=False,old_feature_tensors_retained=False,component_and_spectrum_plan='Record during first authorized reconstruction; no GPU replay solely to fill missing historical terms',method_changed=False))
    return {'edits':len(rows),'minimum_margin':min(r['margin'] for r in rows),'maximum_margin':max(r['margin'] for r in rows)}

if __name__=='__main__':
    cfg=read(sys.argv[2]);print(json.dumps(build(cfg) if sys.argv[1]=='build' else hsic_audit(cfg),indent=2))
