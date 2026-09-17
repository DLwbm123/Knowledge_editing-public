"""Build a score-independent DEV candidate packet from existing authorized sources.

This creates no training dispatch. Unknown clinical/patient relations stay unverified.
"""
from pathlib import Path
import json
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import train_row,validate_task
from scripts.medtrace.stage13r_sources import canonical,conflict
from scripts.medtrace.prepare_stage2_sources import reviewed_attribute


def group(r):return canonical(r['dataset'],r['image_path'])


def qualify(packet, reviews, base, verdicts):
    """Join fresh Base decisions and image review; never reinterpret an unknown as pass."""
    from scripts.medtrace.astra_judge_bundle import validate
    if digest({k:v for k,v in packet.items() if k!='freeze_id'})!=packet['freeze_id']:
        raise ValueError('Source packet changed')
    if base['packet_binding']!=digest(packet) or verdicts['source_binding']!=digest(base):
        raise ValueError('Base/Judge lineage changed')
    batch=dict(batch_id='qualification',records=[dict(opaque_query_id=digest(r)) for r in base['records']])
    decisions=validate(batch,dict(batch_id='qualification',decisions=verdicts['decisions']))
    correct={r['opaque_query_id']:r['is_correct'] for r in decisions}
    by_query={r['query_id']:r for r in base['records']}
    relation={r['review_id']:r for r in reviews['records']}
    if len(relation)!=len(reviews['records']):raise ValueError('Duplicate review IDs')
    expected=set(); results=[]
    for p in packet['candidate_packages']:
        t=p['training']; n=t['native']; selected={}
        for role,rows in [('H_fit',t['H_fit']),('H_eval',[r for r in p['evaluation'] if r['role']=='H_eval'])]:
            ids=[digest([t['canonical_edit_id'],role,r])[:20] for r in rows];expected.update(ids)
            selected[role]=sum(relation[i]['verdict']=='SUPPORTED' for i in ids)
        row=by_query[digest([n['image_sha256'],n['question']])]
        if row['source']['reference']!=n['reference']:raise ValueError('Native reference changed')
        wrong=not correct[digest(row)]
        results.append(dict(candidate_id=p['candidate_id'],native_Base_wrong=wrong,
            reviewed_H_fit=selected['H_fit'],reviewed_H_eval=selected['H_eval'],
            strict_image_level_DEV_eligible=wrong and selected['H_fit']>0 and selected['H_eval']>0,
            patient_study_independence='UNKNOWN',formal_eligible=False))
    if expected!=set(relation):raise ValueError('Review coverage mismatch')
    return results


def isolation(training,evaluation):
    train_groups={group(r) for r in training};train_hashes={r['image_sha256'] for r in training}
    native={group(r) for r in training if r['role']=='native'}
    if any(group(r) in native for r in training if r['role']!='native'):raise ValueError('Auxiliary support overlaps a current/future native')
    if any(group(r) in train_groups or r['image_sha256'] in train_hashes for r in evaluation):raise ValueError('Evaluation overlaps global training')
    return sorted(train_groups),sorted({group(r) for r in evaluation})


def build(source,private):
    read=lambda p:json.loads(p.read_text())
    overlay=read(source/'SOURCE_OVERLAY.json');join=read(source/'ROLE_BOUNDARY_JOIN.json')
    audit=read(private/'H_EVAL_EXPOSURE_AUDIT.json');old=read(private/'TRAINING_TASKS.json')['tasks']
    pool=overlay['rows'];by_qid={r['source_qid']:r for r in pool}
    allowed={(r['source_group'],r['source_qid']) for r in join['support_candidate_source_ids']}
    aux=[r for r in pool if (r['source_group'],r['source_qid']) in allowed]
    reserved={tuple(g) for g in join['reserved_source_groups']}
    eval_pool=[r for r in pool if overlay['image_roles'][r['source_group']]=='evaluation'
        and not audit['group_training_uses'][r['source_group']] and group(r) not in reserved]
    tasks=list(old)
    # Existing native=100 proposal would collide with both smoke tasks' shared H/U/G.
    # Native=203 is the only compatible extra source-level package in this finite pool.
    native=by_qid[725];edit_id='stage18:SLAKE:'+native['source_release']+':725'
    q=native['question']
    extra=dict(canonical_edit_id=edit_id,order=595,seed=int(digest([20260912,edit_id])[:8],16),native=train_row(native,'native'),
        fit_questions=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}'],
        H_fit=[train_row(by_qid[8],'H_fit')],G_fit=[train_row(by_qid[7],'G_fit')],U_fit=[train_row(by_qid[11],'U_fit')])
    tasks.append(validate_task(extra))
    training=[r for t in tasks for r in [t['native']]+t['H_fit']+t['G_fit']+t['U_fit']]
    train_groups={group(r) for r in training};train_hashes={r['image_sha256'] for r in training}
    packages=[];reviews=[];evaluation=[]
    for t in tasks:
        validate_task(t)
        hs=[r for r in eval_pool if conflict(t['native'],r) and group(r) not in train_groups and r['image_sha256'] not in train_hashes]
        hs.sort(key=lambda r:(r['source_group'],r['source_qid']))
        # Fixed unrelated modality source on another evaluation-only image.
        us=[r for r in eval_pool if r['source_qid']==3639 and reviewed_attribute(r['question'])!=reviewed_attribute(t['native']['question'])]
        if not hs or not us:raise ValueError('Incomplete source-level candidate package')
        erows=[train_row(r,'H_eval') for r in hs]+[train_row(r,'U_eval') for r in us];evaluation+=erows
        packages.append(dict(candidate_id=t['canonical_edit_id'],training=t,evaluation=erows,
            source_label_and_known_image_isolation='SUPPORTED',clinical_relationship='UNVERIFIED',patient_study_separation='UNVERIFIED',
            fresh_Base_wrong='UNVERIFIED',status='UNVERIFIED_NOT_DISPATCHABLE',
            historical_exposure='DEV: previous evaluation/model exposure disclosed; not a fresh confirmation set'))
        for role,rows in [('H_fit',t['H_fit']),('H_eval',[r for r in erows if r['role']=='H_eval'])]:
            for r in rows:
                reviews.append(dict(review_id=digest([t['canonical_edit_id'],role,r])[:20],edit=t['canonical_edit_id'],role=role,
                    native=t['native'],support=r,source_evidence='Author dataset answer and exact question match; finite proposition conflict predicate',
                    to_verify=['same proposition and question scope','source answer consistent with image; no negation/severity/location ambiguity',
                        'different patient/study where recoverable; record unknown rather than certify','not a crop/near-duplicate or adjacent slice of training image'],
                    status='UNVERIFIED',reviewer=None,verdict=None,evidence=None))
    training_groups,evaluation_groups=isolation(training,evaluation)
    # Preserve the entire old candidate native/eval exclusion for auxiliary rows.
    blocked={tuple(x) for x in read(private/'SOURCE_ROLE_LEDGER.json')['protected_groups']}
    if any(group(r) in blocked for r in training if r['role']!='native'):raise ValueError('Historical/future role leakage')
    used={(group(r),r['source_qid'],r['question']) for r in training+evaluation}
    packet=dict(version='SOURCE_CANDIDATES_V1',selection='Existing labels and prior role/exposure metadata only; no student predictions or scores read',
        status='PARTIAL_DEV_SOURCE_PACKAGES_NOT_READY_FOR_TRAINING',target=16,candidate_packages=packages,
        source_package_count=len(packages),launch_ready_count=0,shortfall_to_16=16-len(packages),
        training_groups=training_groups,evaluation_groups=evaluation_groups,training_evaluation_overlap=0,
        independent_patient_groups=None,clinical_review_signed=False,unique_QA=len(used),
        historical_smoke_support=dict(selected_H_QA=1,selected_H_images=1,supported_edits=2,
            eligible_H_QA_before_new_pilot_native=2,eligible_H_images_before_new_pilot_native=2),
        new_pilot_auxiliary_change='source203 is a pilot native; its H candidate q726 is excluded globally from pilot auxiliary training',
        remaining=['human relation/near-duplicate review','patient/study metadata or explicit unknown limitation decision','fresh Base and approved uniform Judge','13 additional source-qualified packages; all candidate counts remain pre-Base-filter'],
        authorization='Existing authorized SLAKE train/previous evaluation roles only; no new download, no heldout-to-train conversion, no GPU dispatch')
    packet['freeze_id']=digest(packet)
    for name,data in [('PILOT_CANDIDATES.json',packet),('H_RELATION_REVIEW_QUEUE.json',reviews)]:
        with (private/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    return packet,reviews


if __name__=='__main__':
    import sys
    p,reviews=build(Path(sys.argv[1]),Path(sys.argv[2]))
    print(json.dumps(dict(packages=p['source_package_count'],missing=p['shortfall_to_16'],review_relations=len(reviews),unique_QA=p['unique_QA'],freeze_id=p['freeze_id'])))
