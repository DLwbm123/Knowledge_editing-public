"""Versioned exploratory DEV from existing train sources; preserves old evaluation roles."""
from collections import defaultdict
from pathlib import Path
import random

from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage13r_sources import canonical, RELEASE
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import train_row, validate_task, conflict
from scripts.medtrace.stage18_pilot import isolation
from scripts.medtrace.prepare_stage2_sources import reviewed_attribute


def source_groups(value,dataset='SLAKE'):
    """Identity projection, including original/derived paths; no QA or scores exported."""
    result=set()
    if isinstance(value,dict):
        dataset=value.get('source_dataset',value.get('dataset',dataset))
        for key in ('image_name','image_path','relative_image_path','image_id','original_image_path'):
            if value.get(key):
                try:result.add(canonical(dataset,str(value[key])))
                except ValueError:pass  # Generated/non-source files have no dataset identity.
        for child in value.values():
            if isinstance(child,(dict,list)):result|=source_groups(child,dataset)
    elif isinstance(value,list):
        for child in value:result|=source_groups(child,dataset)
    return result


def collect(storage,private):
    """Existing author train only; frozen reservation metadata and historical eval denylist."""
    import json
    outputs=storage/'Knowledge_editing/outputs'
    join=read(private/'ROLE_BOUNDARY_JOIN.json')
    ledger=read(private.parent/'SOURCE_ROLE_LEDGER.json')
    natives={canonical(t['native']['dataset'],t['native']['image_path']) for t in ledger['tasks']}
    queries={canonical(r['dataset'],r.get('original_image_path',r['image_path'])) for r in ledger['evaluation_queries'].values()}
    blocked={tuple(g) for g in join['reserved_source_groups']+join['evaluation_source_groups']}|(queries-natives)
    evidence=[]
    for rel in ('20260906T030312Z/private/frozen_data.json','20260905T134727Z/private/frozen_data.json'):
        value=read(storage/'medtrace_runs'/rel)
        scopes=value['scopes'].values() if 'scopes' in value else [value.get('scope',{})]
        groups=set()
        for scope in scopes:
            for key in ('negative_roles','positives'):
                for role,rows in scope.get(key,{}).items():
                    if role!='fit':groups|=source_groups(rows)
        blocked|=groups;evidence.append(dict(path=rel,nontraining_groups=len(groups)))
    rel='20260905T125112Z/track_b/roles_private.json';value=read(storage/'medtrace_runs'/rel);groups=set()
    for key in ('negatives','positives'):
        item=value.get(key,{})
        if isinstance(item,dict):
            for role,rows in item.items():
                if role!='fit':groups|=source_groups(rows)
        else:groups|=source_groups([r for r in item if r.get('role')!='fit'])
    blocked|=groups;evidence.append(dict(path=rel,nontraining_groups=len(groups)))
    later=set()
    def visit(value):
        if isinstance(value,dict):
            role=str(value.get('role','')).lower()
            if any(k in role for k in ('eval','calib','challenge','probe')) and 'fit' not in role:later.update(source_groups(value))
            else:
                for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    files=list(outputs.glob('medtrace_stage*/private/edits/e*.json'))
    for file in files:visit(read(file))
    blocked|=later
    hashes=defaultdict(set); originals={}
    catalog=outputs/'m3bench_data_runtime_finalization_v3/20260904T014138Z/data_static/STATIC_QUERY_INVENTORY.jsonl'
    for line in catalog.open():
        r=json.loads(line)
        try:group=canonical(r['dataset'],r['image_path'])
        except ValueError:continue
        if r.get('image_sha256'):hashes[r['image_sha256']].add(group)
        if r['dataset']=='SLAKE' and Path(r['image_path']).name=='source.jpg':
            if group[1] in originals and originals[group[1]]!=r['image_sha256']:raise ValueError('Ambiguous original identity')
            originals[group[1]]=r['image_sha256']
    while True:
        extra=set().union(*(g for g in hashes.values() if g&blocked),set())-blocked
        if not extra:break
        blocked|=extra
    source=storage/'DataP/knowledge_editing/data/m3bench/SLAKE/train.json';raw=read(source)
    groups={('SLAKE',Path(r['img_name']).parent.name) for r in raw};allowed=groups-blocked
    rows=[r for r in raw if r.get('q_lang')=='en' and ('SLAKE',Path(r['img_name']).parent.name) in allowed]
    if any(Path(r['img_name']).parent.name not in originals for r in rows):raise ValueError('Missing original image identity')
    result=dict(source_file=str(source),raw_rows=len(raw),raw_groups=len(groups),rows=rows,
        allowed_training_groups=sorted(allowed),excluded_groups=sorted(blocked),image_hashes=originals,
        historical_evaluation_metadata_sources=evidence,later_role_metadata_audit=dict(files=len(files),groups=len(later)),
        existing_hash_identity_closure=True,prior_training_exposure='Allowed for explicitly versioned DEV only; never unseen confirmation')
    write_new(private/'CURRENT_DATA_ALL_TRAIN_V2.json',result)
    return result


def build(private):
    src=read(private/'CURRENT_DATA_ALL_TRAIN_V2.json')
    overlay=read(private/'SOURCE_OVERLAY.json')
    join=read(private/'ROLE_BOUNDARY_JOIN.json')
    allowed={tuple(g) for g in src['allowed_training_groups']}
    blocked={tuple(g) for g in src['excluded_groups']}
    hashes={('SLAKE',g):h for g,h in src['image_hashes'].items()}
    rows=[]
    for raw in src['rows']:
        group=('SLAKE',Path(raw['img_name']).parent.name)
        if group not in allowed or group in blocked or raw['q_lang']!='en':
            raise ValueError('Source outside authorized DEV train boundary')
        rows.append(dict(dataset='SLAKE',image_path=str(Path(src['source_file']).parent/'imgs'/raw['img_name']),
            image_sha256=hashes[group],source_group='SLAKE:'+RELEASE+':'+group[1],
            question=raw['question'],reference=raw['answer'],source_qid=raw['qid']))
    groups=sorted({r['source_group'] for r in rows})
    random.Random(20260917).shuffle(groups)
    # Partition before Base/student outputs. The old shared U source stays auxiliary.
    auxiliary={g for i,g in enumerate(groups) if i%3==0}|{'SLAKE:'+RELEASE+':xmlab100'}
    fit=[r for r in rows if r['source_group'] in auxiliary]
    native=[r for r in rows if r['source_group'] not in auxiliary]
    evaluation=[r for r in overlay['rows'] if overlay['image_roles'][r['source_group']]=='evaluation'
        and canonical(r['dataset'],r['image_path']) not in {tuple(g) for g in join['reserved_source_groups']}]
    audit=read(private.parent/'H_EVAL_EXPOSURE_AUDIT.json')
    if any(audit['group_training_uses'][r['source_group']] for r in evaluation):
        raise ValueError('Existing evaluation source was used for training')
    by_group=defaultdict(list)
    for r in fit:by_group[r['source_group']].append(r)
    u=next(r for r in fit if r['source_qid']==7)
    ue=next(r for r in evaluation if r['source_qid']==3639)
    priority=lambda r:(0 if r['reference'].strip().lower() in ('yes','no') else 1,
        digest([20260917,r['source_group']]),r['source_qid'])
    packages=[]
    for n in sorted(native,key=priority):
        attr=reviewed_attribute(n['question'])
        if attr!='body_region' and n['reference'].strip().lower() not in ('yes','no'):
            continue
        he=sorted((r for r in evaluation if conflict(n,r)),key=lambda r:(r['source_group'],r['source_qid']))[:2]
        matched=[]
        for h in sorted((r for r in fit if conflict(n,r) and r['source_qid']!=u['source_qid']),key=priority):
            gs=[r for r in by_group[h['source_group']] if reviewed_attribute(r['question'])=='modality'
                and r['source_qid']!=u['source_qid']]
            if gs:matched.append((h,sorted(gs,key=lambda r:r['source_qid'])[0]))
        if not matched or not he:continue
        h,g=matched[0];edit='stage18-current:'+RELEASE+':'+str(n['source_qid']);q=n['question']
        task=dict(canonical_edit_id=edit,order=600+len(packages),seed=int(digest([20260912,edit])[:8],16),
            native=train_row(n,'native'),H_fit=[train_row(h,'H_fit')],G_fit=[train_row(g,'G_fit')],U_fit=[train_row(u,'U_fit')],
            fit_questions=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}'])
        validate_task(task)
        packages.append(dict(candidate_id=edit,training=task,evaluation=[train_row(r,'H_eval') for r in he]+[train_row(ue,'U_eval')],
            status='SOURCE_CANDIDATE_REVIEW_AND_FRESH_BASE_PENDING',historical_exposure='EXPLORATORY_DEV_PREVIOUS_TRAINING_ALLOWED',
            patient_study_separation='UNKNOWN'))
    available=len(packages)
    training=[r for p in packages for r in [p['training']['native']]+sum((p['training'][k] for k in ('H_fit','G_fit','U_fit')),[])]
    ev=[r for p in packages for r in p['evaluation']]
    tg,eg=isolation(training,ev)
    packet=dict(version='EXISTING_DATA_DEV_V2',authorization='User requested construction using existing data after proposed versioned DEV repartition',
        amendment='Separate exploratory DEV; historical native training exposure permitted in new disjoint train roles; prior evaluation-only and sealed/reserved sources remain excluded from training. No change to Stage17 or V1.',
        selection='Source labels and fixed source-group seed only; no Base/student scores',target=16,structurally_available=available,
        candidate_packages=packages,source_package_count=len(packages),training_groups=tg,evaluation_groups=eg,
        raw_existing_train_rows=src['raw_rows'],selected_source_QA=len(rows),selected_source_images=len(groups),
        launch_ready_count=0,formal_eligible=False,new_downloads=0,patient_study_independence='UNKNOWN')
    packet['freeze_id']=digest(packet)
    write_new(private/'PILOT_CANDIDATES.json',packet)
    return packet


def freeze(private):
    """Freeze at most 16 reviewed Base-wrong edits; never use student outcomes."""
    from scripts.medtrace.astra_judge_bundle import validate
    packet=read(private/'PILOT_CANDIDATES.json');base=read(private/'BASE_OUTPUTS.json')
    verdicts=read(private/'BASE_JUDGE_V2/operator/VERDICTS.json')
    support=read(private/'ASTRA_SUPPORT_REVIEW_V2.json');native=read(private/'ASTRA_NATIVE_RELATION_REVIEW_V2.json')
    if (digest({k:v for k,v in packet.items() if k!='freeze_id'})!=packet['freeze_id']
            or base['packet_binding']!=digest(packet) or verdicts['source_binding']!=digest(base)):
        raise ValueError('Qualification lineage changed')
    batch=dict(batch_id='qualification',records=[dict(opaque_query_id=digest(r)) for r in base['records']])
    correct={r['opaque_query_id']:r['is_correct'] for r in validate(batch,dict(batch_id='qualification',decisions=verdicts['decisions']))}
    key=lambda r:(r['image_sha256'],r['question'],r['reference'])
    reviewed={key(r):r['verdict']=='SUPPORTED' for r in support['records']}
    nr={r['candidate_id']:r for r in native['records']}
    if len(nr)!=len(native['records']) or len(reviewed)!=len(support['records']):raise ValueError('Duplicate reviews')
    by_query={r['query_id']:r for r in base['records']};selected=[];qualification=[]
    for p in packet['candidate_packages']:
        t=p['training'];n=t['native'];review=nr.get(p['candidate_id'])
        row=by_query[digest([n['image_sha256'],n['question']])]
        if key(row['source'])!=key(n):raise ValueError('Native Base source changed')
        supported=all(reviewed.get(key(r),False) for r in t['H_fit']+t['G_fit']+t['U_fit']+p['evaluation'])
        native_ok=bool(review and review['native_verdict']=='SUPPORTED')
        relation_ok=False
        if review:
            if (review['native_image_sha256'],review['native_question'],review['native_reference'])!=key(n):raise ValueError('Native review changed')
            expected={(r['role'],)+key(r) for r in t['H_fit']+p['evaluation'] if r['role'] in ('H_fit','H_eval')}
            if {(r['role'],)+key(r) for r in review['relation_details']}!=expected:raise ValueError('Relation coverage changed')
            relation_ok=review['relation_verdict']=='SUPPORTED' and all(r['relation_verdict']=='SUPPORTED' and r['support_label_verdict']=='SUPPORTED'
                and r['same_proposition_schema'] is True and r['author_answers_differ'] is True for r in review['relation_details'])
        q=dict(candidate_id=p['candidate_id'],Base_wrong=not correct[digest(row)],native_supported=native_ok,relations_supported=relation_ok,support_supported=supported)
        qualification.append(q)
        if all(q[k] for k in ('Base_wrong','native_supported','relations_supported','support_supported')):selected.append(p)
    selected=sorted(selected[:16],key=lambda p:p['candidate_id'])
    if not selected:raise ValueError('No qualified DEV edits')
    tasks=[validate_task(p['training']) for p in selected]
    tg,eg=isolation([r for t in tasks for r in [t['native']]+t['H_fit']+t['G_fit']+t['U_fit']],
        [r for p in selected for r in p['evaluation']])
    training=dict(scope='REVIEWED_EXISTING_DATA_DEV_V2',tasks=tasks,freeze_id=digest(tasks))
    evaluation=dict(candidate_packages=selected,scope=training['scope'],training_groups=tg,evaluation_groups=eg,
        patient_study_independence='UNKNOWN',formal_eligible=False)
    by_id={r['candidate_id']:r for r in qualification}
    gate=dict(training_freeze=training['freeze_id'],evaluation_binding=digest(evaluation),base_binding=digest(base),
        source_packet_binding=digest(packet),review_bindings=[digest(support),digest(native)],judge_binding=digest(verdicts),
        selected=[by_id[p['candidate_id']] for p in selected],all_candidates=qualification,formal_eligible=False,
        interpretation='Exploratory existing-source DEV; historical training exposure and unknown patient/study independence')
    for name,value in [('TRAINING_TASKS.json',training),('DEV_EVALUATION.json',evaluation),('DEV_QUALIFICATION.json',gate),('QUALIFICATION_BASE_OUTPUTS.json',base)]:
        write_new(private/name,value)
    return training,evaluation,gate


if __name__=='__main__':
    import os
    packet=build(Path(os.environ['JOB_PRIVATE']))
    print({k:packet[k] for k in ('source_package_count','structurally_available','selected_source_QA','selected_source_images','freeze_id')})
