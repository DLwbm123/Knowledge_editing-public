"""Stage18 train-only boundary and finite approved-source assembly."""
from pathlib import Path
from scripts.medtrace.astra_judge_bundle import read, write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage13r_sources import canonical, conflict as legacy_conflict
from scripts.medtrace.prepare_stage2_sources import normalized, reviewed_attribute

ROW_FIELDS={'dataset','image_path','image_sha256','source_group','question','reference','role','source_qid'}
TASK_FIELDS={'canonical_edit_id','order','seed','native','fit_questions','U_fit','H_fit','G_fit'}


def conflict(a,b):
    """Structural proposal only; original yes/no labels still require image review."""
    if legacy_conflict(a,b):return True
    return (canonical(a['dataset'],a['image_path'])!=canonical(b['dataset'],b['image_path'])
        and normalized(a['question'])==normalized(b['question'])
        and {normalized(a['reference']),normalized(b['reference'])}=={'yes','no'})


def validate_task(t):
    if set(t)!=TASK_FIELDS: raise ValueError('Train-only schema rejects unknown/evaluation fields')
    if t['seed']!=int(digest([20260912,t['canonical_edit_id']])[:8],16): raise ValueError('Seed must bind canonical ID, not position')
    q=t['native']['question']
    if t['fit_questions']!=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}']:
        raise ValueError('Only frozen native-only fit wrappers accepted')
    for role in ('native','U_fit','H_fit','G_fit'):
        rows=[t[role]] if role=='native' else t[role]
        if not rows: raise ValueError('UNSUPPORTED: missing '+role)
        for row in rows:
            if set(row)!=ROW_FIELDS or row['role']!=role or any(not row[k] for k in ROW_FIELDS-{'source_qid'}):
                raise ValueError('Incomplete train provenance or evaluation data')
    for h in t['H_fit']:
        if canonical(h['dataset'],h['image_path'])==canonical(t['native']['dataset'],t['native']['image_path']) or not conflict(t['native'],h):
            raise ValueError('UNSUPPORTED: H must be different-image source conflict')
    if len(t['H_fit'])!=len(t['G_fit']): raise ValueError('H/G slot counts must match')
    for h,g in zip(t['H_fit'],t['G_fit']):
        if h['source_group']!=g['source_group']: raise ValueError('H/G must share matched source slots')
        if normalized(g['question'])==normalized(h['question']) or reviewed_attribute(g['question'])==reviewed_attribute(h['question']):
            raise ValueError('G must be a different proposition')
    return t


def train_row(row,role):
    return dict(dataset=row['dataset'],image_path=row.get('original_image_path',row['image_path']),
        image_sha256=row['image_sha256'],source_group=row['source_group'],question=row['question'],
        reference=row['reference'],role=role,source_qid=row.get('source_qid'))


def build(source,output,reports):
    ledger=read(source/'COHORT_AND_SUPPORT_LEDGER.json'); join=read(source/'ROLE_BOUNDARY_JOIN.json'); overlay=read(source/'SOURCE_OVERLAY.json')
    if digest({k:v for k,v in ledger.items() if k!='freeze_id'})!=ledger['freeze_id']: raise ValueError('Source ledger changed')
    identity=lambda r:(r['dataset'],str(r['source_qid']),r['source_group'])
    allowed={identity(x) for x in join['support_candidate_source_ids']}
    pool=[r for r in overlay['rows'] if identity(r) in allowed]
    if len(pool)!=29: raise ValueError('Approved finite source inventory changed')
    protected={tuple(x) for x in join['evaluation_source_groups']+join['reserved_source_groups']}
    query_rows=list(ledger['queries'].values())+[t['native'] for t in ledger['tasks']]
    protected|={canonical(r['dataset'],r.get('original_image_path',r['image_path'])) for r in query_rows if r.get('image_path')}
    hashes={r['image_sha256'] for r in query_rows if r.get('image_sha256')}
    for r in pool:
        if r['source_role']!='train' or overlay['image_roles'][r['source_group']]!='adaptation': raise ValueError('Non-train source')
        if canonical(r['dataset'],r['image_path']) in protected or r['image_sha256'] in hashes: raise ValueError('Global source/hash leakage')
    train=[]; roles=[]
    for t in ledger['tasks']:
        supported=bool(t['H_fit'] and t['U_fit'] and t['G'] and t['fit_status']=='SUPPORTED')
        if any(identity(r) not in allowed for r in t['H_fit']+t['U_fit']+t['G']): raise ValueError('Source outside approved pool')
        roles.append(dict(edit_id=t['edit_id'],native=t['native'],fit=t['fit_questions'],H_fit=t['H_fit'],G_fit=t['G'],U_fit=t['U_fit'],H_eval=t['H_eval'],
            official_events=t['events'],U_eval=[],U_eval_status='UNVERIFIED_NOT_INDEPENDENTLY_ASSEMBLED',
            official_probe_ids=list(dict.fromkeys(q for e in t['events'] for q in e['all_probe_query_ids'])),
            source_label_smoke_status='SUPPORTED' if supported else 'UNSUPPORTED',current_Base_wrong_status='UNVERIFIED',
            patient_study_separation='UNVERIFIED',H_eval_status='SUPPORTED' if t['H_eval'] else 'UNSUPPORTED'))
        if supported:
            train.append(validate_task(dict(canonical_edit_id=t['edit_id'],order=t['order'],seed=t['seed'],native=train_row(t['native'],'native'),
                fit_questions=t['fit_questions'],U_fit=[train_row(r,'U_fit') for r in t['U_fit']],H_fit=[train_row(r,'H_fit') for r in t['H_fit']],G_fit=[train_row(r,'G_fit') for r in t['G']])))
    train.sort(key=lambda t:t['canonical_edit_id'])
    if len(train)!=2 or any(t['H_eval'] for t in roles): raise ValueError('Unexpected scope: review before expansion')
    output.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    write_new(output/'TRAINING_TASKS.json',dict(scope='TWO_SOURCE_LABEL_SMOKES_ONLY',tasks=train,freeze_id=digest(train)))
    write_new(output/'SOURCE_ROLE_LEDGER.json',dict(historical_freeze=ledger['freeze_id'],tasks=roles,evaluation_queries=ledger['queries'],
        protected_groups=sorted(protected),approved_train_pool=pool,provenance=overlay['provenance'],patient_study='UNKNOWN; not certified independent',
        source_review='Original released labels; no new medical signoff'))
    write_new(output/'CANDIDATE_INDEX.json',[dict(native_id=t['edit_id'],proposition=reviewed_attribute(t['native']['question']),
        native_source=t['native']['source_group'],answer=t['native']['reference'],H_candidates=[dict(source_group=r['source_group'],source_qid=r['source_qid']) for r in pool if conflict(t['native'],r)]) for t in ledger['tasks']])
    counts={}
    for role in ('H_fit','G_fit','U_fit'):
        rows=[r for t in train for r in t[role]]; unique={(r['source_group'],str(r['source_qid'])) for r in rows}
        counts[role]=dict(assignments=len(rows),unique_QA=len(unique),source_groups=len({r['source_group'] for r in rows}),assignments_per_unique_QA=len(rows)/len(unique))
    write_new(reports/'SUPPORT_INVENTORY.json',dict(approved_train_QA=29,approved_source_groups=3,historical_task_natives=594,historical_main146_H=0,
        source_label_smoke_supported=2,Q_H_historical_candidate=2,Q_H_current_Base_verified=0,Q_HG_eval=0,H_eval_independent_sources=0,
        pilot_target_not_found=16,formal_target_not_found=[100,150],roles=counts,patient_study_status='UNVERIFIED',clinical_signoff=False,student_output_selection=False))
    write_new(reports/'COHORT_FLOW.json',dict(task_native_candidates=594,native_fit_U_historical=594,H_missing=592,H_and_G_source_supported=2,
        independent_H_eval_missing=2,Q_HG_eval=0,fresh_Base_and_Judge_pending=2,formal_eligible=0,old_main146_H=0,new_downloads=0,exhausted_pool_rescans=0))
    return train


if __name__=='__main__':
    import sys
    build(*(Path(x) for x in sys.argv[1:]))
