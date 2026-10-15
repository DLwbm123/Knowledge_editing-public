"""M13 preparation only: version/source admission and complete coverage reservations."""
import collections
import copy
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
original_path=ROOT/'experiments/medtrace_root_refine_v1/full146_contract.py'
if not original_path.exists():original_path=Path(__file__).with_name('original146_contract.py')
spec=importlib.util.spec_from_file_location('original146_contract',original_path)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
METHODS=('C','M1','M3','M13')
FLAGS={'M1':(True,False,False),'M3':(False,False,True),'M13':(True,False,True)}
PREFIXES=original.PREFIXES
consumer_id=original.consumer_id
fit_rows=original.fit_rows
# Upper reservations, including one Fisher reconstruction if no lawful cache survives.
CAPS=dict(updates=93440,backwards=282272,operator_backwards=46720,
          non_generation_forwards=500000,local_solves=0,
          evaluation_consumers=24867,generation_attempts=24867,Judge_attempts=24867)


def source_admission(packet):
    tasks=packet['original_tasks'];review=packet['reviews'];errors=[]
    assert len(tasks)==146 and [t['order'] for t in tasks]==list(range(1,147))
    assert all(len(t['fit_questions'])==4 for t in tasks)
    forbidden=set(packet['forbidden']);by_owner=collections.defaultdict(list)
    for anchor in review['qualified_anchors']:by_owner[anchor['owner']].append(anchor)
    extra=packet['extra_authorization']
    assert extra['status']=='AUTHORIZED_FINITE_ADDITIONAL_SOURCE_REVIEW'
    assert extra['owner']==8 and extra['total_candidate_limit']==12
    assert extra['original_question_unchanged'] and extra['source_evidence_gate_unchanged']
    reviews=copy.deepcopy(review['owners']);reviews.setdefault('8',[])
    known={r['candidate_id'] for r in reviews['8']}
    reviews['8'] += [r for r in packet['extra_review']['rows'] if r['candidate_id'] not in known]
    qualified=[];exhausted=[]
    required=dict(relation_status='PASS_SOURCE_EVIDENCE',image_review='PASS_READABLE_SOURCE_FIGURE',
        basis_conflict_review='PASS_NO_CONTRADICTORY_TARGET',role_check='PASS_KNOWN_IMAGE_AND_CASE_EXCLUSION')
    for task in tasks:
        owner=task['order'];rows=by_owner[owner];examined=reviews.get(str(owner),[]);cap=12 if owner==8 else 6
        good=len(rows)==2 and len({r['source_group'] for r in rows})==2 and len({r['image_path'] for r in rows})==2
        good=good and 2<=len(examined)<=cap and len({r['candidate_id'] for r in examined})==len(examined)
        for row in rows:
            evidence=next((v for v in examined if v['candidate_id']==row.get('candidate_id')), {})
            case=row['source_group'].split(':')[0] if row['source_group'].startswith('PMC') else row['source_group']
            good=good and row.get('role')=='GFIT' and row['source_group'] not in forbidden and case not in forbidden
            good=good and bool(evidence.get('admitted') and evidence.get('license') and evidence.get('source_evidence'))
            good=good and evidence.get('task_binding')==original.task_binding(task)
            good=good and evidence.get('image_binding',{}).get('path')==row['image_path']
            good=good and packet['image_availability'].get(row['image_path'],False)
            good=good and all(evidence.get(k)==v for k,v in required.items())
        if good:qualified.append(owner)
        else:
            errors.append(owner)
            if len(examined)>=cap:exhausted.append(owner)
    return dict(status='PASS' if not errors else 'SOURCE_ADMISSION_PENDING',qualified_owners=qualified,
        missing_owners=errors,exhausted_owners=exhausted,qualified_anchor_count=sum(len(by_owner[o]) for o in qualified),
        original_edits=146,source_recipe_unchanged=True,training_ready=False,
        reason='Source evidence and a separate executable runtime admission are both required')


def coverage_reservation(tasks,base_mask):
    """Missing GFIT entries are explicit reserved slots, never admitted data."""
    queries={q:{'query_id':q} for q in original.ids(tasks)}
    slots=[dict(owner=t['order'],query_id=f"RESERVED_GFIT_{t['order']}_{i}",unfilled_source_slot=True)
           for t in tasks for i in range(2)]
    rows=original.evaluation_plan(tasks,queries,slots,base_mask)
    for row in rows:
        if row['method']=='REFINE':row['method']='M13'
    rows += [dict(row,method='M3') for row in rows if row['method']=='M1']
    ids=[original.consumer_id(row) for row in rows]
    assert len(set(ids))==len(rows)==CAPS['evaluation_consumers']
    assert collections.Counter(r['method'] for r in rows)==dict(M1=7494,M3=7494,M13=7494,BASE=2385)
    return rows


def evaluation_plan(packet,queries):
    assert source_admission(packet)['status']=='PASS'
    tasks=packet['original_tasks'];anchors=packet['reviews']['qualified_anchors']
    assert original.ids(tasks)<=queries.keys()
    assert all(all(k in queries[q] for k in ('question','reference','image_path')) for q in original.ids(tasks))
    rows=original.evaluation_plan(tasks,queries,anchors,packet['base_mask'])
    for row in rows:
        if row['method']=='REFINE':row['method']='M13'
    rows += [dict(row,method='M3') for row in rows if row['method']=='M1']
    assert len({consumer_id(row) for row in rows})==len(rows)==CAPS['evaluation_consumers']
    return rows


def selfcheck():
    assert FLAGS['M13']==(True,False,True) and all(not f[1] for f in FLAGS.values())
    assert CAPS['updates']==146*160*len(METHODS)
    assert CAPS['backwards']==3*CAPS['updates']+61*32
    assert CAPS['operator_backwards']==2*146*160
    assert CAPS['evaluation_consumers']==3*7494+2385
    assert CAPS['local_solves']==0
    return dict(status='PASS',M13_original_flags=True,soft_Fisher_not_a_hard_budget=True,
        no_90percent_or_direction_gate=True,bounded_counts=True,training_launcher_implemented=True,training_ready=False)
