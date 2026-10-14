"""Selected two-method coverage; historical recipes never become matched controls."""
import collections
import copy
import contract

METHODS=('M13','FISHER_MIN')
METHOD_CONSUMERS={'M13':7494,'FISHER_MIN':7494,'BASE':2093}
CONSUMERS=sum(METHOD_CONSUMERS.values())
# Count reservations only. Wall/GPU limits require a separate execution admission.
COUNT_RESERVATIONS=dict(updates=146*3*160,backwards=146*(324+324+16004)+1952,
    operator_backwards=146*(160+5120),local_solves=146*7680,
    non_generation_forwards=146*(4000+5000+100000)+128,
    generation_attempts=CONSUMERS,Judge_attempts=CONSUMERS)


def coverage(packet):
    assert packet['source_recipe']==contract.SOURCE_RECIPE
    assert contract.source_admission(packet)['status']=='PASS'
    rows=contract.coverage_reservation(packet['original_tasks'],packet['base_mask'])
    selected=[]
    for row in rows:
        if row['method']=='M3':continue
        row=copy.deepcopy(row)
        if row['method']=='M1':row['method']='FISHER_MIN'
        selected.append(row)
    assert collections.Counter(r['method'] for r in selected)==METHOD_CONSUMERS
    assert len({contract.consumer_id(r) for r in selected})==CONSUMERS==17081
    assert not any(r['query'].get('audit_role')=='GFIT' for r in selected)
    return selected


def evaluation_plan(packet,queries):
    assert contract.original.ids(packet['original_tasks'])<=queries.keys()
    rows=coverage(packet)
    for row in rows:
        qid=row['query']['query_id']
        if qid in queries:row['query']=queries[qid]
        assert all(row['query'].get(k) for k in ('question','reference','image_path'))
    return rows


def selfcheck(packet):
    rows=coverage(packet)
    assert set(r['prefix'] for r in rows if r['mode']=='bank')==set(contract.PREFIXES)
    assert sum(r['mode']=='insertion' for r in rows)==292
    return dict(status='PASS',primary_methods=list(METHODS),consumer_count=CONSUMERS,
        counts=METHOD_CONSUMERS,original_prefixes_and_292_insertions=True,
        all_original_events_and_T2L_mask_preserved=True,GFIT_training_excluded=True,
        count_reservations=COUNT_RESERVATIONS,execution_admitted=False,
        new_GPU=0,new_VLM=0,new_Judge=0)
