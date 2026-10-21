"""Fixed full-cohort masks; no source or probe selected by edited outcomes."""
from configuration import PREFIXES,N


def protection_ids(tasks,ledger,task=None):
    assert len(tasks)==N
    active={(t['native']['image_sha256'],t['native']['question']) for t in tasks}
    return list(dict.fromkeys(q for t in tasks for e in t['events']
        if e['task'] in ((task,) if task else ('T1L','T2L')) for q in e['all_probe_query_ids']
        if ledger['Base_correctness'][q] and (ledger['queries'][q]['image_sha256'],ledger['queries'][q]['question']) not in active))


def expected_ids(tasks,ledger,mode,owner,prefix):
    assert len(tasks)==N
    if mode=='protect146_R0':
        assert owner==0 and prefix in PREFIXES
        return protection_ids(tasks,ledger)
    if mode=='bank_R0':
        assert owner==0 and prefix in PREFIXES
        return list(dict.fromkeys(q for t in tasks[:prefix] for e in t['events'] for q in e['all_probe_query_ids']))
    assert mode=='insertion_R0' and 1<=owner<=N and prefix==owner
    t=tasks[owner-1]
    return list(dict.fromkeys([t['edit_id']]+[q for e in t['events'] if e['task']=='T2G'
        for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]))
