"""Bind a fixed formal cohort; retain raw and Base-qualified denominators."""
from collections import Counter


def build(tasks,ledger):
    assert len(tasks)==8 and [t['order'] for t in tasks]==list(range(1,9))
    queries=ledger['queries'];base=ledger['Base_correctness'];aliases={};panels=[];raw=Counter();eligible=Counter()
    for i,t in enumerate(tasks):
        entries=[(t['edit_id'],'T0',True)]
        for e in t['events']:
            if e['task'] in ('T1G','T2G','T1L','T2L'):
                entries += [(q,e['task'],bool(base[q])==e['task'].endswith('L')) for q in e['all_probe_query_ids']]
        panel=[]
        for q,task,ok in entries:
            assert q in queries
            alias=aliases.setdefault(q,'Q'+str(len(aliases)+1).zfill(3))
            panel.append(dict(query=alias,task=task,eligible=ok));raw[task]+=1;eligible[task]+=int(ok)
        panels.append(dict(alias='E'+str(i+1).zfill(2),panel=panel))
    return panels,aliases,dict(raw),dict(eligible)


def run():
    tasks=[dict(order=i,edit_id='s'+str(i),events=[dict(task='T1L',all_probe_query_ids=['l'+str(i)]),dict(task='T1G',all_probe_query_ids=['g'+str(i)])]) for i in range(1,9)]
    q={k:{} for t in tasks for k in [t['edit_id']]+[x for e in t['events'] for x in e['all_probe_query_ids']]};base={k:k.startswith('l') and k.endswith('1') for k in q}
    p,a,r,e=build(tasks,dict(queries=q,Base_correctness=base))
    assert r==dict(T0=8,T1L=8,T1G=8) and e==dict(T0=8,T1L=1,T1G=8) and len(a)==24
    assert len(p)==8 and p[0]['panel'][1]['eligible'] and not p[1]['panel'][1]['eligible']
    return dict(status='PASS',fixed_cohort=True,raw_qualified_denominators_separate=True)
