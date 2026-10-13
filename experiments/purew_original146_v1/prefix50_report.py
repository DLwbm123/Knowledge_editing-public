"""Anonymous interim prefix50 comparison using the frozen original aggregator."""
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from contract import ARM, METHODS

R,c,r=ar.RUN,ar.common,ar.r


def main():
    marker=R/'private/outputs/bank_R0/0/50/COMPLETE.json'
    assert marker.exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    db.execute('BEGIN')
    rows=list(db.execute("SELECT * FROM consumer WHERE mode='bank_R0' AND prefix=50"))
    assert Counter(x['method'] for x in rows)=={m:c.read(marker)['outputs'] for m in METHODS}
    payloads={x['key']:x for x in db.execute("SELECT DISTINCT p.* FROM payload p JOIN consumer c ON p.key=c.payload_key WHERE c.mode='bank_R0' AND c.prefix=50")}
    statuses={m:dict(Counter(payloads[x['payload_key']]['status'] for x in rows if x['method']==m)) for m in METHODS}
    if any(x['status'] not in ('FORMAT_VALID','MISSING') for x in payloads.values()):
        print(json.dumps(dict(status='SCORING_PENDING',consumer_statuses=statuses)))
        return
    scores={k:x['correct'] for k,x in payloads.items()}
    assert all(x['correct'] in (0,1) if x['status']=='FORMAT_VALID' else x['correct'] is None for x in payloads.values())
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'][:50]
    ledger=c.read(R/'private/EVAL_LEDGER.json')
    active={(t['native']['image_sha256'],t['native']['question']) for t in tasks}
    records=[]
    for x in rows:
        if x['method']==ARM:
            d=c.read(x['path'])
            assert c.digest(d)==x['output_binding']
            assert d['binding']['phase']==c.read(marker)['phase']
            assert json.loads(payloads[x['payload_key']]['binding'])['output']==d['R0']
        query=ledger['queries'][x['query_id']]
        records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=50),input=query,mode='bank_R0',owner_order=x['edit_order']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
    panels=[];coefficients={};groups={}
    for method in METHODS:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,coef,g=r.panel(records,scores,'RETRO146',method,0,'bank_R0',task,tasks,50)
            m.pop('route_ON');n=m['observations'];correct=m['known_correct'];missing=m['missing']
            assert 0<=correct<=correct+missing<=n
            m['micro_bounds']=[100*correct/n,100*(correct+missing)/n] if n else [None,None]
            if task.endswith('L'):
                m['Base_correct_to_wrong_count_bounds']=[n-correct-missing,n-correct]
            if task=='T2G':m.update(ar.bootstrap(coef,scores,g))
            panels.append(m);coefficients[method,task]=coef;groups.update(g)
    contrasts=[]
    for baseline in METHODS[1:]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            a,b=coefficients[ARM,task],coefficients[baseline,task];assert set(a)==set(b)
            delta={e:r.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a}
            bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in unit.items()} for unit in delta.values()]),scores) if delta else [None,None]
            item=dict(baseline=baseline,task=task,delta_macro_bounds_pp=[x*100 if x is not None else None for x in bounds],per_edit_negative_lower_bound=sum(r.score_bounds(x,scores)[0]<0 for x in delta.values()))
            if task=='T2G':item.update(ar.bootstrap(delta,scores,groups))
            contrasts.append(item)
    result=dict(status='INTERIM_PREFIX50_TERMINAL_SCORING',checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),prefix=50,panels=panels,contrasts=contrasts,consumer_statuses=statuses,full_run_complete=False,all_original_masks_and_active_target_rule=True,bootstrap_replicates=10000,bootstrap_seed=20260912,limitations=['Fixed prefix snapshot, not full insertion-to-final T2G retention','Small historical locality support; patient independence unknown','Permanent missing retained in full denominators; no unattempted payload counted missing'])
    c.write(R/'public/PREFIX50_INTERIM.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
