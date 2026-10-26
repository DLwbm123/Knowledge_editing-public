"""Read-only terminal MEMIT accounting, including its four declared UNKNOWNs."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time


def main():
    root,parent=Path(os.environ['REFERENCE_RUN']),Path(os.environ['PARENT_RUN']);out=root/'MEMIT_MM'
    load=lambda p:json.loads(Path(p).read_text())
    terminal=load(out/'INCOMPLETE_GENERATION.json') if (out/'INCOMPLETE_GENERATION.json').exists() else load(out/'GENERATION_COMPLETE.json')
    assert load(out/'WORKER_EXIT.json')['code']==load(out/'GUARD_EXIT.json')['code']==0
    plan=[v for v in load(root/'EVAL_PLAN.json') if v['method']=='MEMIT_MM'];assert len(plan)==7494
    db=sqlite3.connect('file:'+str(root/'judge_v131/queue.sqlite')+'?mode=ro',uri=True)
    states=dict(db.execute('SELECT state,count(*) FROM payload GROUP BY state'));assert not states.get('PENDING') and not states.get('RESERVED')
    scores={};evidence=[]
    for method,mode,n,owner,qid,state,correct in db.execute('SELECT c.method,c.mode,c.prefix,c.owner,c.query_id,p.state,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key WHERE c.method=\'MEMIT_MM\''):
        key=(mode,n,owner,qid);assert key not in scores
        scores[key]=bool(correct) if state in ('FORMAT_VALID','REUSED') and correct in (0,1) else None
    aborted=terminal.get('aborted_consumers',[]);assert len(scores)+len(aborted)==7494
    allowed={tuple(json.loads(v['consumer_id'] if isinstance(v,dict) else v)[1:]) for v in aborted}
    for row in plan:
        key=(row['mode'],row['prefix'],row['owner'],row['query']['query_id'])
        if key not in scores:
            assert key in allowed;scores[key]=None
    packet=load(parent/'private/M13_SOURCE_PACKET.json');tasks=packet['original_tasks'];mask=packet['base_mask']
    panels={}
    for mode,n in [('single',1)]+[('bank',n) for n in (1,50,100,146)]:
        panels[f'{mode}_{n}']={};ts=tasks if mode=='single' else tasks[:n]
        active={(t['native']['image_sha256'],t['native']['question']) for t in ts}
        queries={v['query']['query_id']:v['query'] for v in plan}
        for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
            owners={}
            for t in ts:
                qids=[f"edit_{t['order']}_{i}" for i in range(1,5)] if role=='FIT' else list(dict.fromkeys(q for e in t['events'] if e['task']==role for q in e['all_probe_query_ids']))
                if role not in ('T0','FIT'):qids=[q for q in qids if mask[q]==role.endswith('L')]
                if mode=='bank' and role.endswith('L'):qids=[q for q in qids if (queries[q]['image_sha256'],queries[q]['question']) not in active]
                if qids:owners[t['order']]=[scores[mode,n,t['order'] if mode=='single' else 0,q] for q in qids]
            values=[v for rows in owners.values() for v in rows]
            panels[f'{mode}_{n}'][role]=dict(n=len(values),support_edits=len(owners),correct=sum(v is True for v in values),unknown=sum(v is None for v in values),
                macro_percent=100*sum(sum(rows)/len(rows) for rows in owners.values())/len(owners) if values and all(v is not None for v in values) else None)
    result=dict(status='INCOMPLETE_KNOWN_GENERATION_UNKNOWNS' if aborted else ('INCOMPLETE_SCORING' if any(v is None for v in scores.values()) else 'COMPLETE'),
        planned_consumers=7494,durable_generated=terminal['counts']['formal_consumers'],known_generation_UNKNOWN=len(aborted),
        scoring_UNKNOWN=sum(v is None for v in scores.values())-len(aborted),all146_inserted=True,counts=terminal['counts'],
        panels=panels,judge=dict(model='gpt-6-astra',reasoning_effort='high',protocol='MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1'),
        no_failed_retry=True,scientific_limits='Exposed original146 DEV; old M13P medium labels are a different effort, not a fresh matched CORE/FULL epoch',
        time=time.time())
    print(json.dumps(result))

if __name__=='__main__':main()
