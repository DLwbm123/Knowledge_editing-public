"""Six-arm blind queue using the unchanged Stage17 one-attempt protocol."""
import json
import os
from pathlib import Path
import random
import sqlite3
import sys

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import legacy_queue as q
from audit import read,write,digest,OLD
ARMS=('CP_NO_H','CP_H1','DIRECT_NO_H','DIRECT_H1')
ROOT=q.ROOT
MISSING={}
def initialize(db):raise RuntimeError('Use qualified qwen_queue.initialize')
def payload(*args):raise RuntimeError('Use qualified qwen_queue.payload')
def ingest(db):
    if db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone():return
    if not (RUN/'private/GENERATION_COMPLETE.json').exists():return
    complete=read(RUN/'private/GENERATION_COMPLETE.json');assert 'P1' in complete['completed_phases']
    l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');b=read(RUN/'private/legacy_stage17/BINDINGS.json')
    paths=sorted((RUN/'private/outputs').glob('*/*/e*/*/*.json'))
    # A fixed shuffle of complete input units interleaves all six blinded arms.
    groups={}
    for p in paths:
        d=read(p);arm,mode,order,folder=p.parts[-5:-1];assert arm in ARMS and d['binding']['arm']==arm
        row=d['binding']['input'];unit=(mode,order,folder,row['query_id']);groups.setdefault(unit,[]).append((p,d,arm,mode,order,folder))
    units=sorted(groups);random.Random(20260912).shuffle(units)
    for unit in units:
        rows=groups[unit];assert len(rows)==len(ARMS) and {x[2] for x in rows}==set(ARMS)
        random.Random(digest([20260912,unit])).shuffle(rows)
        for p,d,arm,mode,order,folder in rows:
            row=d['binding']['input'];base=d['binding']['judge_input'] if mode=='H_fit' else b[row['opaque_Base_id']]
            assert mode=='H_fit' or row==l['queries'][row['query_id']]
            key=payload(db,row,base,d['R0']);cid=digest([arm,mode,int(order[1:]),folder,row['query_id'],d['execution_key'] if 'execution_key' in d else digest(d)])
            db.execute('INSERT OR IGNORE INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(cid,arm,mode,d['binding']['prefix'],int(order[1:]),folder,row['query_id'],str(p),digest(d),key))
    # New Base decisions are sensitivity only. Original masks remain unchanged.
    queries=[l['queries'][qid] for qid in sorted({d['binding']['input']['query_id'] for p,d,arm,mode,order,folder in (x for xs in groups.values() for x in xs) if mode!='H_fit'})]
    random.Random(20260912).shuffle(queries)
    for row in queries:
        base=b[row['opaque_Base_id']];out=dict(raw_answer=base['output']['model_answer_raw'],raw_token_ids=base['output']['raw_generated_token_ids'])
        key=payload(db,row,base,out);cid=digest(['Base',row['query_id']]);db.execute('INSERT OR IGNORE INTO consumer VALUES (?,?,?,?,?,?,?,?,?,?)',(cid,'Base','base',0,0,'panel',row['query_id'],str(RUN/'private/legacy_stage17/BINDINGS.json'),digest(base),key))
    pending=db.execute("SELECT count(*) FROM payload WHERE status='PENDING'").fetchone()[0]
    write(ROOT/'QUEUE_ADMISSION.json',dict(status='PASS' if pending<=2000 else 'JUDGE_BUDGET_NOT_ADMITTED',new_pending=pending,full146=complete['full146'],four_arm_units=len(units),fixed_input_order=True,old_success_scores_reused="EXACT_PARENT_QWEN_ONLY",epoch_salt_not_in_payload=True))
    assert pending<=2000,'Complete blinded queue exceeds frozen Judge cap; no partial favorable selection'
    db.execute("INSERT INTO done VALUES ('generation')");db.commit()
original_status=q.status
def status(db):
    state=original_status(db);ledger=read(RUN/'RESOURCE_LEDGER.json');batches=ledger.get('Judge_batches',[])
    last=list(reversed(batches));consecutive=0
    for batch in last:
        if batch['status']=='FAILED_NO_RETRY' and batch.get('transport_failure'):consecutive+=1
        else:break
    terminal=not db.execute("SELECT 1 FROM payload WHERE status NOT IN ('FORMAT_VALID','MISSING')").fetchone()
    if consecutive>=3:state['status']='STOPPED'
    elif db.execute("SELECT 1 FROM done WHERE name='generation'").fetchone() and terminal and not (RUN/'STOP').exists():state['status']='GENERATION_COMPLETE'
    return state
q.initialize=initialize;q.payload=payload;q.ingest=ingest;q.status=status
if __name__=='__main__':q.main()
