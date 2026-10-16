"""Actual frozen plan subset and synthetic scoring checks; no VLM/Judge/GPU."""
import copy
import json
import os
from pathlib import Path
import tempfile
import interim20 as protocol
import interim20_queue
from selected_report146 import calculate


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Invalid input accepted')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    run=Path(os.environ['RUN_ROOT']);read=lambda p:json.loads(p.read_text())
    rows=protocol.plan_subset(read(Path(os.environ['INTERIM20_PARENT'])/'private/EVAL_PLAN146.json'))
    rejected(lambda:protocol.plan_subset(rows[:-1]))
    rejected(lambda:protocol.plan_subset(rows+[rows[0]]))
    records={};scores={}
    for row in rows:
        key=tuple(row[k] for k in ('method','mode','prefix','owner'))+(row['query']['query_id'],)
        records[key]={'active_target':False};scores[key]=1
    tasks=read(run/'private/ORIGINAL146_TASKS.json')[:20];mask=read(run/'private/ORIGINAL_BASE_MASK146.json')
    report,_=calculate(tasks,mask,records,scores,interim20=True)
    assert report['complete_no_missing'] and not report['original_joint_scientific_gate_confirmed']
    assert set(report['panels']['M13'])=={'single_1'}
    assert report['panels']['M13']['single_1']['FIT']['observations']==80
    key=next(k for k in scores if k[0]=='FISHER_MIN' and k[4].startswith('edit_') and k[4].endswith('_1'))
    scores[key]=None;missing,_=calculate(tasks,mask,records,scores,interim20=True)
    assert missing['missing_consumers']==1 and not missing['complete_no_missing']
    for key in scores:
        if key[0]=='BASE':scores[key]=0
    zero,_=calculate(tasks,mask,records,scores,interim20=True)
    assert all(v['correct_retention_rate'] is None for v in zero['protection'].values())
    import astra_queue as a
    from fine_queue import full_identity
    q=a.q;base=interim20_queue.base
    judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
    plan=copy.deepcopy(rows)
    for row in plan:row['query']=dict(query_id=row['query']['query_id'],question='fixture',reference='fixture',image_path='fixture')
    def output(row):
        inp=dict(row['query'],image_sha256='fixture')
        binding=dict(inp,prompt_ids=[1],attention_mask=[1],runtime={'precision':'FP32'},generation={'max_new_tokens':1024})
        return dict(consumer_id=protocol.contract.consumer_id(row),**{k:row[k] for k in ('method','mode','prefix','owner')},
            role='FIXTURE',lock='fixture',binding=dict(input=inp,judge_input=binding),R0=dict(raw_answer='fixture',raw_token_ids=[1]))
    with tempfile.TemporaryDirectory(prefix='e.',dir=run/'private') as tmp:
        q.ROOT=Path(tmp);db=q.connect();q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=protocol.EPOCH,**judge))
        outputs=[(str(i),output(row)) for i,row in enumerate(plan)]
        with db:coverage=base.ingest_generated(q,db,plan,outputs,full_identity,judge,'fixture',{})
        assert coverage['consumers']==1077 and coverage['payloads']==335
        key=db.execute('SELECT key FROM payload LIMIT 1').fetchone()[0]
        db.execute("UPDATE payload SET status='MISSING',correct=NULL WHERE key=?",(key,));db.commit()
        rejected(lambda:base.request(a,q,run,db,dict(action='reserve',keys=[key],worker=0)))
    result=dict(status='PASS',consumers=1077,synthetic_unique_payloads=335,first20_original_order=True,
        no_bank20_or_training=True,original_masks_macro_and_double_CI=True,missing_and_zero_Base_NA=True,
        no_M1_M3_success_claim=True,sealed_queue_and_permanent_missing=True,new_GPU=0,new_Judge=0)
    (run/'private/INTERIM20_CPU_CHECKS1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
