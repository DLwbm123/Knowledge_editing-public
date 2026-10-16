"""CPU-only integration admission on the actual frozen runtime and queue."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import cm_eval146 as protocol


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    run=Path(os.environ['RUN_ROOT']);gate=protocol.admit(run,check_weights=True)
    import cm_runtime146 as w
    sys.path.insert(0,str(Path(__file__).parent))
    import bank146
    import torch
    assert not torch.cuda.is_available()
    entries=[dict(logical_edit_id=str(i),key=[float(i)],radius=.1) for i in range(1,147)]
    for method in protocol.METHODS:
        for n in range(1,147):
            assert len(bank146.bank_entries(entries,dict(method=method,mode='insertion',prefix=n,owner=n)))==n
        for n in (1,50,100,146):
            assert len(bank146.bank_entries(entries,dict(method=method,mode='bank',prefix=n,owner=0)))==n
        for owner in (1,146):
            point=w.r.p.load_state(Path(gate['parent'])/f'private/candidates/{method}/{owner}.pt')
            assert sum(v.numel() for v in point['states']['FINAL'].values())==7168
    plan=protocol.read(run/'private/CM_NEW_PLAN1.json')
    parts=[[r for i,r in enumerate(plan) if i%2==p and ((r['mode'] in ('base','single')) if a=='single' else (r['mode'] in ('bank','insertion')))] for a in ('single','bank') for p in (0,1)]
    assert sum(map(len,parts))==len(plan)==16375
    assert len({protocol.contract.consumer_id(r) for part in parts for r in part})==len(plan)
    r=w.r;generate=r.s.fit.short.generate;baseline=r.s.baseline
    try:
        r.s.fit.short.generate=lambda *args:(SimpleNamespace(decoded_text='fixture',raw_token_ids=[1]),{})
        r.s.baseline=lambda *args:run/'private/fixture_base_not_read'
        with tempfile.TemporaryDirectory(prefix='e.',dir=run/'private') as tmp:
            for arm in ('BASE','C','M13'):
                out=Path(tmp)/(arm+'.json');counts=dict(forwards=0,generation_attempts=0)
                row=dict(query_id='fixture',audit_role='EVAL')
                r._emit(None,None,dict(order=1),arm,(row,None,dict(image_sha256='fixture'),None,{}),counts,dest=out)
                data=protocol.read(out)
                assert counts['generation_attempts']==1 and data['R0']['raw_answer']=='fixture'
                assert data['execution']==protocol.read(run/'private/CM_SOURCE1.json')
    finally:r.s.fit.short.generate=generate;r.s.baseline=baseline
    import cm_queue146 as queue
    import astra_queue as a
    from fine_queue import full_identity,selfcheck
    q=a.q;judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
    fixture=copy.deepcopy(protocol.read(run/'private/EVAL_PLAN146.json'))
    for row in fixture:row['query']=dict(query_id=row['query']['query_id'],question='fixture',reference='fixture',image_path='fixture')
    def output(row):
        inp=dict(row['query'],image_sha256='fixture')
        b=dict(inp,prompt_ids=[1],attention_mask=[1],runtime={'precision':'FP32'},generation={'max_new_tokens':1024})
        return dict(consumer_id=protocol.contract.consumer_id(row),**{k:row[k] for k in ('method','mode','prefix','owner')},role='FIXTURE',lock='fixture',binding=dict(input=inp,judge_input=b),R0=dict(raw_answer='fixture',raw_token_ids=[1]))
    with tempfile.TemporaryDirectory(prefix='e.',dir=run/'private') as tmp:
        q.ROOT=Path(tmp);db=q.connect();q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=protocol.EPOCH,**judge))
        with db:coverage=queue.base.ingest_generated(q,db,fixture,[(str(i),output(row)) for i,row in enumerate(fixture)],full_identity,judge,'fixture',{})
        assert coverage['consumers']==17081
        q.ingest=lambda _:None;first=q.request(db,dict(action='next'))['rows'];assert len(first)==50
        with db:db.executemany("UPDATE payload SET status='MISSING' WHERE key=?",[(v['key'],) for v in first])
        second=q.request(db,dict(action='next'))['rows'];assert len(second)==50 and not {r['key'] for r in first}&{r['key'] for r in second}
        db.close()
    protocol.project_budget(run)
    checks=dict(status='PASS',actual_CPU_import=True,actual_C_M13_FINAL_readable=True,parameters=7168,
        real_emit_write_mock_checked=True,complete_partition_sizes=list(map(len,parts)),queue_consumers=17081,
        serial_scoring_and_permanent_missing_checked=True,identity_checks=selfcheck(),GPU=0,Judge=0)
    (run/'private/CM_CPU_CHECK1.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))


if __name__=='__main__':main()
