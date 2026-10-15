"""CPU-only checks: complete pairing, semantic flags, missingness, queue identities."""
import copy
import json
import os
from pathlib import Path
import tempfile
import wording_pair as p
from wording_report import analyze


def reject(f):
    try:f()
    except AssertionError:return
    raise AssertionError('Invalid input accepted')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    run=Path(os.environ['RUN_ROOT']);read=lambda x:json.loads(x.read_text())
    gate=p.check(run);pairs=read(run/'private/WORDING_PAIRS.json')
    plan=read(run/'private/EVAL_PLAN146.json');original=read(Path(gate['original'])/'private/EVAL_PLAN146.json')
    reject(lambda:p.make_plan(original,pairs[:-1]))
    changed=copy.deepcopy(pairs);changed[0]['question']=changed[0]['original_question']
    reject(lambda:p.make_plan(original,changed))
    old={m:[dict(correct=0,answer='x',route=[None,None]) for _ in pairs] for m in p.METHODS}
    new=copy.deepcopy(old)
    for m in p.METHODS:
        for row in new[m]:row['correct']=1
    result=analyze(pairs,old,new)
    assert result['status']=='COMPLETE_VALID'
    assert all(v['primary']['edit_macro_pp']==100 and v['material_wording_signal'] for v in result['methods'].values())
    assert all(v['edit_macro_pp']==0 for v in result['change_contrasts'].values())
    new['M13'][0]['correct']=None
    result=analyze(pairs,old,new)
    assert result['status']=='COMPLETE_WITH_MISSING' and result['methods']['M13']['primary'] is None
    assert result['methods']['M13']['delta_missing_bounds_pp'][0] < 100
    import wording_queue as w
    import astra_queue as a
    from fine_queue import full_identity
    q=a.q;judge=dict(model='gpt-6-astra',reasoning_effort='medium',protocol=q.PROTOCOL,prompt=q.PROMPT)
    fixture=copy.deepcopy(plan)
    for row in fixture:row['query']=dict(query_id=row['query']['query_id'],question='fixture',reference='fixture',image_path='fixture')
    def output(row):
        inp=dict(row['query'],image_sha256='fixture')
        b=dict(inp,prompt_ids=[1],attention_mask=[1],runtime={'precision':'FP32'},generation={'max_new_tokens':1024})
        return dict(consumer_id=w.contract.consumer_id(row),**{k:row[k] for k in ('method','mode','prefix','owner')},role='FIXTURE',lock='fixture',binding=dict(input=inp,judge_input=b),R0=dict(raw_answer='fixture',raw_token_ids=[1]))
    with tempfile.TemporaryDirectory(prefix='e.',dir=run/'private') as tmp:
        q.ROOT=Path(tmp);db=q.connect();q.write(q.ROOT/'EPOCH_MANIFEST.json',dict(epoch=p.EPOCH,**judge))
        with db:c=w.base.ingest_generated(q,db,fixture,[(str(i),output(row)) for i,row in enumerate(fixture)],full_identity,judge,'fixture',{})
        assert c['consumers']==234 and c['payloads']==78
        q.ingest=lambda _:None
        first=q.request(db,dict(action='next'))['rows'];assert len(first)==50
        with db:db.executemany("UPDATE payload SET status='MISSING' WHERE key=?",[(v['key'],) for v in first])
        second=q.request(db,dict(action='next'))['rows'];assert len(second)==28
    import wording_runtime
    # Exercise the real inherited emit/write path with a fake generation result.
    from types import SimpleNamespace
    r=wording_runtime.r
    generate=r.s.fit.short.generate;baseline=r.s.baseline
    try:
        r.s.fit.short.generate=lambda *args:(SimpleNamespace(decoded_text='fixture',raw_token_ids=[1]),{})
        r.s.baseline=lambda *args:run/'private/fixture_base_not_read'
        with tempfile.TemporaryDirectory(prefix='e.',dir=run/'private') as tmp:
            out=Path(tmp)/'out.json';counts=dict(forwards=0,generation_attempts=0)
            row=dict(query_id='fixture',audit_role='EVAL')
            r._emit(None,None,dict(order=1),'BASE',(row,None,dict(image_sha256='fixture'),None,{}),counts,dest=out)
            data=read(out)
            assert counts['generation_attempts']==1 and data['R0']['raw_answer']=='fixture'
            assert data['execution']==read(run/'private/WORDING_SOURCE1.json')
    finally:r.s.fit.short.generate=generate;r.s.baseline=baseline
    assert not (run/'private/WORDING_GPU_ONCE_0').exists()
    result=dict(status='PASS',pairs=78,consumers=234,originals_immutable=True,missing_and_macros_checked=True,synthetic_dedup_payloads=78,real_emit_write_mock_checked=True,serial_second_batch_checked=True,CPU_import=True,GPU=False,Judge=0)
    (run/'private/WORDING_CPU_CHECK1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
