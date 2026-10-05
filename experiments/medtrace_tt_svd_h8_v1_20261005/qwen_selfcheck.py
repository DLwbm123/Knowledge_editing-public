"""CPU-only boundary checks in a temporary synthetic queue, zero formal attempts."""
import json
import os
from pathlib import Path
import tempfile

import qwen_queue as q

def main():
    real=Path(os.environ['RUN_ROOT']);original=q.read(real/'RESOURCE_LEDGER.json')
    q.initialize(q.connect())
    frozen=q.lock();manifest=q.read(real/'RUN_MANIFEST.json')
    with tempfile.TemporaryDirectory(dir=real/'private/tmp',prefix='qwen_check_') as name:
        root=Path(name);judge=root/'private/judge_common';judge.mkdir(parents=True)
        for module in (q,q.q,q.h):module.RUN=root
        q.ROOT=q.q.ROOT=q.h.ROOT=judge
        q.write(judge/'EPOCH_MANIFEST.json',frozen)
        q.write(root/'RUN_MANIFEST.json',manifest)
        q.write(root/'RESOURCE_LEDGER.json',dict(Judge_attempts=0,Judge_failed_payloads=0,gpu_seconds_used=0,gpu_sessions=[]))
        q.parent_result=lambda full:None
        db=q.connect();db.execute("CREATE TABLE IF NOT EXISTS inherited (key TEXT PRIMARY KEY,parent_row TEXT,batch_receipt TEXT)")
        row=dict(query_id='synthetic',question='synthetic',reference='synthetic',image_sha256='image')
        base=dict(question='synthetic',reference='synthetic',image_sha256='image',image_path='synthetic',prompt_ids=[1],attention_mask=[1],runtime={'fixture':True},generation={'fixture':True})
        out=dict(raw_answer='synthetic',raw_token_ids=[1])
        key=q.payload(db,row,base,out)
        assert q.payload(db,row,base,out)==key
        other=q.payload(db,row,base,dict(out,raw_token_ids=[2]));assert other!=key
        altered=dict(frozen,judge_identity=dict(frozen['judge_identity'],snapshot='different'))
        q.write(judge/'EPOCH_MANIFEST.json',altered)
        assert q.payload(db,row,base,out)!=key
        q.write(judge/'EPOCH_MANIFEST.json',frozen);db.commit()
        q.request(db,dict(action='reserve',batch_id='test',keys=[key]))
        try:q.request(db,dict(action='reserve',batch_id='duplicate',keys=[key]));raise RuntimeError('duplicate accepted')
        except AssertionError:pass
        record=json.loads(db.execute('SELECT record FROM payload WHERE key=?',(key,)).fetchone()[0]);batch=dict(batch_id='test',records=[record])
        response=dict(batch_id='test',decisions=[dict(opaque_query_id=record['opaque_query_id'],is_correct=True)])
        ev=dict(status='FORMAT_VALID',actual_model=frozen['model'],reasoning_effort='disabled',snapshot=frozen['snapshot'],judge_identity=frozen['judge_identity'],input_binding=q.digest(batch),exit_code=0,errors=[],tool_event_types=[],isolation_checks={'fixture':True})
        try:q.request(db,dict(action='publish',batch_id='test',evidence=dict(ev,snapshot='wrong'),response=response));raise RuntimeError('wrong snapshot accepted')
        except AssertionError:pass
        q.request(db,dict(action='publish',batch_id='test',evidence=ev,response=response))
        assert db.execute('SELECT correct FROM payload WHERE key=?',(key,)).fetchone()[0]==1
        q.request(db,dict(action='reserve',batch_id='failed',keys=[other]))
        import qwen_scorer as scorer
        scorer.RUN=root;scorer.ROOT=judge
        scorer.settle_reserved(db)
        assert db.execute('SELECT status FROM payload WHERE key=?',(other,)).fetchone()[0]=='MISSING'
        try:q.request(db,dict(action='reserve',batch_id='forbidden-retry',keys=[other]));raise RuntimeError('missing retried')
        except AssertionError:pass
        import sqlite3
        inherited_db=sqlite3.connect(':memory:');inherited_db.row_factory=sqlite3.Row
        inherited_db.execute('CREATE TABLE payload (key TEXT PRIMARY KEY,record TEXT,binding TEXT,status TEXT,batch TEXT,correct INTEGER)')
        inherited_db.execute('CREATE TABLE inherited (key TEXT PRIMARY KEY,parent_row TEXT,batch_receipt TEXT)')
        for k in [key,other]:
            parent_row=dict(db.execute('SELECT * FROM payload WHERE key=?',(k,)).fetchone());full=json.loads(parent_row['binding'])
            saved=q.read(judge/'evidence'/(parent_row['batch']+'.json'));attempt=next(x for x in q.read(root/'RESOURCE_LEDGER.json')['Judge_batches'] if x['id']==parent_row['batch'])
            assert q.validate_inherited(full,parent_row,saved,attempt,frozen['judge_identity'])
            q.parent_result=lambda obj,r=parent_row,e=saved:(r,e)
            inherited_key=q.payload(inherited_db,row,base,out if k==key else dict(out,raw_token_ids=[2]))
            assert inherited_key==k and inherited_db.execute('SELECT status FROM payload WHERE key=?',(k,)).fetchone()[0]==parent_row['status']
            try:q.request(inherited_db,dict(action='reserve',batch_id='inherit-no-retry',keys=[k]));raise RuntimeError('inherited payload retried')
            except AssertionError:pass
            altered=dict(full,output=dict(full['output'],raw_token_ids=[123]))
            try:q.validate_inherited(altered,parent_row,saved,attempt,frozen['judge_identity']);raise RuntimeError('unbound inherited result accepted')
            except AssertionError:pass
        inherited_db.close();db.close()
    current=q.read(real/'RESOURCE_LEDGER.json')
    assert current['Judge_attempts']==original['Judge_attempts'] and current.get('Judge_batches',[])==original.get('Judge_batches',[]), 'CPU test touched formal Judge ledger'
    print(json.dumps(dict(status='PASS',checks=['full identity dedup','tokens and snapshot separate keys','one attempt','wrong snapshot rejected','valid response publication','missing terminal','interrupted reservation settles without request','parent valid and permanent missing inherited without attempt','mismatched inherited tokens rejected'],formal_Judge_attempts=original['Judge_attempts'],GPU_loads=0)))

if __name__=='__main__':main()
